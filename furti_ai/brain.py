"""The DeepSeek fallback layer.

``BrainPlanner`` is only invoked for novel tasks or when a compiled reflex
fails. It screenshots the screen, asks a vision-capable LLM for a structured
plan (via OpenAI-style function calling), then *compiles* that plan into a
local reflex: a cropped template image plus metadata stored by ``MemoryManager``.

The LLM call is intentionally decoupled behind the :class:`LLMClient` protocol
so the model, vendor, or even a fully local model can be swapped without
touching the rest of the system.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

from .config import Settings
from .memory import MemoryManager
from .models import ActionPlan, ActionType, Skill
from .screen import ScreenCapture

logger = logging.getLogger(__name__)

# Instructs the model to return exactly one structured action object.
SYSTEM_PROMPT = (
    "You are the visual planner for Furti AI, a desktop automation agent. "
    "Given a screenshot and a user instruction, locate the target UI element "
    "and return a single JSON object with keys: action, bbox, text, "
    "confidence, description, params. The action must be one of the "
    "allowed values click, double_click, right_click, type, scroll, key_press. "
    "Coordinates are in pixels relative to the top-left of the screenshot. "
    "Return bbox only as an object with integer x, y, width, and height "
    "(never as [left, top, right, bottom]). The bounding box must tightly "
    "surround the target element so it can be "
    "cropped and re-matched later. Return JSON only, with no markdown and no tool name."
)


class LLMClient(Protocol):
    """Minimal contract for the reasoning backend."""

    def chat_with_vision(self, prompt: str, image_b64: str) -> dict[str, Any]:
        """Return a raw plan dict for the given prompt + PNG screenshot.

        The returned dict must contain at least ``action`` and ``bbox`` keys
        (see :meth:`ActionPlan.from_dict`).
        """
        ...


def _plan_tool_schema() -> dict[str, Any]:
    """OpenAI-style tool definition that forces a structured JSON plan."""
    return {
        "type": "function",
        "function": {
            "name": "plan_action",
            "description": "Produce one executable UI action plan from a screenshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [a.value for a in ActionType],
                        "description": "Low-level action to perform.",
                    },
                    "bbox": {
                        "type": "object",
                        "description": (
                            "Pixel bounding box as an object with x, y, width, "
                            "and height. Do not return a four-item array."
                        ),
                        "properties": {
                            "x": {"type": "integer"},
                            "y": {"type": "integer"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                        },
                        "required": ["x", "y", "width", "height"],
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type when action is 'type'.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Model confidence between 0 and 1.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short human-readable summary of the action.",
                    },
                    "params": {
                        "type": "object",
                        "description": "Extra parameters (e.g. key, scroll_clicks).",
                        "additionalProperties": True,
                    },
                },
                "required": ["action", "bbox", "description"],
            },
        },
    }


class DeepSeekClient:
    """OpenAI-SDK client pointed at DeepSeek's OpenAI-compatible endpoint.

    The default ``deepseek-flash`` configuration is used for image-capable
    task calls. Sending screenshots still depends on the configured
    OpenAI-compatible endpoint accepting the ``image_url`` payload; the
    abstraction allows another vision-capable model to be selected without
    changing :class:`BrainPlanner`.
    """

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        usage_callback=None,
    ) -> None:
        if not settings.deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is not set. Export it or pass it via Settings."
            )
        try:
            from openai import OpenAI  # lazy import keeps the package import-light
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Install the 'openai' package to use DeepSeekClient."
            ) from exc

        self._settings = settings
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._model = model or settings.deepseek_model
        self.usage_callback = usage_callback
        self.call_count = 0

    def _build_request_kwargs(
        self,
        messages: list[dict[str, Any]],
        use_tools: bool = True,
    ) -> dict[str, Any]:
        """Build the OpenAI-compatible payload safely for both chat and reasoning models.

        Reasoning / thinking endpoints reject ``tool_choice`` when a model is
        in thinking mode, even if the tool call schema is otherwise valid. We
        therefore omit tools and tool_choice whenever configured for a
        thinking model or when function calling is deliberately disabled.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
        }

        if use_tools and self._settings.deepseek_use_function_calling and not self._settings.deepseek_thinking_mode:
            kwargs["tools"] = [_plan_tool_schema()]
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": "plan_action"},
            }

        return kwargs

    def chat_with_vision(self, prompt: str, image_b64: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + " Return a JSON object only."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]

        kwargs = self._build_request_kwargs(messages, use_tools=True)
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc)
            # DeepSeek thinking-mode and similar endpoints reject tool_choice
            # even though the rest of the chat request is valid. Retry with the
            # same prompt but without tools. This gives us a content-only JSON
            # answer that can be parsed in the same way as the function-call path.
            if (
                "tool_choice" not in message.lower()
                and "thinking mode" not in message.lower()
                and "tool" not in message.lower()
            ):
                raise
            logger.warning(
                "Falling back from function calling to plain content JSON for DeepSeek: %s",
                exc,
            )
            kwargs = self._build_request_kwargs(messages, use_tools=False)
            response = self._client.chat.completions.create(**kwargs)

        message = response.choices[0].message

        # Preferred path: the model answered with a tool call.
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            self._record_usage(response, purpose="plan_action")
            arguments = tool_calls[0].function.arguments
            return json.loads(arguments)

        # Fallback path: content-only answer or reasoning model content stream.
        self._record_usage(response, purpose="plan_action")
        content = message.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            raise ValueError("The model response did not include a usable content string.")

        # The model may answer with a fenced JSON object. Strip fences if needed.
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        return json.loads(cleaned)

    # ----------------------------------------------- multi-step task support
    def chat_text(self, system: str, user: str, purpose: str = "") -> str:
        """Plain text completion; returns the raw assistant string."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        self._record_usage(response, purpose)
        return self._content_to_str(response.choices[0].message.content)

    def chat_vision(self, system: str, user: str, image_b64: str, purpose: str = "") -> str:
        """Vision completion returning raw text (no forced tool schema)."""
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        self._record_usage(response, purpose)
        return self._content_to_str(response.choices[0].message.content)

    # ---------------------------------------------------------------- utils
    def _record_usage(self, response: Any, purpose: str) -> None:
        self.call_count += 1
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        if self.usage_callback is not None:
            self.usage_callback(
                self._model, prompt_tokens, completion_tokens, purpose
            )

    @staticmethod
    def _content_to_str(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not isinstance(content, str):
            raise ValueError(
                "The model response did not include a usable content string."
            )
        return content


class GeminiClient:
    """Google GenAI client adapter that returns the same structured plan JSON."""

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        usage_callback=None,
    ) -> None:
        if not settings.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Export it or pass it via Settings."
            )
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:  # pragma: no cover
            try:
                import google.generativeai as genai  # older SDK layout
                genai_types = genai.types
            except ImportError as exc:
                raise ImportError(
                    "Install the 'google-genai' package to use GeminiClient."
                ) from exc

        self._settings = settings
        self._client = genai.Client(api_key=settings.google_api_key)
        self._model = model or settings.gemini_model
        self._genai_types = genai_types
        self.usage_callback = usage_callback
        self.call_count = 0

    def _extract_text(self, response: Any) -> str:
        """Extract the model answer from the google.genai response object."""
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        try:
            candidates = getattr(response, "candidates", None)
            if candidates:
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    parts = getattr(content, "parts", None)
                    if parts:
                        collected = []
                        for part in parts:
                            part_text = getattr(part, "text", None)
                            if isinstance(part_text, str):
                                collected.append(part_text)
                        if collected:
                            return "".join(collected)
        except Exception:
            pass

        raise ValueError("The Gemini response did not include a usable content string.")

    def chat_with_vision(self, prompt: str, image_b64: str) -> dict[str, Any]:
        image_data = base64.b64decode(image_b64)

        # Use the new google.genai Part API, with the screenshot bytes sent as an
        # image/png attachment and the instruction sent as text parts.
        contents = [
            self._genai_types.Part.from_text(
                text=SYSTEM_PROMPT + " Return a JSON object only."
            ),
            self._genai_types.Part.from_text(text=prompt),
            self._genai_types.Part.from_bytes(data=image_data, mime_type="image/png"),
        ]

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
            )
        except Exception as exc:
            logger.warning(
                "Gemini multimodal request failed; retrying with a text-only payload: %s",
                exc,
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    self._genai_types.Part.from_text(
                        text=SYSTEM_PROMPT + " Return a JSON object only."
                    ),
                    self._genai_types.Part.from_text(text=prompt),
                ],
            )

        self._record_usage(response, purpose="plan_action")
        text = self._extract_text(response)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        return json.loads(cleaned)

    # ----------------------------------------------- multi-step task support
    def chat_text(self, system: str, user: str, purpose: str = "") -> str:
        """Plain text completion; returns the raw assistant string."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=[
                self._genai_types.Part.from_text(text=system),
                self._genai_types.Part.from_text(text=user),
            ],
        )
        self._record_usage(response, purpose)
        return self._extract_text(response)

    def chat_vision(self, system: str, user: str, image_b64: str, purpose: str = "") -> str:
        """Vision completion returning raw text (no forced tool schema)."""
        image_data = base64.b64decode(image_b64)
        contents = [
            self._genai_types.Part.from_text(text=system),
            self._genai_types.Part.from_text(text=user),
            self._genai_types.Part.from_bytes(data=image_data, mime_type="image/png"),
        ]
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
        )
        self._record_usage(response, purpose)
        return self._extract_text(response)

    # ---------------------------------------------------------------- utils
    def _record_usage(self, response: Any, purpose: str) -> None:
        self.call_count += 1
        metadata = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(metadata, "prompt_token_count", 0) or 0
        # Thinking/thought tokens are billed on the output side.
        completion_tokens = (getattr(metadata, "candidates_token_count", 0) or 0) + (
            getattr(metadata, "thoughts_token_count", 0) or 0
        )
        if self.usage_callback is not None:
            self.usage_callback(
                self._model, prompt_tokens, completion_tokens, purpose
            )


class MockLLMClient:
    """Deterministic stand-in for the reasoning backend.

    Used by the offline demo and by unit tests. It can return a fixed plan, or
    a plan produced by ``plan_fn(prompt)``. ``call_count`` tracks how often the
    brain consulted the model, which proves cache hits skip the network.
    """

    def __init__(
        self,
        plan: dict[str, Any] | None = None,
        plan_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self._plan = plan
        self._plan_fn = plan_fn
        self.call_count = 0

    def chat_with_vision(self, prompt: str, image_b64: str) -> dict[str, Any]:
        self.call_count += 1
        if self._plan_fn is not None:
            return self._plan_fn(prompt)
        if self._plan is not None:
            return dict(self._plan)
        return {
            "action": "click",
            "bbox": {"x": 320, "y": 260, "width": 160, "height": 80},
            "description": "mock plan: click the centre of the button",
        }


class BrainPlanner:
    """Compiles an LLM plan into a cached skill (template image + metadata)."""

    def __init__(
        self,
        llm: LLMClient,
        screen: ScreenCapture,
        memory: MemoryManager,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._screen = screen
        self._memory = memory
        self._settings = settings

    def plan(self, user_prompt: str, task_name: str) -> Skill | None:
        """Capture the screen, ask the model for a plan, then compile & cache it.

        If the model answers without a usable bounding box, the planner must
        degrade to a clean ``None`` result rather than crash the run loop.
        """
        logger.debug("BrainPlanner.plan start: task=%r prompt=%r", task_name, user_prompt)
        screen_img = self._screen.capture()
        logger.debug("Captured screenshot size=%s shape=%s", screen_img.shape[:2], screen_img.shape)
        image_b64 = self._encode_png(screen_img)
        logger.debug("Encoded screenshot bytes=%d", len(image_b64))

        raw_plan = self._llm.chat_with_vision(user_prompt, image_b64)
        logger.debug("LLM returned raw plan payload: %r", raw_plan)

        raw_plan = self._normalize_model_bbox(
            raw_plan,
            screen_width=screen_img.shape[1],
            screen_height=screen_img.shape[0],
        )
        plan = ActionPlan.from_dict(raw_plan)
        logger.debug("ActionPlan normalized from raw payload: %s", plan)

        if plan.bbox is None:
            logger.warning(
                "Planner cannot compile a skill for %r because the LLM plan has no bbox: %s",
                task_name,
                raw_plan,
            )
            return None

        template_path = self._compile_template(screen_img, plan, task_name)
        logger.debug("Template crop compiled at %s", template_path)

        skill = Skill(
            name=task_name,
            template_path=str(template_path),
            action=plan.action,
            metadata={
                "description": plan.description or user_prompt,
                "action": plan.action.value,
                "text": plan.text,
                "expected_bbox": asdict(plan.bbox) if plan.bbox else None,
                "screen_size": [screen_img.shape[1], screen_img.shape[0]],
                "confidence_at_compile": plan.confidence,
                **plan.params,
            },
        )
        self._memory.save_skill(skill)
        logger.info("Compiled new skill %r -> %s", task_name, template_path)
        logger.debug("Saved skill state: %s", skill)
        return skill

    @staticmethod
    def _normalize_model_bbox(
        raw_plan: dict[str, Any],
        screen_width: int,
        screen_height: int,
    ) -> dict[str, Any]:
        """Normalize a model's occasional ``[left, top, right, bottom]`` output.

        The public ``BoundingBox`` list format remains ``[x, y, width, height]``.
        Model-generated arrays are treated as corners because vision models
        commonly emit this shape despite the structured schema. The public
        ``ActionPlan`` parser still accepts ``[x, y, width, height]`` for
        callers that construct plans directly.
        """
        raw_bbox = raw_plan.get("bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return raw_plan

        try:
            left, top, third, fourth = (int(value) for value in raw_bbox)
        except (TypeError, ValueError):
            return raw_plan

        corners_form_valid = (
            0 <= left < third <= screen_width
            and 0 <= top < fourth <= screen_height
        )
        if not corners_form_valid:
            return raw_plan

        normalized = dict(raw_plan)
        normalized["bbox"] = {
            "x": left,
            "y": top,
            "width": third - left,
            "height": fourth - top,
        }
        logger.warning(
            "Model returned bbox as [left, top, right, bottom]; normalized to %s.",
            normalized["bbox"],
        )
        return normalized

    @staticmethod
    def _encode_png(image: np.ndarray) -> str:
        """Encode a BGR image as a base64 PNG string for the vision API."""
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Failed to encode screenshot as PNG.")
        return base64.b64encode(buffer.tobytes()).decode("ascii")

    def _compile_template(
        self, screen_img: np.ndarray, plan: ActionPlan, task_name: str
    ) -> Path:
        """Crop the target region out of the screenshot and save it as a PNG."""
        bbox = plan.bbox.clamp(screen_img.shape[1], screen_img.shape[0])
        if bbox.area <= 0:
            raise ValueError(
                "The plan's bounding box is empty after clamping to the screen."
            )

        crop = screen_img[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]

        self._settings.ensure_dirs()
        safe_name = task_name.replace("/", "_").replace("\\", "_")
        template_path = self._settings.templates_dir / f"{safe_name}.png"
        cv2.imwrite(str(template_path), crop)
        return template_path
