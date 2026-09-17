import numpy as np

from furti_ai.ocr import TextDetector


class PaddleThreeCompatibilityStub:
    """Simulate a 3.x wrapper whose deprecated ocr method rejects cls."""

    def predict(self, image):
        raise RuntimeError("predict unavailable in this stub")

    def ocr(self, image, **kwargs):
        if "cls" in kwargs:
            raise TypeError("predict() got an unexpected keyword argument 'cls'")
        return [
            {
                "rec_texts": ["Export"],
                "rec_scores": [0.99],
                "rec_polys": [[[1, 2], [9, 2], [9, 8], [1, 8]]],
            }
        ]


def test_paddleocr_three_compatibility_path_does_not_pass_cls():
    detector = TextDetector(enabled=False)
    detector._ocr = PaddleThreeCompatibilityStub()
    detector.available = True

    lines = detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert [line.text for line in lines] == ["Export"]
    assert lines[0].bbox.center == (5, 5)


def test_ocr_rescales_boxes_after_reduced_resolution_detection():
    detector = TextDetector(enabled=False, max_dim=320)
    detector._ocr = PaddleThreeCompatibilityStub()
    detector.available = True

    lines = detector.detect(np.zeros((320, 640, 3), dtype=np.uint8))

    assert lines[0].bbox.x == 2
    assert lines[0].bbox.y == 4
    assert lines[0].bbox.width == 16
    assert lines[0].bbox.height == 12


def test_rapidocr_compact_result_shape_is_parsed():
    detector = TextDetector(enabled=False)
    detector._ocr = lambda _image: (
        [
            [
                [[2, 3], [18, 3], [18, 12], [2, 12]],
                "Save",
                0.92,
            ]
        ],
        0.01,
    )
    detector._active_backend = "rapidocr"
    detector.available = True

    lines = detector.detect(np.zeros((20, 30, 3), dtype=np.uint8))

    assert [line.text for line in lines] == ["Save"]
    assert lines[0].bbox.center == (10, 7)


def test_rapidocr_parallel_array_result_shape_is_parsed():
    class RapidResult:
        boxes = np.array([[[2, 3], [18, 3], [18, 12], [2, 12]]])
        txts = np.array(["Save"])
        scores = np.array([0.92])

    detector = TextDetector(enabled=False)
    detector._ocr = lambda _image: RapidResult()
    detector._active_backend = "rapidocr"
    detector.available = True

    lines = detector.detect(np.zeros((20, 30, 3), dtype=np.uint8))

    assert [line.text for line in lines] == ["Save"]
    assert lines[0].confidence == 0.92


def test_icon_matcher_uses_cached_templates_and_exact_scale_by_default(tmp_path):
    import cv2
    from furti_ai.ocr import IconMatcher

    template = np.zeros((10, 10), dtype=np.uint8)
    template[2:8, 3:7] = 255
    path = tmp_path / "save.png"
    assert cv2.imwrite(str(path), template)
    screen = np.zeros((60, 80, 3), dtype=np.uint8)
    screen[20:30, 30:40] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)

    matcher = IconMatcher(tmp_path, threshold=0.99, max_screen_dim=80)
    first = matcher.find_icons(screen)
    second = matcher.find_icons(screen)

    assert first and second
    assert first[0].name == "save"
    assert len(matcher._template_cache) == 1
    assert matcher.multi_scale is False


def test_icon_matcher_skips_executor_anchor_crops(tmp_path):
    import cv2
    from furti_ai.ocr import IconMatcher, is_auto_crop_template

    template = np.zeros((10, 10), dtype=np.uint8)
    template[2:8, 3:7] = 255
    for name in ("save.png", "task_1.png", "task_ocr_2.png", "run.auto.png"):
        assert cv2.imwrite(str(tmp_path / name), template)

    matcher = IconMatcher(tmp_path, threshold=0.99, max_screen_dim=80)

    assert [path.name for path in matcher._template_files()] == ["save.png"]
    assert is_auto_crop_template("task_1") is True
    assert is_auto_crop_template("task_ocr_2") is True
    assert is_auto_crop_template("save.auto") is True
    # Real user templates are still advertised, including "task"-like names.
    assert is_auto_crop_template("task") is False
    assert is_auto_crop_template("task_1_backup") is False


def test_describe_reports_requested_and_active_backend():
    detector = TextDetector(enabled=False)
    detector.enabled = True
    detector.available = True
    detector._active_backend = "paddle"
    detector.backend_note = (
        "RapidOCR unavailable (missing module); using the PaddleOCR fallback"
    )

    description = detector.describe()

    assert "requested=rapidocr" in description
    assert "active=paddle" in description
    assert "using the PaddleOCR fallback" in description


def test_repeated_inference_failures_disable_ocr_visibly():
    detector = TextDetector(enabled=False)
    detector.enabled = True
    detector.available = True
    detector._active_backend = "paddle"

    def failing_ocr(_image):
        raise NotImplementedError("oneDNN path is unavailable in this build")

    detector._ocr = failing_ocr
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    for _ in range(detector.MAX_CONSECUTIVE_FAILURES - 1):
        assert detector.detect(frame) == []
        assert detector.available is True

    assert detector.detect(frame) == []
    assert detector.available is False
    assert "oneDNN" in detector.runtime_failure
    assert "OCR UNAVAILABLE" in detector.describe()
    # Later frames are skipped instead of retrying a known-dead backend.
    assert detector.detect(frame) == []
