# Icon library (CV template matching)

Drop cropped PNG screenshots of UI elements here. The **file name is the icon
label the AI uses** - for example `chrome.png` can be clicked with:

    {"type": "click_icon", "icon": "chrome"}

Furti matches these images against the live screen with OpenCV template
matching and clicks the best match. The LLM never sees or guesses coordinates.

## Tips
- Crop tightly around the icon/logo - avoid surrounding background.
- One image per label, lowercase snake_case (`file_explorer.png`).
- Matching threshold is ~0.82; exact-size crops work best. If your display
  scaling changes, take a fresh crop.
- Works best for pinned taskbar buttons, window controls, and toolbar icons.
