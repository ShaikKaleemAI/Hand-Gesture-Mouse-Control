# Air Mouse — Hand Gesture Mouse Control

Control your entire mouse — cursor, clicks, scroll, zoom, minimize, even the on-screen keyboard — using only hand gestures in front of a webcam. Windows-focused, built on MediaPipe hand-landmark tracking with a custom geometric gesture-recognition layer (no ML model training required — everything is derived from joint angles and distances in real time).

## Gesture set

| Gesture | Action |
|---|---|
| **Pointer** — index extended, middle curled | Moves the cursor continuously (thumb/pinky/ring don't affect it) |
| **Left click** — curl thumb down while pointing | Click (extend thumb back out to re-arm). Two clicks within 0.35s → double click |
| **Right click** — curl pinky down while pointing | Right click (same re-arm pattern) |
| **Scroll up** — index + middle both extended ("scissors") | Scrolls up continuously while held |
| **Scroll down** — index + middle both curled | Scrolls down continuously while held |
| **Zoom** — pinch thumb + index to a point, then spread/pinch | Standard Ctrl+Scroll zoom — works in browsers, image viewers, PDF readers, code editors |
| **Minimize** — full fist, fingertips pulled to a point, held ~0.8s | Minimizes the active window |
| **On-screen keyboard** — draw a full circle in the air with the index finger while pointing | Opens the on-screen keyboard |
| **Corner dwell** — hover cursor in top-right ~1.4s | Minimize |
| **Corner dwell** — hover cursor in top-left ~1.4s | Close window (Alt+F4) |

## How it works

- **Cursor movement and click detection are fully decoupled.** The cursor tracks the index fingertip any time index is extended and middle is curled, regardless of what the thumb/pinky are doing. Click gestures are layered independently on top.
- **Clicks use an arm/disarm trigger-pull pattern** — a click fires once when the finger curls down, and won't fire again until it's extended back out, so holding the pose doesn't spam clicks.
- **Finger extension is detected via the bend angle at each finger's PIP (middle knuckle) joint**, which is orientation-independent — it works whether the palm faces the camera, faces away, or is rotated to the side. This is the core design choice that makes the gesture set reliable across hand orientations, rather than relying on fragile fingertip-position heuristics.
- **Click-pose confirmation over consecutive frames** filters single-frame jitter while staying responsive.
- **Zoom engage/disengage uses a pinch-distance ratio scaled to palm length**, tuned so a normal click's thumb curl can't be mistaken for a pinch.
- **Fist detection for minimize** requires both a fingertip-cluster distance threshold and near-zero wrist velocity, held for ~0.8s — prevents accidental triggers from a fast-moving closed hand.

## Requirements

```
opencv-python
mediapipe==0.10.14
pyautogui
numpy
```

Install:
```bash
pip install opencv-python mediapipe==0.10.14 pyautogui numpy
```

## Run

```bash
py hand_mouse_control_v7.py
```

Focus the preview window and press `q` to quit.

## Notes

- Built and tuned for Windows (uses Windows-specific window management for minimize/close actions).
- All thresholds — extension angle, click cooldown, corner dwell time, scroll speed, cursor smoothing/gain — are exposed as constructor parameters on `HandMouseController`, so behavior can be retuned without touching the detection logic.
- Cursor smoothing uses a dual-speed filter (fast/slow smoothing windows switched by movement speed) so small precise movements stay stable while fast swipes don't feel laggy.
