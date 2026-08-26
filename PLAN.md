# Air Writer → Virtual Touchscreen for Mac

Turn a Mac into a virtual touchscreen by mapping webcam hand tracking to OS cursor/touch actions, with two display modes (cursor-only and semi-transparent camera overlay) and gesture shortcuts for media/system control.

---

## Current State
- Single-file Python/OpenCV app (`main.py`, 674 lines)
- 4 gestures: point (draw), peace (color), fist (clear), palm (erase)
- Neon drawing canvas, no system integration, no modularity

## Vision
Your webcam watches your hand → hand position maps to screen coordinates → gestures fire OS actions (click, drag, scroll, volume, etc.). No touchscreen hardware needed.

---

## Phase 1 — Architecture Refactor
Break the monolith so each capability can evolve independently.

```
air-writer/
  config.py            # Settings, gesture→action map, screen calibration
  gestures.py          # GestureEngine — expanded detection + debounce
  hand_tracking.py     # HandState, Smoother, coordinate mapping
  system_control.py    # OS dispatch: cursor, click, scroll, media, volume
  overlay.py           # Semi-transparent camera overlay window (PyObjC/Quartz)
  canvas.py            # Drawing mode (existing neon canvas, particles, history)
  hud.py               # HUD rendering, mode indicator
  recorder.py          # SessionRecorder (MP4 + GIF)
  main.py              # Slim loop: camera → detection → mode router → display
  requirements.txt     # + pyautogui, pyobjc-framework-Quartz
```

## Phase 2 — Screen Coordinate Mapping
Map hand position in camera space to macOS screen coordinates.

- **Calibration** — on first run, user points at 4 screen corners to compute a homography matrix (camera → screen)
- **Dead zone** — small region in center of motion range where cursor doesn't jitter
- **Smoothing** — reuse existing `Smoother` class with tuned factor for cursor (lower latency than drawing mode)
- **Screen bounds** — clamp to display resolution from `pyautogui.size()`

## Phase 3 — Expanded Gesture Set

| Gesture | Detection Logic |
|---------|----------------|
| **Point** | Index up, others down (existing) |
| **Pinch** | Thumb tip ↔ index tip distance < threshold |
| **Pinch hold** | Pinch sustained > 200ms |
| **Open palm** | All 5 fingers extended (existing) |
| **Fist** | All fingers curled (existing) |
| **Swipe** | Palm/point + velocity > threshold in a direction |
| **Thumbs up** | Only thumb extended, pointing up |
| **Thumbs down** | Only thumb extended, pointing down |
| **L-shape** | Thumb + index extended at ~90° angle |

All gestures get debounce (existing framework) + cooldowns to prevent repeats.

## Phase 4 — Touch + Media System Control
Wire gestures to OS actions via `pyautogui` + macOS `osascript`:

### Touch
| Gesture | Action |
|---------|--------|
| Point + move | Move system cursor |
| Pinch (quick) | Left click |
| Pinch hold + move | Click-drag |
| L-shape | Right click |
| Open palm swipe ↑↓ | Scroll up/down |
| Two-hand pinch/spread | Zoom (Cmd+/Cmd-) |

### Media & System
| Gesture | Action |
|---------|--------|
| Thumbs up | Volume up |
| Thumbs down | Volume down |
| Fist → open palm | Play/pause |
| Palm swipe ← | Previous track |
| Palm swipe → | Next track |
| Three-finger swipe ↑ | Mission Control |
| Three-finger swipe ←→ | Switch desktop |

Implemented via:
- `pyautogui` — cursor movement, clicks, keyboard shortcuts
- `subprocess` + `osascript` — volume, brightness, media keys on macOS

## Phase 5 — Display Modes

### Mode A: Cursor Only (default)
- No visible window — just moves the macOS cursor
- Small system tray icon shows status (tracking active, gesture recognized)
- Minimal resource usage

### Mode B: Semi-Transparent Overlay
- Fullscreen transparent always-on-top window (via PyObjC / Quartz `NSWindow`)
- Shows semi-transparent camera feed with hand skeleton drawn on top
- Click-through — overlay doesn't capture mouse events
- Toggle with keyboard shortcut (e.g., `Cmd+Shift+O`) or a gesture (e.g., peace sign hold)

### Mode C: Draw (preserved)
- Existing neon canvas mode, toggled with keyboard `D`
- Drawing does NOT move system cursor
- Useful for annotations, demo, or fun

Mode switching via keyboard shortcuts or a dedicated three-finger tap gesture.

## Phase 6 — Config & Calibration
- `config.json` with editable gesture→action bindings
- Re-run calibration anytime (keyboard shortcut)
- Sensitivity, smoothing, dead-zone tunable
- Per-gesture enable/disable

## Phase 7 — Polish
- On-screen action toast ("Click", "Scrolling ↓", "Vol +") — small fade-in labels
- Optional audio feedback
- README with setup guide, permissions instructions, gesture cheat sheet, demo GIF
- Unit tests for gesture detection

---

## Implementation Order

| Step | Phase | What you get |
|------|-------|-------------|
| 1 | Phase 1 | Clean module structure, existing features still work |
| 2 | Phase 2 | Hand position → screen cursor mapping with calibration |
| 3 | Phase 3 | Pinch, swipe, thumbs, L-shape detection |
| 4 | Phase 4 | Actual click/scroll/volume/media control |
| 5 | Phase 5 | Cursor-only + overlay toggle |
| 6 | Phase 6 | User-configurable bindings |
| 7 | Phase 7 | Polish, toasts, README |

## Dependencies to Add
- `pyautogui` — cursor/keyboard simulation
- `pyobjc-framework-Quartz` — transparent overlay window on macOS
- `pyobjc-framework-Cocoa` — NSWindow for overlay

## macOS Permissions Required
- **Accessibility** — for `pyautogui` cursor/keyboard control (System Settings → Privacy → Accessibility)
- **Camera** — webcam access
- **Screen Recording** — if overlay uses screen compositing

## Risks
- **Latency** — cursor control needs <50ms gesture-to-action. May need to bypass AR stabilizer and reduce smoothing in control mode
- **Jitter** — dead zones + smoothing critical for usable cursor
- **Permissions** — user must grant Accessibility access or `pyautogui` silently fails
- Drawing mode stays fully functional as a fallback throughout all phases
