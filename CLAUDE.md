# Air Writer — agent handoff

Webcam hand-tracking app. You draw neon ink in the air with your index
finger; if you draw a **closed loop around someone's face**, it captures them
as an ASCII portrait and collects their email.

**`PLAN.md` is stale.** It describes an older "virtual touchscreen for macOS"
direction and claims `main.py` is a 674-line monolith. Neither is true now.
Read this file instead; treat `PLAN.md` as historical.

---

## The demo this exists for

Onboarding/demo day. The pitch is:

> "Come up here — this is Air Writer. Draw a circle around your face.
> That's your portrait. Want us to email it to you? Type your address."

So the whole app is judged on: *does it look good on a projector, and does it
never dead-end in front of a crowd?* Reliability beats features.

## Run it

```bash
python3 main.py
```

Needs macOS camera permission for whatever terminal launches it.

## Demo flow (the state machine that matters)

1. **Idle** — bright cyan brackets + `FACE FOUND · CIRCLE IT` on any detected
   face, so people know they've been seen before they do anything.
2. **Lasso** — point-draw a closed loop. `closed_loop()` fires when a stroke
   is >28 pts, >260px of travel, and returns within 70px of its start.
3. **Countdown** — 1.2s, big 3-2-1 over the face + progress arc + `HOLD STILL`.
   `_Subject.idx` stays `None` throughout; nothing is captured yet.
4. **Capture** — portrait **freezes** (it is a keepsake, not a live filter),
   white shutter flash, card slides in from the right.
5. **Email** — type, `ENTER`. Writes to `captures/`.
6. **Clear** — hold a fist 0.7s. Next person.

## Module map

| file | ~LOC | what |
|---|---|---|
| `main.py` | 720 | camera loop, gestures, canvas/ink, lasso, key routing |
| `face_ascii.py` | 838 | face tracking, ASCII renderer, subject cards |
| `capture.py` | 147 | `CaptureStore` (JSON+PNG on disk), `EmailField` |
| `hud.py` | 127 | the 62px header strip |
| `ui.py` | 123 | real-font text, alpha blit, rounded pills |

Output is `(720 + hud.HEADER_H) x 1280` — the header is composited *above*
the camera frame into one preallocated buffer (`out`, with `bar`/`view`
slices), not drawn onto it.

### Key seams

- `AsciiRenderer.indices(crop) -> (rows, cols) int grid` — all the image
  science lives here. **Change the look here and nothing else breaks.**
- `.compose(idx, cell=None)` — grid → image at any cell size. Screen uses
  4x8; export uses 14x28 for a crisp ~1900px PNG (not an upscale).
- `.to_text(idx)` — plain ASCII, saved alongside for pasting into email.
- `FaceTracker.lockable` — faces + manual regions; what the lasso can toggle.

## Tuning knobs (all at the top of `face_ascii.py`)

| knob | now | effect |
|---|---|---|
| `CROP_PAD` | 1.45 | **the single most important one.** 2.05 swallowed the whole room and the face got ~40x25 cells — eyes unreadable |
| `INK_FLOOR` | 0.10 | lower = heavier/darker portrait |
| `INK_GAMMA` | 0.88 | midtone bias |
| `EDGE_MIX` | 0.24 | adds crisp facial detail without erasing smooth tone |
| `VIGNETTE` / `_POW` | 0.72 / 1.20 | soft superellipse falloff that suppresses the room |
| `COUNTDOWN` | 1.2s | user explicitly wanted **under 2s** |
| `FIST_HOLD` (main) | 0.7s | intent gate on the destructive gesture |

## Gotchas that cost real time — do not rediscover these

1. **The frame is mirrored BEFORE face detection.** All track coordinates are
   in mirrored/display space. Any fake tracker in a test must use mirrored
   coords (`W - x - w`) or the reticle lands in the wrong place. This wasted
   two debugging rounds.
2. **OpenCV is BGR.** Cyan is `(255, 238, 120)`, not `(120, 238, 255)`.
   Got this wrong twice.
3. **Gesture detection must stay rotation-invariant.** The original
   `_up()` compared image-space `y`, which inverts the moment a hand tilts —
   a genuine point was read as **`fist`** (destructive: clears everything)
   across a 180° range of wrist rotation. Now: a finger is extended if its tip
   is farther from the wrist than its own PIP joint. **Never reintroduce a
   y-comparison here.**
4. **The portrait is light-on-light.** Blit it with `cv2.addWeighted`, never
   `cv2.add` — additive blows a white image out to pure white.
5. **The ASCII ramp is derived at runtime by measuring ink coverage.**
   Hand-picked ramps like `" .:-=+*#%@"` are *not* monotonic in brightness
   (`-` has less ink than `:`, `+` less than `=`, `%` less than `#`) which
   shows up as banding and speckle. `_build_ramp()` picks evenly-spaced
   glyphs for whatever font/cell size is in use.
6. **While the email field has focus it swallows every printable key** —
   single-letter shortcuts don't fire, and `q` types a `q` instead of
   quitting. `ESC` drops focus first, *then* `q`/`ESC` quits. This is correct
   (people have `q` in their addresses) but is a demo-day trap.
7. **`matchTemplate` at scale 0.3 is ~7x slower than at 0.25** — non-DFT-
   friendly sizes hit a slow path. Don't "improve" `_ManualTrack.SCALE`
   without benchmarking.
8. **Never refresh a tracking template every frame.** It re-anchors to
   wherever the box already is and freezes on background. Refresh on a timer
   (`REFRESH = 2.0`) only.
9. **`ls -la` hangs in this shell.** Use `printf '%s\n' *` instead.

## Testing — read this before you try to "just run it"

**You (the agent) have no camera access.** `cv2.VideoCapture(0)` returns
"not authorized"; that permission prompt only goes to the user. Everything
below was verified with a **headless simulation harness** that monkeypatches
OpenCV before importing `main`:

```python
cv2.VideoCapture = FakeCap          # yields synthetic frames
cv2.namedWindow = cv2.resizeWindow = lambda *a, **k: None
cv2.imshow = lambda n, i: frames.append(i.copy())
cv2.waitKey = scripted_keys         # drive email typing, then 'q' to exit
cv2.destroyAllWindows = lambda: None
import main
main.HandLandmarker = FakeDetector  # synthetic 21-point hand landmarks
main.FaceTracker = FakeTracker      # optional: a reliable stand-in detector
main.main()
```

A working copy is committed as **`sim_harness.py`** — just `python3
sim_harness.py`. It runs the real path end to end (lasso → countdown →
capture → email typed key-by-key → `ENTER` → written to `captures/`) and
drops rendered frames in `simout/` so you can actually look at your change.

Two things this harness is genuinely good for:
- Driving the full loop including the lasso, by scripting hand landmarks
  around a circle.
- Proving gesture correctness — sweep a synthetic hand through 360° and
  assert the detected gesture at every angle.

**Its blind spot:** synthetic faces are not real faces. The ASCII look has
been tuned against synthetic tonal heads and a reference image, never against
a real webcam frame. If you're changing the portrait look, **get a real frame
first** — press `S` in the app, which writes to `screenshots/` where you can
read it. The user has been asked for this twice and it hasn't happened yet;
ask again rather than tuning blind.

## Storage

```
captures/
  captures.json              growing index, atomic write (tmp + replace)
  <stamp>_NN.png             ~1900px portrait, for emailing
  <stamp>_NN.txt             plain ASCII
  <stamp>_NN_photo.png       source crop
```

`captures/` is **gitignored — it holds real people's email addresses.**
Never commit it. Records carry `"emailed": false` for a future send script
(not written yet).

## Dependencies & models

opencv 4.13, mediapipe 0.10.33, numpy 2.4, Pillow. Both model files are
gitignored, so a fresh clone needs:

```bash
curl -fsSL -o blaze_face_short_range.tflite "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
```

`hand_landmarker.task` must also be present. `FaceTracker` falls back to
OpenCV's bundled Haar cascade if BlazeFace is missing — much worse under
backlight, so check the startup line says `Face backend: blazeface`.

## Performance

~12-28ms/frame headless (36-80fps headroom); real runs are camera-bound at
~20-25fps. BlazeFace is 5.6ms and runs every frame; Haar was 20-50ms and ran
every 2nd. `indices()` is ~2.4ms but runs **once per capture**, not per frame.

## Rejected approaches — don't re-try these

- **Additive glow / bloom on the portrait.** This is exactly what made it
  read as a "Photo Booth filter". The target aesthetic is flat ink on paper.
- **Directional edge glyphs** (`- \ | /` from a Sobel structure tensor).
  Added to compensate for a 56x28 grid being too coarse; at 128x64 features
  emerge from tone alone and the line glyphs were just noise. Resolution was
  the real fix. (It worked and was verified — it's simply unnecessary now.)
- **PyInstaller packaging** (`airwriter.spec`, `paths.py`). Existed briefly,
  removed at the user's request. Paths are plain `Path(__file__).parent`.

## Open / known-imperfect

- **Portrait look is unvalidated against a real face** (see Testing).
- `_ManualTrack` (the fallback when no face is detected inside a lasso)
  template-matches whatever was circled. A loose lasso that's mostly
  background will track the background. It auto-promotes to a real face lock
  once the detector finds a face nearby, so it's a bridge, not a solution.
- No email is actually *sent* — addresses and images are only collected.
- `MAX_SUBJECTS = 1` deliberately: two keyboard-focusable email fields with
  one keyboard is a worse demo, not a richer one.
