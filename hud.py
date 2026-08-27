"""
hud — the header strip above the camera image.

The window is letterboxed anyway, so instead of leaving dead grey space
we render an actual header band and composite the camera frame beneath
it.  Everything here uses the real-font helpers in `ui`.
"""

import cv2
import numpy as np
import math, time

from ui import text, text_width, pill

HEADER_H = 62

BG        = (22, 19, 28)
BG_EDGE   = (52, 48, 66)
DIM       = (118, 112, 136)
BRIGHT    = (232, 228, 240)
GOOD      = (150, 220, 180)
WARN      = (90, 190, 255)
BAD       = (90, 90, 245)

MODE_LABEL = {
    "point": ("DRAW",  None),
    "three": ("ERASE", (185, 185, 195)),
    "palm":  ("IDLE",  DIM),
    "fist":  ("CLEAR", BAD),
    "peace": ("COLOR", None),
}


def _fps_color(fps):
    return GOOD if fps >= 24 else WARN if fps >= 15 else BAD


def draw(bar, hands, n_active, palette, rainbow, rainbow_col, ar, rec,
         thick, fps, n_sub, n_faces):
    """Render the header in place.  `bar` is a (HEADER_H, W, 3) strip."""
    H, W = bar.shape[:2]
    bar[:] = BG
    cv2.line(bar, (0, H - 1), (W, H - 1), BG_EDGE, 1, cv2.LINE_AA)

    mid = H // 2
    accent = hands[0].color if n_active else DIM

    # ── wordmark ──
    cv2.circle(bar, (26, mid), 5, accent, -1, cv2.LINE_AA)
    text(bar, "AIR WRITER", (42, mid - 9), 14, BRIGHT, tracking=2)

    # ── mode chip ──
    g = hands[0].ge.gesture if n_active else "none"
    label, forced = MODE_LABEL.get(g, ("IDLE", DIM))
    mcol = forced or accent
    x = 176
    tw = text_width(label, 12, tracking=2)
    pill(bar, x, mid - 13, tw + 26, 26, mcol, alpha=0.18, border=mcol)
    text(bar, label, (x + 13, mid - 8), 12, mcol, tracking=2)
    x += tw + 26 + 14
    if n_active:
        text(bar, f"{n_active}H", (x, mid - 6), 11, DIM, tracking=1)
    x += 34

    # ── palette ──
    sel = {hs.cidx for hs in hands[:n_active]}
    for i, col in enumerate(palette):
        cx = x + i * 26
        on = i in sel
        cv2.circle(bar, (cx, mid), 8 if on else 5, col, -1, cv2.LINE_AA)
        if on:
            cv2.circle(bar, (cx, mid), 11, BRIGHT, 1, cv2.LINE_AA)
    x += len(palette) * 26 + 6
    if rainbow:
        text(bar, "RGB", (x, mid - 6), 11, rainbow_col, tracking=1)
    x += 42

    # ── thickness ──
    text(bar, "SIZE", (x, mid - 15), 9, DIM, tracking=1)
    bw = 64
    cv2.rectangle(bar, (x, mid + 1), (x + bw, mid + 7), (44, 41, 56), -1)
    fill = int(bw * min(1.0, max(0.0, (thick - 0.3) / 2.7)))
    if fill:
        cv2.rectangle(bar, (x, mid + 1), (x + fill, mid + 7), accent, -1)

    # ── right cluster ──
    rx = W - 22

    fps_s = f"{int(fps)}"
    fw = text_width(fps_s, 20, mono=False)
    rx -= fw
    text(bar, fps_s, (rx, mid - 14), 20, _fps_color(fps), mono=False)
    text(bar, "FPS", (rx, mid + 8), 8, DIM, tracking=1)
    rx -= 24

    def badge(label, value, active):
        nonlocal rx
        s = f"{label} {value}"
        w = text_width(s, 11, tracking=1)
        rx -= w + 22
        col = accent if active else DIM
        pill(bar, rx, mid - 12, w + 18, 24, col,
             alpha=0.16 if active else 0.07,
             border=col if active else None)
        text(bar, s, (rx + 9, mid - 7), 11, col, tracking=1)
        rx -= 10

    badge("FACE", n_faces, n_faces > 0)
    badge("LOCK", n_sub, n_sub > 0)

    if ar:
        s = "AR"
        w = text_width(s, 11, tracking=1)
        rx -= w + 22
        pill(bar, rx, mid - 12, w + 18, 24, GOOD, alpha=0.16, border=GOOD)
        text(bar, s, (rx + 9, mid - 7), 11, GOOD, tracking=1)
        rx -= 10

    if rec.recording:
        secs = int(rec.elapsed)
        s = f"REC {secs // 60}:{secs % 60:02d}"
        w = text_width(s, 11, tracking=1)
        rx -= w + 34
        pill(bar, rx, mid - 12, w + 30, 24, BAD, alpha=0.20, border=BAD)
        pulse = 0.45 + 0.55 * abs(math.sin(time.time() * 3.2))
        cv2.circle(bar, (rx + 13, mid), 4,
                   tuple(int(c * pulse) for c in BAD), -1, cv2.LINE_AA)
        text(bar, s, (rx + 22, mid - 7), 11, BAD, tracking=1)


def guide(view, hands, n_active, n_faces, stage="draw"):
    """Contextual tutorial card for people seeing Air Writer cold.

    The app is used on a projector, often without somebody explaining every
    gesture.  One changing instruction is easier to follow than a permanent
    wall of controls, while the compact legend keeps the gesture vocabulary
    discoverable.
    """
    H, W = view.shape[:2]
    g = hands[0].ge.gesture if n_active else "none"
    accent = hands[0].color if n_active else (255, 238, 120)

    if stage == "countdown":
        kicker, message = "STEP 2 OF 3  ·  CAPTURE", "HOLD STILL — MAKING YOUR ASCII PORTRAIT"
    elif stage == "email":
        kicker, message = "STEP 3 OF 3  ·  KEEP IT", "TYPE YOUR EMAIL, THEN PRESS ENTER"
    elif stage == "saved":
        kicker, message = "PORTRAIT SAVED", "MAKE A FIST TO RESET FOR THE NEXT PERSON"
        accent = GOOD
    elif g == "point":
        kicker, message = "STEP 1 OF 3  ·  DRAW", "TRACE A LOOP AROUND YOUR FACE"
    elif g == "peace":
        kicker, message = "COLOR CHANGED", "LOWER ONE FINGER TO START DRAWING"
    elif g == "three":
        kicker, message = "ERASER ACTIVE", "MOVE THREE FINGERS ACROSS THE INK"
    elif g == "palm":
        kicker, message = "PAUSED", "OPEN PALM IS YOUR SAFE IDLE GESTURE"
        accent = DIM
    elif g == "fist":
        kicker, message = "RESET", "KEEP HOLDING TO CLEAR FOR THE NEXT PERSON"
        accent = BAD
    elif n_faces:
        kicker, message = "START HERE", "RAISE 1 FINGER AND CIRCLE THE FACE FOUND BRACKETS"
    else:
        kicker, message = "AIR WRITER", "RAISE 1 FINGER TO DRAW IN THE AIR"

    x, h = 18, 76
    w = min(650, W - x * 2)
    y = H - h - 18
    pill(view, x, y, w, h, BG, alpha=0.88, border=(62, 58, 74), rad=12)
    cv2.line(view, (x, y + 1), (x + 78, y + 1), accent, 3, cv2.LINE_AA)
    text(view, kicker, (x + 16, y + 10), 10, accent, tracking=2)
    text(view, message, (x + 16, y + 29), 15, BRIGHT, mono=False)
    text(view, "1  DRAW     2  COLOR     3  ERASE     5  PAUSE",
         (x + 16, y + 55), 9, DIM, tracking=1)
