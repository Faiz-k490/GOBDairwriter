"""
Air Writer — AR Spatial Canvas v2.0
Draw neon ink in the air with full AR stabilisation, multi-hand
support, particle effects, rainbow mode, and session recording.

Gestures (per hand):
  ☝️  Point          → DRAW (pinch controls thickness)
  ✌️  Peace          → CYCLE COLOR
  ✊  Fist           → CLEAR CANVAS
  🖐️  Open palm      → ERASE

Keyboard:
  S   Screenshot       R   Toggle recording (MP4 + GIF)
  Z   Undo             Y   Redo
  A   Toggle AR mode   B   Toggle rainbow mode
  H   Toggle HUD       M   Toggle mirror
  C   Clear canvas     Q / ESC   Quit
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)
import time, os, math
from collections import deque
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

WINDOW_NAME = "Air Writer"
CAM_WIDTH, CAM_HEIGHT = 1280, 720
MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")

# Landmark indices
WRIST = 0
THUMB_TIP = 4;  THUMB_IP = 3
INDEX_TIP = 8;  INDEX_DIP = 7;  INDEX_PIP = 6;  INDEX_MCP = 5
MIDDLE_TIP = 12; MIDDLE_DIP = 11; MIDDLE_MCP = 9
RING_TIP = 16;  RING_DIP = 15;  RING_MCP = 13
PINKY_TIP = 20; PINKY_DIP = 19; PINKY_MCP = 17

# Neon palette (BGR)
PALETTE = [
    (255, 240,   0),   # Cyan
    (255,   0, 255),   # Magenta
    (  0, 255, 120),   # Lime
    (  0, 240, 255),   # Yellow
    (255, 130, 100),   # Light-blue
    (100, 100, 255),   # Salmon
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

COL_BG   = (15, 15, 22)
COL_TXT  = (220, 220, 230)
COL_DIM  = (100, 100, 120)


# ──────────────────────────────────────────────────────────────
# Particle system
# ──────────────────────────────────────────────────────────────

class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")
    def __init__(self, x, y, vx, vy, life, color, size):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = self.max_life = life
        self.color, self.size = color, size


class ParticleSystem:
    def __init__(self, cap=300):
        self.particles: list[Particle] = []
        self._cap = cap

    def spawn(self, x, y, color, count=3):
        import random
        for _ in range(count):
            a = random.uniform(0, 2 * math.pi)
            s = random.uniform(0.5, 3.0)
            self.particles.append(
                Particle(x, y, math.cos(a)*s, math.sin(a)*s,
                         random.uniform(0.3, 0.8), color,
                         random.uniform(1.5, 4.0))
            )
        if len(self.particles) > self._cap:
            self.particles = self.particles[-self._cap:]

    def tick(self, dt):
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life > 0:
                p.x += p.vx; p.y += p.vy; p.vy += 0.5
                alive.append(p)
        self.particles = alive

    def draw(self, img):
        for p in self.particles:
            a = max(0.0, p.life / p.max_life)
            c = tuple(int(ch * a) for ch in p.color)
            r = max(1, int(p.size * a))
            cv2.circle(img, (int(p.x), int(p.y)), r, c, -1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# Canvas history  (undo / redo)
# ──────────────────────────────────────────────────────────────

class CanvasHistory:
    def __init__(self, limit=25):
        self._undo: deque[np.ndarray] = deque(maxlen=limit)
        self._redo: deque[np.ndarray] = deque(maxlen=limit)

    def save(self, canvas):
        self._undo.append(canvas.copy())
        self._redo.clear()

    def undo(self, canvas):
        if self._undo:
            self._redo.append(canvas.copy())
            return self._undo.pop()
        return canvas

    def redo(self, canvas):
        if self._redo:
            self._undo.append(canvas.copy())
            return self._redo.pop()
        return canvas


# ──────────────────────────────────────────────────────────────
# AR stabiliser  (optical-flow based)
# ──────────────────────────────────────────────────────────────

class ARStabilizer:
    def __init__(self):
        self._prev_gray = None
        self._prev_pts = None
        self._feat = dict(maxCorners=200, qualityLevel=0.01,
                          minDistance=30, blockSize=7)
        self._lk = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS |
                                  cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self._refresh = 30
        self._tick = 0

    # noinspection PyUnresolvedReferences
    def _hand_mask(self, shape, hands, w, h):
        mask = np.ones(shape[:2], np.uint8) * 255
        for lms in hands:
            xs = [int(lm.x * w) for lm in lms]
            ys = [int(lm.y * h) for lm in lms]
            if xs and ys:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                r = max(max(xs)-min(xs), max(ys)-min(ys))//2 + 80
                cv2.circle(mask, (cx, cy), r, 0, -1)
        return mask

    def process(self, frame, canvas, hands, w, h):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._tick += 1

        if self._prev_gray is None:
            self._prev_gray = gray
            m = self._hand_mask(gray.shape, hands, w, h)
            self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=m,
                                                     **self._feat)
            return canvas

        if self._prev_pts is None or len(self._prev_pts) < 10:
            m = self._hand_mask(gray.shape, hands, w, h)
            self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=m,
                                                     **self._feat)
            self._prev_gray = gray
            return canvas

        new_pts, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk)

        if new_pts is not None and st is not None:
            ok = st.flatten() == 1
            old_g, new_g = self._prev_pts[ok], new_pts[ok]
            if len(old_g) >= 4:
                M, _ = cv2.estimateAffinePartial2D(old_g, new_g)
                if M is not None:
                    canvas = cv2.warpAffine(canvas, M, (w, h),
                                            borderMode=cv2.BORDER_CONSTANT,
                                            borderValue=(0, 0, 0))
                self._prev_pts = new_g.reshape(-1, 1, 2)
            else:
                self._prev_pts = None

        if self._tick % self._refresh == 0:
            m = self._hand_mask(gray.shape, hands, w, h)
            self._prev_pts = cv2.goodFeaturesToTrack(gray, mask=m,
                                                     **self._feat)
        self._prev_gray = gray
        return canvas

    def reset(self):
        self._prev_gray = None
        self._prev_pts = None


# ──────────────────────────────────────────────────────────────
# Session recorder  (MP4 + optional GIF export)
# ──────────────────────────────────────────────────────────────

class SessionRecorder:
    def __init__(self, out_dir):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.recording = False
        self._writer = None
        self._gif_buf: list[np.ndarray] = []
        self._n = 0; self._t0 = 0.0; self._path = None

    def start(self, w, h, fps=24):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self.dir / f"airwriter_{ts}.mp4"
        self._writer = cv2.VideoWriter(
            str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        self._gif_buf.clear(); self._n = 0
        self._t0 = time.time(); self.recording = True

    def feed(self, frame):
        if not self.recording: return
        self._writer.write(frame); self._n += 1
        if self._n % 4 == 0:
            self._gif_buf.append(cv2.resize(frame, (480, 270)))

    def stop(self):
        if not self.recording:
            return None, None
        self.recording = False
        if self._writer:
            self._writer.release(); self._writer = None
        mp4 = self._path; gif = None
        if HAS_PIL and self._gif_buf:
            gif = mp4.with_suffix(".gif")
            pil = [PILImage.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                   for f in self._gif_buf]
            pil[0].save(str(gif), save_all=True, append_images=pil[1:],
                        duration=100, loop=0, optimize=True)
        self._gif_buf.clear()
        return mp4, gif

    @property
    def elapsed(self):
        return (time.time() - self._t0) if self.recording else 0.0


# ──────────────────────────────────────────────────────────────
# Gesture engine  (debounced, per-hand)
# ──────────────────────────────────────────────────────────────

class GestureEngine:
    def __init__(self, threshold=3):
        self.gesture = "none"
        self._prev = "none"; self._hold = 0; self._th = threshold

    @staticmethod
    def _up(lm, tip, dip, mcp):
        return lm[tip].y < lm[dip].y and lm[dip].y < lm[mcp].y

    @staticmethod
    def _thumb_up(lm):
        return abs(lm[THUMB_TIP].x - lm[WRIST].x) > 0.06

    def detect(self, lm):
        i = self._up(lm, INDEX_TIP, INDEX_DIP, INDEX_MCP)
        m = self._up(lm, MIDDLE_TIP, MIDDLE_DIP, MIDDLE_MCP)
        r = self._up(lm, RING_TIP, RING_DIP, RING_MCP)
        p = self._up(lm, PINKY_TIP, PINKY_DIP, PINKY_MCP)
        t = self._thumb_up(lm)
        up = sum((i, m, r, p, t))

        if i and not m and not r and not p:
            raw = "point"
        elif i and m and not r and not p:
            raw = "peace"
        elif (not i and not m and not r and not p
              and lm[INDEX_TIP].y > lm[INDEX_MCP].y):
            raw = "fist"
        elif up >= 4:
            raw = "palm"
        else:
            raw = "other"

        if raw == self._prev:
            self._hold = min(self._hold + 1, self._th + 1)
        else:
            self._hold = 0; self._prev = raw
        if self._hold >= self._th:
            self.gesture = raw
        return self.gesture


# ──────────────────────────────────────────────────────────────
# Coordinate smoother
# ──────────────────────────────────────────────────────────────

class Smoother:
    def __init__(self, f=0.5):
        self.f = f; self.x = self.y = None
    def update(self, x, y):
        if self.x is None:
            self.x, self.y = float(x), float(y)
        else:
            self.x += self.f * (x - self.x)
            self.y += self.f * (y - self.y)
        return int(self.x), int(self.y)
    def reset(self):
        self.x = self.y = None


# ──────────────────────────────────────────────────────────────
# Per-hand state
# ──────────────────────────────────────────────────────────────

class HandState:
    def __init__(self, cidx=0):
        self.ge = GestureEngine()
        self.sm = Smoother(0.5)
        self.prev_pt = None
        self.cidx = cidx
        self.color = PALETTE[cidx % len(PALETTE)]
        self.was_drawing = False
        self._color_cd = 0.0

    def cycle_color(self, now):
        if now - self._color_cd < 1.0:
            return
        self._color_cd = now
        self.cidx = (self.cidx + 1) % len(PALETTE)
        self.color = PALETTE[self.cidx]


# ──────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────

def put_text(img, txt, pos, sc=0.5, col=COL_TXT, th=1):
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th,
                cv2.LINE_AA)

def rounded_rect(img, p1, p2, color, rad=15, alpha=0.6):
    ov = img.copy()
    x1, y1 = p1; x2, y2 = p2
    cv2.rectangle(ov, (x1+rad, y1), (x2-rad, y2), color, -1)
    cv2.rectangle(ov, (x1, y1+rad), (x2, y2-rad), color, -1)
    for cx, cy in ((x1+rad,y1+rad),(x2-rad,y1+rad),
                   (x1+rad,y2-rad),(x2-rad,y2-rad)):
        cv2.circle(ov, (cx, cy), rad, color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def neon_line(canvas, p1, p2, color, thickness=1.0):
    """Layered glow: dim outer → bright inner → white core."""
    outer = max(4, int(14 * thickness))
    inner = max(2, int(6 * thickness))
    core  = max(1, int(2 * thickness))
    dim   = tuple(int(c * 0.35) for c in color)
    cv2.line(canvas, p1, p2, dim, outer, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, color, inner, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, (255, 255, 255), core, cv2.LINE_AA)


def rainbow_color(t):
    hue = int(t * 60) % 180
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0][0])


def pinch_thickness(lm):
    d = math.hypot(lm[THUMB_TIP].x - lm[INDEX_TIP].x,
                   lm[THUMB_TIP].y - lm[INDEX_TIP].y)
    return float(np.interp(d, [0.03, 0.18], [0.3, 3.0]))


# ──────────────────────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────────────────────

def draw_hud(frame, w, h, hands, n_active, rainbow, ar, rec, thick, fps):
    pw, ph = 540, 78
    x, y = (w - pw) // 2, h - ph - 12
    rounded_rect(frame, (x, y), (x+pw, y+ph), COL_BG, rad=18, alpha=0.65)
    cv2.rectangle(frame, (x, y), (x+pw, y+ph), (55, 55, 70), 1, cv2.LINE_AA)

    # — mode —
    g = hands[0].ge.gesture if n_active > 0 else "none"
    pc = hands[0].color if n_active > 0 else COL_DIM
    labels = {"point": ("DRAW", pc), "palm": ("ERASE", (160,160,160)),
              "fist": ("CLEAR", (80,80,255)), "peace": ("COLOR", pc)}
    mtxt, mcol = labels.get(g, ("IDLE", COL_DIM))
    put_text(frame, mtxt, (x+18, y+30), 0.55, mcol, 2)
    if n_active:
        put_text(frame, f"{n_active}H", (x+18, y+55), 0.32, COL_DIM)

    # — palette dots —
    px, py = x + 120, y + 28
    for i, pc in enumerate(PALETTE):
        sel = any(hs.cidx == i for hs in hands[:n_active])
        r = 8 if sel else 5
        cv2.circle(frame, (px + i*24, py), r, pc, -1, cv2.LINE_AA)
        if sel:
            cv2.circle(frame, (px + i*24, py), r+2, (255,255,255), 1,
                       cv2.LINE_AA)
    if rainbow:
        put_text(frame, "RAINBOW", (px, py+22), 0.3,
                 rainbow_color(time.time()), 1)

    # — thickness bar —
    bx, by = x + 290, y + 20
    put_text(frame, "T", (bx, by+3), 0.35, COL_DIM)
    bs = bx + 15; bw = 55
    cv2.rectangle(frame, (bs, by-2), (bs+bw, by+5), (40,40,55), -1)
    f = int(bw * min(1.0, (thick - 0.3) / 2.7))
    cv2.rectangle(frame, (bs, by-2), (bs+f, by+5),
                  hands[0].color if n_active else COL_DIM, -1)

    # — AR tag —
    ac = (0, 220, 180) if ar else (50, 50, 60)
    put_text(frame, "AR", (bx, by+28), 0.35, ac)

    # — recording —
    rx = x + 400
    if rec.recording:
        pulse = int(200 + 55 * math.sin(time.time() * 4))
        cv2.circle(frame, (rx, y+23), 6, (0, 0, pulse), -1, cv2.LINE_AA)
        s = int(rec.elapsed); mm, ss = s // 60, s % 60
        put_text(frame, f"REC {mm}:{ss:02d}", (rx+12, y+28), 0.35,
                 (0, 0, 220))

    # — fps —
    fx = x + pw - 60
    fc = (80,200,120) if fps >= 24 else (60,180,255) if fps >= 15 \
         else (80,80,255)
    put_text(frame, f"{int(fps)}", (fx, y+30), 0.55, fc, 2)
    put_text(frame, "FPS", (fx+4, y+50), 0.28, COL_DIM)


# ──────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}"); return

    print("╔════════════════════════════════════════════╗")
    print("║    ✦  A I R   W R I T E R   v 2 . 0  ✦   ║")
    print("╚════════════════════════════════════════════╝")

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    det = HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    w, h = CAM_WIDTH, CAM_HEIGHT
    for _ in range(30):
        ok, t = cap.read()
        if ok and t is not None:
            h, w = t.shape[:2]; break
        time.sleep(0.1)
    print(f"[✓] Webcam: {w}×{h}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, w, h)

    # ── state ──
    canvas   = np.zeros((h, w, 3), np.uint8)
    hands    = [HandState(0), HandState(1)]
    history  = CanvasHistory(25)
    parts    = ParticleSystem(300)
    ar_stab  = ARStabilizer()
    rec      = SessionRecorder(Path(__file__).parent / "recordings")
    shots_dir = Path(__file__).parent / "screenshots"
    shots_dir.mkdir(exist_ok=True)

    rainbow  = False; ar_on = False
    hud_on   = True;  mirror = True
    thick    = 1.0;   hue_t  = 0.0
    clear_cd = 0.0

    fps = 0.0; ftimes = deque(maxlen=30)
    t0_mono = time.monotonic()

    print("[✓] Ready!  S=Screenshot  R=Record  Z/Y=Undo/Redo")
    print("    A=AR  B=Rainbow  H=HUD  M=Mirror  C=Clear  Q=Quit")

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            continue
        if mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts  = int((time.monotonic() - t0_mono) * 1000)
        res = det.detect_for_video(mpi, ts)

        lm_list = res.hand_landmarks if res.hand_landmarks else []
        n_hands = len(lm_list)
        now = time.time()
        dt  = ftimes[-1] if ftimes else 0.033

        # ── AR stabilisation ──
        if ar_on:
            canvas = ar_stab.process(frame, canvas, lm_list, w, h)

        # ── per-hand processing ──
        for hi in range(min(n_hands, 2)):
            lm = lm_list[hi]
            hs = hands[hi]
            g  = hs.ge.detect(lm)

            rx, ry = int(lm[INDEX_TIP].x * w), int(lm[INDEX_TIP].y * h)
            sx, sy = hs.sm.update(rx, ry)

            thick = pinch_thickness(lm)

            # dim skeleton
            if hud_on:
                pts = [(int(l.x*w), int(l.y*h)) for l in lm]
                sc = hs.color if g == "point" else (70, 70, 90)
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], sc, 1, cv2.LINE_AA)

            # ── draw ──
            if g == "point":
                if not hs.was_drawing:
                    history.save(canvas)
                    hs.was_drawing = True
                col = rainbow_color(hue_t) if rainbow else hs.color
                if rainbow:
                    hue_t += dt * 80
                if hs.prev_pt is not None:
                    if math.hypot(sx-hs.prev_pt[0], sy-hs.prev_pt[1]) < 200:
                        neon_line(canvas, hs.prev_pt, (sx, sy), col, thick)
                hs.prev_pt = (sx, sy)
                # cursor
                cr = max(4, int(8 * thick))
                cv2.circle(frame, (sx, sy), cr, col, -1, cv2.LINE_AA)
                cv2.circle(frame, (sx, sy), max(2, cr//2),
                           (255,255,255), -1, cv2.LINE_AA)
                parts.spawn(sx, sy, col, count=2)
            else:
                hs.prev_pt = None
                if hs.was_drawing:
                    hs.was_drawing = False

            # ── erase ──
            if g == "palm":
                ex = int(lm[MIDDLE_MCP].x * w)
                ey = int(lm[MIDDLE_MCP].y * h)
                er = 60
                cv2.circle(canvas, (ex, ey), er, (0,0,0), -1, cv2.LINE_AA)
                cv2.circle(frame, (ex, ey), er, (255,255,255), 2, cv2.LINE_AA)
                cv2.circle(frame, (ex, ey), er-4, (80,80,80), 1, cv2.LINE_AA)

            # ── clear ──
            if g == "fist" and now - clear_cd > 1.5:
                history.save(canvas)
                canvas = np.zeros((h, w, 3), np.uint8)
                clear_cd = now

            # ── color cycle ──
            if g == "peace":
                hs.cycle_color(now)

        # reset inactive hands
        for hi in range(n_hands, 2):
            hs = hands[hi]
            hs.ge.gesture = "none"; hs.sm.reset(); hs.prev_pt = None
            if hs.was_drawing:
                hs.was_drawing = False

        # ── particles ──
        parts.tick(dt)

        # ── composite ──
        out = cv2.add(frame, canvas)
        parts.draw(out)

        # ── fps ──
        ftimes.append(time.time() - t0)
        fps = len(ftimes) / sum(ftimes) if len(ftimes) > 1 else 30.0

        if hud_on:
            draw_hud(out, w, h, hands, n_hands, rainbow, ar_on, rec,
                     thick, fps)

        if rec.recording:
            rec.feed(out)

        cv2.imshow(WINDOW_NAME, out)

        # ── keyboard ──
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("c"), ord("C")):
            history.save(canvas)
            canvas = np.zeros((h, w, 3), np.uint8)
        elif key in (ord("z"), ord("Z")):
            canvas = history.undo(canvas)
        elif key in (ord("y"), ord("Y")):
            canvas = history.redo(canvas)
        elif key in (ord("s"), ord("S")):
            p = shots_dir / f"air_{datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(str(p), out)
            print(f"[📸] Saved {p}")
        elif key in (ord("r"), ord("R")):
            if rec.recording:
                mp4, gif = rec.stop()
                print(f"[🎬] MP4 → {mp4}")
                if gif: print(f"[🎞️] GIF → {gif}")
            else:
                rec.start(w, h, fps=24)
                print("[⏺️]  Recording…")
        elif key in (ord("b"), ord("B")):
            rainbow = not rainbow
            print(f"[🌈] Rainbow {'ON' if rainbow else 'OFF'}")
        elif key in (ord("a"), ord("A")):
            ar_on = not ar_on
            if not ar_on: ar_stab.reset()
            print(f"[📌] AR {'ON' if ar_on else 'OFF'}")
        elif key in (ord("h"), ord("H")):
            hud_on = not hud_on
        elif key in (ord("m"), ord("M")):
            mirror = not mirror

    # ── cleanup ──
    if rec.recording:
        mp4, gif = rec.stop()
        if mp4: print(f"[🎬] MP4 → {mp4}")
    cap.release(); cv2.destroyAllWindows(); det.close()
    print("[✦] Air Writer closed.")


if __name__ == "__main__":
    main()
