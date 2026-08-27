"""
Air Writer — AR Spatial Canvas v2.0
Draw neon ink in the air with full AR stabilisation, multi-hand
support, particle effects, rainbow mode, and session recording.

Gestures (per hand):
  ☝️  Point          → DRAW (pinch controls thickness)
  ✌️  Peace          → CYCLE COLOR
  3️⃣  Three fingers  → ERASE
  🖐️  Open palm      → IDLE / SAFE PAUSE
  ✊  Fist           → CLEAR CANVAS + release locks

Capture flow (the demo):
  1. Draw a closed loop around someone's face.
  2. Their ASCII portrait freezes in a card, black on nothing.
  3. Type an email into the field, press ENTER.
  4. Portrait + address land in captures/ so they can be mailed later.
  5. Hold a fist to clear, and the next person steps up.

Keyboard:
  S   Screenshot       R   Toggle recording (MP4 + GIF)
  Z   Undo             Y   Redo
  A   Toggle AR mode   B   Toggle rainbow mode
  H   Toggle HUD       M   Toggle mirror
  F   Portrait style   (CLASSIC / BLOCKS / MATRIX / HALFTONE / COLOR)
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

import hud
import capture
import mailer
from face_ascii import (
    FaceTracker, AsciiRenderer, SubjectManager,
    closed_loop, enclosed_faces, draw_candidates,
)

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

WINDOW_NAME = "Air Writer"
FIST_HOLD = 0.7        # seconds a fist must be held before it wipes anything
CAM_WIDTH, CAM_HEIGHT = 1280, 720
MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")

# Landmark indices
WRIST = 0
THUMB_TIP = 4;  THUMB_IP = 3
INDEX_TIP = 8;  INDEX_DIP = 7;  INDEX_PIP = 6;  INDEX_MCP = 5
MIDDLE_TIP = 12; MIDDLE_DIP = 11; MIDDLE_PIP = 10; MIDDLE_MCP = 9
RING_TIP = 16;  RING_DIP = 15;  RING_PIP = 14;  RING_MCP = 13
PINKY_TIP = 20; PINKY_DIP = 19; PINKY_PIP = 18; PINKY_MCP = 17

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
    def _d(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    @classmethod
    def _palm(cls, lm):
        return max(1e-6, cls._d(lm[WRIST], lm[MIDDLE_MCP]))

    @classmethod
    def _extended(cls, lm, tip, pip):
        """Is this finger out?  Rotation-invariant by construction.

        An extended fingertip sits farther from the wrist than its own PIP
        joint; a curled one folds back inside it.  Comparing image-space y
        instead (the old test) silently inverts the moment the hand tilts
        past horizontal, which read a genuine point as a fist.
        """
        w = lm[WRIST]
        return cls._d(lm[tip], w) > cls._d(lm[pip], w) * 1.10

    @classmethod
    def _thumb_out(cls, lm):
        return cls._d(lm[THUMB_TIP], lm[INDEX_MCP]) > cls._palm(lm) * 0.75

    def detect(self, lm):
        i = self._extended(lm, INDEX_TIP, INDEX_PIP)
        m = self._extended(lm, MIDDLE_TIP, MIDDLE_PIP)
        r = self._extended(lm, RING_TIP, RING_PIP)
        p = self._extended(lm, PINKY_TIP, PINKY_PIP)
        t = self._thumb_out(lm)
        if i and not m and not r and not p:
            raw = "point"
        elif i and m and not r and not p:
            raw = "peace"
        elif sum((i, m, r, p)) == 3:
            # Thumb detection is much less stable than the four long
            # fingers, so the eraser deliberately keys off exactly three
            # long fingers.  This stays reliable as the hand rotates.
            raw = "three"
        elif not (i or m or r or p):
            raw = "fist"
        elif i and m and r and p:
            # An open hand is a safe pause.  Requiring a trustworthy thumb
            # here made "five fingers" flicker as the palm turned sideways.
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
        self.stroke: list[tuple[int, int]] = []
        self.fist_since: float | None = None

    def cycle_color(self, now):
        if now - self._color_cd < 1.0:
            return
        self._color_cd = now
        self.cidx = (self.cidx + 1) % len(PALETTE)
        self.color = PALETTE[self.cidx]


# ──────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────

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
    cv2.resizeWindow(WINDOW_NAME, w, h + hud.HEADER_H)

    # one output buffer for header + frame, reused every frame
    out = np.zeros((h + hud.HEADER_H, w, 3), np.uint8)
    bar, view = out[:hud.HEADER_H], out[hud.HEADER_H:]

    # ── state ──
    canvas   = np.zeros((h, w, 3), np.uint8)
    hands    = [HandState(0), HandState(1)]
    history  = CanvasHistory(25)
    parts    = ParticleSystem(300)
    ar_stab  = ARStabilizer()
    rec      = SessionRecorder(Path(__file__).parent / "recordings")
    shots_dir = Path(__file__).parent / "screenshots"
    shots_dir.mkdir(exist_ok=True)

    faces    = FaceTracker(model_dir=Path(__file__).parent)
    store    = capture.CaptureStore(Path(__file__).parent / "captures")
    mail_cfg = mailer.load_config()
    sender   = mailer.EmailSender(mail_cfg)
    ascii_r  = AsciiRenderer()
    subjects = SubjectManager(ascii_r)
    print(f"[✓] Face backend: {faces.backend}")
    print(f"[✓] Captures on file: {store.count}")
    if sender.enabled:
        print(f"[✓] Emailing as: {mail_cfg['user']}")
    else:
        print("[·] Emailing off — collecting addresses only (see .env.example)")
    _owed = len(store.pending())
    if _owed:
        print(f"[·] {_owed} unsent from a previous run — run send_pending.py")

    rainbow  = False; ar_on = False
    hud_on   = True;  mirror = True
    thick    = 1.0;   hue_t  = 0.0
    clear_cd = 0.0

    fps = 0.0; ftimes = deque(maxlen=30)
    t0_mono = time.monotonic()

    print("[✓] Ready!  S=Screenshot  R=Record  Z/Y=Undo/Redo")
    print("    A=AR  B=Rainbow  F=Style  H=HUD  M=Mirror  C=Clear  X=Unlock  Q=Quit")
    print("    ◎ Circle a face → portrait freezes → type email → ENTER")
    print("    ☝ Draw  ·  ✌ Color  ·  3 fingers Erase  ·  🖐 Pause")
    print("    ✊ Hold a fist to clear for the next person.")

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            continue
        if mirror:
            frame = cv2.flip(frame, 1)

        faces.update(frame)

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
                    hs.stroke = []
                    hs.was_drawing = True
                col = rainbow_color(hue_t) if rainbow else hs.color
                if rainbow:
                    hue_t += dt * 80
                if hs.prev_pt is not None:
                    if math.hypot(sx-hs.prev_pt[0], sy-hs.prev_pt[1]) < 200:
                        neon_line(canvas, hs.prev_pt, (sx, sy), col, thick)
                hs.prev_pt = (sx, sy)
                hs.stroke.append((sx, sy))

                # ── lasso: a closed loop locks on to whatever it encircles ──
                if closed_loop(hs.stroke):
                    hit = enclosed_faces(hs.stroke, faces.lockable)
                    if not hit:
                        # no face recognised — lock the encircled region itself
                        xs = [p[0] for p in hs.stroke]
                        ys = [p[1] for p in hs.stroke]
                        bx, by = max(0, min(xs)), max(0, min(ys))
                        bw = min(w, max(xs)) - bx
                        bh = min(h, max(ys)) - by
                        # inset to the circle's inscribed box — the bounding
                        # box of a hand-drawn loop is mostly background
                        ix, iy = bx + bw * 0.14, by + bh * 0.14
                        iw, ih = bw * 0.72, bh * 0.72
                        if iw > 60 and ih > 60:
                            hit = [faces.add_manual((ix, iy, iw, ih), frame)]
                    if hit:
                        canvas = history.undo(canvas)   # loop was a gesture
                        for tr in hit[:2]:
                            act = subjects.toggle(tr, col)
                            kind = getattr(tr, "kind", "face")
                            if act == "released" and kind == "manual":
                                faces.drop_manual(tr.id)
                            print(f"[◎] Subject {act} ({kind})")
                        cx, cy = hit[0].center
                        parts.spawn(int(cx), int(cy), col, count=40)
                        hs.prev_pt = None
                        hs.was_drawing = False
                    hs.stroke = []

                # cursor
                cr = max(4, int(8 * thick))
                cv2.circle(frame, (sx, sy), cr, col, -1, cv2.LINE_AA)
                cv2.circle(frame, (sx, sy), max(2, cr//2),
                           (255,255,255), -1, cv2.LINE_AA)
                parts.spawn(sx, sy, col, count=2)
            else:
                hs.prev_pt = None
                hs.stroke = []
                if hs.was_drawing:
                    hs.was_drawing = False

            # ── erase: exactly three raised fingers ──
            if g == "three":
                ex = int(lm[MIDDLE_MCP].x * w)
                ey = int(lm[MIDDLE_MCP].y * h)
                er = 60
                cv2.circle(canvas, (ex, ey), er, (0,0,0), -1, cv2.LINE_AA)
                cv2.circle(frame, (ex, ey), er, (255,255,255), 2, cv2.LINE_AA)
                cv2.circle(frame, (ex, ey), er-4, (80,80,80), 1, cv2.LINE_AA)

            # ── clear (held, not tapped) ──
            # Clearing is the one destructive gesture, so it needs intent:
            # a fist must be held, and the ring shows how far along it is.
            if g == "fist":
                if hs.fist_since is None:
                    hs.fist_since = now
                held = now - hs.fist_since
                cx, cy = int(lm[MIDDLE_MCP].x * w), int(lm[MIDDLE_MCP].y * h)
                frac = min(1.0, held / FIST_HOLD)
                cv2.circle(frame, (cx, cy), 46, (60, 60, 75), 2, cv2.LINE_AA)
                if frac > 0:
                    cv2.ellipse(frame, (cx, cy), (46, 46), -90, 0,
                                int(360 * frac), (80, 80, 255), 3, cv2.LINE_AA)
                if held >= FIST_HOLD and now - clear_cd > 1.5:
                    history.save(canvas)
                    canvas = np.zeros((h, w, 3), np.uint8)
                    subjects.release_all(); faces.manual.clear()
                    clear_cd = now
                    hs.fist_since = None
                    parts.spawn(cx, cy, (80, 80, 255), count=30)
                    print("[✋] Cleared (fist held)")
            else:
                hs.fist_since = None

            # ── color cycle ──
            if g == "peace":
                hs.cycle_color(now)

        # reset inactive hands
        for hi in range(n_hands, 2):
            hs = hands[hi]
            hs.ge.gesture = "none"; hs.sm.reset(); hs.prev_pt = None
            hs.stroke = []; hs.fist_since = None
            if hs.was_drawing:
                hs.was_drawing = False

        # ── particles ──
        parts.tick(dt)

        # ── composite into the frame half of the output buffer ──
        cv2.add(frame, canvas, view)
        parts.draw(view)

        # ── send outcomes: the worker posts, the main thread writes ──
        for rec_id, st_, err, tries in sender.results():
            rec_ = store.mark(rec_id, st_, err)
            who = rec_["email"] if rec_ else rec_id
            if st_ == "sent":
                print(f"[✓] Emailed {who}")
            else:
                print(f"[!] Email to {who} failed after {tries}: {err}")
            for sb_ in subjects.subjects:
                if sb_.field is not None and sb_.sent_id == rec_id:
                    if st_ == "sent":
                        sb_.field.sent("SENT  ·  CHECK YOUR INBOX")
                    else:
                        # On disk either way; send_pending.py drains it later.
                        sb_.field.failed("SAVED  ·  WE'LL EMAIL IT LATER")

        # ── locked subjects: reticles + live ASCII cards ──
        subjects.update(frame, faces, dt)
        for sb in subjects.subjects:
            if sb.field is None:
                sb.field = capture.EmailField()
        if hud_on:
            draw_candidates(view, faces.visible,
                            {s.tid for s in subjects.subjects})
        subjects.draw(view, faces)

        if subjects.subjects:
            lead = subjects.subjects[0]
            if lead.idx is None:
                guide_stage = "countdown"
            elif lead.field is not None and lead.field.status == "saved":
                guide_stage = "saved"
            else:
                guide_stage = "email"
        else:
            guide_stage = "draw"
        if hud_on:
            hud.guide(view, hands, n_hands, faces.raw_count, guide_stage)

        # ── fps ──
        ftimes.append(time.time() - t0)
        fps = len(ftimes) / sum(ftimes) if len(ftimes) > 1 else 30.0

        hud.draw(bar, hands, n_hands, PALETTE, rainbow,
                 rainbow_color(hue_t), ar_on, rec, thick, fps,
                 subjects.active, faces.raw_count,
                 face_styles[subjects.style][0] if subjects.style else "")

        if rec.recording:
            rec.feed(out)

        cv2.imshow(WINDOW_NAME, out)

        # ── keyboard ──
        key = cv2.waitKey(1) & 0xFF

        # While an email field has focus every printable key belongs to it,
        # so the single-letter shortcuts below must not see them.  ESC drops
        # focus first, and only then quits.
        sub_ = subjects.subjects[0] if subjects.subjects else None
        fld = sub_.field if sub_ is not None else None
        if fld is not None and fld.active and key != 255:
            act = fld.key(key)
            if act == "submit":
                if sub_.idx is None:
                    # The field takes focus the moment a subject locks, which
                    # is before the 3-2-1 has run.  Submitting here would hand
                    # a None grid to compose() and kill the loop mid-demo.
                    fld.error("HOLD STILL  ·  ALMOST THERE")
                elif not capture.valid_email(fld.text):
                    fld.error("THAT DOESN'T LOOK LIKE AN EMAIL")
                else:
                    try:
                        r_ = store.save(fld.text, sub_.idx, ascii_r,
                                        sub_.photo, sub_.style)
                    except OSError as e:
                        # A full disk after a few hundred captures must not
                        # end the demo; the address is still on screen and
                        # ENTER can be pressed again.
                        print(f"[!] Could not save capture: {e}")
                        fld.error("COULDN'T SAVE  ·  PRESS ENTER TO RETRY")
                    else:
                        print(f"[✉️] {r_['email']} → {r_['image']}")
                        # Hand the worker copies and absolute paths only —
                        # never the live record or the store.
                        sub_.sent_id = r_["id"]
                        queued = sender.submit(
                            r_["id"], r_["email"],
                            store.dir / r_["image"],
                            ascii_r.to_text(sub_.idx),
                            r_.get("captured_at", ""))
                        if queued:
                            fld.sending("SENDING TO YOUR INBOX…")
                            fld.active = False
                        else:
                            fld.saved(f"SAVED  ·  {store.count} ON FILE")
            continue

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
                rec.start(w, h + hud.HEADER_H, fps=24)
                print("[⏺️]  Recording…")
        elif key in (ord("x"), ord("X")):
            faces.manual.clear()
            print(f"[◎] Cleared {subjects.release_all()} capture(s)")
        elif key in (ord("b"), ord("B")):
            rainbow = not rainbow
            print(f"[🌈] Rainbow {'ON' if rainbow else 'OFF'}")
        elif key in (ord("a"), ord("A")):
            ar_on = not ar_on
            if not ar_on: ar_stab.reset()
            print(f"[📌] AR {'ON' if ar_on else 'OFF'}")
        elif key in (ord("f"), ord("F")):
            subjects.style = (subjects.style + 1) % len(face_styles)
            subjects.restyle(subjects.style)
            print(f"[◈] Style: {face_styles[subjects.style][0]}")
        elif key in (ord("h"), ord("H")):
            hud_on = not hud_on
        elif key in (ord("m"), ord("M")):
            mirror = not mirror

    # ── cleanup ──
    if rec.recording:
        mp4, gif = rec.stop()
        if mp4: print(f"[🎬] MP4 → {mp4}")
    # Give an in-flight send a moment to land; the worker is a daemon, so a
    # hung socket can never stop the app from exiting.
    sender.stop(drain=1.5)
    cap.release(); cv2.destroyAllWindows(); det.close()
    print("[✦] Air Writer closed.")


if __name__ == "__main__":
    main()
