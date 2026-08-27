"""
face_ascii — lasso a face, lock on, render it as live ASCII.

Three pieces:
  FaceTracker    persistent, smoothed face boxes with stable IDs
  AsciiRenderer  glyph-atlas ASCII art (vectorised, runs every frame)
  SubjectManager the lock-on lifecycle: reticles, cards, connectors

Detection uses OpenCV's bundled Haar cascade so there is nothing to
download.  Drop `blaze_face_short_range.tflite` next to the hand model
and it will be used instead — noticeably steadier on profile views.
"""

import cv2
import numpy as np
import math, time
from pathlib import Path

from PIL import Image as PILImage, ImageDraw

from ui import TEXT, blit, text, text_width, pill

# ──────────────────────────────────────────────────────────────
# Tunables
# ──────────────────────────────────────────────────────────────

# The ramp is derived from measured ink coverage at runtime (see
# AsciiRenderer._build_ramp) — hand-picked ramps are not monotonic in
# brightness, which shows up as banding and speckle on smooth skin.
RAMP_LEVELS  = 12

# Ink on paper, not glow on black.  The portrait is meant to read like
# something printed, so dark glyphs sit on a light ground and there is no
# bloom pass at all — bloom is what made the old version look like a filter.
INK          = (34, 32, 30)
PAPER        = (236, 236, 232)

ASCII_COLS   = 128
ASCII_ROWS   = 64
CELL_W, CELL_H = 4, 8

# Structure comes from resolution here, not from tricks: at 128x64 the face
# emerges from tone alone, so the old edge-glyph pass is gone.
# 2.05 swallowed the whole room: at 128x64 the face got ~40x25 cells and
# the eyes had nowhere to live.  A portrait wants the head to OWN the frame.
CROP_PAD     = 1.45
CROP_DROP    = 0.12        # nudge the frame down; faces sit high in the box

INK_FLOOR    = 0.10        # highlights below this leave bare paper
INK_GAMMA    = 0.88
EDGE_MIX     = 0.24        # facial edges survive even on evenly-lit skin
VIGNETTE     = 0.72        # full-strength centre of the portrait mask
VIGNETTE_POW = 1.20

COUNTDOWN    = 1.2         # seconds of 3-2-1 before the shutter fires

CARD_PAD     = 12
CARD_HEAD    = 30
FIELD_H      = 40          # email input
FOOT_H       = 24          # status line under the field
CARD_W       = ASCII_COLS * CELL_W + CARD_PAD * 2
CARD_H       = (ASCII_ROWS * CELL_H + CARD_PAD * 3 + CARD_HEAD
                + FIELD_H + FOOT_H)
CARD_MARGIN  = 18
CARD_GAP     = 14

# One at a time: the card carries a keyboard-focused email field, and two
# focusable fields with one keyboard is a worse demo, not a richer one.
# ── portrait styles ────────────────────────────────────────────────────
#
# Every pool goes through _build_ramp(), which measures real ink coverage at
# the live font and cell size and picks evenly-spaced glyphs.  Hand-picking a
# ramp is what produces banding: " .:-=+*#%@" is not monotonic ('-' carries
# less ink than ':', '+' less than '=').  Add pools, never hand-ordered ramps.
#
# ink/paper are passed straight to compose(), which already takes both, so a
# style that only recolours costs nothing but a dict lookup.

STYLES = [
    # name        ink              paper            pool
    ("CLASSIC",  (34, 32, 30),   (236, 236, 232), " .,:;-~=+*ox%#@"),
    ("BLOCKS",   (28, 26, 24),   (240, 240, 236), " .:-=+*#%@█▓▒░"),
    ("MATRIX",   (120, 255, 140), (8, 14, 8),     " .:-=+*01#%@$&"),
    ("HALFTONE", (30, 28, 34),   (242, 240, 238), " .,·:;∘o*O0@#%"),
    ("COLOR",    (34, 32, 30),   (236, 236, 232), " .,:;-~=+*ox%#@"),
]
STYLE_COLOR = 4          # the one that samples the subject's real colours

MAX_SUBJECTS = 1

LOST_GRACE   = 1.6          # seconds a subject survives without detection
SLIDE_TIME   = 0.30         # card slide-in duration


# ──────────────────────────────────────────────────────────────
# Face tracking
# ──────────────────────────────────────────────────────────────

class _Track:
    __slots__ = ("id", "box", "last_seen", "hits", "confirmed")

    def __init__(self, tid, box, now):
        self.id = tid
        self.box = np.array(box, np.float32)   # x, y, w, h
        self.last_seen = now
        self.hits = 1
        self.confirmed = False

    def blend(self, box, now, f=0.45):
        self.box += f * (np.array(box, np.float32) - self.box)
        self.last_seen = now
        self.hits += 1
        if self.hits >= 3:
            self.confirmed = True

    @property
    def rect(self):
        return tuple(int(v) for v in self.box)

    @property
    def center(self):
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)


class _ManualTrack:
    """A lassoed region with no face in it — followed by template matching."""

    SCALE = 0.25         # quarter-res is both the fastest matchTemplate size and plenty
    MOVE = 0.8           # near-direct follow; heavy damping deadlocks on the grid
    REFRESH = 2.0        # seconds between template refreshes

    __slots__ = ("id", "kind", "confirmed", "box", "_tpl", "score",
                 "last_seen", "_refreshed")

    def __init__(self, tid, box, frame):
        self.id = tid
        self.kind = "manual"
        self.confirmed = True
        self.box = np.array(box, np.float32)
        self.score = 1.0
        self.last_seen = time.time()
        self._refreshed = self.last_seen
        self._tpl = None
        self._grab(frame)

    def _small(self, frame):
        return cv2.cvtColor(
            cv2.resize(frame, None, fx=self.SCALE, fy=self.SCALE,
                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    def _grab(self, frame):
        g = self._small(frame)
        x, y, w, h = (self.box * self.SCALE).astype(int)
        x, y = max(0, x), max(0, y)
        w, h = max(8, min(w, g.shape[1] - x)), max(8, min(h, g.shape[0] - y))
        if w >= 8 and h >= 8:
            self._tpl = g[y:y + h, x:x + w].copy()

    def update(self, frame):
        if self._tpl is None:
            return
        g = self._small(frame)
        th, tw = self._tpl.shape
        x, y, w, h = (self.box * self.SCALE).astype(int)
        # search a margin around the last known position, not the whole frame
        mx, my = int(tw * 0.6), int(th * 0.6)
        sx0, sy0 = max(0, x - mx), max(0, y - my)
        sx1, sy1 = min(g.shape[1], x + tw + mx), min(g.shape[0], y + th + my)
        roi = g[sy0:sy1, sx0:sx1]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return
        res = cv2.matchTemplate(roi, self._tpl, cv2.TM_CCOEFF_NORMED)
        _, self.score, _, loc = cv2.minMaxLoc(res)
        if self.score < 0.35:
            return                                  # too weak — hold position
        nx = (sx0 + loc[0]) / self.SCALE
        ny = (sy0 + loc[1]) / self.SCALE
        self.box[0] += self.MOVE * (nx - self.box[0])
        self.box[1] += self.MOVE * (ny - self.box[1])
        now = time.time()
        self.last_seen = now
        # Refresh on a timer, never per-frame: re-grabbing every frame lets the
        # template re-anchor to wherever the box already is and freeze there.
        if self.score > 0.6 and now - self._refreshed > self.REFRESH:
            self._refreshed = now
            self._grab(frame)

    @property
    def rect(self):
        return tuple(int(v) for v in self.box)

    @property
    def center(self):
        x, y, w, h = self.box
        return (x + w / 2, y + h / 2)


class FaceTracker:
    """Detect-every-N-frames plus EMA smoothing and ID persistence."""

    def __init__(self, model_dir=None, every=2, det_width=480):
        self.every = every
        self.det_width = det_width
        self._tick = 0
        self._next_id = 1
        self.tracks: list[_Track] = []
        self.manual: list[_ManualTrack] = []
        self.raw_count = 0
        self.backend = "haar"
        self._mp = None

        blaze = Path(model_dir or ".") / "blaze_face_short_range.tflite"
        if blaze.exists():
            try:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import (
                    FaceDetector, FaceDetectorOptions, RunningMode,
                )
                self._mp = FaceDetector.create_from_options(
                    FaceDetectorOptions(
                        base_options=BaseOptions(model_asset_path=str(blaze)),
                        running_mode=RunningMode.IMAGE,
                        min_detection_confidence=0.5,
                    )
                )
                self.backend = "blazeface"
                self.every = 1        # ~5ms — cheap enough to run every frame
            except Exception:
                self._mp = None

        if self._mp is None:
            # alt2 is markedly better than `default` on backlit / tilted faces
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            self._haar2 = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self._det_clahe = cv2.createCLAHE(clipLimit=3.0,
                                              tileGridSize=(8, 8))

    # ── detection ──

    def _detect(self, frame):
        H, W = frame.shape[:2]
        scale = self.det_width / float(W)
        small = cv2.resize(frame, (self.det_width, int(H * scale)))

        if self._mp is not None:
            import mediapipe as mp
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = self._mp.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            boxes = []
            for d in res.detections:
                bb = d.bounding_box
                boxes.append((bb.origin_x, bb.origin_y, bb.width, bb.height))
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            # CLAHE, not equalizeHist — backlit scenes wreck a global histogram
            gray = self._det_clahe.apply(gray)
            mn = (int(self.det_width * 0.07),) * 2
            boxes = [tuple(b) for b in self._haar.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=mn)]
            if not boxes:                      # second opinion before giving up
                boxes = [tuple(b) for b in self._haar2.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=mn)]

        inv = 1.0 / scale
        return [(x * inv, y * inv, w * inv, h * inv) for x, y, w, h in boxes]

    # ── association ──

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        if inter <= 0:
            return 0.0
        return inter / (aw * ah + bw * bh - inter)

    def update(self, frame):
        now = time.time()
        self._tick += 1
        if self._tick % self.every == 0:
            dets = self._detect(frame)
            unmatched = list(range(len(dets)))

            for tr in self.tracks:
                best, best_iou = -1, 0.25
                for di in unmatched:
                    v = self._iou(tr.box, dets[di])
                    if v > best_iou:
                        best, best_iou = di, v
                if best >= 0:
                    tr.blend(dets[best], now)
                    unmatched.remove(best)

            for di in unmatched:
                self.tracks.append(_Track(self._next_id, dets[di], now))
                self._next_id += 1

        self.tracks = [t for t in self.tracks if now - t.last_seen < LOST_GRACE]
        self.raw_count = len(self.visible)

        for m in self.manual:
            m.update(frame)
        return self.tracks

    def add_manual(self, box, frame):
        m = _ManualTrack(self._next_id, box, frame)
        self._next_id += 1
        self.manual.append(m)
        return m

    def drop_manual(self, tid):
        self.manual = [m for m in self.manual if m.id != tid]

    def get(self, tid):
        for t in self.tracks:
            if t.id == tid:
                return t
        for m in self.manual:
            if m.id == tid:
                return m
        return None

    @property
    def visible(self):
        """Confirmed *face* tracks — what the lasso snaps to."""
        return [t for t in self.tracks if t.confirmed]

    @property
    def lockable(self):
        """Everything a lasso can toggle, faces and manual regions alike."""
        return self.visible + self.manual


# ──────────────────────────────────────────────────────────────
# ASCII renderer
# ──────────────────────────────────────────────────────────────

class AsciiRenderer:
    """Pre-baked glyph atlas → one vectorised reshape per frame."""

    # A deliberately small visual vocabulary reads as intentional ASCII,
    # rather than a soup of random letters.  Coverage is still measured at
    # runtime, so the ramp remains monotonic for the active font and size.
    POOL = " .,:;-~=+*ox%#@"

    def __init__(self, levels=RAMP_LEVELS, cell=(CELL_W, CELL_H)):
        self.cw, self.ch = cell
        self.ramp = self._build_ramp(levels)
        self.atlas = self._bake(self.ramp)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._vig = self._vignette(ASCII_COLS, ASCII_ROWS)
        # One ramp+atlas per style, built once.  Keyed by (style, cell) so the
        # 14x28 export bake is cached too and a save does not re-render glyphs.
        self._styles = {}
        self._levels = levels

    # ── styles ──

    def style(self, i, cell=None):
        """(ramp, atlas) for style i at a cell size.  Built once, then cached."""
        i = i % len(STYLES)
        key = (i, cell)
        got = self._styles.get(key)
        if got is None:
            pool = STYLES[i][3]
            if cell is None and i == 0:
                # Style 0 is what __init__ already built; do not rebuild it.
                got = (self.ramp, self.atlas)
            else:
                ramp = self._build_ramp(self._levels, pool)
                got = (ramp, self._bake(ramp, cell))
            self._styles[key] = got
        return got

    def compose_style(self, idx, i, cell=None):
        """Render a frozen grid in style i."""
        ink, paper = STYLES[i % len(STYLES)][1], STYLES[i % len(STYLES)][2]
        ramp, atlas = self.style(i, cell)
        return self._compose_atlas(idx, atlas, ink, paper)

    def compose_color(self, idx, src, i=STYLE_COLOR, cell=None):
        """Style i, but every glyph tinted by the mean colour of its cell.

        cv2.resize with INTER_AREA *is* the per-cell mean: box-averaging over
        each source block is exactly that average, so there is no separate
        sampling pass.
        """
        rows, cols = idx.shape
        ramp, atlas = self.style(i, cell)
        n, ch, cw = atlas.shape
        alpha = np.clip(atlas[idx].transpose(0, 2, 1, 3)
                        .reshape(rows * ch, cols * cw) * 1.35, 0, 1)[..., None]

        cell_bgr = cv2.resize(src, (cols, rows), interpolation=cv2.INTER_AREA)
        # Saturate on the grid, not the full image: every pixel in a cell
        # shares one colour, so the result is identical and ~30x cheaper.
        hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1].astype(np.int16) * 3 // 2,
                              0, 255).astype(np.uint8)
        cell_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        big = cv2.resize(cell_bgr, (cols * cw, rows * ch),
                         interpolation=cv2.INTER_NEAREST).astype(np.float32)

        paper = np.array(STYLES[i % len(STYLES)][2], np.float32)
        # Darken the sampled colour so a pale face still reads as ink.
        return (paper * (1 - alpha) + big * 0.82 * alpha).astype(np.uint8)

    def export(self, idx, style=0, src=None, cell=None):
        """The emailed render: style i at export cell size."""
        if style % len(STYLES) == STYLE_COLOR and src is not None:
            return self.compose_color(idx, src, style, cell)
        return self.compose_style(idx, style, cell)

    def _compose_atlas(self, idx, atlas, ink, paper):
        rows, cols = idx.shape
        ch, cw = atlas.shape[1], atlas.shape[2]
        alpha = atlas[idx].transpose(0, 2, 1, 3).reshape(rows * ch, cols * cw)
        alpha = np.clip(alpha * 1.35, 0, 1)[..., None]
        return (np.array(paper, np.float32) * (1 - alpha) +
                np.array(ink, np.float32) * alpha).astype(np.uint8)

    # ── ramp construction ──

    def _coverage(self, c):
        img = PILImage.new("L", (self.cw, self.ch), 0)
        ImageDraw.Draw(img).text((0, -2), c, font=TEXT.font(int(self.ch * 1.05)),
                                 fill=255)
        return float(np.asarray(img, np.float32).mean() / 255.0)

    def _build_ramp(self, n, pool=None):
        """Pick n glyphs whose ink coverage is as evenly spaced as possible.

        Guarantees a monotonic brightness ramp for whatever font and cell
        size we actually render at, which a hand-written ramp does not.
        """
        cov = {c: self._coverage(c) for c in (pool or self.POOL)}
        top = max(cov.values())
        out = []
        for i in range(n):
            target = top * i / (n - 1)
            for c in sorted(cov, key=lambda c: abs(cov[c] - target)):
                if c not in out:
                    out.append(c)
                    break
        return "".join(sorted(out, key=lambda c: cov[c]))

    def _bake(self, ramp, cell=None):
        cw, ch = cell if cell else (self.cw, self.ch)
        font = TEXT.font(int(ch * 1.05))
        tiles = np.zeros((len(ramp), ch, cw), np.float32)
        for i, c in enumerate(ramp):
            img = PILImage.new("L", (cw, ch), 0)
            ImageDraw.Draw(img).text((0, -2), c, font=font, fill=255)
            tiles[i] = np.asarray(img, np.float32) / 255.0
        return tiles

    @staticmethod
    def _vignette(cols, rows):
        """Soft portrait-shaped falloff that pushes the room to bare paper.

        A face crop is centred by SubjectManager.  Keeping the middle 72%
        untouched protects hair, ears and jaw; the superellipse then fades
        walls and windows near the crop boundary without drawing a visible
        oval around the subject.
        """
        yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
        dx = (xx - (cols - 1) / 2) / ((cols - 1) / 2)
        dy = (yy - (rows - 1) / 2) / ((rows - 1) / 2)
        # Slightly narrower than the crop, with a superellipse rather than a
        # perfect oval so temples and shoulders are not clipped.
        d = (np.abs(dx / 0.92) ** 3.0 +
             np.abs(dy / 1.04) ** 3.0) ** (1.0 / 3.0)
        fade = np.clip((1.0 - d) / (1.0 - VIGNETTE), 0, 1)
        return fade ** VIGNETTE_POW


    # ── rendering ──

    def render(self, face_bgr, cols=ASCII_COLS, rows=ASCII_ROWS):
        """Convenience: quantise then compose in one step."""
        return self.compose(self.indices(face_bgr, cols, rows))

    def compose(self, idx, cell=None, ink=INK, paper=PAPER):
        """Render a frozen index grid as ink on paper, at any cell size.

        No blur, no additive glow: a printed portrait has hard edges, and
        the bloom pass is exactly what made this read as a webcam filter.
        """
        rows, cols = idx.shape
        atlas = self.atlas if cell is None else self._bake(self.ramp, cell)
        ch, cw = atlas.shape[1], atlas.shape[2]
        tiles = atlas[idx]
        alpha = tiles.transpose(0, 2, 1, 3).reshape(rows * ch, cols * cw)
        alpha = np.clip(alpha * 1.35, 0, 1)[..., None]
        return (np.array(paper, np.float32) * (1 - alpha) +
                np.array(ink, np.float32) * alpha).astype(np.uint8)

    def to_text(self, idx):
        """The frozen grid as plain ASCII — nice to paste into an email."""
        return "\n".join("".join(self.ramp[i] for i in row) for row in idx)

    def indices(self, face_bgr, cols=ASCII_COLS, rows=ASCII_ROWS):
        """Quantise a crop to glyph indices, densest glyph = darkest pixel."""
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        work = cv2.resize(gray, (cols * 2, rows * 2),
                          interpolation=cv2.INTER_AREA)
        work = cv2.bilateralFilter(work, 5, 40, 40)
        work = self._clahe.apply(work)
        blur = cv2.GaussianBlur(work, (0, 0), 2)
        work = cv2.addWeighted(work, 1.5, blur, -0.5, 0)

        small = cv2.resize(work, (cols, rows),
                           interpolation=cv2.INTER_AREA).astype(np.float32)

        # Estimate exposure from the face-sized centre, not the surrounding
        # room.  A dark wall or bright window can otherwise compress the face
        # into two or three glyphs even after CLAHE.
        yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
        cx = (cols - 1) / 2
        cy = (rows - 1) / 2
        centre = (((xx - cx) / max(1.0, cols * 0.34)) ** 2 +
                  ((yy - cy) / max(1.0, rows * 0.43)) ** 2) <= 1.0
        sample = small[centre]
        lo, hi = np.percentile(sample, 3), np.percentile(sample, 97)
        norm = np.clip((small - lo) / max(1.0, hi - lo), 0, 1)

        # Invert: this is ink, so dark pixels get the heaviest glyphs.
        ink = 1.0 - norm
        ink = np.clip((ink - INK_FLOOR) / (1.0 - INK_FLOOR), 0, 1) ** INK_GAMMA

        # Detail is additive, not a gate.  Multiplying tone by edge strength
        # used to erase smooth cheeks, forehead and flat dark hair.  Adding a
        # restrained edge layer keeps eyes, brows, nostrils and mouth crisp
        # without turning the portrait into a noisy Sobel sketch.
        f = work.astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.resize(cv2.magnitude(gx, gy), (cols, rows),
                         interpolation=cv2.INTER_AREA)
        detail_scale = np.percentile(mag[centre], 92) + 1e-6
        detail = np.sqrt(np.clip(mag / detail_scale, 0, 1))
        ink = np.clip(ink + EDGE_MIX * detail * (1.0 - ink * 0.55), 0, 1)

        vig = self._vig if ink.shape == self._vig.shape else \
            self._vignette(cols, rows)
        ink = ink * vig

        return np.clip((ink * (len(self.ramp) - 1)).round(), 0,
                       len(self.ramp) - 1).astype(np.int32)


# ──────────────────────────────────────────────────────────────
# Lasso geometry
# ──────────────────────────────────────────────────────────────

def closed_loop(stroke, close_px=70, min_pts=28, min_len=260):
    """True when a stroke has wandered far enough and returned to its start."""
    if len(stroke) < min_pts:
        return False
    sx, sy = stroke[0]
    cx, cy = stroke[-1]
    if math.hypot(cx - sx, cy - sy) > close_px:
        return False
    length = sum(math.dist(stroke[i], stroke[i - 1])
                 for i in range(1, len(stroke)))
    return length >= min_len


def enclosed_faces(stroke, tracks, min_area=5000):
    """Face tracks whose centre falls inside the lasso polygon."""
    poly = np.array(stroke, np.int32).reshape(-1, 1, 2)
    if abs(cv2.contourArea(poly)) < min_area:
        return []
    return [t for t in tracks
            if cv2.pointPolygonTest(poly, t.center, False) >= 0]


# ──────────────────────────────────────────────────────────────
# Lock-on subjects
# ──────────────────────────────────────────────────────────────

class _Subject:
    def __init__(self, slot, track, color, now):
        self.slot = slot
        self.tid = track.id
        self.color = color
        self.born = now
        self.anim = 0.0
        self.last_box = track.rect
        self.lost_since = None
        # Frozen at capture: the portrait is a keepsake, not a live filter.
        self.idx = None          # quantised glyph grid
        self.ascii_img = None    # composed for screen
        self.photo = None        # the source crop, saved alongside
        self.field = None        # capture.EmailField, attached by main
        self.sent_id = None      # capture-record id, once submitted
        self.style = 0           # index into STYLES
        self.flash = 0.0         # shutter flash after the countdown


class SubjectManager:
    def restyle(self, i):
        """Switch every captured portrait to style i.

        Re-composing from the frozen grid is sub-millisecond, so nobody has to
        circle their face again to try a different look.  The colour style
        needs the source crop, which _Subject keeps in .photo.
        """
        for s in self.subjects:
            s.style = i % len(STYLES)
            if s.idx is not None:
                s.ascii_img = (self.r.compose_color(s.idx, s.photo, s.style)
                               if s.style == STYLE_COLOR and s.photo is not None
                               else self.r.compose_style(s.idx, s.style))

    def __init__(self, renderer: AsciiRenderer):
        self.r = renderer
        self.subjects: list[_Subject] = []
        self._slot = 0
        self.style = 0           # session default, cycled with F

    # ── lifecycle ──

    def toggle(self, track, color):
        """Lasso a locked face to release it, an unlocked one to lock it."""
        for s in self.subjects:
            if s.tid == track.id:
                self.subjects.remove(s)
                return "released"
        if len(self.subjects) >= MAX_SUBJECTS:
            self.subjects.pop(0)
        self._slot += 1
        self.subjects.append(_Subject(self._slot, track, color, time.time()))
        return "locked"

    def release_all(self):
        n = len(self.subjects)
        self.subjects.clear()
        self._slot = 0
        return n

    @property
    def active(self):
        return len(self.subjects)

    # ── per-frame ──

    def update(self, frame, tracker: FaceTracker, dt):
        H, W = frame.shape[:2]
        now = time.time()
        alive = []

        # drop manual tracks nothing is locked onto any more
        keep = {s.tid for s in self.subjects}
        tracker.manual = [m for m in tracker.manual if m.id in keep]

        # Promote a manual lock to a real face lock as soon as the detector
        # finds a face anywhere near it.  Deliberately generous: the manual
        # track is only ever a bridge until detection catches up, and a
        # template on a half-background crop drifts if we make it wait.
        taken = {s.tid for s in self.subjects}
        for s in self.subjects:
            m = next((m for m in tracker.manual if m.id == s.tid), None)
            if m is None:
                continue
            mx, my, mw, mh = m.box
            cx, cy = mx + mw / 2, my + mh / 2
            reach = max(mw, mh)                     # ~one box-width of slack
            best, best_d = None, reach
            for f in tracker.visible:
                if f.id in taken:
                    continue
                fx, fy = f.center
                d = math.hypot(fx - cx, fy - cy)
                if d < best_d:
                    best, best_d = f, d
            if best is not None:
                tracker.drop_manual(s.tid)
                s.tid = best.id
                taken.add(best.id)

        for s in self.subjects:
            s.anim = min(1.0, s.anim + dt / SLIDE_TIME)
            tr = tracker.get(s.tid)

            if tr is not None:
                s.last_box = tr.rect
                s.lost_since = None
                if s.idx is None and now - s.born >= COUNTDOWN:
                    # Capture once, after the 3-2-1 has run its course.
                    x, y, w, h = s.last_box
                    cx = x + w / 2
                    cy = y + h / 2 + h * CROP_DROP
                    side = max(w, h) * CROP_PAD
                    x0 = int(max(0, cx - side / 2))
                    x1 = int(min(W, cx + side / 2))
                    y0 = int(max(0, cy - side / 2))
                    y1 = int(min(H, cy + side / 2))
                    if x1 - x0 > 8 and y1 - y0 > 8:
                        crop = frame[y0:y1, x0:x1].copy()
                        s.photo = crop
                        s.idx = self.r.indices(crop)
                        s.style = self.style
                        s.ascii_img = (
                            self.r.compose_color(s.idx, crop, s.style)
                            if s.style == STYLE_COLOR
                            else self.r.compose_style(s.idx, s.style))
                        s.flash = now
                alive.append(s)
            else:
                if s.lost_since is None:
                    s.lost_since = now
                if now - s.lost_since < LOST_GRACE:
                    alive.append(s)

        self.subjects = alive

    # ── drawing ──

    def draw(self, out, tracker: FaceTracker):
        H, W = out.shape[:2]
        now = time.time()

        for i, s in enumerate(self.subjects):
            e = _ease(s.anim)
            card_x = int(W - CARD_W - CARD_MARGIN + (1 - e) * (CARD_W + 40))
            card_y = CARD_MARGIN + i * (CARD_H + CARD_GAP)
            lost = s.lost_since is not None

            _reticle(out, s.last_box, s.color, s.slot, now, e, lost)
            if s.idx is None:
                _countdown(out, s, now)
            else:
                _connector(out, s.last_box, card_x, card_y, s.color, e, lost)
                _card(out, s, card_x, card_y, now, e, lost)
            _flash(out, s, now)


def _ease(t):
    return 1 - (1 - t) ** 3


def _countdown(out, s, now):
    """Big 3-2-1 over the subject, so nobody is captured mid-blink."""
    left = COUNTDOWN - (now - s.born)
    n = max(1, min(3, int(left / (COUNTDOWN / 3)) + 1))
    x, y, w, h = s.last_box
    cx, cy = x + w // 2, y + h // 2

    # each digit swells and fades over its own slice of the countdown
    step = COUNTDOWN / 3.0
    frac = 1.0 - ((left % step) / step)          # 0 -> 1 within this digit
    size = int(66 + 26 * frac)
    fade = max(0.0, 1.0 - frac * 0.55)

    label = str(n)
    tw = text_width(label, size, mono=False)
    text(out, label, (cx - tw // 2, cy - int(size * 0.72)), size,
         (255, 255, 255), strength=fade, mono=False)

    ring = int(max(w, h) * 0.78)
    cv2.ellipse(out, (cx, cy), (ring, ring), -90, 0,
                int(360 * (1 - left / COUNTDOWN)), s.color, 3, cv2.LINE_AA)

    hint = "HOLD STILL"
    hw = text_width(hint, 15, tracking=3)
    text(out, hint, (cx - hw // 2, y + h + 16), 15, (235, 233, 242),
         tracking=3, strength=0.9)


def _flash(out, s, now):
    """A brief white bloom at the moment of capture."""
    if s.flash <= 0:
        return
    k = 1.0 - (now - s.flash) / 0.28
    if k <= 0:
        s.flash = 0.0
        return
    out[:] = np.clip(out.astype(np.float32) + 255.0 * k * 0.55, 0,
                     255).astype(np.uint8)


def _reticle(out, box, color, slot, now, e, lost):
    x, y, w, h = box
    pad = int(10 + 6 * math.sin(now * 2.4))
    x0, y0 = x - pad, y - pad
    x1, y1 = x + w + pad, y + h + pad
    arm = max(12, int(min(w, h) * 0.28))
    col = tuple(int(c * (0.35 if lost else 1.0) * e) for c in color)
    th = 2

    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(out, (cx, cy), (cx + dx * arm, cy), col, th, cv2.LINE_AA)
        cv2.line(out, (cx, cy), (cx, cy + dy * arm), col, th, cv2.LINE_AA)

    if not lost:
        # scan line sweeping the locked region (ROI-only, not a full-frame copy)
        H, W = out.shape[:2]
        sy = int(y0 + (y1 - y0) * ((now * 0.6) % 1.0))
        lx0, lx1 = max(0, x0 + 4), min(W, x1 - 4)
        if 0 <= sy < H and lx0 < lx1:
            row = out[sy:sy + 1, lx0:lx1]
            tint = np.full_like(row, np.array(color, np.uint8))
            cv2.addWeighted(tint, 0.45 * e, row, 1 - 0.45 * e, 0, row)

    label = f"{slot:02d}"
    text(out, label, (x0, y1 + 6), 12, col, tracking=1, strength=e)
    if lost:
        text(out, "SIGNAL LOST", (x0 + 22, y1 + 7), 10, (90, 90, 110),
             tracking=1, strength=e)


def _connector(out, box, card_x, card_y, color, e, lost):
    x, y, w, h = box
    sx, sy = x + w, y + h // 2
    ex, ey = card_x, card_y + CARD_HEAD // 2 + 6
    if sx >= ex - 20:
        return
    mx = sx + (ex - sx) // 2
    col = tuple(int(c * 0.55 * e * (0.4 if lost else 1.0)) for c in color)
    for a, b in (((sx, sy), (mx, sy)), ((mx, sy), (mx, ey)), ((mx, ey), (ex, ey))):
        cv2.line(out, a, b, col, 1, cv2.LINE_AA)
    cv2.circle(out, (sx, sy), 3, col, -1, cv2.LINE_AA)


def _card(out, s, x, y, now, e, lost):
    H, W = out.shape[:2]
    x0, y0 = x, y
    x1, y1 = x + CARD_W, y + CARD_H
    vx0, vy0 = max(0, x0), max(0, y0)
    vx1, vy1 = min(W, x1), min(H, y1)
    if vx0 >= vx1 or vy0 >= vy1:
        return

    # frosted panel — blur at quarter res; indistinguishable at this radius
    roi = out[vy0:vy1, vx0:vx1]
    small = cv2.resize(roi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.5)
    panel = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    panel = (panel.astype(np.float32) * 0.30 +
             np.array((14, 12, 20), np.float32) * 0.70).astype(np.uint8)
    cv2.addWeighted(panel, 0.92 * e, roi, 1 - 0.92 * e, 0, roi)

    accent = tuple(int(c * (0.4 if lost else 1.0)) for c in s.color)
    cv2.rectangle(out, (vx0, vy0), (vx1 - 1, vy1 - 1), (48, 46, 62), 1,
                  cv2.LINE_AA)
    cv2.line(out, (vx0, vy0), (vx0, vy1 - 1), accent, 2, cv2.LINE_AA)

    # header
    text(out, f"ASCII PORTRAIT {s.slot:02d}", (x0 + CARD_PAD, y0 + 7), 12, accent,
         tracking=1, strength=e)
    # The chip tracks the send, so the person can see their portrait is
    # actually on its way rather than just filed.
    fs = s.field.status if s.field is not None else ""
    status, dot = {
        "sending": ("SENDING", (250, 200, 120)),
        "sent": ("SENT", (150, 220, 180)),
        "failed": ("QUEUED", (250, 200, 120)),
        "saved": ("SAVED", (150, 220, 180)),
    }.get(fs, ("CAPTURED", (150, 220, 180)))
    if fs == "sending" and int(now * 2) % 2:
        dot = (90, 84, 110)                       # slow pulse while in flight
    sw = text_width(status, 10, tracking=1)
    text(out, status, (x1 - CARD_PAD - sw - 12, y0 + 9), 10,
         dot if fs != "sending" else (250, 200, 120), tracking=1, strength=e)
    cv2.circle(out, (x1 - CARD_PAD - 4, y0 + 15), 3, dot, -1,
               cv2.LINE_AA)
    cv2.line(out, (vx0 + 1, y0 + CARD_HEAD), (vx1 - 1, y0 + CARD_HEAD),
             (48, 46, 62), 1, cv2.LINE_AA)

    # ── frozen portrait ──
    if s.ascii_img is None:
        return
    ah, aw = s.ascii_img.shape[:2]
    ax, ay = x0 + CARD_PAD, y0 + CARD_HEAD + CARD_PAD
    bx0, by0 = max(0, ax), max(0, ay)
    bx1, by1 = min(W, ax + aw), min(H, ay + ah)
    if bx0 < bx1 and by0 < by1:
        src = s.ascii_img[by0 - ay:by1 - ay, bx0 - ax:bx1 - ax]
        dst = out[by0:by1, bx0:bx1]
        # alpha blend, not additive — the portrait is light on light now
        cv2.addWeighted(src, e, dst, 1 - e, 0, dst)

    # ── email field ──
    if s.field is None:
        return
    _field(out, s, x0, ay + ah + CARD_PAD, now, e, accent)


def _field(out, s, x0, y, now, e, accent):
    f = s.field
    fx, fw = x0 + CARD_PAD, CARD_W - CARD_PAD * 2

    if f.status in ("saved", "sent"):
        col = (150, 220, 180)
    elif f.status == "sending":
        col = (120, 200, 250)
    elif f.status in ("error", "failed"):
        col = (90, 90, 245)
    elif f.active:
        col = accent
    else:
        col = (110, 106, 130)

    # Dark well with a coloured border — tinting the *fill* with the ink
    # colour made light accents (yellow) unreadable behind the text.
    pill(out, fx, y, fw, FIELD_H, (16, 14, 20), alpha=0.85, rad=8)
    pill(out, fx, y, fw, FIELD_H, col, alpha=0.06 + 0.14 * f.flash,
         border=col, rad=8)

    ty = y + (FIELD_H - 15) // 2 - 1
    if f.text:
        shown = f.text[-34:]
        text(out, shown, (fx + 13, ty), 15, (242, 241, 246), strength=e)
        cw = text_width(shown, 15)
    else:
        if f.active:
            text(out, "your@email.com", (fx + 13, ty), 15, (128, 125, 148),
                 strength=e)
        cw = 0

    # caret
    if f.active and (time.time() * 1.6) % 1.0 < 0.55:
        cx = fx + 13 + cw + 2
        cv2.line(out, (cx, y + 10), (cx, y + FIELD_H - 10), (242, 241, 246),
                 2, cv2.LINE_AA)

    # ── status line ──
    if f.status in ("saved", "sent"):
        msg, mc = f.message, (150, 220, 180)
    elif f.status == "sending":
        msg, mc = f.message, (150, 210, 250)
    elif f.status in ("error", "failed"):
        msg, mc = f.message, (120, 120, 235)
    elif f.active:
        msg, mc = "TYPE EMAIL  ·  ENTER TO SAVE", (150, 146, 172)
    else:
        msg, mc = "HOLD FIST TO CLEAR", (140, 136, 162)
    text(out, msg, (fx + 3, y + FIELD_H + 6), 10, mc, tracking=1, strength=e)


def draw_candidates(out, tracks, locked_ids):
    """Bright brackets on detected-but-unlocked faces.

    People need to see they have been found *before* they draw, or they
    have no idea the thing is working — the old ticks were a dim hairline
    and invisible against a bright room.
    """
    pulse = 0.80 + 0.20 * abs(math.sin(time.time() * 2.2))
    for t in tracks:
        if t.id in locked_ids:
            continue
        x, y, w, h = t.rect
        arm = max(14, int(min(w, h) * 0.26))
        col = tuple(int(c * pulse) for c in (255, 238, 120))   # BGR: cyan
        for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1),
                               (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            cv2.line(out, (cx, cy), (cx + dx * arm, cy), (20, 20, 24), 5,
                     cv2.LINE_AA)                       # dark keyline first,
            cv2.line(out, (cx, cy), (cx, cy + dy * arm), (20, 20, 24), 5,
                     cv2.LINE_AA)                       # so it reads on white
            cv2.line(out, (cx, cy), (cx + dx * arm, cy), col, 2, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * arm), col, 2, cv2.LINE_AA)
        tag = "FACE FOUND  ·  CIRCLE IT"
        tw = text_width(tag, 12, tracking=2)
        text(out, tag, (x + w // 2 - tw // 2, y + h + 10), 12, col,
             tracking=2, strength=0.95)
