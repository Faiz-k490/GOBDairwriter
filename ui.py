"""
ui — shared drawing primitives: real-font text, alpha blitting, panels.

OpenCV's Hershey fonts are the single biggest visual tell of a CV demo,
so everything user-facing renders through PIL with a real system font
and composites as an alpha mask.  Masks are cached by (text, size,
tracking), so repeated labels cost one blit.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont

MONO_CANDIDATES = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Courier.ttc",
]
UI_CANDIDATES = [
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
] + MONO_CANDIDATES


class _Text:
    def __init__(self):
        self._cache: dict[tuple, np.ndarray] = {}
        self._fonts: dict[tuple, ImageFont.FreeTypeFont] = {}

    def font(self, size, mono=True):
        key = (size, mono)
        if key not in self._fonts:
            f = None
            for path in (MONO_CANDIDATES if mono else UI_CANDIDATES):
                if Path(path).exists():
                    try:
                        f = ImageFont.truetype(path, size)
                        break
                    except OSError:
                        continue
            self._fonts[key] = f or ImageFont.load_default()
        return self._fonts[key]

    def mask(self, txt, size, tracking=0, mono=True):
        key = (txt, size, tracking, mono)
        if key in self._cache:
            return self._cache[key]
        font = self.font(size, mono)
        probe = ImageDraw.Draw(PILImage.new("L", (1, 1)))
        if tracking:
            w = sum(int(probe.textlength(c, font=font)) + tracking for c in txt)
        else:
            w = int(probe.textlength(txt, font=font))
        w, h = max(1, w + 2), size + size // 2
        img = PILImage.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        if tracking:
            x = 1
            for c in txt:
                d.text((x, 0), c, font=font, fill=255)
                x += int(d.textlength(c, font=font)) + tracking
        else:
            d.text((1, 0), txt, font=font, fill=255)
        m = np.asarray(img, np.float32) / 255.0
        self._cache[key] = m
        return m


TEXT = _Text()


def blit(dst, alpha, x, y, color, strength=1.0):
    """Alpha-composite a mask onto a BGR image, clipped to bounds."""
    ah, aw = alpha.shape
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + aw), min(H, y + ah)
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x, None] * strength
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = np.clip(
        roi * (1 - a) + np.array(color, np.float32) * a, 0, 255).astype(np.uint8)


def text(dst, txt, pos, size, color, tracking=0, strength=1.0, mono=True):
    blit(dst, TEXT.mask(txt, size, tracking, mono), pos[0], pos[1],
         color, strength)


def text_width(txt, size, tracking=0, mono=True):
    return TEXT.mask(txt, size, tracking, mono).shape[1]


def pill(dst, x, y, w, h, color, alpha=0.22, border=None, rad=None):
    """Rounded translucent chip — the HUD's basic building block."""
    rad = h // 2 if rad is None else rad
    x, y, w, h = int(x), int(y), int(w), int(h)
    H, W = dst.shape[:2]
    if x >= W or y >= H or x + w <= 0 or y + h <= 0:
        return
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (rad, 0), (w - rad, h), 255, -1)
    cv2.rectangle(mask, (0, rad), (w, h - rad), 255, -1)
    for cx, cy in ((rad, rad), (w - rad, rad), (rad, h - rad), (w - rad, h - rad)):
        cv2.circle(mask, (cx, cy), rad, 255, -1)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    sub = mask[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    a = sub[..., None] * alpha
    dst[y0:y1, x0:x1] = np.clip(
        roi * (1 - a) + np.array(color, np.float32) * a, 0, 255).astype(np.uint8)
    if border is not None:
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
        sub = edge[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0
        roi = dst[y0:y1, x0:x1].astype(np.float32)
        a = sub[..., None] * 0.9
        dst[y0:y1, x0:x1] = np.clip(
            roi * (1 - a) + np.array(border, np.float32) * a, 0,
            255).astype(np.uint8)
