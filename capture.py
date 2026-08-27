"""
capture — freeze a subject's ASCII portrait and collect their email.

The demo flow: someone circles their face, the portrait freezes, they type
their email, and both are written to disk so the images can be mailed out
afterwards.  Storage is a directory of PNG/TXT files plus one JSON index
that grows with every capture.

  captures/
    captures.json          the growing index
    2026-08-27_141133_01.png
    2026-08-27_141133_01.txt

The JSON is rewritten atomically (temp file + replace) on every save, so a
crash or a yanked power cable during a demo cannot truncate the list of
addresses collected so far.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Deliberately permissive: this validates shape, not deliverability, and a
# demo should never reject someone's real address on a technicality.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")

MAX_EMAIL = 64
EXPORT_CELL = (14, 28)          # glyph cell for the emailed PNG
EXPORT_MARGIN = 40


def valid_email(s):
    return bool(EMAIL_RE.match(s.strip()))


class CaptureStore:
    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "captures.json"
        self.records = self._load()

    def _load(self):
        if not self.index.exists():
            return []
        try:
            data = json.loads(self.index.read_text())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            # Never lose a demo's worth of addresses to a bad parse — keep
            # the old file aside and start a fresh index.
            self.index.rename(self.index.with_suffix(
                f".corrupt-{int(time.time())}.json"))
            return []

    def _write_index(self):
        tmp = self.index.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.records, indent=2))
        tmp.replace(self.index)

    @property
    def count(self):
        return len(self.records)

    def save(self, email, idx_grid, renderer, photo=None):
        """Write the portrait + text + index entry.  Returns the record."""
        stamp = datetime.now()
        base = f"{stamp:%Y-%m-%d_%H%M%S}_{len(self.records) + 1:02d}"

        art = renderer.compose(idx_grid, cell=EXPORT_CELL)
        m = EXPORT_MARGIN
        canvas = np.full((art.shape[0] + m * 2, art.shape[1] + m * 2, 3),
                         art[0, 0], np.uint8)          # match the paper
        canvas[m:m + art.shape[0], m:m + art.shape[1]] = art
        png = self.dir / f"{base}.png"
        cv2.imwrite(str(png), canvas)

        txt = self.dir / f"{base}.txt"
        txt.write_text(renderer.to_text(idx_grid) + "\n")

        rec = {
            "email": email.strip(),
            "captured_at": stamp.isoformat(timespec="seconds"),
            "image": png.name,
            "text": txt.name,
            "emailed": False,
        }
        if photo is not None:
            ph = self.dir / f"{base}_photo.png"
            cv2.imwrite(str(ph), photo)
            rec["photo"] = ph.name

        self.records.append(rec)
        self._write_index()
        return rec


class EmailField:
    """A tiny text input driven by cv2.waitKey codes."""

    def __init__(self):
        self.text = ""
        self.active = True
        self.status = ""          # "", "saved", "error"
        self.message = ""
        self._flash = 0.0

    # ── input ──

    def key(self, code):
        """Feed a keycode.  Returns 'submit', 'cancel', 'edit', or None."""
        if not self.active:
            return None
        if code in (13, 10):
            return "submit"
        if code == 27:
            self.active = False
            return "cancel"
        if code in (8, 127):
            self.text = self.text[:-1]
            self.status = ""
            return "edit"
        if 32 <= code < 127 and len(self.text) < MAX_EMAIL:
            self.text += chr(code)
            self.status = ""
            return "edit"
        return None

    def saved(self, msg):
        self.status = "saved"
        self.message = msg
        self.active = False
        self._flash = time.time()

    def error(self, msg):
        self.status = "error"
        self.message = msg
        self._flash = time.time()

    @property
    def flash(self):
        return max(0.0, 1.0 - (time.time() - self._flash) * 1.5)
