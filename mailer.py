"""
mailer — send each portrait to the person in it, without stalling the demo.

Sending happens on a daemon thread.  The render loop must never block on a
socket: a fair hall's wifi drops, resolves slowly, and occasionally disappears
between one person and the next, and a frozen window in front of a crowd is
worse than a late email.

Threading rule, and it is the important one: **this module never mutates
anything the render loop reads.**  Work goes in through submit(), outcomes come
back out through results(), and the main thread is what writes them to the
store.  The worker gets copies and absolute paths, never the live record or the
CaptureStore — capture.py appends to that list and serialises it, and json.dumps
over a list being appended to from another thread is a real race.

Configuration lives in `.env` (gitignored — it holds a password).  See
.env.example for how to get a Gmail app password.
"""

import os
import queue
import random
import smtplib
import threading
import time
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from pathlib import Path

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
TIMEOUT = 20

# Five tries over ~6.5 minutes.  Long enough to ride out a dead spot in the
# hall, short enough that a genuinely wrong password gives up while you can
# still fix it.
BACKOFF = (5, 20, 60, 180, 300)


def load_config(path=None):
    """Read .env, then the environment (which wins).  No new dependency."""
    cfg = {}
    env_file = Path(path) if path else Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("AIRWRITER_SMTP_USER", "AIRWRITER_SMTP_PASS",
              "AIRWRITER_FROM_NAME", "AIRWRITER_REPLY_TO",
              "AIRWRITER_CLUB_NAME", "AIRWRITER_ENABLED"):
        if os.environ.get(k):
            cfg[k] = os.environ[k]

    user = cfg.get("AIRWRITER_SMTP_USER", "").strip()
    # Google prints app passwords in four groups; people paste them that way.
    pw = cfg.get("AIRWRITER_SMTP_PASS", "").replace(" ", "").strip()
    enabled = cfg.get("AIRWRITER_ENABLED", "1").strip() not in ("0", "false",
                                                                "no", "")
    return {
        "user": user,
        "password": pw,
        "from_name": cfg.get("AIRWRITER_FROM_NAME", "Air Writer").strip(),
        "reply_to": cfg.get("AIRWRITER_REPLY_TO", "").strip() or user,
        "club": cfg.get("AIRWRITER_CLUB_NAME", "").strip(),
        # Placeholder credentials in .env.example must not read as configured.
        "enabled": bool(enabled and user and pw and "x" * 8 not in pw),
    }


def build_message(cfg, to_addr, png_path=None, ascii_text="", when=""):
    """Compose the portrait email.  Pure — no network, so it is testable."""
    club = cfg.get("club") or ""
    msg = EmailMessage()
    msg["Subject"] = "Your ASCII portrait"
    msg["From"] = formataddr((cfg.get("from_name") or "Air Writer",
                              cfg["user"]))
    msg["To"] = to_addr
    if cfg.get("reply_to"):
        msg["Reply-To"] = cfg["reply_to"]

    made_at = f" at {club}" if club else ""
    # The ASCII itself is the keepsake, so it goes in the body as text rather
    # than only as an attachment.
    body = (
        "Here's your portrait.\n\n"
        "A webcam found your face, you drew a circle around it in the air, "
        f"and a program turned it into letters{made_at}.\n\n"
        "The full-size image is attached. The text version:\n\n"
        f"{ascii_text}\n\n"
        "-- \nMade with Air Writer, a hand-tracking drawing toy "
        "(Python + OpenCV + MediaPipe).\n"
    )
    msg.set_content(body)

    cid = make_msgid()[1:-1]
    esc = (ascii_text.replace("&", "&amp;").replace("<", "&lt;")
           .replace(">", "&gt;"))
    msg.add_alternative(f"""\
<html><body style="margin:0;padding:24px;background:#14121a;
 font-family:-apple-system,Helvetica,Arial,sans-serif;color:#e8e6ef">
<div style="max-width:640px;margin:0 auto">
  <h1 style="font-size:20px;letter-spacing:.14em;font-weight:600;
   color:#8ee6ff;margin:0 0 6px">YOUR ASCII PORTRAIT</h1>
  <p style="color:#a9a5bd;font-size:14px;line-height:1.6;margin:0 0 20px">
   A webcam found your face, you drew a circle around it in the air, and a
   program turned it into letters{made_at}.</p>
  <img src="cid:{cid}" alt="Your portrait in ASCII"
   style="width:100%;border-radius:10px;display:block;background:#eceae6">
  <pre style="margin:22px 0 0;padding:16px;background:#0d0b12;color:#c9c5da;
   border-radius:10px;overflow-x:auto;font-size:5px;line-height:1.05">{esc}</pre>
  <p style="color:#6f6b85;font-size:12px;margin:22px 0 0">
   Made with Air Writer — Python, OpenCV and MediaPipe.{
   ' Captured ' + when if when else ''}</p>
</div></body></html>""", subtype="html")

    if png_path and Path(png_path).exists():
        data = Path(png_path).read_bytes()
        # Attach to the HTML part so the cid: reference resolves.
        msg.get_payload()[1].add_related(data, "image", "png", cid=f"<{cid}>")
        msg.add_attachment(data, maintype="image", subtype="png",
                           filename=Path(png_path).name)
    return msg


class _Permanent(Exception):
    """A failure that retrying cannot fix (bad auth, refused address)."""


def send_once(cfg, msg, transport=None):
    """Send one message.  `transport` is injectable so tests stay offline."""
    if transport is not None:
        return transport(cfg, msg)
    # A fresh connection per send: a held-open SMTP session does not survive a
    # venue's wifi, and a stale one fails in confusing ways.
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT) as smtp:
        smtp.login(cfg["user"], cfg["password"])
        smtp.send_message(msg)


class EmailSender:
    """Background sender.  submit() from the loop, drain results() each frame."""

    def __init__(self, cfg, transport=None, backoff=BACKOFF):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled"))
        self._transport = transport
        self._backoff = tuple(backoff)
        self._in = queue.Queue()
        self._out = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="airwriter-mailer")
            self._thread.start()

    # ── main thread ──

    def submit(self, rec_id, to_addr, png_path, ascii_text, when=""):
        """Queue one send.  Everything passed here must be a copy."""
        if not self.enabled:
            return False
        self._in.put({"id": rec_id, "to": to_addr,
                      "png": str(png_path) if png_path else "",
                      "ascii": ascii_text, "when": when})
        return True

    def results(self):
        """Outcomes since the last call: (rec_id, status, error, attempts)."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    @property
    def queued(self):
        return self._in.qsize()

    def stop(self, drain=0.0):
        """Ask the worker to finish.  Daemon, so this can never wedge exit."""
        if not self._thread:
            return
        deadline = time.time() + drain
        while drain and time.time() < deadline and self._in.qsize():
            time.sleep(0.05)
        self._stop.set()

    # ── worker thread ──

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._in.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._deliver(job)
            except Exception as e:                      # never kill the worker
                self._out.put((job["id"], "failed", f"{type(e).__name__}: {e}",
                               0))

    def _deliver(self, job):
        msg = build_message(self.cfg, job["to"], job["png"], job["ascii"],
                            job["when"])
        last = ""
        for attempt in range(1, len(self._backoff) + 2):
            if self._stop.is_set():
                self._out.put((job["id"], "pending", "shutting down", attempt))
                return
            try:
                send_once(self.cfg, msg, self._transport)
                self._out.put((job["id"], "sent", "", attempt))
                return
            except (smtplib.SMTPAuthenticationError,
                    smtplib.SMTPRecipientsRefused,
                    smtplib.SMTPSenderRefused, _Permanent) as e:
                # Retrying a rejected password or a refused address just burns
                # the daily quota and delays the real diagnosis.
                self._out.put((job["id"], "failed",
                               f"{type(e).__name__}: {e}", attempt))
                return
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                if attempt > len(self._backoff):
                    break
                delay = self._backoff[attempt - 1]
                # Jitter so a wifi blip does not resend everything in lockstep.
                self._sleep(delay + random.uniform(0, 0.4 * delay))
        self._out.put((job["id"], "failed", last, len(self._backoff) + 1))

    def _sleep(self, secs):
        """Interruptible: shutdown must not wait out a 5-minute backoff."""
        self._stop.wait(secs)
