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
        "from_name": cfg.get("AIRWRITER_FROM_NAME", "HackBama").strip(),
        "reply_to": cfg.get("AIRWRITER_REPLY_TO", "").strip() or user,
        "club": cfg.get("AIRWRITER_CLUB_NAME", "HackBama").strip(),
        # Placeholder credentials in .env.example must not read as configured.
        "enabled": bool(enabled and user and pw and "x" * 8 not in pw),
    }


# ── HackBama brand ────────────────────────────────────────────────────
#
# Taken from the club site (hackbama.org): a warm, editorial, print-inspired
# palette — bone paper, Alabama crimson, hairline rules, square corners, no
# shadows and no glow.  Deliberately the opposite of a neon-on-black demo
# aesthetic, so the mail reads as coming from the club rather than from a toy.
BONE = "#F1EDE4"          # page background
PAPER = "#F9F7F1"         # card surface
INK = "#191715"           # headings
INK_SOFT = "#5C5751"      # body
INK_FAINT = "#958D83"     # labels
CRIMSON = "#8D1B30"       # accent, CTA
RULE = "#D8D2C9"          # hairline dividers

GROUPME = "https://groupme.com/join_group/105495989/hs26yOC2"
SITE = "https://hackbama.org"

# Inter Tight and Bodoni Moda are Google Fonts and will not load in Outlook or
# Gmail webmail, so every stack ends in something universally present.
SANS = "'Inter Tight',Inter,system-ui,-apple-system,Helvetica,Arial,sans-serif"
SERIF = "'Bodoni Moda',Georgia,'Times New Roman',serif"
MONO = "'JetBrains Mono',ui-monospace,Menlo,Consolas,monospace"


def build_message(cfg, to_addr, png_path=None, ascii_text="", when=""):
    """Compose the portrait email.  Pure — no network, so it is testable."""
    club = cfg.get("club") or "HackBama"
    msg = EmailMessage()
    msg["Subject"] = f"Your portrait, and what {club} actually is"
    msg["From"] = formataddr((cfg.get("from_name") or club, cfg["user"]))
    msg["To"] = to_addr
    if cfg.get("reply_to"):
        msg["Reply-To"] = cfg["reply_to"]

    # The club voice: short declarative sentences, no contractions, no hype,
    # no exclamation points.  Matched here so the mail sounds like the site.
    body = f"""\
{club.upper()}
The build club at The University of Alabama.

Here is your portrait.

You drew a circle in the air and a webcam turned your face into letters.
That is roughly 800 lines of Python — hand tracking, a gesture that knows a
closed loop from a stray scribble, and a renderer that measures the ink
coverage of every glyph so the shading does not band. A student wrote it. That
is the entire point of the booth.

Nobody is missing talent. They are missing a door.

Twice a month we put students in a room and ship something before they leave
it. Tool sessions, skill nights, and every third meeting a mini-hackathon:
two hours from scratch, or one hour with AI wide open. You leave with
something that runs. That is the entire bar.

No application, no experience bar, no dues. Freshmen welcome. Come to one
night and decide from there.

Join the GroupMe: {GROUPME}
{SITE}

Your portrait is attached. The text version:

{ascii_text}

--
{club} · The University of Alabama
A proposed student organization of the Department of Computer Science.
"""
    msg.set_content(body)

    cid = make_msgid()[1:-1]
    esc = (ascii_text.replace("&", "&amp;").replace("<", "&lt;")
           .replace(">", "&gt;"))
    captured = (f'<div style="font-family:{MONO};font-size:11px;'
                f'letter-spacing:.12em;text-transform:uppercase;'
                f'color:{INK_FAINT};margin:0 0 6px">Captured {when}</div>'
                if when else "")

    msg.add_alternative(f"""\
<html><body style="margin:0;padding:0;background:{BONE};
 font-family:{SANS};color:{INK_SOFT};-webkit-font-smoothing:antialiased">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"
 border="0" style="background:{BONE}"><tr><td align="center"
 style="padding:32px 16px">
<table role="presentation" width="600" cellpadding="0" cellspacing="0"
 border="0" style="width:600px;max-width:600px">

  <tr><td style="padding:0 0 18px">
    <div style="font-family:{SERIF};font-weight:700;font-size:22px;
     letter-spacing:.12em;color:{INK}">{club.upper()}</div>
    <div style="font-size:13px;color:{INK_FAINT};margin-top:4px">
      The build club at The University of Alabama</div>
  </td></tr>

  <tr><td style="border-top:1px solid {RULE};padding:26px 0 0">
    {captured}
    <h1 style="margin:0 0 14px;font-size:30px;line-height:1.08;
     letter-spacing:-.03em;font-weight:600;color:{INK}">
      Here is your portrait.</h1>
    <p style="margin:0 0 22px;font-size:16px;line-height:1.625;
     color:{INK_SOFT}">
      You drew a circle in the air and a webcam turned your face into
      letters.</p>
  </td></tr>

  <tr><td style="padding:0 0 26px">
    <img src="cid:{cid}" alt="Your portrait, rendered in ASCII" width="600"
     style="width:100%;max-width:600px;display:block;border:1px solid {RULE};
     background:{PAPER}"></td></tr>

  <tr><td style="background:{PAPER};border:1px solid {RULE};
   padding:26px 28px">
    <div style="font-family:{MONO};font-size:11px;letter-spacing:.12em;
     text-transform:uppercase;color:{INK_FAINT};margin:0 0 10px">
      How it was made</div>
    <p style="margin:0;font-size:15px;line-height:1.65;color:{INK_SOFT}">
      Roughly 800 lines of Python — hand tracking, a gesture that knows a
      closed loop from a stray scribble, and a renderer that measures the ink
      coverage of every glyph so the shading does not band. A student wrote
      it. That is the entire point of the booth.</p>
  </td></tr>

  <tr><td style="padding:34px 0 0">
    <h2 style="margin:0 0 12px;font-size:23px;line-height:1.1;
     letter-spacing:-.03em;font-weight:600;color:{INK}">
      Nobody is missing talent. They are missing a door.</h2>
    <p style="margin:0 0 18px;font-size:16px;line-height:1.625;
     color:{INK_SOFT}">
      Twice a month we put students in a room and ship something before they
      leave it. Tool sessions, skill nights, and every third meeting a
      mini-hackathon: two hours from scratch, or one hour with AI wide open.
      You leave with something that runs. That is the entire bar.</p>
  </td></tr>

  <tr><td style="padding:0 0 30px">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0"
     width="100%" style="border-collapse:separate">
      <tr>
        <td width="8" style="background:{CRIMSON};font-size:0;
         line-height:0">&nbsp;</td>
        <td style="background:{PAPER};border:1px solid {RULE};
         border-left:0;padding:16px 20px;font-size:15px;line-height:1.6;
         color:{INK_SOFT}">
          No application, no experience bar, no dues. Freshmen welcome. Come
          to one night and decide from there.</td>
      </tr>
    </table>
  </td></tr>

  <tr><td style="padding:0 0 34px">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0">
      <tr><td style="background:{CRIMSON}">
        <a href="{GROUPME}" style="display:inline-block;padding:15px 34px;
         font-family:{SANS};font-size:16px;font-weight:500;color:{PAPER};
         text-decoration:none">Join the GroupMe</a>
      </td></tr>
    </table>
  </td></tr>

  <tr><td style="border-top:1px solid {RULE};padding:22px 0 0">
    <div style="font-family:{MONO};font-size:11px;letter-spacing:.12em;
     text-transform:uppercase;color:{INK_FAINT};margin:0 0 10px">
      Your portrait as text</div>
    <div style="background:{PAPER};border:1px solid {RULE};padding:14px;
     overflow-x:auto">
      <pre style="margin:0;font-family:{MONO};font-size:4px;line-height:1.02;
       color:{INK};white-space:pre">{esc}</pre></div>
  </td></tr>

  <tr><td style="border-top:1px solid {RULE};margin-top:26px;
   padding:22px 0 0">
    <p style="margin:26px 0 0;font-size:13px;line-height:1.6;
     color:{INK_FAINT}">
      <a href="{SITE}" style="color:{CRIMSON};text-decoration:none">
        hackbama.org</a><br>
      The University of Alabama · A proposed student organization of the
      Department of Computer Science.<br>
      You are receiving this because you asked for your portrait at our
      booth.</p>
  </td></tr>

</table></td></tr></table></body></html>""", subtype="html")

    if png_path and Path(png_path).exists():
        data = Path(png_path).read_bytes()
        # One copy, not two.  Attaching it separately as well would base64 a
        # ~1MB portrait twice into every message for no visible gain.  Giving
        # the inline part a filename and inline disposition means clients that
        # render it show it in place, and the rest offer it as a download.
        msg.get_payload()[1].add_related(
            data, "image", "png", cid=f"<{cid}>",
            filename=Path(png_path).name,
            disposition="inline")
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
