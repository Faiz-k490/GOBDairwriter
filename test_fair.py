"""
test_fair — the paths that must not break in front of a crowd.

Runs offline: every send goes through an injected transport, so the suite
never touches the network and never mails anyone.  Run it before the fair.

    .venv/bin/python test_fair.py
"""

import shutil
import smtplib
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import capture
import face_ascii as fa
import mailer

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}{'' if cond else '  — ' + detail}")


def head(size=300):
    """A synthetic tonal face.  Not a real one — see CLAUDE.md."""
    im = np.full((size, size, 3), 90, np.uint8)
    c = size // 2
    cv2.ellipse(im, (c, c), (int(size * .27), int(size * .35)), 0, 0, 360,
                (150, 170, 200), -1)
    cv2.circle(im, (c - 30, c - 20), 10, (40, 40, 45), -1)
    cv2.circle(im, (c + 30, c - 20), 10, (40, 40, 45), -1)
    cv2.ellipse(im, (c, c + 45), (34, 16), 0, 0, 180, (70, 60, 90), -1)
    return im


def test_ramps(r):
    """Every style ramp must be monotonic in ink coverage, or it bands."""
    for i, (name, _, _, pool) in enumerate(fa.STYLES):
        ramp, _ = r.style(i)
        cov = [r._coverage(c) for c in ramp]
        mono = all(a <= b + 1e-9 for a, b in zip(cov, cov[1:]))
        check(f"ramp {name} monotonic", mono, f"coverage {cov}")


def test_styles_render(r, img):
    idx = r.indices(img)
    seen = {}
    for i, (name, *_) in enumerate(fa.STYLES):
        out = (r.compose_color(idx, img, i) if i == fa.STYLE_COLOR
               else r.compose_style(idx, i))
        check(f"style {name} renders", out.shape == (512, 512, 3),
              str(out.shape))
        seen[name] = round(float(out.mean()), 1)
    # If two styles produce the same image, one of them is not doing anything.
    check("styles are visually distinct", len(set(seen.values())) == len(seen),
          str(seen))


def test_export_parity(r, img):
    d = Path(tempfile.mkdtemp())
    try:
        st = capture.CaptureStore(d)
        idx = r.indices(img)
        for i, (name, *_) in enumerate(fa.STYLES):
            rec = st.save(f"s{i}@x.com", idx, r, img, i)
            im = cv2.imread(str(d / rec["image"]))
            check(f"export {name} written", im is not None and im.shape[0] > 1500,
                  str(None if im is None else im.shape))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_live(r):
    """The mirror must stay far under the frame budget, and stay glow."""
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3),
                                              dtype=np.uint8)
    out = np.zeros((782, 1280, 3), np.uint8)
    view = out[62:]
    lv = view.reshape(fa.LIVE_ROWS, fa.LIVE_CH, fa.LIVE_COLS,
                      fa.LIVE_CW, 3).transpose(0, 2, 1, 3, 4)

    check("live grid divides the frame exactly",
          1280 % fa.LIVE_CW == 0 and 720 % fa.LIVE_CH == 0
          and fa.LIVE_COLS * fa.LIVE_CW == 1280
          and fa.LIVE_ROWS * fa.LIVE_CH == 720,
          "a non-dividing grid is ~20x slower through INTER_AREA")

    def mono():
        lv[...] = r._live_atlas[r.live_indices(frame)]

    def colour():
        r.live_color(r.live_indices(frame), frame, view)

    for label, fn in (("mono", mono), ("colour", colour)):
        fn(); fn()
        t = time.perf_counter()
        for _ in range(30):
            fn()
        ms = (time.perf_counter() - t) / 30 * 1000
        check(f"live {label} under budget ({ms:.2f}ms < 15ms)", ms < 15.0,
              f"{ms:.2f}ms")

    # A lit face on a dark room should leave most of the frame blank.
    room = np.full((720, 1280, 3), 42, np.uint8)
    room[300:520, 560:760] = 190
    r._elo = r._ehi = None
    r._lut_lo = r._lut_hi = -99.0
    idx = r.live_indices(room)
    blank = float((idx == 0).mean())
    check(f"live mode is glow, not a wall of glyphs ({blank:.0%} blank)",
          blank > 0.6, f"only {blank:.0%} blank")


def test_capture_guards(r, img):
    d = Path(tempfile.mkdtemp())
    try:
        st = capture.CaptureStore(d)
        try:
            st.save("a@b.com", None, r)
            check("null grid still raises inside the store", False,
                  "expected an exception")
        except Exception:
            # main.py guards this before calling; the store itself may raise.
            check("null grid raises in the store (main.py guards it)", True)

        rec = st.save("a@b.com", r.indices(img), r, img)
        check("record has a stable id", len(rec.get("id", "")) == 32)
        check("record starts pending", rec["status"] == capture.PENDING)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_store_lifecycle(r, img):
    d = Path(tempfile.mkdtemp())
    try:
        st = capture.CaptureStore(d)
        idx = r.indices(img)
        a = st.save("a@x.com", idx, r)
        b = st.save("b@x.com", idx, r)
        st.mark(a["id"], capture.SENT)
        st.mark(b["id"], capture.FAILED, "Connection timed out")

        check("sent is counted", st.sent_count == 1, str(st.sent_count))
        check("failed is not pending", not st.pending())
        check("failure records the error",
              "timed out" in st.records[1]["last_error"])

        # A restart mid-fair must resume, not silently drop anyone.
        c = st.save("c@x.com", idx, r)
        st2 = capture.CaptureStore(d)
        check("restart rehydrates the queue",
              [x["email"] for x in st2.pending()] == ["c@x.com"],
              str([x["email"] for x in st2.pending()]))
        check("marks survive a reload", st2.sent_count == 1)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_legacy_index(r):
    """An index from before send-tracking is still real addresses."""
    import json
    d = Path(tempfile.mkdtemp())
    try:
        (d / "captures.json").write_text(json.dumps([
            {"email": "old@x.com", "captured_at": "2026-08-01T10:00:00",
             "image": "x.png", "text": "x.txt", "emailed": False},
            {"email": "done@x.com", "captured_at": "2026-08-01T10:01:00",
             "image": "y.png", "text": "y.txt", "emailed": True},
        ]))
        st = capture.CaptureStore(d)
        check("legacy records gain ids", all(len(x["id"]) == 32
                                             for x in st.records))
        check("legacy unsent becomes pending",
              [x["email"] for x in st.pending()] == ["old@x.com"])
        check("legacy emailed becomes sent", st.sent_count == 1)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_message():
    cfg = dict(user="c@g.com", password="p" * 16, from_name="Air Writer",
               reply_to="club@x.org", club="Coding Club", enabled=True)
    d = Path(tempfile.mkdtemp())
    try:
        png = d / "p.png"
        cv2.imwrite(str(png), np.full((900, 900, 3), 220, np.uint8))
        m = mailer.build_message(cfg, "you@x.com", png, "@@##\n##@@", "now")
        types = [p.get_content_type() for p in m.walk()]
        check("message has a plain-text part", "text/plain" in types)
        check("message has an html part", "text/html" in types)
        check("portrait is embedded once", types.count("image/png") == 1,
              f"{types.count('image/png')} copies")
        html = [p for p in m.walk()
                if p.get_content_type() == "text/html"][0].get_content()
        cid = [p for p in m.walk() if p.get("Content-ID")][0]
        check("html references the inline image",
              f"cid:{cid['Content-ID'].strip('<>')}" in html)
        check("reply-to is set", m["Reply-To"] == "club@x.org")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _drain(sender, timeout=6.0):
    t = time.time()
    while time.time() - t < timeout:
        got = sender.results()
        if got:
            return got
        time.sleep(0.02)
    return []


def test_send_retry():
    cfg = dict(user="c@g.com", password="p" * 16, from_name="AW",
               reply_to="", club="", enabled=True)

    tries = {"n": 0}

    def flaky(c, m):
        tries["n"] += 1
        if tries["n"] < 3:
            raise TimeoutError("network unreachable")

    s = mailer.EmailSender(cfg, transport=flaky, backoff=(0.01, 0.01, 0.01))
    s.submit("r1", "a@b.com", None, "art")
    got = _drain(s)
    check("transient failure retries then sends",
          got and got[0][1] == "sent" and tries["n"] == 3, str(got))

    hits = {"n": 0}

    def badauth(c, m):
        hits["n"] += 1
        raise smtplib.SMTPAuthenticationError(535, b"bad password")

    s2 = mailer.EmailSender(cfg, transport=badauth, backoff=(0.01,) * 4)
    s2.submit("r2", "a@b.com", None, "art")
    got = _drain(s2)
    check("bad password gives up immediately",
          got and got[0][1] == "failed" and hits["n"] == 1,
          f"{got} after {hits['n']} tries")

    s3 = mailer.EmailSender(cfg, transport=lambda c, m: (_ for _ in ()).throw(
        OSError("no route")), backoff=(0.01, 0.01))
    s3.submit("r3", "a@b.com", None, "art")
    got = _drain(s3)
    check("unreachable network ends as failed, not lost",
          got and got[0][1] == "failed", str(got))

    off = mailer.EmailSender(dict(cfg, enabled=False))
    check("disabled sender collects without sending",
          off.submit("r4", "a@b.com", None, "") is False)


def test_config():
    d = Path(tempfile.mkdtemp())
    try:
        f = d / ".env"
        f.write_text("AIRWRITER_SMTP_USER=you@gmail.com\n"
                     "AIRWRITER_SMTP_PASS=xxxxxxxxxxxxxxxx\n")
        check("placeholder credentials do not count as configured",
              not mailer.load_config(f)["enabled"])
        f.write_text("AIRWRITER_SMTP_USER=club@gmail.com\n"
                     "AIRWRITER_SMTP_PASS=abcd efgh ijkl mnop\n")
        c = mailer.load_config(f)
        check("app password survives being pasted with spaces",
              c["password"] == "abcdefghijklmnop", c["password"])
        check("reply-to falls back to the sender",
              c["reply_to"] == "club@gmail.com")
        check("missing .env is not an error",
              mailer.load_config(d / "nope")["enabled"] is False)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    print("Air Writer — pre-fair checks (offline; nothing is mailed)\n")
    r = fa.AsciiRenderer()
    img = head()

    for title, fn in (
        ("portrait styles", lambda: (test_ramps(r), test_styles_render(r, img))),
        ("export parity", lambda: test_export_parity(r, img)),
        ("live ASCII mirror", lambda: test_live(r)),
        ("capture guards", lambda: test_capture_guards(r, img)),
        ("store lifecycle", lambda: test_store_lifecycle(r, img)),
        ("legacy index", lambda: test_legacy_index(r)),
        ("email message", test_message),
        ("send + retry", test_send_retry),
        ("configuration", test_config),
    ):
        print(f"{title}:")
        fn()
        print()

    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f"  FAILED: {n}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
