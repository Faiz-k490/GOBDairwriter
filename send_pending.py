"""
send_pending — mail every portrait the demo could not send at the time.

The app queues sends in the background, but a fair hall's wifi is not a thing
to rely on, and the whole point of collecting an address is that the person
eventually gets their picture.  Anything still marked pending — because the
network was down, because the laptop was closed mid-send, because emailing was
switched off for the day — is drained by this.

    python send_pending.py --dry-run     # show who is owed, send nothing
    python send_pending.py               # actually send
    python send_pending.py --retry-failed

Safe to re-run: each record is marked as it lands, so a second run only picks
up what is still outstanding.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import capture
import mailer

HERE = Path(__file__).parent


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be sent, without sending")
    ap.add_argument("--retry-failed", action="store_true",
                    help="also retry records that previously gave up")
    ap.add_argument("--limit", type=int, default=0,
                    help="send at most N (a consumer Gmail caps near 500/day)")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="seconds between sends (default 2)")
    args = ap.parse_args(argv)

    store = capture.CaptureStore(HERE / "captures")
    cfg = mailer.load_config()

    owed = store.pending()
    if args.retry_failed:
        owed += [r for r in store.records if r.get("status") == capture.FAILED]
    if args.limit:
        owed = owed[:args.limit]

    print(f"{store.count} captures on file · {store.sent_count} sent · "
          f"{len(owed)} to send")
    if not owed:
        print("Nothing outstanding.")
        return 0

    for r in owed:
        print(f"  {r['email']:38} {r.get('captured_at', '')}"
              f"{'  (previously failed)' if r.get('status') == capture.FAILED else ''}")

    if args.dry_run:
        print("\n--dry-run: nothing sent.")
        return 0

    if not cfg["enabled"]:
        print("\nEmailing is not configured — set AIRWRITER_SMTP_USER and "
              "AIRWRITER_SMTP_PASS in .env (see .env.example).",
              file=sys.stderr)
        return 1

    print(f"\nSending as {cfg['user']}…")
    sent = failed = 0
    for i, r in enumerate(owed, 1):
        png = store.dir / r.get("image", "")
        txt = store.dir / r.get("text", "")
        art = txt.read_text() if txt.exists() else ""
        msg = mailer.build_message(cfg, r["email"], png if png.exists() else None,
                                   art, r.get("captured_at", ""))
        try:
            mailer.send_once(cfg, msg)
        except Exception as e:
            failed += 1
            store.mark(r["id"], capture.FAILED, f"{type(e).__name__}: {e}")
            print(f"  [{i}/{len(owed)}] ✗ {r['email']}: {e}")
        else:
            sent += 1
            store.mark(r["id"], capture.SENT)
            print(f"  [{i}/{len(owed)}] ✓ {r['email']}")
        # Spacing out a bulk run looks less like spam to the receiving side.
        if i < len(owed) and args.delay:
            time.sleep(args.delay + random.uniform(0, 0.5))

    print(f"\nSent {sent}, failed {failed}.")
    if failed:
        print("Re-run with --retry-failed once the network is better.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
