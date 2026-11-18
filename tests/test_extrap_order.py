"""Extrapolated AI frames must never be shown after a real frame that is newer than the one they were drawn from.

The window (mac/play.py) orders frames by their time. An extrapolated frame sits half a game gap after its real frame;
when the next real frame arrives sooner than that (arrival jitter), the extrapolated one would come after it and the
picture would jump back. Run: python tests/test_extrap_order.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("play_mac", ROOT / "mac" / "play.py")
play = importlib.util.module_from_spec(spec)
sys.modules["play_mac"] = play
spec.loader.exec_module(play)

GAP = 1 / 30


def show_order(events, delay=0.05):
    """events: (time the message reaches the window, pts, base or None). Returns the pts shown, in order, using the
    window's own rules; frames are shown at their pts + delay."""
    pending, newest, stale = [], -1.0, 0
    for _, pts, base in sorted(events):
        if play.is_stale(base, newest, pts):
            stale += 1
            continue
        if base is None:
            newest = max(newest, pts)
            pending, dropped = play.drop_stale(pending, pts)
            stale += len(dropped)
        pending.append((None, pts + delay, base, pts))
    return [e[3] for e in sorted(pending, key=lambda e: e[1])], stale


def main() -> int:
    ok = True
    # steady 30 fps: real k at k*GAP, its extrapolated frame half a gap later
    ev = []
    for k in range(10):
        ev += [(k * GAP, k * GAP, None), (k * GAP + 0.015, k * GAP + GAP / 2, k * GAP)]
    shown, stale = show_order(ev)
    good = shown == sorted(shown) and stale == 0 and len(shown) == 20
    print(f"{'ok  ' if good else 'FAIL'} steady 30 fps: 20 frames in order, {stale} dropped")
    ok &= good

    # the next real frame arrives early (10 ms after the last, jitter): the extrapolated frame of the one before is stale
    ev = [(0.0, 0.0, None), (0.015, GAP / 2, 0.0), (0.040, 0.010 + 0.0, None)]  # real at 0.010 < the extrapolated at 0.0167
    shown, stale = show_order(ev)
    good = stale == 1 and shown == [0.0, 0.010]
    print(f"{'ok  ' if good else 'FAIL'} early real frame: stale {stale}, shown {shown}")
    ok &= good

    # the extrapolated frame is computed late and reaches the window after the next real frame, which is later than it
    ev = [(0.0, 0.0, None), (0.040, GAP, None), (0.045, GAP / 2, 0.0)]
    shown, stale = show_order(ev)
    good = stale == 0 and shown == sorted(shown) and len(shown) == 3
    print(f"{'ok  ' if good else 'FAIL'} late but still before the next real frame: kept, in order {shown}")
    ok &= good

    # ... and one that would fall after the next real frame is dropped even if it arrives after it
    ev = [(0.0, 0.0, None), (0.030, 0.010, None), (0.045, GAP / 2, 0.0)]
    shown, stale = show_order(ev)
    good = stale == 1 and shown == [0.0, 0.010]
    print(f"{'ok  ' if good else 'FAIL'} late and after the next real frame: dropped, shown {shown}")
    ok &= good
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
