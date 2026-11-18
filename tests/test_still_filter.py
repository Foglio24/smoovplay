"""The still-picture filter of mac/play.py must pass slow motion and drop true repeats.

The stream is 60 Hz and the game 30 fps, so every picture arrives twice. A screen that moves slowly changes by far less
than STILL_LEVEL between two pictures; comparing each frame only with the last one *shown* held the slow motion back
until the changes added up, so a slow pan was shown at a few fps.

Run: python tests/test_still_filter.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("play_mac", ROOT / "mac" / "play.py")
play = importlib.util.module_from_spec(spec)
sys.modules["play_mac"] = play
spec.loader.exec_module(play)


def slow_pan(n: int, step: float, h: int = 1080, w: int = 1920) -> list[np.ndarray]:
    """n distinct pictures, each differing from the one before by `step` grey levels on average."""
    rng = np.random.default_rng(1)
    base = rng.uniform(40, 200, (h // 16, w // 16)).repeat(16, 0).repeat(16, 1)
    return [np.repeat(np.round(base + k * step)[..., None], 3, axis=2).astype(np.uint8) for k in range(n)]


def emitted(frames: list[np.ndarray]) -> int:
    f = play.StillFilter()
    return sum(1 for fr in frames if f.is_new(fr))


def check(name: str, got: int, want: range) -> bool:
    ok = got in want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} passed (want {want.start}..{want.stop - 1})")
    return ok


def main() -> int:
    ok = True
    pics = slow_pan(30, 0.08)  # 30 distinct pictures of a slow pan
    doubled = [p for p in pics for _ in range(2)]  # the 60 Hz stream: each one twice
    ok &= check("slow pan, every picture twice", emitted(doubled), range(29, 31))
    ok &= check("true still screen (60 identical frames)", emitted([pics[0]] * 60), range(1, 2))
    cut = [pics[0]] * 5 + [255 - pics[0]] * 5
    ok &= check("hard cut in a still screen", emitted(cut), range(2, 3))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
