"""Colour probe: what the PS5 stream really carries, next to what it says it carries.

Run it while SmoovPlay (or chiaki-ng with the frame tap) is streaming, or just before. It joins a running stream
with the codec header play.py saved and asks chiaki-ng for one keyframe (or reads a new stream from its start), decodes a few seconds on its own (play.py is not disturbed)
and reports, per decoded frame: the declared range/matrix/transfer, and the real luma levels (share of pixels
above 235 and below 16: in a limited-range stream those should be rare). It saves the last frame as PNG twice:
converted as declared, and converted as full range, to compare by eye.

  .venv-mac/bin/python mac/color_probe.py [--seconds 5]

Output in ~/Library/Logs/SmoovPlay/color_probe/<date-time>/.
"""

from __future__ import annotations

import argparse
import struct
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from framegen.chiaki_tap import CODECS, NAME, PARAM_NALS, SLOTS, FrameTap, nal_units, param_sets  # noqa: E402

RANGES = {0: "unspecified", 1: "limited (16-235)", 2: "full (0-255)"}


def luma(fr) -> np.ndarray:
    p = fr.planes[0]
    a = np.frombuffer(p, np.uint8 if "10" not in fr.format.name else np.uint16)
    return a.reshape(fr.height, -1)[:, :fr.width]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args()

    tap = FrameTap()
    h = tap.header()
    t_wait = time.monotonic() + 60
    if not h.live:
        print("Waiting for a stream (60 s)...", flush=True)
    while not h.live:
        if time.monotonic() > t_wait:
            sys.exit("No stream started: open SmoovPlay, connect to the PS5, then run this again.")
        time.sleep(0.05)
        h = tap.header()
    codec = CODECS.get(h.codec, "hevc")
    if h.write_seq <= SLOTS - 4:  # the stream just started: its codec header is still in the ring
        params, seq = b"", 0
    else:  # joining a running stream: the header play.py saved for it
        hdr = Path(tempfile.gettempdir()) / (NAME.split("\\")[-1] + "-header.bin")
        try:
            b = hdr.read_bytes()
        except OSError:
            sys.exit(f"No saved codec header ({hdr}): start the stream with SmoovPlay running first.")
        if struct.unpack_from("<II", b) != (h.session, h.codec):
            sys.exit("The saved codec header is from another stream: reconnect to the PS5, then run this again.")
        params, seq = b[8:], h.write_seq
        tap.request_idr()  # a keyframe to start decoding from (the PS5 sends one on request)
    ctx = av.CodecContext.create(codec, "r")
    ctx.flags |= av.codec.context.Flags.low_delay

    out = Path.home() / "Library" / "Logs" / "SmoovPlay" / "color_probe" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    t_end = time.monotonic() + a.seconds + 3
    first, rows, last = True, [], None
    while time.monotonic() < t_end:
        h = tap.header()
        if h.write_seq <= seq:
            time.sleep(0.005)
            continue
        if h.write_seq - seq > 24:
            seq = h.write_seq - 1
        p = tap.read(seq, h.qpc_freq, h.slots, h.slot_size)
        seq += 1
        if p is None or not p.data:
            continue
        ps = param_sets(p.data, codec)
        if ps:
            params = ps
            if all(t in PARAM_NALS[codec] for t, _ in nal_units(p.data, codec)):  # the header on its own
                continue
        if not params:
            continue
        data = (params + p.data) if first else p.data
        try:
            frames = ctx.decode(av.Packet(data))
        except av.error.InvalidDataError:
            continue
        for fr in frames:
            first = False
            y = luma(fr)
            hi = 940 if y.dtype == np.uint16 else 235
            lo = 64 if y.dtype == np.uint16 else 16
            rows.append((fr.format.name, fr.color_range, fr.colorspace, getattr(fr, "color_trc", None),
                         float((y > hi).mean() * 100), float((y < lo).mean() * 100), int(y.max()), int(y.min()),
                         float(np.percentile(y, 99))))
            last = fr
    if not rows:
        sys.exit("No frame decoded (no keyframe came). Move in the game a bit and run this again.")

    fmt, rng, cs, trc = rows[-1][:4]
    above = [r[4] for r in rows]
    below = [r[5] for r in rows]
    lines = [
        f"{len(rows)} frames decoded. Declared: {fmt}, range {RANGES.get(rng, rng)}, matrix {cs} "
        f"(1 = BT.709), transfer {trc} (1 = BT.709, 16 = PQ/HDR10, 18 = HLG).",
        f"Luma above 235 (whiter than limited-range white): mean {np.mean(above):.2f}% of pixels, "
        f"max {max(above):.2f}%.",
        f"Luma below 16 (blacker than limited-range black): mean {np.mean(below):.2f}%, max {max(below):.2f}%.",
        f"Luma max {max(r[6] for r in rows)}, min {min(r[7] for r in rows)}, 99th percentile (last frame) "
        f"{rows[-1][8]:.0f}.",
        "",
        "Reading it: in a real limited-range picture only a fraction of a percent goes above 235. Several percent "
        "means the PS5 puts full-range values in a stream it labels limited, and converting it as declared burns "
        "the whites (compare as_declared.png with as_full_range.png).",
    ]
    text = "\n".join(lines) + "\n"
    (out / "report.txt").write_text(text)
    from PIL import Image
    Image.fromarray(last.to_ndarray(format="rgb24")).save(out / "as_declared.png")
    last.color_range = 2
    Image.fromarray(last.to_ndarray(format="rgb24")).save(out / "as_full_range.png")
    print(text + f"\nSaved in {out}")


if __name__ == "__main__":
    main()
