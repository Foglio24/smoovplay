"""Can an AI frame be made from the two real frames that have already arrived (no waiting for the next one)?

Interpolation draws a frame halfway between two real frames, so it can't be shown before the later one has arrived
and been processed: the display delay has a floor of half a frame interval plus the model's time. Extrapolation draws a
frame *after* the latest real one, from the motion between the last two, and has no such floor. This measures how much
picture quality that costs on real gameplay, with the same RIFE 4.22 lite flow the app uses:

  hold    show the latest real frame again (what no frame generation does)
  interp  the app's AI frame, halfway between two real frames (needs the later one: the delay floor)
  extrap  the latest real frame warped on by the motion between the last two (needs nothing later)

All three are measured against the real frame at the same moment, on a clip whose frames are `gap` apart (two clip
frames apart, so the motion is twice what a live 30 fps stream has: a hard case for the app).

  python scripts/extrapolation_probe.py CLIP.mp4 [--frames 140] [--out DIR]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import coreml_mac as cm  # noqa: E402
from coreml_mac import _resize  # noqa: E402


def flows_and_interp(model: cm.RifePair, a: torch.Tensor, b: torch.Tensor, t: float = 0.5):
    """lean2's forward, returning the interpolated frame and the two flows (mid -> frame a, mid -> frame b)."""
    tt = torch.full((1, 1, 1, 1), t, device=a.device)
    src0 = torch.cat([a, model.encode(a)], 1)
    src1 = torch.cat([b, model.encode(b)], 1)
    s0 = [_resize(src0, sz) for sz in model.sizes]
    s1 = [_resize(src1, sz) for sz in model.sizes]
    flow = mask = feat = None
    for i, (blk, s, sz) in enumerate(zip(model.blocks, model.scales, model.sizes)):
        if flow is None:
            x, y = s0[0], s1[0]
            parts = [x[:, :3], y[:, :3], x[:, 3:], y[:, 3:], tt.expand(1, 1, sz[0], sz[1])]
        else:
            fl = _resize(flow, sz)
            x, y = model._warp_k(s0[i], fl[:, :2], i), model._warp_k(s1[i], fl[:, 2:4], i)
            parts = [x[:, :3], y[:, :3], x[:, 3:], y[:, 3:], tt.expand(1, 1, sz[0], sz[1]), mask, feat, fl / s]
        tmp = blk.lastconv(blk.convblock(blk.conv0(torch.cat(parts, 1))))
        fd = tmp[:, :4] * s
        flow = fd if flow is None else fl + fd
        mask, feat = tmp[:, 4:5], tmp[:, 5:]
        if i < 3:
            mask, feat = _resize(mask, model.sizes[i + 1]), _resize(feat, model.sizes[i + 1])
    flow, mask = _resize(flow, model.full), _resize(mask, model.full)
    w0 = model._warp(src0[:, :3], flow[:, :2])
    w1 = model._warp(src1[:, :3], flow[:, 2:4])
    m = torch.sigmoid(mask)
    return (w0 * m + w1 * (1 - m)).clamp(0, 1), flow[:, :2], flow[:, 2:4]


# f1 - f0 of two identical frames, measured over 15 frames of the demo clip at 1080p (sd 0.01 px): the flow network
# reports this much motion between frames that don't move at all, so a still picture would be drawn shifted (0.2 px
# at tau = 0.5), a tremble against the real frames. It is subtracted from the motion.
MOTION_BIAS = (-0.034, -0.403)


def extrap(model: cm.RifePair, b: torch.Tensor, f0: torch.Tensor, f1: torch.Tensor, tau: float, regrid: bool) -> torch.Tensor:
    """b warped on by tau of the motion between the two real frames. D = f1 - f0 is that motion (per gap) but sits on the
    grid of the frame halfway between them; with `regrid` it is first read at p - tau*D, where the new frame's content
    was, so the motion field moves with the objects instead of staying behind."""
    d = f1 - f0 - torch.tensor(MOTION_BIAS, device=f1.device).view(1, 2, 1, 1)
    if regrid:
        d = model._warp(d, -tau * d)  # the same bilinear sampler the warps use, on the 2-channel motion field
    return model._warp(b, -tau * d)


def shift_of(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(dx, dy) in pixels that moves picture a onto b, by phase correlation on a centre crop."""
    import cv2
    h, w = a.shape[:2]
    c = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    g = lambda x: cv2.cvtColor(x[c], cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(g(a), g(b), cv2.createHanningWindow(g(a).shape[::-1], cv2.CV_32F))
    return dx, dy


def translation_test(model, tens, back, frames, dev) -> None:
    """Pure integer translations and identical pairs through the extrapolation: the result must be b moved on by half the
    motion between the two frames (tau = 0.5), nothing more or less, and b itself when the two frames are the same."""
    import cv2
    print("translation test: shift of the extrapolated frame from the real frame b, in px (x, y)")
    print("   motion per gap (x,y)   wanted      f1 only (x,y)       (f1-f0)  (x,y)       (f1-f0) regrid (x,y)")
    rows = []
    for img in (frames[0], frames[40], frames[80]):
        h, w = img.shape[:2]
        for vx, vy in ((0, 0), (4, 0), (-4, 0), (0, 4), (0, -6), (6, 6), (-6, 4)):
            m = lambda dx, dy: cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_NEAREST,
                                              borderMode=cv2.BORDER_REFLECT)
            fa, fb = m(0, 0), m(vx, vy)
            _, f0, f1 = flows_and_interp(model, tens(fa), tens(fb))
            tb = tens(fb)
            got = []
            for var in (lambda: model._warp(tb, -f1), lambda: extrap(model, tb, f0, f1, 0.5, False),
                        lambda: extrap(model, tb, f0, f1, 0.5, True)):
                got.append(shift_of(fb, back(var())))
            rows.append(((vx, vy), got))
    for (vx, vy) in sorted({r[0] for r in rows}):
        sel = [r[1] for r in rows if r[0] == (vx, vy)]
        mean = lambda k: tuple(np.mean([g[k][i] for g in sel], axis=0) for i in (0, 1))
        cells = "".join(f"   ({mean(k)[0]:+5.2f},{mean(k)[1]:+5.2f})   " for k in range(3))
        print(f"   ({vx:+d},{vy:+d})            ({vx / 2:+4.1f},{vy / 2:+4.1f}) {cells}")
    ident = [g[1] for (v, g) in rows if v == (0, 0)]
    print("   identical pair, (f1-f0): worst |shift| %.3f px" % max(max(abs(x), abs(y)) for x, y in ident))


def hud_test(model, tens, back, frames, g: int) -> None:
    """A static interface over a moving picture must stay put. Pastes the same patch (a strip of the clip's top-left
    and one of its bottom-left corner: world texture, standing in for HUD elements) at the same place into both real frames and into the real frame being predicted,
    then measures how far the patch ends up from where it really is in the extrapolated frame (and in the interpolated
    one, for comparison). The world moves under it exactly as in the clip; the patch never moves."""
    boxes = {"top-left strip": (30, 10, 570, 90), "bottom-left patch": (24, 960, 210, 1050)}
    patches = {n: frames[0][y0:y1, x0:x1].copy() for n, (x0, y0, x1, y1) in boxes.items()}

    def stamp(img):
        img = img.copy()
        for n, (x0, y0, x1, y1) in boxes.items():
            img[y0:y1, x0:x1] = patches[n]
        return img

    res: dict = {}
    with torch.no_grad():
        for i in range(0, len(frames) - 2 * g - 1, 6):
            fa, fb = stamp(frames[i]), stamp(frames[i + g])
            mid_gt, nxt_gt = stamp(frames[i + g // 2]), stamp(frames[i + g + g // 2])
            interp, f0, f1 = flows_and_interp(model, tens(fa), tens(fb))
            e = back(extrap(model, tens(fb), f0, f1, 0.5, False))
            moving = float(np.abs(frames[i + g].astype(np.float32) - frames[i].astype(np.float32)).mean())
            for n, (x0, y0, x1, y1) in boxes.items():
                for kind, img in (("extrap", e), ("interp", back(interp))):
                    crop = img[y0:y1, x0:x1]
                    res.setdefault((n, kind), []).append((psnr(crop, patches[n]), *shift_of_crop(crop, patches[n]), moving))
    print("static interface over moving gameplay: the pasted patch vs where it really is (shift in px; 0 = it stayed put)")
    for (n, kind), v in res.items():
        v = np.array(v)
        print(f"   {n:13s} {kind:7s} PSNR {v[:, 0].mean():5.1f} dB (worst {v[:, 0].min():5.1f})   shift x {np.abs(v[:, 1]).mean():.2f}"
              f" (worst {np.abs(v[:, 1]).max():.2f})  y {np.abs(v[:, 2]).mean():.2f} (worst {np.abs(v[:, 2]).max():.2f})")


def shift_of_crop(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    import cv2
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(g(a), g(b), cv2.createHanningWindow(g(a).shape[::-1], cv2.CV_32F))
    return dx, dy


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--gap", type=int, default=2, help="clip frames between the two real frames")
    ap.add_argument("--out", default=None, help="a directory for side-by-side pictures of the worst cases")
    a = ap.parse_args()
    import av
    from PIL import Image

    frames = []
    with av.open(a.clip) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i < a.start:
                continue
            if i >= a.start + a.frames:
                break
            frames.append(fr.to_ndarray(format="rgb24"))
    h, w = frames[0].shape[:2]
    pw, ph = cm.padded(w, h, 0.5)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = cm.RifePair(0.5, ph, pw, "lean2").eval().to(dev)

    def tens(img: np.ndarray) -> torch.Tensor:
        x = torch.zeros(1, 3, ph, pw, device=dev)
        x[0, :, :h, :w] = torch.from_numpy(img).to(dev).permute(2, 0, 1).float() / 255
        return x

    def back(x: torch.Tensor) -> np.ndarray:
        return (x[0, :, :h, :w].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()

    g = a.gap
    rows = []
    with torch.no_grad():
        translation_test(model, tens, back, frames, dev)
        hud_test(model, tens, back, frames, g)
        for i in range(0, len(frames) - 2 * g - 1, 6):
            fa, fb = frames[i], frames[i + g]
            mid_gt, nxt_gt = frames[i + g // 2], frames[i + g + g // 2]  # halfway between them / half a gap after b
            interp, f0, f1 = flows_and_interp(model, tens(fa), tens(fb))
            tb = tens(fb)
            ext = {"extrap (f1)": model._warp(tb, -f1), "extrap (f1-f0)/2": extrap(model, tb, f0, f1, 0.5, False),
                   "extrap regrid": extrap(model, tb, f0, f1, 0.5, True)}
            row = {"hold": psnr(fb, nxt_gt), "interp": psnr(back(interp), mid_gt), "hold_mid": psnr(fa, mid_gt)}
            for k, v in ext.items():
                row[k] = psnr(back(v), nxt_gt)
            rows.append((i, row, back(ext["extrap (f1)"]), nxt_gt, fb))
    names = ["hold", "extrap (f1)", "extrap (f1-f0)/2", "extrap regrid", "interp", "hold_mid"]
    print(f"{len(rows)} samples, real frames {g} clip frames apart, PSNR dB against the real frame at that moment")
    for n in names:
        v = np.array([r[1][n] for r in rows])
        print(f"  {n:18s} mean {v.mean():5.2f}  worst10% {np.percentile(v, 10):5.2f}  best10% {np.percentile(v, 90):5.2f}")
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        worst = sorted(rows, key=lambda r: r[1]["extrap (f1)"])[:2]
        for i, _, ex, gt, hold in worst:
            Image.fromarray(np.concatenate([hold, ex, gt], 1)).resize((1920, 360)).save(out / f"extrap_{i}.png")
        print("pictures (hold | extrapolated | real):", out)


if __name__ == "__main__":
    main()
