"""RIFE 4.22 lite as a Core ML model that takes and gives plain RGB pictures (macOS, Apple Silicon GPU).

Image in, image out: Core ML turns the 8-bit frames into the model's float16 0..1 itself, so Python does no
per-pixel work (at 1080p a NumPy conversion each way would cost more than 10% of the frame budget). The model pads
to the size the network needs and crops back inside, so frames go in and come out at their own size.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from coreml_mac import MODULO, RifePair, VARIANTS  # noqa: E402  the forward pass, checked against vs-rife there

MODEL_DIR = ROOT / "models" / "coreml"


def padded(w: int, h: int, scale: float) -> tuple[int, int]:
    step = max(MODULO, int(MODULO / scale))
    return math.ceil(w / step) * step, math.ceil(h / step) * step


class RifeImages(nn.Module):
    """(img0, img1) as (1,3,h,w) 0..1 RGB, t (1,1,1,1) -> (1,3,h,w) 0..255."""

    def __init__(self, w: int, h: int, scale: float, variant: str = "base") -> None:
        super().__init__()
        pw, ph = padded(w, h, scale)
        self.core = RifePair(scale, ph, pw, variant)
        self.pad = (0, pw - w, 0, ph - h)
        self.w, self.h = w, h

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.pad[1] or self.pad[3]:
            img0, img1 = F.pad(img0, self.pad), F.pad(img1, self.pad)
        return self.core(img0, img1, t)[:, :, : self.h, : self.w] * 255.0


def model_path(w: int, h: int, scale: float, variant: str = "base") -> Path:
    # A changed graph needs its own name: an existing package is reused as it is.
    tag = "" if variant == "base" else f"_{variant}"
    return MODEL_DIR / f"rife422lite_img_{w}x{h}_{'full' if scale == 1.0 else 'half'}{tag}.mlpackage"


def convert(w: int, h: int, scale: float, variant: str = "base") -> Path:
    """Build (once) and return the .mlpackage for this frame size, motion detail and variant."""
    path = model_path(w, h, scale, variant)
    if path.exists():
        return path
    import coremltools as ct
    import numpy as np
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[coreml] building {path.name} (first time only, about a minute)...", flush=True)
    model = RifeImages(w, h, scale, variant).eval()
    ex = (torch.rand(1, 3, h, w), torch.rand(1, 3, h, w), torch.full((1, 1, 1, 1), 0.5))
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    rgb = ct.colorlayout.RGB
    ml = ct.convert(
        traced,
        inputs=[ct.ImageType("img0", shape=ex[0].shape, scale=1 / 255.0, color_layout=rgb),
                ct.ImageType("img1", shape=ex[1].shape, scale=1 / 255.0, color_layout=rgb),
                ct.TensorType("t", shape=ex[2].shape, dtype=np.float16)],
        outputs=[ct.ImageType("out", color_layout=rgb)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    tmp = path.with_suffix(".part.mlpackage")
    ml.save(str(tmp))
    tmp.rename(path)
    return path


class CoreMLRife:
    """Draws in-between frames on the GPU. Frames are HxWx3 uint8 RGB NumPy arrays."""

    def __init__(self, w: int, h: int, scale: float, variant: str = "base") -> None:
        import coremltools as ct
        import numpy as np
        self._np = np
        self.model = ct.models.MLModel(str(convert(w, h, scale, variant)), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        self._t = {}

    def interpolate(self, img0, img1, t: float):
        """img0/img1: PIL images (reuse them: each real frame is both the end of one pair and the start of the
        next). Returns the frame at fraction t as an HxWx3 uint8 array."""
        if t not in self._t:
            self._t[t] = self._np.full((1, 1, 1, 1), t, self._np.float16)
        out = self.model.predict({"img0": img0, "img1": img1, "t": self._t[t]})["out"]
        return self._np.asarray(out.convert("RGB"))
