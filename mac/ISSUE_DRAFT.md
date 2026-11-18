**Title:** macOS (Apple Silicon) port: feasibility numbers and a proposed plan

Hi! I really like SmoovPlay and wanted to see whether a Mac version is realistic, so I measured RIFE 4.22 lite on an Apple Silicon Mac. Short version: **1080p 30 → 60 fps fits on the GPU of an M5 Pro with Core ML; 720p → 90 fps also fits; 120 fps doesn't.** Details, scripts and a possible plan below, in case it's useful.

### Setup

Mac mini, Apple M5 Pro, macOS 27.2.1, Python 3.12, PyTorch 2.15 (MPS), coremltools 9.0. Synthetic timing plus a real frame pair (`framegen/samples/gta6-beach.jpg`, panned), with the picture compared against a CPU fp32 run of the unmodified vs-rife forward (`reference=True`), so a fast but wrong backend can't pass.

Budget: a 30 fps stream gives 33.3 ms per real frame. I kept 20% for decode and display, so **≤ 26.7 ms** per real frame for prepare + 1 (60 fps), 2 (90) or 3 (120) AI frames.

### PyTorch on MPS (`RifeInterpolator(device="mps", cuda_graph=False)`, unchanged code)

Everything runs on the GPU (no MPS CPU fallbacks), picture matches (PSNR 54–101 dB vs the reference).

| Stream | Motion | fp16, fast path: cost for 60 / 90 / 120 fps (ms) |
|---|---|---|
| 720p | full | 20.1 / 37.7 / 55.2 |
| 720p | half | 15.9 / 29.1 / 42.4 |
| 1080p | full | 40.6 / 75.8 / 111.1 |
| 1080p | half | 29.9 / 54.4 / 78.9 |

One AI frame at 1080p full is ~35 ms, about 4× the 9 ms on an RTX 4070 Ti SUPER. FastIFNet beats the reference forward on MPS as well (35 vs 51 ms at 1080p full).

### Core ML (same forward, converted with coremltools, fp16, one call = both encoders + flow net)

| Stream | Motion | GPU, one AI frame incl. copies (ms) | PSNR vs reference |
|---|---|---|---|
| 720p | full | 14.2 | 54.4 |
| 720p | half | 11.0 | 49.6 |
| 1080p | full | 29.6 | 52.7 |
| 1080p | half | **22.4** | 51.9 |

About 25% faster than MPS, which puts **1080p half at 60 fps** inside the budget (and 720p half at 90). These numbers are slightly pessimistic, since each call re-encodes both frames. The **Neural Engine is ~20× slower** than the GPU here (probably the `grid_sample` warps), so the GPU is the target.

To convert, I had to replace `scale_factor` resizes with fixed sizes; otherwise the traced shape arithmetic doesn't convert. The forward is FastIFNet's, as a plain `nn.Module` with the timestep as an input.

### What a port would touch

| Today (Windows / NVIDIA) | macOS |
|---|---|
| chiaki-ng + frame tap (`CreateFileMapping`) | chiaki-ng already builds on macOS; the tap would need POSIX `shm_open` + `mmap` (same layout) |
| NVDEC decode | VideoToolbox (FFmpeg `hwaccel=videotoolbox`) |
| RIFE: TensorRT / CUDA graphs / DirectML | Core ML on the GPU (`CPU_AND_GPU`), ideally fed from `CVPixelBuffer`/IOSurface to avoid copies |
| FSR 1 / RTX VSR | MetalFX spatial upscaler, or FSR 1 as a Metal/GL 4.1 shader |
| DXGI present, DDA capture, HDR | Metal (CAMetalLayer) present; ScreenCaptureKit for App window / capture card |
| OpenGL window + menus | GL is capped at 4.1 on macOS and deprecated: shaders need a check, Metal long-term |
| Hotkeys, `winutil` | Cocoa / Quartz equivalents |
| NVIDIA VFG, RTX VSR | not available |

Realistic Mac targets: 1080p → 60 fps, 720p → 90 fps.

### Prototype

I also put together a small standalone prototype, `mac/play.py`: a 30 fps source (demo pan or a video file, paced like a live stream) → Core ML RIFE in a separate process → OpenGL window, with Space to toggle AI frames. On the M5 Pro (2560×1440 at 240 Hz), 1080p 30 → 60 fps with half motion held **60 fps for about 2.5 minutes: 22.2 ms per AI frame, no frame late, 8 dropped out of about 8,000**. Toggling the AI frames off and on goes cleanly between 30 and 60 fps. The delay from a frame's arrival to the screen is fixed at 80 ms; I haven't tuned it.

Scripts (happy to open a PR if you want them in the repo):
- `scripts/bench_mac.py`: the MPS table above
- `scripts/coreml_mac.py`: Core ML conversion + timing + picture check
- `mac/rife_coreml.py`, `mac/play.py`: the prototype

Would you be open to a macOS port? If so, which parts would you rather keep in your hands?
