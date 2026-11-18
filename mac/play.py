"""SmoovPlay on a Mac, first prototype: a 30 fps video played at 60 (or 90) fps with RIFE on the Apple GPU.

No console yet. The source is a demo (a camera panning over the sample screenshots) or any video file, read at
its real speed as if it were a live stream: each frame becomes available only at its own time, so what you see
is what a Remote Play stream would get, delay included.

  python3 mac/play.py                    # demo, 1080p 30 fps -> 60 fps
  python3 mac/play.py --res 720p --mult 3   # demo, 720p -> 90 fps
  python3 mac/play.py ~/Movies/clip.mp4  # a video of yours (30 fps works best)
  python3 mac/play.py --tap              # live from the frame tap: chiaki-ng (a PS5) or scripts/frametap_fake.py

Keys: SPACE turns the AI frames on and off (compare 30 and 60), F full screen, Q or ESC quits.

Two processes: the worker reads frames and runs Core ML, the main one only shows them, so a slow AI frame never
holds up the screen. Frames pass through shared memory.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import sys
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = {"720p": (1280, 720), "1080p": (1920, 1080)}
SLOTS = 10  # frames in flight between the two processes
STILL_LEVEL = 0.35  # change on a thumbnail, summed up since the last picture shown, that makes a picture new
STEP_LEVEL = 0.04  # change on a thumbnail from the frame just before that makes a picture new (a repeat differs by ~0)
GAP_S = 0.09  # frames further apart than this (a still screen that started moving) get no AI frame between them
LATE_S = 0.026  # a real frame older than this when the AI would start can't get its AI frame out in time: skipped
CUT_LEVEL = 35.0  # mean difference (0..255) on a thumbnail above which two frames are a scene cut: no AI frame


# ---------------------------------------------------------------------------------------------------- sources
class DemoSource:
    """A camera panning over the sample screenshots, a few seconds each (the switch is a scene cut)."""

    def __init__(self, w: int, h: int, fps: float = 30.0, seconds: float = 6.0) -> None:
        from PIL import Image
        self.w, self.h, self.fps, self.n = w, h, fps, int(fps * seconds)
        bw, bh = int(w * 1.35), int(h * 1.2)
        self.scenes = []
        for p in sorted((ROOT / "framegen" / "samples").glob("*.jpg")):
            img = Image.open(p).convert("RGB").resize((bw, bh), Image.LANCZOS)
            self.scenes.append(np.asarray(img))

    def __iter__(self):
        k = 0
        while True:
            img = self.scenes[(k // self.n) % len(self.scenes)]
            u = (k % self.n) / self.n
            x = int(round((img.shape[1] - self.w) * (0.5 - 0.5 * math.cos(2 * math.pi * u))))
            y = int(round((img.shape[0] - self.h) * (0.5 - 0.5 * math.cos(4 * math.pi * u))))
            yield np.ascontiguousarray(img[y:y + self.h, x:x + self.w])
            k += 1


class FileSource:
    """A video file, decoded by FFmpeg (PyAV), looped. Above 1080p it's scaled down to 1080p."""

    def __init__(self, path: str) -> None:
        import av
        self.path = path
        with av.open(path) as c:
            s = c.streams.video[0]
            self.fps = float(s.average_rate or 30)
            w, h = s.codec_context.width, s.codec_context.height
        if h > 1080:
            w, h = int(round(w * 1080 / h / 2)) * 2, 1080
        self.w, self.h = w, h

    def __iter__(self):
        import av
        while True:
            with av.open(self.path) as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                for f in c.decode(s):
                    yield f.to_ndarray(format="rgb24", width=self.w, height=self.h)


class StillFilter:
    """Says whether a decoded frame is a new picture or a repeat of the one before it.

    A frame is new when it differs from the frame received just before it (STEP_LEVEL: a repeat of a 30 fps game's
    picture differs by about nothing, a slow pan by a little), or when the changes since the last picture passed add up
    to STILL_LEVEL (slower than that is a still screen with something tiny moving). Comparing only with the last
    picture passed, as this did before, held a slow pan back until its changes added up: a few fps instead of 30."""

    def __init__(self) -> None:
        self.held: np.ndarray | None = None  # thumbnail of the last picture passed
        self.last: np.ndarray | None = None  # thumbnail of the last frame received
        self.step = 0.0  # the last frame's difference from the frame received just before it (for the statistics)

    def is_new(self, rgb: np.ndarray) -> bool:
        thumb = rgb[::16, ::16].astype(np.int16)
        last, self.last = self.last, thumb
        if self.held is None:
            self.held = thumb
            return True
        self.step = float(np.abs(thumb - last).mean())
        if self.step >= STEP_LEVEL or np.abs(thumb - self.held).mean() >= STILL_LEVEL:
            self.held = thumb
            return True
        return False


class TapSource:
    """Live frames from the frame tap (chiaki_tap/, docs/frame-tap.md): chiaki-ng streaming a PS5, or the fake PS5
    (scripts/frametap_fake.py). The stream is 60 Hz with every frame of a 30 fps game sent twice; ChiakiSource decodes
    it (software, 1080p HEVC takes about 2 ms) and says which frames are new from their compressed size.
    Yields (RGB frame, arrival time on time.monotonic(), damaged) for each *new* frame only."""

    def __init__(self, name: str | None, wait_s: float = 3600.0) -> None:
        sys.path.insert(0, str(ROOT))
        from framegen.chiaki_tap import NAME, ChiakiSource
        self.src = ChiakiSource(decoder="sw", wait_s=wait_s, tap_name=name or NAME)
        self.src.pix_fmt = "rgb24"  # decoded straight to RGB: no per-frame channel flip in this process
        self.w, self.h = self.src.w, self.src.h
        self.fps = 30.0  # the game's rate; the stream itself is 60 Hz
        self.still = 0  # frames skipped as the same picture as the one before
        self.recv = self.size_dup = self.passed = 0  # per second, for snapshot(): stream frames, repeats by size, passed on
        self.steps_pass: list = []  # the difference from the frame before, of the frames passed on / of the ones skipped
        self.steps_skip: list = []

    def snapshot(self) -> dict:
        """What the filters did since the last call (the worker sends it to the window once a second)."""
        q = lambda v, f: round(sorted(v)[int(len(v) * f)], 2) if v else None
        out = {"recv": self.recv, "size_dup": self.size_dup, "still": len(self.steps_skip), "passed": self.passed,
               "pass": [q(self.steps_pass, f) for f in (0.1, 0.5, 0.9)], "skip": [q(self.steps_skip, f) for f in (0.5, 0.9)]}
        self.recv = self.size_dup = self.passed = 0
        self.steps_pass, self.steps_skip = [], []
        return out

    def __iter__(self):
        from framegen.io import Tick
        filt = StillFilter()
        for tick in self.src:
            if not isinstance(tick, Tick):  # a Waiting: the stream stalled for a moment
                continue
            self.recv += 1
            if tick.new_hint is False:  # a repeat of the frame before
                self.size_dup += 1
                continue
            rgb = tick.frame
            # On a near-still screen (a menu) every compressed frame is tiny, so the size says "new" for nearly all of
            # them and the AI would have twice the frames it can do. A picture that hasn't changed is a repeat, and
            # there is nothing to interpolate between two equal pictures anyway.
            if not filt.is_new(rgb):
                self.still += 1
                self.steps_skip.append(filt.step)
                continue
            self.passed += 1
            self.steps_pass.append(filt.step)
            yield rgb, tick.t_arrival, tick.damaged

    def close(self) -> None:
        self.src.close()


def is_cut(a: np.ndarray, b: np.ndarray) -> bool:
    ta, tb = a[::16, ::16].astype(np.int16), b[::16, ::16].astype(np.int16)
    return float(np.abs(ta - tb).mean()) > CUT_LEVEL


def is_stale(base: float | None, newest_real: float, pts: float) -> bool:
    """An extrapolated frame (drawn after the real frame `base`, to be shown at `pts`) goes back in time if a real frame
    newer than its base is due at or before it: showing it would undo that real frame."""
    return base is not None and base < newest_real <= pts


def drop_stale(pending: list, pts: float) -> tuple[list, list]:
    """A real frame at `pts` just arrived: (kept, dropped) of the pending (slot, due, base, pts) entries. Dropped are
    the extrapolated ones drawn after an older real frame that fall at or after `pts`."""
    keep, drop = [], []
    for e in pending:
        (drop if e[2] is not None and e[2] < pts <= e[3] else keep).append(e)
    return keep, drop


# ---------------------------------------------------------------------------------------------------- worker
def worker(args, free_q, ready_q, ai_on, stop) -> None:
    """Reads the source in real time, draws the AI frames, and hands every frame over with its time."""
    from multiprocessing import shared_memory
    shm = live_src = None
    skipped = 0
    try:
        from PIL import Image
        live = args.tap is not None
        if live:  # the model loads before waiting for the stream: a stream's codec header comes only once, at its start
            (w, h), fps = RES[args.res], 30.0
        else:
            src = FileSource(args.source) if args.source else DemoSource(*RES[args.res])
            w, h, fps = src.w, src.h, src.fps
        n = args.mult - 1
        scale = {"full": 1.0, "half": 0.5}.get(args.motion) or (0.5 if h >= 1000 or n >= 2 else 1.0)
        rife = None
        if n > 0 and args.test_blend:  # for testing without a Mac: a plain cross-fade
            class _Blend:
                def interpolate(self, a, b, t):
                    if args.mode == "extrap":
                        return np.asarray(b)
                    return (np.asarray(a) * (1 - t) + np.asarray(b) * t).astype(np.uint8)
            rife = _Blend()
        elif n > 0:
            from rife_coreml import CoreMLRife
            rife = CoreMLRife(w, h, scale, "extrap" if args.mode == "extrap" else args.model)
            warm = Image.fromarray(np.zeros((h, w, 3), np.uint8))
            for _ in range(3):
                rife.interpolate(warm, warm, 0.5)

        if live:
            print("[smoovplay] ready, waiting for the stream", flush=True)  # smoovplay_mac.sh starts chiaki-ng on this
            src = live_src = TapSource(args.tap or None)
            if (src.w, src.h) != (w, h):
                raise RuntimeError(f"the stream is {src.w}x{src.h}, but SmoovPlay started for {w}x{h}: use --res")

        size = w * h * 3
        shm = shared_memory.SharedMemory(create=True, size=size * SLOTS)
        frames = [np.ndarray((h, w, 3), np.uint8, buffer=shm.buf, offset=i * size) for i in range(SLOTS)]
        for i in range(SLOTS):
            free_q.put(i)
        t0 = time.monotonic() + (0.0 if live else 0.2)
        ready_q.put(("info", shm.name, w, h, fps, n, scale, t0))

        def slot() -> int | None:
            while not stop.is_set():
                try:
                    return free_q.get(timeout=0.2)
                except queue.Empty:
                    pass
            return None

        prev = prev_img = None
        prev_pts = 0.0
        skipped = 0
        gaps: deque = deque(maxlen=8)  # recent gaps between real frames: where an extrapolated frame falls in time
        t_snap = time.monotonic()
        for k, item in enumerate(src):
            if live and time.monotonic() - t_snap >= 1.0:
                t_snap = time.monotonic()
                ready_q.put(("tapstat", live_src.snapshot()))
            if live:  # a real frame exists from the moment it arrived
                frame, t_arr, damaged = item
                pts = t_arr - t0
            else:  # a file: frame k exists only from its own time
                frame, damaged, pts = item, False, k / fps
                wait = t0 + pts - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            if stop.is_set():
                break
            t_body = time.perf_counter()
            img = Image.fromarray(frame)
            behind = live and (time.monotonic() - (t0 + pts) > LATE_S or pts - prev_pts > GAP_S)
            if behind:
                skipped += 1
            # The real frame goes out first: it's due at its own time and needn't wait for the AI frame, which falls
            # before it (halfway from the frame before) and so is the one that sets how short the delay can be.
            s = slot()
            if s is None:
                return
            frames[s][:] = frame
            ready_q.put(("frame", s, pts, False, (time.perf_counter() - t_body) * 1000, None))
            if prev is not None and 0 < pts - prev_pts <= GAP_S:
                gaps.append(pts - prev_pts)
            if rife is not None and prev is not None and ai_on.value and not damaged and not behind and not is_cut(prev, frame):
                # extrap: the AI frame is a fraction t of ONE game frame after this one, in the game's time (the flow
                # is the motion over one game frame), put where the game's pace says (the typical gap, not this
                # arrival's jitter). The window drops it if a newer real frame turns out to come before it.
                gap = sorted(gaps)[len(gaps) // 2] if gaps else 1.0 / fps
                # A pair that spans k game frames (one was lost, or never passed the still filter) has k frames of motion
                # between it: the model's t is a fraction of that, so t / k. k is rounded, so arrival jitter can't move it.
                k_span = min(4, max(1, round((pts - prev_pts) / gap)))
                for j in range(1, n + 1):
                    t = j / (n + 1)
                    s = slot()
                    if s is None:
                        return
                    a = time.perf_counter()
                    frames[s][:] = rife.interpolate(prev_img, img, t / k_span if args.mode == "extrap" else t)
                    ms = (time.perf_counter() - a) * 1000
                    if args.mode == "extrap":
                        ready_q.put(("frame", s, pts + gap * t, True, ms, pts))
                    else:
                        ready_q.put(("frame", s, prev_pts + (pts - prev_pts) * t, True, ms, None))
            prev, prev_img, prev_pts = frame, img, pts
    except Exception:  # noqa: BLE001 - report it to the window process, which prints it
        ready_q.put(("error", traceback.format_exc()))
    finally:
        if live_src is not None:
            print(f"[smoovplay] same-picture frames skipped: {live_src.still}, AI frames skipped for being late: {skipped}")
            live_src.close()  # prints how many frames the reader dropped or skipped
        stop.wait(5)  # the window process must detach before the memory goes
        if shm is not None:
            shm.close()
            shm.unlink()


# ---------------------------------------------------------------------------------------------------- window
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="a video file (default: the demo)")
    ap.add_argument("--res", default="1080p", choices=list(RES), help="demo resolution")
    ap.add_argument("--mult", type=int, default=2, choices=[1, 2, 3], help="2 = 30 -> 60 fps, 3 = 30 -> 90 fps")
    ap.add_argument("--motion", default="auto", choices=["auto", "full", "half"])
    ap.add_argument("--tap", nargs="?", const="", default=None, metavar="NAME",
                    help="live from the frame tap (default name Local\\chiaki-frametap) instead of a video")
    ap.add_argument("--model", default="lean3", choices=["base", "fold", "lean", "lean2", "lean2g", "lean3"],
                    help="Core ML graph: base = as first converted (21.8 ms at 1080p), lean = no full-size round trip "
                         "(17.5 ms, 66 dB from base), lean2 = also warps at block sizes (14.2 ms, 46-48 dB from base), "
                         "lean3 = lean2 with the output offset corrected (14.3 ms; the AI frame no longer sits ~0.17 px "
                         "off the real frames, which showed as a tremble on still pictures)")
    ap.add_argument("--test-blend", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode", default="interp", choices=["interp", "extrap"],
                    help="interp: the AI frame is drawn halfway between two real frames, so it needs the later one and "
                         "the picture is shown ~60 ms late. extrap: it is drawn after the latest real frame from the "
                         "motion between the last two: ~30 ms less delay, a little less exact (see "
                         "scripts/extrapolation_probe.py)")
    ap.add_argument("--vsync", action="store_true",
                    help="wait for the screen's vsync on every loop, as before (the loop then runs at only ~100 a second "
                         "on macOS, however fast the screen is, and drops frames that fall in the same loop)")
    ap.add_argument("--delay", type=float, default=None,
                    help="ms from a frame's arrival to the screen (default: 80 for a video; with --tap it adapts, "
                         "as low as the frames allow)")
    a = ap.parse_args()
    if a.source:
        a.source = str(Path(a.source).expanduser().resolve())

    import glfw
    from multiprocessing import shared_memory
    from OpenGL import GL

    ctx = mp.get_context("spawn")
    free_q, ready_q = ctx.Queue(), ctx.Queue()
    ai_on, stop = ctx.Value("b", 1), ctx.Event()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    proc = ctx.Process(target=worker, args=(a, free_q, ready_q, ai_on, stop), daemon=True)
    proc.start()

    print("Starting (the first time it builds the Core ML model: about a minute)...", flush=True)
    while True:
        try:
            msg = ready_q.get(timeout=1)
        except queue.Empty:
            if not proc.is_alive():
                sys.exit("The worker stopped before starting.")
            continue
        if msg[0] == "error":
            sys.exit(msg[1])
        if msg[0] == "info":
            break
    _, shm_name, w, h, fps, n, scale, t0 = msg
    shm = shared_memory.SharedMemory(name=shm_name)
    size = w * h * 3
    frames = [np.ndarray((h, w, 3), np.uint8, buffer=shm.buf, offset=i * size) for i in range(SLOTS)]
    out_fps = fps * (n + 1)
    how = "no AI" if n == 0 else f"{n} AI frame(s) per real frame, motion {'full' if scale == 1 else 'half'}"
    print(f"Source {w}x{h} at {fps:g} fps -> {out_fps:g} fps ({how})")

    if not glfw.init():
        sys.exit("Couldn't start the window system (glfw).")
    mon = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(mon)
    hz = mode.refresh_rate
    print(f"Screen: {mode.size.width}x{mode.size.height} at {hz} Hz")
    if 0 < hz < out_fps - 1:
        print(f"Note: this screen shows {hz} frames a second at most, fewer than {out_fps:g}.")
    k = min(1.0, 0.8 * mode.size.width / w, 0.8 * mode.size.height / h)
    win = glfw.create_window(int(w * k), int(h * k), "SmoovPlay Mac prototype", None, None)
    glfw.make_context_current(win)
    # With vsync on, macOS blocks swap_buffers for 8-11 ms (measured: ~95 loops a second on a 240 Hz screen) and the
    # frames due within one loop were dropped but the last. Off, a frame goes to the compositor the moment it is due and
    # the loop sleeps in between; a window has no tearing on macOS either way.
    glfw.swap_interval(1 if a.vsync else 0)

    GL.glEnable(GL.GL_TEXTURE_2D)
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    for p in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
        GL.glTexParameteri(GL.GL_TEXTURE_2D, p, GL.GL_LINEAR)
    for p in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
        GL.glTexParameteri(GL.GL_TEXTURE_2D, p, GL.GL_CLAMP_TO_EDGE)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)

    state = {"full": False, "saved": None}

    def on_key(window, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE and n > 0:
            ai_on.value = 0 if ai_on.value else 1
            print("AI frames", "ON" if ai_on.value else "OFF", flush=True)
        elif key == glfw.KEY_F:
            if state["full"]:
                x, y, ww, hh = state["saved"]
                glfw.set_window_monitor(window, None, x, y, ww, hh, 0)
            else:
                state["saved"] = (*glfw.get_window_pos(window), *glfw.get_window_size(window))
                glfw.set_window_monitor(window, mon, 0, 0, mode.size.width, mode.size.height, hz)
            state["full"] = not state["full"]

    glfw.set_key_callback(win, on_key)
    redraw = {"need": True}  # the window was exposed or resized: draw the picture again even if no new one is due
    glfw.set_window_refresh_callback(win, lambda _w: redraw.update(need=True))
    glfw.set_framebuffer_size_callback(win, lambda _w, _x, _y: redraw.update(need=True))
    if a.tap is not None:  # chiaki-ng's own window is open behind: ours stays in front of it
        glfw.set_window_attrib(win, glfw.FLOATING, True)

    auto = a.delay is None and a.tap is not None  # live: the lowest delay the frames can keep up with
    delay = (0.040 if a.mode == "extrap" else 0.060) if auto else (80.0 if a.delay is None else a.delay) / 1000.0
    needs: deque = deque()  # (when, how long after its own time a frame was ready) over the last few seconds
    t_adapt = time.monotonic()
    pending: list = []  # (slot, due time, base real frame's pts if extrapolated, pts), earliest first
    newest_real = -1.0  # pts of the newest real frame received
    stats = {"shown": 0, "drop": 0, "late": 0, "stale": 0, "ms": [], "body": [], "t": time.monotonic()}
    min_delay = 0.015 if a.mode == "extrap" else 0.030  # the floor of the adaptive delay
    have_frame = False
    try:
        while not glfw.window_should_close(win):
            while True:
                try:
                    msg = ready_q.get_nowait()
                except queue.Empty:
                    break
                if msg[0] == "error":
                    raise RuntimeError(msg[1])
                if msg[0] == "tapstat":
                    t = msg[1]
                    print(f"   tap: stream {t['recv']}/s, repeats by size {t['size_dup']}, repeats by picture {t['still']}, "
                          f"passed on {t['passed']} | step of passed p10/p50/p90 {t['pass']}, of skipped p50/p90 {t['skip']}",
                          flush=True)
                    continue
                _, s, pts, is_ai, ms, base = msg
                if is_stale(base, newest_real, pts):
                    free_q.put(s)  # a real frame newer than its base comes first: it would go back in time
                    stats["stale"] += 1
                    continue
                need = time.monotonic() - (t0 + pts)
                needs.append((time.monotonic(), need))
                due = t0 + pts + delay
                if time.monotonic() > due + 0.5 / out_fps:
                    stats["late"] += 1  # arrived after its time: the AI frame took too long
                if is_ai:
                    stats["ms"].append(ms)
                else:
                    stats["body"].append(ms)
                    newest_real = max(newest_real, pts)
                    keep, drop = drop_stale(pending, pts)
                    for e in drop:
                        free_q.put(e[0])
                        stats["stale"] += 1
                    pending[:] = keep
                pending.append((s, due, base, pts))
                pending.sort(key=lambda p: p[1])

            now = time.monotonic()
            if auto and now - t_adapt >= 0.25:
                # The delay has to cover the slowest frames (their AI frame is ready late); up at once, down slowly so
                # the spacing of the frames on screen doesn't visibly change.
                while needs and needs[0][0] < now - 4.0:
                    needs.popleft()
                if len(needs) >= 30:
                    ns = sorted(n for _, n in needs)
                    target = min(0.150, max(min_delay, ns[int(len(ns) * 0.98)] + 0.006))
                    delay = target if target > delay else max(target, delay - 0.004 * (now - t_adapt))
                t_adapt = now
            chosen = None
            while pending and pending[0][1] <= now:
                s = pending.pop(0)[0]
                if chosen is not None:
                    free_q.put(chosen)
                    stats["drop"] += 1
                chosen = s
            if chosen is not None:
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, frames[chosen])
                free_q.put(chosen)
                stats["shown"] += 1
                have_frame = True

            if chosen is not None or redraw["need"] or a.vsync:
                redraw["need"] = False
                fw, fh = glfw.get_framebuffer_size(win)
                GL.glViewport(0, 0, fw, fh)
                GL.glClearColor(0, 0, 0, 1)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                if have_frame and fw and fh:
                    sx = min(1.0, (fw / fh) / (w / h))
                    sy = min(1.0, (w / h) / (fw / fh))
                    GL.glBegin(GL.GL_QUADS)
                    for u, v, x, y in ((0, 1, -sx, -sy), (1, 1, sx, -sy), (1, 0, sx, sy), (0, 0, -sx, sy)):
                        GL.glTexCoord2f(u, v)
                        GL.glVertex2f(x, y)
                    GL.glEnd()
                glfw.swap_buffers(win)
            glfw.poll_events()
            if not a.vsync and chosen is None:
                # Nothing was due: sleep until the next frame is (or 2 ms, to take in the ones the worker sends).
                nxt = pending[0][1] - time.monotonic() if pending else 0.002
                time.sleep(max(0.0002, min(0.002, nxt - 0.0004)))

            el = now - stats["t"]
            if el >= 1.0:
                ms = sorted(stats["ms"])
                rife = f"AI frame {ms[len(ms) // 2]:.1f} ms" if ms else "AI off"
                bd = sorted(stats["body"])
                if bd:
                    rife += f" | per frame {bd[len(bd) // 2]:.1f}/{bd[-1]:.0f} ms"
                ready = sorted(nd for t_n, nd in needs if t_n > now - el)
                if ready:
                    rife += f" | ready {ready[len(ready) // 2] * 1000:.0f}/{ready[int(len(ready) * 0.9)] * 1000:.0f}/{ready[-1] * 1000:.0f} ms"
                line = (f"{'AI ON ' if ai_on.value and n else 'AI OFF'} | {stats['shown'] / el:5.1f} fps shown | "
                        f"{rife} | delay {delay * 1000:.0f} ms | late {stats['late']} | dropped {stats['drop']}"
                        + (f" | stale {stats['stale']}" if a.mode == "extrap" else ""))
                glfw.set_window_title(win, f"SmoovPlay Mac prototype | {line}")
                print(line, flush=True)
                stats.update(shown=0, drop=0, late=0, stale=0, ms=[], body=[], t=now)
    finally:
        stop.set()
        for e in pending:
            free_q.put(e[0])
        del frames
        shm.close()
        glfw.terminate()
        proc.join(3)


if __name__ == "__main__":
    main()
