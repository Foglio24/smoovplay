"""Long-session recorder for the Mac prototype: one CSV row per second while SmoovPlay plays.

Each row has what mac/play.py prints every second (fps shown, AI frame ms, adaptive display delay, late/dropped,
stream and game rate from the tap), the memory footprint of play.py (main and worker) and chiaki-ng, the SoC die
temperatures (IOHID sensors, no root needed) and, when powermetrics is piped in, CPU/GPU clocks, GPU busy share,
power and thermal pressure. play.py is not touched: its log is followed like `tail -f`.

Started by run_session.sh, which pipes `sudo powermetrics -f plist` into this script:

  bash run_session.sh --check   # 8 seconds, no game needed: checks powermetrics, sensors and the log
  bash run_session.sh           # waits for SmoovPlay, records until it closes (Q), then writes a summary

Output: ~/Library/Logs/SmoovPlay/sessions/<date-time>/ with session.csv, summary.txt, powermetrics.plist (raw) and
copies of play.log and chiaki.log (the app overwrites those at each launch).

"delay" is play.py's adaptive display delay (how long a frame waits before it's shown), not the end-to-end latency.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import os
import plistlib
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

LOGS = Path.home() / "Library" / "Logs" / "SmoovPlay"
PLAY_PATTERN = r"python.* mac/play\.py"  # the interpreter, not a shell whose command line names it
CHIAKI_PATTERN = "chiaki.app/Contents/MacOS/chiaki"
WARMUP_S = 60  # left out of the "first minutes" comparison: model load, delay settling

COLUMNS = [
    "t_s", "clock", "play_running", "ai_on", "fps_shown", "ai_ms", "per_frame_p50_ms", "per_frame_max_ms",
    "ready_p50_ms", "ready_p90_ms", "ready_max_ms", "delay_ms", "late", "dropped", "stats_lines",
    "stream_fps", "repeats_size", "repeats_picture", "passed", "game_fps", "tap_ai",
    "mem_play_main_mb", "mem_play_children_mb", "mem_chiaki_mb", "temp_max_c", "temp_mean_c",
    "cpu_clusters_mhz", "cpu_p_max_mhz", "gpu_mhz", "gpu_busy_pct", "cpu_w", "gpu_w", "ane_w", "package_w",
    "thermal_pressure", "pm_age_s",
]

STATS_RE = re.compile(r"^AI (ON |OFF) \|\s*([\d.]+) fps shown \| (.*)$")
TAP_RE = re.compile(r"tap: stream (\d+)/s, repeats by size (\d+), repeats by picture (\d+), passed on (\d+).*"
                    r"game ~(\d+) fps, AI (on|off)")


# ---------------------------------------------------------------- play.log


class LogFollower(threading.Thread):
    """Follows play.log like `tail -F`: starts at the end, starts over when the file is replaced or truncated (the
    launcher opens it with `>` at every launch). Keeps the latest per-second stats line and tap line."""

    def __init__(self, path: Path) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.lock = threading.Lock()
        self.stats: dict | None = None
        self.tap: dict | None = None
        self.n_stats = 0  # stats lines since the last take()
        self.seen = False

    def run(self) -> None:
        f = None
        ino = None
        pos = 0
        buf = b""
        first = True
        while True:
            try:
                st = self.path.stat()
                if f is None or st.st_ino != ino or st.st_size < pos:
                    if f:
                        f.close()
                    f = open(self.path, "rb")
                    ino = st.st_ino
                    pos = st.st_size if first else 0  # old content of a log from before we started: skip it
                    f.seek(pos)
                    buf = b""
                first = False
                chunk = f.read()
                if chunk:
                    pos += len(chunk)
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for ln in lines:
                        self._line(ln.decode("utf-8", "replace"))
            except FileNotFoundError:
                first = False
            except OSError:
                pass
            time.sleep(0.1)

    def _line(self, line: str) -> None:
        m = STATS_RE.match(line)
        if m:
            d = {"ai_on": int(m.group(1).strip() == "ON"), "fps_shown": float(m.group(2))}
            rest = m.group(3)
            for key, pat in (("ai_ms", r"AI frame ([\d.]+) ms"), ("delay_ms", r"delay (\d+) ms"),
                             ("late", r"late (\d+)"), ("dropped", r"dropped (\d+)")):
                mm = re.search(pat, rest)
                if mm:
                    d[key] = float(mm.group(1))
            mm = re.search(r"per frame ([\d.]+)/(\d+) ms", rest)
            if mm:
                d["per_frame_p50_ms"], d["per_frame_max_ms"] = float(mm.group(1)), float(mm.group(2))
            mm = re.search(r"ready (\d+)/(\d+)/(\d+) ms", rest)
            if mm:
                d["ready_p50_ms"], d["ready_p90_ms"], d["ready_max_ms"] = (float(x) for x in mm.groups())
            with self.lock:
                self.stats = d
                self.n_stats += 1
                self.seen = True
            return
        m = TAP_RE.search(line)
        if m:
            with self.lock:
                self.tap = {"stream_fps": int(m.group(1)), "repeats_size": int(m.group(2)),
                            "repeats_picture": int(m.group(3)), "passed": int(m.group(4)),
                            "game_fps": int(m.group(5)), "tap_ai": int(m.group(6) == "on")}

    def take(self) -> tuple[dict | None, dict | None, int]:
        """The latest lines, and how many stats lines came since the last call (0: play.py printed nothing)."""
        with self.lock:
            s, t, n = self.stats, self.tap, self.n_stats
            self.stats, self.tap, self.n_stats = None, None, 0
        return s, t, n


# ---------------------------------------------------------------- powermetrics


class PowerMetrics(threading.Thread):
    """Reads `powermetrics -f plist` from stdin: one plist per sample. Saves every raw sample, keeps the latest
    parsed one."""

    def __init__(self, stream, raw_path: Path) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.raw = open(raw_path, "wb")
        self.lock = threading.Lock()
        self.latest: dict | None = None
        self.at = 0.0
        self.count = 0
        self.errors: list[str] = []
        self.keys: list[str] = []  # top-level and processor keys of the first sample, for --check

    def run(self) -> None:
        buf = b""
        end = b"</plist>"
        while True:
            chunk = self.stream.read1(65536) if hasattr(self.stream, "read1") else self.stream.read(4096)
            if not chunk:
                break
            buf += chunk
            while end in buf:
                i = buf.index(end) + len(end)
                doc, buf = buf[:i], buf[i:]
                doc = doc.strip(b"\x00\r\n\t ")
                self.raw.write(doc + b"\n\x00\n")
                self.raw.flush()
                try:
                    sample = plistlib.loads(doc)
                except Exception as e:  # noqa: BLE001
                    self.errors.append(f"{type(e).__name__}: {e}")
                    continue
                parsed = self._parse(sample)
                with self.lock:
                    if not self.keys:
                        self.keys = sorted(sample) + [f"processor.{k}" for k in sorted(sample.get("processor", {}))] \
                            + [f"gpu.{k}" for k in sorted(sample.get("gpu", {}))]
                    self.latest, self.at = parsed, time.monotonic()
                    self.count += 1

    @staticmethod
    def _parse(s: dict) -> dict:
        proc = s.get("processor", {}) or {}
        gpu = s.get("gpu", {}) or {}
        el = (s.get("elapsed_ns") or 1e9) / 1e9

        def watts(power_key: str, energy_key: str, *dicts) -> float | None:
            for d in dicts:
                if power_key in d:
                    return round(float(d[power_key]) / 1000.0, 2)  # mW
                if energy_key in d:
                    return round(float(d[energy_key]) / 1000.0 / el, 2)  # mJ over the interval
            return None

        def mhz(v) -> int:  # CPU clusters report Hz, the GPU MHz (macOS 27)
            v = float(v)
            return round(v / 1e6 if v > 1e5 else v)

        clusters = []
        p_max = None
        for c in proc.get("clusters", []) or []:
            f_mhz = mhz(c.get("freq_hz", 0))
            name = str(c.get("name", "?")).replace("-Cluster", "").replace("_CLUSTER", "")
            clusters.append(f"{name}:{f_mhz}")
            if not name.upper().startswith("E"):  # M5 Pro: P0, P1 and S clusters, no E
                p_max = f_mhz if p_max is None else max(p_max, f_mhz)
        busy = None
        if "idle_ratio" in gpu:
            busy = round((1.0 - float(gpu["idle_ratio"])) * 100.0, 1)
        return {
            "cpu_clusters_mhz": " ".join(clusters),
            "cpu_p_max_mhz": p_max,
            "gpu_mhz": mhz(gpu["freq_hz"]) if "freq_hz" in gpu else None,
            "gpu_busy_pct": busy,
            "cpu_w": watts("cpu_power", "cpu_energy", proc),
            "gpu_w": watts("gpu_power", "gpu_energy", proc, gpu),
            "ane_w": watts("ane_power", "ane_energy", proc),
            "package_w": watts("combined_power", "combined_energy", proc),
            "thermal_pressure": s.get("thermal_pressure"),
        }

    def get(self) -> tuple[dict | None, float | None]:
        with self.lock:
            return self.latest, (time.monotonic() - self.at) if self.latest else None


# ---------------------------------------------------------------- temperatures (IOHID, no root)


class Temps:
    """SoC die temperatures from the "PMU tdie*" HID sensors (the same ones `stats`-style apps read)."""

    def __init__(self) -> None:
        V = ctypes.c_void_p
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        io = ctypes.CDLL(ctypes.util.find_library("IOKit"))
        cf.CFStringCreateWithCString.restype = V
        cf.CFStringCreateWithCString.argtypes = [V, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFNumberCreate.restype = V
        cf.CFNumberCreate.argtypes = [V, ctypes.c_int, V]
        cf.CFDictionaryCreate.restype = V
        cf.CFDictionaryCreate.argtypes = [V, V, V, ctypes.c_long, V, V]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetCount.argtypes = [V]
        cf.CFArrayGetValueAtIndex.restype = V
        cf.CFArrayGetValueAtIndex.argtypes = [V, ctypes.c_long]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [V, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFRelease.argtypes = [V]
        io.IOHIDEventSystemClientCreate.restype = V
        io.IOHIDEventSystemClientCreate.argtypes = [V]
        io.IOHIDEventSystemClientSetMatching.argtypes = [V, V]
        io.IOHIDEventSystemClientCopyServices.restype = V
        io.IOHIDEventSystemClientCopyServices.argtypes = [V]
        io.IOHIDServiceClientCopyProperty.restype = V
        io.IOHIDServiceClientCopyProperty.argtypes = [V, V]
        io.IOHIDServiceClientCopyEvent.restype = V
        io.IOHIDServiceClientCopyEvent.argtypes = [V, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]
        io.IOHIDEventGetFloatValue.restype = ctypes.c_double
        io.IOHIDEventGetFloatValue.argtypes = [V, ctypes.c_int32]
        self.cf, self.io = cf, io
        utf8 = 0x08000100

        def cfstr(x: str):
            return cf.CFStringCreateWithCString(None, x.encode(), utf8)

        def cfnum(n: int):
            v = ctypes.c_int32(n)
            return cf.CFNumberCreate(None, 3, ctypes.byref(v))  # kCFNumberSInt32Type

        keys = (V * 2)(cfstr("PrimaryUsagePage"), cfstr("PrimaryUsage"))
        vals = (V * 2)(cfnum(0xFF00), cfnum(5))  # Apple vendor page, temperature sensor
        kcb = V.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")
        vcb = V.in_dll(cf, "kCFTypeDictionaryValueCallBacks")
        match = cf.CFDictionaryCreate(None, keys, vals, 2, ctypes.addressof(kcb), ctypes.addressof(vcb))
        self.client = io.IOHIDEventSystemClientCreate(None)
        io.IOHIDEventSystemClientSetMatching(self.client, match)
        services = io.IOHIDEventSystemClientCopyServices(self.client)
        prod = cfstr("Product")
        buf = ctypes.create_string_buffer(256)
        self.sensors = []  # the die sensors' service handles (kept alive by `services`)
        for i in range(cf.CFArrayGetCount(services) if services else 0):
            sv = cf.CFArrayGetValueAtIndex(services, i)
            name = io.IOHIDServiceClientCopyProperty(sv, prod)
            if name and cf.CFStringGetCString(name, buf, 256, utf8) and buf.value.startswith(b"PMU tdie"):
                self.sensors.append(sv)
            if name:
                cf.CFRelease(name)
        self._services = services

    def read(self) -> tuple[float | None, float | None]:
        vals = []
        for sv in self.sensors:
            ev = self.io.IOHIDServiceClientCopyEvent(sv, 15, 0, 0)  # kIOHIDEventTypeTemperature
            if ev:
                v = self.io.IOHIDEventGetFloatValue(ev, 15 << 16)
                self.cf.CFRelease(ev)
                if 0.0 < v < 150.0:
                    vals.append(v)
        if not vals:
            return None, None
        return round(max(vals), 1), round(sum(vals) / len(vals), 1)


# ---------------------------------------------------------------- processes and memory


def pgrep(*args: str) -> list[int]:
    r = subprocess.run(["pgrep", *args], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def footprints(pids: list[int]) -> dict[int, float]:
    """Physical footprint in MB per pid (what Activity Monitor calls Memory; includes GPU/IOSurface memory)."""
    if not pids:
        return {}
    cmd = ["footprint"]
    for p in pids:
        cmd += ["-p", str(p)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    out = {}
    for m in re.finditer(r"\[(\d+)\]:.*?Footprint: ([\d.]+) (KB|MB|GB)", r.stdout):
        mult = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}[m.group(3)]
        out[int(m.group(1))] = round(float(m.group(2)) * mult, 1)
    return out


def processes() -> tuple[int | None, list[int], list[int]]:
    """(play.py main, its children, chiaki-ng). The spawned worker's command line doesn't name play.py."""
    play = pgrep("-f", PLAY_PATTERN)
    play_set = set(play)
    main = None
    for p in play:  # the one whose parent isn't another play.py
        parent = subprocess.run(["ps", "-o", "ppid=", "-p", str(p)], capture_output=True, text=True).stdout.strip()
        if not parent or int(parent) not in play_set:
            main = p
            break
    children = pgrep("-P", str(main)) if main else []
    return main, children, pgrep("-f", CHIAKI_PATTERN)


# ---------------------------------------------------------------- summary


def summary(rows: list[dict]) -> str:
    def nums(rs, k):
        return [r[k] for r in rs if isinstance(r.get(k), (int, float))]

    def mean(rs, k):
        v = nums(rs, k)
        return f"{statistics.fmean(v):.1f}" if v else "-"

    def pct(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(len(v) * q))] if v else None

    play = [r for r in rows if r["play_running"]]
    if not play:
        return "SmoovPlay never ran during the recording.\n"
    t0 = play[0]["t_s"]
    dur = play[-1]["t_s"] - t0
    ai = [r for r in play if r.get("ai_on") == 1 and r["t_s"] - t0 >= WARMUP_S]
    out = [f"Session: {dur / 60:.1f} min with SmoovPlay running ({len(play)} rows), "
           f"{len([r for r in play if r.get('ai_on') == 1])} s with AI on, "
           f"{len([r for r in play if r.get('ai_on') == 0])} s with AI off."]
    if ai:
        fps = nums(ai, "fps_shown")
        aims = nums(ai, "ai_ms")
        out.append(f"AI on, after the first {WARMUP_S} s: fps shown mean {statistics.fmean(fps):.1f}, "
                   f"p5 {pct(fps, 0.05):.1f}, p95 {pct(fps, 0.95):.1f}; "
                   f"AI frame median {statistics.median(aims) if aims else float('nan'):.1f} ms, "
                   f"p95 {pct(aims, 0.95) or float('nan'):.1f} ms; display delay mean {mean(ai, 'delay_ms')} ms.")
        out.append(f"Late AI frames: {int(sum(nums(play, 'late')))} total; dropped: {int(sum(nums(play, 'dropped')))} "
                   f"total; seconds without a stats line: {len([r for r in play if r['stats_lines'] == 0])}.")
        gf = nums(ai, "game_fps")
        if gf:
            out.append(f"Game rate seen by the tap: median {statistics.median(gf):.0f} fps.")
        a = [r for r in ai if r["t_s"] - t0 < WARMUP_S + 120]
        b = [r for r in ai if r["t_s"] >= play[-1]["t_s"] - 120]
        if a and b and dur > 6 * 60:
            out.append("\nFirst 2 minutes (after warm-up) vs last 2 minutes:")
            for k, label in (("fps_shown", "fps shown"), ("ai_ms", "AI frame ms"), ("delay_ms", "display delay ms"),
                             ("gpu_mhz", "GPU MHz"), ("gpu_busy_pct", "GPU busy %"), ("cpu_p_max_mhz", "CPU P MHz"),
                             ("temp_max_c", "die temp max C"), ("gpu_w", "GPU W"), ("package_w", "package W")):
                out.append(f"  {label:<18} {mean(a, k):>8} -> {mean(b, k):>8}")
    tp = [r.get("thermal_pressure") for r in play if r.get("thermal_pressure")]
    if tp:
        counts = {x: tp.count(x) for x in dict.fromkeys(tp)}
        out.append(f"\nThermal pressure (seconds): {counts}")
    tm = nums(play, "temp_max_c")
    if tm:
        out.append(f"Die temperature: max {max(tm):.1f} C, mean of the per-second max {statistics.fmean(tm):.1f} C.")
    # after the warm-up, and not the last seconds (play.py shutting down frees memory before it exits)
    mem = [(r["t_s"], (r.get("mem_play_main_mb") or 0) + (r.get("mem_play_children_mb") or 0))
           for r in play if r["t_s"] - t0 >= WARMUP_S and r["t_s"] <= play[-1]["t_s"] - 5
           and r.get("mem_play_main_mb")]
    if len(mem) > 30:
        ys = [m[1] for m in mem]
        first = statistics.median([y for t, y in mem if t <= mem[0][0] + 120])
        last = statistics.median([y for t, y in mem if t >= mem[-1][0] - 120])
        mins = (mem[-1][0] - mem[0][0]) / 60
        out.append(f"play.py memory (main + worker), median of the first vs last 2 minutes: {first:.0f} MB -> "
                   f"{last:.0f} MB over {mins:.1f} min (max {max(ys):.0f} MB).")
    ch = nums(play, "mem_chiaki_mb")
    if ch:
        out.append(f"chiaki-ng memory: {ch[0]:.0f} MB -> {ch[-1]:.0f} MB, max {max(ch):.0f} MB.")
    out.append("\n'display delay' is play.py's adaptive delay, not the end-to-end latency.")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=LOGS / "play.log", help="play.py's log to follow")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: a new one under sessions/)")
    ap.add_argument("--check", action="store_true", help="8 seconds, then print what each source gives")
    ap.add_argument("--no-powermetrics", action="store_true", help="don't read powermetrics from stdin")
    ap.add_argument("--max-minutes", type=float, default=30.0, help="stop after this long (a backstop)")
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = a.out or LOGS / "sessions" / (f"check-{stamp}" if a.check else stamp)
    out.mkdir(parents=True, exist_ok=True)

    pm = None
    if not a.no_powermetrics and not sys.stdin.isatty():
        pm = PowerMetrics(sys.stdin.buffer, out / "powermetrics.plist")
        pm.start()
    log = LogFollower(a.log)
    log.start()
    try:
        temps = Temps()
    except Exception as e:  # noqa: BLE001
        print(f"(temperatures unavailable: {e})", flush=True)
        temps = None

    f = open(out / "session.csv", "w", newline="")
    w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    rows: list[dict] = []
    start = time.monotonic()
    seen_play = False
    gone_since = None
    last_wait_msg = 0.0
    print(f"Recording to {out}", flush=True)
    if not a.check:
        print("Now open SmoovPlay and play. The recording stops by itself when SmoovPlay closes (Q), "
              "or press Ctrl-C.", flush=True)
    tick = start
    try:
        while True:
            tick += 1.0
            time.sleep(max(0.0, tick - time.monotonic()))
            now = time.monotonic()
            main_pid, children, chiaki = processes()
            fp = footprints(([main_pid] if main_pid else []) + children + chiaki)
            stats, tap, n = log.take()
            row: dict = {"t_s": round(now - start, 1), "clock": datetime.now().strftime("%H:%M:%S"),
                         "play_running": int(main_pid is not None), "stats_lines": n}
            if stats:
                row.update(stats)
            if tap:
                row.update(tap)
            if main_pid:
                row["mem_play_main_mb"] = fp.get(main_pid)
                row["mem_play_children_mb"] = round(sum(fp.get(c, 0) for c in children), 1) if children else None
            if chiaki:
                row["mem_chiaki_mb"] = round(sum(fp.get(c, 0) for c in chiaki), 1)
            if temps:
                row["temp_max_c"], row["temp_mean_c"] = temps.read()
            if pm:
                s, age = pm.get()
                if s:
                    row.update(s)
                    row["pm_age_s"] = round(age, 1)
            w.writerow(row)
            f.flush()
            rows.append(row)

            if a.check:
                if now - start >= 8:
                    break
                continue
            if main_pid:
                if not seen_play:
                    print(f"[{row['clock']}] SmoovPlay is running, recording.", flush=True)
                seen_play, gone_since = True, None
                if int(row["t_s"]) % 60 == 0 and stats:
                    print(f"[{row['clock']}] {stats.get('fps_shown', 0):.0f} fps shown, "
                          f"AI {stats.get('ai_ms', 0):.1f} ms, delay {stats.get('delay_ms', 0):.0f} ms, "
                          f"die {row.get('temp_max_c')} C", flush=True)
            elif seen_play:
                gone_since = gone_since or now
                if now - gone_since >= 3:
                    print("SmoovPlay closed: stopping.", flush=True)
                    break
            elif now - last_wait_msg >= 15:
                print("  waiting for SmoovPlay...", flush=True)
                last_wait_msg = now
            if now - start >= a.max_minutes * 60:
                print(f"{a.max_minutes:g} minutes reached: stopping.", flush=True)
                break
    except KeyboardInterrupt:
        print("\nStopped (Ctrl-C).", flush=True)
    finally:
        f.close()

    if a.check:
        report = [f"Check, {len(rows)} seconds:"]
        if pm is None:
            report.append("  powermetrics: NOT piped in (run through run_session.sh)")
        else:
            s, age = pm.get()
            report.append(f"  powermetrics: {pm.count} samples, {len(pm.errors)} parse errors"
                          + (f" (first: {pm.errors[0]})" if pm.errors else ""))
            report.append(f"    keys: {', '.join(pm.keys) or '-'}")
            report.append(f"    parsed: {s}")
        last = rows[-1] if rows else {}
        report.append(f"  temperatures: {len(temps.sensors) if temps else 0} die sensors, "
                      f"max {last.get('temp_max_c')} C, mean {last.get('temp_mean_c')} C")
        report.append(f"  play.log: {a.log} {'found' if a.log.exists() else 'not there yet (fine before a session)'}; "
                      f"SmoovPlay running now: {'yes' if last.get('play_running') else 'no'}")
        text = "\n".join(report) + "\n"
    else:
        text = summary(rows)
        for name in ("play.log", "chiaki.log"):
            src = a.log if name == "play.log" else a.log.with_name("chiaki.log")
            if src.exists():
                shutil.copy2(src, out / name)
    (out / "summary.txt").write_text(text)
    print("\n" + text + f"\nSaved in {out}", flush=True)
    os._exit(0)  # the reader threads may be blocked on stdin/the log; powermetrics ends on the closed pipe


if __name__ == "__main__":
    main()
