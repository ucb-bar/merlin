#!/usr/bin/env python3
"""Read-only preflight for a FireSim run on this host.

Every check here exists because it once cost an afternoon: an error message that named neither
the setting nor the path it wanted, a queue daemon that had quietly been dead for two weeks, a
kernel module whose file name gained a `.zst`.  This script touches NOTHING — it reads config
files, `/proc/modules`, and the last heartbeat sample.  It never submits a job, never talks to
the FPGA, never starts the daemon.

    .venv/bin/python build_tools/firesim/preflight.py
    .venv/bin/python build_tools/firesim/preflight.py --json

Exit code is 0 when every check is OK or WARN, 1 when any check FAILs.  See
docs/guides/firesim.md for what to do about each finding.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))
from merlin.common.paths import env  # noqa: E402

OK, WARN, FAIL = "OK", "WARN", "FAIL"
# The queue daemon writes its pid file on start and refreshes a heartbeat row; a pid file older
# than this with no live process is the "dead for two weeks" case.
STALE_DAEMON_S = 24 * 3600


def _r(name: str, status: str, detail: str) -> dict:
    return {"check": name, "status": status, "detail": detail}


def check_chipyard() -> list[dict]:
    cy = env("MERLIN_CHIPYARD") or env("MERLIN_EXT_CHIPYARD")
    if not cy:
        return [_r("chipyard", FAIL, "MERLIN_CHIPYARD unset (set it in .env)")]
    deploy = Path(cy) / "sims" / "firesim" / "deploy"
    if not deploy.is_dir():
        return [_r("chipyard", FAIL, f"no sims/firesim/deploy under {cy}")]
    return [_r("chipyard", OK, str(deploy))]


def check_modelblaster() -> list[dict]:
    """run_on_firesim() reuses ModelBlaster's queue-safe runner.  When MERLIN_MODELBLASTER is
    unset the failure is a bare `ModuleNotFoundError: No module named 'modelblaster'` that names
    neither the setting nor the path — so check it up front."""
    mb = env("MERLIN_MODELBLASTER")
    if not mb:
        return [_r("modelblaster", FAIL, "MERLIN_MODELBLASTER unset -> run_on_firesim() will die "
                                         "with a bare ModuleNotFoundError")]
    # run_on_firesim() puts both `<mb>/src` and `<mb>` on sys.path and tries the packaged import
    # (`modelblaster.validation.firesim_runner`) before the flat one, so accept either layout.
    for runner in (Path(mb) / "src" / "modelblaster" / "validation" / "firesim_runner.py",
                   Path(mb) / "validation" / "firesim_runner.py"):
        if runner.is_file():
            return [_r("modelblaster", OK, str(runner))]
    return [_r("modelblaster", FAIL, f"no validation/firesim_runner.py under {mb}")]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def check_queue() -> list[dict]:
    """The single FPGA is shared: every run must go through the queue, and the queue only moves
    while its daemon is alive."""
    q = env("MERLIN_EXT_FIRESIM_QUEUE") or env("FIRESIM_QUEUE_ROOT")
    if not q:
        return [_r("queue", FAIL, "MERLIN_EXT_FIRESIM_QUEUE unset")]
    root = Path(q)
    if not (root / "bin" / "firesim-queue").is_file():
        return [_r("queue", FAIL, f"no bin/firesim-queue under {root}")]
    pidf = root / "daemon.pid"
    if not pidf.is_file():
        return [_r("queue", FAIL, f"daemon not running (no {pidf}); start it with "
                                  f"`{root}/bin/firesim-queue daemon`")]
    try:
        pid = int(pidf.read_text().strip())
    except ValueError:
        return [_r("queue", WARN, f"unparseable {pidf}")]
    age_h = (time.time() - pidf.stat().st_mtime) / 3600.0
    if not _pid_alive(pid):
        return [_r("queue", FAIL, f"daemon.pid={pid} is DEAD (pid file {age_h:.1f}h old). Check "
                                  f"`firesim-queue status` for PENDING jobs, then restart the "
                                  f"daemon — the pid file outlives the process.")]
    status = WARN if age_h > STALE_DAEMON_S / 3600.0 else OK
    return [_r("queue", status, f"daemon pid {pid} alive, started {age_h:.1f}h ago ({root})")]


def check_xdma() -> list[dict]:
    """`insmod: ERROR: could not load module poll_mode=1` at INFRASETUP means the module is not
    loaded: FireSim's helper looks for a literal `xdma.ko`, modern kernels ship `xdma.ko.zst`, the
    search comes back empty and `poll_mode=1` gets mistaken for the module path."""
    nodes = sorted(Path("/dev").glob("xdma0_*"))
    if nodes:
        return [_r("xdma", OK, f"xdma character devices present ({len(nodes)} under /dev)")]
    mods = Path("/proc/modules")
    try:
        listed = any(line.split(" ", 1)[0] == "xdma" for line in mods.read_text().splitlines())
    except OSError:
        return [_r("xdma", WARN, "/proc/modules unreadable and no /dev/xdma0_* nodes — cannot tell")]
    if listed:
        return [_r("xdma", WARN, "xdma module loaded but no /dev/xdma0_* nodes")]
    return [_r("xdma", FAIL, "xdma module NOT loaded -> INFRASETUP fails with "
                             "`insmod: ERROR: could not load module poll_mode=1`")]


def _yaml_scalar(path: Path, key: str) -> str | None:
    """Read one `key: value` scalar out of a config file without importing yaml (these configs
    are edited by several tools and may be mid-write; a line read is the cheap, safe thing)."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            return stripped[len(key) + 1:].strip()
    return None


def check_bitstream() -> list[dict]:
    cy = env("MERLIN_CHIPYARD") or env("MERLIN_EXT_CHIPYARD")
    if not cy:
        return []
    deploy = Path(cy) / "sims" / "firesim" / "deploy"
    hw = _yaml_scalar(deploy / "config_runtime.yaml", "default_hw_config")
    if not hw:
        return [_r("bitstream", WARN, "no default_hw_config in config_runtime.yaml")]
    hwdb = deploy / "config_hwdb.yaml"
    text = hwdb.read_text(encoding="utf-8", errors="replace") if hwdb.is_file() else ""
    if f"\n{hw}:" not in f"\n{text}":
        return [_r("bitstream", FAIL, f"default_hw_config={hw} has no entry in config_hwdb.yaml")]
    tar = None
    seen = False
    for line in text.splitlines():
        if line.strip().startswith(f"{hw}:"):
            seen = True
            continue
        if seen and line.strip().startswith("bitstream_tar:"):
            tar = line.split(":", 1)[1].strip()
            break
        if seen and line and not line[0].isspace():
            break
    detail = f"default_hw_config={hw}"
    if tar and tar.startswith("file://"):
        p = Path(tar[len("file://"):])
        if not p.is_file():
            return [_r("bitstream", FAIL, f"{detail}; bitstream_tar missing: {p}")]
        detail += f"; tar present ({p.stat().st_size / 1e6:.0f} MB)"
    return [_r("bitstream", OK, detail + "  [NOTE: config_runtime.yaml is SHARED — back it up "
                                         "before repointing default_hw_config]")]


def check_heartbeat() -> list[dict]:
    """heartbeat.csv is `target cycles, seconds since start`.  Its slope is the effective clock,
    and that number is how you tell a slow run from a hung one."""
    cy = env("MERLIN_CHIPYARD") or env("MERLIN_EXT_CHIPYARD")
    if not cy:
        return []
    simdir = _yaml_scalar(Path(cy) / "sims/firesim/deploy/config_runtime.yaml",
                          "default_simulation_dir")
    if not simdir:
        return [_r("heartbeat", WARN, "no default_simulation_dir in config_runtime.yaml")]
    hb = Path(simdir) / "sim_slot_0" / "heartbeat.csv"
    if not hb.is_file():
        return [_r("heartbeat", WARN, f"no {hb} (no run has executed in this slot yet)")]
    rows: list[tuple[int, int]] = []
    for line in hb.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            rows.append((int(parts[0].strip()), int(parts[1].strip())))
        except ValueError:
            continue
    if len(rows) < 2:
        return [_r("heartbeat", WARN, f"{hb}: not enough samples yet")]
    dc, ds = rows[-1][0] - rows[0][0], rows[-1][1] - rows[0][1]
    age_min = (time.time() - hb.stat().st_mtime) / 60.0
    if ds <= 0:
        return [_r("heartbeat", WARN, f"{hb}: degenerate time span")]
    mhz = dc / ds / 1e6
    return [_r("heartbeat", OK, f"last run advanced {mhz:.2f} MHz effective "
                                f"({rows[-1][0]:,} cycles in {rows[-1][1]}s, "
                                f"file {age_min:.0f} min old)")]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)

    results: list[dict] = []
    for fn in (check_chipyard, check_modelblaster, check_queue, check_xdma,
               check_bitstream, check_heartbeat):
        results.extend(fn())

    if a.json:
        print(json.dumps(results, indent=2))
    else:
        width = max(len(r["check"]) for r in results)
        for r in results:
            print(f"[{r['status']:4s}] {r['check']:{width}s}  {r['detail']}")
    return 1 if any(r["status"] == FAIL for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
