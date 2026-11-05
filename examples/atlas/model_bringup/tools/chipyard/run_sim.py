#!/usr/bin/env python3
"""Run a prebuilt Chipyard simulator with explicit inputs and bounded execution."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_tooling import sha256, write_json


def safe_path(value):
    """Reject restricted names before performing any filesystem admission."""
    path = Path(value)
    if any(any(word in part.casefold() for word in ("hammer", "vlsi")) for part in path.parts):
        raise ValueError("restricted path component")
    return checked(path)


def positive(value):
    number = int(value)
    if not 0 < number <= (1 << 63) - 1:
        raise argparse.ArgumentTypeError("positive bounded integer required")
    return number


def signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass  # The simulator can exit between the wait timeout and signal.


def run_simulator(simulator, elf, out, *, dramsim_ini=None, max_cycles=2000000000,
                  wall_seconds=43200, verbose=False, extra_args=(), env=None):
    simulator, elf, out = (safe_path(p) for p in (simulator, elf, out))
    for value in (max_cycles, wall_seconds):
        positive(value)
    if not simulator.is_file() or not os.access(simulator, os.X_OK) or not elf.is_file():
        raise ValueError("executable simulator and existing ELF required")
    if out.exists():
        raise ValueError("fresh run output directory required")
    command = [str(simulator), "+permissive"]
    if dramsim_ini is not None:
        dramsim_ini = safe_path(dramsim_ini)
        if not dramsim_ini.is_dir():
            raise ValueError("DRAMSim configuration directory required")
        command += ["+dramsim", "+dramsim_ini_dir=" + str(dramsim_ini)]
    command += [f"+max-cycles={max_cycles}", "+ntb_random_seed=1", "+loadmem=" + str(elf)]
    if verbose:
        command.append("+verbose")
    # Extra simulator flags cannot replace the runner's bound or loaded program.
    for arg in extra_args:
        if (not arg.startswith("+") or arg.startswith(("+max-cycles", "+loadmem", "+permissive", "+dramsim_ini_dir"))):
            raise ValueError("extra arguments must be non-reserved simulator plusargs")
    command += list(extra_args) + ["+permissive-off", str(elf)]
    out.mkdir(parents=True)
    child_env = dict(os.environ if env is None else env, TMPDIR=str(out))
    started = time.monotonic()
    status = {"command": command, "elf": str(elf), "simulator": str(simulator),
              "elf_sha256": sha256(elf),
              "max_cycles": max_cycles, "wall_limit_seconds": wall_seconds,
              "timed_out": False, "returncode": 127}
    with (out / "sim.log").open("w") as log:
        try:
            with subprocess.Popen(command, cwd=out, env=child_env, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                try:
                    status["returncode"] = process.wait(timeout=wall_seconds)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    status["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
                    signal_group(process, signal.SIGINT)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        signal_group(process, signal.SIGKILL)
                        process.wait()
                    status["process_returncode"] = process.returncode
                    status["returncode"] = 124 if status["timed_out"] else 130
        except OSError as error:
            status["error"] = str(error)
            log.write(str(error) + "\n")
    status["wall_seconds"] = time.monotonic() - started
    (out / "exit-status.txt").write_text(str(status["returncode"]) + "\n")
    write_json(out / "run.json", status)
    return status


def add_simulator_arguments(parser):
    parser.add_argument("--simulator", type=Path, default=os.environ.get("ATLAS_SOC_SIMULATOR"),
                        help="prebuilt executable (or ATLAS_SOC_SIMULATOR)")
    parser.add_argument("--dramsim-ini", type=Path, default=os.environ.get("ATLAS_DRAMSIM_INI"),
                        help="enable DRAMSim with this configuration directory (or ATLAS_DRAMSIM_INI)")
    parser.add_argument("--max-cycles", type=positive, default=2000000000)
    parser.add_argument("--wall-seconds", type=positive, default=43200)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sim-arg", action="append", default=[], help="additional simulator plusarg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    add_simulator_arguments(parser)
    args = parser.parse_args()
    if args.simulator is None:
        parser.error("--simulator or ATLAS_SOC_SIMULATOR is required")
    status = run_simulator(args.simulator, args.elf, args.out, dramsim_ini=args.dramsim_ini,
                           max_cycles=args.max_cycles, wall_seconds=args.wall_seconds,
                           verbose=args.verbose, extra_args=args.sim_arg)
    print(json.dumps(status, sort_keys=True))
    return status["returncode"] if status["returncode"] >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
