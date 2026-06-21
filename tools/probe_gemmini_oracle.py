#!/usr/bin/env python3
"""Probe the Gemmini execution oracles (Step 0 for the C0 RTL-certification path).

Not fancy: resolve toolchain/sim paths from env (with chipyard defaults), report which
oracle layers are present, and optionally run a known-good prebuilt bareMetalC binary on
spike-gemmini and/or the Verilator RTL sim to confirm output + exit.

Env (all optional; defaults target /scratch2/agustin/chipyard):
  MERLIN_CHIPYARD            chipyard root
  MERLIN_GEMMINI_SPIKE       spike binary (with gemmini extension available)
  MERLIN_GEMMINI_VERILATOR   prebuilt Verilator sim binary
  MERLIN_RISCV_GCC           riscv64-unknown-elf-gcc
  MERLIN_GEMMINI_HARNESS_DIR gemmini-rocc-tests root (includes + benchmarks/common)

Usage:
  python tools/probe_gemmini_oracle.py                 # status only
  python tools/probe_gemmini_oracle.py --run spike     # + run known-good on spike-gemmini
  python tools/probe_gemmini_oracle.py --run verilator # + run known-good on Verilator RTL
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CHIPYARD = "/scratch2/agustin/chipyard"
VERILATOR_CONFIG = "GemminiAndOPUShuttleConfig"
KNOWN_GOOD = {  # prebuilt bareMetalC binaries (relative to gemmini-rocc-tests/build)
    "spike": "bareMetalC/matmul-baremetal",
    "verilator": "bareMetalC/mvin_mvout_zeros-baremetal",  # small -> tractable on RTL
}


def chipyard() -> Path:
    return Path(os.environ.get("MERLIN_CHIPYARD", DEFAULT_CHIPYARD))


def spike_bin() -> Path:
    return Path(os.environ.get("MERLIN_GEMMINI_SPIKE",
                               chipyard() / ".conda-env/riscv-tools/bin/spike"))


def libgemmini() -> Path:
    return chipyard() / ".conda-env/riscv-tools/lib/libgemmini.so"


def verilator_sim() -> Path:
    return Path(os.environ.get(
        "MERLIN_GEMMINI_VERILATOR",
        chipyard() / "sims/verilator" / f"simulator-chipyard.harness-{VERILATOR_CONFIG}"))


def riscv_gcc() -> Path:
    return Path(os.environ.get("MERLIN_RISCV_GCC",
                               chipyard() / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"))


def rocc_tests() -> Path:
    return Path(os.environ.get("MERLIN_GEMMINI_HARNESS_DIR",
                               chipyard() / "generators/gemmini/software/gemmini-rocc-tests"))


def test_ld() -> Path:
    return rocc_tests() / "riscv-tests/benchmarks/common/test.ld"


def _ok(p: Path) -> str:
    return "OK  " if p.exists() else "MISS"


def status() -> dict:
    rows = {
        "chipyard": chipyard(),
        "spike": spike_bin(),
        "libgemmini.so": libgemmini(),
        "verilator_sim": verilator_sim(),
        "riscv_gcc": riscv_gcc(),
        "rocc_tests": rocc_tests(),
        "test.ld": test_ld(),
    }
    print("=== Gemmini oracle probe ===")
    for name, p in rows.items():
        print(f"  [{_ok(p)}] {name}: {p}")
    spike_ready = spike_bin().is_file()
    verilator_ready = verilator_sim().is_file()
    gcc_ready = riscv_gcc().is_file() and test_ld().is_file()
    print(f"  L1 spike-gemmini (bootstrap, derived_from_rtl=false): "
          f"{'available' if spike_ready else 'unavailable'}")
    print(f"  L2 Verilator RTL (certification, derived_from_rtl=true): "
          f"{'available' if verilator_ready else 'unavailable'}")
    print(f"  riscv gcc + test.ld (to compile generated C): "
          f"{'available' if gcc_ready else 'unavailable'}")
    return {"spike": spike_ready, "verilator": verilator_ready, "gcc": gcc_ready}


def run_known_good(which: str, timeout: int) -> int:
    elf = rocc_tests() / "build" / KNOWN_GOOD[which]
    if not elf.is_file():
        print(f"  known-good binary missing: {elf}")
        return 2
    if which == "spike":
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (str(libgemmini().parent) + ":"
                                  + env.get("LD_LIBRARY_PATH", ""))
        cmd = [str(spike_bin()), "--extension=gemmini", str(elf)]
    else:
        env = dict(os.environ)
        cmd = [str(verilator_sim()), str(elf)]
    print(f"\n=== running {which}: {' '.join(cmd)} ===")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return 3
    tail = "\n".join(proc.stdout.splitlines()[-15:])
    print(tail)
    print(f"  exit={proc.returncode}")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["spike", "verilator"], action="append", default=[])
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()
    st = status()
    rc = 0
    for which in args.run:
        if not st[which]:
            print(f"\n  skipping --run {which}: oracle unavailable")
            continue
        rc |= run_known_good(which, args.timeout)
    return rc


if __name__ == "__main__":
    sys.exit(main())
