#!/usr/bin/env python3
"""Run every registered kernel-golden fixture through the npu_model simulator.

For each ``KernelGolden_<kernel>`` Program class in
``third_party/npu_model/npu_model/configs/programs/kernel_golden_suite.py``,
this script:

  1. Instantiates the Program (inputs laid out in DRAM, instructions stitched).
  2. Runs ``Simulation`` for up to ``--max-cycles`` cycles.
  3. Reads back the fixture's expected output address, reshapes into the
     fixture tensor shape.
  4. Compares against the fixture's torch reference at the fixture's
     declared tolerance (rtol, atol); ``rtol=atol=1e-2`` by default.

Exits 1 if ANY kernel fails allclose, 0 otherwise. Prints one row per kernel
with PASS/FAIL + max diff + cycles.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "third_party" / "npu_model"))

    import npu_model  # noqa: F401
    import torch
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.configs.programs import kernel_golden_suite as suite
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        action="append",
        help="Run only the named kernel (may be repeated). Defaults to all.",
    )
    parser.add_argument("--max-cycles", type=int, default=50_000)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print simulator progress for each kernel.",
    )
    args = parser.parse_args()

    selected = suite.__all__
    if args.kernel:
        wanted = {f"KernelGolden_{k}" for k in args.kernel}
        selected = [name for name in suite.__all__ if name in wanted]
        missing = wanted - set(selected)
        if missing:
            print(
                f"ERROR: unknown kernel(s): {sorted(missing)}",
                file=sys.stderr,
            )
            return 2

    header = f"{'kernel':<22}  {'result':<6}  {'max_diff':>12}  {'cycles':>8}"
    print(header)
    print("-" * len(header))

    failures: list[str] = []
    xfails: list[tuple[str, str]] = []
    passes = 0
    for cls_name in selected:
        cls = getattr(suite, cls_name)
        prog = cls()
        sim = Simulation(
            hardware_config=DefaultHardwareConfig(),
            logger_config=LoggerConfig(filename="/tmp/kernel_golden_trace.json"),
            program=prog,
        )
        if args.verbose:
            sim.run(max_cycles=args.max_cycles)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.run(max_cycles=args.max_cycles)

        base, golden = prog.golden_result
        size = golden.numel() * golden.element_size()
        mem = sim.core.arch_state.read_dram(base, size).view(golden.dtype).reshape(golden.shape).clone()
        rtol, atol = getattr(cls, "kernel_tolerance", (1e-2, 1e-2))
        ok = torch.allclose(mem.float(), golden.float(), rtol=rtol, atol=atol)
        max_diff = (mem.float() - golden.float()).abs().max().item()
        kernel_name = cls_name[len("KernelGolden_") :]
        xfail_reason = getattr(cls, "xfail_reason", "") or ""
        cycles = sim.cycle_count
        if xfail_reason:
            status = "XFAIL" if not ok else "XPASS"
            if not ok:
                xfails.append((kernel_name, xfail_reason))
            else:
                # XPASS means the broken fixture unexpectedly passes — worth
                # surfacing so we can drop the xfail_reason.
                failures.append(f"{kernel_name} (XPASS)")
        else:
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures.append(kernel_name)
            else:
                passes += 1
        print(f"{kernel_name:<22}  {status:<6}  {max_diff:>12.6f}  {cycles:>8d}")

    print("-" * len(header))
    if xfails:
        print(f"{len(xfails)} expected failure(s):")
        for name, reason in xfails:
            print(f"  {name}: {reason}")
    if failures:
        print(f"{len(failures)} unexpected failure(s): {', '.join(failures)}")
        return 1
    print(f"all {passes}/{len(selected)} non-xfail kernel(s) PASSED " f"({len(xfails)} xfail)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
