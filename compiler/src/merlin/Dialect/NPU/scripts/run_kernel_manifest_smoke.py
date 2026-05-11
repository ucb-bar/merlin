#!/usr/bin/env python3
"""Run a stitched SaturnNPU kernel-manifest program in npu_model."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _int(value: str) -> int:
    return int(value, 0)


def _kernel_sequence(args: argparse.Namespace) -> list[str]:
    if args.all:
        from npu_model.configs.programs.smolvla_kernel_library import kernel_names

        return kernel_names()
    if args.sequence:
        return [item.strip() for item in args.sequence.split(",") if item.strip()]
    return args.kernel or []


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root / "third_party" / "npu_model"))

    import npu_model.configs.isa_definition  # noqa: F401
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.configs.programs.smolvla_kernel_library import stitch_kernels
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software import Program

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        action="append",
        help="Kernel name to run; repeat to concatenate",
    )
    parser.add_argument("--sequence", help="Comma-separated kernel sequence to run")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all copied manifest kernels in sorted order",
    )
    parser.add_argument(
        "--base-address",
        type=_int,
        default=0,
        help="DRAM base for invocation 0",
    )
    parser.add_argument(
        "--stride",
        type=_int,
        default=0x10000,
        help="DRAM stride per invocation",
    )
    parser.add_argument(
        "--connect-linear",
        action="store_true",
        help="Map each invocation's stores to the next invocation's leading loads",
    )
    parser.add_argument("--max-cycles", type=int, default=20000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    names = _kernel_sequence(args)
    if not names:
        parser.error("provide --kernel, --sequence, or --all")

    instructions = stitch_kernels(
        names,
        base_address=args.base_address,
        stride=args.stride,
        connect_linear=args.connect_linear,
    )
    program_cls = type(
        "StitchedKernelManifestProgram",
        (Program,),
        {"instructions": instructions, "memory_regions": []},
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as trace:
        trace_path = trace.name
    try:
        sim = Simulation(
            hardware_config=DefaultHardwareConfig(),
            logger_config=LoggerConfig(filename=trace_path),
            program=program_cls(),
            verbose=args.verbose,
        )
        sim.run(max_cycles=args.max_cycles)
    finally:
        Path(trace_path).unlink(missing_ok=True)

    print(f"stitched manifest smoke PASSED: {len(names)} kernel(s), " f"{len(instructions)} instruction(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
