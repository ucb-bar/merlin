#!/usr/bin/env python3
"""Emit native npu_model ISA from the SaturnNPU kernel manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _int(value: str) -> int:
    return int(value, 0)


def _kernel_sequence(args: argparse.Namespace) -> list[str]:
    if args.all:
        from npu_model.configs.programs._stitch import manifest_kernel_names

        return manifest_kernel_names()
    if args.sequence:
        return [item.strip() for item in args.sequence.split(",") if item.strip()]
    return args.kernel or []


def main() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party" / "npu_model"))

    import npu_model.configs.isa_definition  # noqa: F401
    from npu_model.configs.programs._stitch import (
        instruction_to_text,
        manifest_kernel_names,
        stitch_kernels,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        action="append",
        help="Kernel name to emit; repeat to concatenate",
    )
    parser.add_argument("--sequence", help="Comma-separated kernel sequence to emit")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Emit all manifest kernels in sorted order",
    )
    parser.add_argument("--list", action="store_true", help="List available kernels and exit")
    parser.add_argument("--output", type=Path, help="Output path; defaults to stdout")
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
    args = parser.parse_args()

    if args.list:
        for name in manifest_kernel_names():
            print(name)
        return 0

    names = _kernel_sequence(args)
    if not names:
        parser.error("provide --kernel, --sequence, --all, or --list")

    instructions = stitch_kernels(
        names,
        base_address=args.base_address,
        stride=args.stride,
        connect_linear=args.connect_linear,
    )
    lines = [f"# kernels: {', '.join(names)}"]
    lines.extend(instruction_to_text(instruction) for instruction in instructions)
    text = "\n".join(lines) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
