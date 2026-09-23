"""Explicit whole-model MLIR lowering and inspection, without capture or deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlin lower", description="Lower supplied MLIR through the shared whole-model pipeline; never run it."
    )
    parser.add_argument("input", type=Path, help="existing linalg-on-tensors MLIR file")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory (must not already exist)")
    parser.add_argument(
        "--target",
        action="append",
        choices=("host", "riscv"),
        default=[],
        help="also build host shared library or RISC-V object; repeat for both (default: LLVM IR only)",
    )
    parser.add_argument("--textual", action="store_true", help="use existing text-only preprocessing instead of xDSL")
    parser.add_argument("--feature", action="append", help="explicit registered lowering feature; repeat to combine")
    parser.add_argument(
        "--ir-audit",
        nargs="?",
        const="exact",
        default=False,
        choices=("exact", "compact", "both"),
        help="retain exact stages (may be large), compact inspection-only views, or both",
    )
    parser.add_argument(
        "--audit-sidecar",
        type=Path,
        action="append",
        default=[],
        help="bind an existing weights/manifest file in the audit; repeat for each dependency",
    )
    args = parser.parse_args(argv)
    if args.audit_sidecar and not args.ir_audit:
        parser.error("--audit-sidecar requires --ir-audit")
    for source in (args.input, *args.audit_sidecar):
        if not source.is_file():
            parser.error(f"input is not an existing file: {source}")
    # Reserve a fresh destination before invoking the pipeline. Existing output,
    # including a dangling symlink, is never an implicit overwrite permission.
    output = args.out.absolute()
    if output.exists() or output.is_symlink():
        parser.error(f"output already exists: {output}; choose a fresh --out directory")

    try:
        from .lower import lower_model_file

        output.mkdir(parents=True, exist_ok=False)
        result = lower_model_file(
            args.input,
            output,
            targets=tuple(dict.fromkeys(args.target)),
            textual=args.textual,
            features=frozenset(args.feature) if args.feature else None,
            ir_audit=args.ir_audit,
            audit_sidecars=tuple(args.audit_sidecar),
        )
    except Exception as exc:
        print(f"merlin lower: {type(exc).__name__}: {exc}; output location: {output}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "workdir": str(result.workdir),
                "ll_path": str(result.ll_path),
                "host_so": str(result.host_so) if result.host_so is not None else None,
                "riscv_obj": str(result.riscv_obj) if result.riscv_obj is not None else None,
                "audit_index": str(result.audit_index) if result.audit_index is not None else None,
                "stats": result.stats,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
