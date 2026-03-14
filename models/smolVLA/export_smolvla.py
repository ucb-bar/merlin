#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UNDERSTANDING_PI0_ROOT = REPO_ROOT / "third_party" / "Understanding-PI0"
UNDERSTANDING_EXPORT_SCRIPT = UNDERSTANDING_PI0_ROOT / "scripts" / "export_iree.py"
INT8_EXPORT_SCRIPT = Path(__file__).resolve().parent / "export_smolvla_int8.py"


def resolve_python(python_arg: str | None) -> Path:
    if python_arg:
        return Path(python_arg).expanduser().resolve()
    preferred = UNDERSTANDING_PI0_ROOT / ".venv" / "bin" / "python"
    if preferred.exists():
        return preferred
    return Path(sys.executable).resolve()


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("+ " + shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_common_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--model-id",
        args.model_id,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--batch-size",
        str(args.batch_size),
        "--image-h",
        str(args.image_h),
        "--image-w",
        str(args.image_w),
        "--prompt-len",
        str(args.prompt_len),
        "--mx-kernel-preference",
        args.mx_kernel_preference,
    ]
    if args.no_vision:
        common.append("--no-vision")
    if args.skip_patches:
        common.append("--skip-patches")
    if args.no_exportable_mx:
        common.append("--no-exportable-mx")
    return common


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Export SmolVLA MLIR artifacts by interfacing with third_party/Understanding-PI0. "
            "Produces fp32, int8, and mixed fp8/int8 variants."
        )
    )
    ap.add_argument(
        "--mode",
        choices=["all", "fp32", "int8", "fp8"],
        default="all",
        help="Which export(s) to run.",
    )
    ap.add_argument("--python", help="Python interpreter to run Understanding-PI0 exports.")
    ap.add_argument("--model-id", default="lerobot/smolvla_base")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--image-h", type=int, default=256)
    ap.add_argument("--image-w", type=int, default=256)
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--mx-kernel-preference", default="AUTO")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--skip-patches", action="store_true")
    ap.add_argument("--no-exportable-mx", action="store_true")
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="Output directory for generated MLIR files.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not UNDERSTANDING_EXPORT_SCRIPT.exists():
        raise FileNotFoundError(f"Expected script not found: {UNDERSTANDING_EXPORT_SCRIPT}")
    if not INT8_EXPORT_SCRIPT.exists():
        raise FileNotFoundError(f"Expected script not found: {INT8_EXPORT_SCRIPT}")

    py = resolve_python(args.python)
    if not py.exists():
        raise FileNotFoundError(f"Python interpreter not found: {py}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp32 = out_dir / "smolVLA.mlir"
    out_int8 = out_dir / "smolVLA.q.int8.mlir"
    out_fp8 = out_dir / "smolVLA.q.fp8.mlir"

    common = build_common_args(args)
    steps: list[tuple[str, list[str]]] = []
    if args.mode in ("all", "fp32"):
        steps.append(
            (
                "fp32",
                [str(py), str(UNDERSTANDING_EXPORT_SCRIPT), *common, "--no-quant", "--out", str(out_fp32)],
            )
        )
    if args.mode in ("all", "int8"):
        steps.append(("int8", [str(py), str(INT8_EXPORT_SCRIPT), *common, "--out", str(out_int8)]))
    if args.mode in ("all", "fp8"):
        steps.append(("fp8/int8-mixed", [str(py), str(UNDERSTANDING_EXPORT_SCRIPT), *common, "--out", str(out_fp8)]))

    print(f"[smolVLA-export] python={py}")
    print(f"[smolVLA-export] out_dir={out_dir}")
    for label, cmd in steps:
        print(f"[smolVLA-export] exporting {label} ...")
        run_cmd(cmd, dry_run=args.dry_run)

    print("[smolVLA-export] done")
    print(f"  - {out_fp32}")
    print(f"  - {out_int8}")
    print(f"  - {out_fp8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
