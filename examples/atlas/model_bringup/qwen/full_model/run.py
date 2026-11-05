"""Reproduce the real-Qwen inventory and model bench using the kernel-library coverage workflow.

python3 examples/atlas/model_bringup/qwen/full_model/run.py --setup
python3 examples/atlas/model_bringup/qwen/full_model/run.py
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source_roots import MERLIN_ROOT as ROOT, bootstrap_compiler, observed_revision
HERE = Path(__file__).resolve().parent
KERNELS = "ce465827a977ba8451b585f71910726e7f12b885"
HISTORICAL_KERNELS = "80c5f27d4bcded9c1ca7618a17ffcdbae2be7a39"
MODEL = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"


def run(command, **kwargs):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def verify_checkout(destination, revision):
    observed = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    modified = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=no"]
    )
    if observed != revision or modified:
        raise SystemExit(f"{destination} must be clean at {revision}")


def checkout(destination, remote, revision):
    if not destination.exists():
        run(["git", "init", "--quiet", destination])
        run(["git", "-C", destination, "remote", "add", "origin", remote])
        run(["git", "-C", destination, "fetch", "--depth", "1", "origin", revision])
        run(["git", "-C", destination, "checkout", "--detach", "FETCH_HEAD"])
    verify_checkout(destination, revision)


def setup(args):
    if not all(shutil.which(name) for name in ("uv", "git")):
        raise SystemExit("Setup requires uv and git")
    args.deps.mkdir(parents=True, exist_ok=True)
    checkout(
        args.deps / "atlas-kernels",
        "https://github.com/ucb-bar/atlas-kernels.git",
        KERNELS,
    )
    checkout(
        args.deps / "npu-model",
        "https://github.com/ucb-ee194-tapeout/npu_model.git",
        MODEL,
    )
    for name, version, torch in (
        ("capture", "3.12", "2.10.0"),
        ("model", "3.14", "2.11.0"),
    ):
        env = args.deps / name
        if not env.exists():
            run(["uv", "venv", "--python", version, env])
        python = env / "bin/python"
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                f"torch=={torch}",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ]
        )
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "numpy",
                "safetensors",
                "pyyaml",
                "tabulate",
            ]
        )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            args.deps / "capture/bin/python",
            "transformers==4.44.0",
            "accelerate",
            "torch-mlir==20261001",
            "xdsl==0.65.0",
        ]
    )
    run(
        [
            args.deps / "capture/bin/python",
            HERE / "download_checkpoint.py",
            "--out",
            args.checkpoint,
        ],
        env=dict(os.environ, HF_HUB_DISABLE_XET="1"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--deps", type=Path, default=ROOT / "out/qwen-full-deps")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "out/qwen-checkpoint")
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "out/qwen-full-model")
    parser.add_argument("--prefill", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--job-timeout", type=int, default=1800)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-real-capture", action="store_true")
    parser.add_argument("--capture-python", type=Path)
    parser.add_argument("--model-python", type=Path)
    parser.add_argument("--library-root", type=Path)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--atlas-bin", type=Path)
    parser.add_argument("--atlas-mlir-root", type=Path)
    args = parser.parse_args()
    args.deps, args.checkpoint, args.out = (
        p.resolve() for p in (args.deps, args.checkpoint, args.out)
    )
    if args.setup:
        setup(args)
        return
    compiler_root = bootstrap_compiler(args.atlas_mlir_root)
    from atlas_mlir.toolchain import bin_dir
    args.atlas_bin = (args.atlas_bin or bin_dir()).resolve()
    capture = (args.capture_python or args.deps / "capture/bin/python").absolute()
    model = (args.model_python or args.deps / "model/bin/python").absolute()
    library = Path(args.library_root or os.environ.get("ATLAS_KERNELS_ROOT") or args.deps / "atlas-kernels").resolve()
    model_root = Path(args.model_root or os.environ.get("NPU_MODEL_ROOT") or args.deps / "npu-model").resolve()
    if not (args.library_root or os.environ.get("ATLAS_KERNELS_ROOT")):
        verify_checkout(library, KERNELS)
    if not (args.model_root or os.environ.get("NPU_MODEL_ROOT")):
        verify_checkout(model_root, MODEL)
    if args.prefill < 1 or args.decode_steps < 0 or args.workers < 1:
        raise SystemExit(
            "Prefill/workers must be positive and decode steps nonnegative"
        )
    if args.out.exists():
        raise SystemExit("Choose a fresh output directory with -o")
    for path in (
        capture,
        model,
        library / "coverage/export_ops.py",
        model_root / "npu_model/simulation.py",
        args.atlas_bin / "atlas-opt",
        args.atlas_bin / "atlas-emit",
    ):
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}; run --setup and build atlas-opt/atlas-emit first"
            )
    args.out.mkdir(parents=True)
    env = dict(
        os.environ,
        QWEN_CHECKPOINT=str(args.checkpoint),
        QWEN_PREFILL=str(args.prefill),
        QWEN_DECODE_STEPS=str(args.decode_steps),
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    run(
        [
            capture,
            library / "coverage/export_ops.py",
            "qwen",
            "--loader",
            HERE / "loader.py",
            "--out",
            args.out / "export",
            "--fixture",
            args.out / "qwen.json",
        ],
        env=env,
    )
    run(
        [
            capture,
            HERE / "inventory.py",
            "--library",
            library,
            "--export",
            args.out / "export",
            "--checkpoint",
            args.checkpoint,
            "--prefill",
            args.prefill,
            "--decode-steps",
            args.decode_steps,
        ],
        env=env,
    )
    run(
        [
            capture,
            library / "coverage/build_table.py",
            "--inventory",
            f"qwen={args.out / 'qwen.json'}",
            "--mapping",
            HERE / "mapping.yaml",
            "--allow-unmapped",
            "--out",
            args.out / "coverage",
        ],
        env=env,
    )
    run(
        [
            model,
            library / "coverage/bench_model.py",
            "--coverage",
            args.out / "coverage/coverage.json",
            "--model-root",
            model_root,
            "--atlas-mlir-bin",
            args.atlas_bin.resolve(),
            "--min-tier",
            "verified",
            "--workers",
            args.workers,
            "--job-timeout",
            args.job_timeout,
            "--out",
            args.out / "bench",
            *(["--plan-only"] if args.plan_only else []),
        ],
        env=env,
    )
    if not args.plan_only and not args.skip_real_capture:
        run(
            [
                capture,
                HERE / "capture_boundaries.py",
                "--out",
                args.out / "real-capture",
            ],
            env=env,
        )
        run(
            [
                model,
                HERE / "qualify_projections.py",
                "--library",
                library,
                "--model-root",
                model_root,
                "--atlas-bin",
                args.atlas_bin.resolve(),
                "--capture",
                args.out / "real-capture",
                "--out",
                args.out / "qualification",
            ],
            env=env,
        )
    record = dict(
        kernels_revision=observed_revision(library),
        kernels_root=str(library),
        historical_kernels_revision=HISTORICAL_KERNELS,
        model_revision=observed_revision(model_root),
        model_root=str(model_root),
        historical_model_revision=MODEL,
        compiler_revision=observed_revision(Path(os.environ.get("ATLAS_MLIR_ROOT", compiler_root))),
        compiler_support_root=str(compiler_root),
        mapping_sha256=hashlib.sha256((HERE / "mapping.yaml").read_bytes()).hexdigest(),
        compiler_sha256={
            name: hashlib.sha256((args.atlas_bin / name).read_bytes()).hexdigest()
            for name in ("atlas-opt", "atlas-emit")
        },
        compiler_route="hand-written kernel generators; Atlas compiler verification/encoding gate",
        model_bench=str(args.out / "bench/MODEL-BENCH.md"),
    )
    (args.out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Model bench:", record["model_bench"])


if __name__ == "__main__":
    main()
