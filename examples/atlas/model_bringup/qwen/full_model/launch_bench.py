"""Export the complete random-weight Qwen session and qualify library launch shapes.

The shared atlas-kernels exporter, coverage planner, generators, arithmetic
goldens, and simulator perform the work. This is not a connected device model.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from bench_admission import admit_bench

HERE = Path(__file__).resolve().parent


def run(command, environment):
    subprocess.run(list(map(str, command)), env=environment, check=True)


def annotate(library, output):
    spec = importlib.util.spec_from_file_location("qwen_exporter", library / "coverage/export_ops.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    records = json.loads((output / "export/nodes.json").read_text())
    step = -1
    for row in records:
        if row["op"] == "aten.embedding.default":
            step += 1
        row.update(step=max(step, 0), phase="prefill" if step <= 0 else "decode")
    if step != 2:
        raise RuntimeError(f"Expected two cached decodes, observed {step}")
    ordered = exporter.calls(records)
    counts = {phase: sum(call["phase"] == phase for call in ordered) for phase in ("prefill", "decode")}
    if counts != dict(prefill=197, decode=394):
        raise RuntimeError(f"Incomplete Qwen projection census: {counts}")
    lengths = [row["outputs"][0][0][1] for row in records if row["op"] == "aten.embedding.default"]
    if lengths != [32, 1, 1]:
        raise RuntimeError(f"Unexpected cached sequence geometry: {lengths}")
    attention = exporter.attention(records)
    norms = sum(row["op"] == "aten.rsqrt.default" for row in records)
    if len(attention) != 84 or norms != 171:
        raise RuntimeError(f"Incomplete attention/norm census: {len(attention)}/{norms}")
    document = json.loads((output / "export/inventory.json").read_text())
    document.update(ops=exporter.aggregate(records), calls=ordered, attention=attention)
    document["session"] = dict(prefill=32, decode_steps=2, projection_counts=counts,
                               real_checkpoint=False, tensor_source="seeded random FP32 full-model weights")
    document["sources"]["random_weights"] = json.loads((output / "metadata/loader-metadata.json").read_text())
    (output / "export/nodes.json").write_text(json.dumps(records) + "\n")
    (output / "export/inventory.json").write_text(json.dumps(document, indent=2) + "\n")
    (output / "inventory.json").write_text(json.dumps(document, indent=2) + "\n")
    return document["session"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--atlas-bin", type=Path, required=True)
    parser.add_argument("--model-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tmp", type=Path, help="subprocess scratch directory, default OUT/tmp")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--job-timeout", type=int, default=1800)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--projection-fusions", action="store_true",
                        help="also qualify opt-in exact QKV bias and o/down residual panels; no connected model claim")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    for name in ("library", "model_root", "atlas_bin", "out"):
        setattr(args, name, getattr(args, name).absolute())
    args.out.mkdir(parents=True, exist_ok=False)
    scratch = (args.tmp or args.out / "tmp").absolute()
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, QWEN_RANDOM_OUT=str(args.out / "metadata"), QWEN_PREFILL="32",
               QWEN_DECODE_STEPS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(scratch), TEMP=str(scratch), TMP=str(scratch))
    run([sys.executable, args.library / "coverage/export_ops.py", "qwen", "--loader",
         HERE / "random_loader.py", "--out", args.out / "export"], env)
    annotate(args.library, args.out)
    run([sys.executable, args.library / "coverage/build_table.py", "--inventory",
         f"qwen={args.out / 'inventory.json'}", "--mapping", HERE / "mapping.yaml",
         "--allow-unmapped", "--out", args.out / "coverage"], env)
    run([args.model_python, args.library / "coverage/bench_model.py", "--coverage",
         args.out / "coverage/coverage.json", "--model-root", args.model_root,
         "--atlas-mlir-bin", args.atlas_bin, "--min-tier", "verified", "--workers", args.workers,
         "--job-timeout", args.job_timeout, "--out", args.out / "bench",
         *(["--plan-only"] if args.plan_only else [])], env)
    launch_qualification = dict(status="PENDING", reason="plan-only")
    if not args.plan_only:
        launch_qualification = admit_bench(args.out / "bench", expected_variants=9)
        run([sys.executable, HERE / "capture_boundaries.py", "--loader", HERE / "random_loader.py",
             "--skip-linalg", "--out", args.out / "capture"], env)
        capture_record = args.out / "capture/capture.json"
        captured = json.loads(capture_record.read_text())
        captured["tensor_source"] = "seeded random FP32 full-model graph boundary tensors; no pretrained checkpoint"
        capture_record.write_text(json.dumps(captured, indent=2) + "\n")
        run([args.model_python, HERE / "qualify_projections.py", "--library", args.library,
             "--model-root", args.model_root, "--atlas-bin", args.atlas_bin,
             "--capture", args.out / "capture", "--out", args.out / "qualification"], env)
    if args.projection_fusions and not args.plan_only:
        run([args.model_python, args.library / "kernels/projection_gemm/qualify_qwen_fusions.py",
             "--model-root", args.model_root, "--atlas-bin", args.atlas_bin,
             "--out", args.out / "projection-fusions", "--timed"], env)
    record = dict(tensor_source="seeded random FP32 full-model weights; no pretrained checkpoint",
                  kernels_revision=subprocess.check_output(["git", "-C", str(args.library), "rev-parse", "HEAD"], text=True).strip(),
                  model_revision=subprocess.check_output(["git", "-C", str(args.model_root), "rev-parse", "HEAD"], text=True).strip(),
                  compiler_sha256={name: hashlib.sha256((args.atlas_bin / name).read_bytes()).hexdigest()
                                   for name in ("atlas-opt", "atlas-emit")},
                  mapping_sha256=hashlib.sha256((HERE / "mapping.yaml").read_bytes()).hexdigest(),
                  connected_device_execution="PENDING", plan_only=args.plan_only,
                  launch_qualification=launch_qualification)
    if args.projection_fusions:
        record["projection_fusions"] = dict(
            selection="exact QKV bias and o/down residual panel epilogues",
            qualification="PENDING" if args.plan_only else str(args.out / "projection-fusions/summary.json"),
            tensor_source="seeded synthetic panels, separate from model capture",
            connected_device_execution="PENDING", host_time_saved="unmeasured",
        )
    (args.out / "run.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
