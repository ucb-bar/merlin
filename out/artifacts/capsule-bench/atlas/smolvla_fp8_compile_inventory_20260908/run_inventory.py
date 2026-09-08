#!/usr/bin/env python3
"""Bounded compile-only inventory for the native-FP8 Atlas SmolVLA path.

This is deliberately not a simulator test.  It inventories every captured op, records
the target router's decision, identifies the actual rank-2 and batched contraction
kernels structurally, and invokes the generated Atlas backend once per unique shape.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import signal
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from xdsl.dialects.builtin import StringAttr, TensorType

from merlin.compile_cli import _mesh_tile_binding
from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.capsule_source import model_op_demands
from merlin.targetgen.corpus_spec import build_gemv_batched, build_matmul
from merlin.targetgen.routing import route_plan


REPO = Path(__file__).resolve().parents[5]
DEFAULT_CAPTURE = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
DEFAULT_PACKAGE = (
    REPO
    / "out/runs/atlas/capsule-bench/merlin_assisted/"
      "merlincirct_atlas_fresh_func_20260907/submission"
)
DEFAULT_GSIM_HEADER = Path("/scratch/agustin/tmp/gsim-atlas-core/AtlasCore.h")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root: Path) -> tuple[str, list[dict]]:
    rows = []
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()
                       and "__pycache__" not in p.parts and p.suffix != ".pyc"):
        rel = path.relative_to(root).as_posix()
        digest = sha256(path)
        size = path.stat().st_size
        rows.append({"path": rel, "bytes": size, "sha256": digest})
        h.update(rel.encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\n")
    return h.hexdigest(), rows


def sattr(op, key: str) -> str:
    value = (getattr(op, "attributes", {}) or {}).get(key)
    return value.data if isinstance(value, StringAttr) else ""


def shape(value) -> tuple[int, ...]:
    typ = value.type
    return tuple(int(x) for x in typ.get_shape()) if isinstance(typ, TensorType) else ()


def physical_layers(module) -> list[dict]:
    """Actual contraction kernels, not every support op inheriting a provenance tag."""
    rows: list[dict] = []
    for op in module.walk():
        prov_op = sattr(op, "prov.op")
        region = sattr(op, "prov.region_id")
        if op.name == "linalg.matmul":
            inputs = list(op.inputs)
            if len(inputs) < 2:
                continue
            lhs, rhs = shape(inputs[0]), shape(inputs[1])
            if len(lhs) != 2 or len(rhs) != 2 or lhs[1] != rhs[0]:
                continue
            rows.append({
                "kind": "matmul",
                "source_op": prov_op,
                "region_id": region,
                "fqn": sattr(op, "prov.fqn"),
                "M": lhs[0], "K": lhs[1], "N": rhs[1],
                "lhs_shape": list(lhs), "rhs_shape": list(rhs),
            })
            continue

        # model2MLIR represents attention BMMs as a generic with the canonical
        # [B,M,K] x [B,K,N] -> [B,M,N] shapes.  Constants/fills carrying the same
        # region tag are intentionally not kernels.
        if op.name != "linalg.generic" or prov_op != "batch_matmul":
            continue
        inputs = list(op.inputs)
        outputs = list(op.outputs)
        if len(inputs) != 2 or len(outputs) != 1:
            continue
        lhs, rhs, out = shape(inputs[0]), shape(inputs[1]), shape(outputs[0])
        if (len(lhs) != 3 or len(rhs) != 3 or len(out) != 3
                or lhs[0] != rhs[0] or lhs[0] != out[0]
                or lhs[1] != out[1] or lhs[2] != rhs[1] or rhs[2] != out[2]):
            continue
        rows.append({
            "kind": "matmul_batched",
            "source_op": prov_op,
            "region_id": region,
            "fqn": sattr(op, "prov.fqn"),
            "B": lhs[0], "M": lhs[1], "K": lhs[2], "N": rhs[2],
            "lhs_shape": list(lhs), "rhs_shape": list(rhs),
        })
    return rows


def kernel_key(row: dict) -> tuple:
    if row["kind"] == "matmul":
        return ("matmul", row["M"], row["K"], row["N"])
    return ("matmul_batched", row["B"], row["M"], row["K"], row["N"])


def kernel_id(key: tuple) -> str:
    return "_".join(str(x) for x in key)


def emit_interface(row: dict, binding) -> str:
    base = {
        "name": kernel_id(kernel_key(row)),
        "kind": "op",
        "source_role": "whole_model_layer_compile_inventory",
        "source_reference": "smolvla_fp32_consistent/model.mlir",
    }
    if row["kind"] == "matmul":
        base.update(M=row["M"], K=row["K"], N=row["N"])
        return build_matmul(base, binding)[1]
    base.update(B=row["B"], M=row["M"], H=row["K"], N=row["N"])
    return build_gemv_batched(base, binding)[1]


def run_one(tool: Path, interface: Path, kernel: Path, stderr: Path, timeout_s: float) -> dict:
    start = time.monotonic()
    command = [str(tool), "--emit-target-artifact", str(interface)]
    with kernel.open("wb") as out, stderr.open("wb") as err:
        proc = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
            status = "accepted" if rc == 0 else "refused"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc, status = 124, "timeout"
    elapsed = time.monotonic() - start
    words = 0
    if kernel.exists():
        with kernel.open("rt", encoding="utf-8", errors="replace") as stream:
            words = sum(line.lstrip().startswith(".word") for line in stream)
    original_size = kernel.stat().st_size if kernel.exists() else 0
    original_sha = sha256(kernel) if kernel.exists() else None
    compressed = kernel.with_suffix(kernel.suffix + ".gz")
    if kernel.exists():
        # Preserve every emitted program without leaving hundreds of MiB of repetitive
        # `.word` text in the durable artifact.  mtime=0 makes reruns byte-reproducible.
        with kernel.open("rb") as source, compressed.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as target:
                shutil.copyfileobj(source, target)
        kernel.unlink()
    return {
        "status": status,
        "returncode": rc,
        "compile_seconds": round(elapsed, 6),
        "instruction_words": words,
        "instruction_bytes": words * 4,
        "assembly_file_bytes": original_size,
        "assembly_sha256": original_sha,
        "assembly_gzip_bytes": compressed.stat().st_size if compressed.exists() else 0,
        "assembly_gzip_sha256": sha256(compressed) if compressed.exists() else None,
        "stderr": stderr.read_text(encoding="utf-8", errors="replace")[-4000:],
        "command": command,
    }


def route_rows(model_text: str) -> tuple[list[dict], dict]:
    demands = model_op_demands(model_text, "fp8_e4m3")
    plan = route_plan(demands, "atlas")
    placement = {}
    for bucket in ("mesh", "fallback", "scalar_rvv"):
        for result in plan[bucket]:
            placement[id(result)] = bucket
    rows = []
    for index, result in enumerate(plan["results"]):
        row = asdict(result.demand)
        row.update(index=index, bucket=placement[id(result)], unit=result.unit,
                   accumulator=result.acc, gap=result.gap)
        rows.append(row)
    summary = {
        "total": len(rows),
        "by_bucket": dict(Counter(row["bucket"] for row in rows)),
        "by_op": dict(Counter(row["op"] for row in rows)),
        "mesh_by_op": dict(Counter(row["op"] for row in rows if row["bucket"] == "mesh")),
    }
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    ap.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--per-shape-timeout", type=float, default=60.0)
    ap.add_argument("--total-timeout", type=float, default=600.0)
    ap.add_argument("--gsim-header", type=Path, default=DEFAULT_GSIM_HEADER)
    args = ap.parse_args()
    if not 0 < args.per_shape_timeout <= 60:
        ap.error("--per-shape-timeout must be in (0, 60]")
    if not 0 < args.total_timeout <= 600:
        ap.error("--total-timeout must be in (0, 600]")

    output = args.output.resolve()
    interfaces, kernels, errors = output / "interfaces", output / "kernels", output / "stderr"
    for path in (output, interfaces, kernels, errors):
        path.mkdir(parents=True, exist_ok=True)

    capture = args.capture.resolve()
    package = args.package.resolve()
    tool = package / "mlir_oot/atlas-opt"
    started = time.monotonic()
    text = capture.read_text(encoding="utf-8")
    module = parse_mlir_file(capture)
    demands, routing_summary = route_rows(text)
    layers = physical_layers(module)
    counts = Counter(kernel_key(row) for row in layers)
    representatives = {kernel_key(row): row for row in layers}
    binding = _mesh_tile_binding("atlas", "fp8_e4m3", "bf16")

    unique = []
    results = {}
    for key in sorted(representatives, key=lambda x: tuple(str(y) for y in x)):
        kid = kernel_id(key)
        row = representatives[key]
        iface = interfaces / f"{kid}.mlir"
        asm = kernels / f"{kid}.S"
        err = errors / f"{kid}.txt"
        iface.write_text(emit_interface(row, binding), encoding="utf-8")
        remaining = args.total_timeout - (time.monotonic() - started)
        if remaining <= 0:
            result = {
                "status": "not_run_total_timeout", "returncode": None,
                "compile_seconds": 0.0, "instruction_words": 0,
                "instruction_bytes": 0, "assembly_file_bytes": 0,
                "assembly_sha256": None, "stderr": "", "command": [],
            }
        else:
            result = run_one(tool, iface, asm, err, min(args.per_shape_timeout, remaining))
        result.update({
            "kernel_id": kid,
            "shape": list(key[1:]),
            "kind": key[0],
            "layer_occurrences": counts[key],
            "interface": str(iface.relative_to(output)),
            "interface_sha256": sha256(iface),
            "assembly_gzip": str(asm.with_suffix(asm.suffix + ".gz").relative_to(output))
            if asm.with_suffix(asm.suffix + ".gz").exists() else None,
            "stderr_file": str(err.relative_to(output)) if err.exists() else None,
        })
        unique.append(result)
        results[key] = result

    # This is a generated RTL fact, not a guessed limit.  If the exact GSIM header is
    # unavailable, fit remains unknown instead of silently assuming a capacity.
    imem = {"words": None, "derived_from": None, "sha256": None, "source_line": None}
    gsim_header = args.gsim_header.resolve()
    if gsim_header.is_file():
        for number, line in enumerate(gsim_header.read_text(encoding="utf-8").splitlines(), 1):
            marker = "uint32_t imem$mem["
            if marker not in line:
                continue
            body = line.split(marker, 1)[1].split("]", 1)[0]
            if body.isdigit():
                imem = {"words": int(body), "derived_from": str(gsim_header),
                        "sha256": sha256(gsim_header), "source_line": number}
                break
    for result in unique:
        result["instruction_memory_fit"] = (
            None if imem["words"] is None else result["instruction_words"] <= imem["words"]
        )

    for index, row in enumerate(layers):
        key = kernel_key(row)
        result = results[key]
        row.update(index=index, kernel_id=result["kernel_id"], compile_status=result["status"],
                   instruction_memory_fit=result["instruction_memory_fit"])

    package_hash, package_files = tree_digest(package)
    codegen_text = (package / "mlir_oot/codegen.py").read_text(encoding="utf-8")
    plateau_path = package.parent / "plateau.json"
    package_score = None
    if plateau_path.is_file():
        plateau = json.loads(plateau_path.read_text(encoding="utf-8"))
        package_score = {
            "passed": plateau.get("latest_passed"), "total": plateau.get("n_capsules"),
            "path": str(plateau_path), "sha256": sha256(plateau_path),
        }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
    ).stdout.strip()
    elapsed = time.monotonic() - started
    physical_by_kind = dict(Counter(row["kind"] for row in layers))
    unique_by_status = dict(Counter(row["status"] for row in unique))
    layers_by_status = dict(Counter(row["compile_status"] for row in layers))
    unique_imem_fit = dict(Counter(str(row["instruction_memory_fit"]).lower() for row in unique))
    layers_imem_fit = dict(Counter(str(row["instruction_memory_fit"]).lower() for row in layers))
    summary = {
        "schema": "atlas_smolvla_compile_inventory_v1",
        "claim": "compile-only; no simulator, numeric, or whole-image execution claim",
        "target": "atlas",
        "operand_dtype": "fp8_e4m3",
        "accumulator_dtype": "bf16",
        "capture": {"path": str(capture), "bytes": capture.stat().st_size, "sha256": sha256(capture)},
        "package": {"path": str(package), "tree_sha256": package_hash, "files": package_files},
        "package_capsule_score": package_score,
        "control_flow_fix_present": {
            "branch_displacement_scale_2_occurrences": codegen_text.count(
                "(self.labels[label] - index) * 2"),
            "large_tile_compact_schedule_threshold_present": (
                "((m + 31) // 32) * ((k + 31) // 32) * ((n + 31) // 32) >= 16"
                in codegen_text
            ),
        },
        "tool": str(tool),
        "git_head": git_head,
        "limits": {"per_shape_seconds": args.per_shape_timeout, "total_seconds": args.total_timeout},
        "elapsed_seconds": round(elapsed, 6),
        "routing": routing_summary,
        "physical_contraction_layers": len(layers),
        "physical_contractions_by_kind": physical_by_kind,
        "unique_kernel_shapes": len(unique),
        "unique_compile_status": unique_by_status,
        "physical_layers_by_compile_status": layers_by_status,
        "instruction_memory": imem,
        "unique_instruction_memory_fit": unique_imem_fit,
        "physical_layers_instruction_memory_fit": layers_imem_fit,
        "emitted_programs": {
            "instruction_words_min": min((row["instruction_words"] for row in unique), default=0),
            "instruction_words_max": max((row["instruction_words"] for row in unique), default=0),
            "assembly_bytes_total": sum(row.get("assembly_file_bytes", 0) for row in unique),
            "assembly_gzip_bytes_total": sum(row.get("assembly_gzip_bytes", 0) for row in unique),
            "compile_seconds_sum": round(sum(row["compile_seconds"] for row in unique), 6),
        },
        "all_unique_shapes_emitted": bool(unique) and all(row["status"] == "accepted" for row in unique),
        "notes": [
            "Routing demands preserve every captured compute/support op; physical layers count only actual matmul kernels.",
            "model2MLIR propagates contraction provenance onto bias/im2col support ops, so mesh-routed demand count is not a kernel count.",
            "A successful emission proves frontend+lowering+codegen acceptance only; it does not prove halt, numeric correctness, or a single whole-model image.",
            "Instruction-memory fit is checked independently against the generated GSIM RTL header; non-fitting kernels need compact runtime loops before hardware execution.",
        ],
    }

    (output / "demands.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in demands), encoding="utf-8"
    )
    (output / "layers.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in layers), encoding="utf-8"
    )
    (output / "unique_shapes.json").write_text(
        json.dumps(unique, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_unique_shapes_emitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
