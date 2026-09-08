#!/usr/bin/env python3
"""Plan the full capture and compile every referenced Atlas kernel variant."""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
CAPTURE = REPO / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
TOOL = ROOT / "submission/mlir_oot/atlas-opt"
PLAN_ROOT = ROOT / "whole_capture_plan"
INTERFACES = PLAN_ROOT / "interfaces"
KERNELS = PLAN_ROOT / "kernels"
COMMANDS = PLAN_ROOT / "command_buffers"
for directory in (PLAN_ROOT, INTERFACES, KERNELS, COMMANDS):
    directory.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "submission"))

from mlir_oot.frontend import parse_verified  # noqa: E402
from mlir_oot.planner import plan_full_graph  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def interface_for(partition: dict) -> str:
    geometry = partition["geometry"]
    if partition["kind"] == "matmul_batched":
        b, m, k, n = (geometry[key] for key in ("B", "M", "K", "N"))
        return f'''module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"}} {{
  %A0 = merlin_iface.tensor {{name = "A0", role = "input"}} : tensor<{b}x{m}x{k}xf8E4M3FN>
  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{b}x{k}x{n}xf8E4M3FN>
  %Y0 = merlin_iface.matmul_batched %A0, %W {{name = "Y0", batch = {b} : i64, output_dtype = "bf16"}} : (tensor<{b}x{m}x{k}xf8E4M3FN>, tensor<{b}x{k}x{n}xf8E4M3FN>) -> tensor<{b}x{m}x{n}xbf16>
}}
'''
    m, k, n = (geometry[key] for key in ("M", "K", "N"))
    bias = (
        f'  %B = merlin_iface.tensor {{name = "B", role = "bias"}} : tensor<{n}xbf16>\n'
        if partition["bias_fused"] else ""
    )
    epilogue = '["bias_add"]' if partition["bias_fused"] else "[]"
    bias_attr = ', bias = "B"' if partition["bias_fused"] else ""
    return f'''module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"}} {{
  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{k}x{n}xf8E4M3FN>
  %A0 = merlin_iface.tensor {{name = "A0", role = "input"}} : tensor<{m}x{k}xf8E4M3FN>
{bias}  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : (tensor<{k}x{n}xf8E4M3FN>) -> !merlin_iface.resident
  %acc0 = merlin_iface.matmul %A0, %W_res : (tensor<{m}x{k}xf8E4M3FN>, !merlin_iface.resident) -> !merlin_iface.acc<bf16>
  %Y0 = merlin_iface.commit %acc0 {{name = "Y0", epilogue = {epilogue}, output_dtype = "bf16"{bias_attr}}} : (!merlin_iface.acc<bf16>) -> tensor<{m}x{n}xbf16>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}}
'''


source = CAPTURE.read_text(encoding="utf-8")
plan = plan_full_graph(parse_verified(source))
representatives = {}
for partition in plan["partitions"]:
    representatives.setdefault(partition["kernel_id"], partition)

kernel_library = []
for kernel_id in sorted(representatives):
    partition = representatives[kernel_id]
    interface = interface_for(partition)
    interface_path = INTERFACES / f"{kernel_id}.mlir"
    command_path = COMMANDS / f"{kernel_id}.json"
    kernel_path = KERNELS / f"{kernel_id}.S.gz"
    command_path.unlink(missing_ok=True)
    kernel_path.unlink(missing_ok=True)
    interface_path.write_text(interface, encoding="utf-8")
    proc = subprocess.run(
        [str(TOOL), f"--emit-command-buffer={command_path}",
         "--emit-target-artifact", str(interface_path)],
        capture_output=True, text=True, timeout=60,
    )
    assembly = proc.stdout.encode()
    with kernel_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as target:
            target.write(assembly)
    words = sum(line.lstrip().startswith(".word") for line in proc.stdout.splitlines())
    command_buffer = json.loads(command_path.read_text()) if command_path.is_file() else None
    occurrences = sum(p["kernel_id"] == kernel_id for p in plan["partitions"])
    kernel_library.append({
        "kernel_id": kernel_id,
        "kind": partition["kind"],
        "geometry": partition["geometry"],
        "bias_fused": partition["bias_fused"],
        "partition_occurrences": occurrences,
        "returncode": proc.returncode,
        "stderr": proc.stderr,
        "instruction_words": words,
        "imem_words": 32768,
        "fits_imem": proc.returncode == 0 and words <= 32768,
        "interface": interface_path.relative_to(ROOT).as_posix(),
        "interface_sha256": digest(interface.encode()),
        "assembly_gzip": kernel_path.relative_to(ROOT).as_posix(),
        "assembly_sha256": digest(assembly),
        "assembly_gzip_sha256": digest(kernel_path.read_bytes()),
        "command_buffer": command_path.relative_to(ROOT).as_posix(),
        "command_count": len(command_buffer["commands"]) if command_buffer else None,
    })

receipts = {row["kernel_id"]: row for row in kernel_library}
for partition in plan["partitions"]:
    receipt = receipts[partition["kernel_id"]]
    partition["image"] = {
        "kernel_id": partition["kernel_id"],
        "instruction_words": receipt["instruction_words"],
        "imem_words": receipt["imem_words"],
        "fits_imem": receipt["fits_imem"],
        "command_count": receipt["command_count"],
    }

inventory = json.loads((ROOT / "full_capture_partition_inventory.json").read_text())
origin_counts = Counter(
    value["origin"]["kind"]
    for partition in plan["partitions"]
    for value in partition["abi"]["inputs"]
)
plan.update({
    "capture": {"path": "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir",
                "bytes": len(source.encode()), "sha256": digest(source.encode())},
    "kernel_library": kernel_library,
    "compile_coverage": {
        "imem_fit_structural_partitions": sum(receipts[p["kernel_id"]]["fits_imem"]
                                              for p in plan["partitions"]),
        "structural_partitions_total": len(plan["partitions"]),
        "capture_semantics_executable_partitions": 0,
        "unique_kernel_variants_fitting_imem": sum(r["fits_imem"] for r in kernel_library),
        "unique_kernel_variants_total": len(kernel_library),
    },
    "host_required": inventory["host_required_breakdown"],
    "boundary_input_origins": dict(sorted(origin_counts.items())),
    "limitations": [
        "partition images are compiled but not assembled into a whole-model dispatcher",
        "host-required regions retain capture semantics and are not replaced by invented commands",
        "FP8/BF16 quantization is not yet calibrated or numerically graded for the full model",
        "output lifetime endpoints follow layout bridges to the first real consumer but do not allocate storage",
    ],
})

(PLAN_ROOT / "partition_plan.json").write_text(
    json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(PLAN_ROOT / "dependency_manifest.json").write_text(
    json.dumps({
        "schema": "atlas_partition_dependencies_v1",
        "edges": plan["accelerator_dependency_edges"],
        "maximal_accelerator_islands": plan["maximal_accelerator_islands"],
        "boundary_input_origins": plan["boundary_input_origins"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(PLAN_ROOT / "lifetime_manifest.json").write_text(
    json.dumps({
        "schema": "atlas_partition_lifetimes_v1",
        "basis": "capture top-level operation ordinals; view chains followed to first real consumer",
        "outputs": [{"partition_id": p["partition_id"], **p["lifetime"],
                     "device_bytes": p["abi"]["outputs"][0]["device_bytes"]}
                    for p in plan["partitions"]],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
(PLAN_ROOT / "abi_manifest.json").write_text(
    json.dumps({
        "schema": "atlas_partition_abi_v1",
        "partitions": [{"partition_id": p["partition_id"], "kernel_id": p["kernel_id"],
                        "capture_regions": p["capture_regions"], "abi": p["abi"]}
                       for p in plan["partitions"]],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
summary = {
    "partitions": plan["partition_count"],
    "kernel_variants": plan["kernel_variant_count"],
    "maximal_accelerator_islands": plan["maximal_accelerator_island_count"],
    "accelerator_edges": len(plan["accelerator_dependency_edges"]),
    "compile_coverage": plan["compile_coverage"],
    "host_required": plan["host_required"],
}
print(json.dumps(summary, indent=2, sort_keys=True))
raise SystemExit(0 if all(row["fits_imem"] for row in kernel_library) else 1)
