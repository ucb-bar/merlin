"""Merlin precision lowering, instruction selection and Atlas compilation."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from safetensors.numpy import load_file

from source_roots import MERLIN_ROOT as ROOT, QWEN_ROOT, bootstrap_compiler, observed_revision

COMPILER_ROOT = bootstrap_compiler()
from atlas_mlir_native.binding import AtlasNativeBinding, diagnostic_mxu0_f32_profile
from atlas_mlir_native.graph_precision import lower_diagnostic_graph
from atlas_mlir_native.graph_abi import prepare_graph_inputs
from atlas_mlir_native.precision_abi import pack_source_operand
from merlin.semantic_compiler.snapshot import (
    build_native_snapshot,
    packaged_egg_bridge_path,
    open_native_snapshot,
)
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.target_binding import (
    verify_native_publication,
    NativeCompilationError,
)
from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
from merlin.semantic_compiler.precision_transform import build_diagnostic_matmul_tile

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    root = args.out.resolve()
    policy = json.loads(
        (QWEN_ROOT / "tests/examples/attention_mlp_graph_precision.json").read_text()
    )
    policy["interlayer_pack"] = "scaled-e8m0-rtl-rne-v1"
    snapshot_dir = root / "snapshot"
    snap = (
        open_native_snapshot(snapshot_dir)
        if snapshot_dir.exists()
        else build_native_snapshot(
            AtlasNativeBinding.profile(),
            destination=snapshot_dir,
            crate=packaged_egg_bridge_path(),
            cargo_target_dir=root / "cargo",
            source_revision=observed_revision(ROOT) or "installed-merlin",
        )
    )
    rows = []
    for workload in json.loads((root / "workloads.json").read_text())["workloads"]:
        name = workload["name"]
        if args.only and name not in args.only:
            continue
        dest = root / name
        source = (dest / "capture/linalg.mlir").read_text()
        started = time.monotonic()
        row = dict(name=name, source_sha256=hashlib.sha256(source.encode()).hexdigest())
        try:
            if workload["route"] == "projection":
                profile = diagnostic_mxu0_f32_profile()
                selected_policy = dict(profile.required_policy_fields)
                selected_policy.update(
                    source_sha256=row["source_sha256"],
                    matmul_regions={"matmul#0": dict(profile.selected_decision)},
                )
                plan = build_diagnostic_matmul_tile(
                    source,
                    entry="forward",
                    policy_document=json.dumps(selected_policy).encode(),
                    original_source_sha256=row["source_sha256"],
                    region_id="matmul#0",
                    profile=profile,
                    tile_origin=(0, 0),
                    ordered_k_continuation=True,
                )
                request = plan.request
                record = plan.record()
                outputs = (127,)
            else:
                selected_policy = policy
                plan = lower_diagnostic_graph(source, entry="forward", policy=policy)
                request, record = plan.request, plan.record()
                outputs = tuple(range(64, 64 + len(request.outputs)))
            write(dest / "policy.json", selected_policy)
            write(dest / "precision.json", record)
            names = [name for name, _ in request.input_storages] + [
                c.node_id for c in request.constants
            ]
            fixed = {name: i for i, name in enumerate(names)}
            write(dest / "abi.json", dict(fixed_inputs=fixed, fixed_outputs=outputs))
            manifest = AtlasNativeBinding.compile(
                snap,
                request,
                fixed_inputs=fixed,
                fixed_outputs=outputs,
                target_source=COMPILER_ROOT,
                destination=dest / "native",
                limits=SearchLimits(
                    candidate_nodes=256,
                    candidates=512,
                    orders_per_candidate=64,
                    wall_timeout_s=60 if workload["route"] == "projection" else 120,
                ),
            )
            verify_native_publication(
                dest / "native",
                manifest,
                engine="merlin_native",
                request_digest=request.digest(),
                target_identity=request.target_identity,
            )
            execution = json.loads((dest / "native/execution_plan.json").read_text())
            values = load_file(str(dest / "capture/inputs.safetensors"))
            payloads = dest / "inputs"
            payloads.mkdir()
            if workload["route"] == "projection":
                for role, slot in fixed.items():
                    source_values = (
                        values["x"] if role.startswith("lhs") else values["weight"].T
                    )
                    (payloads / f"{slot}.bin").write_bytes(
                        pack_source_operand(plan, role, source_values)
                    )
            else:
                prepared = prepare_graph_inputs(
                    plan,
                    policy,
                    record,
                    manifest,
                    execution,
                    json.loads((dest / "capture/arguments.json").read_text()),
                    weights=load_file(str(dest / "capture/weights.safetensors")),
                    invocation=values,
                )
                for item in prepared:
                    (
                        payloads / f"{(item.address - 0x90000000) // 2048}.bin"
                    ).write_bytes(item.payload)
                for item in execution.get("constant_inputs", []):
                    (
                        payloads / f"{(item['address'] - 0x90000000) // 2048}.bin"
                    ).write_bytes((dest / "native" / item["path"]).read_bytes())
            row.update(
                status="compiled",
                request_digest=request.digest(),
                words=(dest / "native/program.bin").stat().st_size // 4,
            )
        except (LinalgBridgeError, NativeCompilationError) as error:
            row.update(
                status=error.status
                if isinstance(error, NativeCompilationError)
                else "unsupported",
                reason=str(error),
            )
            if workload["route"] == "graph" or name.startswith("projection_k"):
                write(dest / "compile-status.json", row)
                raise
        row["compile_seconds"] = time.monotonic() - started
        write(dest / "compile-status.json", row)
        rows.append(row)
        print(f"{name}: {row['status']}", flush=True)
    write(root / "compilation.json", rows)


if __name__ == "__main__":
    main()
