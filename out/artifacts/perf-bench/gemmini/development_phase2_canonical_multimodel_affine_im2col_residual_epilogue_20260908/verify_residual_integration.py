#!/usr/bin/env python3
"""Fail closed on the canonical affine-im2col plus exact residual checkpoint."""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


BUNDLE = Path(__file__).resolve().parent
GEMMINI = BUNDLE.parent
PARENT = GEMMINI / "development_phase2_canonical_multimodel_affine_im2col_20260908"
Q535 = GEMMINI / "resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_firesim_candidate_20260908"
SKIP = {"build", "__pycache__", ".git"}
EXPECTED_TREE = ("6fe21d20a4f4b138102bfa9506b460c74eaa21a4ea64941a5192d78275daf19a", 43)
EXPECTED_SUPPORT = ("0714adc43f25efb2421afc37519190f9a2b58c75ad75d7a08b9ff3537936498a", 1016)
EXPECTED_FILES = {
    "compiler/mlir_oot/frontend/residual_epilogue.py": "8d9619c6ff971fbfad94d9834e7f4c3039f06a876e84379ed94a8f865c2e3f0e",
    "compiler/mlir_oot/lowering/model_lane.py": "3ed79f23e0c5beae701e3db4b12fb7ff66e2824a9408d89bf2c2f90126d78306",
    "compiler/mlir_oot/lowering/source_conv_model_lane.py": "b43601ae05e9d8059f930f502d56aa8e9efaed6f58d932145a0572b55ae503b5",
    "compiler/mlir_oot/codegen/loop_host_linalg.py": "097bf6906098fdf4681a0f893da8c287664bdb2b7ff9315de05b70d3e815d14a",
    "compiler/mlir_oot/codegen/llvm_emit.py": "f57d0b4de259992ccd443ed44fb4f166e5e4baaff3e752d22b07ca7f34e370f9",
    "compiler/manifest.yaml": "8f2618ffb5493b7f6b31e5160bd1c0e82d5bc127f7678d3273e9e0c9c4e57783",
    "tests/fixtures/residual_two_branch_template.mlir": "f0111490e81771b2bf640c1a2ad16ec999cd79269f0355b4a1591b5c3c58d9f1",
    "tests/test_residual_epilogue_fusion.py": "ca6401373869b41d9098b287cfaf739881e00722c4c4accc2cd0dd9024d97e19",
    "residual_epilogue_integration.patch": "d4382a84f7f93e898f795d935c3f8c320d71cee88c202aef054d92c8820603af",
}
DTYPE_BYTES = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
               "f16": 2, "f32": 4, "f64": 8}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_tree(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or SKIP.intersection(path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        files += 1
    return digest.hexdigest(), files


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise SystemExit(detail)


def same_file(left: Path, right: Path, label: str) -> None:
    require(left.read_bytes() == right.read_bytes(), f"{label} is not byte-identical")


def definition(path: Path, name: str, owner: str | None = None) -> str:
    text = path.read_text()
    nodes = ast.parse(text).body
    if owner is not None:
        cls = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner)
        nodes = cls.body
    node = next(node for node in nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)
    return ast.get_source_segment(text, node) or ""


def normalized(command_buffer: dict) -> dict:
    result = json.loads(json.dumps(command_buffer))
    result.get("params", {}).pop("target_neutral_residual_epilogues", None)
    for segment in result.get("params", {}).get("host_lane_segments", []):
        segment.pop("residual_fusion_sites", None)
    return result


def structural_summary(command_buffer: dict) -> dict:
    tasks = command_buffer["params"]["global_program_plan"]["tasks"]
    total = intermediate = 0
    for name, tensor in command_buffer["tensors"].items():
        size = (command_buffer["params"]["storage_encodings"][name]["storage_elements"]
                * DTYPE_BYTES[tensor["dtype"]])
        total += size
        if tensor["role"] == "intermediate":
            intermediate += size
    storage = command_buffer["params"]["host_storage"]
    boundaries = Counter((task.get("boundary_sync") or {}).get("direction") for task in tasks)
    return {
        "source_ops": command_buffer["params"]["global_program_plan"]["source_op_count"],
        "tasks": len(tasks),
        "host": sum(task["kind"] == "host" for task in tasks),
        "accelerator": sum(task["kind"] != "host" for task in tasks),
        "abi": len(command_buffer["kernel_abi"]["args"]),
        "total_bytes": total,
        "intermediate_bytes": intermediate,
        "workspace_requested": storage["workspace_requested_bytes"],
        "workspace_peak": storage["workspace_global_bytes"],
        "allocations": storage["allocation_count"],
        "host_writes": sum(len(task["writes"]) for task in tasks if task["kind"] == "host"),
        "d2h": boundaries["device_to_host"],
        "h2d": boundaries["host_to_device"],
    }


def main() -> int:
    require(hash_tree(BUNDLE / "compiler") == EXPECTED_TREE,
            f"compiler tree changed: {hash_tree(BUNDLE / 'compiler')}")
    require(hash_tree(BUNDLE / "support/merlin") == EXPECTED_SUPPORT,
            f"support tree changed: {hash_tree(BUNDLE / 'support/merlin')}")
    for relative, expected in EXPECTED_FILES.items():
        require(sha256(BUNDLE / relative) == expected, f"sealed file changed: {relative}")

    # The isolated child adds one generic recognizer and changes only the five intended
    # integration surfaces plus the manifest. Everything else remains canonical.
    new_file = Path("mlir_oot/frontend/residual_epilogue.py")
    changed = {
        Path("manifest.yaml"), Path("mlir_oot/codegen/llvm_emit.py"),
        Path("mlir_oot/codegen/loop_host_linalg.py"), Path("mlir_oot/lowering/model_lane.py"),
        Path("mlir_oot/lowering/source_conv_model_lane.py"), new_file,
    }
    parent_files = {path.relative_to(PARENT / "compiler")
                    for path in (PARENT / "compiler").rglob("*")
                    if path.is_file() and not SKIP.intersection(path.parts)}
    child_files = {path.relative_to(BUNDLE / "compiler")
                   for path in (BUNDLE / "compiler").rglob("*")
                   if path.is_file() and not SKIP.intersection(path.parts)}
    require(child_files == parent_files | {new_file}, "compiler inventory changed outside residual pass")
    for relative in sorted(parent_files - changed):
        same_file(PARENT / "compiler" / relative, BUNDLE / "compiler" / relative,
                  f"canonical compiler {relative}")

    parent_emit = PARENT / "compiler/mlir_oot/codegen/llvm_emit.py"
    child_emit = BUNDLE / "compiler/mlir_oot/codegen/llvm_emit.py"
    for name, owner in (("emit_loop_conv_ws", "Emitter"),
                        ("emit_im2col_row", "Emitter"),
                        ("_universal_valid_x_span", None)):
        require(definition(child_emit, name, owner) == definition(parent_emit, name, owner),
                f"canonical affine/native emitter changed: {name}")
    manifest = yaml.safe_load((BUNDLE / "compiler/manifest.yaml").read_text())
    surfaces = [row for row in manifest["optimization_surfaces"]
                if row["id"] == "exact_second_tensor_residual_epilogue"]
    require(len(surfaces) == 1, "exact residual optimization surface missing or duplicated")

    current = BUNDLE / "validation/residual_integration"
    parent = PARENT / "validation/affine_im2col/default"
    baseline_cb = json.loads((parent / "resnet50/command_buffer.json").read_text())
    candidate_cb = json.loads((current / "resnet50/command_buffer.json").read_text())
    require(structural_summary(baseline_cb) == {
        "source_ops": 1240, "tasks": 109, "host": 55, "accelerator": 54, "abi": 393,
        "total_bytes": 116646848, "intermediate_bytes": 90079488,
        "workspace_requested": 23405568, "workspace_peak": 7341056,
        "allocations": 66, "host_writes": 122, "d2h": 54, "h2d": 50,
    }, "canonical ResNet baseline structural summary changed")
    require(structural_summary(candidate_cb) == {
        "source_ops": 1240, "tasks": 105, "host": 51, "accelerator": 54, "abi": 389,
        "total_bytes": 109306816, "intermediate_bytes": 82739456,
        "workspace_requested": 7349248, "workspace_peak": 7341056,
        "allocations": 54, "host_writes": 118, "d2h": 50, "h2d": 50,
    }, "residual ResNet structural summary changed")
    tasks = candidate_cb["params"]["global_program_plan"]["tasks"]
    owned = [index for task in tasks for index in task["source_op_indices"]]
    require(len(owned) == len(set(owned)) == 1240 and sorted(owned) == list(range(1240)),
            "ResNet source ownership is no longer exact")
    residual = candidate_cb["params"]["target_neutral_residual_epilogues"]
    require((residual["formed_site_count"], residual["selected_site_count"],
             residual["deferred_branch_count"]) == (15, 15, 4), "residual site counts changed")
    require(residual["refused"] == [{
        "source_op_index": 1204,
        "reason": "residual float result has no unique exact roundeven/clamp i8 sink",
    }], "fail-closed ResNet refusal changed")
    expected_opcodes = Counter({"RES_PACK": 54, "CONV2D": 53,
                                "MATMUL_RESIDENT": 1, "COMMIT": 1})
    require(Counter(row["opcode"] for row in baseline_cb["commands"]) == expected_opcodes
            and Counter(row["opcode"] for row in candidate_cb["commands"]) == expected_opcodes,
            "accelerator command stream changed")

    base_target = (parent / "resnet50/target.mlir").read_text()
    cand_target = (current / "resnet50/target.mlir").read_text()
    expected_sites = {
        "llvm.load": (1867, 1827), "llvm.store": (1682, 1666),
        "llvm.br": (4584, 4416), "llvm.cond_br": (2292, 2208),
        "llvm.fmul": (158, 158), "llvm.fadd": (121, 121),
        "llvm.intr.roundeven": (50, 50), "llvm.fptosi": (50, 50),
    }
    require({name: (base_target.count(f'"{name}"'), cand_target.count(f'"{name}"'))
             for name in expected_sites} == expected_sites,
            "ResNet target site A/B changed")

    for model, expected in {
        "tinyllama": (718, 1, "859f0f45c349f4e206dbc21127a804fa2148de410529d699c63c286eb0ea1fc5"),
        "lstmnetvit": (2302, 75, "4c912f9ac995f8eb2677be1ff286369c32178166d6a02a037e49f7f4dbeed0f7"),
        "smolvla": (11910, 233, "f1cdc9aed4b3db465733b5af4e201e92c6eddafb685b76f7607daeb607969b3b"),
    }.items():
        candidate_dir, parent_dir = current / model, parent / model
        candidate = json.loads((candidate_dir / "command_buffer.json").read_text())
        canonical = json.loads((parent_dir / "command_buffer.json").read_text())
        tasks = candidate["params"]["global_program_plan"]["tasks"]
        require((candidate["params"]["global_program_plan"]["source_op_count"], len(tasks),
                 sha256(candidate_dir / "target.mlir")) == expected, f"{model} summary changed")
        same_file(candidate_dir / "target.mlir", parent_dir / "target.mlir", f"{model} target")
        require(normalized(candidate) == normalized(canonical), f"{model} normalized CB changed")
        require(candidate["params"]["target_neutral_residual_epilogues"]["formed_site_count"] == 0,
                f"{model} acquired an unexpected residual site")

    stack = json.loads((current / "resnet50/object_build/kernel.stack_frame.json").read_text())
    require(stack["status"] == "passed" and stack["frame_bytes"] == 1296
            and stack["headroom_bytes"] == 64240, "candidate stack preflight changed")

    qrun = current / "resnet50/q535_spike"
    prep = json.loads((qrun / "preparation_receipt.json").read_text())
    require(prep["constant_blob_byte_identical_to_q535"] is True
            and prep["read_only_prefix_byte_layout_identical_to_q535"] is True
            and (prep["kernel_arg_count"], prep["read_args"], prep["mutable_args"]) == (389, 217, 172),
            "q535 ABI rebind changed")
    baseline_log = Q535 / "validation/spike_warm_measured.log"
    candidate_log = qrun / "spike_warm_measured.log"
    require(sha256(baseline_log) == "25a320f2ee3bbb614ee557334ec711e34b92ef3fc30bd9111fd65f04ac39f659",
            "q535 baseline log changed")
    require(sha256(candidate_log) == "a0ee2c2739bb8677c5f7dcb1f7f75e52c38a9839d2b4bfb55a202accc47a42b5",
            "candidate exact Spike log changed")
    lines = set(candidate_log.read_text().splitlines())
    require({
        "MERLIN_METRIC cycles=492147976",
        "MERLIN_METRIC instret=492147980",
        "MERLIN_RESULT checksum_fnv1a64=c6e777c3fe0aae90",
        "MERLIN_RESULT logits_checked=1000 bad=0 nonfinite=0 top1=258 expected_top1=258",
        "PASS: warm-then-measured Merlin W8A8 ResNet-50 and all-logit check",
    }.issubset(lines), "candidate exact result markers changed")
    require(sha256(qrun / "spike_warm_measured.stderr") ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "candidate Spike stderr is no longer empty")

    receipt = json.loads((current / "receipt.json").read_text())
    require(receipt["status"] == "passed_exact_full_model_and_portfolio",
            "integration receipt status changed")
    spike = receipt["full_model_spike_ab"]
    require(spike["arguments"] == ["--isa=rv64gc_zicntr", "--extension=gemmini"]
            and spike["external_extension_loaded"] is False,
            "Spike extension regression guard failed")
    require(spike["baseline"]["cycles"] == 506265226
            and spike["candidate"]["cycles"] == 492147976
            and spike["cycles_saved"] == 14117250,
            "Spike A/B changed")
    require(receipt["hardware"] == {
        "firesim_run_performed": False, "l3_run_performed": False,
        "queue_action_performed": False, "hardware_claim": "none",
    }, "receipt unexpectedly claims expensive execution")
    print("PASS canonical affine/native/dynamic mechanisms, exact generic residual fusion, "
          "four-model gates, and 492147976-cycle exact warm/measured ResNet proxy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
