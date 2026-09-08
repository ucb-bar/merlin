#!/usr/bin/env python3
"""Fail closed on the final exact-epilogue/residual/global-encoding artifact."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
GEMMINI = BUNDLE.parent
RESIDUAL = GEMMINI / "development_phase2_canonical_multimodel_affine_im2col_residual_epilogue_20260908"
Q535 = GEMMINI / "resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_firesim_candidate_20260908"
SKIP = {"build", "__pycache__", ".git"}
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
    print(f"{'PASS' if condition else 'FAIL'} {detail}")
    if not condition:
        raise SystemExit(2)


def structural_summary(cb: dict) -> dict:
    tasks = cb["params"]["global_program_plan"]["tasks"]
    total = intermediate = 0
    for name, tensor in cb["tensors"].items():
        size = (cb["params"]["storage_encodings"][name]["storage_elements"]
                * DTYPE_BYTES[tensor["dtype"]])
        total += size
        if tensor["role"] == "intermediate":
            intermediate += size
    storage = cb["params"]["host_storage"]
    return {
        "source_ops": cb["params"]["global_program_plan"]["source_op_count"],
        "tasks": len(tasks),
        "host": sum(task["kind"] == "host" for task in tasks),
        "accelerator": sum(task["kind"] != "host" for task in tasks),
        "abi": len(cb["kernel_abi"]["args"]),
        "total_bytes": total,
        "intermediate_bytes": intermediate,
        "workspace_requested": storage["workspace_requested_bytes"],
        "workspace_peak": storage["workspace_global_bytes"],
        "allocations": storage["allocation_count"],
    }


def main() -> int:
    receipt_path = BUNDLE / "validation/final_combined_receipt.json"
    require(sha256(receipt_path) ==
            "235ee8843be114bc59a8298b91f78dbe74499ba249ec1216434f999b865019cd",
            "final receipt seal")
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "passed_exact_combined_and_transferred_full_model",
            "final receipt status")
    require(hash_tree(BUNDLE / "compiler") == (
        "48694957d14c9608960f7ac2b6cd08b72b15c55e4a11ff26e1cf952e0cd7607e", 45),
        f"compiler seal {hash_tree(BUNDLE / 'compiler')}")
    require(hash_tree(BUNDLE / "tests") == (
        "b542aa95c5a2c652c6cc30e49e45d62583027b0833c4f9d903310b148ee6523c", 18),
        f"tests seal {hash_tree(BUNDLE / 'tests')}")
    require(hash_tree(BUNDLE / "support/merlin") == (
        "0714adc43f25efb2421afc37519190f9a2b58c75ad75d7a08b9ff3537936498a", 1016),
        "support snapshot seal")
    require(sha256(BUNDLE / "final_combined_compiler.patch") ==
            "0e611e3c9c2231196653bfdb906a45843b89f8ee14f4ffbefe683116a9f2477a",
            "review patch seal")

    exact_path = BUNDLE / "validation/exact_epilogue_warm_spike/receipt.json"
    exact = json.loads(exact_path.read_text())
    require(sha256(exact_path) ==
            "5e358eb793173c90d8dc2d8d678da47b448692a9841d5bea5706548316b14024",
            "exact epilogue witness receipt seal")
    require(exact["status"] == "passed" and exact["actual"] == exact["expected"]
            and len(exact["actual"]) == 323 and exact["metrics"] == {"cycles": 56},
            "exact epilogue same-process warm/measured output and cycles")
    require(set(exact["markers"].values()) == {1}, "exact epilogue profile markers")

    encoded_path = BUNDLE / "validation/global_encoding/reduced_chain_spike/receipt.json"
    encoded = json.loads(encoded_path.read_text())
    require(sha256(encoded_path) ==
            "885e1b9492be41d8fc051b95e7ae0f010fff598991f82d3443627fb90f9e603b",
            "global encoding witness receipt seal")
    require(encoded["status"] == "passed" and encoded["actual"] == encoded["expected"]
            and encoded["outputs_checked"] == 96 and encoded["metrics"] == {"cycles": 102}
            and (encoded["warmup_invocations"], encoded["measured_invocations"]) == (1, 1)
            and (encoded["native_convolution_tasks"], encoded["im2col_tasks"]) == (2, 0),
            "global encoding two-layer warm/measured exact witness")
    chain = json.loads((BUNDLE /
        "validation/global_encoding/reduced_chain_spike/command_buffer.json").read_text())
    require(chain["params"]["global_program_plan"]["tasks"][1]["boundary_sync"] == {
        "direction": "device_dependency", "tensors": ["Y0"]},
        "native producer-consumer hazard fence")

    portfolio = BUNDLE / "validation/final_portfolio"
    expected = {
        "resnet50": (1240, 105,
            "c29ba22408242dcad9aa215c33a4bbca8ac45d12f87352360afeb16ee5167883",
            "e0054a7660956a526d7a972c6e29a85976c027071efa4e0272af697810187665"),
        "tinyllama": (718, 1,
            "859f0f45c349f4e206dbc21127a804fa2148de410529d699c63c286eb0ea1fc5",
            "cdbda3674aed5731910326cb4c9a0140d159ad07bff61cd825a415b2c5f284c6"),
        "lstmnetvit": (2302, 75,
            "4c912f9ac995f8eb2677be1ff286369c32178166d6a02a037e49f7f4dbeed0f7",
            "d011d36c9506ddf42241c806058e3333f000dfb5f6c7ad334857161426c4e7fe"),
        "smolvla": (11910, 233,
            "f1cdc9aed4b3db465733b5af4e201e92c6eddafb685b76f7607daeb607969b3b",
            "bf7a4a1a30910ac676941f51d955f7b7cddb4f977fdb3ffcdbc460b5620a5aef"),
    }
    for model, (source_ops, task_count, target_hash, cb_hash) in expected.items():
        directory = portfolio / model
        cb = json.loads((directory / "command_buffer.json").read_text())
        tasks = cb["params"]["global_program_plan"]["tasks"]
        require((cb["params"]["global_program_plan"]["source_op_count"], len(tasks),
                 sha256(directory / "target.mlir"),
                 sha256(directory / "command_buffer.json")) ==
                (source_ops, task_count, target_hash, cb_hash), f"{model} compile identity")
        require("Exit status: 0" in (directory / "time.txt").read_text()
                and not (directory / "compile.stderr").read_text(),
                f"{model} sequential compile status")

    bridge = portfolio / "tinyllama_dynamic_bridge"
    bridge_cb = json.loads((bridge / "command_buffer.json").read_text())
    require(sha256(bridge / "target.mlir") ==
            "a1d39869e27efe2d5f3b4be5f8cb75020746423b761e69d96e69c1ec7d015082"
            and sha256(bridge / "command_buffer.json") ==
            "aea701e7ce1a449a62f8b4604664e677f164d1db10403fab0bf24c7d5644ffda"
            and bridge_cb["params"]["dynamic_weight_only_contract"]
                ["source_f32_bit_equivalence"] == "NOT_CLAIMED",
            "explicit opt-in dynamic-weight bridge")

    resnet_dir = portfolio / "resnet50"
    cb = json.loads((resnet_dir / "command_buffer.json").read_text())
    require(structural_summary(cb) == {
        "source_ops": 1240, "tasks": 105, "host": 51, "accelerator": 54,
        "abi": 389, "total_bytes": 109306816, "intermediate_bytes": 82739456,
        "workspace_requested": 7349248, "workspace_peak": 7341056,
        "allocations": 54}, "ResNet structural summary")
    owned = [index for task in cb["params"]["global_program_plan"]["tasks"]
             for index in task["source_op_indices"]]
    require(len(owned) == len(set(owned)) == 1240 and sorted(owned) == list(range(1240)),
            "ResNet exact unique source ownership")
    require(Counter(row["opcode"] for row in cb["commands"]) == Counter({
        "RES_PACK": 54, "CONV2D": 53, "MATMUL_RESIDENT": 1, "COMMIT": 1}),
        "ResNet accelerator command inventory")
    quantized = cb["params"]["target_neutral_quantized_epilogues"]
    require(quantized["selected_count"] == 0 and Counter(
        row["reason"] for row in quantized["refused"]) == Counter({
            "accumulator_scale_is_not_proven_identity": 32,
            "dynamic second tensor/residual terminates epilogue": 20,
            "chain does not end in exact roundeven/clamp i8 quantize": 2}),
        "ResNet exact native epilogue fail-closed inventory")
    residual = cb["params"]["target_neutral_residual_epilogues"]
    require((residual["formed_site_count"], residual["selected_site_count"],
             residual["deferred_branch_count"]) == (15, 15, 4)
            and residual["refused"] == [{
                "source_op_index": 1204,
                "reason": "residual float result has no unique exact roundeven/clamp i8 sink"}],
            "ResNet exact residual formation and refusal inventory")
    encoding = cb["params"]["global_encoding"]
    require((encoding["status"], len(encoding["dependencies"]),
             encoding["logical_boundary_bytes"],
             encoding["encoded_boundary_bytes_if_legal"],
             encoding["potential_boundary_byte_reduction"],
             encoding["capacity"]["peak_live_encoded_bytes_if_legal"]) ==
            ("refused_exact_semantic_boundary", 53, 44455936, 11113984, 33341952,
             802816), "ResNet post-fusion global encoding ledger")
    require(all(not row["exact_narrow_epilogue_selected"]
                and row["current"]["output_dtype"] == "i32"
                for row in encoding["dependencies"]),
            "ResNet encoding rows reflect actual post-fusion boundaries")

    final_target = resnet_dir / "target.mlir"
    residual_target = RESIDUAL / "validation/residual_integration/resnet50/target.mlir"
    require(final_target.read_bytes() == residual_target.read_bytes(),
            "ResNet target byte identity transfers full-model result")
    prior = json.loads((RESIDUAL / "validation/residual_integration/receipt.json").read_text())
    spike = prior["full_model_spike_ab"]
    require(spike["candidate"]["cycles"] == 492147976
            and spike["correctness"] == {
                "logits_checked": 1000, "bad": 0, "nonfinite": 0,
                "top1": 258, "expected_top1": 258,
                "checksum_fnv1a64": "c6e777c3fe0aae90",
                "max_abs_bits": "00000000", "max_rel_bits": "00000000"}
            and spike["arguments"] == ["--isa=rv64gc_zicntr", "--extension=gemmini"]
            and spike["external_extension_loaded"] is False,
            "transferred exact ResNet warm/measured Spike result")
    require(receipt["hardware"] == {
        "firesim_run_performed": False, "l3_run_performed": False,
        "queue_action_performed": False, "hardware_claim": "none"},
        "no expensive execution or hardware claim")
    q535_receipt = Q535 / "validation/firesim_queue_job_535_success.json"
    require(sha256(q535_receipt) ==
            "399fb17188aa7da4bc1b0e61d6876ef0e2657165ac87e3da6627bb11681571c1",
            "q535 hardware-lineage receipt seal")
    q535 = json.loads(q535_receipt.read_text())
    require(q535["status"] == "passed"
            and q535["measured_metrics"]["cycles"] == 1316619699,
            "q535 remains latest full-Merlin hardware checkpoint")
    alias = json.loads((BUNDLE / "validation/combined_receipt.json").read_text())
    require(alias == {
        "schema": "phase2_final_combined_receipt_alias_v1",
        "status": "canonical_receipt_is_final_combined_receipt_json",
        "canonical_receipt": "final_combined_receipt.json",
        "canonical_receipt_sha256":
            "235ee8843be114bc59a8298b91f78dbe74499ba249ec1216434f999b865019cd",
    }, "unambiguous combined receipt alias")
    print("PASS final combined compiler: 65 tests, both warm witnesses, four-model + bridge "
          "gates, exact ownership/ABI/refusal/encoding audits, and byte-identical ResNet transfer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
