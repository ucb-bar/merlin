#!/usr/bin/env python3
"""Fail closed on the exact native-epilogue compiler and its scoped evidence."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
CANONICAL = BUNDLE.parent / "development_phase2_canonical_multimodel_affine_im2col_20260908"
Q535 = BUNDLE.parent / "q535_affine_im2col_hardware_receipt.json"
SKIP = {"build", "__pycache__", ".git"}


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


def stable(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def check(label: str, condition: bool, detail: object) -> None:
    print(f"{'PASS' if condition else 'FAIL'} {label}: {detail}")
    if not condition:
        raise SystemExit(2)


def main() -> int:
    receipt = json.loads((BUNDLE / "validation/exact_native_epilogue_receipt.json").read_text())
    check("receipt-status", receipt["status"] == "passed_scoped_exact_no_resnet_codegen_change",
          receipt["status"])
    check("compiler-tree", hash_tree(BUNDLE / "compiler") ==
          ("c6429ca7055b8b9df3ae2fc4dc0bf311a0b0d92678cc7661fbe75fdc5f6b61fa", 43),
          hash_tree(BUNDLE / "compiler"))
    check("tests-tree", hash_tree(BUNDLE / "tests") ==
          ("0a2a419fc88924991f02308f7f6948449c2b7a52487ba9d026f34b44bab47ec4", 14),
          hash_tree(BUNDLE / "tests"))
    check("review-patch", sha256(BUNDLE / "exact_native_epilogue_compiler.patch") ==
          "0e882560625d17c1ff9840d26670f0a492e4c5a24d4cbdef9ee0d9fa600ae88a",
          sha256(BUNDLE / "exact_native_epilogue_compiler.patch"))

    baseline = json.loads((BUNDLE / "validation/exact_epilogue_warm_spike/baseline/receipt.json").read_text())
    candidate = json.loads((BUNDLE / "validation/exact_epilogue_warm_spike/receipt.json").read_text())
    for name, record in (("baseline", baseline), ("candidate", candidate)):
        check(f"{name}-warm-status", record["status"] == "passed", record["status"])
        check(f"{name}-same-process-protocol",
              record["markers"] == {
                  "invocation_declaration": 1, "warm_begin": 1, "warm_end": 1,
                  "measured_begin": 1, "measured_end": 1, "cycle_metric": 1},
              record["markers"])
        check(f"{name}-exact-output", record["actual"] == record["expected"],
              f"elements={len(record['actual'])}")
    check("micro-real-code-change",
          baseline["object_sha256"] != candidate["object_sha256"] and
          baseline["metrics"]["cycles"] == 13760 and candidate["metrics"]["cycles"] == 56,
          f"{baseline['object_sha256']} -> {candidate['object_sha256']}")
    check("micro-physical-deltas",
          candidate["full_width_intermediate_bytes_eliminated"] == 1292 and
          candidate["output_dma_bytes_eliminated"] == 969 and
          candidate["physical_fences_eliminated"] == 0 and
          baseline["fence_count"] == candidate["fence_count"] == 2,
          {key: candidate[key] for key in (
              "full_width_intermediate_bytes_eliminated", "output_dma_bytes_eliminated",
              "physical_fences_eliminated", "fence_count")})

    models = {
        "resnet50": "c18b8671a2c2ee3af17e0a89309a4cdc82dd2f4e3afac2fe65787743a01db859",
        "tinyllama_default": "859f0f45c349f4e206dbc21127a804fa2148de410529d699c63c286eb0ea1fc5",
        "lstmnetvit": "4c912f9ac995f8eb2677be1ff286369c32178166d6a02a037e49f7f4dbeed0f7",
        "smolvla": "f1cdc9aed4b3db465733b5af4e201e92c6eddafb685b76f7607daeb607969b3b",
    }
    for model, expected in models.items():
        target = BUNDLE / f"validation/portfolio/{model}/target.mlir"
        check(f"{model}-target", sha256(target) == expected, sha256(target))

    cb_path = BUNDLE / "validation/portfolio/resnet50/command_buffer.json"
    cb = json.loads(cb_path.read_text())
    ep = cb["params"]["target_neutral_quantized_epilogues"]
    histogram = Counter(row["reason"] for row in ep["refused"])
    check("resnet-exact-refusals", ep["selected_count"] == 0 and histogram == Counter({
        "accumulator_scale_is_not_proven_identity": 32,
        "dynamic second tensor/residual terminates epilogue": 20,
        "chain does not end in exact roundeven/clamp i8 quantize": 2,
    }), dict(histogram))
    check("resnet-native-loop-conv-after-formation",
          cb["params"]["convolution_lowering"]["native_loop_conv_count"] == 0,
          cb["params"]["convolution_lowering"]["native_loop_conv_count"])
    cb["params"].pop("target_neutral_quantized_epilogues")
    check("resnet-normalized-command-buffer",
          stable(cb) == "431727281a0a84db34d0672e542c92e5716a1d778966115a9310f85632d8babc",
          stable(cb))

    canonical_paths = {
        "tinyllama_default": "tinyllama", "lstmnetvit": "lstmnetvit", "smolvla": "smolvla"}
    for candidate_name, canonical_name in canonical_paths.items():
        current = json.loads((BUNDLE / f"validation/portfolio/{candidate_name}/command_buffer.json").read_text())
        current["params"].pop("target_neutral_quantized_epilogues")
        prior = json.loads((CANONICAL / f"validation/native_scalar_ops/{canonical_name}/command_buffer.json").read_text())
        check(f"{candidate_name}-command-buffer", current == prior, "semantic identity")

    object_path = BUNDLE / "validation/portfolio/resnet50/object_build/kernel.o"
    q535 = json.loads(Q535.read_text())
    measured_object = q535["artifact_identity"]["measured_kernel_object_sha256"]
    check("resnet-object-transfer", sha256(object_path) == measured_object ==
          "e251d0e4d508af30f0828dd1394095de6a580c40c961cd6a8efe240ed3310618",
          sha256(object_path))
    check("q535-exact-hardware-receipt",
          q535["status"] == "passed" and q535["measured_metrics"]["cycles"] == 1316619699
          and q535["correctness"]["bad"] == 0,
          {"status": q535["status"], "cycles": q535["measured_metrics"]["cycles"]})
    print("PASS exact native epilogue scoped evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
