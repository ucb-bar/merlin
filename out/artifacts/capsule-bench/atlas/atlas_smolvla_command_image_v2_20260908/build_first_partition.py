#!/usr/bin/env python3
"""Compile the first target-compatible SmolVLA addmm partition."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARTITION = ROOT / "partitions/first_addmm_matmul_0"
TOOL = ROOT / "submission/mlir_oot/atlas-opt"
INTERFACE = PARTITION / "interface.mlir"
ASSEMBLY = PARTITION / "kernel.S"
COMMAND_BUFFER = PARTITION / "command_buffer.json"

proc = subprocess.run(
    [
        str(TOOL),
        f"--emit-command-buffer={COMMAND_BUFFER}",
        "--emit-target-artifact",
        str(INTERFACE),
    ],
    capture_output=True,
    text=True,
    timeout=60,
)
ASSEMBLY.write_text(proc.stdout, encoding="utf-8")
(PARTITION / "compile.stderr").write_text(proc.stderr, encoding="utf-8")
words = sum(line.lstrip().startswith(".word") for line in proc.stdout.splitlines())
command_buffer = json.loads(COMMAND_BUFFER.read_text()) if COMMAND_BUFFER.is_file() else None
receipt = {
    "schema": "atlas_smolvla_partition_compile_v1",
    "claim": "compile-only partition; no numeric or whole-model execution claim",
    "capture_regions": ["matmul_0", "add_3"],
    "capture_semantic": "addmm",
    "capture_shapes": {
        "activation": [1024, 768],
        "weight_after_aot_transpose": [768, 768],
        "bias": [768],
        "output": [1024, 768],
    },
    "device_quantization": {
        "activation": "fp8_e4m3",
        "weight": "fp8_e4m3",
        "bias": "bf16",
        "accumulator": "bf16",
        "output": "bf16",
        "qualified_against_full_model": False,
    },
    "returncode": proc.returncode,
    "instruction_words": words,
    "imem_words": 32768,
    "fits_imem": words <= 32768,
    "command_count": len(command_buffer["commands"]) if command_buffer else None,
    "command_opcodes": [c["opcode"] for c in command_buffer["commands"]]
    if command_buffer
    else [],
    "interface_sha256": hashlib.sha256(INTERFACE.read_bytes()).hexdigest(),
    "assembly_sha256": hashlib.sha256(ASSEMBLY.read_bytes()).hexdigest(),
}
(PARTITION / "partition_manifest.json").write_text(
    json.dumps(
        {
            "schema": "atlas_smolvla_concrete_partition_v1",
            "claim": "one compile-only device partition selected from the full capture",
            "capture": {
                "path": str(
                    ROOT.parents[4]
                    / "out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir"
                ),
                "sha256": "27256e2414be16eb0f783a547e79102c9b31afad76cbfd13c2f4b421ec7d8e4c",
            },
            "selection": {
                "kind": "first target-compatible physical contraction after the unsupported patch convolution",
                "source_candidate_window": {
                    "index": 10,
                    "first_region": "view_5",
                    "last_region": "view_16",
                    "logical_region_count": 21,
                },
                "fused_capture_regions": ["matmul_0", "add_3"],
                "provenance_fqn": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.self_attn.q_proj",
                "semantic": "aten.addmm.default",
            },
            "capture_boundary": {
                "inputs": [
                    {
                        "name": "A0",
                        "capture_value": "%929",
                        "capture_type": "tensor<1024x768xf32>",
                        "producer_region": "view_5",
                        "upstream_region": "layer_norm_0",
                        "device_type": "tensor<1024x768xf8E4M3FN>",
                    },
                    {
                        "name": "W",
                        "capture_value": "%931",
                        "capture_type": "tensor<768x768xf32>",
                        "producer": "unattributed linalg.transpose of function argument %7",
                        "host_aot_preprocess": "transpose permutation [1, 0], then quantize to fp8_e4m3",
                        "device_type": "tensor<768x768xf8E4M3FN>",
                    },
                    {
                        "name": "B",
                        "capture_value": "%8",
                        "capture_type": "tensor<768xf32>",
                        "producer": "function argument %8",
                        "host_aot_preprocess": "quantize to bf16",
                        "device_type": "tensor<768xbf16>",
                    },
                ],
                "outputs": [
                    {
                        "name": "Y0",
                        "capture_value": "%937",
                        "capture_type": "tensor<1024x768xf32>",
                        "producer_region": "add_3",
                        "next_consumer_region": "view_6",
                        "device_type": "tensor<1024x768xbf16>",
                    }
                ],
                "host_device_edges": {"read": 3, "write": 1},
            },
            "image": {
                "instruction_words": words,
                "measurement": "exact emitted .word count",
                "imem_words": 32768,
                "fits_imem": words <= 32768,
                "command_count": len(command_buffer["commands"]),
                "command_opcodes": [c["opcode"] for c in command_buffer["commands"]],
            },
            "remaining_proofs": [
                "derive full-model FP8/BF16 calibration rather than merely choosing storage types",
                "grade this partition numerically against the captured f32 region",
                "schedule the other six mesh operations in the source candidate window",
                "connect view_5/view_6 without a redundant materialization",
            ],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
(PARTITION / "compile_receipt.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(receipt, indent=2, sort_keys=True))
raise SystemExit(proc.returncode)
