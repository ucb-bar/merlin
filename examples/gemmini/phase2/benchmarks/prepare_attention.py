"""Capture complete public attention for compiler experiments and source baselines.

This supplements the exact integer GEMM/convolution corpus. It preserves the
existing complete-attention source arithmetic, including FP32 PV. It does not
admit a new quantization recipe or selected full-model floating tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from merlin_experiments.phase0 import development_attention as attention
from merlin_experiments.phase0.development_attention_diagnostic import retain

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.capsule_source import PytorchRefSource


def prepare(profile, member, output):
    profile = Path(profile)
    document = yaml.safe_load(profile.read_text())
    if document.get("schema") != "merlin.example.public_phase2_attention.v1":
        raise ValueError("an explicit public attention profile is required")
    matches = [row for row in document["members"] if row["id"] == member]
    if len(matches) != 1:
        raise ValueError("select one uniquely declared attention member")
    g = matches[0]["geometry"]
    declaration = attention.declaration(
        member, heads=g["query_heads"], kv=g["kv_heads"], width=g["head_dim"], value_width=g["value_dim"],
        prefix=g["prefix_tokens"], queries=g["query_tokens"], mask=g["mask"],
    )
    description = attention.describe(declaration)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")

    write("declaration.json", declaration)
    write("source-description.json", description)
    loader = output / "loader.py"
    loader.write_text(attention.render(declaration))
    with shared_compute_pool(timeout_s=300):
        captured = PytorchRefSource(timeout=240).capture_loader(
            loader, "int8", workdir=output / "capture", already_quantized=True,
        )
        inputs = attention.inputs(declaration)
        expected = attention.evaluate(declaration, inputs)
    if captured.meta["opaque"] or len(captured.inputs) != len(inputs):
        raise ValueError("attention capture has opaque operations or changed its input roster")
    for actual, wanted in zip(captured.inputs, inputs, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    if len(captured.golden) != len(expected):
        raise ValueError("attention capture omitted a required output or carried state")
    diagnostics, golden = [], []
    for abi, value, wanted in zip(captured.meta["output_abi"], captured.golden, expected, strict=True):
        if abi["shape"] != list(wanted.shape) or abi["dtype"] != {"int8": "i8", "float32": "f32"}[str(wanted.dtype)]:
            raise ValueError("attention capture changed an output ABI")
        actual = np.asarray(value, dtype=wanted.dtype)
        golden.append(actual)
        if wanted.dtype == np.int8:
            np.testing.assert_array_equal(actual, wanted)
            diagnostics.append({"comparison": "exact_carried_state", "passed": True})
        else:
            if not np.isfinite(actual).all() or not np.isfinite(wanted).all():
                raise ValueError("the public diagnostic produced nonfinite attention outputs")
            diagnostics.append({
                "comparison": "diagnostic_only_no_new_tolerance",
                "bit_equal": actual.tobytes() == wanted.tobytes(),
                "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - wanted), initial=0)),
            })
    closure = retain(description, captured, expected, golden, output / "capture-closure")
    portable_source = (output / "capture-closure/capture.interface.mlir").read_text()
    (output / "source.mlir").write_text(portable_source)
    for name in ("capsule.weights.safetensors", "capsule.weights.safetensors.manifest.json"):
        retained = output / "capture-closure" / name
        if retained.is_file():
            shutil.copyfile(retained, output / name)
    write("capture-meta.json", captured.meta)
    for name, values in (("inputs", inputs), ("captured-reference", golden), ("independent-reference", expected)):
        prefix = "input" if name == "inputs" else "output"
        np.savez(output / (name + ".npz"), **{f"{prefix}_{i}": x for i, x in enumerate(values)})
    record = {
        "schema": "merlin.example.public_phase2_attention_capture.v1", "member": member,
        "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(portable_source.encode()).hexdigest(),
        "source_description": description, "campaign_policy": document["campaign"],
        "capture_closure": closure,
        "complete_outputs": sum(x.size for x in expected), "floating_diagnostics": diagnostics,
        "scope": "complete attention and carried KV; projections and RoPE are outside this source boundary",
        "device_execution": "not_yet_run", "timing": "not_yet_observed",
        "full_attainable_roofline": None, "selected_full_model_qualification": False,
    }
    write("attention-member.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("attention-profile.yaml"))
    parser.add_argument("--member", default="gqa_prefill32_decode1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.profile, args.member, args.output)
    print(json.dumps({key: result[key] for key in ("member", "complete_outputs", "scope")}, indent=2))


if __name__ == "__main__":
    main()
