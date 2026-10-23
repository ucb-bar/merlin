"""A PT2E model is admitted only when the whole quantized contraction region was rewritten."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.capture.integerization import contraction_partition
from merlin.targetgen import capsule_source as source
from merlin.targetgen.integerization_export import receipt_export_complete

_ONE_INTEGER_MM = """builtin.module {
  func.func @forward(%a: tensor<2x4xi8>, %b: tensor<4x2xi8>) -> tensor<2x2xi32> {
    %z = tensor.empty() : tensor<2x2xi32>
    %r = linalg.matmul ins(%a, %b : tensor<2x4xi8>, tensor<4x2xi8>)
      outs(%z : tensor<2x2xi32>) -> tensor<2x2xi32>
    func.return %r : tensor<2x2xi32>
  }
}"""


def _receipt(*, seen: int = 1, rewritten: int = 1, remaining: int = 0, status: str = "passed") -> dict:
    return {
        "schema": "m2m.pt2e-integerize.v1",
        "quantized_contractions_seen": seen,
        "quantized_contractions_integerized": rewritten,
        "quantized_contractions_remaining": remaining,
        "quantized_by_kind": {
            "linear": {"seen": seen, "integerized": rewritten, "remaining": remaining},
            "conv2d": {"seen": 0, "integerized": 0, "remaining": 0},
            "matmul": {"seen": 0, "integerized": 0, "remaining": 0},
            "unsupported": {"seen": 0, "integerized": 0, "remaining": 0},
        },
        "integer_mm_emitted": 1,
        "exported_integer_mm_count": 1,
        "max_reduction_k": 4,
        "accumulator_bound_checked": True,
        "refusals": [],
        "golden_agreement": {
            "status": status,
            "samples": 1,
            "atol": 0.03125,
            "rtol": 0.02,
            "max_abs": 0.001,
            "max_rel": 0.001,
            "outputs": [
                {
                    "max_abs": 0.001,
                    "max_rel": 0.001,
                    "within_tolerance": status == "passed",
                    "atol": 0.03125,
                    "rtol": 0.02,
                }
            ],
        },
    }


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda r: r.update({"quantized_contractions_seen": 2}), "census"),
        (
            lambda r: (
                r.update({"quantized_contractions_seen": 2, "quantized_contractions_remaining": 1}),
                r["quantized_by_kind"]["linear"].update({"seen": 2, "remaining": 1}),
            ),
            "remaining",
        ),
        (lambda r: r.update({"refusals": [{"kind": "conv2d", "reason": "unsupported"}]}), "refusal"),
        (lambda r: r.update({"accumulator_bound_checked": False}), "accumulator"),
        (lambda r: r.update({"max_reduction_k": 131072}), "accumulator"),
        (lambda r: r["golden_agreement"].update({"status": "failed"}), "agreement"),
        (lambda r: r["golden_agreement"].update({"atol": 1.0}), "tolerance"),
        (lambda r: r.update({"integer_mm_emitted": 2}), "export"),
        (lambda r: r.update({"exported_integer_mm_count": 2}), "export"),
        (lambda r: r["golden_agreement"]["outputs"][0].update({"rtol": 1.0}), "tolerance"),
    ],
)
def test_static_pt2e_receipt_rejects_partial_or_unproved_capture(mutate, reason):
    receipt = _receipt()
    mutate(receipt)
    with pytest.raises(source.M2MUnavailable, match=reason):
        source._require_pt2e_integerization_receipt(
            _ONE_INTEGER_MM,
            {
                "integerization_receipt": receipt,
                "quantization_stats": {"annotated_contractions": receipt["quantized_contractions_seen"]},
            },
            agreement_tolerance=(0.03125, 0.02),
        )


def test_static_pt2e_receipt_accepts_complete_proof():
    source._require_pt2e_integerization_receipt(
        _ONE_INTEGER_MM,
        {"integerization_receipt": _receipt(), "quantization_stats": {"annotated_contractions": 1}},
        agreement_tolerance=(0.03125, 0.02),
    )


def test_worker_accepts_v1_receipt_without_batched_matmul_field():
    receipt = _receipt()
    receipt["exported_integer_bmm_count"] = 0
    assert "integer_bmm_emitted" not in receipt
    assert receipt_export_complete(receipt, contraction_partition(receipt))


def test_worker_does_not_accept_unreported_batched_matmul():
    receipt = _receipt()
    receipt["exported_integer_bmm_count"] = 1
    assert not receipt_export_complete(receipt, contraction_partition(receipt))
    receipt["integer_bmm_emitted"] = 1
    assert receipt_export_complete(receipt, contraction_partition(receipt))


def test_preserved_qdq_partition_requires_complete_typed_accounting():
    receipt = _receipt(seen=2, rewritten=1, remaining=1)
    receipt["precision_decisions"] = [
        {"kind": "linear", "node": "int_linear", "decision": "integerized_i32"},
        {
            "kind": "linear",
            "node": "bf16_linear",
            "decision": "preserve_float_qdq",
            "source_dtype": "torch.bfloat16",
            "required_numeric_semantics": "dequantize-before-floating-contraction",
        },
    ]
    receipt["precision_decision_counts"] = {"integerized_i32": 1, "preserve_float_qdq": 1, "unresolved": 0}
    receipt["refusals"] = [{"kind": "linear", "node": "bf16_linear", "reason": "non-f32 Q/DQ precision"}]
    assert contraction_partition(receipt) == {"seen": 2, "integerized": 1, "preserved": 1}
    receipt["precision_decisions"][1]["source_dtype"] = "torch.float32"
    with pytest.raises(ValueError, match="non-FP32"):
        contraction_partition(receipt)
    receipt["precision_decisions"][1]["source_dtype"] = "torch.bfloat16"
    receipt["refusals"].append({"kind": "linear", "node": "hidden", "reason": "unsupported layout"})
    with pytest.raises(ValueError, match="exact preserved"):
        contraction_partition(receipt)
    receipt["refusals"].pop()
    receipt["precision_decisions"][1]["decision"] = "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        contraction_partition(receipt)


def test_selected_integer_reference_requires_exact_byte_bound_proof(tmp_path: Path):
    reference = tmp_path / "integer-reference.json"
    reference.write_text('{"outputs": [1]}\n')
    receipt = _receipt()
    proof = receipt["golden_agreement"]
    proof.update(
        reference="pt2e_integer",
        atol=0.0,
        rtol=0.0,
        max_abs=0.0,
        max_rel=0.0,
        source={"path": "pt2e_integer_reference.py", "sha256": "a" * 64},
        output={"path": reference.name, "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()},
        executed_contractions={"conv2d": 0, "linear": 1, "matmul": 0, "total": 1, "selected": 1, "observed": 1},
    )
    proof["outputs"][0].update(atol=0.0, rtol=0.0, max_abs=0.0, max_rel=0.0)
    meta = {
        "recipe": {"software_numerical_engine": "integer_reference"},
        "integerization_receipt": receipt,
        "quantization_stats": {"annotated_contractions": 1},
    }
    source._require_pt2e_integerization_receipt(
        _ONE_INTEGER_MM,
        meta,
        agreement_tolerance=(0.03125, 0.02),
        capture_root=tmp_path,
    )
    reference.write_text('{"outputs": [2]}\n')
    with pytest.raises(source.M2MUnavailable, match="artifact does not match"):
        source._require_pt2e_integerization_receipt(
            _ONE_INTEGER_MM,
            meta,
            agreement_tolerance=(0.03125, 0.02),
            capture_root=tmp_path,
        )


def test_static_pt2e_receipt_cannot_hide_an_annotated_contraction():
    with pytest.raises(source.M2MUnavailable, match="PT2E's 2 annotated contractions"):
        source._require_pt2e_integerization_receipt(
            _ONE_INTEGER_MM,
            {"integerization_receipt": _receipt(), "quantization_stats": {"annotated_contractions": 2}},
            agreement_tolerance=(0.03125, 0.02),
        )


def test_w8a8_model_fresh_capture_rejects_missing_receipt_even_with_integer_mm(tmp_path: Path, monkeypatch):
    """Exercise the actual cache-admission boundary, not only the validator."""
    payload = {
        "ok": True,
        "opaque": 0,
        "scheme": None,
        "capture_abi_version": source._MODEL_CAPTURE_ABI_VERSION,
        "input_abi": [],
        "output_abi": [{"shape": [2, 2], "dtype": "i32"}],
        "loader_provenance_status": "declared",
        "loader_dependency_sources": [],
    }

    def worker(cmd, **_kw):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "linalg.mlir").write_text(_ONE_INTEGER_MM)
        (out / "inputs.json").write_text("[]")
        (out / "golden.json").write_text("[]")
        (out / "meta.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(source.subprocess, "run", worker)
    loader = tmp_path / "loader.py"
    loader.write_text("# synthetic loader\n")
    capture = source.PytorchRefSource(m2m_dir=tmp_path / "upstream", python=tmp_path / "python")
    monkeypatch.setattr(type(capture), "available", lambda self: True)
    monkeypatch.setattr(type(capture), "_cache_slot", lambda *args, **kwargs: None)
    recipe = {"status": "derived", "activation": {"dtype": "int8", "mode": "static"}, "weight": {"dtype": "int8"}}
    with pytest.raises(source.M2MUnavailable, match="integerization receipt"):
        capture.capture_loader(
            loader, "i8", workdir=tmp_path / "capture", recipe=recipe, agreement_tolerance=(0.03125, 0.02)
        )
