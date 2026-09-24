"""A PT2E model is admitted only when the whole quantized contraction region was rewritten."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_source as source

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
