"""Final issuer guards, with the outer source/materialization boundary isolated.

The cases strengthen sweep3/validator_probe.py with real saved program bytes,
typed producer comparisons, and a digest-bound independent reference artifact.
No capture, replay, or execution attestation is mocked into existence.
"""

import copy
import json

import numpy as np
import pytest
from merlin_experiments.capture_execution import sealed_m2m as S

from merlin.capture.integerization import NON_CONTRACTION_ANNOTATIONS, contraction_partition
from merlin.targetgen.integerization_export import receipt_export_complete


def metadata(contractions=1, *, batched=False, values=None):
    values = values if values is not None else [np.array([1.0], dtype=np.float32)]
    types = {
        "float32": ("torch.float32", "f32"),
        "complex64": ("torch.complex64", "complex<f32>"),
        "float64": ("torch.float64", "f64"),
        "complex128": ("torch.complex128", "complex<f64>"),
    }
    rows = [
        dict(
            finite=bool(np.isfinite(value).all()),
            atol=0,
            rtol=0,
            max_abs=0,
            max_rel=0,
            within_tolerance=True,
            byte_equal=True,
            shape=list(value.shape),
            dtype_before=types[str(value.dtype)][0],
            dtype_after=types[str(value.dtype)][0],
        )
        for value in values
    ]
    counts = {kind: dict(seen=0, integerized=0, remaining=0) for kind in ("linear", "conv2d", "matmul", "unsupported")}
    kind = "matmul" if batched else "linear"
    counts[kind].update(seen=contractions, integerized=contractions)
    agreement = dict(
        status="passed",
        samples=1,
        finite=all(row["finite"] for row in rows),
        atol=0,
        rtol=0,
        max_abs=0,
        max_rel=0,
        outputs=rows,
    )
    agreement.update(
        reference="pt2e_integer",
        executed_contractions={
            key: contractions if key in ("total", "selected", "observed", kind) else 0
            for key in ("total", "selected", "observed", "linear", "conv2d", "matmul")
        },
    )
    receipt = dict(
        schema="m2m.pt2e-integerize.v1",
        quantized_contractions_seen=contractions,
        quantized_contractions_integerized=contractions,
        quantized_contractions_remaining=0,
        quantized_by_kind=counts,
        refusals=[],
        max_reduction_k=contractions,
        accumulator_bound_checked=bool(contractions),
        integer_mm_emitted=0 if batched else contractions,
        exported_integer_mm_count=0 if batched else contractions,
        integer_bmm_emitted=contractions if batched else 0,
        exported_integer_bmm_count=contractions if batched else 0,
        golden_agreement=agreement,
    )
    stats = dict(annotated_contractions=contractions)
    if not contractions:
        annotations = {key: int(key == "annotated_movements") for key in NON_CONTRACTION_ANNOTATIONS}
        stats.update(annotations)
        receipt.update(integer_program="non_contraction", non_contraction_annotations=annotations.copy())
    return dict(
        integerization_receipt=receipt,
        quantization_stats=stats,
        output_abi=[dict(shape=list(value.shape), dtype=types[str(value.dtype)][1]) for value in values],
        recipe=dict(software_numerical_engine="integer_reference"),
    )


def save(stage, meta):
    reference = stage / "integer-reference.json"
    reference.write_text(
        json.dumps(dict(schema="merlin.capture.integer_reference.v1", output_abi=meta["output_abi"], outputs=[]))
    )
    meta["integerization_receipt"]["golden_agreement"]["output"] = dict(
        path=reference.name, sha256=S._file_digest(reference)
    )
    (stage / "meta.json").write_text(json.dumps(meta))
    (stage / "model.mlir").write_text("module { func.func @forward() { return } }")


def run_guard(stage, meta, monkeypatch):
    save(stage, meta)
    monkeypatch.setattr(S, "_materialized_v2", lambda *_: dict(status="verified_materialized"))
    return S._materialized_v3(stage, stage, stage, dict(dtype="int8"))


@pytest.mark.parametrize("case", ["non_contraction", "exact_nonfinite_output", "bmm_only", "complex", "mixed"])
def test_sweep3_probes_with_complete_evidence(tmp_path, monkeypatch, case):
    values = [np.array([float("inf"), float("nan")], dtype=np.float32)] if case == "exact_nonfinite_output" else None
    if case == "complex":
        values = [np.array([1 + 2j, complex(3, float("nan"))], dtype=np.complex64)]
    meta = metadata(0 if case == "non_contraction" else 1, batched=case == "bmm_only", values=values)
    if case == "mixed":
        meta["integerization_receipt"].update(integer_bmm_emitted=2, exported_integer_bmm_count=2)
    receipt = meta["integerization_receipt"]
    assert receipt_export_complete(receipt, contraction_partition(receipt))
    assert run_guard(tmp_path, meta, monkeypatch)["integer_contractions"] == (
        0 if case == "non_contraction" else 3 if case == "mixed" else 1
    )


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["integerization_receipt"].pop("integer_program"),
        lambda m: m["quantization_stats"].update(annotated_movements=2),
        lambda m: m["quantization_stats"].update(annotated_movements=True),
        lambda m: m["integerization_receipt"].update(exported_integer_mm_count=1),
        lambda m: m["integerization_receipt"].update(integer_bmm_emitted=1, exported_integer_bmm_count=1),
        lambda m: m["integerization_receipt"].update(refusals=[{"reason": "unresolved"}]),
        lambda m: m["integerization_receipt"]["quantized_by_kind"].pop("linear"),
        lambda m: m["integerization_receipt"]["quantized_by_kind"]["linear"].update(seen=False),
        lambda m: m["integerization_receipt"].update(quantized_contractions_seen=False),
        lambda m: m["integerization_receipt"]["golden_agreement"]["executed_contractions"].update(total=1),
        lambda m: m["integerization_receipt"].update(max_reduction_k=1),
        lambda m: m["integerization_receipt"].update(precision_decisions=[{"decision": "preserve_float_qdq"}]),
        lambda m: m["integerization_receipt"].update(precision_decision_counts={"unresolved": 1}),
    ],
)
def test_zero_partition_cannot_waive_missing_or_contradictory_evidence(tmp_path, monkeypatch, edit):
    meta = metadata(0)
    edit(meta)
    with pytest.raises(S.SealedM2MError):
        run_guard(tmp_path, meta, monkeypatch)


def test_non_contraction_guard_parses_saved_integer_work(tmp_path, monkeypatch):
    meta = metadata(0)
    save(tmp_path, meta)
    (tmp_path / "model.mlir").write_text("""module {
      func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>, %c: tensor<2x4xi32>) -> tensor<2x4xi32> {
        %r = linalg.matmul ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%c : tensor<2x4xi32>) -> tensor<2x4xi32>
        return %r : tensor<2x4xi32>
      }
    }""")
    monkeypatch.setattr(S, "_materialized_v2", lambda *_: dict(status="verified_materialized"))
    with pytest.raises(S.SealedM2MError, match="contains integer contractions"):
        S._materialized_v3(tmp_path, tmp_path, tmp_path, dict(dtype="int8"))


@pytest.mark.parametrize(
    "key,value",
    [
        ("integer_bmm_emitted", -1),
        ("integer_bmm_emitted", True),
        ("integer_bmm_emitted", 0),
        ("exported_integer_bmm_count", 2),
        ("accumulator_bound_checked", False),
        ("max_reduction_k", 0),
        ("max_reduction_k", 131072),
        ("integer_mm_emitted", 1),
    ],
)
def test_bmm_keeps_counts_accumulator_and_k_guards(tmp_path, monkeypatch, key, value):
    meta = metadata(batched=True)
    meta["integerization_receipt"][key] = value
    with pytest.raises(S.SealedM2MError):
        run_guard(tmp_path, meta, monkeypatch)


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: m["integerization_receipt"]["golden_agreement"]["outputs"][0].update(byte_equal=False),
        lambda m: m["integerization_receipt"]["golden_agreement"]["outputs"][0].pop("byte_equal"),
        lambda m: m["integerization_receipt"]["golden_agreement"]["outputs"][0].update(atol=0.1),
        lambda m: m["integerization_receipt"]["golden_agreement"].update(rtol=0.1),
        lambda m: m["integerization_receipt"]["golden_agreement"]["outputs"][0].update(max_rel=1),
        lambda m: m["integerization_receipt"]["golden_agreement"]["outputs"][0].update(dtype_after="torch.float64"),
        lambda m: m["integerization_receipt"]["golden_agreement"]["outputs"][0].update(shape=[2]),
        lambda m: m["output_abi"].append(copy.deepcopy(m["output_abi"][0])),
        lambda m: m["integerization_receipt"]["golden_agreement"]["outputs"][0].update(within_tolerance=False),
        lambda m: m["integerization_receipt"]["golden_agreement"].update(reference="portable_pt2e"),
        lambda m: m["integerization_receipt"]["golden_agreement"].update(finite=True),
    ],
)
def test_special_outputs_require_exact_complete_typed_comparisons(tmp_path, monkeypatch, edit):
    meta = metadata(values=[np.array([float("nan")], dtype=np.float32)])
    edit(meta)
    with pytest.raises(S.SealedM2MError):
        run_guard(tmp_path, meta, monkeypatch)


@pytest.mark.parametrize("change", ["digest", "abi", "symlink"])
def test_special_reference_must_remain_bound(tmp_path, monkeypatch, change):
    meta = metadata(values=[np.array([float("nan")], dtype=np.float32)])
    save(tmp_path, meta)
    reference = tmp_path / "integer-reference.json"
    if change == "digest":
        reference.write_text(reference.read_text() + " ")
    elif change == "abi":
        reference.write_text(json.dumps(dict(schema="merlin.capture.integer_reference.v1", output_abi=[])))
        meta["integerization_receipt"]["golden_agreement"]["output"]["sha256"] = S._file_digest(reference)
        (tmp_path / "meta.json").write_text(json.dumps(meta))
    else:
        target = tmp_path / "indirect"
        reference.rename(target)
        reference.symlink_to(target)
    monkeypatch.setattr(S, "_materialized_v2", lambda *_: dict(status="verified_materialized"))
    with pytest.raises(S.SealedM2MError, match="special/complex"):
        S._materialized_v3(tmp_path, tmp_path, tmp_path, dict(dtype="int8"))


@pytest.mark.parametrize("complex_values", [False, True])
def test_nan_payload_words_are_compared_bitwise_through_issuer(tmp_path, monkeypatch, complex_values):
    torch = pytest.importorskip("torch")

    from merlin.targetgen._m2m_capture_worker import _integerized_agreement

    words = torch.tensor([0x7FC00001, 0xFFC00023, 0x7F800000, 0x80000000], dtype=torch.uint32)
    left = words.view(torch.complex64 if complex_values else torch.float32)
    meta = metadata(values=[left.numpy()])
    assert run_guard(tmp_path, meta, monkeypatch)["integer_contractions"] == 1
    changed = words.clone()
    changed[1 if complex_values else 0] = 0x7FC00002
    right = changed.view(left.dtype)
    comparison = _integerized_agreement([left], [right], atol=0, rtol=0)
    assert comparison["status"] == "failed"
    assert comparison["outputs"][0]["byte_equal"] is False
    assert comparison["finite"] is False
    meta["integerization_receipt"]["golden_agreement"].update(comparison)
    with pytest.raises(S.SealedM2MError):
        run_guard(tmp_path, meta, monkeypatch)


def test_complex_diagnostics_keep_imaginary_errors_and_zero_signs():
    torch = pytest.importorskip("torch")

    from merlin.targetgen._m2m_capture_worker import _integerized_agreement

    comparison = _integerized_agreement(torch.tensor([1 + 2j]), torch.tensor([1 + 4j]), atol=0, rtol=0)
    assert comparison["status"] == "failed"
    assert comparison["max_abs"] == 2
    assert comparison["max_rel"] > 0
    left = torch.tensor([complex(0.0, -0.0)], dtype=torch.complex128)
    right = torch.tensor([complex(-0.0, 0.0)], dtype=torch.complex128)
    comparison = _integerized_agreement(left, right, atol=0, rtol=0)
    assert comparison["status"] == "failed"
    assert comparison["outputs"][0]["signed_zero_differences"] == 2
    comparison = _integerized_agreement(
        torch.tensor([complex(1, float("inf"))]), torch.tensor([complex(1, float("inf"))]), atol=0, rtol=0
    )
    assert comparison["status"] == "passed" and comparison["finite"] is False


def test_special_policy_checks_every_leaf_including_finite_ones(tmp_path, monkeypatch):
    meta = metadata(values=[np.array([float("nan")], dtype=np.float32), np.array([1.0], dtype=np.float32)])
    assert run_guard(tmp_path, meta, monkeypatch)["integer_contractions"] == 1
    meta["integerization_receipt"]["golden_agreement"]["outputs"][1]["byte_equal"] = False
    with pytest.raises(S.SealedM2MError):
        run_guard(tmp_path, meta, monkeypatch)


def test_zero_partition_does_not_skip_declared_dynamic_work(tmp_path, monkeypatch):
    meta = metadata(0)
    meta["quantization_stats"]["dynamic_contractions"] = dict(sites=[{"status": "rewritten"}])
    with pytest.raises(S.SealedM2MError, match="dynamic contraction work"):
        run_guard(tmp_path, meta, monkeypatch)


def test_zero_partition_accepts_only_complete_empty_dynamic_domain(tmp_path, monkeypatch):
    meta = metadata(0)
    meta["quantization_stats"]["dynamic_contractions"] = dict(
        sites=[],
        rewritten=0,
        kept_in_float=[],
        device_calls=0,
        host_elements=0,
        int8_macs=0,
    )
    save(tmp_path, meta)
    trace = dict(
        graphs={"prepared": {"nodes": []}}, mlir={"sha256": S._file_digest(tmp_path / "model.mlir"), "operations": []}
    )
    (tmp_path / "frontend-trace.json").write_text(json.dumps(trace))
    assert run_guard(tmp_path, meta, monkeypatch)["integer_contractions"] == 0
    trace["graphs"]["prepared"]["nodes"].append(dict(id="unemitted", target="aten._int_mm.default"))
    (tmp_path / "frontend-trace.json").write_text(json.dumps(trace))
    with pytest.raises(S.SealedM2MError, match="exactly once"):
        run_guard(tmp_path, meta, monkeypatch)


def test_batched_counts_cannot_hide_unexplained_matrix_work(tmp_path, monkeypatch):
    meta = metadata(batched=True)
    meta["integerization_receipt"].update(exported_integer_mm_count=1, exported_integer_bmm_count=0)
    with pytest.raises(S.SealedM2MError):
        run_guard(tmp_path, meta, monkeypatch)
