"""Complete attention semantics retain causal/GQA state and missing numeric gates."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from merlin_experiments.phase0 import development_attention as A
from merlin_experiments.phase0 import development_capture as C
from merlin_experiments.phase0 import development_cohort as F
from merlin_experiments.phase0 import development_envelopes as E

from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.paths import repo_root


def small(**kwargs):
    return A.declaration("complete_attention", heads=6, kv=2, width=7, value_width=5, prefix=1, queries=[1], **kwargs)


def eager(declaration):
    pytest.importorskip("torch")
    namespace = {}
    exec(compile(A.render(declaration), "public-complete-attention", "exec"), namespace)
    model, inputs = namespace["get_model_and_inputs"]()
    with shared_compute_pool(timeout_s=120):
        result = model(*inputs)
    return model, inputs, result


def test_exact_uniform_softmax_case_matches_independent_reference_without_any_tolerance():
    declaration = small()
    model, inputs, _ = eager(declaration)
    inputs = list(inputs)
    inputs[2] = inputs[2].clone().zero_()
    with shared_compute_pool(timeout_s=120):
        actual = model(*inputs)
    wanted = A.evaluate(declaration, [x.numpy() for x in inputs])
    for a, w in zip(actual, wanted, strict=True):
        np.testing.assert_array_equal(a.numpy(), w)
    description = A.describe(declaration)
    assert description["contractions"] == 6 and description["logical_contractions"] == 12
    assert not description["score_pressure"]["intermediate_scores_required_liveouts"]
    assert all("score" not in row["name"] and "probability" not in row["name"] for row in description["outputs"])
    assert description["pending_numeric_policy"]["outputs"]["attention0"]["compare"] == "UNKNOWN"


def test_causal_mask_prevents_future_value_effects_and_preserves_grouped_head_mapping():
    declaration = A.declaration("causal_groups", heads=6, kv=2, width=7, value_width=5, prefix=0, queries=[3])
    model, inputs, _ = eager(declaration)
    inputs = list(inputs)
    assert [row["name"] for row in A.describe(declaration)["inputs"]] == ["q0", "k0", "v0"]
    inputs[0] = inputs[0].clone().zero_()
    with shared_compute_pool(timeout_s=120):
        original = model(*inputs)
        changed = [x.clone() for x in inputs]
        changed[2][0, 2] += 2
        after = model(*changed)
    np.testing.assert_array_equal(original[0][:, :2].numpy(), after[0][:, :2].numpy())
    np.testing.assert_array_equal(original[0][3:].numpy(), after[0][3:].numpy())
    assert not np.array_equal(original[0][:3, 2].numpy(), after[0][:3, 2].numpy())


@pytest.mark.parametrize("mask", ["causal", "explicit_boolean", "none"])
def test_signed_source_inputs_and_complete_independent_output_roster(mask):
    declaration = A.declaration("mask_pattern", heads=6, kv=2, width=17, prefix=3, queries=[5, 1], mask=mask)
    _, inputs, result = eager(declaration)
    independent = A.inputs(declaration)
    for a, b in zip(inputs, independent, strict=True):
        np.testing.assert_array_equal(a.numpy(), b)
    reference = A.evaluate(declaration, independent)
    for row, a, b in zip(A.describe(declaration)["outputs"], result, reference, strict=True):
        assert list(a.shape) == row["shape"] == list(b.shape)
        assert a.numpy().dtype == b.dtype and np.isfinite(a.numpy()).all() and np.isfinite(b).all()
        if row["dtype"] == "int8":
            np.testing.assert_array_equal(a.numpy(), b)
    # Floating errors are diagnostic; no selected tolerance is invented here.
    assert len(reference) == 4


def test_public_complete_attention_envelope_keeps_prefill_decode_and_capacity_scale():
    requests = E.complete_attention_requests(repo_root() / "experiments/reference-data/phase0/public-architectures")
    assert len(requests) == 10
    descriptions = [A.describe(row["declaration"]) for row in requests]
    assert any(d["declaration"]["geometry"]["query_tokens"] == [32, 1, 1] for d in descriptions)
    assert max(d["score_pressure"]["materialized_fp32_score_bytes"] for d in descriptions) > 64 * 1024 * 1024
    assert all(d["pending_contract"] for d in descriptions)


def test_ordinary_capture_retains_pending_attention_instead_of_inventing_float_policy(tmp_path):
    from merlin.targetgen.capsule_source import PytorchRefSource

    if not PytorchRefSource().available():
        pytest.skip("selected Model2MLIR capture runtime unavailable")
    declaration = small()
    root = tmp_path / "sources"
    manifest = F.freeze(
        [{"declaration": declaration}],
        root,
        provenance={},
        budgets=dict(
            max_members=4,
            max_source_bytes=65536,
            max_reference_elements=10000,
            max_reference_work=100000,
            max_total_reference_work=100000,
            max_total_reference_elements=10000,
        ),
    )
    assert manifest["members"][0]["pending_contract"]
    report = C.capture(
        root,
        manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        ordinals=[0],
        output=tmp_path / "capture",
        max_work=100000,
        max_elements=10000,
    )
    row = report["members"][0]
    assert row["status"] == "complete_attention_capture_reference_diagnostic_contract_pending", row
    assert Path(row["pending_capsule"]).is_file()
    assert row["floating_diagnostics"][0]["comparison"].startswith("UNKNOWN")
    assert not (tmp_path / "capture/capsules").exists()
