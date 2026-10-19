"""Blocked preparation preserves exact seeds, full outputs and old refusals."""

import hashlib

import numpy as np
import pytest
from merlin_experiments.phase0 import development_blocked_reference as B
from merlin_experiments.phase0 import development_cohort as F
from merlin_experiments.phase0 import development_envelopes as E
from merlin_experiments.phase0 import development_parameter_stress as P
from merlin_experiments.phase0 import development_reference as R
from merlin_experiments.phase0 import development_reference_supplement as S
from merlin_experiments.phase0 import development_sources as D


def cases():
    conv = dict(H=5, W=7, CI=3, CO=5, KH=3, KW=3, SH=2, SW=2, PH=1, PW=1)
    grouped = E.declaration(
        D.PATTERN, dict(steps=2, prefix_tokens=3, query_tokens=2, kv_heads=2, query_heads=6, head_dim=7), ordinal=6
    )
    grouped.update(schema=D.SCHEMA, max_logical_elements=1000000)
    return [
        E.declaration("matmul", dict(M=7, K=17, N=11), ordinal=0),
        E.declaration("matmul_rhs_nk", dict(M=7, K=17, N=11), ordinal=1),
        E.declaration("conv_nhwc", conv, ordinal=2),
        E.declaration("conv_transition_nhwc", {**conv, "OUT": 7}, ordinal=3),
        E.declaration("conv_transition_nhwc", {**conv, "OUT": 7, "KH": 7, "KW": 7, "PH": 3, "PW": 3}, ordinal=4),
        E.declaration("resident_rhs_reuse", dict(M=[1, 7, 3], K=17, N=11), ordinal=5),
        grouped,
    ]


@pytest.mark.parametrize("declaration", cases(), ids=lambda row: row["pattern"])
def test_every_blocked_output_is_exact_against_independent_full_reference(tmp_path, declaration):
    description = D.describe(declaration)
    result = B.materialize(declaration, description, tmp_path / "member", block=dict(M=3, K=5, N=4), timeout_s=10)
    for path, wanted in zip(result["outputs"], R.evaluate(declaration, R.inputs(declaration)), strict=True):
        actual = np.load(path)
        np.testing.assert_array_equal(actual, wanted)
        assert actual.dtype == wanted.dtype
    assert result["computed_scalar_products"] == description["scalar_products"]


def test_supplement_retains_original_members_and_denials_without_modifying_source(tmp_path):
    declaration = cases()[1]
    source = tmp_path / "source"
    manifest = F.freeze(
        [{"declaration": declaration}, {"status": "source_refused", "reason": "selected readout absent"}],
        source,
        provenance={},
        budgets=dict(
            max_members=4,
            max_source_bytes=65536,
            max_reference_elements=1,
            max_reference_work=1,
            max_total_reference_work=1,
            max_total_reference_elements=1,
        ),
    )
    assert manifest["members"][0]["status"] == "source_frozen_reference_budget_denied"
    pin = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    report = S.supplement(
        source,
        manifest_sha256=pin,
        output=tmp_path / "supplement",
        ordinals=[0],
        block=dict(M=3, N=4, K=5),
        budgets=dict(
            max_workspace_bytes=8 * 1024**2,
            max_output_bytes=10000,
            max_total_output_bytes=10000,
            max_member_work=100000,
            max_total_work=100000,
            max_dense_source_bytes=1,
            member_timeout_s=10,
            total_timeout_s=30,
        ),
    )
    assert len(report["members"]) == 2
    first = report["members"][0]
    assert first["status"] == "complete_exact_reference_materialized"
    assert first["cost_admission"]["dense_source_memory"].startswith("REFUSED")
    assert report["members"][1]["original_reason"] == "selected readout absent"
    assert F.verify(source, manifest_sha256=pin) == manifest
    assert not report["phase0_admitted"] and not report["performance_objective_changed"]
    index = tmp_path / "supplement/index.json"
    index_pin = hashlib.sha256(index.read_bytes()).hexdigest()
    assert S.verify(index, index_sha256=index_pin, sources=source, manifest_sha256=pin) == report
    output = tmp_path / "supplement" / first["outputs"][0]["path"]
    output.chmod(0o644)
    output.write_bytes(b"changed")
    with pytest.raises(ValueError, match="output bytes"):
        S.verify(index, index_sha256=index_pin, sources=source, manifest_sha256=pin)


@pytest.mark.parametrize("width,depth,shared", [(17, 4, False), (23, 3, True), (31, 4, True)])
def test_parameter_chain_preserves_prior_state_and_every_parameter_use(tmp_path, width, depth, shared):
    declaration = P.declaration("bounded_parameter_chain", width=width, depth=depth, shared=shared)
    description = P.describe(declaration)
    result = B.materialize(declaration, description, tmp_path / "member", block=dict(M=1, N=4, K=5), timeout_s=10)
    for path, expected in zip(result["outputs"], P.evaluate(declaration, P.inputs(declaration)), strict=True):
        actual = np.load(path)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
    assert result["computed_scalar_products"] == depth * width**2
    admission = B.plan(
        description,
        block=dict(M=1, N=4, K=5),
        max_workspace_bytes=8 * 1024**2,
        max_output_bytes=100000,
        max_dense_source_bytes=16 * width**2 - 1,
    )
    assert admission["dense_source_seed_peak_lower_bytes"] == 16 * width**2
    assert admission["dense_source_memory"].startswith("REFUSED")
