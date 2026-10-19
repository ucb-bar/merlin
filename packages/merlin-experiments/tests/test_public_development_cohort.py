"""Public source/reference spectra retain every budget and semantic refusal."""

import hashlib

import numpy as np
import pytest
from merlin_experiments.phase0 import development_cohort as C
from merlin_experiments.phase0 import development_envelopes as E
from merlin_experiments.phase0 import development_reference as R
from merlin_experiments.phase0 import development_sources as D

from merlin.common.cpu_pool import shared_compute_pool
from merlin.common.paths import repo_root


def cases():
    conv = dict(H=5, W=7, CI=3, CO=5, OUT=7, KH=3, KW=3, SH=1, SW=1, PH=1, PW=1)
    action = dict(prefix=35, hidden=7, action=5, chunk=3, steps=2, decode=2)
    return [
        E.declaration("matmul", dict(M=3, K=17, N=5), ordinal=0),
        E.declaration("matmul_rhs_nk", dict(M=5, K=7, N=3), ordinal=1),
        E.declaration("conv_transition_nhwc", conv, ordinal=2),
        E.declaration(
            "conv_transition_nhwc", {**conv, "KH": 7, "KW": 7, "SH": 2, "SW": 2, "PH": 3, "PW": 3}, ordinal=3
        ),
        E.declaration("conv_nhwc", {key: value for key, value in conv.items() if key != "OUT"}, ordinal=4),
        E.declaration("prefix_action_recurrence", action, ordinal=5),
        E.declaration("vision_prefix_action", {"vision": conv, "action": action}, ordinal=6),
        E.declaration("resident_rhs_reuse", dict(M=[1, 7, 3], K=17, N=5), ordinal=7),
    ]


@pytest.mark.parametrize("declaration", cases(), ids=lambda d: d["name"] + "_" + d["pattern"])
def test_complete_fresh_pytorch_source_matches_independent_integer_reference(declaration):
    pytest.importorskip("torch")
    namespace = {}
    exec(compile(D.render(declaration), "public-development-source", "exec"), namespace)
    with shared_compute_pool(timeout_s=120):
        model, source_inputs = namespace["get_model_and_inputs"]()
        actual = model(*source_inputs)
    values = R.inputs(declaration)
    for source, independent in zip(source_inputs, values, strict=True):
        np.testing.assert_array_equal(source.numpy(), independent)
    expected = R.evaluate(declaration, values)
    assert len(actual) == len(D.describe(declaration)["outputs"])
    for observed, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed.numpy(), wanted)
        assert observed.numpy().dtype == wanted.dtype


def test_public_architecture_plan_contains_actual_widths_and_above_upper_envelopes():
    path = repo_root() / "experiments/reference-data/phase0/public-architectures"
    result = E.derive(path, axis_granules={"M": 16, "K": 16, "N": 16}, resource_sha256="a" * 64, max_spectrum=32)
    assert result["scope"] == "public_architecture_informed_development"
    assert any(row["shape"]["N"] == 32000 for row in result["summary"]["members"])
    assert any(row["shape"]["N"] > 32000 for row in result["spectrum"]["members"] + result["spectrum"]["deferred"])
    assert {row["declaration"]["pattern"] for row in result["requests"] if "declaration" in row} >= {
        "matmul",
        "matmul_rhs_nk",
        "conv_transition_nhwc",
        D.PATTERN,
        "prefix_action_recurrence",
        "vision_prefix_action",
    }
    assert len(result["requests"]) <= 256
    assert sum(row.get("status") == "source_selection_budget_denied" for row in result["requests"]) == (
        2 * len(result["spectrum"]["deferred"])
    )
    assert not result["spectrum"]["claims"]["validation_execution"]
    assert result["sources"][-1]["source_kind"] == "operator_declared_public_development_envelope"


def budgets():
    return dict(
        max_members=128,
        max_source_bytes=131072,
        max_reference_elements=10000,
        max_reference_work=1000000,
        max_total_reference_work=1000000,
        max_total_reference_elements=10000,
    )


def test_materialized_roster_freezes_all_liveouts_and_keeps_expensive_refusals(tmp_path):
    small, large = cases()[0], E.declaration("matmul", dict(M=1, K=2048, N=40000), ordinal=9)
    root = tmp_path / "cohort"
    manifest = C.freeze(
        [{"declaration": small}, {"declaration": large}, {"status": "source_refused", "reason": "unsupported readout"}],
        root,
        provenance={"scope": "public_development"},
        budgets=budgets(),
    )
    assert [row["status"] for row in manifest["members"]] == [
        "source_and_independent_reference_frozen",
        "source_frozen_reference_budget_denied",
        "source_refused",
    ]
    first = manifest["members"][0]
    assert (root / first["loader"]).is_file()
    with np.load(root / first["reference"]) as reference:
        assert reference["output_0"].shape == (3, 5)
        assert set(reference.files) == {"input_0", "input_1", "output_0"}
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    assert C.verify(root, manifest_sha256=digest) == manifest
    assert not manifest["phase0_admitted"] and not manifest["performance_objective_frozen"]
    import json

    coverage = json.loads((root / "coverage.json").read_bytes())
    assert len(coverage["members"]) == 3 and len(coverage["unavailable_members"]) == 2
    loader = root / first["loader"]
    loader.chmod(0o644)
    loader.write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        C.verify(root, manifest_sha256=digest)


def test_aggregate_reference_budget_denial_preserves_entire_requested_denominator(tmp_path):
    requests = [{"declaration": row} for row in cases()[:2]]
    manifest = C.freeze(
        requests, tmp_path / "cohort", provenance={}, budgets={**budgets(), "max_total_reference_work": 1}
    )
    assert len(manifest["members"]) == 2 and manifest["reference_work"] == 0
    assert all(row["status"] == "source_frozen_reference_budget_denied" for row in manifest["members"])


def test_invalid_composition_join_and_overflow_refuse_before_materialization(tmp_path):
    declaration = cases()[6]
    declaration["geometry"]["action"]["hidden"] += 1
    with pytest.raises(ValueError, match="join exactly"):
        D.describe(declaration)
    overflow = E.declaration("matmul", dict(M=1, K=65536, N=3), ordinal=0)
    overflow["arithmetic"]["readout_shift"] = 30
    # Domain proof includes rounding; this remains below the exact i32 ceiling.
    assert D.describe(overflow)["scalar_products"] == 196608


def test_logical_payload_never_mints_physical_liveness_or_complete_roofline():
    from merlin_experiments.phase0.development_coverage import resources

    result = resources(
        D.describe(cases()[0]),
        dict(
            tile_rows=16,
            tile_cols=16,
            operand_bytes=1,
            accumulate_bytes=4,
            operand_store_bytes=1024,
            accumulator_bytes=1024,
            peak_macs_per_cycle=256,
        ),
    )
    assert result["logical_operand_bytes"] == 3 * 17 + 17 * 5
    assert not result["simultaneous_liveness_proven"]
    assert result["roofline"]["status"] == "UNKNOWN" and result["roofline"]["attainment"] is None


@pytest.mark.parametrize("declaration", [cases()[3], cases()[6], cases()[7]], ids=lambda row: row["pattern"])
def test_ordinary_capture_writer_checks_new_connected_source_patterns(tmp_path, declaration):
    from merlin_experiments.phase0 import development_capture as capture

    from merlin.targetgen import capsule_source

    source = capsule_source.PytorchRefSource(timeout=120)
    if not source.available():
        pytest.skip("selected Model2MLIR capture runtime unavailable")
    root = tmp_path / "sources"
    C.freeze([{"declaration": declaration}], root, provenance={}, budgets=budgets())
    report = capture.capture(
        root,
        manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        ordinals=[0],
        output=tmp_path / "capture",
        max_work=1000000,
        max_elements=10000,
    )
    row = report["members"][0]
    assert row["status"] == "complete_capture_and_independent_reference_match"
    assert row["capture_freshness"].startswith("UNKNOWN") and row["independent_reference_execution"] == "fresh"
    assert row["capture_metadata"]["opaque"] == 0
    assert row["capture_metadata"]["materialized_contractions"] == D.describe(declaration)["contractions"]
    assert "capsule.yaml" in row["files"]
