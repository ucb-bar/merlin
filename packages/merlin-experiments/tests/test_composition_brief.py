"""Only pinned applied evidence becomes compact public feedback."""

import hashlib
import json

from merlin_experiments.phase1.feedback.composition_brief import build


def pin(path, record):
    path.write_text(json.dumps(record))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def record(tmp_path, *, report=None, role="tensor_preparation_sharing"):
    preparation = {} if report is None else {role: pin(tmp_path / "tensor-preparation-cse.json", report)}
    return {
        "source_composition_selection": {
            "preferences": {
                "prepack_weights": False,
                "shared_input_siblings": True,
                "share_exact_tensor_preparations": True,
                "dense_input_views": True,
            }
        },
        "build": {
            "inputs": {"share_exact_tensor_preparations": True, "dense_input_views": True},
            "output": {"compilation_recipe": pin(tmp_path / "recipe.json", {"preparation": preparation})},
        },
    }


def test_selection_is_not_application_and_missing_is_not_zero(tmp_path):
    result = build(record(tmp_path))
    assert result["features"]["dense_input_views"]["status"] == "applied_evidence_unavailable"
    assert "effects" not in result["features"]["dense_input_views"]
    assert result["features"]["shared_input_siblings"]["status"] == "not_selected"
    assert result["performance_ranking_eligible"] is False


def test_group_profile_is_compact_diagnostic_not_performance(tmp_path):
    r = record(tmp_path)
    r["source_composition_selection"]["preferences"]["group_profile"] = True
    r["build"]["inputs"]["group_profile"] = True
    result = build(r, tiers={"L2": {"group_profile": {
        "status": "observed", "unit": "counter_ticks", "timing_qualified": False,
        "performance_ranking_eligible": False, "group_counter_ticks": 20,
        "between_group_counter_ticks": 12, "total_counter_ticks": 32,
        "private_path": str(tmp_path),
    }}})
    feature = result["features"]["group_profile"]
    assert feature["requested"] and feature["selected"] and feature["status"] == "observed"
    assert feature["observations"]["L2"]["between_group_counter_ticks"] == 12
    assert "private_path" not in str(feature)
    assert not result["performance_ranking_eligible"]


def test_pinned_source_effect_is_compact_and_has_no_paths_or_values(tmp_path):
    r = record(
        tmp_path,
        report={
            "schema": "merlin.tensor_preparation_cse.v1",
            "rewrites": {"tensor.splat": 3},
            "private_values": [99, 100],
        },
    )
    result = build(r)
    feature = result["features"]["share_exact_tensor_preparations"]
    assert feature["status"] == "applied" and feature["effects"]["merged_preparation_operations"] == 3
    assert str(tmp_path) not in json.dumps(result)
    assert "private_values" not in json.dumps(result)


def test_changed_report_does_not_claim_application(tmp_path):
    r = record(tmp_path, report={"schema": "merlin.tensor_preparation_cse.v1", "rewrites": {"tensor.splat": 3}})
    (tmp_path / "tensor-preparation-cse.json").write_text("{}")
    assert build(r)["features"]["share_exact_tensor_preparations"]["status"] == "applied_evidence_unavailable"


def test_storage_feedback_exposes_actual_barriers_without_private_allocations(tmp_path):
    r = record(tmp_path)
    r["source_composition_selection"]["preferences"]["reuse_intermediate_storage"] = True
    r["build"]["inputs"]["device"] = {"intermediate_storage": {"original_policy": True}}
    report = pin(tmp_path / "storage-report.json", {
        "schema": "merlin.host_intermediate_storage.v1",
        "automatic_selection": {"status": "applied", "reason": None},
        "arena": {"bound": 3, "refused": 2, "arena_bytes": 256, "naive_total_bytes": 384,
                  "refusals": {"raw_pointer_escapes": 1, "in_loop": 1,
                               "private_allocation_name": 7, "multiple_frees": True},
                  "placements": [{"allocation": "private_buffer"}]}},
    )
    r["build"]["output"]["compilation_recipe"] = pin(tmp_path / "recipe.json", {
        "preparation": {"intermediate_storage_report": report}})
    feature = build(r)["features"]["reuse_intermediate_storage"]
    assert feature["status"] == "applied"
    assert feature["allocation_refusals"] == {"in_loop": 1, "raw_pointer_escapes": 1}
    assert feature["effects"]["arena_bytes"] == 256
    assert feature["storage_metric_scope"] == "compiled_allocation_intervals_not_runtime_peak_or_cycles"
    assert "private_" not in json.dumps(feature)
    assert build(r)["performance_ranking_eligible"] is False


def test_head_priority_does_not_pretend_sibling_mode_applied(tmp_path):
    r = record(tmp_path)
    r["source_composition_selection"].update(kind="independent_heads", status="source_selected_not_qualified")
    r["source_device_join"] = {
        "composition": {
            "schema": "merlin.original_head_source_composition.v1",
            "components": [{}],
            "matrix_products": 32,
            "batch_abi_calls": 1,
        }
    }
    result = build(r)
    assert result["selected_route"] == "independent_heads"
    assert result["features"]["shared_input_siblings"]["requested"] is True
    assert result["features"]["shared_input_siblings"]["status"] == "not_selected"
    assert result["features"]["independent_heads"]["effects"]["accelerator_execution_count_reduced"] is False


def test_verified_joint_composition_reports_both_actual_modes(tmp_path):
    r = record(tmp_path)
    r["source_device_join"] = {"composition": {
        "schema": "merlin.original_head_source_composition.v1",
        "components": [{"schema": "merlin.original_head_family.v1"},
                       {"schema": "merlin.original_sibling_source_composition.v1", "members": [{}, {}]}],
        "matrix_products": 2, "batch_abi_calls": 1, "ordinary_joined_groups": {3: 6},
        "original_source_denominator_unchanged": True,
    }}
    summary = build(r)
    assert summary["selected_route"] == "independent_heads_and_shared_input_siblings"
    for name in ("independent_heads", "shared_input_siblings"):
        assert summary["features"][name]["status"] == "applied"
    assert summary["features"]["independent_heads"]["effects"]["matrix_products"] == 2
    assert summary["features"]["shared_input_siblings"]["effects"]["composed_original_groups"] == 2


def test_actual_zero_rewrites_stays_distinct_from_absent_evidence(tmp_path):
    r = record(tmp_path, report={"schema": "merlin.tensor_preparation_cse.v1", "rewrites": {}})
    assert build(r)["features"]["share_exact_tensor_preparations"]["status"] == "selected_no_rewrite"


def test_ordinary_public_projection_retains_only_summary(tmp_path):
    from merlin_experiments.phase1.feedback.public_scalability import _observed_result, public_result

    summary = build(record(tmp_path))
    case = {
        "capsule": "public_program",
        "status": "pass",
        "tiers": {},
        "owned_host_model_check": {"source_composition": summary, "private": [1, 2]},
    }
    observed = _observed_result(case)
    public = public_result({"observed": {"public_program": observed}})
    assert public["cases"][0]["source_composition"] == summary
    assert "owned_host_model_check" not in json.dumps(public)


def test_preparation_report_is_pinned_without_an_optional_pass_policy(tmp_path):
    from merlin.runtime.source_optimization_build import SourceOptimizationBuild

    class Recipe:
        def __init__(self):
            self.pins = {}

        def bind_preparation(self, name, path):
            self.pins[name] = path

    (tmp_path / "tensor-preparation-cse.json").write_text("{}")
    (tmp_path / "model.tensor-preparations.mlir").write_text("module {}")
    recipe = Recipe()
    SourceOptimizationBuild(None).consumed(recipe, tmp_path)
    assert set(recipe.pins) == {"tensor_preparation_sharing", "tensor_preparation_shared_source"}


def test_selfcheck_diagnostic_does_not_promote_missing_qualification(tmp_path):
    from merlin_experiments.phase1.feedback.selfcheck_results import _candidate_selfcheck_row

    summary = build(record(tmp_path))
    row, certified = _candidate_selfcheck_row(
        {"owned_host_model_check": {"source_composition": summary}}, None, name="public_program", barrier_tier="L2"
    )
    assert row["source_composition"] == summary
    assert row["pass"] is False and certified is False


def test_pure_profile_reader_is_an_explicit_full_export_helper():
    from merlin_experiments.phase1.full_compiler_workspace import PUBLIC_HELPERS, forbidden

    assert "merlin.runtime.operation_profile" in PUBLIC_HELPERS
    assert forbidden("merlin.runtime.operation_profile") is False


def test_bounded_rne_application_requires_original_pinned_source_report(tmp_path):
    from pathlib import Path

    from merlin.llvmlower import late_quant_rne

    r = record(tmp_path)
    r["source_composition_selection"]["preferences"]["host_bounded_rne"] = True
    r["build"]["inputs"]["host_bounded_rne"] = {"host_isa": "rv64gc", "combine_clamp": True}
    assert build(r)["features"]["host_bounded_rne"]["status"] == "applied_evidence_unavailable"
    proof = pin(tmp_path / "rne-proof.json", {
        "schema": "merlin_late_bounded_rne_v1", "host_isa": "rv64gc", "combine_clamp": True,
        "legalizer_sha256": hashlib.sha256(Path(late_quant_rne.__file__).read_bytes()).hexdigest(),
        "routes": [{"source_form": "bounded_roundeven_conversion"}],
    })
    r["build"]["output"]["compilation_recipe"] = pin(tmp_path / "recipe.json", {
        "preparation": {"host_bounded_rne": proof},
    })
    feature = build(r)["features"]["host_bounded_rne"]
    assert feature["requested"] and feature["selected"] and feature["status"] == "applied"
    assert feature["effects"] == {
        "proven_original_chains": 1, "historical_observer_permissions_imported": False,
    }
    (tmp_path / "rne-proof.json").write_text("{}")
    assert build(r)["features"]["host_bounded_rne"]["status"] == "applied_evidence_unavailable"
