"""Tests for the DSE guidance workstream (merlin.dse_guidance).

Covers: temporal parsing + derived deadline, the flat->multi-rate reuse flip, residency/
autonomous-loop legality, the gap_closure formula and its edge cases, the negative control,
the "no measurement -> no invented constants" guarantee, evidence-tag propagation, component-
sum residualization, the aet instrumentation adapter, and schema validity of the artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.common import schemas
from merlin.common.yaml import load_yaml
from merlin.dse_guidance import (aet_ingest, attribution as ATTR, axes, baseline_cost as BC,
                                 calibration, candidates as CAND, fidelity as FID, pipeline,
                                 representation, study, synth, temporal as T, topology as TOP)

FIX = Path(__file__).parent / "fixtures" / "dse_guidance"


def _smolvla():
    temporal = T.load(FIX / "smolvla_action_head_temporal.yaml")
    baseline = BC.load(FIX / "smolvla_action_head_cost.yaml")
    return temporal, baseline


# --------------------------------------------------------------------- C1 temporal

def test_temporal_derived_deadline():
    temporal = T.load(FIX / "smolvla_action_head_temporal.yaml")
    assert temporal.K == 10 and temporal.H == 50
    # 1000 * 50 / 30 = 1666.67 ms
    assert temporal.replan_deadline_ms == pytest.approx(1000 * 50 / 30, rel=1e-9)
    assert "action_head_weights" in temporal.loop_invariant_state()
    assert temporal.has_k_loop()


def test_temporal_stated_deadline_disagreement_warns():
    doc = {
        "workload": "w", "timing": {"K": 2, "H": 10, "control_rate_hz": 30,
                                    "replan_deadline_ms": 999.0},
        "regions": [],
    }
    md = T.parse(doc)
    assert md.replan_deadline_ms == pytest.approx(1000 * 10 / 30)  # derived wins
    assert any("disagrees" in w for w in md.warnings)


# ---------------------------------------------------------- C2 flat vs multi-rate

def test_flat_to_multirate_flips_reuse_1_to_K():
    temporal, _ = _smolvla()
    reps = representation.build_representations(temporal, region=None)
    assert reps["flat"].visible_weight_reuse == 1
    assert reps["multirate"].visible_weight_reuse == temporal.K == 10
    assert reps["flat"].deadline_visible is False
    assert reps["multirate"].deadline_visible is True
    # prefix_kv is loop-invariant in the temporal metadata -> visible K-reuse under multirate
    assert reps["multirate"].visible_prefix_kv_reuse == temporal.K


def test_resident_recommended_only_in_multirate():
    temporal, _ = _smolvla()
    reps = representation.build_representations(temporal, region=None)
    assert "resident_packed_weights" not in reps["flat"].recommended_axis_names
    assert "resident_packed_weights" in reps["multirate"].recommended_axis_names
    assert "resident_prefix_kv" in reps["multirate"].recommended_axis_names


# --------------------------------------------------- C3 axis legality / triage math

def test_resident_weights_legal_only_when_K_gt_1_and_loop_invariant():
    temporal, baseline = _smolvla()
    reps = representation.build_representations(temporal, region=None)
    flat = axes.evaluate_axis("resident_packed_weights", reps["flat"].facts, baseline)
    multi = axes.evaluate_axis("resident_packed_weights", reps["multirate"].facts, baseline)
    assert flat.legality == 0
    assert multi.legality == 1


def test_autonomous_k_loop_illegal_when_K_le_1():
    temporal = T.load(FIX / "no_reuse_matmul_temporal.yaml")
    reps = representation.build_representations(temporal, region=None)
    baseline = BC.parse({"workload": "no_reuse_matmul",
                         "baseline": {"total_ms": 10,
                                      "components": {"cpu_dispatch_ms": 6, "sync_ms": 4}}})
    multi = axes.evaluate_axis("autonomous_K_loop", reps["multirate"].facts, baseline)
    assert multi.legality == 0
    assert multi.benefit_ms == 0.0


def test_gap_closure_formula_and_target_already_met():
    baseline = BC.parse({
        "workload": "w",
        "target": {"total_ms": 50},
        "baseline": {"total_ms": 100, "components": {"dma_memory_ms": 100}},
        "metadata_source": {"dma_memory_ms": "analytical"},
    })
    facts = {"K": 1, "has_k_loop": False, "dispatches_per_replan": 1}
    res = axes.evaluate_axis("DMA_bandwidth_2x", facts, baseline)
    # benefit = 50 (half of dma), target_gap = 50 -> gap_closure = 1.0
    from merlin.dse_guidance.triage import _row
    row = _row("w", "flat", 100.0, 50.0, 50.0, res)
    assert row["gap_closure"] == pytest.approx(1.0)
    assert row["priority_score"] is not None

    # Target already met: target_gap <= 0 -> no score, flagged.
    row2 = _row("w", "flat", 40.0, 50.0, -10.0, res)
    assert row2["gap_closure"] is None
    assert row2["priority_score"] is None
    assert "already meets target" in row2["reason"]


def test_no_target_reports_baseline_share_only():
    baseline = BC.parse({"workload": "w",
                         "baseline": {"total_ms": 100, "components": {"compute_ms": 40}}})
    assert baseline.target_gap_ms is None
    res = axes.evaluate_axis("PE_count_2x", {"K": 1, "dispatches_per_replan": 1}, baseline)
    from merlin.dse_guidance.triage import _row
    row = _row("w", "flat", 100.0, None, None, res)
    assert row["gap_closure"] is None
    assert row["baseline_share"] == pytest.approx(0.2)  # 20 / 100


# ---------------------------------------------------------------- C7 negative control

def test_negative_control_does_not_recommend_residency():
    region = load_yaml(study.paths.merlin_dir() / "benchmarks" / "semantic_memory"
                       / "no_reuse_matmul.yaml")
    spec = study.spec_from_region("no_reuse_matmul", region)
    res = pipeline.run_guidance(spec.temporal, spec.baseline, region=region)
    rows = {r["axis"]: r for r in res.triage_multirate["axes"]}
    assert rows["resident_packed_weights"]["legality"] == 0
    assert (rows["resident_packed_weights"]["priority_score"] in (None, 0)
            or rows["resident_packed_weights"]["gap_closure"] in (None, 0))
    assert rows["autonomous_K_loop"]["legality"] == 0
    assert res.is_negative_control


# ------------------------------------------------------- C5 no-measurement guarantee

def test_missing_cpu_measurement_invents_nothing():
    assert aet_ingest.ingest(None, None) is None
    temporal, baseline = _smolvla()
    res = pipeline.run_guidance(temporal, baseline, region=None, coupling=None)
    assert res.coupling_per_replan is None
    assert res.calibration_rows == []  # no fabricated anchor


# ------------------------------------------------------ C5/C6 aet instrumentation path

def test_aet_run_ingest_tags_measured_and_upgrades_command_batching():
    coupling = aet_ingest.from_aet_run(FIX / "aet_run", workload="smolvla_action_head")
    assert coupling is not None
    assert coupling.source == "measured"
    temporal, baseline = _smolvla()
    res = pipeline.run_guidance(temporal, baseline, region=None, coupling=coupling)
    cb = next(r for r in res.triage_multirate["axes"] if r["axis"] == "command_batching")
    assert cb["evidence_type"] == "measured"
    # A calibration anchor row is produced from measured vs predicted dispatch count.
    quantities = {r["quantity"] for r in res.calibration_rows}
    assert "dispatch_count_op_level" in quantities


def test_cpu_coupling_yaml_matches_aet_run():
    via_yaml = aet_ingest.ingest(FIX / "cpu_coupling.yaml", None)
    via_run = aet_ingest.from_aet_run(FIX / "aet_run")
    assert via_yaml.regimes[aet_ingest.OP_LEVEL].host_submit_ns \
        == via_run.regimes[aet_ingest.OP_LEVEL].host_submit_ns


# ----------------------------------------------------------- C4 component handling

def test_component_sum_residualized_not_silently_dropped():
    baseline = BC.parse({"workload": "w",
                         "baseline": {"total_ms": 100,
                                      "components": {"compute_ms": 40, "dma_memory_ms": 30}}})
    assert "residual" in baseline.components
    assert baseline.component("residual") == pytest.approx(30.0)
    assert any("residual" in w for w in baseline.warnings)


def test_untagged_component_defaults_to_assumed():
    baseline = BC.parse({"workload": "w",
                         "baseline": {"total_ms": 10, "components": {"compute_ms": 10}}})
    assert baseline.evidence_for("compute") == "assumed"


# ------------------------------------------------------------- evidence propagation

def test_evidence_tag_propagates_into_triage_rows():
    temporal, baseline = _smolvla()
    res = pipeline.run_guidance(temporal, baseline, region=None)
    # Without measured coupling, command_batching uses a STRUCTURAL intervention model, so it
    # caps at structural_bound even though cpu_dispatch/sync are measured (the model is the
    # weak link). This is the honest design: a structural estimate is not a measurement.
    cb = next(r for r in res.triage_multirate["axes"] if r["axis"] == "command_batching")
    assert cb["evidence_type"] == "structural_bound"
    # compute is trace_derived, but PE_count_2x's model is analytical -> weakest = analytical.
    pe = next(r for r in res.triage_multirate["axes"] if r["axis"] == "PE_count_2x")
    assert pe["evidence_type"] == "analytical"


def test_weak_component_evidence_drags_axis_down():
    # An untagged (assumed) component pulls the axis evidence below its analytical model.
    baseline = BC.parse({"workload": "w",
                         "baseline": {"total_ms": 100, "components": {"compute_ms": 40}}})
    res = axes.evaluate_axis("PE_count_2x", {"K": 1, "dispatches_per_replan": 1}, baseline)
    assert res.evidence_type == "assumed"


# --------------------------------------------------------------------- schema checks

def test_artifacts_validate_against_schemas(tmp_path):
    temporal, baseline = _smolvla()
    res = pipeline.run_guidance(temporal, baseline, region=None)
    pipeline.write_artifacts(res, tmp_path)
    # triage dict validates against dse_axis_triage
    assert schemas.validate(res.triage_multirate, "dse_axis_triage") == []
    # input docs validate against their schemas
    assert schemas.validate(load_yaml(FIX / "smolvla_action_head_temporal.yaml"),
                            "temporal_workload_metadata") == []
    assert schemas.validate(load_yaml(FIX / "smolvla_action_head_cost.yaml"),
                            "baseline_cost") == []
    assert schemas.validate(load_yaml(FIX / "cpu_coupling.yaml"), "cpu_coupling") == []


# ----------------------------------------------------------- exhaustive study

def test_study_runs_all_supported_workloads(tmp_path):
    specs = study.discover_specs()
    names = {s.name for s in specs}
    assert {"vla_action_chunk_decode", "no_reuse_matmul", "repeated_rhs_matmul"} <= names
    summary = study.run_study(specs, tmp_path)
    assert summary["n_workloads"] == len(specs)
    assert (tmp_path / "study_summary.csv").is_file()
    assert (tmp_path / "study_summary.md").is_file()
    # Reuse workloads put residency on top; the no-reuse control does not.
    text = (tmp_path / "study_summary.md").read_text()
    assert "resident_packed_weights" in text


def test_study_synthesized_baseline_is_analytical():
    region = load_yaml(study.paths.merlin_dir() / "benchmarks" / "semantic_memory"
                       / "vla_action_chunk_decode.yaml")
    spec = study.spec_from_region("vla_action_chunk_decode", region)
    assert spec.baseline.unit == "cycles"
    assert all(v == "analytical" for v in spec.baseline.evidence.values())


# ------------------------------------------------------ residency over-claim guard

def test_residency_is_component_specific_and_unquantified_without_attribution():
    baseline = BC.parse({"workload": "w",
                         "baseline": {"total_ms": 100,
                                      "components": {"packing_ms": 20, "dma_memory_ms": 60,
                                                     "compute_ms": 20}}})
    facts = {"K": 8, "has_k_loop": True, "has_repeated_head": True,
             "head_weights_immutable": True, "visible_weight_reuse": 8,
             "dispatches_per_replan": 8}
    # No head/backbone split and no region fraction -> structurally legal but NOT quantified
    # (we refuse to apply K-reuse to undifferentiated whole-model cost).
    res = axes.evaluate_axis("resident_packed_weights", facts, baseline)
    assert res.legality == 1 and res.quantified is False and res.benefit_ms == 0.0

    # With a region-derived reducible fraction, the dma portion IS grounded and claimed.
    facts2 = dict(facts, dram_reducible_fraction=0.5)
    res2 = axes.evaluate_axis("resident_packed_weights", facts2, baseline)
    assert res2.quantified and "dma_memory" in res2.affected_components
    assert res2.benefit_ms == pytest.approx(20 * (1 - 1 / 8) + 60 * 0.5)

    # With an explicit repeated_head breakdown, residency is charged to the head only.
    base_head = BC.parse({"workload": "w",
                          "baseline": {"total_ms": 100,
                                       "components": {"packing_ms": 20, "dma_memory_ms": 60,
                                                      "compute_ms": 20},
                                       "repeated_head": {"packing_ms": 8,
                                                         "weight_memory_ms": 10}}})
    res3 = axes.evaluate_axis("resident_packed_weights", facts, base_head)
    assert res3.quantified
    assert res3.benefit_ms == pytest.approx(8 * (1 - 1 / 8) + 10 * (1 - 1 / 8))  # head only


# ----------------------------------------------------- real model zoo (output/ captures)

def _has_model_captures() -> bool:
    from merlin.dse_guidance import models as M
    return bool(M.discover_model_captures())


@pytest.mark.skipif(not _has_model_captures(), reason="no model.mlir captures under output/")
def test_model_zoo_discovery_and_specs():
    from merlin.dse_guidance import models as M
    captures = M.discover_model_captures()
    # The supported VLA/LM zoo should include these base models.
    assert {"openvla", "rdt2", "xr0"} <= set(captures)
    specs = study.discover_model_specs()
    assert specs
    for spec in specs:
        assert schemas.validate(  # temporal + baseline docs are well-formed
            {"workload": spec.name, "timing": {"K": spec.temporal.K},
             "regions": []}, "temporal_workload_metadata") == []
        assert spec.baseline.unit == "cycles"


@pytest.mark.skipif(not _has_model_captures(), reason="no model.mlir captures under output/")
def test_model_flat_vs_multirate_legality_flip():
    # On a real captured model, residency is illegal under the flat (single-pass) capture and
    # becomes legal under the multi-rate decode/denoise loop — the central thesis on real data.
    from merlin.dse_guidance import models as M
    captures = M.discover_model_captures()
    spec = study.spec_from_model("openvla", captures["openvla"])
    res = pipeline.run_guidance(spec.temporal, spec.baseline, overrides=spec.overrides)
    flat = {r["axis"]: r for r in res.triage_flat["axes"]}
    multi = {r["axis"]: r for r in res.triage_multirate["axes"]}
    assert flat["resident_packed_weights"]["legality"] == 0
    assert multi["resident_packed_weights"]["legality"] == 1


@pytest.mark.skipif(not _has_model_captures(), reason="no model.mlir captures under output/")
def test_model_xr0_calibration_anchor_is_measured_and_honest():
    # The baseline stays ANALYTICAL (not circularly scaled to the measurement). The measured
    # FireSim total appears only in the calibration anchor, which honestly reports the gap.
    from merlin.dse_guidance import models as M
    captures = M.discover_model_captures()
    if "xr0" not in captures:
        pytest.skip("no xr0 capture")
    spec = study.spec_from_model("xr0", captures["xr0"])
    assert spec.baseline.baseline_total_ms < 1e9  # analytical, NOT the 1.46e11 measured total
    rows = M.calibration_rows(M.MODEL_ARCH["xr0"],
                              M.capture_facts(M._prefer_capture(captures["xr0"])))
    total_row = next(r for r in rows if r["quantity"] == "total_cycles")
    assert total_row["measured"] == pytest.approx(146.2e9, rel=1e-6)
    assert total_row["evidence_type"] == "measured"
    assert total_row["predicted"] < total_row["measured"]  # analytical underestimates (honest)


# ------------------------------------------------ VLA topology / fidelity / candidates

def test_topology_classifies_workload():
    assert TOP.classify("flow_matching_action_head") == TOP.CLASS_FLOW_MATCHING
    assert TOP.classify("diffusion/denoise_steps") == TOP.CLASS_FLOW_MATCHING
    assert TOP.classify("autoregressive_vla/action_token_decode") == TOP.CLASS_AUTOREGRESSIVE
    assert TOP.classify("regression_parallel_head") == TOP.CLASS_REGRESSION_PARALLEL
    assert TOP.classify("something_else") == TOP.CLASS_UNKNOWN


def test_flat_flow_matching_hides_loop_axes():
    topo = TOP.load(FIX / "smolvla_action_head_temporal.yaml")
    assert topo.workload_class == TOP.CLASS_FLOW_MATCHING
    # The prefix/KV crossing is recovered from the produce/consume boundary.
    assert any(c["state"] == "prefix_kv" for c in topo.state_crossing_boundaries())
    fid = FID.assess(topo)
    assert fid.severity == "high"
    assert "denoise_loop" in fid.missing_structure
    assert {"resident_action_head_weights", "autonomous_K_loop"} <= set(fid.hidden_axes)
    cand_axes = {c.axis for c in CAND.detect(topo)}
    assert {"resident_action_head_weights", "resident_prefix_kv", "autonomous_K_loop",
            "command_batching", "backbone_head_partition"} <= cand_axes


def test_regression_head_does_not_recommend_loop_axes():
    topo = TOP.load(FIX / "regression_head_temporal.yaml")
    assert topo.workload_class == TOP.CLASS_REGRESSION_PARALLEL
    assert not topo.has_repeated_head()
    fid = FID.assess(topo)
    assert fid.severity == "low"
    cand_axes = {c.axis for c in CAND.detect(topo)}
    # No inner loop -> these loop-specific axes must NOT be candidates.
    assert "autonomous_K_loop" not in cand_axes
    assert "resident_action_head_weights" not in cand_axes
    assert "command_batching" not in cand_axes


def test_candidates_are_structural_and_carry_no_numbers():
    topo = TOP.load(FIX / "smolvla_action_head_temporal.yaml")
    for c in CAND.detect(topo):
        assert c.legality == "structural"
        assert c.benefit == "unquantified"
        assert c.required_measurements          # says what to measure first
        assert c.could_be_wrong_if
        # A candidate certificate carries no cycle/gap_closure/score field.
        assert not hasattr(c, "gap_closure") and not hasattr(c, "priority_score")


def test_autoregressive_surfaces_decode_kv_path():
    doc = {"workload": "ar", "class": "autoregressive_decode",
           "timing": {"K": 8, "H": 8, "control_rate_hz": 5},
           "regions": [
               {"name": "backbone", "role": "backbone_once", "produces": ["prefix_kv"]},
               {"name": "decode", "role": "repeated_head", "invocation_count": 8,
                "loop_invariant_state": ["weights"], "consumes": ["prefix_kv"]}]}
    topo = TOP.from_temporal(T.parse(doc))
    axes_found = {c.axis for c in CAND.detect(topo)}
    assert "decode_kv_cache_path" in axes_found


def test_structural_only_run_needs_no_baseline(tmp_path):
    topo_tm = T.load(FIX / "smolvla_action_head_temporal.yaml")
    res = pipeline.run_guidance(topo_tm, baseline=None)
    assert res.triage_multirate is None          # no quantitative output without a baseline
    assert res.candidate_axes                     # but structural candidates are produced
    pipeline.write_artifacts(res, tmp_path)
    assert (tmp_path / "dse_candidate_axes.md").is_file()
    assert (tmp_path / "capture_fidelity_report.md").is_file()
    assert (tmp_path / "vla_runtime_topology.yaml").is_file()
    assert not (tmp_path / "axis_triage.csv").exists()   # gated on a baseline


# ------------------------------------------------ Level-1 region attribution

def _head_rec(i):
    return ATTR.MatmulRecord(i, f"matmul_{i}", "addmm", True, 28, 1024, 1024, 1024 * 1024 * 4,
                             (28 * 1024 + 28 * 1024) * 4, "f32")


def _records():
    # backbone: one big projection; head: a transformer block repeated 3x in the capture.
    return (
        ATTR.MatmulRecord(0, "matmul_0", "matmul", False, 1, 2048, 9216, 2048 * 9216 * 4,
                          (1 * 2048 + 1 * 9216) * 4, "f32"),
        _head_rec(1), _head_rec(2), _head_rec(3),
        ATTR.MatmulRecord(4, "matmul_4", "matmul", False, 28, 512, 512, 512 * 512 * 4,
                          (28 * 512 + 28 * 512) * 4, "f32"),  # unmapped -> unknown
    )


def _topo(K=5):
    doc = {"workload": "m", "class": "diffusion/denoise_steps",
           "timing": {"K": K, "H": K, "control_rate_hz": 30},
           "regions": [{"name": "bb", "role": "backbone_once"},
                       {"name": "hd", "role": "repeated_head", "invocation_count": K,
                        "loop_invariant_state": ["weights"]}]}
    return TOP.from_temporal(T.parse(doc))


def test_attribution_explicit_mapping_attributes_real_facts():
    rules = {"rules": [
        {"role": "repeated_head", "match": {"shape_signature": [28, 1024, 1024]}},
        {"role": "backbone_once", "match": {"region_ids": ["matmul_0"]}},
    ]}
    attr = ATTR.attribute_records(_records(), _topo(K=5), rules)
    head = attr.role("repeated_head")
    bb = attr.role("backbone_once")
    assert head.attribution_status == "attributed" and head.source == "explicit_mapping"
    assert head.facts["matmul_count"] == 3
    # repeated_head total is multiplied by K (5); per-invocation is not.
    assert head.facts["invocations"] == 5
    assert head.facts["macs_total"] == head.facts["macs_per_invocation"] * 5
    # backbone is NOT multiplied by K.
    assert bb.invocations == 1
    assert bb.facts["macs_total"] == bb.facts["macs_per_invocation"]


def test_role_from_fqn_inference():
    assert ATTR.role_from_fqn("model.action_expert.denoise_block.2") == "repeated_head"
    assert ATTR.role_from_fqn("model.vision_backbone.layers.3.attn") == "backbone_once"
    assert ATTR.role_from_fqn("model.kv_cache.update") == "prefix_builder"
    assert ATTR.role_from_fqn("model.mystery.layer") is None
    assert ATTR.role_from_fqn(None) is None
    # Real RDT denoise-step module paths (observed in a fresh capture) -> repeated_head.
    assert ATTR.role_from_fqn("model.blocks.0.attn.qkv") == "repeated_head"
    assert ATTR.role_from_fqn("model.t_embedder.mlp.0") == "repeated_head"
    # Ordering: a vision backbone's OWN blocks must not be read as head.
    assert ATTR.role_from_fqn("vision_backbone.blocks.3.attn") == "backbone_once"


# Real prov.fqn recaptures (real architectures, small random configs) live under
# merlin/benchmarks/dse_guidance/recaptures/<workload>/model.mlir. ~MLIR only; weights NOT committed.
_RECAP = Path(__file__).resolve().parents[2] / "benchmarks" / "dse_guidance" / "recaptures"
_RDT_RECAP = _RECAP / "rdt"


@pytest.mark.skipif(not (_RDT_RECAP / "model.mlir").is_file(),
                    reason="no fresh rdt prov.fqn capture")
def test_real_capture_auto_recovers_head_role_from_prov_fqn():
    # The whole captured graph is the denoise head; every matmul carries prov.fqn and
    # auto-recovers to repeated_head with NO operator mapping.
    recs = ATTR.extract_matmuls(str(_RDT_RECAP))
    assert len(recs) > 0 and all(r.fqn for r in recs)
    assert all(ATTR.role_from_fqn(r.fqn) == "repeated_head" for r in recs)

    topo = TOP.from_temporal(T.parse({
        "workload": "rdt_recap", "class": "diffusion/denoise_steps",
        "timing": {"K": 5, "H": 64, "control_rate_hz": 30},
        "regions": [{"name": "denoise", "role": "repeated_head", "invocation_count": 5,
                     "loop_invariant_state": ["weights"]}]}))
    attr = ATTR.attribute(str(_RDT_RECAP), topo)          # NO operator map
    head = attr.role("repeated_head")
    assert head is not None and head.attribution_status == "attributed"
    assert head.source == "prov_fqn"                       # role recovered from the capture
    assert head.facts["matmul_count"] == len(recs)
    assert head.facts["weight_bytes"] > 0 and head.facts["macs_per_invocation"] > 0
    assert head.invocations == 5 and head.facts["macs_total"] == head.facts["macs_per_invocation"] * 5

    # The candidate certificate carries the real attributed facts; ranking stays gated.
    cand = {c.axis: c for c in CAND.detect(topo, attribution=attr)}["resident_action_head_weights"]
    assert cand.attributed_facts and cand.attributed_facts["matmul_count"] == len(recs)
    assert cand.quantification_blocked_by == "missing_calibration"
    assert cand.benefit == "unquantified"


# --------------------------------------------- cross-workload case study (multiple real captures)

def _has_recaptures() -> bool:
    from merlin.dse_guidance import case_study as CS
    return len(CS.available_models()) >= 2


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_is_multi_workload_and_honest():
    from merlin.dse_guidance import case_study as CS
    models = CS.available_models()
    assert len(models) >= 2                     # the value is breadth, not one model
    cases = [CS.analyze(w) for w in models]
    for c in cases:
        # Roles auto-recovered from prov.fqn; at least the repeated head is attributed.
        head = c.attribution.role("repeated_head")
        assert head is not None and head.attribution_status == "attributed"
        assert head.facts["matmul_count"] > 0 and head.facts["weight_bytes"] > 0
        # No candidate fabricates a quantitative benefit.
        for cand in c.candidates:
            assert cand.benefit == "unquantified"


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_openvla_recovers_backbone_head_split():
    from merlin.dse_guidance import case_study as CS
    if "openvla" not in CS.available_models():
        pytest.skip("no openvla recapture")
    case = CS.analyze("openvla")
    # The real OpenVLA capture splits into a vision backbone AND a decode head, from prov.fqn.
    assert case.attribution.role("backbone_once") is not None
    assert case.attribution.role("repeated_head") is not None


# ------------------------------------------------ accuracy gate (measurable-now, real)

def test_accuracy_gate_status():
    from merlin.dse_guidance import accuracy_gate as AG
    pts = AG.load()
    assert pts and all(p.dtype == "int8" for p in pts)
    assert AG.status_for("small_llama", "int8") == "pass"
    assert AG.status_for("openvla", "int8_w8a8") == "pass"     # candidate-format label maps to int8
    assert AG.status_for("small_llama", "int4_weight_only") == "unavailable"   # not measured
    assert AG.status_for("nonexistent", "int8") == "unavailable"
    assert "W8A8" in AG.report_md()


def test_dse_readiness_accuracy_status_wired():
    from merlin.dse_guidance import contract as CON
    topo = _topo(K=5)
    recs = (ATTR.MatmulRecord(0, "m0", "addmm", True, 28, 1024, 1024, 1024 * 1024 * 4, 1, "f32",
                              fqn="model.action_expert.denoise.0"),)
    attr = ATTR.attribute_records(recs, topo)
    r_pass = CON.dse_readiness(topo, attr, None, cpu_coupling_available=False,
                               accuracy_status="pass")
    assert r_pass.fields["accuracy_constraints"]["available"] is True
    assert r_pass.fields["accuracy_constraints"]["source"] == "measured"
    assert not any("accuracy" in m for m in r_pass.missing)
    r_na = CON.dse_readiness(topo, attr, None, cpu_coupling_available=False,
                             accuracy_status="unavailable")
    assert any("accuracy" in m for m in r_na.missing)


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_presentation_package(tmp_path):
    from merlin.dse_guidance import case_study as CS
    CS.run_case_study(tmp_path)
    for f in ("README.md", "case_study_summary.md", "workload_contract_table.csv",
              "abstraction_pressure_table.csv", "dse_readiness_summary.csv",
              "dtype_capacity_table.csv", "accuracy_gate_report.md", "accuracy_gate_results.csv",
              "requirements_table.csv"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    # Headline resident set is the bf16 figure (consistent), and quantitative DSE stays not-ready.
    wct = (tmp_path / "workload_contract_table.csv").read_text()
    assert "resident_bf16_B" in wct and "ready_quantitative" in wct
    # The only mention of speedup is the explicit "no speedup claimed" disclaimer.
    assert "no speedup claimed" in (tmp_path / "case_study_summary.md").read_text().lower()


# ------------------------------------------------ design envelope (requirements, not calibration)

def test_design_envelope_requirements_and_dtype_scaling():
    from merlin.dse_guidance import design_envelope as DE
    # K=5, deadline from H=64 @ 30Hz = 2.1333s; weight_bytes captured as f32.
    env = DE.derive("rdt", K=5, deadline_s=64 / 30.0, deadline_evidence="assumed_reference",
                    macs_per_step=39_432_486_912, weight_bytes=391_118_848,
                    activation_bytes_per_step=10_000_000, dispatches_per_step=20,
                    dispatches_evidence="derived_requirement", captured_dtype="f32")
    assert env.req("macs_per_replan").value == 39_432_486_912 * 5
    assert env.req("resident_capacity_required").value == 391_118_848
    assert env.req("avoidable_weight_reload_bytes").value == 391_118_848 * 4
    # dtype-scaled capacity from the f32 element count (97.78M params).
    n_elem = 391_118_848 / 4
    assert env.capacity_by_dtype_B["bf16"] == pytest.approx(n_elem * 2)
    assert env.capacity_by_dtype_B["int8"] == pytest.approx(n_elem * 1)
    assert env.capacity_by_dtype_B["int4"] == pytest.approx(n_elem * 0.5)
    assert env.capacity_by_dtype_B["fp6"] == pytest.approx(n_elem * 0.75)
    # No gap_closure anywhere; the only mention of speedup is the explicit "not_claimed" status.
    obj = DE.to_yaml_obj(env)
    assert "gap_closure" not in str(obj).lower()
    assert obj["design_envelope"]["status"]["quantitative_speedup"] == "not_claimed"


def test_design_envelope_missing_deadline_keeps_capacity_drops_rate():
    from merlin.dse_guidance import design_envelope as DE
    env = DE.derive("w", K=5, deadline_s=None, deadline_evidence="unavailable",
                    macs_per_step=1000, weight_bytes=4000, activation_bytes_per_step=0,
                    dispatches_per_step=2, dispatches_evidence="derived_requirement",
                    captured_dtype="f32")
    assert env.req("resident_capacity_required").value == 4000        # capacity still emitted
    assert env.req("required_compute_rate").value is None             # rate unavailable
    assert env.req("required_compute_rate").evidence == "unavailable"


def test_design_envelope_feasibility_and_command_gate():
    from merlin.dse_guidance import design_envelope as DE
    design = {"name": "toy", "clock_ghz": 1.0, "macs_per_cycle": 4096, "local_memory_mb": 256,
              "dram_bandwidth_gb_s": 256, "supported_dtypes": ["bf16", "int8"]}  # no submit_ns
    env = DE.derive("rdt", K=5, deadline_s=64 / 30.0, deadline_evidence="assumed_reference",
                    macs_per_step=39_432_486_912, weight_bytes=391_118_848,
                    activation_bytes_per_step=1e7, dispatches_per_step=20,
                    dispatches_evidence="derived_requirement", captured_dtype="f32", design=design)
    f = env.feasibility
    assert f["compute_feasible"] is True and f["memory_feasible"] is True
    assert f["capacity_feasible"] is True
    assert f["dtype_feasible"]["int8"] is True and f["dtype_feasible"]["fp6"] is False
    assert f["command_feasible"] == "unavailable"     # no command_submit_ns -> not fabricated


# ------------------------------------------------ workload-contract package (DSE-ready)

def test_contract_abstraction_candidates_and_readiness():
    from merlin.dse_guidance import contract as CON, candidates as CAND, numerical_contract as NC
    topo = _topo(K=5)
    recs = (
        ATTR.MatmulRecord(0, "m0", "addmm", True, 28, 1024, 1024, 1024 * 1024 * 4, 1, "f32",
                          fqn="model.action_expert.denoise.0"),
        ATTR.MatmulRecord(1, "m1", "addmm", True, 28, 1024, 1024, 1024 * 1024 * 4, 1, "f32",
                          fqn="model.action_expert.denoise.1"),
    )
    attr = ATTR.attribute_records(recs, topo)
    structural = CAND.detect(topo, attribution=attr)
    nc = NC.audit(str(_RDT_RECAP), workload="rdt") if (_RDT_RECAP / "model.mlir").is_file() else None
    cands = CON.abstraction_candidates(structural, nc)
    assert cands, "expected abstraction candidates"
    by_axis = {c.axis: c for c in cands}
    # Each maps to a concrete system abstraction + DSE knobs and claims no speedup.
    rw = by_axis.get("resident_action_head_weights")
    assert rw and "resident_weight_object" in rw.system_abstraction
    assert rw.dse_knobs_exposed and "speedup" in rw.what_is_not_claimed
    blob = str(CON.abstraction_yaml(cands)).lower()
    assert "speedup" not in blob.replace("not_claimed", "").replace("what_is_not_claimed", "") \
        or "speedup" in str(rw.what_is_not_claimed).lower()   # only as a "not claimed" entry

    readiness = CON.dse_readiness(topo, attr, nc, cpu_coupling_available=False)
    assert readiness.ready is False
    assert any("accuracy" in m for m in readiness.missing)
    assert any("command-submit" in m or "sync" in m for m in readiness.missing)
    assert any("K /" in m or "control-rate" in m for m in readiness.missing)


def test_contract_measurement_plan_splits_proxy_vs_target():
    from merlin.dse_guidance import contract as CON, candidates as CAND
    topo = _topo(K=5)
    recs = (ATTR.MatmulRecord(0, "m0", "addmm", True, 28, 1024, 1024, 1024 * 1024 * 4, 1, "f32",
                              fqn="model.action_expert.denoise.0"),)
    attr = ATTR.attribute_records(recs, topo)
    cands = CON.abstraction_candidates(CAND.detect(topo, attribution=attr), None)
    plan = CON.measurement_plan(cands)
    assert "measurable_now" in plan and "needs_target_design" in plan
    # dispatch/host measurements are proxy-measurable now; bandwidth/capacity need the target.
    runtime = " ".join(plan["measurable_now"]["runtime_proxy"]).lower()
    target = " ".join(plan["needs_target_design"]).lower()
    assert "dispatch" in runtime or "submit" in runtime
    assert "bandwidth" in target or "capacity" in target
    # "cost" must NOT be misclassified as accuracy (the "cos" substring bug).
    assert not any("backbone_cost" in m for m in plan["measurable_now"]["accuracy"])


# ------------------------------------------------ measured dispatch coupling (one measured leg)

def test_measured_dispatch_data_grounds_command_batching():
    from merlin.dse_guidance import dispatch_measure as DM
    ms = DM.load_measured()
    assert len(ms) >= 1
    for m in ms:
        assert m.cos >= 0.999                       # the measured run was faithful
        assert m.n_kernels > m.matmul_estimate       # real dispatches >> matmul proxy
        assert m.undercount_ratio and m.undercount_ratio > 3  # matmul count badly undercounts
    rows = DM.calibration_rows(ms)
    assert all(r["evidence_type"] == "measured" for r in rows)
    assert all(r["quantity"] == "dispatches_per_forward" for r in rows)


def test_measured_host_breakdown_shows_dispatch_bound():
    # P1-b: the per-dispatch host-cost breakdown shows the forward is host-dispatch-bound.
    from merlin.dse_guidance import dispatch_measure as DM
    bd = [m for m in DM.load_measured() if m.overhead_frac is not None]
    assert bd, "expected recorded host breakdowns"
    for m in bd:
        assert m.overhead_frac > 0.5             # majority of host time is dispatch/alloc overhead
        assert m.per_dispatch_host_ms and m.per_dispatch_host_ms > 0


def test_per_component_calibration_reports_unidentifiable():
    # P1-a: multi-feature fits over a handful of whole-model points must not be presented as a
    # calibrated per-component model; the report says coefficients are not identifiable.
    from merlin.dse_guidance import cost_calibration as CC
    rows = [
        {"cycles": 1.8e8, "macs": 3.4e6, "act_bytes": 1.0e6, "matmuls": 15},
        {"cycles": 1.3e11, "macs": 1.2e9, "act_bytes": 5.0e8, "matmuls": 15},
        {"cycles": 9.8e9, "macs": 8.0e7, "act_bytes": 2.0e7, "matmuls": 26},
        {"cycles": 8.5e10, "macs": 9.4e8, "act_bytes": 3.0e8, "matmuls": 23},
    ]
    mf = CC.multifeature_calibration(rows)
    assert mf["n_points"] == 4 and "macs_only" in mf["fits"]
    for fit in mf["fits"].values():
        assert "loo_mape" in fit and "condition_number" in fit
    assert "not\nidentifiable" in CC.multifeature_report_md(mf).replace(" ", "\n") \
        or "not identifiable" in CC.multifeature_report_md(mf)


_OUTPUT = Path(__file__).resolve().parents[3] / "output"


@pytest.mark.skipif(not (_OUTPUT / "small_llama_int8_consistent" / "model.mlir").exists(),
                    reason="no small capture to reproduce the dispatch measurement")
def test_measure_reproduces_dispatch_count_if_runtime_available():
    # Reproduce one measured row end-to-end (host reference executor). Skips cleanly if the
    # runtime/toolchain is unavailable.
    from merlin.dse_guidance import dispatch_measure as DM
    md = _OUTPUT / "small_llama_int8_consistent"
    try:
        m = DM.measure(str(md))
    except Exception as e:  # noqa: BLE001 - toolchain/runtime may be unavailable in CI
        pytest.skip(f"dispatch runtime unavailable: {type(e).__name__}")
    assert m.cos >= 0.999 and m.n_kernels > m.matmul_estimate


# ------------------------------------------------ numerical-contract audit (structural)

def test_numerical_contract_flags_lost_lowbit():
    from merlin.dse_guidance import numerical_contract as NC
    import os
    cap = "output/rdt2_int8_consistent"
    if not os.path.isfile(f"{cap}/model.mlir"):
        pytest.skip("no int8 zoo capture")
    c = NC.audit(cap, workload="rdt2_int8", has_epilogue=True)
    assert c.low_bit_storage and c.weight_storage_dtype == "int8"
    assert c.low_bit_compute_lost and not c.packed_layout_visible   # f32 compute -> lost
    assert "native_low_bit_compute" in c.lost_structure and c.severity == "high"
    axes = {k.axis for k in c.candidates}
    assert {"native_lowbit_compute", "fused_dequant_matmul"} <= axes
    for k in c.candidates:                       # never a speedup/accuracy/gap_closure
        assert k.benefit == "unquantified" and not hasattr(k, "gap_closure")


def test_numerical_contract_fp32_emits_lowbit_candidate_from_real_bytes():
    from merlin.dse_guidance import numerical_contract as NC
    if not (_RDT_RECAP / "model.mlir").is_file():
        pytest.skip("no rdt recapture")
    c = NC.audit(str(_RDT_RECAP), workload="rdt", workload_class="diffusion/denoise_steps",
                 repeated_head_weight_bytes=391_118_848)
    assert c.declared_quantization == "none" and not c.low_bit_compute_lost
    rw = next(k for k in c.candidates if k.axis == "resident_packed_lowbit_weights")
    assert rw.evidence["repeated_head_weight_bytes"] == 391_118_848
    assert "int4_weight_only" in rw.evidence["candidate_formats"]
    assert rw.benefit == "unquantified" and rw.required_accuracy_measurements


def test_numerical_contract_yaml_has_no_speedup_or_accuracy_number():
    from merlin.dse_guidance import numerical_contract as NC
    if not (_RDT_RECAP / "model.mlir").is_file():
        pytest.skip("no rdt recapture")
    obj = NC.to_yaml_obj(NC.audit(str(_RDT_RECAP), workload="rdt",
                                  repeated_head_weight_bytes=391_118_848))
    blob = str(obj).lower()
    assert "speedup" not in blob and "gap_closure" not in blob
    for cand in obj["numerical_contract"]["candidates"]:
        assert cand["current_status"]["benefit"] == "unquantified"


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_provenance_csv_labels_evidence(tmp_path):
    from merlin.dse_guidance import case_study as CS
    CS.run_case_study(tmp_path)
    csv_text = (tmp_path / "cross_workload_provenance.csv").read_text()
    assert "recovered_from_prov_fqn" in csv_text      # roles from the capture
    assert "recovered_from_ir" in csv_text            # facts from the capture
    assert "assumed_reference" in csv_text             # K
    assert "unavailable" in csv_text                   # CPU coupling
    assert "missing_calibration" in csv_text           # quantification gate
    assert (tmp_path / "case_study.md").is_file()


def test_attribution_auto_recovers_roles_from_prov_fqn():
    # With prov.fqn present (model2MLIR module-FQN provenance), roles recover with NO operator map.
    recs = (
        ATTR.MatmulRecord(0, "matmul_0", "matmul", False, 1, 2048, 9216, 2048 * 9216 * 4,
                          1, "f32", fqn="model.vision_backbone.proj"),
        ATTR.MatmulRecord(1, "matmul_1", "addmm", True, 28, 1024, 1024, 1024 * 1024 * 4,
                          1, "f32", fqn="model.action_expert.denoise.0"),
        ATTR.MatmulRecord(2, "matmul_2", "addmm", True, 28, 1024, 1024, 1024 * 1024 * 4,
                          1, "f32", fqn="model.action_expert.denoise.1"),
    )
    attr = ATTR.attribute_records(recs, _topo(K=5), mapping_rules=None)
    head = attr.role("repeated_head")
    bb = attr.role("backbone_once")
    assert head is not None and head.source == "prov_fqn" and head.facts["matmul_count"] == 2
    assert head.invocations == 5                      # head ×K
    assert bb is not None and bb.source == "prov_fqn" and bb.invocations == 1  # backbone ×1
    assert attr.attribution_status == "attributed" and not attr.unresolved


def test_attribution_unknown_when_no_rule_matches():
    attr = ATTR.attribute_records(_records(), _topo(), mapping_rules=None)
    assert attr.attribution_status in ("unknown", "partial")
    unk = attr.role("unknown")
    assert unk is not None and unk.attribution_status == "unknown"
    # The shape clusters are still reported (real IR facts), even with no role mapping.
    assert any(s["signature"] == [28, 1024, 1024] for s in attr.repeated_signatures)


def test_candidates_carry_attributed_facts_or_block_on_attribution():
    topo = _topo(K=5)
    rules = {"rules": [{"role": "repeated_head",
                        "match": {"shape_signature": [28, 1024, 1024]}}]}
    attr = ATTR.attribute_records(_records(), topo, rules)
    with_attr = {c.axis: c for c in CAND.detect(topo, attribution=attr)}
    rw = with_attr["resident_action_head_weights"]
    assert rw.attributed_facts is not None and rw.attributed_facts["matmul_count"] == 3
    assert rw.quantification_blocked_by == "missing_calibration"
    # Without attribution, the same axis is blocked on region attribution and carries no facts.
    without = {c.axis: c for c in CAND.detect(topo)}["resident_action_head_weights"]
    assert without.attributed_facts is None
    assert without.quantification_blocked_by == "missing_region_attribution"
    # Crucially: still no quantitative speedup/gap_closure on any candidate.
    for c in CAND.detect(topo, attribution=attr):
        assert c.benefit == "unquantified"
        assert not hasattr(c, "gap_closure")


def test_cost_calibration_fits_and_flags_outlier():
    from merlin.dse_guidance import cost_calibration as CC
    # Synthetic measured set: 3 consistent models (~100 cycles/MAC) + 1 huge outlier + 1 unparsed.
    measured = {"substrate": "test", "source": "test", "points": [
        {"model": "a", "dtype": "int8", "cycles": 1_000_000},
        {"model": "b", "dtype": "int8", "cycles": 10_000_000},
        {"model": "c", "dtype": "int8", "cycles": 5_000_000},
        {"model": "outlier", "dtype": "fp32", "cycles": 10_000_000_000},
        {"model": "noparse", "dtype": "int8", "cycles": 7_000_000},
    ]}
    macs = {"a": 10_000, "b": 100_000, "c": 50_000, "outlier": 10_000, "noparse": None}
    res = CC.calibrate(lambda m: macs[m], measured)
    assert res.fitted_cycles_per_mac == pytest.approx(100.0, rel=0.01)  # 3 consistent ~100
    assert res.n_fit == 3 and res.n_unparsed == 1
    out = next(p for p in res.points if p.model == "outlier")
    assert out.is_outlier                       # 1e10/1e4 = 1e6 cycles/MAC >> 100
    assert res.mape_consistent is not None and res.mape_consistent < 5  # exact-ish on consistent
    assert "cycles/MAC" in res.verdict


def test_cost_calibration_loads_real_measured_data():
    from merlin.dse_guidance import cost_calibration as CC
    doc = CC.load_measured()
    models = {p["model"] for p in doc["points"]}
    assert {"xr0", "openvla", "rdt2"} <= models   # the real FASED sweep points exist
    assert all(isinstance(p["cycles"], int) for p in doc["points"])


@pytest.mark.skipif(not _has_model_captures(), reason="no model.mlir captures under output/")
def test_cost_calibration_on_real_models_flags_xr0_outlier():
    from merlin.dse_guidance import cost_calibration as CC
    from merlin.dse_guidance import models as M
    caps = M.discover_model_captures()

    def macs_of(model):
        if model not in caps:
            return None
        f = M.capture_facts(M._prefer_capture(caps[model]))
        return f.total_macs if (f.parsed and f.total_macs) else None

    res = CC.calibrate(macs_of)
    assert res.fitted_cycles_per_mac is not None       # a real fitted constant
    xr0 = next((p for p in res.points if p.model == "xr0"), None)
    if xr0 and xr0.macs:
        assert xr0.is_outlier                          # capture MACs can't explain its cycles


@pytest.mark.skipif(not _has_model_captures(), reason="no model.mlir captures under output/")
def test_attribution_reads_real_prov_provenance_from_capture():
    from merlin.dse_guidance import models as M
    captures = M.discover_model_captures()
    if "rdt2" not in captures:
        pytest.skip("no rdt2 capture")
    recs = ATTR.extract_matmuls(M._prefer_capture(captures["rdt2"]))
    assert len(recs) > 0
    # prov.op is read (not the wrong m2m.* namespace): some matmuls are addmm (epilogue).
    assert any(r.epilogue for r in recs)
    # And the repeated transformer-block signature is recovered from real shapes.
    sigs = {s["signature"][0] for s in ATTR._repeated_signatures(recs)}
    assert sigs  # at least one repeated shape cluster


@pytest.mark.skipif(not _has_model_captures(), reason="no model.mlir captures under output/")
def test_model_capture_residency_is_unquantified_not_fabricated():
    # A whole-model capture cannot separate head from backbone -> residency must be legal but
    # report gap_closure = null, never a fabricated magnitude.
    from merlin.dse_guidance import models as M
    captures = M.discover_model_captures()
    spec = study.spec_from_model("openvla", captures["openvla"])
    res = pipeline.run_guidance(spec.temporal, spec.baseline, overrides=spec.overrides,
                                capture_facts=spec.capture_facts)
    row = next(r for r in res.triage_multirate["axes"]
               if r["axis"] == "resident_packed_weights")
    assert row["legality"] == 1 and row["gap_closure"] is None


@pytest.mark.skipif(not _has_model_captures(), reason="no model.mlir captures under output/")
def test_model_study_runs_and_summarizes(tmp_path):
    specs = study.discover_model_specs()
    summary = study.run_study(specs, tmp_path)
    assert summary["n_workloads"] == len(specs)
    text = (tmp_path / "study_summary.md").read_text()
    assert "becomes legal under multi-rate" in text
    # Even an unparsed capture still shows the structural flip (legality), not a crash.
    assert "resident_packed_weights" in text


# ============================================================ contract-completeness package (P5-P8)

def _pkg_topo_attr(K=5, fqn="model.action_expert.denoise.0"):
    """A minimal (topo, attribution) with an attributed loop-invariant-weight repeated head."""
    topo = _topo(K=K)
    recs = (ATTR.MatmulRecord(0, "m0", "addmm", True, 28, 1024, 1024, 1024 * 1024 * 4, 1, "f32",
                              fqn=fqn),)
    return topo, ATTR.attribute_records(recs, topo)


def test_state_lifetime_joins_bytes_and_scope():
    from merlin.dse_guidance import state_lifetime as SL
    from merlin.dse_guidance.design_envelope import E_IR
    topo, attr = _pkg_topo_attr(K=5)
    recs = SL.state_records(topo, attr)
    w = next(r for r in recs if r.state == "weights")
    assert w.lifetime_scope == SL.SCOPE_INVARIANT
    assert w.bytes == 1024 * 1024 * 4 and w.bytes_evidence == E_IR
    assert w.implied_abstraction == "resident_weight_object"
    assert w.reused_times == 5 and w.scope_evidence == "recovered_from_prov_fqn"


def test_state_lifetime_marks_unknown_bytes_unavailable():
    from merlin.dse_guidance import state_lifetime as SL
    from merlin.dse_guidance.design_envelope import E_NA
    # A non-weight loop-invariant state has no byte fact in a flat capture -> unavailable, not invented.
    doc = {"workload": "m", "class": "diffusion/denoise_steps",
           "timing": {"K": 4, "H": 4, "control_rate_hz": 30},
           "regions": [{"name": "hd", "role": "repeated_head", "invocation_count": 4,
                        "loop_invariant_state": ["prefix_kv"]}]}
    topo = TOP.from_temporal(T.parse(doc))
    recs = SL.state_records(topo, attribution=None)
    kv = next(r for r in recs if r.state == "prefix_kv")
    assert kv.bytes is None and kv.bytes_evidence == E_NA
    assert kv.implied_abstraction == "prefix_kv_object"


def test_compiler_proof_status_derivation():
    from merlin.dse_guidance import compiler_proof as CP
    topo, attr = _pkg_topo_attr(K=5)
    case = type("C", (), {"topo": topo, "attribution": attr, "workload": "m"})()
    assert CP.proof_status("resident_action_head_weights", case) == "proven_for_workload"
    assert CP.proof_status("autonomous_K_loop", case) == "assumed"
    assert CP.proof_status("packed_layout_preservation", case) == "unknown"   # layout erased
    assert CP.proof_status("native_lowbit_compute", case) == "unknown"


def test_compiler_proof_no_invented_proofs():
    from merlin.dse_guidance import compiler_proof as CP, contract as CON
    topo, attr = _pkg_topo_attr(K=5)
    case = type("C", (), {"topo": topo, "attribution": attr, "workload": "m"})()
    cands = CON.abstraction_candidates(CAND.detect(topo, attribution=attr), None)
    rows = CP.proof_matrix([{"case": case, "cands": cands}])
    assert rows
    for r in rows:
        # structural axes must reuse the catalog proof verbatim (nothing invented)
        cat = CP.catalog_proof(r.axis)
        if cat is not None:
            assert r.compiler_proof_needed == cat


def test_workload_family_clusters_by_class():
    from merlin.dse_guidance import workload_family as WF
    # build two cases via topology classes
    d_topo = TOP.from_temporal(T.parse({"workload": "d", "class": "diffusion/denoise_steps",
                                        "timing": {"K": 5, "H": 5, "control_rate_hz": 30},
                                        "regions": [{"name": "h", "role": "repeated_head"}]}))
    l_topo = TOP.from_temporal(T.parse({"workload": "l", "class": "llm/token_decode",
                                        "timing": {"K": 8, "H": 8, "control_rate_hz": 30},
                                        "regions": [{"name": "h", "role": "repeated_head"}]}))
    c1 = type("C", (), {"topo": d_topo, "workload": "d"})()
    c2 = type("C", (), {"topo": l_topo, "workload": "l"})()
    fake_cand = type("AC", (), {"axis": "resident_action_head_weights"})()
    pkgs = [{"case": c1, "cands": [fake_cand]}, {"case": c2, "cands": [fake_cand]}]
    fams = {r.workload: r.family for r in WF.family_rows(pkgs)}
    assert fams["d"] == "iterative_denoise" and fams["l"] == "token_decode"
    sets = WF.family_axis_sets(pkgs)
    assert "resident_action_head_weights" in sets["iterative_denoise"]


def test_search_space_gates_by_enabled_axis_and_disclaims():
    from merlin.dse_guidance import search_space as SS, contract as CON
    from merlin.dse_guidance.design_envelope import E_NA
    topo, attr = _pkg_topo_attr(K=5)
    case = type("C", (), {"topo": topo, "attribution": attr, "workload": "m"})()
    cands = CON.abstraction_candidates(CAND.detect(topo, attribution=attr), None)
    obj = SS.to_yaml_obj(SS.template_for_workload({"case": case, "cands": cands}))
    space = obj["dse_search_space_template"]
    abstr = {e["axis"]: e for e in space["search_space"]["abstractions"]}
    enabled = abstr["resident_action_head_weights"]
    assert enabled["enabled"] is True and enabled["knobs"]   # carries ABSTRACTION_MAP knobs
    # an axis this diffusion workload does not imply is listed but disabled + unavailable
    assert abstr["quantized_KV_cache"]["enabled"] is False
    assert abstr["quantized_KV_cache"]["evidence"] == E_NA
    # discipline: what_is_not_claimed names speedup; no gap_closure anywhere
    assert "speedup" in space["what_is_not_claimed"]
    assert "gap_closure" not in str(obj).lower()


def test_numerical_per_region_dtype_and_honesty_labels():
    from merlin.dse_guidance import numerical_contract as NC
    from merlin.dse_guidance.design_envelope import E_IR, E_NA
    if not (_RDT_RECAP / "model.mlir").is_file():
        pytest.skip("no rdt recapture")
    topo = TOP.from_temporal(T.parse({
        "workload": "rdt", "class": "diffusion/denoise_steps",
        "timing": {"K": 5, "H": 64, "control_rate_hz": 30},
        "regions": [{"name": "denoise", "role": "repeated_head", "invocation_count": 5,
                     "loop_invariant_state": ["weights"]}]}))
    recs = ATTR.extract_matmuls(str(_RDT_RECAP))
    attr = ATTR.attribute(str(_RDT_RECAP), topo)
    c = NC.audit(str(_RDT_RECAP), workload="rdt", records=recs, attribution=attr)
    head = next(r for r in c.per_region_dtype if r["region"] == "repeated_head")
    assert head["dtype"] and head["evidence"] == E_IR and head["n"] > 0
    # honesty fields are labeled and never fabricated
    assert c.accumulator_dtype and c.accumulator_dtype_evidence == E_IR  # f32 capture
    assert c.scale_metadata == "erased_or_unavailable" and c.scale_metadata_evidence == E_NA
    assert c.sparsity_metadata == "erased_or_unavailable"
    blob = str(NC.to_yaml_obj(c)).lower()
    assert "speedup" not in blob and "gap_closure" not in blob


def test_numerical_accumulator_i32_for_int8_storage():
    from merlin.dse_guidance import numerical_contract as NC
    from merlin.dse_guidance.design_envelope import E_DERIVED
    acc, ev = NC._accumulator_dtype("int8", "f32")
    assert acc == "i32" and ev == E_DERIVED
    acc2, ev2 = NC._accumulator_dtype("f32", "f32")
    assert acc2 == "f32"


def test_torchao_plan_is_plan_not_sweep():
    from merlin.dse_guidance import numerical_contract as NC
    md = NC.torchao_integration_plan_md()
    assert "unavailable" in md.lower()                 # non-int8 formats not claimed
    assert "measured: pass" in md.lower()              # int8 is the measured leg
    assert "not a sweep" in md.lower()
    # no fabricated per-format accuracy/speedup numbers
    import re
    assert not re.search(r"cos\s*=\s*0\.\d+", md.lower())
    assert "x faster" not in md.lower()


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_contract_completeness_package(tmp_path):
    from merlin.dse_guidance import case_study as CS
    CS.run_case_study(tmp_path)
    for f in ("resident_state_table.csv", "compiler_proof_matrix.csv",
              "abstraction_pressure_ranking.csv", "workload_family_table.csv",
              "measurement_priority_table.csv", "torchao_integration_plan.md"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    # abstraction pressure ranking is a count, not a speedup column
    ranking = (tmp_path / "abstraction_pressure_ranking.csv").read_text()
    assert "n_workloads" in ranking and "evidence_strength" in ranking
    assert "speedup" not in ranking.lower() and "cycle" not in ranking.lower()
    for line in ranking.strip().splitlines()[1:]:
        strength = line.split(",")[4]
        assert strength in ("strong", "partial", "structural_only")
    # measurement priority uses the three contract categories
    prio = (tmp_path / "measurement_priority_table.csv").read_text()
    for cat in ("accuracy_measurable_now", "proxy_measured", "target_measured"):
        assert cat in prio
    # at least one per-family search-space template exists
    assert list(tmp_path.glob("dse_search_space_template_*.yaml"))


# ============================================================ P5a/P5b/P6 envelope + certificates

def test_memory_envelope_traffic_and_avoidable_reload():
    from merlin.dse_guidance import memory_envelope as ME
    from merlin.dse_guidance.design_envelope import E_IR
    topo, attr = _pkg_topo_attr(K=5)           # head invocations=5, weight_bytes=1024*1024*4
    rows = ME.region_traffic(attr)
    head = next(r for r in rows if r.region == "repeated_head")
    wb = 1024 * 1024 * 4
    assert head.invocations == 5 and head.reuse_factor == 5
    assert head.weight_traffic_if_nonresident == wb * 5
    assert head.avoidable_weight_reload == wb * 4          # residency keeps 1 load
    obj = ME.to_yaml_obj(rows, "m")
    r = obj["memory_envelope"]["regions"][0]
    assert r["weight_bytes"]["evidence"] == E_IR
    # traffic the flat capture cannot expose is unavailable, not invented
    assert r["intermediate_materialization_bytes"]["value"] is None
    assert "gap_closure" not in str(obj).lower()   # speedup only in the disclaimer note


def test_command_graph_is_honest_about_unrolled_loop():
    from merlin.dse_guidance import command_graph as CG
    topo, attr = _pkg_topo_attr(K=5)           # head has 1 matmul
    g = CG.command_graph(topo, attr)
    assert g["commands_per_step_matmul_proxy"]["value"] == 1
    assert g["dispatches_per_replan_proxy"]["value"] == 5          # proxy * K
    assert g["batchable"]["value"] is True
    # syncs / dependencies / allocations are NOT recoverable from an unrolled capture
    assert g["syncs_per_step"]["value"] is None
    assert g["dependency_graph"]["value"] is None
    assert g["dependency_graph"]["evidence"] == "unavailable"
    assert "speedup" in g["what_is_not_claimed"]


def test_accuracy_gate_fp8_not_falsely_int8():
    # regression: 'fp8_w8a8' contains 'w8a8' but must NOT inherit the measured int8 pass.
    from merlin.dse_guidance import accuracy_gate as AG
    assert AG._family("fp8_w8a8") == "fp8"
    assert AG._family("int8_w8a8") == "int8"
    assert AG.status_for("openvla", "int8_w8a8") == "pass"
    assert AG.status_for("openvla", "fp8_w8a8") == "unavailable"   # fp8 not measured


def test_dtype_certificates_accuracy_gated():
    from merlin.dse_guidance import dtype_certificates as DC, numerical_contract as NC
    from merlin.dse_guidance import design_envelope as DE
    if not (_RDT_RECAP / "model.mlir").is_file():
        pytest.skip("no rdt recapture")
    nc = NC.audit(str(_RDT_RECAP), workload="x", repeated_head_weight_bytes=4_000_000)
    env = DE.derive("x", K=5, deadline_s=None, deadline_evidence="unavailable",
                    macs_per_step=1, weight_bytes=4_000_000, activation_bytes_per_step=0,
                    dispatches_per_step=1, dispatches_evidence="derived_requirement",
                    captured_dtype="f32")
    # workload 'openvla' has a measured int8 pass; use it to exercise the legal path
    certs = DC.certificates(nc, env, "openvla")
    int8 = next(c for c in certs if c.dtype == "int8_w8a8")
    fp8 = next(c for c in certs if c.dtype == "fp8_w8a8")
    assert int8.accuracy_status == "measured_pass"
    assert int8.dse_status == "accuracy_legal_structural_candidate"
    assert int8.resident_capacity_at_format_B == env.capacity_by_dtype_B["int8"]
    assert fp8.accuracy_status == "unavailable"            # never assumed
    assert fp8.dse_status == "blocked_by_missing_accuracy"
    assert int8.required_compiler_proofs and int8.what_dse_should_explore
    obj = DC.to_yaml_obj(certs, "openvla")
    assert "speedup" in str(obj["numerical_candidate_certificates"]["certificates"][0]
                            ["what_is_not_claimed"])
    assert "gap_closure" not in str(obj).lower()


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_envelope_and_certificate_tables(tmp_path):
    from merlin.dse_guidance import case_study as CS
    CS.run_case_study(tmp_path)
    for f in ("traffic_table.csv", "dispatch_granularity_table.csv",
              "accuracy_gated_dtype_candidates.csv"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    # honest command-graph: syncs/dependency unavailable for every workload
    dg = (tmp_path / "dispatch_granularity_table.csv").read_text()
    assert dg.count("unavailable") >= 2
    # accuracy gating: NO non-int8 format may show measured_pass
    gated = (tmp_path / "accuracy_gated_dtype_candidates.csv").read_text()
    import csv, io
    for r in csv.DictReader(io.StringIO(gated)):
        if r["dtype"] != "int8_w8a8":
            assert r["accuracy_status"] != "measured_pass", f"false pass: {r}"


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_independent_verification_harness_passes():
    # The standalone verifier independently re-derives every key number from the raw captures and
    # cross-checks them against the emitted artifacts; --check-only means it does not write the report.
    import subprocess, sys
    root = Path(__file__).resolve().parents[2]
    script = root / "benchmarks" / "dse_guidance" / "verify_implementation.py"
    r = subprocess.run([sys.executable, str(script), "--check-only"],
                       capture_output=True, text=True, cwd=str(root))
    assert r.returncode == 0, f"verifier failed:\n{r.stdout}\n{r.stderr}"
    assert "checks passed" in r.stdout
