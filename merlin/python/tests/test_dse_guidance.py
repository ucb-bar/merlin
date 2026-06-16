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
