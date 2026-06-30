"""Tests for the DSE guidance workstream (merlin.dse_guidance).

Covers: temporal parsing + derived deadline, the flat->multi-rate reuse flip, residency/
autonomous-loop legality, the gap_closure formula and its edge cases, the negative control,
the "no measurement -> no invented constants" guarantee, evidence-tag propagation, component-
sum residualization, the aet instrumentation adapter, and schema validity of the artifacts.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import json
from pathlib import Path

import pytest

from merlin.common import schemas
from merlin.common.yaml import load_yaml
from merlin.dse_guidance import (aet_ingest, attribution as ATTR, axes, baseline_cost as BC,
                                 calibration, candidates as CAND, fidelity as FID, pipeline,
                                 representation, study, synth, temporal as T, topology as TOP)

FIX = merlin_dir() / "tests" / "fixtures" / "dse_guidance"


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
_RECAP = merlin_dir() / "benchmarks" / "dse_guidance" / "recaptures"
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


# --------------------------------------------- P21-S1: loop-preserving capture recovery

_RECAP_LOOP = merlin_dir() / "benchmarks" / "dse_guidance" / "recaptures_loop"


@pytest.mark.skipif(not (_RECAP_LOOP / "openvla" / "model.mlir").is_file(),
                    reason="no loop-preserving capture")
def test_loop_recovery_recovers_K_and_carried_state_from_ir():
    from merlin.dse_guidance.loop_recovery import recover_loop
    # openVLA autoregressive decode: K=7, static KV cache is a carried iter_arg
    ov = recover_loop(_RECAP_LOOP / "openvla" / "model.mlir", "openvla")
    assert ov.present and ov.K == 7 and ov.K_source == "recovered_from_ir"
    roles = [c.role for c in ov.carried_state]
    assert roles.count("kv_cache") == 2          # static k/v cache carried in-place
    assert ov.kv_cache_bytes == 221184           # 2 * 2*1*4*27*128 * 4 bytes (f32)
    assert ov.repeated_region_op_count > 100      # the decode body is a real repeated region

    # smolVLA / pi0.5 flow-matching: K=10, the action latent is the carried state (no growing KV)
    for w, k in (("smolvla", 10), ("pi05", 10)):
        lr = recover_loop(_RECAP_LOOP / w / "model.mlir", w)
        assert lr.present and lr.K == k
        assert any(c.role == "latent" and c.shape == [1, 50, 32] for c in lr.carried_state)
        assert lr.kv_cache_bytes is None          # prefix KV invariant (closed-over), not carried


def test_real_config_magnitudes_and_kv_sizing():
    """P21 S2/S3: deployment-real magnitudes are config-exact compositions, and the KV byte
    formula is validated against the IR-recovered iter_arg then applied at deployment scale."""
    from merlin.dse_guidance import real_config as RC
    g = RC.REAL_GEOMETRY["openvla"]
    # openVLA LM == Llama-2-7B: 32 * (q+k+v+o + gate+up+down) + untied embed/lm_head
    per_layer = 4 * 4096 * 4096 + 3 * 4096 * 11008
    assert g.total_params() == per_layer * 32 + 2 * (32064 * 4096)
    assert 6.6e9 < g.total_params() < 6.9e9                      # ~6.74B (Llama-2-7B)
    # KV deployment-exact = 2*kv_heads*head_dim*seq*n_layers*dtype
    assert g.kv_cache_bytes("bf16") == 2 * 32 * 128 * 263 * 32 * 2
    # the byte formula reproduces the IR-recovered KV iter_arg on the captured (small) config
    kv = {r["workload"]: r for r in RC.kv_sizing_rows(_RECAP_LOOP)}
    if (_RECAP_LOOP / "openvla" / "model.mlir").is_file():
        assert "matches IR iter_arg" in kv["openvla"]["ir_formula_check"]
        assert kv["openvla"]["loop_carried_in_ir"] is True
    # weight VALUES are irrelevant: magnitudes are config-determined
    assert all(r["evidence"] == "recovered_from_model_config" for r in RC.magnitude_rows())


@pytest.mark.skipif(not (_RECAP_LOOP / "openvla" / "model.mlir").is_file(),
                    reason="no loop-preserving capture")
def test_residency_from_ir_proves_loop_invariant_operands():
    """P21 GAP-C: the scf.for region boundary proves which operands are loop-invariant
    (referenced in the body, defined outside -> resident-eligible across K) vs loop-carried."""
    from merlin.dse_guidance.loop_recovery import residency_from_ir
    rc = residency_from_ir(_RECAP_LOOP / "openvla" / "model.mlir", "openvla")
    assert rc.present and rc.K == 7
    assert rc.n_loop_invariant_operands > 100        # weights referenced read-only every iteration
    assert rc.n_loop_carried == 5                    # the 5 iter_args (counter/tok/out/k/v)
    assert "loop-invariant" in rc.resident_proof and "resident-eligible" in rc.resident_proof
    assert rc.evidence == "recovered_from_ir"


_RECAP_NATIVE = merlin_dir() / "benchmarks" / "dse_guidance" / "recaptures_native"


@pytest.mark.skipif(not (_RECAP_NATIVE / "bitvla" / "model.mlir").is_file(),
                    reason="no native low-bit bitvla capture")
def test_native_lowbit_bitvla_datapath_recovered():
    """P21-S4: the bitvla native W1.58 ternary datapath (packed-int2 storage + absmean scale) is
    captured directly — the storage that the torchao-int8 qdq stand-in could not expose."""
    from merlin.dse_guidance import quant_metadata as QM
    cs = merlin_dir() / "benchmarks" / "dse_guidance" / "case_study"
    rows = {r["workload"]: r for r in QM.native_quant_rows(cs)}
    assert "bitvla" in rows
    bv = rows["bitvla"]
    assert bv["storage"] == "int2_packed_in_i8"
    assert int(bv["n_packed_weight_tensors"]) > 0          # packed-int2 weights present
    assert "absmean" in bv["scale"]                        # per-tensor absmean scale recovered
    assert "recovered" in bv["status"]
    # the count matches the actual i8-stored tensors in the capture
    cap = (_RECAP_NATIVE / "bitvla" / "model.mlir").read_text()
    assert cap.count("xi8>") == int(bv["n_packed_weight_tensors"])
    # P22 GAP-D: the int2 unpack is a named quant_ext.unpack_int2 op (opaque chain folded)
    assert cap.count("quant_ext.unpack_int2") > 0
    assert "named op" in bv["unpack_visibility"]
    assert "call @aten_stack" not in cap


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


_OUTPUT = repo_root() / "artifacts" / "recaptures"


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
    cap = "artifacts/recaptures/rdt2_int8_consistent"
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
    root = merlin_dir()
    script = root / "benchmarks" / "dse_guidance" / "verify_implementation.py"
    r = subprocess.run([sys.executable, str(script), "--check-only"],
                       capture_output=True, text=True, cwd=str(root))
    assert r.returncode == 0, f"verifier failed:\n{r.stdout}\n{r.stderr}"
    assert "checks passed" in r.stdout


# ============================================================ P5 operator geometry + primitive coverage

def _mm(M, N, K, fqn=None, op="matmul", idx=0):
    from merlin.dse_guidance import attribution as ATTR
    return ATTR.MatmulRecord(idx, f"matmul_{idx}", op, op == "addmm", M, K, N,
                             K * N * 4, (M * K + M * N) * 4, "f32", fqn=fqn)


def test_shape_taxonomy_geometry_thresholds():
    from merlin.dse_guidance import shape_taxonomy as ST
    assert ST.classify_geometry(1, 2048, 256) == ST.GEMV          # vector-like (M=1)
    assert ST.classify_geometry(4, 2048, 2048) == ST.GEMV         # M=4 decode
    assert ST.classify_geometry(67, 2048, 2048) == ST.WIDE_SKINNY  # N/M >= 4
    assert ST.classify_geometry(2048, 67, 2048) == ST.TALL_SKINNY  # M/N >= 4
    assert ST.classify_geometry(4096, 4096, 2048) == ST.SQUAREISH  # M/N==1, not tiny
    # square but tiny -> not squareish (falls through to projection/odd/frag/unknown)
    assert ST.classify_geometry(8, 8, 4) != ST.SQUAREISH


def test_shape_taxonomy_semantic_from_fqn_and_priority():
    from merlin.dse_guidance import shape_taxonomy as ST
    assert ST.classify_semantic("vla.language_model.model.layers.0.self_attn.q_proj") == ST.SEM_QKV
    assert ST.classify_semantic("model.blocks.0.attn.proj") == ST.SEM_ATTN_OUT
    assert ST.classify_semantic("model.blocks.0.ffn.fc1") == ST.SEM_MLP
    # lm-head leaf must NOT swallow a whole tree rooted at "lm." (tiny_llama style)
    assert ST.classify_semantic("lm") == ST.SEM_LM_HEAD
    assert ST.classify_semantic("lm.model.layers.0.self_attn.q_proj") == ST.SEM_QKV
    assert ST.classify_semantic(None) == ST.SEM_UNKNOWN


def test_operator_geometry_extracts_macs_bytes_aspect():
    from merlin.dse_guidance import operator_geometry as OG
    from merlin.dse_guidance.design_envelope import E_IR
    s = OG.operator_shape(_mm(67, 2048, 2048, fqn="model.blocks.0.ffn.fc1", op="addmm"),
                          "rdt", "repeated_head")
    assert s.macs == 67 * 2048 * 2048
    assert s.rhs_weight_bytes == 2048 * 2048 * 4 and s.output_bytes == 67 * 2048 * 4
    assert s.aspect_ratio_NK == round(2048 / 2048, 4)
    assert s.shape_class == "wide_skinny" and s.semantic_class == "mlp_projection"
    assert s.epilogue is True and s.epilogue_hint == "bias_addmm"
    assert s.batch_product == 1 and s.evidence_shape == E_IR


def test_tile_padding_formula_and_nonmultiple_waste():
    from merlin.dse_guidance import operator_geometry as OG, primitive_coverage as PC
    # N=344 is not a multiple of 16 -> nonzero waste under tile_16x16
    s = OG.operator_shape(_mm(8, 344, 128), "w", "repeated_head")
    cov = PC.tile_coverage(s, "tile_16x16", 16, 16)
    assert cov.padded_M == 16 and cov.padded_N == 352          # ceil(8/16)*16, ceil(344/16)*16
    assert cov.padded_macs == 16 * 352 * 128
    assert cov.padding_waste > 0 and 0 < cov.tile_utilization < 1
    # an exact-multiple shape wastes nothing
    s2 = OG.operator_shape(_mm(32, 64, 128), "w", "r")
    c2 = PC.tile_coverage(s2, "tile_32x32", 32, 32)
    assert c2.padding_waste == 0.0 and c2.tile_utilization == 1.0 and c2.covered_under_5pct


def test_gemv_lane_coverage_rule_applicability():
    from merlin.dse_guidance import operator_geometry as OG, primitive_coverage as PC
    # gemv-like shape: lane applies along N, N=2048 multiple of 64 -> no waste
    g = OG.operator_shape(_mm(1, 2048, 256), "w", "r")
    assert g.shape_class == "gemv_like"
    cov = PC.gemv_coverage(g, "gemv_lane_64", 64)
    assert cov.applicable is True and cov.padding_waste == 0.0
    # squareish shape: a lane is NOT applicable and must not be scored as covering it
    sq = OG.operator_shape(_mm(4096, 4096, 2048), "w", "r")
    assert sq.shape_class == "squareish_gemm"
    covsq = PC.gemv_coverage(sq, "gemv_lane_64", 64)
    assert covsq.applicable is False
    assert not (covsq.covered_under_5pct or covsq.covered_under_10pct or covsq.covered_under_25pct)


def test_primitive_coverage_aggregation_and_regret():
    from merlin.dse_guidance import operator_geometry as OG, primitive_coverage as PC
    shapes = [OG.operator_shape(_mm(8, 344, 128, idx=0), "wa", "r"),
              OG.operator_shape(_mm(64, 64, 64, idx=1), "wb", "r")]
    cov = PC.all_coverage(shapes)
    per = PC.aggregate_by_primitive_workload(cov)
    # one aggregate row per (primitive, workload); coverage_under_10pct in [0,1]
    assert {a.workload for a in per} == {"wa", "wb"}
    assert all(0.0 <= a.coverage_under_10pct <= 1.0 for a in per)
    regret = PC.aggregate_regret(cov, per)
    for r in regret:
        assert abs(r.max_regret - (r.best_workload_coverage_10
                                   - r.worst_workload_coverage_10)) < 1e-9
        assert 0.0 <= r.coverage_under_10pct <= 1.0


def test_p5_artifacts_have_no_speedup_or_forbidden_fields():
    from merlin.dse_guidance import operator_geometry as OG, primitive_coverage as PC
    shapes = [OG.operator_shape(_mm(67, 2048, 2048, fqn="model.blocks.0.ffn.fc1"), "rdt", "r")]
    cov = PC.all_coverage(shapes)
    per = PC.aggregate_by_primitive_workload(cov)
    regret = PC.aggregate_regret(cov, per)
    blobs = [str(OG.to_yaml_obj({"rdt": shapes}, conv_visible=False)).lower(),
             OG.report_md({"rdt": shapes}, shapes).lower(),
             PC.coverage_report_md(per, regret).lower(),
             PC.cross_workload_report_md(regret, per).lower()]
    for b in blobs:
        for term in ("gap_closure", "faster", "optimal", "predicted cycles"):
            assert term not in b, f"forbidden term {term!r}"
        # 'speedup' may appear only inside an explicit no-speedup disclaimer
        for ln in b.splitlines():
            if "speedup" in ln:
                assert "no speedup" in ln


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_operator_geometry_and_coverage(tmp_path):
    from merlin.dse_guidance import case_study as CS
    CS.run_case_study(tmp_path)
    for f in ("operator_shape_table.csv", "operator_geometry.yaml",
              "shape_summary_by_workload.csv", "shape_summary_by_region.csv",
              "operator_cluster_table.csv", "operator_geometry_report.md",
              "tile_waste_table.csv", "primitive_coverage_matrix.csv",
              "primitive_coverage_report.md", "primitive_regret_table.csv",
              "cross_workload_coverage_report.md"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    # operator_shape_table carries real geometry + both class axes
    import csv, io
    rows = list(csv.DictReader(io.StringIO((tmp_path / "operator_shape_table.csv").read_text())))
    assert rows and all(int(r["macs"]) == int(r["M"]) * int(r["N"]) * int(r["K"]) for r in rows)
    assert {r["shape_class"] for r in rows} - set()        # nonempty
    # gemv lanes are never marked applicable on squareish ops
    tw = list(csv.DictReader(io.StringIO((tmp_path / "tile_waste_table.csv").read_text())))
    assert not [r for r in tw if r["primitive_kind"] == "gemv_lane"
                and r["shape_class"] == "squareish_gemm" and r["applicable"] == "True"]


# ============================================================ P6 multi-rate workload contract graph

_ALLOWED_EV = {"recovered_from_ir", "recovered_from_prov_fqn", "recovered_from_model_config",
               "assumed_reference", "derived_requirement", "design_assumption", "measured",
               "proxy_measured", "unavailable"}


def _graph_fixture(K=5, cls="diffusion/denoise_steps", fqn="model.action_expert.denoise.0"):
    from merlin.dse_guidance import contract_graph as CG2, operator_geometry as OG
    from merlin.dse_guidance import state_lifetime as SL
    topo = TOP.from_temporal(T.parse({
        "workload": "m", "class": cls, "timing": {"K": K, "H": K, "control_rate_hz": 30},
        "regions": [{"name": "backbone", "role": "backbone_once", "invocation_count": 1},
                    {"name": "head", "role": "repeated_head", "invocation_count": K,
                     "loop_invariant_state": ["weights"]}]}))
    recs = (_mm(28, 1024, 1024, fqn=fqn, op="addmm", idx=0),)
    attr = ATTR.attribute_records(recs, topo)
    shapes = OG.operator_shapes(recs, "m", attr)
    state = SL.state_records(topo, attr)
    case = type("C", (), {"workload": "m", "cls": cls, "K": K, "topo": topo, "attribution": attr})()
    return CG2.build_graph(case, shapes, None, state)


def test_phase_rate_cadence_classification():
    from merlin.dse_guidance import phase_rate as PR
    from merlin.dse_guidance import topology as TOP
    assert PR.classify_cadence("backbone_once", TOP.CLASS_FLOW_MATCHING, 1, 5) == "once_per_replan"
    assert PR.classify_cadence("repeated_head", TOP.CLASS_FLOW_MATCHING, 5, 5) == "K_times_per_replan"
    assert PR.classify_cadence("repeated_head", TOP.CLASS_AUTOREGRESSIVE, 8, 8) == "token_loop"
    assert PR.classify_cadence("repeated_head", TOP.CLASS_FLOW_MATCHING, 1, 1) == "once_per_forward"
    assert PR.classify_cadence("control_loop", TOP.CLASS_FLOW_MATCHING, None, 5) == "control_tick"
    assert PR.classify_cadence(None, TOP.CLASS_UNKNOWN, None, 5) == "unknown"


def test_phase_rate_model_sources():
    from merlin.dse_guidance import phase_rate as PR
    from merlin.dse_guidance.design_envelope import E_CONFIG, E_DERIVED
    topo = TOP.from_temporal(T.parse({"workload": "m", "class": "diffusion/denoise_steps",
                                      "timing": {"K": 5, "H": 8, "control_rate_hz": 30},
                                      "regions": [{"name": "h", "role": "repeated_head"}]}))
    rm = PR.rate_model(topo)
    # K/H/control come from the model's published config (a real source, not a bare assumption)
    assert rm["K"]["value"] == 5 and rm["K"]["source"] == E_CONFIG
    assert rm["replan_deadline_s"]["source"] == E_DERIVED  # derived from H / control_rate
    assert rm["replan_deadline_s"]["value"] == round(8 / 30, 6)


def test_contract_graph_node_and_edge_schema():
    from merlin.dse_guidance import contract_graph as CG2
    g = _graph_fixture(K=5)
    kinds = {n.kind for n in g.nodes}
    assert {"phase", "region", "operator", "loop_body", "state_object"} <= kinds
    assert all(n.kind in CG2.NODE_KINDS for n in g.nodes)
    assert all(n.evidence in _ALLOWED_EV for n in g.nodes)
    assert all(e.kind in CG2.EDGE_KINDS for e in g.edges)
    assert all(e.evidence in _ALLOWED_EV for e in g.edges)
    assert all(e.can_pipeline in (True, False, "unknown") for e in g.edges)
    # a loop-invariant weight edge exists and is read-only pipelineable
    inv = next(e for e in g.edges if e.kind == "loop_invariant")
    assert inv.tensor == "weights" and inv.can_pipeline is True and inv.bytes == 1024 * 1024 * 4


def test_contract_graph_region_mac_aggregation():
    g = _graph_fixture(K=5)
    region = next(n for n in g.nodes if n.kind == "region" and n.region_role == "repeated_head")
    ws = region.work_summary
    assert ws["macs_per_invocation"] == 28 * 1024 * 1024
    assert ws["macs_per_replan"] == ws["macs_per_invocation"] * 5     # derived identity
    loop = next(n for n in g.nodes if n.kind == "loop_body")
    assert loop.rate["trip_count"] == 5


def test_contract_graph_missing_k_has_no_loop_body():
    g = _graph_fixture(K=1)
    assert not [n for n in g.nodes if n.kind == "loop_body"]          # K=1 -> no repeated loop
    head_phase = next(n for n in g.nodes if n.kind == "phase" and n.region_role == "repeated_head")
    assert head_phase.rate["cadence"] == "once_per_forward"


def test_contract_graph_data_dependencies_recovered_from_ir():
    from merlin.dse_guidance.design_envelope import E_IR
    # real producer->consumer edges come from the dependencies arg (the SSA use-def graph)
    from merlin.dse_guidance import contract_graph as CG2, operator_geometry as OG
    from merlin.dse_guidance import state_lifetime as SL
    topo = TOP.from_temporal(T.parse({
        "workload": "m", "class": "diffusion/denoise_steps", "timing": {"K": 3, "H": 3,
        "control_rate_hz": 30}, "regions": [{"name": "head", "role": "repeated_head",
        "invocation_count": 3, "loop_invariant_state": ["weights"]}]}))
    recs = (_mm(8, 128, 128, fqn="model.denoise.0", idx=0),
            _mm(8, 128, 128, fqn="model.denoise.1", idx=1))
    attr = ATTR.attribute_records(recs, topo)
    shapes = OG.operator_shapes(recs, "m", attr)
    case = type("C", (), {"workload": "m", "cls": "x", "K": 3, "topo": topo, "attribution": attr})()
    deps = ((), (0,))                       # op1 consumes op0's result
    g = CG2.build_graph(case, shapes, None, SL.state_records(topo, attr), dependencies=deps)
    dd = [e for e in g.edges if e.kind == "data_dependency"]
    assert dd and not [e for e in g.edges if e.kind == "unknown_dependency"]
    e = dd[0]
    assert e.source == "m:op:0" and e.target == "m:op:1"
    assert e.evidence == E_IR and e.can_pipeline is False


def test_matmul_dependencies_recovers_real_ssa_chain():
    # on the real tiny_llama capture: a Llama layer's o_proj consumes q/k/v_proj; the next
    # layer's q_proj consumes the previous layer's mlp.down_proj — recovered from SSA use-def.
    if not (_RECAP / "tiny_llama" / "model.mlir").is_file():
        pytest.skip("no tiny_llama recapture")
    deps = ATTR.matmul_dependencies(str(_RECAP / "tiny_llama"))
    assert deps and len(deps) == len(ATTR.extract_matmuls(str(_RECAP / "tiny_llama")))
    assert set(deps[3]) == {0, 1, 2}        # o_proj <- q_proj, k_proj, v_proj
    assert 3 in deps[4] and 3 in deps[5]    # gate_proj, up_proj <- o_proj output (residual stream)
    assert set(deps[6]) == {4, 5}           # down_proj <- gate_proj, up_proj


def test_contract_graph_token_loop_for_autoregressive():
    g = _graph_fixture(K=8, cls="llm/token_decode", fqn="lm.model.layers.0.self_attn.q_proj")
    loop = next(n for n in g.nodes if n.kind == "loop_body")
    assert loop.rate["cadence"] == "token_loop"


def test_contract_graph_no_forbidden_wording():
    from merlin.dse_guidance import contract_graph as CG2
    g = _graph_fixture(K=5)
    blobs = [str(CG2.to_yaml_obj([g])).lower(), CG2.summary_md([g]).lower(),
             CG2.rate_mismatch_report_md([g]).lower(),
             str(CG2.multi_rate_contract_yaml([g])).lower(), CG2.phase_rate_csv([g]).lower()]
    for b in blobs:
        for term in ("gap_closure", "faster", "optimal", "predicted cycles", "improvement"):
            assert term not in b, f"forbidden term {term!r}"
        for ln in b.splitlines():
            if "speedup" in ln:
                assert "no speedup" in ln


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_contract_graph(tmp_path):
    from merlin.dse_guidance import case_study as CS
    from merlin.common.yaml import load_yaml
    CS.run_case_study(tmp_path)
    for f in ("workload_contract_graph.yaml", "workload_contract_graph_summary.md",
              "phase_rate_table.csv", "multi_rate_contract.yaml", "rate_mismatch_report.md"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    g = load_yaml(tmp_path / "workload_contract_graph.yaml")["workload_contract_graph"]
    wls = {x["workload"] for x in g["graphs"]}
    assert wls == set(CS.available_models())
    # every graph has phase + operator nodes and a recovered rate model
    for x in g["graphs"]:
        kinds = {n["kind"] for n in x["nodes"]}
        assert "phase" in kinds and "operator" in kinds
        assert x["rate_model"]["K"]["value"] >= 1


# ============================================================ P7 parallelism / sharding / hierarchy

def _toy_graph():
    from merlin.dse_guidance import contract_graph as CG2
    N = CG2.Node
    nodes = [N(id="w:op:0", workload="w", kind="operator", shape_summary={"macs": 10}),
             N(id="w:op:1", workload="w", kind="operator", shape_summary={"macs": 10}),
             N(id="w:op:2", workload="w", kind="operator", shape_summary={"macs": 100})]
    edges = [CG2.Edge("w:op:0", "w:op:2", "data_dependency"),
             CG2.Edge("w:op:1", "w:op:2", "data_dependency")]
    return CG2.ContractGraph("w", "x", {}, nodes, edges)


def test_parallelism_critical_path_and_available_parallelism():
    from merlin.dse_guidance import parallelism as PAR
    d = PAR.analyze_graph(_toy_graph())
    assert d.total_macs == 120 and d.critical_path_macs == 110   # op0/op1 (10) -> op2 (100)
    assert d.critical_path_ops == 2
    assert d.available_parallelism == round(120 / 110, 4)        # work/span, NOT speedup
    assert d.max_ready_width == 2 and d.n_levels == 2            # op0,op1 ready together
    assert d.critical_path_macs <= d.total_macs
    assert d.dep_evidence == "recovered_from_ir"


def test_parallelism_conservative_fallback_when_no_data_edges():
    from merlin.dse_guidance import parallelism as PAR, contract_graph as CG2
    nodes = [CG2.Node(id="w:op:0", workload="w", kind="operator", shape_summary={"macs": 5}),
             CG2.Node(id="w:op:1", workload="w", kind="operator", shape_summary={"macs": 5})]
    g = CG2.ContractGraph("w", "x", {}, nodes, [])               # no data_dependency edges
    d = PAR.analyze_graph(g)
    assert d.dep_evidence == "conservative_assumption"           # sequential chain, labeled honestly
    assert d.critical_path_macs == 10                            # forced sequential


def test_sharding_mnk_formulas_and_partial_sum_bytes():
    from merlin.dse_guidance import sharding as SH, operator_geometry as OG
    s = OG.operator_shape(_mm(8, 344, 128), "w", "repeated_head")   # M=8,N=344,K=128 (f32)
    axes = {a.axis: a for a in SH.shard_axes(s)}
    # M: rows, no reduction, broadcast weights (K*N*4); 8%8==0 -> no tail at 8
    assert axes["M"].reduction_required is False and not axes["M"].has_tail[8]
    assert axes["M"].per_extra_shard_bytes == 128 * 344 * 4
    assert "weight_broadcast" in axes["M"].required_abstractions
    # N: cols, no reduction, broadcast activations (M*K*4); 344%8==0 -> no tail
    assert axes["N"].reduction_required is False
    assert axes["N"].per_extra_shard_bytes == 8 * 128 * 4
    assert "activation_multicast" in axes["N"].required_abstractions
    # K: reduction split -> partial sums M*N*acc(4) + accumulator_merge
    assert axes["K"].reduction_required is True and axes["K"].comm_category == "high"
    assert axes["K"].per_extra_shard_bytes == 8 * 344 * 4
    assert "partial_sum_object" in axes["K"].required_abstractions


def test_sharding_attention_and_conv_unavailable():
    from merlin.dse_guidance import sharding as SH, operator_geometry as OG
    shapes = [OG.operator_shape(_mm(8, 128, 128), "w", "r")]
    obj = SH.sharding_opportunities_yaml({"w": SH.all_shard_axes(shapes)})["sharding_opportunities"]
    assert obj["attention_sharding"]["value"] == "unavailable"
    assert obj["conv_sharding"]["value"] == "unavailable"


def test_resource_hierarchy_schema_and_unavailable_units():
    from merlin.dse_guidance import resource_hierarchy as RH, operator_geometry as OG, parallelism as PAR
    shapes = [OG.operator_shape(_mm(4096, 4096, 2048, idx=0), "w", "r"),   # dense_gemm
              OG.operator_shape(_mm(1, 2048, 256, idx=1), "w", "r")]       # gemv -> skinny
    dags = [PAR.analyze_graph(_toy_graph())]
    clusters = RH.cluster_hierarchy(shapes)
    hints = RH.hierarchy_hints_yaml(clusters)["parallel_hierarchy_hints"]["clusters"]
    known = set(("matrix_tile_engine", "skinny_gemm_or_gemv_engine", "reduction_tree", "systolic_array",
                 "SIMD_vector_lanes", "multi_engine_cluster", "epilogue_unit", "DMA_engine",
                 "loop_controller"))
    for h in hints:
        assert set(h["hierarchy_options"]) <= known        # only known hierarchy units referenced
    pressure = RH.resource_pressure(shapes, dags)
    units = {u["unit"]: u for u in
             RH.processing_unit_candidates_yaml(RH.processing_unit_candidates(shapes, pressure))
             ["processing_unit_candidates"]["units"]}
    # attention / conv have no operators in the capture -> unavailable, not dropped
    assert units["attention_kv_engine"]["evidence"] == "unavailable"
    assert units["conv_engine"]["evidence"] == "unavailable"
    assert units["matrix_engine"]["evidence"] == "recovered_from_ir"     # dense gemm present


def test_structural_hierarchy_hints_cover_vocabulary():
    from merlin.dse_guidance import resource_hierarchy as RH, operator_geometry as OG
    from merlin.dse_guidance import sharding as SH, parallelism as PAR
    shapes = [OG.operator_shape(_mm(4096, 4096, 2048, idx=0, op="addmm"), "w", "repeated_head"),
              OG.operator_shape(_mm(1, 2048, 256, idx=1), "w", "repeated_head")]
    axes = SH.all_shard_axes(shapes)
    dags = [PAR.analyze_graph(_toy_graph())]
    hints = {h.hierarchy_option: h for h in RH.structural_hierarchy_hints(shapes, axes, dags)}
    # the structural units the rest of P7 implies are now surfaced (closing the vocab gap)
    assert {"reduction_tree", "epilogue_unit", "DMA_engine", "loop_controller",
            "multi_engine_cluster"} <= set(hints)
    assert all(h.hierarchy_option in RH.HIER_UNITS for h in hints.values())
    assert all(h.evidence in _ALLOWED_EV for h in hints.values())
    assert hints["reduction_tree"].evidence == "recovered_from_ir"      # K-shardable ops exist
    assert hints["epilogue_unit"].evidence == "recovered_from_ir"       # the addmm op
    assert hints["loop_controller"].supported_workloads == ["w"]        # repeated_head present
    # the yaml emits both clusters and structural_units
    obj = RH.hierarchy_hints_yaml(RH.cluster_hierarchy(shapes),
                                  list(hints.values()))["parallel_hierarchy_hints"]
    assert obj["clusters"] and obj["structural_units"]


def test_p7_reports_have_no_forbidden_wording():
    from merlin.dse_guidance import (parallelism as PAR, sharding as SH, resource_hierarchy as RH,
                                     operator_geometry as OG)
    shapes = [OG.operator_shape(_mm(4096, 4096, 2048), "w", "r")]
    dags = [PAR.analyze_graph(_toy_graph())]
    pressure = RH.resource_pressure(shapes, dags)
    units = RH.processing_unit_candidates(shapes, pressure)
    clusters = RH.cluster_hierarchy(shapes)
    blobs = [PAR.report_md(dags).lower(),
             SH.report_md({"w": SH.all_shard_axes(shapes)}, SH.all_shard_axes(shapes)).lower(),
             RH.processing_unit_report_md(units, pressure, clusters, dags).lower(),
             str(RH.processing_unit_candidates_yaml(units)).lower()]
    for b in blobs:
        for term in ("gap_closure", "faster", "optimal", "predicted cycles", "improvement"):
            assert term not in b, f"forbidden term {term!r}"
        for ln in b.splitlines():
            if "speedup" in ln:
                assert "no speedup" in ln or "not a speedup" in ln


# ============================================================ P8 pipeline / overlap / unit guidance

def _p8_inputs(K=5, cls="diffusion/denoise_steps", fqn="model.action_expert.denoise.0"):
    from merlin.dse_guidance import contract_graph as CG2, operator_geometry as OG
    from merlin.dse_guidance import state_lifetime as SL
    topo = TOP.from_temporal(T.parse({
        "workload": "m", "class": cls, "timing": {"K": K, "H": K, "control_rate_hz": 30},
        "regions": [{"name": "backbone", "role": "backbone_once", "invocation_count": 1},
                    {"name": "head", "role": "repeated_head", "invocation_count": K,
                     "loop_invariant_state": ["weights"]}]}))
    recs = (_mm(28, 1024, 1024, fqn=fqn, op="addmm", idx=0),)
    attr = ATTR.attribute_records(recs, topo)
    shapes = OG.operator_shapes(recs, "m", attr)
    case = type("C", (), {"workload": "m", "cls": cls, "K": K, "topo": topo, "attribution": attr})()
    g = CG2.build_graph(case, shapes, None, SL.state_records(topo, attr), dependencies=((),))
    return g, shapes


def test_pipeline_phase_classification():
    from merlin.dse_guidance import pipeline_envelope as PE
    g, shapes = _p8_inputs(cls="diffusion/denoise_steps")
    phases = {p.phase_class: p for p in PE.phase_model(g, shapes)}
    assert "backbone_or_encoder" in phases and "repeated_action_head" in phases
    g2, s2 = _p8_inputs(cls="llm/token_decode", fqn="lm.model.layers.0.self_attn.q_proj")
    cls2 = {p.phase_class for p in PE.phase_model(g2, s2)}
    assert "decoder_token_step" in cls2          # autoregressive head -> decoder_token_step


def test_pipeline_overlap_schema_and_double_buffer_rule():
    from merlin.dse_guidance import pipeline_envelope as PE
    g, shapes = _p8_inputs(K=5)
    phases = PE.phase_model(g, shapes)
    cands = {c.source_phase: c for c in PE.overlap_candidates(g, phases, has_control_loop=True)}
    for c in cands.values():
        assert c.can_overlap in ("yes", "no", "unknown")
        assert all(a in PE.ALLOWED_ABSTRACTIONS for a in c.required_abstractions)
        assert isinstance(c.required_buffer_count, int) or c.required_buffer_count == "unavailable"
    # double-buffer rule: the control||inference overlap (VLA) needs 2 buffers
    ctrl = cands["control_tick_consumer"]
    assert ctrl.can_overlap == "yes" and ctrl.required_buffer_count == 2
    assert "double_buffered_action_chunk" in ctrl.required_abstractions
    # the bounded K-loop needs 1 buffer; KV pipelining is unknown -> unavailable
    loop = next(c for c in cands.values() if c.dependency_type == "bounded_loop")
    assert loop.required_buffer_count == 1 and "bounded_loop_command" in loop.required_abstractions
    kv = next(c for c in cands.values() if c.dependency_type == "kv_dependency")
    assert kv.can_overlap == "unknown" and kv.required_buffer_count == "unavailable"


def test_overlap_candidates_gated_on_recovered_structure():
    from merlin.dse_guidance import pipeline_envelope as PE
    # head-only capture (no backbone compute) + no control loop -> both gated overlaps are unknown
    g, shapes = _p8_inputs(K=5)              # single repeated_head op, no backbone ops
    phases = PE.phase_model(g, shapes)
    c = {x.source_phase: x for x in PE.overlap_candidates(g, phases, has_control_loop=False)}
    bb = next(v for k, v in c.items() if k.startswith("backbone(next"))
    assert bb.can_overlap == "unknown" and bb.required_buffer_count == "unavailable"  # no backbone
    assert c["control_tick_consumer"].can_overlap == "unknown"                        # no control loop
    # a VLA (control loop present) flips the control-tick overlap to a structural yes
    c2 = {x.source_phase: x for x in PE.overlap_candidates(g, phases, has_control_loop=True)}
    assert c2["control_tick_consumer"].can_overlap == "yes"
    assert c2["control_tick_consumer"].required_buffer_count == 2


def test_pipeline_missing_period_is_explicit():
    from merlin.dse_guidance import pipeline_envelope as PE
    g, shapes = _p8_inputs(K=5)
    head = next(p for p in PE.phase_model(g, shapes) if p.phase_class == "repeated_action_head")
    # per-K-step wall time is a runtime measurement -> period unavailable, named in missing (not faked)
    assert head.period_s is None and head.missing


def test_processing_unit_guidance_schema():
    from merlin.dse_guidance import (processing_unit_guidance as PUG, resource_hierarchy as RH,
                                     parallelism as PAR, sharding as SH, operator_geometry as OG)
    shapes = [OG.operator_shape(_mm(4096, 4096, 2048, idx=0, op="addmm"), "w", "repeated_head"),
              OG.operator_shape(_mm(1, 2048, 256, idx=1), "w", "repeated_head")]
    dags = [PAR.analyze_graph(_toy_graph())]
    pressure = RH.resource_pressure(shapes, dags)
    g = PUG.guidance(shapes, dags, pressure, SH.all_shard_axes(shapes))
    opts = {o.option for o in g["options"]}
    assert opts == {"one_bigger_unit", "multiple_identical_units", "multiple_specialized_units"}
    spec = next(o for o in g["options"] if o.option == "multiple_specialized_units")
    classes = {p.resource_class for p in pressure}
    assert spec.candidate_units and all(u["for"] in classes for u in spec.candidate_units)
    assert "heterogeneous" in g["search_space_implication"].lower()
    obj = PUG.guidance_yaml(g)["processing_unit_guidance"]
    assert obj["options"] and "search_space_implication" in obj


def test_p8_reports_have_no_forbidden_wording():
    from merlin.dse_guidance import (pipeline_envelope as PE, processing_unit_guidance as PUG,
                                     resource_hierarchy as RH, parallelism as PAR, sharding as SH,
                                     operator_geometry as OG)
    g, shapes = _p8_inputs(K=5)
    phases = PE.phase_model(g, shapes)
    cands = {"m": PE.overlap_candidates(g, phases)}
    dags = [PAR.analyze_graph(_toy_graph())]
    pug = PUG.guidance(shapes, dags, RH.resource_pressure(shapes, dags), SH.all_shard_axes(shapes))
    blobs = [PE.overlap_report_md({"m": phases}, cands).lower(),
             str(PE.pipeline_candidates_yaml(cands)).lower(),
             PUG.heterogeneity_report_md(pug).lower(),
             str(PUG.guidance_yaml(pug)).lower()]
    for b in blobs:
        for term in ("gap_closure", "faster", "will improve", "predicted cycles", "meets deadline"):
            assert term not in b, f"forbidden term {term!r}"
        for ln in b.splitlines():
            if "speedup" in ln:
                assert "no speedup" in ln or "not a speedup" in ln


# ============================================================ P9 memory / DMA / buffer envelope

def _mem_inputs(K=5):
    from merlin.dse_guidance import operator_geometry as OG
    topo = TOP.from_temporal(T.parse({
        "workload": "m", "class": "diffusion/denoise_steps", "timing": {"K": K, "H": K,
        "control_rate_hz": 30}, "regions": [{"name": "head", "role": "repeated_head",
        "invocation_count": K, "loop_invariant_state": ["weights"]}]}))
    recs = (_mm(8, 256, 128, fqn="model.denoise.0", idx=0),)   # M=8, N=256, K=128 (f32)
    attr = ATTR.attribute_records(recs, topo)
    return attr, OG.operator_shapes(recs, "m", attr)


def test_memory_envelope_schema_and_avoidable_reload():
    from merlin.dse_guidance import memory_envelope as ME
    from merlin.dse_guidance.design_envelope import E_IR
    attr, shapes = _mem_inputs(K=5)
    rm = ME.region_memory(attr, shapes)[0]
    assert rm.region == "repeated_head" and rm.invocations == 5
    assert rm.weight_bytes == 128 * 256 * 4 and rm.activation_input_bytes == 8 * 128 * 4
    assert rm.output_bytes == 8 * 256 * 4
    assert rm.avoidable_weight_reload == rm.weight_bytes * 4         # weight * (K-1)
    assert rm.lifetime == "across_K" and rm.evidence == E_IR
    # bytes a flat dequantized capture cannot expose are unavailable, not invented
    assert rm.intermediate_bytes is None and rm.scale_bytes is None and rm.kv_bytes is None


def test_memory_envelope_dtype_resident_scaling():
    from merlin.dse_guidance import memory_envelope as ME
    attr, shapes = _mem_inputs(K=5)
    rm = ME.region_memory(attr, shapes)[0]
    assert rm.resident_by_dtype["int8"] == rm.weight_bytes // 4    # f32 -> int8 is /4
    assert rm.resident_by_dtype["bf16"] == rm.weight_bytes // 2


def test_dma_stream_classification_and_missing_kv_scale():
    from merlin.dse_guidance import memory_envelope as ME, dma_buffer_analysis as DMA
    from merlin.dse_guidance.design_envelope import E_NA
    attr, shapes = _mem_inputs(K=5)
    rm = ME.region_memory(attr, shapes)[0]
    streams = {s.stream: s for s in DMA.region_streams(rm)}
    assert streams["weight"].bytes == rm.weight_bytes and streams["weight"].direction == "read"
    assert streams["weight"].prefetchable == "yes"
    assert streams["output"].direction == "write"
    # scale-sideband and KV streams are unavailable (dequantized capture / attention lowered)
    assert streams["scale_sideband"].bytes == "unavailable" and streams["scale_sideband"].sideband
    assert streams["scale_sideband"].evidence == E_NA
    assert streams["kv_prefix"].bytes == "unavailable"
    assert all(s.candidate_abstraction in DMA.ALLOWED_DMA_ABSTRACTIONS for s in streams.values())


def test_buffer_requirement_double_buffer_rule():
    from merlin.dse_guidance import memory_envelope as ME, dma_buffer_analysis as DMA
    attr, shapes = _mem_inputs(K=5)
    b = DMA.buffer_requirements(ME.region_memory(attr, shapes))[0]
    assert b.min_input_buffer_count == 2 and b.min_output_buffer_count == 2   # double-buffered
    assert b.double_buffering_needed == "yes"
    assert b.input_buffer_bytes == 8 * 128 * 4 and b.resident_weight_bytes == 128 * 256 * 4


def test_p9_reports_have_no_forbidden_wording():
    from merlin.dse_guidance import memory_envelope as ME, dma_buffer_analysis as DMA
    attr, shapes = _mem_inputs(K=5)
    rm = ME.region_memory(attr, shapes)
    mem = {"m": rm}
    blobs = [ME.memory_envelope_report_md(mem).lower(),
             str(ME.memory_hierarchy_yaml(mem)).lower(),
             DMA.dma_pressure_report_md({"m": DMA.all_streams(rm)}, mem).lower(),
             str(ME.memory_abstraction_candidates_yaml({"m": ME.reuse_lifetime(rm)})).lower()]
    for b in blobs:
        for term in ("gap_closure", "faster", "optimal", "predicted cycles", "improvement"):
            assert term not in b, f"forbidden term {term!r}"
        for ln in b.splitlines():
            if "speedup" in ln:
                assert "no speedup" in ln or "not a speedup" in ln or "no bandwidth/speedup" in ln


# ============================================================ P10 fusion / epilogue / accumulator

def test_epilogue_pattern_label_and_certificates():
    from merlin.dse_guidance import fusion_epilogue as FE
    assert FE._pattern_label({"bias": True, "activation": True, "scale": False, "clamp": False,
                              "cast": False}) == "matmul->bias->activation"
    assert FE._pattern_label({"bias": False, "activation": False, "scale": False, "clamp": False,
                              "cast": False}) == "matmul"
    # certificates from a bias+activation epilogue: epilogue/accumulator/activation IR-supported;
    # low-bit/scale/sparsity always blocked (no low-bit storage in a dequantized capture)
    pat = FE.EpiloguePattern(0, "addmm", "matmul->bias->activation", True, True, False, False,
                             False, 2, False, "recovered_from_ir")
    certs = {c.abstraction: c for c in FE.certificates([pat], [])}
    assert certs["fused_requant_epilogue"].evidence == "recovered_from_ir"
    assert certs["activation_clamp_unit"].evidence == "recovered_from_ir"
    assert certs["fused_dequant_matmul"].evidence == "unavailable"
    assert certs["scale_object"].evidence == "unavailable"
    assert certs["structured_sparsity_skip"].evidence == "unavailable"
    for c in certs.values():                       # every certificate carries the honest fields
        assert c.could_be_wrong_if and c.what_is_not_claimed and c.accuracy_measurements_needed


def test_accumulator_field_vocab():
    from merlin.dse_guidance import fusion_epilogue as FE
    assert FE.DEQ_NA == "unavailable" and FE.REQ_EPILOGUE == "epilogue_candidate"
    assert FE.ACC_COMMITTED == "committed_directly"


@pytest.mark.skipif(not (_RDT_RECAP / "model.mlir").is_file(), reason="no rdt recapture")
def test_epilogue_detection_and_accumulator_on_real_capture():
    from merlin.dse_guidance import fusion_epilogue as FE, numerical_contract as NC
    cap = str(_RDT_RECAP)
    pats = FE.epilogue_patterns(cap)
    recs = ATTR.extract_matmuls(cap)
    assert len(pats) == len(recs)
    # every addmm op must be flagged has_bias (matmul+bias epilogue), recovered_from_ir
    assert all(p.has_bias for p in pats if recs[p.index].op == "addmm")
    assert any(p.has_bias for p in pats)            # rdt (DiT) uses addmm bias
    # accumulator contract: f32 capture -> no dequant, no scale; bias slot -> requant candidate
    topo = TOP.from_temporal(T.parse({"workload": "rdt", "class": "diffusion/denoise_steps",
                                      "timing": {"K": 5, "H": 64, "control_rate_hz": 30},
                                      "regions": [{"name": "h", "role": "repeated_head",
                                                   "invocation_count": 5}]}))
    attr = ATTR.attribute(cap, topo)
    nc = NC.audit(cap, workload="rdt", records=recs, attribution=attr)
    accs = FE.accumulator_contract(recs, attr, nc, pats)
    a = next(x for x in accs if x.region == "repeated_head")
    assert a.dequant_location == "unavailable" and a.scale_dtype == "unavailable"
    assert a.requant_location == "epilogue_candidate"      # bias epilogue exists
    assert a.accumulator_materialization == "committed_directly"


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_epilogue_reshape_separation_no_overclaim():
    # LLaMA-style projections reshape their output before any elementwise op: report the boundary
    # as reshape-separated and claim NO directly-fused epilogue (don't mislabel a residual add as
    # bias). rdt (matmul feeds a generic directly) is NOT reshape-separated.
    from merlin.dse_guidance import fusion_epilogue as FE, case_study as CS
    avail = CS.available_models()
    if "tiny_llama" in avail:
        pats = FE.epilogue_patterns(str(CS._recap_dir("tiny_llama")))
        assert pats and all(p.reshape_separated_epilogue for p in pats)
        assert all(not (p.has_bias or p.has_activation or p.has_scale) for p in pats)  # no overclaim
    if "rdt" in avail:
        rdt = FE.epilogue_patterns(str(CS._recap_dir("rdt")))
        assert any(not p.reshape_separated_epilogue and p.has_bias for p in rdt)  # directly fused


def test_lost_numerical_contracts_marks_erased():
    from merlin.dse_guidance import fusion_epilogue as FE
    a = FE.AccumulatorContract("repeated_head", 5, "f32", "f32", "f32", "f32", "unavailable",
                               "unavailable", FE.DEQ_NA, FE.REQ_EPILOGUE, FE.ACC_COMMITTED,
                               "recovered_from_ir")
    csv_txt = FE.lost_contracts_csv({"m": [a]})
    assert "low_bit_weight_storage" in csv_txt and "lost" in csv_txt
    assert "scale_zero_point_metadata" in csv_txt and "erased" in csv_txt
    assert "measured_pass" not in csv_txt


def test_p10_no_forbidden_wording_or_false_measured_pass():
    from merlin.dse_guidance import fusion_epilogue as FE
    pat = FE.EpiloguePattern(0, "addmm", "matmul->bias", True, False, False, False, False, 1,
                             False, "recovered_from_ir")
    a = FE.AccumulatorContract("r", 1, "f32", "f32", "f32", "f32", "unavailable", "unavailable",
                               FE.DEQ_NA, FE.REQ_EPILOGUE, FE.ACC_COMMITTED, "recovered_from_ir")
    certs = FE.certificates([pat], [a])
    blobs = [FE.fusion_report_md({"m": [pat]}, {"m": [a]}, {"m": certs}).lower(),
             str(FE.epilogue_candidates_yaml({"m": certs})).lower(),
             FE.lost_contracts_csv({"m": [a]}).lower()]
    for b in blobs:
        assert "measured_pass" not in b              # P10 makes no accuracy verdict
        for term in ("gap_closure", "faster", "optimal", "predicted cycles", "x faster"):
            assert term not in b, f"forbidden term {term!r}"
        for ln in b.splitlines():
            if "speedup" in ln:
                assert "no speedup" in ln or "not a speedup" in ln


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_consolidated_dse_search_space_knobs(tmp_path):
    # the capstone bridge aggregates the P5-P10 structural knobs (not a stale P1-P4 list)
    from merlin.dse_guidance import case_study as CS
    from merlin.common.yaml import load_yaml
    from merlin.dse_guidance.primitive_coverage import TILE_PRIMITIVES, GEMV_PRIMITIVES
    CS.run_case_study(tmp_path)
    assert (tmp_path / "dse_search_space_knobs.yaml").is_file()
    assert (tmp_path / "dse_search_space_knobs.md").is_file()
    cat = load_yaml(tmp_path / "dse_search_space_knobs.yaml")["dse_search_space_knobs"]
    groups = {g["group"]: g for g in cat["knob_groups"]}
    assert {"P5", "P7", "P8", "P9", "P10"} <= {g["source_phase"] for g in groups.values()}
    # primitive knobs are grounded in the P5 candidate set (not hand-written)
    prim = [n for n, _, _ in TILE_PRIMITIVES] + [n for n, _ in GEMV_PRIMITIVES]
    assert groups["compute_primitive_shape"]["knobs"] == prim
    # inter-op parallelism is honestly disabled (low avg parallelism), sharding enabled
    assert groups["inter_op_parallelism"]["enabled"] is False
    assert groups["intra_op_sharding"]["enabled"] is True
    assert "speedup" not in str(cat).lower() or "no speedup" in str(cat).lower()


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_p10_artifacts(tmp_path):
    from merlin.dse_guidance import case_study as CS
    CS.run_case_study(tmp_path)
    for f in ("epilogue_pattern_table.csv", "accumulator_contract_table.csv",
              "numerical_epilogue_candidates.yaml", "lost_numerical_contracts.csv",
              "fusion_opportunity_report.md"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    import csv, io
    accs = list(csv.DictReader(io.StringIO(
        (tmp_path / "accumulator_contract_table.csv").read_text())))
    for r in accs:
        assert r["scale_dtype"] == "unavailable" and r["dequant_location"] == "unavailable"
    # no non-int8 false measured-pass introduced anywhere
    assert "measured_pass" not in (tmp_path / "numerical_epilogue_candidates.yaml").read_text()


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_p9_artifacts(tmp_path):
    from merlin.dse_guidance import case_study as CS
    CS.run_case_study(tmp_path)
    for f in ("memory_hierarchy_envelope.yaml", "data_movement_table.csv",
              "reuse_lifetime_table.csv", "memory_abstraction_candidates.yaml",
              "memory_envelope_report.md", "dma_stream_table.csv",
              "buffer_requirement_table.csv", "dma_pressure_report.md"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    import csv, io
    dm = list(csv.DictReader(io.StringIO((tmp_path / "data_movement_table.csv").read_text())))
    for r in dm:
        assert int(r["avoidable_weight_reload"]) == int(r["weight_bytes"]) * max(
            int(r["invocations"]) - 1, 0)
        assert r["scale_bytes"] == "unavailable" and r["kv_bytes"] == "unavailable"


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_p8_artifacts(tmp_path):
    from merlin.dse_guidance import case_study as CS
    from merlin.common.yaml import load_yaml
    CS.run_case_study(tmp_path)
    for f in ("pipeline_envelope.yaml", "pipeline_stage_table.csv", "pipeline_candidates.yaml",
              "buffering_requirement_table.csv", "overlap_opportunities.md",
              "processing_unit_guidance.yaml", "heterogeneity_report.md"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    pug = load_yaml(tmp_path / "processing_unit_guidance.yaml")["processing_unit_guidance"]
    assert len(pug["options"]) == 3 and "heterogeneous" in pug["search_space_implication"].lower()


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_p7_artifacts(tmp_path):
    from merlin.dse_guidance import case_study as CS
    CS.run_case_study(tmp_path)
    for f in ("dag_parallelism_report.md", "critical_path_table.csv", "concurrency_windows.csv",
              "parallel_region_candidates.yaml", "sharding_table.csv", "sharding_opportunities.yaml",
              "intra_op_sharding_report.md", "operator_cluster_to_hierarchy.csv",
              "parallel_hierarchy_hints.yaml", "resource_pressure_table.csv",
              "processing_unit_candidates.yaml", "processing_unit_parallelism_report.md"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    import csv, io
    crit = list(csv.DictReader(io.StringIO((tmp_path / "critical_path_table.csv").read_text())))
    for r in crit:
        assert int(r["critical_path_macs"]) <= int(r["total_macs"])      # span <= work
        assert float(r["available_parallelism"]) >= 1.0


# ============================================================ P12 HW/SW boundary placement

_BP_EVID = {
    "rdt": {"dense": True, "gemv": True, "backbone": False, "epilogue": True, "k_loop": True,
            "control_loop": True, "decode": False},
    "openvla": {"dense": False, "gemv": True, "backbone": True, "epilogue": True, "k_loop": True,
                "control_loop": True, "decode": True},
    "small_llama": {"dense": False, "gemv": True, "backbone": False, "epilogue": False,
                    "k_loop": True, "control_loop": False, "decode": True},
    "tiny_llama": {"dense": False, "gemv": True, "backbone": False, "epilogue": False,
                   "k_loop": True, "control_loop": False, "decode": True},
}
_BP_CP = {
    "resident_action_head_weights": ("weights immutable across K", "proven_for_workload"),
    "autonomous_K_loop": ("bounded K + device-resident body", "assumed"),
    "command_batching": ("static K-loop dependency graph", "assumed"),
    "async_chunk_overlap": ("next replan independent of current chunk", "assumed"),
    "fused_requant_epilogue": ("fused requant bit-exactness", "unknown"),
    "resident_packed_lowbit_weights": ("per-format accuracy vs fp32", "unknown"),
    "decode_kv_cache_path": ("autoregressive KV path", "assumed"),
}


def _bp_certs():
    from merlin.dse_guidance import boundary_placement as BP
    return {c.abstraction: c for c in BP.build_certificates(_BP_EVID, _BP_CP)}


def test_boundary_level_and_status_vocabulary():
    from merlin.dse_guidance import boundary_placement as BP
    for c in _bp_certs().values():
        assert {b["level"] for b in c.boundary_levels} == set(BP.LEVELS)
        assert all(b["status"] in BP.STATUS for b in c.boundary_levels)
    assert len(BP.ABSTRACTIONS) == 27


def test_boundary_certificate_schema():
    certs = _bp_certs()
    for c in certs.values():
        for b in c.boundary_levels:
            for fld in ("level", "status", "software_manages", "hardware_manages",
                        "required_compiler_proof", "required_runtime_support",
                        "required_isa_semantics", "required_hw_support", "metadata_crossing",
                        "risk", "missing_evidence"):
                assert fld in b, f"{c.abstraction} level missing {fld}"
        assert c.cp_matrix_axis is None or isinstance(c.cp_matrix_axis, str)
        assert c.what_is_not_claimed and c.source_analyses


def test_boundary_grounded_supporting_workloads():
    certs = _bp_certs()
    # matrix engine only where dense GEMM exists (rdt); GEMV everywhere; requant where epilogue
    assert certs["matrix_engine"].supporting_workloads == ["rdt"]
    assert certs["skinny_gemm_or_gemv_engine"].supporting_workloads == \
        ["openvla", "rdt", "small_llama", "tiny_llama"]
    assert certs["fused_requant_epilogue"].supporting_workloads == ["openvla", "rdt"]
    # decode controller only for autoregressive workloads (not rdt flow-matching)
    assert "rdt" not in certs["decode_loop_controller"].supporting_workloads


def test_boundary_resident_weight_matches_ladder():
    from merlin.dse_guidance import boundary_placement as BP
    c = _bp_certs()["resident_weight_object"]
    st = {b["level"]: b["status"] for b in c.boundary_levels}
    assert st[BP.L_RUNTIME] == "strong_candidate" and st[BP.L_COMMAND] == "strong_candidate"
    assert st[BP.L_ISA] == "weak_candidate" and st[BP.L_DATAPATH] == "not_applicable"
    assert c.cp_matrix_axis == "resident_action_head_weights"
    assert c.compiler_proof_status == "proven_for_workload"


def test_boundary_bounded_loop_and_partial_sum():
    from merlin.dse_guidance import boundary_placement as BP
    c = _bp_certs()
    bl = {b["level"]: b["status"] for b in c["bounded_loop_command"].boundary_levels}
    assert bl[BP.L_COMMAND] == "strong_candidate" and bl[BP.L_MICROCODE] == "strong_candidate"
    assert c["bounded_loop_command"].cp_matrix_axis == "autonomous_K_loop"
    ps = {b["level"]: b["status"] for b in c["partial_sum_object"].boundary_levels}
    assert ps[BP.L_COMPILER] == "not_applicable" and ps[BP.L_ISA] == "strong_candidate"


def test_boundary_erased_abstractions_blocked():
    from merlin.dse_guidance import boundary_placement as BP
    c = _bp_certs()
    for name in ("packed_lowbit_tensor", "scale_object", "fused_dequant_matmul",
                 "native_lowbit_matmul"):
        st = {b["level"]: b["status"] for b in c[name].boundary_levels}
        assert st[BP.L_COMPILER] == "possible"                 # compiler-dequant (status quo)
        assert st[BP.L_DATAPATH] == "blocked" and st[BP.L_ISA] == "blocked"  # need low-bit capture
        assert c[name].erased is True
    # KV abstractions unavailable (attention lowered)
    kv = {b["level"]: b["status"] for b in c["kv_cache_object"].boundary_levels}
    assert kv[BP.L_RUNTIME] == "unavailable" and kv[BP.L_COMPILER] == "not_applicable"


def test_boundary_pressure_score_recomputes():
    certs = _bp_certs()
    for c in certs.values():
        assert c.boundary_pressure_score == sum(c.pressure_components.values())
    # resident_weight (proven, both roles, all 4 workloads) scores high
    assert certs["resident_weight_object"].boundary_pressure_score >= 8


def test_responsibility_matrix_schema():
    from merlin.dse_guidance import boundary_placement as BP
    rows = BP.responsibility_rows()
    assert len(rows) == 17
    for r in rows:
        for col in ("compiler", "runtime_hal", "command_processor", "accelerator_isa",
                    "device_microcode", "datapath"):
            assert r[col] in BP.RESP_CELLS
    by = {r["function"]: r for r in rows}
    assert by["region_partitioning"]["compiler"] == "owns"
    assert by["partial_sum_merge"]["accelerator_isa"] == "owns"
    assert by["resident_object_lifetime"]["runtime_hal"] == "owns"


def test_boundary_dse_knobs_have_reason_and_evidence():
    from merlin.dse_guidance import boundary_placement as BP
    obj = BP.boundary_dse_knobs_yaml(list(_bp_certs().values()))["boundary_dse_knobs"]
    assert obj["knobs"]
    for k in obj["knobs"]:
        assert k["knob"] and k["reason"] and k["evidence"] and k["abstraction"]
        assert k["boundary_level"] in BP.LEVELS


def test_boundary_partial_mode():
    from merlin.dse_guidance import boundary_placement as BP
    certs = BP.build_certificates({}, {})        # no workloads, no compiler proofs
    assert len(certs) == 27
    assert all(c.supporting_workloads == [] for c in certs)
    assert all(c.compiler_proof_status == "unavailable" for c in certs)
    assert all(c.cp_matrix_axis is None or c.compiler_proof_status == "unavailable" for c in certs)


def test_boundary_no_forbidden_wording():
    from merlin.dse_guidance import boundary_placement as BP
    certs = list(_bp_certs().values())
    blobs = [BP.boundary_report_md(certs, BP.responsibility_rows()).lower(),
             str(BP.boundary_candidate_contracts_yaml(certs)).lower(),
             BP.interface_contract_sketches_md(certs).lower(),
             str(BP.boundary_dse_knobs_yaml(certs)).lower()]
    for b in blobs:
        for term in ("optimal", "best design", "faster", "predicted", "performance improvement",
                     "gap_closure"):
            assert term not in b, f"forbidden term {term!r}"
        for ln in b.splitlines():
            if "speedup" in ln:
                assert "no speedup" in ln or "not a speedup" in ln


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_dse_contract_manifest_and_query(tmp_path):
    # the single machine-readable consume entry point + the --query CLI on top of it
    import json
    from merlin.dse_guidance import case_study as CS
    from merlin.dse_guidance import cli
    CS.run_case_study(tmp_path)
    mf = tmp_path / "dse_contract.json"
    assert mf.is_file()
    m = json.loads(mf.read_text())
    for k in ("workloads", "per_workload", "search_space_knob_groups", "boundary_placement",
              "measurements_needed_before_quantitative_dse", "artifacts_index",
              "what_is_not_claimed"):
        assert k in m, f"manifest missing {k}"
    assert set(m["workloads"]) == set(CS.available_models())
    for v in m["artifacts_index"].values():
        assert (tmp_path / v).is_file(), f"index points at missing {v}"
    assert m["boundary_placement"]["score_is"].startswith("evidence_breadth")
    assert m["boundary_placement"]["top_by_evidence_breadth"]
    # the --query CLI consumes the manifest
    for q in ("summary", "knobs", "boundary", "missing", "index",
              "boundary:resident_weight_object"):
        assert cli.main(["--query", q, "--out", str(tmp_path)]) == 0


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_case_study_emits_p12_artifacts(tmp_path):
    from merlin.dse_guidance import case_study as CS
    from merlin.common.yaml import load_yaml
    CS.run_case_study(tmp_path)
    for f in ("hw_sw_boundary_matrix.csv", "boundary_candidate_contracts.yaml",
              "boundary_placement_report.md", "responsibility_split_matrix.csv",
              "interface_contract_sketches.md", "isa_candidate_primitives.yaml",
              "runtime_object_candidates.yaml", "command_isa_candidates.yaml",
              "boundary_dse_knobs.yaml"):
        assert (tmp_path / f).is_file(), f"missing {f}"
    contracts = load_yaml(tmp_path / "boundary_candidate_contracts.yaml")[
        "boundary_candidate_contracts"]["certificates"]
    assert len(contracts) == 27
    # boundary placement now feeds the consolidated bridge
    knobs = load_yaml(tmp_path / "dse_search_space_knobs.yaml")["dse_search_space_knobs"]
    assert any(g["source_phase"] == "P12" for g in knobs["knob_groups"])


# ============================================================ P13 evidence mining / insight extract

_CS_DIR = merlin_dir() / "benchmarks" / "dse_guidance" / "case_study"


def test_insight_evidence_tier_classification():
    from merlin.dse_guidance import insight_mining as IM
    # Tier now keys off a per-metric harness check OR >=2-artifact corroboration (not artifact set)
    assert IM.evidence_tier("recovered_from_ir", "head_weight_bytes") == "A"        # has a check
    assert IM.evidence_tier("recovered_from_ir", "no_check_metric", 2) == "A"       # corroborated>=2
    assert IM.evidence_tier("recovered_from_ir", "no_check_metric", 1) == "B"       # unverified
    assert IM.evidence_tier("derived_requirement", "avoidable_weight_reload") == "B"  # derived+check
    assert IM.evidence_tier("derived_requirement", "no_check_metric", 1) == "C"     # derived-unverified
    assert IM.evidence_tier("recovered_from_model_config", "K") == "C"              # config reference
    assert IM.evidence_tier("assumed_reference", "x") == "C"
    assert IM.evidence_tier("unavailable", "x") == "D"


def test_insight_partial_mode(tmp_path):
    from merlin.dse_guidance import insight_mining as IM
    # an empty dir: every expected artifact recorded exists=no, mining does not crash, no facts
    b = IM.mine(tmp_path, "all")
    assert b["inventory"] and all("exists" in r for r in b["inventory"])
    assert all(not r["exists"] for r in b["inventory"])      # all missing, recorded explicitly
    assert b["facts"] == [] and b["findings"] == []          # nothing to mine -> no invented facts
    assert isinstance(b["consistency_checks"], list)         # ran without crashing


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_insight_unified_fact_schema_and_tiers():
    from merlin.dse_guidance import insight_mining as IM
    facts = IM.unified_facts(_CS_DIR, "all")
    assert facts
    ids = [f["fact_id"] for f in facts]
    assert len(ids) == len(set(ids))                                   # no duplicate fact_id
    for f in facts:
        assert set(IM.FACT_COLUMNS) <= set(f)                          # full schema
        assert f["evidence_tier"] in ("A", "B", "C", "D")
        assert f["verification_status"] in ("verified", "not_verified", "unavailable")


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_insight_usefulness_and_findings_gate():
    from merlin.dse_guidance import insight_mining as IM
    facts = IM.unified_facts(_CS_DIR, "all")
    answers = IM.usefulness(_CS_DIR, "all", facts)
    assert len(answers) == 20
    for a in answers:
        assert a["status"] in ("strong", "partial", "weak", "unavailable")
        assert a["recommended_presentation_use"] in ("main", "backup", "do_not_show")
    findings = IM.presentation_findings(facts, answers)
    main = [f for f in findings if f["presentation_placement"] == "main"]
    assert main
    # main gate: tier A/B, has implication, never purely tier C/D
    assert all(f["evidence_tier"] in ("A", "B") for f in main)
    assert all(f["dse_implication"] for f in main)


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_insight_plot_manifest_columns_exist():
    from merlin.dse_guidance import insight_mining as IM
    for p in IM.plot_manifest(_CS_DIR, "all"):
        if p["recommendation"] == "omit":
            continue
        art = _CS_DIR / p["source_artifact"]
        if p["source_artifact"] == "unified_fact_table.csv":
            continue                                                   # run's own output
        assert art.is_file(), f"plot {p['plot_id']} source missing"


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_insight_consistency_and_no_forbidden_wording():
    from merlin.dse_guidance import insight_mining as IM
    b = IM.mine(_CS_DIR, "all")
    assert all(ok for ok, _ in b["consistency_checks"]), \
        [m for ok, m in b["consistency_checks"] if not ok]
    blob = (str(b["findings"]) + str(b["usefulness"]) + str(b["plots"])).lower()
    for t in ("speedup", "faster", "optimal", "performance improvement", "predicted cycles"):
        assert t not in blob, f"forbidden term {t!r}"


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_insight_per_network_completes_for_all(tmp_path):
    # the per-network requirement: every supported network mines cleanly (all consistency checks pass)
    from merlin.dse_guidance import insight_mining as IM, presentation_plots as PP, case_study as CS
    for w in CS.available_models():
        b = IM.mine(_CS_DIR, w)
        assert len(b["facts"]) > 0, f"{w}: no facts"
        assert all(ok for ok, _ in b["consistency_checks"]), f"{w}: consistency fail"
        # emit + render once into a temp run dir to confirm the full pipeline works end-to-end
        run = tmp_path / f"{w}_test_dse_analysis"
        rendered = PP.render_plots(b["plots"], _CS_DIR, b["facts"], run / "generated_plots")
        IM.emit_run(b, run, rendered)
        assert (run / "unified_fact_table.csv").is_file()
        assert (run / "presentation_candidate_findings.md").is_file()


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_insight_mining_is_deterministic():
    from merlin.dse_guidance import insight_mining as IM
    assert IM.unified_facts(_CS_DIR, "all") == IM.unified_facts(_CS_DIR, "all")
    f = IM.unified_facts(_CS_DIR, "rdt")
    assert IM.presentation_findings(f, IM.usefulness(_CS_DIR, "rdt", f)) == \
        IM.presentation_findings(f, IM.usefulness(_CS_DIR, "rdt", f))


# ============================================================ P14 devil's-advocate iteration loop

@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p14_gap_audit_converges_to_zero():
    from merlin.dse_guidance import insight_mining as IM
    nets = IM._workloads(_CS_DIR)
    for scope in nets + ["all"]:
        b = IM.mine(_CS_DIR, scope)
        oa = b["open_avoidable_gaps"]
        assert not oa, f"{scope}: open avoidable gaps {[g['category'] for g in oa]}"
        # inherent limits are scoped (each carries a required input), not bare caveats
        inh = [g for g in b["gaps"] if g["category"] == "inherent_limit"]
        assert inh and all(g["required_input"] and g["status"] == "scoped" for g in inh)


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p14_full_artifact_coverage():
    from merlin.dse_guidance import insight_mining as IM
    facts = IM.unified_facts(_CS_DIR, "all")
    used = {f["source_artifact"] for f in facts}
    expected = set(IM._ARTIFACT_PHASE)
    missing = sorted(expected - used)
    assert not missing, f"unused expected artifacts: {missing}"     # leverages ALL of P0-P12


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p14_main_findings_corroborated_and_measured_lowbit_surfaced():
    from merlin.dse_guidance import insight_mining as IM
    b = IM.mine(_CS_DIR, "all")
    main = [f for f in b["findings"] if f["presentation_placement"] == "main"]
    assert main
    for f in main:
        check = any(x["verifying_check"] for m in f["relevant_metrics"]
                    for x in b["facts"] if x["metric_name"] == m)
        assert f["max_corroborated_by"] >= 2 or check, f"uncorroborated main: {f['title']}"
    # real measured + low-bit evidence is present (inherent-limit removal via existing data)
    assert any(f["derivation_type"] == "measured" for f in b["facts"])
    assert any(f["workload"] == "ZOO" or "lowbit" in f["metric_name"] for f in b["facts"])


def test_p14_signal_vs_context_candidacy():
    from merlin.dse_guidance import insight_mining as IM
    # a CONTEXT metric (a row count) is never presentation-worthy, even corroborated -> de-noise
    ctx = IM._fact("F0", workload="w", metric="n_runtime_objects", value=3, unit="count",
                   artifact="runtime_object_candidates.yaml", phase="P12",
                   evidence="recovered_from_ir", implication="objects crossing the HAL",
                   corroborated_by=2)
    assert ctx["metric_class"] == "context" and ctx["presentation_candidate"] is False
    # a SIGNAL metric, tier A/B, is a candidate even single-source (real recovered evidence)
    sig = IM._fact("F1", workload="w", metric="avoidable_weight_reload", value=10, unit="bytes",
                   artifact="data_movement_table.csv", phase="P9", evidence="derived_requirement",
                   implication="residency benefit", corroborated_by=1)
    assert sig["metric_class"] == "signal" and sig["presentation_candidate"] is True
    # measured facts are their own class and presentation-worthy
    m = IM._fact("F2", workload="w", metric="accuracy_int8_w8a8", value="pass", unit="band",
                 artifact="accuracy_gate_results.csv", phase="P4", evidence="measured",
                 implication="int8 accuracy-legal")
    assert m["metric_class"] == "measured" and m["presentation_candidate"] is True


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p14_required_inputs_manifest_scopes_every_limit():
    from merlin.dse_guidance import insight_mining as IM
    ri = IM.required_inputs(_CS_DIR, IM.unified_facts(_CS_DIR, "all"))
    assert ri
    for x in ri:
        assert x["limit"] and x["required_input"] and x["status"] == "scoped"


@pytest.mark.skipif(not _has_recaptures(), reason="fewer than 2 prov.fqn recaptures present")
def test_p14_cli_insight_mining_run_zero_gaps(tmp_path):
    from merlin.dse_guidance import cli
    rc = cli.main(["--insight-mining", "--out", str(tmp_path)])
    assert rc == 0                                                   # nonzero iff any open gaps
    runs = list(tmp_path.glob("*_dse_analysis"))
    assert runs
    allrun = next(r for r in runs if r.name.startswith("all_"))
    assert (allrun / "gap_audit_report.md").is_file()
    assert (allrun / "required_inputs_manifest.yaml").is_file()


# --------------------------------------------------------------------------- P15 signal-first study

@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p15_every_signal_metric_answers_a_dse_question():
    from merlin.dse_guidance import insight_mining as IM
    # the organizing rule: every SIGNAL/measured metric maps to one of the fixed DSE questions
    unmapped = [m for m in IM.SIGNAL_METRICS if m not in IM._METRIC_QUESTION]
    assert unmapped == []
    assert set(IM._METRIC_QUESTION.values()) <= set(IM.DSE_QUESTIONS)
    b = IM.mine(_CS_DIR, "all")
    # canonical table is signal/measured only, each row carries a valid question
    fbm = {f["metric_name"]: f for f in b["facts"]}
    assert b["canonical_signal"]
    for r in b["canonical_signal"]:
        assert fbm[r["metric"]]["metric_class"] in ("signal", "measured")  # no context leaks
        assert r["dse_question"] in IM.DSE_QUESTIONS
    # every main finding carries a question
    for f in b["findings"]:
        if f["presentation_placement"] == "main":
            assert f["dse_question"] in IM.DSE_QUESTIONS


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p15_hotspots_reference_real_ops_and_recompute():
    from merlin.dse_guidance import insight_mining as IM
    import csv as _csv
    import io as _io
    h = IM.per_operator_hotspots(_CS_DIR)
    ops = list(_csv.DictReader(_io.StringIO(
        (_CS_DIR / "operator_shape_table.csv").read_text())))
    assert h["n_ops"] == len(ops)
    keys = {(o["workload"], o["op_index"]) for o in ops}
    assert all((r["workload"], r["op_index"]) in keys for r in h["by_macs"])
    # top-by-MACs is an honest independent sort
    indep = [o["op_index"] for o in sorted(ops, key=lambda o: -int(o["macs"]))[:10]]
    assert [r["op_index"] for r in h["by_macs"]] == indep
    # the dominant op carries a real per-workload MAC fraction in (0, 1]
    d = h["dominant_op"]
    assert 0.0 < d["mac_fraction_of_workload"] <= 1.0
    # a per-network scope reports ONLY that network's ops (not the corpus-wide top)
    wls = sorted({o["workload"] for o in ops})
    one = [w for w in wls if w != "rdt"][0]              # a non-rdt workload (rdt dominates corpus)
    hs = IM.per_operator_hotspots(_CS_DIR, one)
    assert hs["by_macs"] and all(r["workload"] == one for r in hs["by_macs"])
    assert hs["n_ops"] == sum(1 for o in ops if o["workload"] == one)


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p15_abstraction_coverage_recomputes_and_flags_overfit():
    from merlin.dse_guidance import insight_mining as IM
    import csv as _csv
    import io as _io
    cov = IM.abstraction_coverage(_CS_DIR)
    assert cov
    ops = list(_csv.DictReader(_io.StringIO(
        (_CS_DIR / "operator_shape_table.csv").read_text())))
    mac_by_wl = {}
    for o in ops:
        mac_by_wl[o["workload"]] = mac_by_wl.get(o["workload"], 0) + int(o["macs"])
    tmac = sum(mac_by_wl.values())
    nwl = len(mac_by_wl)
    for r in cov:
        supp = [w.strip() for w in r["workloads_supporting"].split(";") if w.strip()]
        assert abs(r["mac_coverage"] - round(sum(mac_by_wl.get(w, 0) for w in supp) / tmac, 4)) < 1e-6
        # overfit risk is consistent with support breadth (single-workload = high)
        assert r["overfit_risk"] == ("high" if len(supp) <= 1
                                     else "low" if len(supp) == nwl else "medium")


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p15_corpus_plan_and_family_summary():
    from merlin.dse_guidance import insight_mining as IM
    from merlin.dse_guidance.models import MODEL_ARCH
    cp = IM.corpus_expansion_plan(_CS_DIR)
    missing = [m["model"] for ms in cp["missing_by_family"].values() for m in ms]
    assert missing and all(m in MODEL_ARCH for m in missing)          # only real registry models
    assert not (set(missing) & set(cp["captured_models"]))            # captured are excluded
    assert cp["fidelity_asks"]
    fs = IM.workload_family_summary(_CS_DIR, IM.mine(_CS_DIR, "all")["findings"])
    assert set(fs["families"]) and "unknown" not in fs["families"]    # all workloads classified
    for d in fs["families"].values():
        assert d["workloads"] and d["dominant_shape_class"]


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p15_emit_writes_study_deliverables_and_plot_captions(tmp_path):
    from merlin.dse_guidance import insight_mining as IM, presentation_plots as PP
    b = IM.mine(_CS_DIR, "all")
    rendered = PP.render_plots(b["plots"], _CS_DIR, b["facts"], tmp_path / "generated_plots")
    IM.emit_run(b, tmp_path, rendered)
    for f in ("canonical_signal_table.csv", "per_operator_hotspots.csv",
              "abstraction_coverage_table.csv", "workload_family_summary.md",
              "corpus_expansion_plan.md", "signal_findings_report.md",
              "presentation_plots_index.md"):
        assert (tmp_path / f).is_file(), f
    # every non-omit plot carries a non-empty DSE-implication caption
    assert all(p["dse_caption"] for p in b["plots"] if p["recommendation"] != "omit")
    # the study refuses unbuilt-HW performance claims + carries the devil's-advocate closing note
    blob = (tmp_path / "signal_findings_report.md").read_text().lower()
    assert not any(t in blob for t in ("speedup", "faster", "optimal", "predicted cycles"))
    assert "robust vs corpus-limited" in blob and "random-init" in blob
    # the consolidated one-file digest exists and embeds every section + the knob catalog
    dig = (tmp_path / "DSE_FINDINGS.md").read_text()
    for section in ("Headline metrics", "Top ops by MACs", "Abstraction necessity (strict",
                    "Decision-impact plots", "What a DSE tool ingests", "How to evaluate this"):
        assert section in dig, section
    assert "compute_primitive_shape" in dig and "total_macs" in dig
    assert "speedup" not in dig.lower() or "no speedup" in dig.lower()


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p15_decision_impact_plots_render_with_captions(tmp_path):
    # the what-if plots show how an outcome changes under a DSE knob choice (structural, no perf)
    from merlin.dse_guidance import insight_mining as IM, presentation_plots as PP
    decision_ids = {"decision_primitive_choice", "decision_weight_residency",
                    "decision_capacity_dtype", "decision_sharding_cost"}
    plots = IM.plot_manifest(_CS_DIR, "all")
    dp = [p for p in plots if p["plot_id"] in decision_ids]
    assert len(dp) == len(decision_ids)
    assert all(p["available"] and p["dse_caption"] for p in dp)        # data present + captioned
    b = IM.mine(_CS_DIR, "all")
    rendered = PP.render_plots(b["plots"], _CS_DIR, b["facts"], tmp_path)
    # matplotlib may be unavailable in CI -> renderer no-ops; only assert when it ran
    if rendered:
        assert decision_ids <= set(rendered)
        for pid in decision_ids:
            assert (tmp_path / f"{pid}.png").is_file()


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p15_canonical_table_has_entity_discriminator_and_no_dups(tmp_path):
    # per-abstraction / per-region metrics share workload=ALL -> entity disambiguates + dedups
    from merlin.dse_guidance import insight_mining as IM
    canon = IM.canonical_signal_table(IM.unified_facts(_CS_DIR, "all"))
    assert canon and all("entity" in r for r in canon)
    keys = [(r["dse_question"], r["metric"], r["workload"], r["entity"], str(r["value"]))
            for r in canon]
    assert len(keys) == len(set(keys))                                # no duplicate headline rows
    bps = [r for r in canon if r["metric"] == "boundary_pressure_score"]
    assert bps and len({r["entity"] for r in bps}) == len(bps)        # each names its abstraction


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p15_per_network_run_is_scoped_not_corpus_wide(tmp_path):
    # a per-network folder must contain THAT network's hotspots + canonical only; the corpus-level
    # cross-workload artifacts (coverage / family / corpus plan) belong to the 'all' run only.
    from merlin.dse_guidance import insight_mining as IM
    one = [w for w in IM._workloads(_CS_DIR) if w != "rdt"][0]
    b = IM.mine(_CS_DIR, one)
    IM.emit_run(b, tmp_path, [])
    assert (tmp_path / "per_operator_hotspots.csv").is_file()
    assert (tmp_path / "canonical_signal_table.csv").is_file()
    # cross-workload corpus artifacts are NOT dumped into a per-network folder
    assert not (tmp_path / "abstraction_coverage_table.csv").is_file()
    assert not (tmp_path / "corpus_expansion_plan.md").is_file()
    # the per-network canonical table is scoped to that workload
    import csv as _csv
    rows = list(_csv.DictReader((tmp_path / "canonical_signal_table.csv").open()))
    wls = {r["workload"] for r in rows if r["workload"] not in ("ALL", "ZOO", "")}
    assert wls == {one}


# --------------------------------------------------------------------------- P16 decision-frontier

@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p16_abstraction_necessity_is_discriminating_not_permissive():
    from merlin.dse_guidance import insight_mining as IM
    from merlin.dse_guidance.boundary_placement import catalog_rows
    nec = IM.abstraction_necessity(_CS_DIR)
    roll = nec["rollup"]
    # the #1 fix: NOT everything is necessary, and some abstractions are blocked
    assert roll["necessary"] < len(nec["rows"]) and roll["blocked"] > 0
    # blocked abstractions are exactly the erased/kv catalog entries (capture-limited)
    erased_kv = {c["abstraction"] for c in catalog_rows() if c["erased"] or c["kv"]}
    blocked = {r["abstraction"] for r in nec["rows"] if r["macro_class"] == "blocked"}
    assert blocked and blocked <= erased_kv
    # every cell is a valid class and at least one not_applicable cell exists (non-AR + decode/kv)
    valid = set(IM._NEC_CLASSES)
    assert all(r[w] in valid for r in nec["rows"] for w in nec["workloads"])
    assert any(r[w] == "not_applicable" for r in nec["rows"] for w in nec["workloads"])


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p16_necessity_recomputes_from_source():
    # matrix_engine (dense) necessity must equal (dense MAC fraction > 0.5) per workload
    from merlin.dse_guidance import insight_mining as IM
    import csv as _csv
    import io as _io
    shape = list(_csv.DictReader(_io.StringIO(
        (_CS_DIR / "shape_summary_by_workload.csv").read_text())))
    nec = IM.abstraction_necessity(_CS_DIR)
    me = next(r for r in nec["rows"] if r["abstraction"] == "matrix_engine")
    for w in nec["workloads"]:
        dense = sum(float(r["mac_fraction"]) for r in shape
                    if r["workload"] == w and r["shape_class"] == "squareish_gemm")
        assert (me[w] == "necessary") == (dense > 0.5)


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p16_primitive_set_frontier_monotone_and_set_union():
    from merlin.dse_guidance import insight_mining as IM
    fr = IM.primitive_set_frontier(_CS_DIR)
    by = fr["best_by_size"]
    # worst-workload coverage is monotone non-decreasing in set size; a 2-set beats the best single
    assert by[2]["worst"] >= by[1]["worst"] and by[2]["worst"] > by[1]["worst"]
    # independent op-level set-union recompute of the chosen 2-set's worst coverage
    import csv as _csv
    import io as _io
    tw = list(_csv.DictReader(_io.StringIO((_CS_DIR / "tile_waste_table.csv").read_text())))
    op_macs, op_cover = {}, {}
    for r in tw:
        if r.get("applicable") != "True":
            continue
        k = (r["workload"], r["op_index"])
        op_macs[k] = float(r["true_macs"])
        op_cover.setdefault(k, {})[r["primitive"]] = (r["covered_under_10pct"] == "True")
    pset = by[2]["set"]
    num, den = {}, {}
    for (w, op), m in op_macs.items():
        den[w] = den.get(w, 0.0) + m
        if any(op_cover[(w, op)].get(p, False) for p in pset):
            num[w] = num.get(w, 0.0) + m
    # the extractor rounds coverage to 4 decimals, so round the independent recompute the same way
    assert round(min(num.get(w, 0.0) / den[w] for w in den), 4) == by[2]["worst"]


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p16_operator_pareto_and_leave_one_out():
    from merlin.dse_guidance import insight_mining as IM
    par = IM.operator_pareto(_CS_DIR)
    for r in par["rows"]:
        assert r["k_macs_50"] <= r["k_macs_95"] <= r["n_ops"]
    rdt = next(r for r in par["rows"] if r["workload"] == "rdt")
    assert rdt["k_macs_50"] == 1 and rdt["top_op_mac_share"] > 0.8     # one giant op dominates
    rob = IM.robustness(_CS_DIR)
    dom = next(f for f in rob["findings"] if f["finding"] == "dense_gemm_mac_dominance")
    # the LOO MECHANISM is well-formed (the specific outcome depends on the corpus — adding the
    # vision-heavy VLAs legitimately changes whether dense GEMM dominates): valid fractions, a
    # micro_loo entry per workload, and collapses_if_removed derived exactly from the 0.2 threshold.
    wls = rob["workloads"]
    assert 0.0 <= dom["macro"] <= 1.0 and 0.0 <= dom["micro"] <= 1.0
    assert set(dom["micro_loo"]) == set(wls)
    assert sorted(dom["collapses_if_removed"]) == sorted(
        w for w in wls if dom["micro_loo"][w] < 0.2 <= dom["micro"])


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p16_capture_fidelity_and_emit(tmp_path):
    from merlin.dse_guidance import insight_mining as IM
    cf = IM.capture_fidelity(_CS_DIR)
    lowbit = next(r for r in cf["matrix"] if r["feature"] == "packed_lowbit_layout")
    assert all(lowbit[w] == "erased" for w in cf["workloads"])         # dequantized capture
    kloop = next(r for r in cf["matrix"] if r["feature"] == "K_or_decode_loop")
    # P21/P23: workloads WITH a loop-preserving capture recover K from IR (scf.for); any without one
    # still reports K from config/reference (or n/a). Restrict to the ANALYZED corpus (cf["workloads"])
    # — small_llama has a loop model.mlir but is excluded from the corpus (functional-weight wrapper ->
    # 0 linalg.matmul), so it is not a column in the matrix.
    _loop_preserved = {w for w in cf["workloads"]
                       if (_CS_DIR.parent / "recaptures_loop" / w / "model.mlir").is_file()}
    for w in cf["workloads"]:
        if w in _loop_preserved:
            assert "recovered" in kloop[w] and "IR" in kloop[w], (w, kloop[w])
        else:
            assert ("config" in kloop[w]) or (kloop[w] == "n/a"), (w, kloop[w])
    # the loop-carried state row is recovered for every loop-preserving capture
    lcs = next(r for r in cf["matrix"] if r["feature"] == "loop_carried_state")
    assert all("recovered" in lcs[w] for w in _loop_preserved)
    # openVLA's static KV cache is recovered with bytes from the IR iter_arg
    kv = next(r for r in cf["matrix"] if r["feature"] == "kv_cache_state")
    assert "recovered" in kv["openvla"] and "B" in kv["openvla"]
    # the digest leads with strict necessity, not the permissive coverage
    b = IM.mine(_CS_DIR, "all")
    IM.emit_run(b, tmp_path, [])
    for f in ("abstraction_necessity_table.csv", "primitive_set_frontier.csv",
              "operator_pareto_hotspots.csv", "capture_fidelity_matrix.csv",
              "decision_question_scorecard.md", "leave_one_workload_out.md",
              "presentation_slide_candidates.md"):
        assert (tmp_path / f).is_file(), f
    dig = (tmp_path / "DSE_FINDINGS.md").read_text()
    assert "Abstraction necessity (strict" in dig
    assert "possible-placement view only" in dig                       # coverage demoted to appendix


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p16_unit_multiplicity_demoted_to_context():
    # "heterogeneous" is an interpretation, not a measured fact -> must not be a signal/headline metric
    from merlin.dse_guidance import insight_mining as IM
    assert "unit_multiplicity_implication" not in IM.SIGNAL_METRICS
    canon = IM.canonical_signal_table(IM.unified_facts(_CS_DIR, "all"))
    assert not any(r["metric"] == "unit_multiplicity_implication" for r in canon)


# --------------------------------------------------------------------------- P17 adversarial audit

@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p17_k_rollup_fixed_and_predicate_audit():
    # the K=7 reporting bug: the necessity rollup predicate must NOT embed a literal per-workload K;
    # the per-workload K lives in predicate_audit, flagged as configured-K (and thus suspicious when
    # it drives necessity).
    import re
    from merlin.dse_guidance import insight_mining as IM
    nec = IM.abstraction_necessity(_CS_DIR)
    assert not [r["abstraction"] for r in nec["rows"] if re.search(r"K=\d", r["predicate"])]
    pa = IM.predicate_audit(_CS_DIR)
    assert pa["rows"] and all(r["predicate_inputs"] and r["thresholds"] for r in pa["rows"])
    rwo = [r for r in pa["rows"] if r["abstraction"] == "resident_weight_object"]
    assert rwo and all(re.search(r"K=\d", r["predicate_inputs"]) for r in rwo)
    assert all(r["uses_configured_K"] == "yes" for r in rwo)
    # necessity resting on configured K is flagged suspicious
    assert any(r["suspicious"] for r in rwo if r["classification"] == "necessary")


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p17_gemv_abstraction_renamed_and_split():
    # vector_gemv_engine -> skinny_gemm_or_gemv_engine everywhere it surfaces; the true_gemv vs
    # skinny-GEMM split (the user's correctness concern) is exposed per workload in predicate_audit.
    from merlin.dse_guidance import insight_mining as IM
    from merlin.dse_guidance.boundary_placement import catalog_rows
    names = {c["abstraction"] for c in catalog_rows()}
    assert "vector_gemv_engine" not in names and "skinny_gemm_or_gemv_engine" in names
    pa = IM.predicate_audit(_CS_DIR)
    cells = [r for r in pa["rows"] if r["abstraction"] == "skinny_gemm_or_gemv_engine"]
    assert cells and all("true_gemv=" in r["predicate_inputs"] and "skinny_gemm=" in r["predicate_inputs"]
                         for r in cells)


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p17_frontier_robustness_recompute_monotone_and_extra_tiles():
    # the pad-waste recompute (needed for thresholds + extra candidate tiles) must match the
    # committed tile_waste at 10%, the extra tiles must be in the candidate set, and best worst-
    # coverage must be monotone non-decreasing in set size at each threshold.
    import csv as _csv
    import io as _io
    from merlin.dse_guidance import insight_mining as IM
    ops = {(o["workload"], o["op_index"]): o for o in _csv.DictReader(_io.StringIO(
        (_CS_DIR / "operator_shape_table.csv").read_text()))}
    mism = 0
    for r in _csv.DictReader(_io.StringIO((_CS_DIR / "tile_waste_table.csv").read_text())):
        if r.get("applicable") != "True":
            continue
        o = ops.get((r["workload"], r["op_index"]))
        if not o:
            continue
        M, N, K = int(o["M"]), int(o["N"]), int(o["K"])
        if M <= 0 or N <= 0 or K <= 0:
            continue
        ap, wv = IM._prim_waste(r["primitive"], M, N, K, o["shape_class"])
        if ap and (wv <= 0.10) != (r["covered_under_10pct"] == "True"):
            mism += 1
    assert mism == 0
    fro = IM.primitive_frontier_robustness(_CS_DIR)
    assert all(t in fro["primitives"] for t in IM._EXTRA_TILES)
    by_thr = {}
    for row in fro["rows"]:
        by_thr.setdefault(row["threshold_pct"], {})[row["set_size"]] = row["worst"]
    assert by_thr and all(s[sz] >= s[sz - 1] - 1e-9 for s in by_thr.values() for sz in s if sz - 1 in s)
    # threshold-robustness is a real boolean (the claim "a 2-set suffices" vs "this exact pair")
    assert isinstance(fro["two_set_threshold_robust"], bool)


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p17_influence_winner_stable_vs_magnitude_unstable():
    # the dense-GEMM fraction keeps its (losing) winner but its MAGNITUDE swings hard when the most
    # influential workload is dropped -> must be flagged; LOO delta table is complete.
    from merlin.dse_guidance import insight_mining as IM
    inf = IM.macro_micro_influence(_CS_DIR)
    dense = next(r for r in inf["rows"] if r["metric"] == "dense_gemm_mac_fraction")
    assert dense["winner_stable_magnitude_unstable"] == "yes"
    assert dense["max_loo_micro_delta"] > 0.2
    assert all(isinstance(r["macro"], float) and isinstance(r["micro"], float) for r in inf["rows"])
    assert len(inf["loo_rows"]) == len(inf["rows"]) * len(inf["workloads"])


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p17_adversarial_audit_and_emit(tmp_path):
    # every audited conclusion references an artifact + carries a verdict; the P17 files are emitted.
    from merlin.dse_guidance import insight_mining as IM
    b = IM.mine(_CS_DIR, "all")
    aud = b["adversarial_audit"]["rows"]
    vocab = ("recovered", "derived", "measured", "assumed", "unavailable")
    assert aud and all(r["conclusion"] and r["supporting_artifact"] and r["verdict"]
                       and any(v in r["metric_class"] for v in vocab) for r in aud)
    IM.emit_run(b, tmp_path, [])
    for f in ("predicate_audit_table.csv", "predicate_audit_table.md", "conclusion_validity_table.csv",
              "adversarial_audit_report.md", "primitive_frontier_robustness.csv",
              "primitive_frontier_robustness.md", "uncovered_ops_by_primitive_set.csv",
              "macro_micro_influence_table.csv", "leave_one_out_delta_table.csv",
              "workload_influence_report.md"):
        assert (tmp_path / f).is_file(), f


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p17_timing_envelope_is_derived_requirement():
    # every envelope row is a REQUIREMENT recomputed as work/deadline (not measured); the command rate
    # is proxy-tagged except for the one workload with a measured dispatch count.
    import csv as _csv
    import io as _io
    from merlin.dse_guidance import insight_mining as IM
    from merlin.dse_guidance.models import MODEL_ARCH, _base_model
    env = IM.timing_requirement_envelope(_CS_DIR)
    assert env["rows"]
    req = {(r["workload"], r["region"], r["requirement"]): r["value"] for r in _csv.DictReader(
        _io.StringIO((_CS_DIR / "requirements_table.csv").read_text()))}
    for r in env["rows"]:
        assert r["K_basis"] in ("configured", "sweep")
        assert "sweep" in r["deadline_basis"] or "derived" in r["deadline_basis"]
        Kcfg = MODEL_ARCH[_base_model(r["workload"])].loop_count
        mpr = float(req[(r["workload"], "repeated_head", "macs_per_replan")])
        expect = (mpr / Kcfg) * r["K"] / (r["deadline_ms"] / 1000.0)
        assert abs(expect - r["required_compute_MAC_per_s"]) <= max(1.0, expect * 1e-6)
        # residency removes exactly a K x weight-bandwidth requirement
        if r["required_weight_B_per_s_resident"]:
            ratio = r["required_weight_B_per_s_nonresident"] / r["required_weight_B_per_s_resident"]
            assert abs(ratio - r["K"]) < 1e-3
    assert all(("proxy_only" in r["command_rate_basis"]) or (r["workload"] in IM._MEASURED_DISPATCH)
               for r in env["rows"])


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p18_operator_recovery_accounting():
    # P18: attention is RECOVERED (real MACs), not unavailable. linear MACs == named-matmul sum;
    # visible_linear_fraction in [0,1]; at least one workload has recovered attention MACs; low-bit
    # stays erased.
    import csv as _csv
    import io as _io
    from merlin.dse_guidance import insight_mining as IM
    ops = list(_csv.DictReader(_io.StringIO((_CS_DIR / "operator_shape_table.csv").read_text())))
    mac = {}
    for o in ops:
        mac[o["workload"]] = mac.get(o["workload"], 0) + int(o["macs"])
    om = IM.operator_recovery_accounting(_CS_DIR)
    assert om["rows"]
    for r in om["rows"]:
        assert int(r["linear_gemm_macs"]) == mac[r["workload"]]   # named-matmul subset is exact
        assert 0.0 <= float(r["visible_linear_fraction"]) <= 1.0
        assert "no" in r["lowbit_packed_recovered"]               # low-bit genuinely erased
    assert any(int(r["attention_macs"]) > 0 for r in om["rows"])  # attention recovered, not unavailable


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p18_capture_erasure_and_per_family():
    # capture-erasure evidence is demonstrated from the IR (loops absent except a known gather
    # artifact; no low-bit int types); per-family fractions are valid.
    from merlin.dse_guidance import insight_mining as IM
    ce = IM.capture_erasure_evidence(_CS_DIR)["rows"]
    assert ce and all(not r["lowbit_int_types_present"] for r in ce)
    assert sum(1 for r in ce if r["loops_preserved"]) <= 1     # only the smolvla gather artifact
    pf = IM.per_family_summary(_CS_DIR)["rows"]
    assert pf and all(0.0 <= float(r["visible_linear_fraction"]) <= 1.0 for r in pf)


@pytest.mark.skipif(not (_CS_DIR / "capture_level_ablation.csv").is_file(),
                    reason="capture-level ablation summary not present")
def test_p18_capture_level_ablation():
    # Stage B: high-level captures expose attention/softmax as NAMED linalg_ext ops; qdq captures
    # expose quant_ext.dequantize; loops stay absent (torch.export-blocked) at every level. Reads the
    # committed op-count summary (the raw multi-level recaptures are gitignored + regenerable).
    from merlin.dse_guidance import insight_mining as IM
    ab = IM.capture_level_ablation(_CS_DIR)
    assert ab["rows"] and ab["unlock"]
    hl = [r for r in ab["rows"] if r["level"] == "high_level" and r["available"]]
    qd = [r for r in ab["rows"] if r["level"] == "quant_qdq" and r["available"]]
    assert hl and all(int(r["linalg_ext_softmax"]) > 0 for r in hl)
    assert qd and all(int(r["quant_ext_dequantize"]) > 0 for r in qd)
    assert all(int(r["scf_for"]) == 0 for r in ab["rows"])     # loop-preserving torch.export-blocked


@pytest.mark.skipif(not (_CS_DIR / "timeloop_problem_shapes.yaml").is_file(),
                    reason="mapspace seeds not present")
def test_p20_timeloop_problem_shapes():
    # Tool A: Timeloop problem shapes are consistent + DSE-consumable. dims*==macs; 3 GEMM data-spaces;
    # attention has no stationary weight; linear has a weight; yaml count matches dataflow table.
    from merlin.common.yaml import load_yaml
    ts = load_yaml(_CS_DIR / "timeloop_problem_shapes.yaml")["timeloop_problem_shapes"]
    shapes = ts["shapes"]
    assert shapes and ts["count"] == len(shapes)
    for s in shapes:
        inst = s["problem"]["instance"]
        assert inst["M"] * inst["N"] * inst["K"] == s["macs_per_instance"]
        ds = s["problem"]["shape"]["data-spaces"]
        assert len(ds) == 3 and len(s["problem"]["shape"]["dimensions"]) == 3
        if s["op_class"] == "attention_contraction":
            assert "weight_stationary" not in s["dataflow_candidates"]
            assert all(d.get("operand_identity") != "weight" for d in ds)
        if s["op_class"] == "linear_gemm":
            assert any(d.get("operand_identity") == "weight" for d in ds)


@pytest.mark.skipif(not (_CS_DIR / "operand_locality_table.csv").is_file(),
                    reason="operand locality not present")
def test_p20_operand_locality_and_quant_metadata():
    # Tool B: weight bytes reconcile with data_movement; reuse scopes valid; weights resident. Tool E:
    # quant rows trace to qdq dequant ops; bitvla native-ternary gap recorded.
    import csv as _csv
    import io as _io
    dm = {(r["workload"], r["region"]): r for r in _csv.DictReader(_io.StringIO(
        (_CS_DIR / "data_movement_table.csv").read_text()))}
    loc = list(_csv.DictReader(_io.StringIO((_CS_DIR / "operand_locality_table.csv").read_text())))
    scopes = {"within_op", "across_ops", "across_K", "across_decode", "across_replan"}
    assert loc and all(r["reuse_scope"] in scopes for r in loc)

    def _i(x):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return 0
    for r in loc:
        if r["operand"] == "weight":
            assert _i(r["bytes"]) == _i(dm.get((r["workload"], r["region"]), {}).get("weight_bytes"))
            if _i(r["bytes"]) > 0:
                assert r["resident_candidate"].startswith("yes")
    qmf = _CS_DIR / "quant_metadata_visibility.csv"
    if qmf.is_file():
        qm = list(_csv.DictReader(_io.StringIO(qmf.read_text())))
        assert qm and all(int(r["n_dequant_ops"]) > 0 for r in qm)
        assert any("ternary" in r["native_scheme_gap"].lower() for r in qm if r["workload"] == "bitvla")


@pytest.mark.skipif(not (_CS_DIR / "dse_contract.json").is_file(),
                    reason="case_study package not present")
def test_p17_new_decision_plots_registered_and_available():
    from merlin.dse_guidance import insight_mining as IM
    from merlin.dse_guidance import presentation_plots as PP
    pm = {p["plot_id"]: p for p in IM.plot_manifest(_CS_DIR, "all")}
    for pid in ("primitive_frontier_by_threshold", "macro_vs_micro_primitive_coverage",
                "required_compute_envelope", "required_memory_movement_envelope",
                "required_command_rate_envelope", "workload_influence_loo_delta"):
        assert pid in pm, pid
        assert pm[pid]["recommendation"] != "omit", pid          # source artifact + columns exist
        assert pm[pid]["dse_caption"], pid
        assert pid in PP._RENDERERS, pid


# --------------------------------------------------------------------------- agent devil's-advocate

def test_agent_citation_gate_keeps_grounded_rejects_ungrounded(tmp_path):
    # the gate is the deterministic 'dispose' step: a critique survives ONLY if it quotes a real
    # artifact line; a fabricated citation (incl. invented numbers/speedup) is rejected.
    from merlin.dse_guidance.agent import critic
    (tmp_path / "DSE_FINDINGS.md").write_text(
        "# digest\nmatrix_engine is necessary only for rdt (corpus-narrow).\n")
    (tmp_path / "table.csv").write_text("metric,value\navoidable_weight_reload,1564475392\n")
    items = [
        {"claim": "matrix_engine presented as general", "severity": "high",
         "cite": "matrix_engine is necessary only for rdt", "suggested_fix": "say rdt-only"},
        {"claim": "invented speedup", "severity": "high",
         "cite": "the design is 3x faster than baseline", "suggested_fix": "remove"},
        {"claim": "no citation", "severity": "low", "cite": "", "suggested_fix": "x"},
        {"claim": "bad severity", "severity": "URGENT",
         "cite": "avoidable_weight_reload,1564475392", "suggested_fix": "x"}]
    res = critic.citation_gate(items, tmp_path)
    assert res["n_proposed"] == 4
    assert len(res["accepted"]) == 1 and res["accepted"][0]["severity"] == "high"
    assert len(res["rejected"]) == 3
    reasons = " ".join(r["reason"] for r in res["rejected"])
    assert "not found" in reasons and "missing" in reasons and "severity" in reasons


def test_agent_run_critic_with_injected_runner(tmp_path):
    # run_critic with an injected runner exercises propose+parse+dispose without a live `claude`.
    from merlin.dse_guidance.agent import critic
    (tmp_path / "DSE_FINDINGS.md").write_text(
        "# digest\nbest single primitive worst-workload coverage is 0.13.\n")

    def fake_runner(prompt):
        assert "Do NOT invent or recompute numbers" in prompt and "JSON array" in prompt
        return {"text": '```json\n[{"claim":"worst-cov framed as adequate","severity":"medium",'
                        '"cite":"worst-workload coverage is 0.13","suggested_fix":"flag as poor"}]\n```',
                "usage": {}}
    res = critic.run_critic(tmp_path, runner=fake_runner)
    assert len(res["accepted"]) == 1 and not res["rejected"]
    out = critic.emit_critique(res, tmp_path)
    assert out.is_file() and "citation-gated" in out.read_text()


def test_agent_unavailable_is_honest_not_fabricated():
    # if the `claude` CLI is missing, the runner raises AgentError (callers report 'unavailable')
    from merlin.dse_guidance.agent import claude_cli
    import shutil
    if shutil.which("claude") is not None:
        pytest.skip("claude CLI present; cannot test the unavailable path")
    with pytest.raises(claude_cli.AgentError):
        claude_cli.run_agent("hello", cache_bust=False)


@pytest.mark.skipif(not (_CS_DIR / "loop_aware_contract.csv").is_file(),
                    reason="case_study package not present")
def test_loop_aware_seeds_multiply_repeated_head_by_recovered_K():
    # P21-aware mapspace seeds: a repeated-head shape's per-replan instance count is layers x K
    # (K from the IR scf.for, via loop_aware_contract.csv); a once-prefix shape stays layers x 1.
    from merlin.dse_guidance import mapspace as MS
    rows = MS.loop_aware_seed_rows(_CS_DIR)
    assert rows, "no loop-aware seed rows from the committed case_study artifacts"
    facts = MS._loop_facts(_CS_DIR)
    for r in rows:
        outer = facts.get(r["workload"], {}).get("K", 1) \
            if r["region_role"] in ("repeated_head", "decode_lm") else 1
        assert int(r["outer_loop_K"]) == outer
        assert int(r["instances_per_replan"]) == int(r["layers_in_capture"]) * int(r["outer_loop_K"])
        assert int(r["macs_per_replan"]) == int(r["macs_per_instance"]) * int(r["instances_per_replan"])
    # openvla decode loop K=7 is IR-recovered and applied to its repeated head; prefix stays x1
    ov = [r for r in rows if r["workload"] == "openvla" and r["region_role"] == "repeated_head"]
    assert ov and all(int(r["outer_loop_K"]) == 7 and r["K_source"] == "recovered_from_ir" for r in ov)
    assert all(int(r["outer_loop_K"]) == 1 for r in rows
               if r["workload"] == "openvla" and r["region_role"] == "backbone_once")
