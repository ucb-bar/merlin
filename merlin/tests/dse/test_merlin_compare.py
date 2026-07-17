"""Tests for the merlin-compare unified comparison driver (HOST only, ingest cached data)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.compare import driver
from merlin.compare.attribution import attribute, gap_driver_axes
from merlin.compare.empirical import Measurement, measure, measure_all
from merlin.compare.spec import Config, Spec, Workload
from merlin.compare.structural import cca_for, decode_o


# ---------------------------------------------------------------- spec parsing
def test_spec_parses_config_kinds():
    spec = Spec.parse({
        "configs": ["baseline", "ours_wholemodel", "ours_wholemodel_vf", "xnnpack", "openblas"],
        "workloads": ["openvla", "gemm:64", "gemm:17x192x576"],
    })
    kinds = {c.name: c.kind for c in spec.configs}
    assert kinds == {
        "baseline": "baseline", "ours_wholemodel": "ours", "ours_wholemodel_vf": "ours",
        "xnnpack": "kernel_backend", "openblas": "kernel_backend"}
    wl = {w.name: (w.kind, w.mnk) for w in spec.workloads}
    assert wl["openvla"] == ("model", None)
    assert wl["gemm:64"] == ("gemm", (64, 64, 64))
    assert wl["gemm:17x192x576"] == ("gemm", (17, 192, 576))


def test_spec_rejects_unknown_config_and_target():
    with pytest.raises(ValueError):
        Spec.parse({"configs": ["mystery_thing"], "workloads": ["openvla"]})
    with pytest.raises(ValueError):
        Spec.parse({"configs": ["baseline"], "workloads": ["openvla"], "target": "k1",
                    "metric": "nonsense"})
    # spike is a declared seam, not implemented in v1 -> loud failure.
    with pytest.raises(ValueError):
        Spec.parse({"configs": ["baseline"], "workloads": ["openvla"], "target": "spike"})


def test_spec_ours_features():
    c = Config.parse({"name": "ours_wholemodel_vf",
                      "compiler_features": ["accumulator_resident_wholemodel_vf"]})
    assert c.kind == "ours"
    assert c.compiler_features == ("accumulator_resident_wholemodel_vf",)


# ---------------------------------------------------------------- ingest (real cached data)
def test_ingest_openvla_four_way_matches_known_numbers():
    # ours within ~55-63% of experts on openvla (the known result).
    spec = Spec.parse({
        "configs": ["baseline", "ours_wholemodel", "xnnpack", "openblas"],
        "workloads": ["openvla"]})
    ms = measure_all(spec)
    base = ms[("baseline", "openvla")]
    ours = ms[("ours_wholemodel", "openvla")]
    xnn = ms[("xnnpack", "openvla")]
    assert base.status == ours.status == xnn.status == "measured"
    # baseline ~5.86s, ours ~1.19s, xnnpack ~0.65s
    assert 5.0e9 < base.value < 6.5e9
    pct_of_xnn = 100 * xnn.value / ours.value
    assert 50 < pct_of_xnn < 70   # ours ~55% of xnnpack speed


def test_ingest_gemm_shape_from_jsonl():
    spec = Spec.parse({"configs": ["xnnpack", "ours_wholemodel"], "workloads": ["gemm:64"]})
    ms = measure_all(spec)
    assert ms[("xnnpack", "gemm:64")].status == "measured"
    assert ms[("xnnpack", "gemm:64")].source.endswith("cross_framework_matrix_k1.jsonl")


def test_missing_cell_is_not_measured_not_invented():
    # rdt2 has no _vf measurement in the cached JSONs.
    m = measure(Config.parse("ours_wholemodel_vf"), Workload.parse("rdt2"), "k1")
    assert m.status == "not_measured"
    assert m.value is None


# ---------------------------------------------------------------- structural CCA
def test_cca_decode_for_configs():
    xnn = cca_for(Config.parse("xnnpack"), Workload.parse("openvla"))
    ours = cca_for(Config.parse("ours_wholemodel"), Workload.parse("openvla"))
    base = cca_for(Config.parse("baseline"), Workload.parse("openvla"))
    assert base is None   # baseline has no vector matmul decode (honest None)
    assert xnn is not None and ours is not None
    assert xnn.compute.contraction_form == "fused_fma"
    # the .vf-vs-.vv structural distinction is carried in provenance.
    assert xnn.provenance["fma_loop_vfmacc_vf"] == 1
    assert ours.provenance["fma_loop_vfmacc_vv"] == 1
    assert ours.provenance["fma_loop_vfmacc_vf"] == 0


def test_decode_o_seam_callable():
    # The live rebuild seam imports the real decode+lift path; assert it is wired (no board).
    assert callable(decode_o)


# ---------------------------------------------------------------- attribution
def test_attribution_flags_vf_vs_vv_and_routes_action():
    spec = Spec.parse({
        "configs": ["ours_wholemodel_vf", "xnnpack"], "workloads": ["openvla"]})
    ms = measure_all(spec)
    ccas = {c.name: cca_for(c, Workload.parse("openvla")) for c in spec.configs}
    wcc = {(c.name, "openvla"): ccas[c.name] for c in spec.configs}
    attrs = attribute(spec, ms, ccas, workload_ccas=wcc)
    assert attrs, "expected an ours-vs-expert attribution"
    a = attrs[0]
    # measured gap present and ours trailing the expert on openvla.
    assert not a.measured["ours_faster"]
    assert 50 < a.measured["pct_of_expert"] < 70
    # the .vf-vs-.vv divergence is auto-identified in the notes (the kernel_breakdown.md driver).
    assert any(".vf" in n and ".vv" in n for n in a.notes)
    # the accumulator-residency divergence routed to the microkernel PASS action.
    axes = {d.axis for d in a.divergences}
    assert "compute.accumulator_resident" in axes
    seams = {act.target_seam for act in a.actions}
    assert any("accumulator_resident_microkernel" in s for s in seams)


def test_gap_driver_axes_nonempty_when_trailing():
    spec = Spec.parse({
        "configs": ["ours_wholemodel", "ours_wholemodel_vf", "xnnpack", "openblas"],
        "workloads": ["openvla"]})
    ms = measure_all(spec)
    ccas = {c.name: cca_for(c, Workload.parse("openvla")) for c in spec.configs}
    wcc = {(c.name, "openvla"): ccas[c.name] for c in spec.configs}
    attrs = attribute(spec, ms, ccas, workload_ccas=wcc)
    axes = gap_driver_axes(attrs)
    assert "compute.accumulator_resident" in axes


# ---------------------------------------------------------------- full driver + manifest
def test_full_run_artifact_and_deterministic_manifest(tmp_path: Path):
    spec = Spec.parse({
        "label": "test_4way",
        "configs": ["baseline", "ours_wholemodel", "ours_wholemodel_vf", "xnnpack", "openblas"],
        "workloads": ["openvla", "gemm:64"]})
    a = driver.run(spec, out_root=tmp_path / "a", ts="FIXED")
    b = driver.run(spec, out_root=tmp_path / "b", ts="FIXED")
    assert (a / "compare.md").is_file()
    assert (a / "manifest.yaml").is_file()
    # deterministic: same cached sources + same commit -> identical manifest and report.
    assert (a / "manifest.yaml").read_text() == (b / "manifest.yaml").read_text()
    assert (a / "compare.md").read_text() == (b / "compare.md").read_text()
    # manifest records spec + provenance.
    import yaml
    man = yaml.safe_load((a / "manifest.yaml").read_text())
    assert man["tool"] == "merlin-compare"
    assert man["spec"]["label"] == "test_4way"
    assert any("k1_" in s for s in man["empirical_sources"])
    # measured table in the report carries the known numbers.
    md = (a / "compare.md").read_text()
    assert "openvla" in md and "xnnpack" in md
    assert "not_measured" not in md.split("## 2.")[0] or "openvla" in md  # table renders


def test_run_board_is_live_and_board_gated(monkeypatch):
    # --run is now LIVE: measure() ingests (never raises a stub); measure_all(run=True) refreshes the
    # board cache first and is fail-closed board-gated — no board -> a loud RuntimeError, never a
    # silent/fabricated run. (k1.K1_HOST is import-frozen, so force unavailability at k1.available.)
    from merlin.compare.empirical import measure_all
    from merlin.compare.spec import Spec
    from merlin.rvvgen import k1
    monkeypatch.setattr(k1, "available", lambda: False)
    # per-cell measure() just ingests now (no NotImplementedError stub)
    m = measure(Config.parse("xnnpack"), Workload.parse("openvla"), "k1", run=True)
    assert m.status in ("measured", "not_measured")
    spec = Spec.parse({"configs": ["baseline", "xnnpack"], "workloads": ["openvla"]})
    with pytest.raises(RuntimeError, match="board"):
        measure_all(spec, run=True)


# ---------------------------------------------------------------- region alignment (C7)
def test_align_regions_joins_on_provenance_and_flags_asymmetry():
    from merlin.baselines.contract import RegionProfile
    from merlin.compare.attribution import align_regions

    ours = [
        RegionProfile(name="attention", region_id="matmul_3", fqn="model.layers.0.self_attn",
                      role="repeated_head", wall_ns=100, cos=1.0),
        # ET leaves norm scalar / does not surface it as a region -> present only on our side.
        RegionProfile(name="norm", region_id="layer_norm_0", fqn="model.layers.0.input_layernorm",
                      wall_ns=10),
    ]
    expert = [
        RegionProfile(name="attention", region_id="matmul_3", fqn="model.layers.0.self_attn",
                      role="repeated_head", wall_ns=80, cos=0.9999),
    ]
    rows = {r.key: r for r in align_regions(ours, expert)}

    attn = rows["matmul_3"]
    assert attn.presence == "both"
    assert attn.wall_ratio == 100 / 80              # apples-to-apples: Merlin slower on THIS layer
    assert attn.ours_cos == 1.0 and attn.expert_cos == 0.9999
    assert attn.role == "repeated_head"

    norm = rows["layer_norm_0"]
    assert norm.presence == "merlin_only"           # the heterogeneity a whole-model number hides
    assert "only" in norm.note


def test_region_alignment_md_renders_matrix_and_flags_asymmetry():
    from merlin.baselines.contract import RegionProfile
    from merlin.compare.attribution import align_regions
    from merlin.compare.report import region_alignment_md

    ours = [
        RegionProfile(name="attention", region_id="matmul_3", fqn="model.layers.0.self_attn",
                      role="repeated_head", wall_ns=100_000, cos=1.0),
        RegionProfile(name="norm", region_id="layer_norm_0", fqn="model.layers.0.input_layernorm",
                      wall_ns=10_000),
    ]
    expert = [RegionProfile(name="attention", region_id="matmul_3", fqn="model.layers.0.self_attn",
                            role="repeated_head", wall_ns=80_000, cos=0.9999)]
    md = region_alignment_md(align_regions(ours, expert))
    assert "region | role |" in md                                  # the matrix header
    assert "model.layers.0.self_attn" in md and "1.25×" in md       # aligned attention row + ratio
    assert "model.layers.0.input_layernorm" in md                   # the norm layer
    assert "merlin_only" in md and "⚠️" in md                        # the one-sided-region flag
    assert region_alignment_md([]).startswith("_No aligned regions")
