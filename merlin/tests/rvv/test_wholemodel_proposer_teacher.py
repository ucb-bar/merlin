"""Per-op-teacher whole-model proposer (GAP 1).

Proves the beam's whole-model proposer USES XNNPACK's CCA as a per-op teacher instead of ignoring
divergences and emitting a fixed lever list:
  * a family whose expert-vs-ours CCA DIVERGES routes to the matching feature fork (gelu -> activation
    vectorization, reduce -> vectorize_reduction);
  * the hybrid UNIONs the per-op-teacher forks with the census hardcodes, deduped, honoring the
    <=1-schedule-replacement composition rule;
  * a family with NO XNNPACK primitive (sdpa/layer_norm) gets an honest no-teacher RECORD, never a
    faked divergence;
  * the section-lift path (ours_section_cca) is wired correctly (mock build_fn; no board/compile).

All CCAs here are lifted from the committed asm fixtures (no toolchain) or crafted mocks — board-free.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.kernels import cca, cca_compare
from merlin.mining import wholemodel_proposer as W


def _ours(op, **compute):
    return cca.CCA(op=op, backend=["rvv"], compute=cca.ComputeFacet(op=op, **compute))


def _feature_sets(forks):
    return [tuple(f.overrides.get("compiler_features", [])) for f in forks if f.forkable]


# --- expert-fixture lifts (the harvested teachers) ------------------------------------------------

def test_expert_fixtures_lift_the_teachable_property():
    """gelu -> vectorized_polynomial; reduce -> vredsum_tree; matmul -> fused_fma. No-teacher -> None."""
    assert W.expert_family_cca("gelu").compute.activation_vectorization == "vectorized_polynomial"
    assert W.expert_family_cca("sigmoid").compute.activation_vectorization == "vectorized_polynomial"
    assert W.expert_family_cca("reduce").compute.reduction_form == "vredsum_tree"
    assert W.expert_family_cca("softmax").compute.reduction_form == "vredsum_tree"
    assert W.expert_family_cca("matmul").compute.contraction_form == "fused_fma"
    # families with no XNNPACK primitive have no teacher.
    for fam in ("sdpa", "layer_norm", "embedding", "index_gather"):
        assert W.expert_family_cca(fam) is None


# --- engine 1: routing a real teacher divergence into a fork --------------------------------------

def test_gelu_teacher_routes_activation_fork():
    expert = W.expert_family_cca("gelu")
    ours = _ours("gelu", activation_vectorization="scalar_libm_call")
    divs = cca_compare.compare(expert, ours, evidence=["xnnpack:gelu"])
    assert any(d.axis == "compute.activation_vectorization" for d in divs)
    forks = W.route_divergence_forks(divs, {"compiler_features": []})
    feats = [f for f in forks if f.forkable and
             "vectorized_transcendental_activation" in f.overrides.get("compiler_features", [])]
    assert feats, "gelu teacher divergence did not route to the activation-vectorization feature"
    assert "teacher:xnnpack-cca" in feats[0].evidence


def test_reduce_teacher_routes_vectorize_reduction():
    expert = W.expert_family_cca("reduce")
    ours = _ours("reduce", reduction_form="none")
    divs = cca_compare.compare(expert, ours, evidence=["xnnpack:reduce"])
    forks = W.route_divergence_forks(divs, {"compiler_features": []})
    assert any("vectorize_reduction" in f.overrides.get("compiler_features", [])
               for f in forks if f.forkable)


# --- the hybrid: teacher UNION census hardcodes ---------------------------------------------------

def test_hybrid_unions_teacher_and_census_hardcodes():
    """propose_wholemodel_levers consumes the beam's divergences AND emits the census hardcodes."""
    expert = W.expert_family_cca("gelu")
    ours = _ours("gelu", activation_vectorization="scalar_libm_call")
    divs = cca_compare.compare(expert, ours)
    forks = W.propose_wholemodel_levers(divs, {"compiler_features": []})
    sets = _feature_sets(forks)
    flat = {f for s in sets for f in s}
    # teacher-driven activation feature present...
    assert "vectorized_transcendental_activation" in flat
    # ...alongside the graph-layout hardcode a facet diff cannot emit.
    assert "fuse_transpose_b" in flat


def test_empty_divergences_degrades_to_census_hardcodes():
    """No divergences -> the proposer is exactly the census hardcodes (backward compatible)."""
    forks = W.propose_wholemodel_levers([], {"compiler_features": []})
    got = {f for s in _feature_sets(forks) for f in s}
    assert got == {name for name, _ in W.RANKED_LEVERS}


def test_no_duplicate_feature_forks():
    """A teacher fork and a census hardcode for the SAME feature collapse to one fork."""
    expert = W.expert_family_cca("reduce")   # routes to vectorize_reduction, also a hardcode lever
    ours = _ours("reduce", reduction_form="none")
    divs = cca_compare.compare(expert, ours)
    forks = W.propose_wholemodel_levers(divs, {"compiler_features": []})
    n_reduction = sum("vectorize_reduction" in s for s in _feature_sets(forks))
    assert n_reduction == 1


# --- composition: never two full-schedule-replacement features ------------------------------------

def test_composition_never_stacks_two_schedule_replace():
    from merlin.llvmlower import impr_features as I
    # parent already carries a schedule-replacement feature.
    parent = {"compiler_features": ["accumulator_resident_wholemodel_vf_mrpad"]}
    expert = W.expert_family_cca("reduce")   # vectorize_reduction is also schedule_replace
    ours = _ours("reduce", reduction_form="none")
    divs = cca_compare.compare(expert, ours)
    forks = W.propose_wholemodel_levers(divs, parent)
    for s in _feature_sets(forks):
        reps = [f for f in s if getattr(I.get(f), "schedule_replace", False)]
        assert len(reps) <= 1, f"two schedule-replace features co-enabled: {s}"


# --- honest no-teacher path -----------------------------------------------------------------------

def test_no_teacher_families_recorded_not_faked():
    """A family with no XNNPACK primitive yields a no-teacher NOTE, no divergence."""
    def expert_fn(fam):
        return None                      # simulate: no teacher for any family
    def ours_fn(fam):
        return _ours(fam)
    divs, notes = W.per_family_teacher_divergences(
        "unused", families=["sdpa", "layer_norm"], expert_fn=expert_fn, ours_fn=ours_fn)
    assert divs == []
    fams = {n[0] for n in notes}
    assert {"sdpa", "layer_norm"} <= fams


def test_make_per_op_teacher_proposer_precomputed():
    """The bound closure emits teacher forks + hardcodes + honest no-teacher records, 2-arg contract."""
    expert = W.expert_family_cca("gelu")
    ours = _ours("gelu", activation_vectorization="scalar_libm_call")
    teacher_divs = cca_compare.compare(expert, ours)
    notes = [("sdpa", "no XNNPACK attention primitive")]
    proposer = W.make_per_op_teacher_proposer(
        precomputed_divergences=teacher_divs, no_teacher_notes=notes)
    forks = proposer([], {"compiler_features": []})
    flat = {f for s in _feature_sets(forks) for f in s}
    assert "vectorized_transcendental_activation" in flat        # teacher
    assert "fuse_transpose_b" in flat                            # hardcode
    noteacher = [f for f in forks if not f.forkable and f.targets == "noteacher:sdpa"]
    assert noteacher and "attention" in noteacher[0].note        # honest record, not a fork


# --- the section-lift path wiring (board-free via a mock build_fn) --------------------------------

_MODEL_MLIR = """\
module {
  func.func @forward(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.matmul {prov.op = "matmul", prov.region_id = "r_mm"} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = linalg.elemwise_unary {prov.op = "gelu", prov.region_id = "r_gelu"} ins(%0 : tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    func.return %1 : tensor<4x4xf32>
  }
}
"""


def test_family_region_ids_filters_by_family(tmp_path):
    (tmp_path / "model.mlir").write_text(_MODEL_MLIR)
    assert W.family_region_ids(tmp_path, "gelu") == ["r_gelu"]
    assert W.family_region_ids(tmp_path, "matmul") == ["r_mm"]
    assert W.family_region_ids(tmp_path, "sdpa") == []


def test_ours_section_cca_wiring(tmp_path, monkeypatch):
    """ours_section_cca: filter regions -> build the section -> build_fn -> lift the emitted CCA.

    build_section_bundle and build_fn are mocked (the board/compile seam); we assert the section was
    built for the right region_ids and the CCA is lifted from the returned objdump with the right op."""
    (tmp_path / "model.mlir").write_text(_MODEL_MLIR)
    seen = {}

    import merlin.mining.section_build as SB

    def fake_build_section_bundle(model_dir, region_ids, out_dir, *, seed=0):
        seen["region_ids"] = list(region_ids)
        seen["out_dir"] = str(out_dir)
        return {"region_ids": list(region_ids)}
    monkeypatch.setattr(SB, "build_section_bundle", fake_build_section_bundle)

    # the "emitted" section asm: reuse the committed gelu fixture text as a stand-in for our object.
    fixture_text = (repo_root() / "merlin/tests/data/cca_asm/xnnpack_gelu_rvv.objdump").read_text()

    def fake_build_fn(section_dir):
        seen["build_fn_dir"] = str(section_dir)
        return fixture_text, ("some_undef_sym",)

    result = W.ours_section_cca(tmp_path, "gelu", build_fn=fake_build_fn, work_root=tmp_path / "work")
    assert seen["region_ids"] == ["r_gelu"]                 # scoped to the gelu region only
    assert seen["build_fn_dir"].endswith("section_gelu")    # section bundle dir passed to build_fn
    assert result is not None and result.op == "gelu"       # CCA lifted from the emitted asm


def test_ours_section_cca_none_when_family_absent(tmp_path):
    (tmp_path / "model.mlir").write_text(_MODEL_MLIR)
    called = {"n": 0}

    def build_fn(_):
        called["n"] += 1
        return "", None

    # no 'reduce' op in the model -> None, and the (board) build_fn is never invoked.
    assert W.ours_section_cca(tmp_path, "reduce", build_fn=build_fn) is None
    assert called["n"] == 0


def test_make_per_op_teacher_proposer_requires_inputs():
    with pytest.raises(ValueError):
        W.make_per_op_teacher_proposer()   # no precomputed divergences and no model_dir+build_fn
