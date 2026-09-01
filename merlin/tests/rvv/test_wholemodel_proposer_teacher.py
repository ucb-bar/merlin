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


# ---------------------------------------------------------------------------------------
# A lever that cannot be REGISTERED is a lever that is silently never proposed:
# `_feature_fork` -> `_composes` -> `impr_features.normalize` raises KeyError on an unknown
# name, `_composes` catches it and returns False, and the fork is dropped without a word.
# So "is every ranked lever registered" is not a tidiness check -- it is the difference
# between a searchable lever and an inert one.
# ---------------------------------------------------------------------------------------

def test_every_ranked_lever_is_registered_so_the_beam_can_actually_propose_it():
    from merlin.llvmlower import impr_features as I
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS

    known = set(I.known())
    missing = [f for f, _ in RANKED_LEVERS if f not in known]
    assert not missing, f"ranked levers the proposer cannot compose (silently dropped): {missing}"


def test_every_ranked_lever_actually_yields_a_forkable_proposal():
    """The stronger form: registration is necessary but not sufficient -- the proposal must come out
    forkable, with the lever in its feature set."""
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS, census_hardcode_forks

    proposed = {tuple(sorted(fp.overrides.get("compiler_features") or ())): fp
                for fp in census_hardcode_forks([])}
    for feat, _ in RANKED_LEVERS:
        fp = proposed.get((feat,))
        assert fp is not None, f"{feat} produced no fork"
        assert fp.forkable, f"{feat} produced a non-forkable proposal"


def test_per_op_register_block_is_a_ranked_lever_and_ranks_above_the_class_wide_clamps():
    """It SUPERSEDES the class-wide MR_mm/NR_bmm clamps rather than competing with them, so it must be
    offered before them: a class is not shape-homogeneous, and one degenerate extent in it otherwise
    forces every member off the vector path."""
    from merlin.llvmlower.impr_features import PEROP_BLOCK_NAME
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS

    names = [f for f, _ in RANKED_LEVERS]
    assert PEROP_BLOCK_NAME in names
    assert names.index(PEROP_BLOCK_NAME) < names.index("accumulator_resident_wholemodel_vf_mrpad")


def test_the_sentinel_fails_loud_if_it_ever_reaches_lowering_unresolved():
    """Registering the sentinel makes it searchable, but it is a REQUEST, not a lowering edit. If it
    survives to `apply_pipeline` nothing tagged the IR, so every contraction silently falls to
    convert-linalg-to-loops with correct numbers and a successful build -- the one failure mode no
    correctness gate can see. It must raise instead."""
    import pytest as _pytest

    from merlin.llvmlower import pipeline as P
    from merlin.llvmlower.impr_features import PEROP_BLOCK_NAME

    with _pytest.raises(RuntimeError) as e:
        P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset([PEROP_BLOCK_NAME]))
    assert "prepare_for_lowering" in str(e.value)
    assert "untagged" in str(e.value) or "scalar" in str(e.value)
    # ...and the frozen baseline is untouched by the registration
    assert P.build_rvv_pipeline("/tmp/s.mlir", features=frozenset()) == P.build_rvv_pipeline("/tmp/s.mlir")


# ---------------------------------------------------------------------------------------
# The transformer tail's teachers. `sin`, `cos` and `rsqrt` are census families the census
# has emitted all along, and they were in NEITHER FAMILY_TEACHERS NOR NO_TEACHER_FAMILIES --
# so no expert was ever lifted for them, no divergence could form, and the loop could not
# propose anything for RoPE or RMSNorm. Measured on small_llama int8: 16.63% of an INT8
# model's binary is scalar FLOAT, calling __kernel_sinf / __kernel_cosf / __kernel_rem_pio2f
# and __ieee754_sqrt per element.
# ---------------------------------------------------------------------------------------

def test_the_transformer_tail_families_are_registered_one_way_or_the_other():
    """A census family that is in neither registry is INVISIBLE, which is the failure this closes.
    Being registered with fixture=None is fine -- that is an honest no-teacher record."""
    from merlin.mining.wholemodel_proposer import (FAMILY_TEACHERS, NO_TEACHER_FAMILIES,
                                                   _coverage_maps)
    fam_map, _ = _coverage_maps()
    known = set(FAMILY_TEACHERS) | set(NO_TEACHER_FAMILIES)
    for f in ("sin", "cos", "rsqrt"):
        assert f in fam_map, f"{f} is expected to be a census family"
        assert f in known, f"census family {f!r} is in neither registry -> silently unteachable"


def test_the_rmsnorm_teacher_is_harvested_and_lifts_as_vectorized():
    """The expert must be LIFTED from a real disassembly, never declared -- the beam's hard principle
    is that the CCA is tool-composed."""
    from merlin.common.paths import repo_root
    from merlin.kernels import cca
    from merlin.kernels.decode import rvv
    from merlin.mining.wholemodel_proposer import FAMILY_TEACHERS

    t = FAMILY_TEACHERS["rsqrt"]
    assert t.fixture == "xnnpack_rsqrt_rvv.objdump"
    fx = repo_root() / "merlin/tests/data/cca_asm" / t.fixture
    if not fx.is_file():
        pytest.skip("rsqrt fixture not harvested in this checkout")
    expert = cca.lift_asm(rvv.decode_text(fx.read_text()), op="rsqrt", source="expert")
    assert expert.compute.activation_vectorization == "vectorized_polynomial"


def test_sin_and_cos_are_honest_no_teacher_records_with_the_reason_named():
    """XNNPACK DOES ship f32-vsin/f32-vcos RVV kernels, but they call a 2-arg xnn_round_f32 that does
    not exist anywhere in this revision. Authoring that helper would make the EXPERT CCA something we
    wrote, and the expert's instruction mix IS the search target -- so it must stay unharvested and
    SAID SO, not quietly absent."""
    from merlin.mining.wholemodel_proposer import FAMILY_TEACHERS

    for f in ("sin", "cos"):
        t = FAMILY_TEACHERS[f]
        assert t.fixture is None, f"{f} must not claim a fixture it cannot harvest"
        assert "xnn_round_f32" in t.note, f"{f}'s note must name the actual blocker"
        # the ukernel_src is kept so a later XNNPACK bump can flip it by re-running the harvester
        assert t.ukernel_src and "rvv" in t.ukernel_src


def test_the_shim_defines_the_macro_that_blocked_the_rsqrt_harvest():
    """`... params) XNN_OOB_READS {` parses as a declarator followed by garbage without it, and clang
    reports "expected function body after function declarator" -- which is what skipped the RMSNorm
    teacher. Empty expansion is faithful: the attributes change neither the instruction mix nor the
    symbol, which is all the harvest reads."""
    from merlin.common.paths import repo_root

    h = (repo_root() / "merlin/python/merlin/kernels/ceiling_drivers/src/xnnpack/common.h").read_text()
    assert "#define XNN_OOB_READS" in h
    # ...and it must be INSIDE the include guard, not trailing after its #endif
    assert h.index("#define XNN_OOB_READS") < h.rindex("#endif  // MERLIN_CEILING_XNN_COMMON_H")


def test_the_loop_closes_from_harvested_expert_to_the_agent_leaf():
    """The whole point, with no agent involved: a real expert fixture and our real emitted shape form
    a divergence, route to a FORKABLE lever with a machine-readable promise, give a residual the beam
    can escalate on, and terminate at a CODEGEN rung that needs new code."""
    from merlin.common.paths import repo_root
    from merlin.kernels import action_catalog as ac
    from merlin.kernels import cca
    from merlin.kernels.cca_compare import Divergence
    from merlin.kernels.decode import rvv

    fx = repo_root() / "merlin/tests/data/cca_asm/xnnpack_rsqrt_rvv.objdump"
    if not fx.is_file():
        pytest.skip("rsqrt fixture not harvested in this checkout")
    expert = cca.lift_asm(rvv.decode_text(fx.read_text()), op="rsqrt", source="expert")
    ours = cca.lift_asm(
        rvv.decode_text("0000000000000000 <rmsnorm>:\n   0:\t000000ef     \tjal\tra, 0x100 "
                        "<__ieee754_sqrtf>\n"), op="rsqrt", source="ours")

    axis = "compute.activation_vectorization"
    d = Divergence(axis=axis, backend="rvv",
                   ours=ac._facet_value(ours, axis), expert=ac._facet_value(expert, axis))
    assert (d.ours, d.expert) == ("scalar_libm_call", "vectorized_polynomial")

    a = ac.route(d)
    assert a.action_class == "PASS" and a.forkable_now is True
    assert a.intended_facet == {axis: "vectorized_polynomial"}
    # the deterministic GATE: non-empty residual when the fork did not deliver, empty when it did
    assert ac.achieved_residual(a, ours) == [axis]
    assert ac.achieved_residual(a, expert) == []
    # ...and the ladder terminates at the leaf a constrained agent would be handed
    up = ac.route_escalated(d, a.action_class)
    assert up is not None and up.action_class == "CODEGEN" and up.forkable_now is False
