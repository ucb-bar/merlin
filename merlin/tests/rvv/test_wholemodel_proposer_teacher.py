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


# ---------------------------------------------------------------------------------------
# DTYPE-MATCHED experts. A CCA diff across dtypes manufactures divergences that are not
# gaps. OBSERVED when the loop was run with an f32 expert against an int8 model: it
# reported `compute.widening ours=True expert=False` and `compute.epilogue
# ours='requant_narrow' expert='none'`, both of which only restate that one side is int8.
# Same comparand-integrity failure the bundle_id guard catches on the wall axis.
# ---------------------------------------------------------------------------------------

def test_the_expert_fixture_is_selected_by_dtype():
    from merlin.mining.wholemodel_proposer import expert_fixture_for

    assert expert_fixture_for("matmul", "int8") == "xnnpack_qd8_gemm_rvv.objdump"
    assert expert_fixture_for("matmul", "fp16") == "xnnpack_f16_gemm_rvv.objdump"
    assert expert_fixture_for("matmul", "fp32") == "xnnpack_f32_gemm_rvv.objdump"
    # dtype=None keeps the registry default, so every pre-existing caller is unchanged
    assert expert_fixture_for("matmul") == "xnnpack_f32_gemm_rvv.objdump"


def test_an_unmatched_dtype_yields_no_expert_rather_than_another_dtype_s():
    """Failing closed costs a divergence and buys the guarantee that a reported one is real. Borrowing
    a different dtype's expert would report the dtype difference as a compiler gap."""
    from merlin.mining.wholemodel_proposer import expert_family_cca, expert_fixture_for

    assert expert_fixture_for("matmul", "fp8") is None
    assert expert_family_cca("matmul", dtype="fp8") is None
    # a family with no dtype map still resolves through the registry, not to None
    assert expert_fixture_for("gelu", "int8") == "xnnpack_gelu_rvv.objdump"


def test_the_matched_expert_removes_the_spurious_widening_divergence():
    """The concrete payoff, on the real fixtures: an int8 expert widens, so `compute.widening` stops
    being a divergence against an int8 model. An f32 expert does not, which is what produced the noise."""
    from merlin.mining.wholemodel_proposer import expert_family_cca

    e8 = expert_family_cca("matmul", dtype="int8")
    e32 = expert_family_cca("matmul", dtype="fp32")
    if e8 is None or e32 is None:
        pytest.skip("gemm fixtures not harvested in this checkout")
    assert e8.compute.widening is True, "a dynamic-int8 GEMM widens"
    assert e32.compute.widening is False, "an f32 GEMM does not"


# ---------------------------------------------------------------------------------------
# Multi-teacher discovery. A whole model is not one kernel, and no single expert can answer
# every axis: an expert GEMM has no activation, so compute.activation_vectorization is
# UNCOMPARABLE against it and raises nothing no matter how large the model's activation
# cost is. MEASURED on small_llama fp32: scalar exp is 16.48% of real model work
# (__ieee754_expf 11.91% + expf 4.57%) against 2.42% for ALL scalar math on the int8 build
# of the same model -- and the matmul-only run reported that axis as uncomparable.
# ---------------------------------------------------------------------------------------

def _ours_with(**compute):
    from merlin.kernels.cca import CCA, ComputeFacet
    return CCA(op="matmul", backend=["rvv"], compute=ComputeFacet(op="matmul", **compute))


def test_consulting_every_teacher_finds_what_one_teacher_cannot():
    """The activation axis is answerable by the gelu/sigmoid teachers and not by matmul, so it must be
    found when all are consulted and missed when only matmul is."""
    from merlin.kernels import cca_compare
    from merlin.mining.wholemodel_proposer import (divergences_across_teachers, expert_family_cca)

    ours = _ours_with(activation_vectorization="scalar_libm_call")
    if expert_family_cca("gelu") is None or expert_family_cca("matmul", dtype="fp32") is None:
        pytest.skip("fixtures not harvested in this checkout")

    matmul_only = [d.axis for d in cca_compare.compare(
        expert_family_cca("matmul", dtype="fp32"), ours)]
    assert "compute.activation_vectorization" not in matmul_only, (
        "a GEMM expert cannot answer the activation axis — that is the premise")

    divs, taught, unanswered = divergences_across_teachers(ours, dtype="fp32")
    axes = {d.axis for d in divs}
    assert "compute.activation_vectorization" in axes
    assert taught["compute.activation_vectorization"] in ("gelu", "sigmoid", "silu", "softmax")


def test_an_agreeing_axis_counts_as_answered_not_as_blindness():
    """Deriving the answered set from the DIVERGENCE list marks every agreeing axis as "nobody could
    teach it", which over-reports blindness and buries the axes genuinely nobody teaches. Observed: it
    listed compute.accumulator_resident as unanswered while the matmul expert lifts it."""
    from merlin.mining.wholemodel_proposer import divergences_across_teachers, expert_family_cca

    e = expert_family_cca("matmul", dtype="fp32")
    if e is None:
        pytest.skip("gemm fixture not harvested in this checkout")
    # copy an expert value verbatim so the axis AGREES; it must not be reported as unanswered
    ours = _ours_with(accumulator_resident=e.compute.accumulator_resident)
    _divs, _taught, unanswered = divergences_across_teachers(ours, dtype="fp32")
    assert "compute.accumulator_resident" not in unanswered


def test_the_op_label_is_not_treated_as_a_divergence():
    """compute.op is an identity field. Comparing our matmul against the `mul` teacher reports
    ours='matmul' expert='mul' -- a label mismatch, not a gap, and it routes nowhere. Multi-teacher
    comparison is only sound if each teacher opines on PROPERTIES, never on which op it is."""
    from merlin.mining.wholemodel_proposer import divergences_across_teachers

    divs, _t, unanswered = divergences_across_teachers(_ours_with(), dtype="fp32")
    assert "compute.op" not in {d.axis for d in divs}
    assert "compute.op" not in unanswered


def test_an_axis_no_teacher_can_answer_is_still_reported():
    """The whole point of returning the third element: a coverage gap in the TEACHER SET must be
    visible, not dropped, because the loop can only discover what it is shown."""
    from merlin.mining.wholemodel_proposer import divergences_across_teachers

    # `mr_adapts_to_m` is populated by no harvested fixture -- verified by enumerating what the
    # teacher set can answer (19 axes) against the facet fields that exist. So it is a genuine
    # coverage gap in the TEACHER SET, which is exactly what this element exists to surface.
    ours = _ours_with(mr_adapts_to_m=False)
    _divs, _taught, unanswered = divergences_across_teachers(ours, dtype="fp32")
    assert "compute.mr_adapts_to_m" in unanswered, (
        "an axis ours populates that no teacher can answer must be REPORTED, or the loop silently "
        "treats a teacher-coverage gap as 'no gap found'")


def test_every_ranked_lever_names_a_registered_feature():
    """A lever the proposer offers must exist, or the fork dies with 'unknown impr feature' -- and
    `_composes` catches that KeyError and returns False, so the lever is silently NEVER PROPOSED
    rather than failing loudly. Exactly one seam was dead that way before a test caught it."""
    from merlin.llvmlower import impr_features as F
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS
    missing = [n for n, _sr in RANKED_LEVERS if n not in F._REGISTRY]
    assert not missing, f"RANKED_LEVERS names unregistered feature(s): {missing}"


def test_ranked_levers_have_no_duplicates():
    """A duplicate would spend a generation's width twice on the same idea."""
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS
    names = [n for n, _ in RANKED_LEVERS]
    assert len(names) == len(set(names)), f"duplicate levers: {sorted(set(n for n in names if names.count(n) > 1))}"


def test_the_locality_lever_is_searched_not_defaulted():
    """promote_buffers_to_stack is 1.34x on small_llama int8, 1.04x on the same model in fp32, and
    ~1.01x SLOWER on spectformer int8. A blanket default would have shipped a regression; it belongs
    to the search, which measures per model."""
    from merlin.llvmlower import impr_features as F
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS
    assert "promote_buffers_to_stack" in [n for n, _ in RANKED_LEVERS]
    assert F._REGISTRY["promote_buffers_to_stack"].edit_pipeline is not None


def test_refinements_cost_the_seed_generation_no_width():
    """A magnitude is only meaningful once the lever it belongs to is enabled.

    Putting every (lever, magnitude) pair in RANKED_LEVERS would multiply generation 1's width by the
    ladder length -- and the run that motivated these knobs already deferred 11 of its 12 proposals
    with `reason: over_width`. So refinements must be EMPTY at the seed and appear only once the
    parent already carries the lever they retune.
    """
    from merlin.mining.wholemodel_proposer import refinement_forks

    assert refinement_forks([]) == []
    assert refinement_forks(["erase_self_copy"]) == []

    mr = refinement_forks(["perop_register_block"])
    assert mr, "per-op blocking in the parent must open the MR-cap axis"
    assert all(f.targets.endswith(":mr_cap") for f in mr)

    stack = refinement_forks(["promote_buffers_to_stack"])
    assert stack, "stack promotion in the parent must open the cap axis"
    assert all(f.targets.endswith(":cap") for f in stack)


def test_a_refinement_replaces_the_magnitude_it_retunes_rather_than_stacking_it():
    """Two caps for one lever describe two different builds; a fork must carry exactly one."""
    from merlin.llvmlower import impr_features as I
    from merlin.mining.wholemodel_proposer import refinement_forks

    for fp in refinement_forks(["perop_register_block", "erase_self_copy"]):
        feats = fp.overrides["compiler_features"]
        assert I.PEROP_BLOCK_NAME not in feats, "the unpinned sentinel must be replaced, not kept"
        assert sum(I.parse_perop_mr_sentinel(f) is not None for f in feats) == 1
        assert "erase_self_copy" in feats, "unrelated parent features must survive"

    for fp in refinement_forks(["promote_buffers_to_stack"]):
        feats = fp.overrides["compiler_features"]
        assert sum(f.startswith(I.PROMOTE_STACK_NAME) for f in feats) == 1


def test_the_named_op_enabler_is_a_base_lever_not_a_refinement():
    """The named-op register block cannot fire until the contraction keeps its named form: the int8
    quant pass rewrites every linalg.matmul into a linalg.generic, and a schedule matching on the op
    NAME then finds an empty handle and does nothing.

    But the enabler is a LEVER, not a magnitude, so it belongs in RANKED_LEVERS -- refinements must
    cost the seed generation no width, and offering it there would widen generation 1 for every run.
    """
    from merlin.llvmlower.impr_features import NAMED_INT8_CONTRACTION_NAME as ENABLER
    from merlin.mining.wholemodel_proposer import RANKED_LEVERS, refinement_forks

    assert ENABLER in [name for name, _ in RANKED_LEVERS]
    # it is NOT a full-schedule replacement -- it changes which op the quant pass emits, nothing else
    assert dict(RANKED_LEVERS)[ENABLER] is False
    # the seed generation stays free, and no tile is offered while the lever cannot fire
    assert refinement_forks([]) == []
    assert not [f for f in refinement_forks(["perop_register_block"]) if "tile" in f.targets]


def test_tiles_are_offered_once_the_enabler_is_on_and_each_is_distinct():
    from merlin.llvmlower.impr_features import MRPAD_INT8_TILES, NAMED_INT8_CONTRACTION_NAME
    from merlin.mining.wholemodel_proposer import refinement_forks

    forks = refinement_forks([NAMED_INT8_CONTRACTION_NAME, "promote_buffers_to_stack"])
    tiles = [f for f in forks if "tile" in f.targets]
    assert len(tiles) == len(MRPAD_INT8_TILES)
    # each fork enables exactly ONE tile -- two full-schedule replacements cannot compose, and the
    # composition guard rejects that pairing outright
    for f in tiles:
        feats = set(f.overrides["compiler_features"])
        assert len(feats & set(MRPAD_INT8_TILES)) == 1
        assert NAMED_INT8_CONTRACTION_NAME in feats
    # and the proposals are distinct
    assert len({tuple(sorted(f.overrides["compiler_features"])) for f in tiles}) == len(tiles)
