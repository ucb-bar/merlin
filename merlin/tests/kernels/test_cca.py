"""R3: Common Compute Abstraction lift + cross-level agreement (target-agnostic)."""
from __future__ import annotations

from pathlib import Path

from merlin.common.paths import merlin_dir

import pytest

from merlin.kernels import cca
from merlin.kernels.decode import objdump, rvv

_ASM_DIR = merlin_dir() / "tests" / "data" / "cca_asm"

# matmul-ish stream: mul_add (vfmul+vfadd), no vfmacc, fixed vsetivli, e32m2.
_MUL_ADD = """\

Disassembly of section .text:
0 <k>:
       0: 00     \tvsetivli\tzero, 0x8, e32, m2, ta, ma
       4: 00     \tvle32.v\tv8, (a0)
       8: 00     \tvfmul.vv\tv8, v8, v9
       c: 00     \tvfadd.vv\tv10, v10, v8
"""
# fused stream: vfmacc present, e32m4.
_FUSED = """\

Disassembly of section .text:
0 <k>:
       0: 00     \tvsetivli\tzero, 0x8, e32, m4, ta, ma
       4: 00     \tvle32.v\tv8, (a0)
       8: 00     \tvfmacc.vv\tv12, v8, v9
"""


# scalar activation: a per-element libm call loop (jal to <expf>), scalar float, no vfmacc.
_SCALAR_LIBM_ACT = """\

Disassembly of section .text:
0 <gelu>:
       0: 00     \tflw\tfa0, 0x0(a0)
       4: 00     \tjal\tra, 0x100 <expf>
       8: 00     \tfmul.s\tfa0, fa0, fa1
       c: 00     \tfsw\tfa0, 0x0(a1)
"""
# vectorized activation: an inline minimax polynomial (vfmacc chain), no libm call.
_VECTOR_POLY_ACT = """\

Disassembly of section .text:
0 <gelu>:
       0: 00     \tvsetivli\tzero, 0x8, e32, m2, ta, ma
       4: 00     \tvle32.v\tv8, (a0)
       8: 00     \tvfmacc.vv\tv12, v8, v9
       c: 00     \tvfmacc.vv\tv12, v12, v10
"""


def _cca(monkeypatch, snippet, source):
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: snippet)
    return cca.lift_asm(rvv.decode("x.o"), op="matmul", source=source)


def _cca_op(monkeypatch, snippet, op):
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: snippet)
    return cca.lift_asm(rvv.decode("x.o"), op=op, source="act")


def test_activation_vectorization_scalar_libm(monkeypatch):
    c = _cca_op(monkeypatch, _SCALAR_LIBM_ACT, "gelu")
    assert c.compute.activation_vectorization == "scalar_libm_call"   # calls <expf>


def test_activation_vectorization_vectorized_poly(monkeypatch):
    c = _cca_op(monkeypatch, _VECTOR_POLY_ACT, "gelu")
    assert c.compute.activation_vectorization == "vectorized_polynomial"  # vfmacc poly, no libm call


def test_activation_vectorization_none_for_matmul(monkeypatch):
    # a plain matmul (not a transcendental activation) must NOT be classified as an activation.
    c = _cca(monkeypatch, _FUSED, "expert")
    assert c.compute.activation_vectorization is None


def test_lift_asm_mul_add(monkeypatch):
    c = _cca(monkeypatch, _MUL_ADD, "ours")
    assert c.backend == ["rvv"]
    assert c.compute.contraction_form == "mul_add"   # no vfmacc -> mul_add
    assert c.vector.sew == 32 and c.vector.lmul == 2.0
    assert c.vector.vl_strategy == "vsetivli_fixed"


def test_lift_asm_fused(monkeypatch):
    c = _cca(monkeypatch, _FUSED, "expert")
    assert c.compute.contraction_form == "fused_fma"
    assert c.vector.lmul == 4.0


def test_agreement_flags_divergences(monkeypatch):
    ours = _cca(monkeypatch, _MUL_ADD, "ours")
    expert = _cca(monkeypatch, _FUSED, "expert")
    rep = cca.cca_agree(expert, ours)
    assert not rep.agree
    axes = {d.split(":")[0] for d in rep.disagreements}
    assert "compute.contraction_form" in axes   # the vfmacc gap
    assert "vector.lmul" in axes                 # 4 vs 2
    # self-agreement is the validity baseline
    assert cca.cca_agree(ours, ours).agree


def test_composite_backend_supported():
    # a heterogeneous region (NPU+RVV) is just a backend list — not a special case
    c = cca.CCA(op="attention", backend=["npu", "rvv"])
    assert c.backend == ["npu", "rvv"]
    # facets are populated only when relevant. (`dataflow` was retired: its fields were data MOVEMENT,
    # which every target has, so they folded into memory/dispatch rather than staying in a facet
    # scoped to one kind of silicon.)
    assert c.spatial is None and c.simt is None and c.dispatch is None and c.layout is None


# ---- accumulator-residency / register-block / VL-NR: the abstraction reads the expert-win
# properties faithfully off REAL disassembly (no regex; via decode.rvv). These pin the ABSTRACTION
# (does the CCA *see* the gap?), not a memorized shape — the asm fixtures are whole-kernel
# disassembly built from the expert ceiling drivers + our own baseline / impr-feature codegen
# (provenance in data/cca_asm/AGENT.md).

def _lift_fixture(monkeypatch, name: str) -> cca.CCA:
    text = (_ASM_DIR / name).read_text()
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: text)
    return cca.lift_asm(rvv.decode(name), op="matmul", source=name)


def test_lift_reads_accumulator_resident_on_experts(monkeypatch):
    # Both expert GEMMs keep the accumulator in vector registers across the whole K loop (no in-loop
    # spill) and commit C once after — the abstraction must read accumulator_resident=True on each.
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump"):
        c = _lift_fixture(monkeypatch, fx)
        assert c.compute.contraction_form == "fused_fma", fx
        assert c.compute.accumulator_resident is True, fx
        # a register block is recovered (MR distinct accumulators, NR lmul-scaled lanes)
        assert c.compute.register_block is not None, fx
        mr, nr = c.compute.register_block
        assert isinstance(mr, int) and mr >= 1, fx
        assert nr[0] == "vsetvlmax", fx


def test_lift_reads_accumulator_dtype(monkeypatch):
    # The accumulate width is captured (ISA-grounded): the f32 GEMM experts accumulate in f32. This is
    # the dtype-datapath axis the compiler exposes via the dtype_strategy knob.
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump"):
        c = _lift_fixture(monkeypatch, fx)
        assert c.compute.accumulator_dtype == "f32", fx


def test_lift_reads_memory_facet(monkeypatch):
    # the CCA now CAPTURES the data-movement/packing dimension (was blind to it) — the expert GEMMs
    # fetch operands as packed unit-stride panels (the #1 expert lever, lifted from decode.memory).
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump"):
        c = _lift_fixture(monkeypatch, fx)
        assert c.memory is not None, fx
        assert c.memory.access_pattern == "unit_stride", fx


def test_memory_divergence_surfaces_packed_vs_strided():
    # the whole point: a kernel whose COMPUTE matches the expert but fetches operands STRIDED now
    # surfaces a memory.access_pattern divergence — the dimension that used to be invisible (the
    # '72% slower with no divergences' case). This is what the CCA had to capture.
    expert = cca.CCA(op="matmul", backend=["rvv"],
                     compute=cca.ComputeFacet(op="matmul", contraction_form="fused_fma"),
                     memory=cca.MemoryFacet(access_pattern="unit_stride"))
    ours = cca.CCA(op="matmul", backend=["rvv"],
                   compute=cca.ComputeFacet(op="matmul", contraction_form="fused_fma"),
                   memory=cca.MemoryFacet(access_pattern="strided"))
    from merlin.kernels import cca_compare
    axes = {d.axis for d in cca_compare.compare(expert, ours)}
    assert axes == {"memory.access_pattern"}      # compute matches; ONLY the memory dimension differs


def test_decode_text_matches_object_path(monkeypatch):
    # a CCA lifted from objdump TEXT (rvv.decode_text — what the beam has in objdump.txt) equals the
    # object-file path (rvv.decode). Enables lifting a fork's CCA with no toolchain.
    text = (_ASM_DIR / "openblas_sgemm_rvv.objdump").read_text()
    from_text = cca.lift_asm(rvv.decode_text(text), op="matmul", source="openblas")
    monkeypatch.setattr(objdump, "disassemble_text", lambda *a, **k: text)
    from_obj = cca.lift_asm(rvv.decode("x.o"), op="matmul", source="openblas")
    assert from_text.compute.contraction_form == from_obj.compute.contraction_form == "fused_fma"
    assert from_text.compute.accumulator_resident == from_obj.compute.accumulator_resident
    assert from_text.vector.tail == from_obj.vector.tail


def _matmul_record(dtype: str):
    from merlin.frontends.linalg_mlir import MatmulRecord
    return MatmulRecord(kind="linalg.matmul", m=64, k=64, n=64, lhs_shape=(64, 64), rhs_shape=(64, 64),
                        dtype=dtype, weight_arg_index=1, weight_name="w", prov={"prov.op": "matmul"})


def test_lift_graph_partial_from_dtype():
    # the flat-graph analyzer derives only op + dtype datapath facets (partial, by design)
    g = cca.lift_graph(_matmul_record("f32"))
    assert g.compute.op == "matmul" and g.compute.accumulator_dtype == "f32"
    assert g.compute.widening is None                          # f32 -> not a widening MAC
    g8 = cca.lift_graph(_matmul_record("i8"))
    assert g8.compute.accumulator_dtype == "i32" and g8.compute.widening is True


def test_asm_and_graph_analyzers_agree(monkeypatch):
    # the two DETERMINISTIC analyzers (asm decode + flat graph) must agree on the shared populated
    # facets — cca_agree is the validity gate that quarantines a bad reconstruction on either side.
    asm = _lift_fixture(monkeypatch, "openblas_sgemm_rvv.objdump")   # f32 GEMM -> acc f32
    graph = cca.lift_graph(_matmul_record("f32"))
    rep = cca.cca_agree(asm, graph)
    assert rep.agree, rep.disagreements
    assert "compute.accumulator_dtype" in rep.compared_fields    # the shared facet actually compared


def test_lift_reads_vector_tail(monkeypatch):
    # The tail policy (ta|tu) is captured from the decoded vsetvl vtype state (not guessed). The GEMM
    # fixtures run tail-agnostic (ta). Populating it feeds the eventual tail route + cca_agree.
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump"):
        c = _lift_fixture(monkeypatch, fx)
        assert c.vector.tail in ("ta", "tu"), fx


def test_lift_reads_xnnpack_nr_tracks_vsetvlmax(monkeypatch):
    # XNNPACK 1x4v is the VL-adaptive expert: a polymorphic vsetvli VL-loop, so NR tracks vsetvlmax.
    c = _lift_fixture(monkeypatch, "xnnpack_f32_gemm_rvv.objdump")
    assert c.vector.vl_strategy == "vsetvl_loop"
    assert c.compute.nr_is_vsetvlmax is True
    assert c.compute.register_block[0] == 1            # MR=1 (one accumulator), 1x4v


def test_lift_reads_ours_baseline_not_resident(monkeypatch):
    # Our FROZEN baseline lowering does not even form a fused MAC (vfmul+vfadd) — the deepest gap.
    c = _lift_fixture(monkeypatch, "ours_baseline_matmul.objdump")
    assert c.compute.contraction_form == "mul_add"
    assert c.compute.accumulator_resident is not True   # None/False — never the expert's True


def test_lift_reads_ours_accum_feature_still_not_resident(monkeypatch):
    # The accumulator_resident_microkernel feature DOES form vfmacc, but the emitted asm still spills
    # the accumulator through the stack inside the K loop (whole-register vsNr/vlNre). The abstraction
    # must honestly read accumulator_resident=False — i.e. it SEES the residual gap to the experts.
    c = _lift_fixture(monkeypatch, "ours_accum_resident_matmul.objdump")
    assert c.compute.contraction_form == "fused_fma"
    assert c.compute.accumulator_resident is False


def test_accumulator_residency_divergence_expert_vs_ours(monkeypatch):
    # End-to-end of the abstraction: expert(resident=True) vs ours(resident=False) DISAGREE on the
    # shared compute.accumulator_resident axis — the gap is now a typed, comparable CCA field.
    expert = _lift_fixture(monkeypatch, "openblas_sgemm_rvv.objdump")
    ours = _lift_fixture(monkeypatch, "ours_accum_resident_matmul.objdump")
    rep = cca.cca_agree(expert, ours)
    assert "compute.accumulator_resident" in rep.compared_fields
    assert any("accumulator_resident" in d for d in rep.disagreements)


def test_accumulator_resident_is_target_agnostic_compute_field():
    # Promoted onto the SHARED ComputeFacet (not just SpatialFacet) — every backend answers the same
    # "is the accumulator resident across the reduction" question on the same axis.
    assert "accumulator_resident" in cca.ComputeFacet().__dataclass_fields__
    # the Gemmini/spatial view of the SAME concept still exists (compared per-facet)
    assert "accumulator_resident" in cca.SpatialFacet().__dataclass_fields__


@pytest.mark.skipif(not _ASM_DIR.is_dir(), reason="cca asm fixtures absent")
def test_fixtures_present():
    for fx in ("openblas_sgemm_rvv.objdump", "xnnpack_f32_gemm_rvv.objdump",
               "ours_baseline_matmul.objdump", "ours_accum_resident_matmul.objdump"):
        assert (_ASM_DIR / fx).is_file()


# ---------------------------------------------------------------------------------------
# Scalar-math call classification. The detector's symbol list was the GELU/softmax family
# only -- ("exp","erf","tanh","sinh","cosh","log","pow") -- matched as a SUBSTRING of the
# call target. That failed in both directions: it missed every decorated spelling a real
# libc emits (__ieee754_sqrt, __kernel_sinf) and it would false-positive on any symbol
# that merely contains a stem (<merlin_single_step> contains "sin", <log_write> contains
# "log"). Consequence, measured on small_llama int8: 16.63% of an INT8 model's binary is
# scalar FLOAT and 36 model symbols are entirely scalar, yet _infer_activation_vectorization
# returned None -- so no divergence was raised, action_catalog was never consulted, and the
# beam never proposed the lever that was forkable the whole time.
# ---------------------------------------------------------------------------------------

def test_the_decorated_libc_spellings_are_classified():
    """These are the exact symbols small_llama int8 calls. Every one of them used to return None."""
    from merlin.kernels.cca import math_call_kind

    assert math_call_kind("__ieee754_sqrt") == "algebraic"          # RMSNorm normaliser
    assert math_call_kind("__ieee754_sqrtf") == "algebraic"
    assert math_call_kind("__kernel_sinf") == "transcendental"      # RoPE
    assert math_call_kind("__kernel_cosf") == "transcendental"
    # glibc's argument reduction for sin/cos is its OWN symbol, so a model can pay for it
    # without any sin/cos call being visible at the call site.
    assert math_call_kind("__kernel_rem_pio2f") == "transcendental"
    assert math_call_kind("__ieee754_rem_pio2f") == "transcendental"
    assert math_call_kind("__extendbfsf2") == "softfloat"           # bf16 -> f32 soft conversion


def test_a_stem_that_merely_appears_inside_a_symbol_is_not_a_math_call():
    """The retired substring test would have called all of these math."""
    from merlin.kernels.cca import math_call_kind

    for sym in ("merlin_single_step", "log_write", "cosine_table", "single", "forward",
                "memcpy", "__errno", "expand_shape", "powerdown_hook"):
        assert math_call_kind(sym) is None, sym


def test_a_stem_that_itself_ends_in_a_suffix_letter_survives():
    """`erf` ends in "f" and `atan` ends in "n"; stripping suffixes to a fixed point turns `erff` into
    `er` and `atanf` into `ata`. Candidates are tested longest-first for exactly this reason."""
    from merlin.kernels.cca import math_call_kind

    assert math_call_kind("erff") == "transcendental"
    assert math_call_kind("erf") == "transcendental"
    assert math_call_kind("atanf") == "transcendental"
    assert math_call_kind("sqrtf_finite") == "algebraic"
    assert math_call_kind("log2f") == "transcendental"


def _stream_calling(*symbols):
    """A decoded stream whose only interesting content is a call to each symbol."""
    from merlin.kernels.decode import rvv

    lines = ["0000000000000000 <forward>:"]
    for i, sym in enumerate(symbols):
        lines.append(f"   {i * 4:x}:\t000000ef     \tjal\tra, 0x100 <{sym}>")
    return rvv.decode_text("\n".join(lines) + "\n")


def test_the_divergence_is_now_observable_for_rope_and_rmsnorm():
    """The whole point: the CCA must LIFT the loss, or nothing downstream can route it."""
    from merlin.kernels import cca

    s = _stream_calling("__kernel_sinf", "__kernel_cosf", "__ieee754_sqrt")
    found = cca.scalar_math_calls(s)
    assert set(found) == {"__kernel_sinf", "__kernel_cosf", "__ieee754_sqrt"}
    assert set(found.values()) == {"transcendental", "algebraic"}
    assert cca._has_transcendental_libm_call(s) is True
    # ...and it fires even when the op label is unknown, since the CALL is the evidence
    assert cca._infer_activation_vectorization(s, None) == "scalar_libm_call"
    assert cca._infer_activation_vectorization(s, "rmsnorm") == "scalar_libm_call"


def test_a_plain_kernel_is_still_never_classified_as_an_activation():
    """The None return is what stops a matmul being mislabelled; widening the symbol set must not
    weaken it."""
    from merlin.kernels import cca

    s = _stream_calling("memcpy", "merlin_single_step")
    assert cca.scalar_math_calls(s) == {}
    assert cca._infer_activation_vectorization(s, "matmul") is None


def test_the_ladder_escalates_past_the_pass_that_cannot_cover_these_ops():
    """The PASS emits exp/erf/tanh polynomials only. Before this rung, a model paying for rsqrt or
    sin/cos routed to that pass, measured no gain, and the ladder ENDED -- with nothing recording that
    the lever was missing rather than useless. Escalation must reach a CODEGEN work-item."""
    from merlin.kernels import action_catalog as ac
    from merlin.kernels.cca_compare import Divergence

    d = Divergence(axis="compute.activation_vectorization", backend="rvv",
                   ours="scalar_libm_call", expert="vectorized_polynomial")
    cheap = ac.route(d)
    assert cheap.action_class == "PASS" and cheap.forkable_now is True

    up = ac.route_escalated(d, "PASS")
    assert up is not None, "the ladder must not be exhausted at PASS"
    assert up.action_class == "CODEGEN" and up.forkable_now is False
    assert "act_poly" in up.target_seam
    for token in ("rsqrt", "sin/cos", "range reduction"):
        assert token in up.change, token
    assert ac.route_escalated(d, "CODEGEN") is None, "and it must terminate, not loop"


def test_the_c23_floatN_spellings_are_classified():
    """The REAL defect behind a false "promise achieved". This glibc exposes its math routines under
    the C23 type-generic names, so a binary that plainly calls sin/cos/sqrt reported ZERO scalar math
    calls, `_infer_activation_vectorization` returned 'vectorized_polynomial' for an UNFIXED binary,
    and `achieved_residual` came back empty -- the gate would have credited a change that never
    happened. Measured on the K1 Linux build of small_llama int8, whose most-called math targets are
    roundevenf (175 calls), cosf32 (128) and sinf32 (128).
    """
    from merlin.kernels.cca import math_call_kind

    assert math_call_kind("sinf32") == "transcendental"
    assert math_call_kind("cosf32") == "transcendental"
    assert math_call_kind("sqrtf32") == "algebraic"
    assert math_call_kind("sqrtf64") == "algebraic"
    assert math_call_kind("expf32x") == "transcendental"
    # the _FloatN suffix must be tried BEFORE the bare `f`, or sinf32 never reduces to sin
    assert math_call_kind("sinf") == "transcendental"


def test_the_trig_range_reduction_helper_is_counted():
    """roundevenf is the 4th most-called symbol in the measured binary. Omitting it under-reports the
    scalar-math share of exactly the path (RoPE) this axis exists to find."""
    from merlin.kernels.cca import math_call_kind

    for sym in ("roundevenf", "roundeven", "rintf", "truncf", "floorf", "fmodf"):
        assert math_call_kind(sym) == "algebraic", sym


def test_a_tail_call_is_a_call_but_an_internal_jump_is_not():
    """glibc reaches routines by tail call (`j 0x... <sym>`), not only `jal`. But an intra-function jump
    renders as `<sym+0x14>`, and treating that as a call would count a routine as calling itself.
    objdump's own rendering is the discriminator -- exact symbol vs symbol+offset -- so no heuristic
    is needed."""
    from merlin.kernels import cca
    from merlin.kernels.decode import rvv

    tail = rvv.decode_text("0000000000000000 <f>:\n   0:\ta055     \tj\t0x100 <sinf32>\n")
    assert cca.scalar_math_calls(tail) == {"sinf32": "transcendental"}

    internal = rvv.decode_text("0000000000000000 <sinf32>:\n   0:\ta055     \tj\t0x14 <sinf32+0x14>\n")
    assert cca.scalar_math_calls(internal) == {}, "an internal jump is not a call to the routine"


def test_libc_internals_that_merely_look_mathy_are_still_not_math():
    """These are real symbols from the measured binary's call-target set. The substring detector this
    replaced would have classified several of them."""
    from merlin.kernels.cca import math_call_kind

    for sym in ("expand_dynamic_string_token", "_nl_expand_alias", "_nl_explode_name",
                "sysinfo", "__gettext_free_exp", "_IO_vtable_check", "__libc_assert_fail"):
        assert math_call_kind(sym) is None, sym


# ---------------------------------------------------------------------------------------
# Uncomparable axes. `compare` emits no divergence when either side is None -- correct, it
# cannot claim a gap it cannot see. But silence is not "no gap", and the loop can only
# discover what it is shown. This module already knew the hazard and patched ONE axis (the
# MR-aware special case, added because an unblocked kernel lifts register_block=None so
# "expert blocks, ours doesn't" never surfaced); every other axis kept the blindness.
# ---------------------------------------------------------------------------------------

def test_an_axis_only_one_side_can_answer_is_reported_not_silently_skipped():
    from merlin.kernels import cca_compare as cc
    from merlin.kernels.cca import CCA, ComputeFacet, MemoryFacet

    full = CCA(op="matmul", backend=["rvv"],
               compute=ComputeFacet(op="matmul", register_block=(4, 16), accumulator_resident=True),
               memory=MemoryFacet(access_pattern="unit_stride"))
    partial = CCA(op="matmul", backend=["rvv"],
                  compute=ComputeFacet(op="matmul"))          # no block, no residency, no memory facet

    axes = dict(cc.uncomparable_axes(partial, full))
    assert axes.get("compute.register_block") == "expert"
    assert axes.get("compute.accumulator_resident") == "expert"
    assert axes.get("memory") == "expert", "a whole missing facet must be named once, not per field"
    # and compare() still refuses to invent a divergence for them
    for d in cc.compare(partial, full):
        assert d.expert is not None and d.ours is not None


def test_both_sides_absent_is_not_an_uncomparable_axis():
    """Neither side lifting an axis is a coverage fact about the LIFTER, not an asymmetry between the
    two CCAs -- reporting it would bury the asymmetries that matter."""
    from merlin.kernels import cca_compare as cc
    from merlin.kernels.cca import CCA, ComputeFacet

    a = CCA(op="matmul", backend=["rvv"], compute=ComputeFacet(op="matmul"))
    b = CCA(op="matmul", backend=["rvv"], compute=ComputeFacet(op="matmul"))
    assert not [x for x in cc.uncomparable_axes(a, b) if x[0].startswith("memory")]


def test_the_measured_qd8_expert_blindness_is_surfaced():
    """The concrete case: the dtype-matched int8 GEMM fixture cannot answer register blocking or
    anything in memory, so choosing it (correctly, to avoid a cross-dtype diff) must not silently
    retire those axes."""
    from merlin.kernels import cca_compare as cc
    from merlin.mining.wholemodel_proposer import expert_family_cca

    e8, e32 = expert_family_cca("matmul", dtype="int8"), expert_family_cca("matmul", dtype="fp32")
    if e8 is None or e32 is None:
        pytest.skip("gemm fixtures not harvested in this checkout")
    axes = dict(cc.uncomparable_axes(e8, e32))
    assert axes.get("compute.register_block") == "expert"
    assert axes.get("memory") == "expert"
