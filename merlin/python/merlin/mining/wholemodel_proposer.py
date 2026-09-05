"""Whole-model beam proposer — a HYBRID of a PER-OP TEACHER and census-ranked hardcodes.

The default CCA proposer (``fork_from_action.propose_forks_from_cca``) diffs the parent's emitted
kernel CCA against a single EXPERT KERNEL objdump. That works for the facets a whole-model asm lift
CAN see against a GEMM expert (contraction form, register block, reduction form, activation
vectorization, envelope), but it is structurally blind to the levers that are GRAPH properties, not
inner-loop properties — a materialized ``linalg.transpose`` (38% of byte-traffic, 57% of openvla
time) and a per-matmul MR clamp are decisions no kernel-vs-kernel facet diff can surface, because the
CCA has no ``layout`` facet and MR-under-M-tail is a shape decision.

So this proposer is a UNION of two engines:

1. **Per-op TEACHER** (``route_divergence_forks`` + the per-family section machinery): route the CCA
   divergences the beam already lifts (parent-vs-expert) into forks via ``action_catalog.route`` /
   ``fork_from_action.propose_forks_from_cca`` — the divergence->route->fork engine that already
   exists. The richer form (``make_per_op_teacher_proposer``) pairs a per-FAMILY expert CCA (an
   XNNPACK ukernel fixture, see ``FAMILY_TEACHERS`` and ``build_tools/scripts/harvest_xnnpack_fixtures.py``)
   against OUR per-family section CCA (``ours_section_cca``: slice the model to that family's regions,
   build, lift the emitted asm). Families with no XNNPACK primitive (``sdpa``/``layer_norm``/gather)
   get an HONEST no-teacher record — never a faked divergence.

2. **Census-ranked hardcodes** (``census_hardcode_forks`` / ``RANKED_LEVERS``): the whole-model levers
   whose route exists but whose facet field the CCA does not carry, so ``compare`` cannot emit them —
   ``layout.transpose_materialized`` (``fuse_transpose_b``) and the per-matmul MR block
   (``accumulator_resident_wholemodel_vf_mrpad``), plus the additive envelope/reduction/activation
   passes as a fallback when the teacher is idle. Ranked by measured byte-traffic
   (``out/artifacts/ceiling/model_op_census.json``).

Contract: ``(divergences, knobs) -> [ForkProposal]`` — a drop-in for ``beam.run_beam(proposer=...)``.
``propose_wholemodel_levers`` is the in-contract default (consumes the beam's own divergences UNION
the census hardcodes); ``make_per_op_teacher_proposer`` binds the richer per-family teacher and
returns the same 2-arg closure. Each feature proposal MERGES one new feature into the parent's
``compiler_features`` (depth-N accumulates a stack), dropping any that cannot compose (two
full-schedule-replacement features clobber).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..kernels.knobs import ForkProposal
# `_composes` resolves every lever name through `impr_features`, and an UNREGISTERED name is
# swallowed there as "does not compose" -- so a lever registered lazily elsewhere would be
# silently never proposed rather than rejected. Register it here, where the list lives.
from ..llvmlower.transpose_maps import ensure_registered as _register_fold_weight_transpose
from ..llvmlower.weight_prepack import ensure_registered as _register_prepack_weight_layout
# `cse_through_provenance` is registered by `llvmlower.lower`, which this module never imports -- so
# `_composes` raised KeyError and returned False for EVERY parent stack carrying it. That is not a
# lever of its own being skipped: it is in the config the search currently calls best, so while it was
# missing here NO lever could be proposed on top of the winner, and the beam silently had nothing to
# build on. Exactly the failure the comment above describes, one import short of being prevented.
from ..llvmlower.concat_dps import ensure_registered as _register_concat_dps
from ..llvmlower.epilogue_fusion import ensure_registered as _register_fuse_epilogue_loops
from ..llvmlower.prov_cse import ensure_registered as _register_cse_through_provenance
from ..llvmlower.perop_blocks import ensure_registered as _register_conv_register_block

_register_fold_weight_transpose()
_register_prepack_weight_layout()
_register_cse_through_provenance()
# Registered here even though neither is in RANKED_LEVERS. Registration and RANKING are different
# decisions: `_composes` catches the KeyError for an unregistered name and returns False, so an
# unregistered feature is not "declined", it is INVISIBLE -- it can never be proposed, and a later
# improvement to it stays invisible too. Ranking stays the owning lever's call.
_register_concat_dps()
_register_fuse_epilogue_loops()
# The direct-conv arm. Registered, deliberately NOT ranked: it is inert on every model whose convs
# model2MLIR expanded into im2col (which is every model captured at the default element budget), so
# ranking it would spend beam width on a fork that cannot differ from its parent. It becomes a
# candidate only alongside a capture whose convs took the direct form, and it has NO hardware
# measurement yet -- see llvmlower.perop_blocks.CONV_ARM_FEATURE.
_register_conv_register_block()

# Whole-model HARDCODE levers, most-impactful first by measured byte-traffic / e2e attribution. Each
# entry is (feature_name, is_full_schedule_replacement). These are the levers a per-facet CCA diff
# CANNOT emit — a GRAPH-layout decision (transpose) or a shape decision (per-matmul MR) with no CCA
# facet field — plus the additive passes as a teacher-idle fallback. The teacher (engine 1) supplies
# the rest from real divergences.
RANKED_LEVERS: list[tuple[str, bool]] = [
    # FIRST, because it is the largest whole-model effect measured on this repo's silicon oracle, and
    # because it is the only lever here that changes what the BUNDLE stores rather than what the
    # compiler emits. Interleaved in one K1 session, alternating bundles, three rounds each, both arms
    # gating ok=True: 3,548,286/3,574,361/3,561,602 ns stock vs 2,125,388/2,086,712/2,127,671 ns
    # prepacked -- 1.70x against a 2.6% noise band. It is NOT the transposes' own byte traffic (0.4 MiB
    # per inference on this model, inside the band); it is that 15 `linalg.transpose` ops and their
    # `tensor.empty` destinations stop being materialized at all, taking their buffers, allocs and
    # copies with them. Bit-exact by construction and asserted per weight as `stored.T`, so a fork
    # carrying it grades against the same goldens. Refuses (rather than silently building stock) on a
    # bundle whose layout cannot be pre-applied soundly.
    ("prepack_weight_layout", False),
    # NOTE: `named_int8_contraction` is deliberately NOT listed here. It is an ENABLER -- inert on
    # its own by construction, since keeping the contraction named changes no emitted code until a
    # schedule acts on the named form -- and `run_beam` EXCLUDES inert forks from the survivor set.
    # Listed as a base lever it was built 10 times in one search, was inert every time, never
    # survived to become a parent, and so the tile refinements that need it as a parent were never
    # proposed: 0 forks in 136 carried the enabler together with a tile. It is offered instead as
    # part of a COMBINED proposal in `refinement_forks`, so it is only ever measured with something
    # that can act on it.
    # KEEP THE CONTRACTION NAMED, first because it is an ENABLER: without it a whole family of levers
    # below cannot fire at all. The int8 quant pass rewrites every `linalg.matmul` into a
    # `linalg.generic` (measured: 15 -> 0 on small_llama int8), and a transform schedule matching on
    # the op NAME then gets an empty handle from `transform.structured.match`, which makes every op
    # downstream of it a no-op. The lever still builds, gates clean and reports as applied. An
    # 87-fork search over exactly those levers emitted 20 distinct binaries and 34 inert nodes.
    # It is not itself an optimization -- MEASURED on the K1, enabling it alone leaves the wall and
    # the output bit-identical (4,140,253 vs 4,131,982 ns; cos 0.9999079, rel 0.0147963915 either
    # way) -- so it earns its rank purely by what it makes reachable.
    # PER-CONTRACTION register blocking, first because it SUPERSEDES the two hand-picked class-wide
    # clamps below it rather than competing with them. `WHOLEMODEL_VF_NR_BMM = 8` and `MR_mm = 1` are
    # single numbers a human chose for a whole op CLASS, and a class is not shape-homogeneous: one
    # degenerate extent in it forces every member off the vector path. This lever derives the widest
    # block legal for each contraction on its own extents and its own element width instead. It could
    # not be listed here before: the name was unregistered, and `_feature_fork` -> `_composes` catches
    # the resulting KeyError and returns False, so the lever would have been silently never proposed
    # rather than rejected. MEASURED on the live K1 (small_llama int8, whole model, interleaved
    # same-session arms, cos identical on every arm): no blocking 290,015,352 cyc -> per-op at the old
    # MR cap 11,338,272 -> per-op at MR=4 8,856,276. The first step is the class-wide clamp being gone;
    # the second is the A-reuse register block.
    ("perop_register_block", True),
    # The N-fill knob, right after the blocking it implies. Ranked here rather than defaulted because
    # its SIGN is model-dependent: MEASURED on the K1, 1.160x FASTER on spectformer int8 and 1.196x
    # SLOWER on small_llama int8, both far outside the 2.6% band. The i32 accumulator is what sets
    # LMUL, so a wider N tile can push it from m4 to m8 and spill (decoded: 0 -> 6 accumulator spill
    # ops at 128^3). A per-model question, which is what the beam is for.
    # Promote small bufferization allocs to the stack. Second only to per-op blocking on the model it
    # was found on, and it is a LOCALITY lever rather than an arithmetic one: bufferization gives each
    # intermediate its own scattered heap buffer (209 of them on small_llama int8), so every one is
    # written and re-read through cache misses. MEASURED sustained on the K1, cos identical: 4,878,645
    # -> 3,649,518 ns, 1.34x, at a 256 KB per-buffer cap. ⚠️ MODEL-DEPENDENT, which is exactly why it
    # belongs to the SEARCH and not to a default: the same lever is 1.04x on small_llama fp32 and
    # ~1.01x SLOWER on spectformer int8. A blanket default would have shipped a regression.
    ("promote_buffers_to_stack", False),
    ("perop_nr_fill_register", False),
    # The M axis of the same question, and the reason it is listed rather than defaulted is the same:
    # it is a bound on the ARCHITECTURE (how many accumulator rows of this op's own block fit the
    # vector register file), not a promise about the register allocator, and only the board can say
    # whether LLVM keeps that many groups live without spilling. What it CLOSES is an asymmetry: N has
    # been derived per-op from target facts for a while, while M was one hand-set number for the whole
    # model (`zephyr_model._PEROP_MR_CAP = 4`, an env var, or a rung of `_MR_CAP_LADDER` reachable only
    # once a parent already carried blocking) -- so two contractions could differ in MR only by gcd(M)
    # clipping the cap they shared. Registered eagerly by `impr_features` next to the N-fill knob whose
    # `edit_pipeline` hook it mirrors, so no `ensure_registered` import belongs above: what an
    # unregistered name costs is not a rejection but INVISIBILITY -- `_composes` catches the KeyError
    # and returns False, and the lever is then never proposed at all.
    ("perop_mr_fill_register", False),
    ("fuse_transpose_b", False),                          # transpose: 38% byte-traffic, measured -6.5% openvla
    # `fold_weight_transpose` IS NOT LISTED, and that is a result rather than an omission. It is the
    # general form of the fold above -- it folds a weight transpose into any linalg consumer's maps,
    # so unlike `fuse_transpose_b` it does fire on a quantized model (15 of small_llama int8's 25
    # transposes, the same 15 an offline pre-transposed bundle hoists). MEASURED on the K1 anyway,
    # interleaved on top of this list's own winner: 3,594,824 ns without it, 3,994,718 ns with it --
    # 1.09x SLOWER at bit-identical output. A map fold can only FLIP which axis is contiguous, and on
    # an n-vectorized contraction the flip lands on the vectorized axis: the B read goes from
    # tensor<1x16xi8> (16 consecutive n) to tensor<16x1xi8> (16 n, 128 B apart). The feature now
    # refuses any fold that increases the stride along a consumer's fastest-varying output dim, which
    # on this model means it folds 0 of 25 -- so ranking it would buy the beam a whole-model lowering
    # to rediscover that it does nothing here. It stays registered and selectable for models whose
    # permutations leave the hot axis alone.
    ("accumulator_resident_wholemodel_vf_mrpad", True),   # matmul MR register block: 1.49x rdt2 matmul bucket
    ("vectorize_reduction", True),                        # reduce/softmax: 2nd byte-traffic family, was unvectorized
    ("erase_self_copy", False),                           # envelope: per-tile memrefCopy elimination
    # The other half of the same axis, and the reason the erase alone never closed it. The erase can
    # only remove copies that are REDUNDANT; a copy into a `memref.subview` moves real data and stays
    # a call to the rank-generic `@memrefCopy`. MEASURED on small_llama int8 (hand_v0_int8, whole
    # model, host-executed, bit-identical output): erase alone leaves `envelope.runtime_calls` at
    # ('free','malloc','memcpy','memrefCopy','memset') with 24 prologue @memrefCopy sites; adding this
    # takes it to ('free','malloc','memset') with 0 memrefCopy and 0 memcpy.
    ("expand_memref_copy", False),                        # envelope: memrefCopy/memcpy -> emitted loops
    # The third lever on the same byte-traffic axis, and the one that attacks the linalg.generic long
    # tail rather than the copies around it. `linalg-specialize-generic-ops` has to run before the
    # schedule (it recovers the contraction NAMES the transform arms match on), and it un-fuses every
    # elementwise chain on the way past -- so each per-row quantize/requant scale is materialized into
    # a full-size temporary before the op that reads it. MEASURED on small_llama int8 at this list's
    # own winner: 50 `linalg.broadcast` writing 242,944 bytes per inference from 17,476 read (13.9x,
    # zero arithmetic) -> 13 writing 60,160; emitted `forward` 35,253 -> 31,348 instructions, model.o
    # 189,008 -> 170,664 B, stack alloca 118 -> 103 sites, output BIT-IDENTICAL on spike. Ranked LAST
    # of the additive passes deliberately: the wall is UNMEASURED, and on this same model a transpose
    # fold with an equally clean static case measured 1.09x SLOWER -- pricing it is the beam's job,
    # which is the whole reason it is listed instead of defaulted. No `ensure_registered` import is
    # needed above: unlike the satellite-module levers this one is registered eagerly by
    # `impr_features` itself, where its `edit_pipeline` hook lives.
    ("fuse_elementwise_post_contraction", False),         # tail: broadcast/elementwise -> fused, 50 -> 13
    ("vectorized_transcendental_activation", True),       # gelu/sigmoid/silu: closes the 10-17x activation gap
    # The convolutional half of the same tail, and the only lever here that DELETES an intermediate
    # tensor rather than scheduling one better. model2MLIR expands every conv into im2col + matmul
    # before merlin sees it, so the operand the int8 pass dynamically quantizes IS the expanded
    # matrix: on deepjscc's `enc.net.1` a 147x4096 f32 im2col matrix, ~41x the 1x3x70x70 activation
    # it was gathered from. A trip-weighted instruction model of `forward` put 44.4% of deepjscc int8
    # in the scalar gather and 31.1% in activation quantize+amax against 18.2% in the vectorized
    # contraction (lstmnetvit: 43.8% / 35.7% / 13.2%). Moving the scale from per-parallel-row to
    # per-tensor makes the quantization commute with the gather, which puts the abs-max and the
    # quantize on the activation, moves the gather in i8, and erases the f32 expansion.
    # Ranked LAST on purpose: the numbers above are STATIC, the wall is UNMEASURED, and it is the
    # only lever on this list that is not bit-exact (a per-tensor activation scale is a real numeric
    # change), so it must earn its rank against the accuracy gate and a board measurement rather than
    # against an op count -- on this same model `fold_weight_transpose` had flawless static evidence
    # and measured 1.09x SLOWER, and `vectorize_non_contraction_generics` emitted 4.9x more vector
    # instructions and 1.28x slower. Its "bit-identical output" was a MISREADING corrected in
    # d4f86238: the digest was compared on deepjscc alone and generalised. On small_llama the lever
    # computed a WRONG answer (cos 0.968) that varied with the initial stack address, because a
    # packed `vector<8xi1>` store was read back one byte per element from the causal mask.
    ("quantize_before_gather", False),                    # im2col: quantize A, not G(A); erase the f32 expansion
]


# ---------------------------------------------------------------------------------------------------
# Per-family TEACHER registry: census op-family -> the XNNPACK expert fixture + CCA op tag. `xnn_family`
# / `status` are DERIVED from the two authoritative maps (kernel_coverage_matrix.FAMILY_MAP,
# xnnpack_kernel_catalog._MAP) so this registry only carries what those maps do not: the harvested
# fixture basename, the CCA op tag, and (for the harvester) the rvv ukernel source. A None `fixture`
# means "no XNNPACK teacher for this family" — an HONEST no-teacher record, never a faked divergence.
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilyTeacher:
    census_family: str
    op: str                        # CCA op tag used when lifting (drives activation/reduction inference)
    fixture: str | None            # basename under merlin/tests/data/cca_asm/ (None => no teacher)
    ukernel_src: str | None = None # rel path under <XNNPACK>/src for the harvester (None => already harvested)
    note: str = ""


# The liftable teachers: a census family that HAS an XNNPACK primitive whose CCA diff against our
# section is meaningful. GEMM fixtures pre-exist (run_expert_gemm); the rest are harvested by
# build_tools/scripts/harvest_xnnpack_fixtures.py into the SAME cca_asm/ dir.
FAMILY_TEACHERS: dict[str, FamilyTeacher] = {
    # contractions -> the existing GEMM expert fixture (MR=1, NR=vsetvlmax, accumulator-resident).
    "matmul": FamilyTeacher("matmul", "matmul", "xnnpack_f32_gemm_rvv.objdump",
                            "f32-gemm/gen/f32-gemm-1x4v-minmax-rvv.c",
                            note="f32 GEMM ukernel 1x4v"),
    "addmm":  FamilyTeacher("addmm", "matmul", "xnnpack_f32_gemm_rvv.objdump",
                            note="linear+bias == GEMM"),
    "linear": FamilyTeacher("linear", "matmul", "xnnpack_f32_gemm_rvv.objdump", note="== GEMM"),
    # activations -> vectorized-polynomial ukernels. The flagship non-GEMM teacher: expert lifts
    # activation_vectorization='vectorized_polynomial', ours (scalar libm) 'scalar_libm_call' ->
    # routes to vectorized_transcendental_activation.
    "gelu": FamilyTeacher("gelu", "gelu", "xnnpack_gelu_rvv.objdump",
                          "f32-vgelu/gen/f32-vgelu-rvv-rational-12-10-div-u4v.c", "rational-12-10 vgelu"),
    "sigmoid": FamilyTeacher("sigmoid", "sigmoid", "xnnpack_sigmoid_rvv.objdump",
                             "f32-vsigmoid/gen/f32-vsigmoid-rvv-rr2-p5-div-u4v.c", "rr2-p5 vsigmoid"),
    "silu": FamilyTeacher("silu", "silu", "xnnpack_sigmoid_rvv.objdump", None,
                          "SiLU = x*sigmoid; f32-vsigmoid is the closest transcendental teacher"),
    # THE TRANSFORMER TAIL. `sin`, `cos` and `rsqrt` are census families already -- the census has
    # emitted them all along -- and they were in NEITHER this registry NOR NO_TEACHER_FAMILIES, so no
    # expert was ever lifted for them and no divergence could form. That is the whole reason the loop
    # never proposed anything for RoPE or RMSNorm, measured on small_llama int8: 16.63% of an INT8
    # model's binary is scalar FLOAT, and it calls __kernel_sinf / __kernel_cosf / __kernel_rem_pio2f
    # (RoPE) and __ieee754_sqrt (RMSNorm's normaliser) per element.
    #
    # XNNPACK ships RVV kernels for exactly these, so the teacher is HARVESTED like every other one --
    # no declared or hand-authored expert, which would have broken the "the CCA is tool-composed"
    # principle the beam rests on.
    "rsqrt": FamilyTeacher("rsqrt", "rsqrt", "xnnpack_rsqrt_rvv.objdump",
                           "f32-vrsqrt/gen/f32-vrsqrt-rvv-rsqrt-u4v.c",
                           "RMSNorm normaliser; XNNPACK uses the native rsqrt estimate + Newton, "
                           "which lifts as a vfmacc chain (the axis distinguishes vector-inline math "
                           "from a scalar libm call, not the specific approximation)"),
    # sin/cos: fixture=None, an HONEST no-teacher record. XNNPACK DOES ship f32-vsin / f32-vcos RVV
    # kernels, but they do not compile in this revision: both call a TWO-argument
    # `xnn_round_f32(vx_div_2pi, vl)`, and no such overload exists anywhere in the tree -- the SIMD
    # headers define a one-argument form for AVX/HVX only, and there is no RVV SIMD header at all
    # (checked: src/xnnpack/simd/ has no rvv file, and no 2-arg xnn_round_f32 in any header). The
    # harvester therefore skips them and says so, which is correct: authoring that helper here would
    # make the EXPERT CCA a thing we wrote, and the expert's instruction mix IS the search target.
    # Recorded rather than dropped so RoPE's missing teacher is a visible gap with a reason, and so
    # the next XNNPACK bump can flip it by supplying the ukernel_src again.
    "sin": FamilyTeacher("sin", "sin", None,
                         "f32-vsin/gen/f32-vsin-rvv-rational-5-4-div-u4v.c",
                         "RoPE rotation. UNHARVESTABLE in this XNNPACK revision: the RVV kernel calls "
                         "a 2-arg xnn_round_f32 that the tree does not define. Ours pays glibc's "
                         "__kernel_rem_pio2f as a scalar call per element; no expert to diff against."),
    "cos": FamilyTeacher("cos", "cos", None,
                         "f32-vcos/gen/f32-vcos-rvv-rational-5-4-div-u4v.c",
                         "RoPE rotation, cos half. Same 2-arg xnn_round_f32 blocker as sin."),
    # reductions -> horizontal-reduce ukernels. Expert lifts reduction_form='vredsum_tree', ours
    # (scalar accumulate) 'none' -> routes to vectorize_reduction.
    "reduce": FamilyTeacher("reduce", "reduce", "xnnpack_reduce_rvv.objdump",
                            "f32-rsum/gen/f32-rsum-rvv-u4v.c", "f32-rsum horizontal reduce"),
    "reduce_mean": FamilyTeacher("reduce_mean", "reduce", "xnnpack_reduce_rvv.objdump", None,
                                 "mean == rsum * 1/N"),
    # softmax's vectorizable reduction is the row-SUM (exp-sum); taught by f32-rsum (reduction_form=
    # vredsum_tree). op tag 'reduce' (not 'softmax') so the lifter reads the reduction facet, not a
    # spurious activation classification. The row-MAX (f32-rmax) uses vfredmax, which the lifter does
    # not classify as a reduction_form -> no divergence -> not a useful teacher, so it is not used.
    "softmax": FamilyTeacher("softmax", "reduce", "xnnpack_reduce_rvv.objdump", None,
                             "softmax sum-reduce taught by f32-rsum"),
    # clamp / elementwise binary -> vectorized ukernels (the CCA diff here is thin; harvested for
    # completeness — the beam wall decides, no fork is forced if compare emits nothing).
    "minmax": FamilyTeacher("minmax", "minmax", "xnnpack_clamp_rvv.objdump",
                            "f32-vclamp/gen/f32-vclamp-rvv-u4v.c", "clamp/relu"),
    "add": FamilyTeacher("add", "add", "xnnpack_vbinary_add_rvv.objdump",
                         "f32-vbinary/gen/f32-vadd-rvv-u4v.c", "elementwise add"),
    "sub": FamilyTeacher("sub", "add", "xnnpack_vbinary_add_rvv.objdump", None, "vsub == vadd family"),
    "mul": FamilyTeacher("mul", "mul", "xnnpack_vbinary_mul_rvv.objdump",
                         "f32-vbinary/gen/f32-vmul-rvv-u4v.c", "elementwise mul"),
}

# NO-TEACHER families: census families with NO XNNPACK vector primitive (FAMILY_MAP element [0] is
# None) — an honest structural gap. Recorded, never faked. batch_matmul/sdpa are attention (no XNN
# batch/attention primitive); the rest are gather/norm/index ops.
NO_TEACHER_FAMILIES: dict[str, str] = {
    "batch_matmul": "no XNNPACK batch-matmul primitive (attention bmm) -> ours-vs-ours only",
    "sdpa": "no XNNPACK/BLAS attention primitive -> no expert teacher",
    "layer_norm": "composite (rsum+rsqrt+vbinary); no single XNNPACK kernel -> no expert teacher",
    "embedding": "gather; no XNNPACK vector primitive -> no expert teacher",
    "index_gather": "gather; no XNNPACK vector primitive -> no expert teacher",
    "select": "predicated select; no XNNPACK primitive -> no expert teacher",
}


def _coverage_maps() -> tuple[dict, dict]:
    """(FAMILY_MAP, _MAP) from the build_tools authoritative maps, lazily + defensively imported.

    Library code importing a build_tools script is fragile (repo_root may not be on sys.path), so we
    retry with repo_root inserted and fall back to ({}, {}) — the registry is self-sufficient without
    them; they only ENRICH the reported xnn_family/status.
    """
    try:
        from build_tools.scripts.kernel_coverage_matrix import FAMILY_MAP  # type: ignore
        from build_tools.scripts.xnnpack_kernel_catalog import _MAP  # type: ignore
        return FAMILY_MAP, _MAP
    except Exception:
        try:
            import sys
            from ..common.paths import repo_root
            root = str(repo_root())
            if root not in sys.path:
                sys.path.insert(0, root)
            from build_tools.scripts.kernel_coverage_matrix import FAMILY_MAP  # type: ignore
            from build_tools.scripts.xnnpack_kernel_catalog import _MAP  # type: ignore
            return FAMILY_MAP, _MAP
        except Exception:
            return {}, {}


def family_coverage(family: str) -> tuple[str | None, str | None]:
    """(xnn_family, status) for a census family from the authoritative maps — ('?', None) if absent.

    xnn_family is FAMILY_MAP[family][0] (None => no XNNPACK primitive => no teacher); status is the
    ``_MAP`` classification (mapped/partial/expert-only) of that xnn_family's leading token."""
    fmap, catalog = _coverage_maps()
    entry = fmap.get(family)
    xnn = entry[0] if entry else None
    status = None
    if xnn:
        head = xnn.split("/")[0].strip()
        for pref, (_, st) in catalog.items():
            if head.startswith(pref):
                status = st
                break
    return xnn, status


# ---------------------------------------------------------------------------------------------------
# Composition helpers (the <=1 schedule-replacement rule).
# ---------------------------------------------------------------------------------------------------
def _composes(features: list[str]) -> bool:
    """True iff the feature set is co-enable-able (no two full-schedule-replacement features)."""
    from ..llvmlower import impr_features as I
    try:
        I.normalize(features)
    except Exception:  # CompositionError (two schedule_replace) or unknown feature
        return False
    reps = [f for f in features if getattr(I.get(f), "schedule_replace", False)]
    return len(reps) <= 1


def _feature_fork(feat: str, parent_feats: list[str], *, targets: str, evidence: list[str],
                  note: str, action: Any = None) -> ForkProposal | None:
    """Merge one feature onto the parent's feature stack, honoring the <=1 schedule-replace rule.

    Returns a forkable ForkProposal, or None if the feature is already enabled or cannot compose even
    after dropping the parent's conflicting schedule-replacement feature."""
    if feat in parent_feats:
        return None
    merged = parent_feats + [feat]
    if not _composes(merged):
        # e.g. a schedule-replace feature on top of a parent that already carries one. Try replacing
        # the conflicting schedule-replacement feature instead of stacking.
        from ..llvmlower import impr_features as I
        base = [f for f in parent_feats if not getattr(I.get(f), "schedule_replace", False)]
        merged = base + [feat]
        if not _composes(merged):
            return None
    return ForkProposal(overrides={"compiler_features": merged}, lever="feature", targets=targets,
                        evidence=list(evidence), forkable=True, note=note, action=action)


#: Search ladders for the two levers whose MAGNITUDE was reachable only through an environment
#: variable, so no fork could vary it. Geometric rather than hand-picked: each rung doubles (MR) or
#: quadruples (stack cap), which brackets the measured knees without asserting where they are --
#: the knee is a per-model fact (stack promotion saturates past 256 KB on small_llama int8, MR gains
#: nothing past 8 on small_llama fp32) and locating it is the search's job, not this list's.
#:
#: 1 IS A RUNG BECAUSE THE EXPERT USES IT. XNNPACK's int8 GEMM is
#: `xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4v__rvv` -- MR=1 with a 4-register N group -- and the
#: lifted expert CCA agrees (`compute.register_block (1, ('vsetvlmax', 4.0))`). The same omission
#: was already found and fixed once on the OTHER register-block family: `MRPAD_INT8_TILES` says in
#: its own comment that the ladder started at MR=2 and "the expert's MR was not in the search space
#: at all". Leaving it out here made the expert's shape reachable only through the fixed-tile
#: family, which replaces the derived per-op N as well as the M -- so the two axes could never be
#: separated.
_MR_CAP_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16)
_STACK_CAP_LADDER: tuple[int, ...] = (16384, 65536, 262144, 1048576)


def refinement_forks(parent_feats: list[str]) -> list[ForkProposal]:
    """Forks that RETUNE a lever the parent already carries, rather than adding a new one.

    Why these are separate from ``RANKED_LEVERS``: a magnitude is only meaningful once the lever it
    belongs to is enabled, and putting every (lever, magnitude) pair in the flat list would multiply
    generation 1's width by the ladder length. The run that motivated this already deferred 11 of 12
    proposals with ``reason: over_width``, so spending width on caps for a lever that is not even in
    the parent is exactly the wrong trade. Proposing them as REFINEMENTS keeps generation 1 the same
    size and spends depth-N width on the axis the parent has already shown is worth having.

    This is also the fix for a measured mis-ranking. Stack promotion was searched only at its 16 KB
    default (env-only cap), measured 1.03x, and its fork came out slower than its own parent -- while
    the same feature at 256 KB is 1.34x on that model. The lever was not weak; the magnitude was
    unreachable.
    """
    from ..llvmlower import impr_features as I

    out: list[ForkProposal] = []
    have = set(parent_feats)

    # -- stack promotion: retune the per-buffer cap the parent is already promoting under.
    if I.PROMOTE_STACK_NAME in have or any(
            f.startswith(f"{I.PROMOTE_STACK_NAME}_") for f in have):
        base = [f for f in parent_feats
                if f != I.PROMOTE_STACK_NAME and not f.startswith(f"{I.PROMOTE_STACK_NAME}_")]
        for nbytes in _STACK_CAP_LADDER:
            name = I.ensure_promote_stack(nbytes)
            if name in have:
                continue
            merged = base + [name]
            if _composes(merged):
                out.append(ForkProposal(
                    overrides={"compiler_features": merged}, lever="knob",
                    targets=f"wholemodel:{I.PROMOTE_STACK_NAME}:cap",
                    evidence=["census:byte-traffic", f"refine:{I.PROMOTE_STACK_NAME}"],
                    forkable=True,
                    note=f"retune the stack-promotion per-buffer cap to {nbytes} bytes"))

    # -- per-op blocking: retune the MR cap the block table is derived under.
    if I.PEROP_BLOCK_NAME in have or any(
            I.parse_perop_mr_sentinel(f) is not None for f in have):
        base = [f for f in parent_feats
                if f != I.PEROP_BLOCK_NAME and I.parse_perop_mr_sentinel(f) is None]
        for mr in _MR_CAP_LADDER:
            name = I.perop_mr_sentinel(mr)
            if name in have:
                continue
            merged = base + [name]
            if _composes(merged):
                out.append(ForkProposal(
                    overrides={"compiler_features": merged}, lever="knob",
                    targets=f"wholemodel:{I.PEROP_BLOCK_NAME}:mr_cap",
                    evidence=["census:byte-traffic", f"refine:{I.PEROP_BLOCK_NAME}"],
                    forkable=True,
                    note=f"retune the per-op register-block MR cap to {mr}"))

    # -- named-op M-pad register block: retune the TILE on the int8 datapath.
    # This lever only exists once the contraction keeps its named form: the int8 quant pass rewrites
    # every linalg.matmul into a linalg.generic, and a transform schedule matching on the op NAME
    # then finds an empty handle and does nothing. So the tile refinements are proposed only
    # ALONGSIDE that feature -- proposing them without it spends width on forks that cannot fire,
    # which is exactly the failure this whole ladder exists to avoid.
    # -- named-op M-pad register block: the ENABLER AND A TILE, always together.
    # Proposing them separately cannot work. The enabler alone emits identical code, so the beam
    # excludes it as inert and it never becomes a parent; the tile alone matches `linalg.matmul`,
    # of which the int8 datapath has none, so it is a no-op the applicability check now names. Only
    # the pair does anything, so the pair is the proposal.
    tiles = getattr(I, "MRPAD_INT8_TILES", ())
    enabler = I.NAMED_INT8_CONTRACTION_NAME
    if tiles and (I.PEROP_BLOCK_NAME in have
                  or any(I.parse_perop_mr_sentinel(f) is not None for f in have)
                  or enabler in have):
        # The tile REPLACES whatever register block the parent carries -- both emit a complete
        # transform schedule, and two of those cannot compose (the feature layer refuses the pair
        # outright). Stacking them produced a CompositionError and no measurement at all.
        base = [f for f in parent_feats
                if f not in set(tiles) and f != enabler and f != I.PEROP_BLOCK_NAME
                and I.parse_perop_mr_sentinel(f) is None]
        for name in tiles:
            if name in have:
                continue
            merged = base + [enabler, name]
            if _composes(merged):
                out.append(ForkProposal(
                    overrides={"compiler_features": merged}, lever="knob",
                    targets=f"wholemodel:{I.MRPAD_NAME}:tile",
                    evidence=["census:byte-traffic", f"refine:{I.MRPAD_NAME}"],
                    forkable=True,
                    note=f"retune the named-op register-block tile to {name.rsplit('_i32_', 1)[-1]}"))
    return out


def _fork_key(fp: ForkProposal) -> tuple:
    """Dedup key. Feature forks collide iff they enable the SAME feature set (teacher & hardcode for
    one feature merge to the identical stack); other forks key on (lever, targets)."""
    feats = fp.overrides.get("compiler_features")
    if feats is not None:
        return ("feature", tuple(sorted(feats)))
    return (fp.lever, fp.targets)


def _union(primary: list[ForkProposal], secondary: list[ForkProposal]) -> list[ForkProposal]:
    """primary UNION secondary, primary winning on a key collision (teacher forks come first)."""
    seen: set[tuple] = set()
    out: list[ForkProposal] = []
    for fp in list(primary) + list(secondary):
        k = _fork_key(fp)
        if k in seen:
            continue
        seen.add(k)
        out.append(fp)
    return out


# ---------------------------------------------------------------------------------------------------
# Engine 2: census-ranked hardcodes.
# ---------------------------------------------------------------------------------------------------
def census_hardcode_forks(parent_feats: list[str]) -> list[ForkProposal]:
    """One fork per not-yet-enabled whole-model hardcode lever, merged onto the parent's features."""
    out: list[ForkProposal] = []
    for feat, _is_replace in RANKED_LEVERS:
        fp = _feature_fork(feat, parent_feats, targets=f"wholemodel:{feat}",
                           evidence=["census:byte-traffic", f"lever:{feat}"],
                           note=f"enable whole-model lever {feat} (byte-traffic ranked)")
        if fp is not None:
            out.append(fp)
    return out


# ---------------------------------------------------------------------------------------------------
# Engine 1: the per-op teacher — route real CCA divergences into forks.
# ---------------------------------------------------------------------------------------------------
def route_divergence_forks(divergences: Any, knobs: dict[str, Any]) -> list[ForkProposal]:
    """Route CCA divergences (expert-vs-ours) into forks via ``propose_forks_from_cca``, then re-merge
    each feature fork onto the parent's stack (composition). Non-feature forks (knob / work_item) pass
    through unchanged, preserving their forkable/deferred status and the CompilerAction for audit."""
    from .fork_from_action import propose_forks_from_cca
    parent_feats = list(knobs.get("compiler_features") or [])
    out: list[ForkProposal] = []
    for fp in propose_forks_from_cca(list(divergences or []), knobs):
        feats = fp.overrides.get("compiler_features") if fp.lever == "feature" and fp.forkable else None
        if feats:
            for feat in feats:  # normally a single feature per routed action
                merged = _feature_fork(feat, parent_feats, targets=fp.targets,
                                       evidence=list(fp.evidence) + ["teacher:xnnpack-cca"],
                                       note=fp.note, action=fp.action)
                if merged is not None:
                    out.append(merged)
        else:
            out.append(fp)   # knob / recorded work-item — keep as-is (honest)
    return out


def no_teacher_records(notes: list[tuple[str, str]]) -> list[ForkProposal]:
    """Honest, non-forkable records for families with no XNNPACK teacher — recorded by the beam as
    deferred, never minted or faked into a divergence."""
    return [ForkProposal(overrides={}, lever="work_item", targets=f"noteacher:{fam}",
                         evidence=[f"census-family:{fam}", "no-xnnpack-primitive"],
                         forkable=False, note=reason)
            for fam, reason in notes]


# ---------------------------------------------------------------------------------------------------
# The in-contract HYBRID proposer (drop-in for the CLI's --proposer wholemodel).
# ---------------------------------------------------------------------------------------------------
def propose_wholemodel_levers(divergences: Any, knobs: dict[str, Any]) -> list[ForkProposal]:
    """HYBRID: route the beam's own CCA divergences (per-op teacher) UNION the census hardcodes.

    This is the minimal in-contract fix: the beam already lifts parent-vs-expert divergences and
    passes them here — the old proposer IGNORED them. Now they drive forks (engine 1); the census
    hardcodes (engine 2) supply the graph-layout / shape levers no facet diff can emit. Teacher forks
    win on a collision. Degrades to pure hardcodes when ``divergences`` is empty (today's behavior)."""
    parent_feats = list(knobs.get("compiler_features") or [])
    teacher = route_divergence_forks(divergences, knobs)
    hardcodes = census_hardcode_forks(parent_feats)
    # Refinements last: they retune a magnitude on a lever the parent ALREADY carries, so they are
    # empty at the seed and cost generation 1 no width at all.
    return _union(_union(teacher, hardcodes), refinement_forks(parent_feats))


# ---------------------------------------------------------------------------------------------------
# The richer per-FAMILY teacher: expert-fixture CCA vs OUR per-family section CCA.
# The section-lift path (ours_section_cca) needs a built section (board/compile work) — it is wired +
# unit-tested here with an injected build_fn, and exercised for real on the board.
# ---------------------------------------------------------------------------------------------------
def dtype_fixture_teachers() -> list["FamilyTeacher"]:
    """The dtype-matched GEMM fixtures as harvestable teachers.

    They are not in FAMILY_TEACHERS (which is keyed by census family, one fixture each), but they must
    be re-harvestable: the pre-existing qd8 and f16 fixtures were built from an UNLINKED object, so
    every loop-scoped facet lifted as None and the expert silently taught nothing about register
    blocking, accumulator residency or memory.
    """
    return [FamilyTeacher(census_family="matmul", op="matmul", fixture=fx, ukernel_src=src,
                          note="dtype-matched GEMM fixture, harvested linked")
            for fx, src in sorted(_DTYPE_UKERNEL_SRC.items())]


def cca_asm_dir() -> Path:
    from ..common.paths import repo_root
    return repo_root() / "merlin" / "tests" / "data" / "cca_asm"


#: DTYPE-MATCHED expert fixtures, per family. A CCA diff across dtypes manufactures divergences that
#: are not gaps: comparing an int8 model to an f32 expert reports `compute.widening ours=True
#: expert=False` and `compute.epilogue ours='requant_narrow' expert='none'` -- both simply restate that
#: one side is int8, and both were OBSERVED as unrouted noise when the loop was run this way. It is the
#: same comparand-integrity failure the bundle_id guard catches on the wall axis, one axis over: the
#: thing being compared has to be comparable before a difference means anything.
#:
#: Only entries whose fixture exists are useful; a family/dtype pair with no fixture FAILS CLOSED to no
#: expert rather than silently borrowing another dtype's. That costs a divergence and buys the guarantee
#: that a reported one is real.
#: Harvestable ukernel sources for the dtype-matched GEMM fixtures. The qd8 and f16 fixtures
#: pre-existed but were harvested UNLINKED, so their loop structure was unreadable and they taught
#: nothing about register blocking. Naming the source lets the harvester regenerate them linked.
_DTYPE_UKERNEL_SRC: dict[str, str] = {
    "xnnpack_qd8_gemm_rvv.objdump": "qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4v-minmax-rvv.c",
    "xnnpack_f32_gemm_rvv.objdump": "f32-gemm/gen/f32-gemm-1x4v-minmax-rvv.c",
    # f16 needs the Zvfh arith variant; the plain `rvv` spelling does not exist for this family.
    "xnnpack_f16_gemm_rvv.objdump": "f16-gemm/gen/f16-gemm-1x4v-minmax-rvvfp16arith.c",
}

_DTYPE_FIXTURES: dict[str, dict[str, str]] = {
    "matmul": {"int8": "xnnpack_qd8_gemm_rvv.objdump",
               "fp16": "xnnpack_f16_gemm_rvv.objdump",
               "fp32": "xnnpack_f32_gemm_rvv.objdump"},
    "addmm":  {"int8": "xnnpack_qd8_gemm_rvv.objdump",
               "fp16": "xnnpack_f16_gemm_rvv.objdump",
               "fp32": "xnnpack_f32_gemm_rvv.objdump"},
    "linear": {"int8": "xnnpack_qd8_gemm_rvv.objdump",
               "fp16": "xnnpack_f16_gemm_rvv.objdump",
               "fp32": "xnnpack_f32_gemm_rvv.objdump"},
}


#: Spelling aliases -> the registry's dtype keys. The repo spells the same dtype several ways at
#: different layers (a CLI flag `--dtype f32`, a ContractionShape `('i8','i8','i32')`, a bundle name
#: `*_int8`), and the fixture registry can only be keyed on one. Normalising is a SPELLING concern, so
#: the alias set is explicit and the valid TARGETS are derived from the registry itself -- an alias
#: pointing at a key no fixture table has is a bug this raises on, not a silent None.
_DTYPE_ALIASES: dict[str, str] = {
    "f32": "fp32", "fp32": "fp32", "float32": "fp32", "float": "fp32",
    "f16": "fp16", "fp16": "fp16", "float16": "fp16", "half": "fp16",
    "i8": "int8", "int8": "int8", "qint8": "int8", "qd8": "int8", "s8": "int8",
}


def canonical_dtype(spelling: str | None) -> str | None:
    """The registry dtype key for a dtype ``spelling``, or None if nothing claims it.

    Fails CLOSED on an unrecognised spelling: None means "no dtype-matched expert", which
    :func:`expert_fixture_for` turns into no expert at all rather than another dtype's fixture. That is
    the right failure -- a cross-dtype expert diff reports differences that are only the dtype, and
    those route to real levers that then measure inert.
    """
    if not spelling:
        return None
    key = _DTYPE_ALIASES.get(str(spelling).strip().lower())
    if key is None:
        return None
    known = {k for table in _DTYPE_FIXTURES.values() for k in table}
    if key not in known:                    # an alias that outlived its fixture table
        raise KeyError(f"dtype alias {spelling!r} -> {key!r} names no key in _DTYPE_FIXTURES {sorted(known)}")
    return key


def expert_fixture_for(family: str, dtype: str | None = None) -> str | None:
    """The fixture basename to lift the expert from, for this family AND dtype.

    ``dtype=None`` keeps the registry's single default, so every existing caller is unchanged. A dtype
    with no matched fixture returns None -- no expert -- rather than falling back to another dtype's,
    because a cross-dtype diff reports differences that are only the dtype (see :data:`_DTYPE_FIXTURES`).
    """
    if dtype:
        by_dtype = _DTYPE_FIXTURES.get(family)
        if by_dtype is not None:
            return by_dtype.get(dtype)          # None => fail closed, no expert for this pair
    t = FAMILY_TEACHERS.get(family)
    return t.fixture if t is not None else None


def expert_family_cca(family: str, *, fixture_dir: Path | None = None, dtype: str | None = None):
    """Lift the per-family EXPERT CCA from its XNNPACK ukernel fixture, or None if the family has no
    teacher / the fixture has not been harvested yet. No LLM authors it — tool-composed from asm.

    ``dtype`` selects a dtype-MATCHED fixture where one exists (see :func:`expert_fixture_for`); a pair
    with no matched fixture yields no expert rather than a cross-dtype comparison."""
    teacher = FAMILY_TEACHERS.get(family)
    if teacher is None:
        return None
    fixture = expert_fixture_for(family, dtype)
    if fixture is None:
        return None
    path = (fixture_dir or cca_asm_dir()) / fixture
    if not path.is_file():
        return None
    from .beam_cli import lift_expert_cca
    return lift_expert_cca(path, teacher.op)


def divergences_across_teachers(ours, *, dtype: str | None = None,
                                families: "tuple[str, ...] | None" = None,
                                fixture_dir: "Path | None" = None):
    """Divergences for OURS against EVERY family teacher, unioned by axis.

    A whole model is not one kernel, and no single expert can answer every axis. An expert GEMM has no
    activation, so ``compute.activation_vectorization`` is UNCOMPARABLE against it -- and comparing
    against the GEMM teacher alone therefore reports no divergence on a model's activation cost no
    matter how large it is.

    MEASURED, and this is why the function exists: on small_llama fp32 the dynamic profile puts scalar
    `exp` at 16.48% of real model work (``__ieee754_expf`` 11.91% + ``expf`` 4.57%), against 2.42% for
    ALL scalar math on the int8 build of the same model. The loop run against the matmul teacher
    reported that axis as uncomparable and raised nothing -- the single largest fp32-specific cost was
    invisible to discovery. The gelu/sigmoid teachers answer that axis; they simply were not consulted.

    Each teacher contributes only the axes IT can answer: a divergence is kept when the teacher's side
    of the axis is populated, so a GEMM never gets to opine on activations and vice versa. First
    teacher to answer an axis wins, in the registry's byte-traffic order, so the ranking stays the
    census's rather than this function's. Returns ``(divergences, taught_by, uncomparable)`` -- and the
    third element is the point: an axis NO teacher could answer is reported, not dropped.
    """
    from ..kernels import cca_compare

    from dataclasses import asdict as _asdict

    #: `compute.op` is the op LABEL, not a property of the code. A cross-family teacher differs on it
    #: by construction -- comparing our matmul against the `mul` teacher reports ours='matmul'
    #: expert='mul' -- which is an identity mismatch, not a gap, and it routes nowhere useful.
    IDENTITY_AXES = ("compute.op",)

    def _populated(c) -> set:
        """Axes this CCA can actually answer -- populated, regardless of VALUE."""
        out = set()
        for facet in cca_compare._facet_names():
            f = getattr(c, facet, None)
            if f is None:
                continue
            out |= {f"{facet}.{k}" for k, v in _asdict(f).items() if v is not None}
        return out

    order = families if families is not None else tuple(default_teacher_families())
    ours_axes = _populated(ours)
    seen: dict[str, object] = {}
    taught_by: dict[str, str] = {}
    answered: set[str] = set()
    for fam in order:
        expert = expert_family_cca(fam, fixture_dir=fixture_dir, dtype=dtype)
        if expert is None:
            continue
        # An axis is ANSWERED when both sides are populated -- whether or not they differ. Deriving
        # this from the divergence list instead would mark every AGREEING axis as unanswered, which
        # over-reports blindness and buries the axes genuinely nobody can teach.
        answered |= _populated(expert) & ours_axes
        # Stamp the TEACHER into the divergence's own evidence, so attribution travels with the
        # divergence instead of in a parallel dict the consumer has to remember to thread. A consumer
        # (the beam) receives only the list; without this it cannot say which expert justified an
        # axis, and "the expert says LMUL=4" is unauditable when there are nine experts.
        ev = [f"teacher:{fam}", str(expert.provenance.get("source", "expert"))]
        for d in cca_compare.compare(expert, ours, evidence=ev):
            if d.axis in IDENTITY_AXES or d.axis in seen:
                continue
            seen[d.axis] = d
            taught_by[d.axis] = fam
    # axes OURS populated that no teacher could answer at all -- genuine coverage gaps in the teacher
    # set, reported rather than dropped, because the loop can only discover what it is shown.
    unanswered = sorted(a for a in ours_axes - answered if a not in IDENTITY_AXES)
    return list(seen.values()), taught_by, unanswered


def teacher_compare_fn(*, dtype: str | None = None,
                       families: "tuple[str, ...] | None" = None,
                       fixture_dir: "Path | None" = None,
                       record: list | None = None):
    """A ``compare_fn(ours) -> [Divergence]`` for :func:`mining.beam.run_beam`, backed by EVERY teacher.

    The beam's default expert side is ONE lifted fixture, which silently bounds what the search can
    discover: an axis the single expert cannot answer is uncomparable, raises no divergence, routes to
    no action, and is therefore never forked -- regardless of how much of the model's wall it owns.
    MEASURED on small_llama fp32: the matmul-teacher-only diff found 5 divergences and 4 mintable forks
    and reported ``compute.activation_vectorization`` as uncomparable, while the dynamic profile puts
    scalar `exp` at 16.48% of real model work. Consulting all teachers: 9 divergences, 6 mintable forks,
    including that axis (taught by gelu) and ``compute.reduction_form`` (taught by softmax).

    ``dtype`` is normalised by :func:`canonical_dtype`, so a caller may pass whatever its layer spells.
    Appends one audit record per invocation to ``record`` when given -- the teacher that justified each
    axis, and the axes NO teacher could answer, which is the coverage gap of the teacher SET and the
    only honest way to read "no divergence found".
    """
    dt = canonical_dtype(dtype)

    def _compare(ours):
        divs, taught_by, unanswered = divergences_across_teachers(
            ours, dtype=dt, families=families, fixture_dir=fixture_dir)
        if record is not None:
            record.append({"dtype": dt, "n_divergences": len(divs),
                           "taught_by": dict(taught_by), "unanswered_axes": list(unanswered)})
        return divs

    return _compare


def family_region_ids(model_dir: str | Path, family: str) -> list[str]:
    """The ``prov.region_id``s of the top-level @forward ops whose family matches ``family`` (matching
    either ``prov.op`` or ``prov.family``) — the section the teacher scopes OUR CCA to."""
    from ..llvmlower.op_profile import find_forward_ops
    text = Path(model_dir, "model.mlir").read_text()
    _, _, ops = find_forward_ops(text)
    rids: list[str] = []
    for rec in ops:
        rid = rec.get("region_id")
        if rid and (rec.get("op") == family or rec.get("family") == family):
            rids.append(rid)
    return rids


#: A build_fn takes a built section bundle dir and returns ``(objdump_text, undefined_symbols)`` for
#: its emitted object — the board/compile seam the section-lift path depends on. The board agent
#: supplies one that runs ``mining.k1.build_k1_binary`` on the section and reads
#: ``decode.objdump.disassemble_text`` / ``undefined_symbols`` off ``generated/model.o``. Tests inject
#: a mock returning a canned objdump.
SectionBuildFn = Callable[[Path], "tuple[str, tuple[str, ...] | None]"]


def ours_section_cca(model_dir: str | Path, family: str, *, build_fn: SectionBuildFn,
                     op: str | None = None, work_root: str | Path | None = None, seed: int = 0):
    """Lift OUR per-family CCA by slicing the model to ``family``'s regions, building that section, and
    lifting the emitted asm — the section-scoped analog of ``beam._cca_divergences``.

    Returns None when the model has no op of that family (nothing to teach). ``build_fn`` is the board
    seam (see :data:`SectionBuildFn`); everything else is host-side and deterministic."""
    from .section_build import build_section_bundle
    op = op or (FAMILY_TEACHERS.get(family).op if FAMILY_TEACHERS.get(family) else family)
    rids = family_region_ids(model_dir, family)
    if not rids:
        return None
    if work_root is None:
        from ..common.artifacts import cache_dir
        work_root = cache_dir("beam-sections")
    work = Path(work_root) / f"section_{family}"
    build_section_bundle(model_dir, rids, work, seed=seed)
    objdump_text, undef = build_fn(work)
    from ..kernels import cca
    from ..kernels.decode import rvv
    return cca.lift_asm(rvv.decode_text(objdump_text), op=op, source="ours",
                        undefined_symbols=undef)


def default_teacher_families() -> list[str]:
    """The liftable teacher families (a fixture is registered), ordered by measured census byte-traffic
    (falls back to registry order when the census json is absent, e.g. a fresh worktree)."""
    liftable = [f for f, t in FAMILY_TEACHERS.items() if t.fixture is not None]
    order = _census_byte_order()
    if not order:
        return liftable
    # census-ranked first (highest bytes_share), then any liftable family the census did not rank.
    ranked = [f for f in order if f in liftable]
    return ranked + [f for f in liftable if f not in ranked]


def _census_byte_order() -> list[str]:
    """Census families ordered by descending ``mean_bytes_share`` (empty if the census json is absent)."""
    import json
    from ..common.paths import artifacts_dir
    p = artifacts_dir() / "ceiling" / "model_op_census.json"
    if not p.is_file():
        return []
    try:
        rows = json.loads(p.read_text()).get("ranking", [])
    except Exception:
        return []
    rows = [r for r in rows if r.get("family")]
    rows.sort(key=lambda r: r.get("mean_bytes_share", 0.0), reverse=True)
    return [r["family"] for r in rows]


def per_family_teacher_divergences(model_dir: str | Path, families: list[str] | None = None, *,
                                   build_fn: SectionBuildFn | None = None,
                                   expert_fn: Callable[[str], Any] | None = None,
                                   ours_fn: Callable[[str], Any] | None = None,
                                   ) -> tuple[list, list[tuple[str, str]]]:
    """Pair a per-family EXPERT CCA against OUR per-family section CCA and compare -> (all divergences,
    no-teacher notes). ``expert_fn``/``ours_fn`` default to the fixture + section-lift paths; tests
    inject mocks so no board/compile is needed. A family with an expert but no emitted section (or vice
    versa) is honestly recorded as a no-teacher note, not a divergence."""
    from ..kernels import cca_compare
    expert_fn = expert_fn or expert_family_cca
    if ours_fn is None:
        if build_fn is None:
            raise ValueError("per_family_teacher_divergences needs build_fn (or an explicit ours_fn)")
        ours_fn = lambda fam: ours_section_cca(model_dir, fam, build_fn=build_fn)  # noqa: E731
    families = families if families is not None else default_teacher_families()
    divergences: list = []
    notes: list[tuple[str, str]] = []
    for fam in families:
        expert = expert_fn(fam)
        if expert is None:
            notes.append((fam, NO_TEACHER_FAMILIES.get(fam, f"no XNNPACK teacher fixture for {fam!r}")))
            continue
        ours = ours_fn(fam)
        if ours is None:
            notes.append((fam, f"expert teacher exists for {fam!r} but the model emits no such section"))
            continue
        divergences.extend(cca_compare.compare(expert, ours, evidence=[f"xnnpack:{fam}"]))
    # families that are structurally no-teacher and were not among `families`: still record them if
    # the model actually runs them (honest surface of where we have no expert).
    for fam, reason in NO_TEACHER_FAMILIES.items():
        if fam not in families and not any(n[0] == fam for n in notes):
            notes.append((fam, reason))
    return divergences, notes


def make_per_op_teacher_proposer(model_dir: str | Path | None = None,
                                 families: list[str] | None = None, *,
                                 build_fn: SectionBuildFn | None = None,
                                 precomputed_divergences: list | None = None,
                                 no_teacher_notes: list[tuple[str, str]] | None = None,
                                 ) -> Callable[[Any, dict], list[ForkProposal]]:
    """Bind the per-FAMILY teacher and return a ``(divergences, knobs) -> [ForkProposal]`` closure.

    The closure routes the precomputed per-family teacher divergences (expert-fixture-vs-our-section,
    computed once here) TOGETHER with the beam's own parent-vs-expert divergences, UNION the census
    hardcodes, plus the honest no-teacher records. Preserves the beam's proposer contract and the
    <=1-schedule-replacement composition rule.

    Two ways to supply the teacher divergences:
      * ``precomputed_divergences`` (+ ``no_teacher_notes``) — pass them directly (tests, or a caller
        that already ran the section builds);
      * ``model_dir`` + ``build_fn`` — compute them here via ``per_family_teacher_divergences`` (the
        board path: build_fn runs the K1 section build)."""
    if precomputed_divergences is None:
        if model_dir is None or build_fn is None:
            raise ValueError("make_per_op_teacher_proposer needs precomputed_divergences, or "
                             "model_dir + build_fn to compute them")
        precomputed_divergences, no_teacher_notes = per_family_teacher_divergences(
            model_dir, families, build_fn=build_fn)
    teacher_divs = list(precomputed_divergences or [])
    notes = list(no_teacher_notes or [])

    def proposer(divergences: Any, knobs: dict[str, Any]) -> list[ForkProposal]:
        parent_feats = list(knobs.get("compiler_features") or [])
        all_divs = teacher_divs + list(divergences or [])
        teacher = route_divergence_forks(all_divs, knobs)
        hardcodes = census_hardcode_forks(parent_feats)
        forks = _union(teacher, hardcodes)
        forks.extend(no_teacher_records(notes))
        return forks

    return proposer
