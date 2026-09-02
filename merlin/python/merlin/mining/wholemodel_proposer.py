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

# Whole-model HARDCODE levers, most-impactful first by measured byte-traffic / e2e attribution. Each
# entry is (feature_name, is_full_schedule_replacement). These are the levers a per-facet CCA diff
# CANNOT emit — a GRAPH-layout decision (transpose) or a shape decision (per-matmul MR) with no CCA
# facet field — plus the additive passes as a teacher-idle fallback. The teacher (engine 1) supplies
# the rest from real divergences.
RANKED_LEVERS: list[tuple[str, bool]] = [
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
    ("perop_nr_fill_register", False),
    ("fuse_transpose_b", False),                          # transpose: 38% byte-traffic, measured -6.5% openvla
    ("accumulator_resident_wholemodel_vf_mrpad", True),   # matmul MR register block: 1.49x rdt2 matmul bucket
    ("vectorize_reduction", True),                        # reduce/softmax: 2nd byte-traffic family, was unvectorized
    ("erase_self_copy", False),                           # envelope: per-tile memrefCopy elimination
    ("vectorized_transcendental_activation", True),       # gelu/sigmoid/silu: closes the 10-17x activation gap
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
    return _union(teacher, hardcodes)


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
