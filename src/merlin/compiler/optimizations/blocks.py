"""On-demand contraction block feature declarations and claimed-op reporting."""

from __future__ import annotations

from merlin.compiler.optimizations.hoisting import _accumulator_resident_v3_pipeline
from merlin.compiler.optimizations.loops import (
    _accumulator_resident_v3_kblocked_pre_schedule,
    _accumulator_resident_v3_pre_schedule,
    _accumulator_resident_v3_scalable_pre_schedule,
    _accumulator_resident_v3_unrolled_pre_schedule,
    scalable_lanes,
)
from merlin.compiler.optimizations.resident import _tile_epilogue_hygiene
from merlin.compiler.selection.features import ImprFeature


def ensure_v3_kblocked_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the K-BLOCKED v3 tuning point — makes the KC lever real (cache blocking)."""
    from merlin.compiler.optimizations.registry import known, register

    name = f"accum_resident_v3kb_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"Accumulator-resident micro-kernel with REAL K-blocking (MR={MR}, NR={NR}, "
                f"KC={KC}): K tiled by KC for cache reuse of the B panel, then by 1 for the "
                f"register-resident accumulation. Compiler-emitted (no ukernel). Default-off."
            ),
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=(
                lambda _t, _MR=MR, _NR=NR, _KC=KC: _accumulator_resident_v3_kblocked_pre_schedule(_MR, _NR, _KC)
            ),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(MR),
        )
    )
    return name


def ensure_v3_unrolled_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the M-UNROLLED v3 tuning point — MR independent accumulators, any MR."""
    from merlin.compiler.optimizations.registry import known, register

    name = f"accum_resident_v3u_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"Accumulator-resident micro-kernel with M UNROLLED into {MR} independent "
                f"accumulators (NR={NR}, KC={KC}) — shape-agnostic in MR, unlike the 2-D "
                f"vector<MRxNR> formulation. Compiler-emitted (no ukernel). Default-off."
            ),
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=(
                lambda _t, _MR=MR, _NR=NR, _KC=KC: _accumulator_resident_v3_unrolled_pre_schedule(_MR, _NR, _KC)
            ),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(MR),
        )
    )
    return name


def ensure_v3_scalable_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the VL-AGNOSTIC v3 tuning point — a scalable (``vsetvli``-sized) N block.

    This is the realization of ``MicrokernelSpec.vl_strategy = VL_DYNAMIC``: the emitted inner loop
    sizes itself to the vector length the hardware reports, so it needs no ``_zvl`` march pin and is
    correct (and full-width) on ANY RVV part."""
    from merlin.compiler.optimizations.registry import known, register

    k = scalable_lanes(NR)
    name = f"accum_resident_v3vl_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"VL-AGNOSTIC accumulator-resident micro-kernel (MR={MR}, NR={NR} lanes at the "
                f"RVV minimum VLEN -> vector<[{k}]xT>, KC={KC}): the N register block is a "
                f"SCALABLE vector, so the emitted loop sizes to the hardware's runtime vector "
                f"length (vsetvli against VLMAX) instead of a compile-time width that the "
                f"backend must widen for the worst-case VLEN. Needs no _zvl march pin. N loop "
                f"is peeled so the main body is unmasked; the remainder is a scalar tail. "
                f"Compiler-emitted (no ukernel). Default-off."
            ),
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=(
                lambda _t, _MR=MR, _NR=NR, _KC=KC: _accumulator_resident_v3_scalable_pre_schedule(_MR, _NR, _KC)
            ),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(MR),
        )
    )
    return name


def ensure_v3_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) and return the name of the v3 accumulator-resident micro-kernel tuning
    point for ANY (MR, NR, KC) — turning the fixed 4-point grid into a CONTINUOUS, beam-tunable space.

    This is how the compiler reaches expert-kernel granularity by CODE GENERATION alone: the v3 recipe
    is fully compiler-emitted (tile [MR,NR] + K-by-1, scoped-vectorize -> vector.contract, PRE-bufferize
    subset-hoisting makes the accumulator a register-resident scf.for iter_arg, contraction ->
    outerproduct -> vector.fma, then A-scalarization -> vfmacc.vf). NO hand ukernel is involved — the
    intrinsic driver remains a ceiling REFERENCE only. Registering a point is idempotent."""
    from merlin.compiler.optimizations.registry import known, register

    name = "accumulator_resident_microkernel_v3" if (MR, NR, KC) == (4, 16, 16) else f"accum_resident_v3_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"Accumulator-resident vfmacc.vf micro-kernel tuning point (MR={MR}, NR={NR}, "
                f"KC={KC}), registered on demand so the beam can tune the register block "
                f"continuously. Compiler-emitted (no ukernel). Default-off."
            ),
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC: _accumulator_resident_v3_pre_schedule(_MR, _NR, _KC)),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(MR),
        )
    )
    return name


def ensure_v3_perop_microkernel(
    MR_mm: int | None, NR_mm: int | None, MR_bmm: int | None, NR_bmm: int | None, KC: int
) -> str:
    """Register (on demand) a v3 tuning point with an INDEPENDENT register block per op class.

    ``_accumulator_resident_v3_pre_schedule`` already emits four separate tile factors — the matmul
    arm uses ``[MR_mm, NR_mm]`` and the batch_matmul arm ``[1, MR_bmm, NR_bmm]`` — but until now the
    only way to reach them was the hand-frozen ``accumulator_resident_wholemodel_vf`` point, whose
    clamps (MR_mm=1, NR_bmm=8) are constants somebody picked once. They are a SHAPE decision: a
    model's matmuls and its attention batch_matmuls have different extents, so the largest
    non-masking block differs per class. This registrar makes that decision expressible, so a
    shape-aware policy can DERIVE the four factors from the workload instead of pinning them.

    ``accumulator_resident_wholemodel_vf`` == ``ensure_v3_perop_microkernel(1, 16, 4, 8, 16)``; that
    point keeps its own name (and its measured history) rather than being aliased here.

    A class's block may be ``None``, meaning DO NOT CLAIM this op class — its contractions are left
    to ``convert-linalg-to-loops``. That is the honest answer when no block wider than one lane is
    legal for the class's extents (see ``_accumulator_resident_v3_pre_schedule``); the alternative is
    a build that fails outright. Both classes ``None`` would vectorize nothing, so it is rejected."""
    from merlin.compiler.optimizations.registry import known, register

    skip_mm = MR_mm is None or NR_mm is None
    skip_bmm = MR_bmm is None or NR_bmm is None
    if skip_mm and skip_bmm:
        raise ValueError(
            "ensure_v3_perop_microkernel: at least one op class must be claimed "
            "(both blocks None vectorizes nothing — use the scalar backend instead)"
        )
    tag_mm = "x_x" if skip_mm else f"{MR_mm}_{NR_mm}"
    tag_bmm = "x_x" if skip_bmm else f"{MR_bmm}_{NR_bmm}"
    name = f"accum_resident_v3p_{tag_mm}_{tag_bmm}_{KC}"
    if name in known():
        return name
    _mm_desc = "linalg.matmul UNCLAIMED (scalar)" if skip_mm else f"linalg.matmul [{MR_mm}, {NR_mm}]"
    _bmm_desc = "linalg.batch_matmul UNCLAIMED (scalar)" if skip_bmm else f"linalg.batch_matmul [1, {MR_bmm}, {NR_bmm}]"
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"Accumulator-resident vfmacc.vf micro-kernel with a per-op-class register "
                f"block ({_mm_desc}, {_bmm_desc}, KC={KC}), registered on demand so a "
                f"shape-aware policy can pick a blocking that masks no parallel dim. "
                f"Compiler-emitted (no ukernel). Default-off."
            ),
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=(
                lambda _t, _mm=MR_mm, _nm=NR_mm, _mb=MR_bmm, _nb=NR_bmm, _KC=KC, _sm=skip_mm, _sb=skip_bmm: (
                    _accumulator_resident_v3_pre_schedule(
                        _mb or 1, _nm or 1, _KC, NR_bmm=_nb or 1, MR_mm=_mm or 1, skip_mm=_sm, skip_bmm=_sb
                    )
                )
            ),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(None if skip_mm else MR_mm),
        )
    )
    return name


_V3P_PREFIX = "accum_resident_v3p_"


_V3P_CLASS_ORDER = ("linalg.matmul", "linalg.batch_matmul")


def unclaimed_op_classes(feature: str) -> list[str]:
    """Contraction op classes a ``accum_resident_v3p_*`` point does NOT tile/vectorize.

    Lives next to :func:`ensure_v3_perop_microkernel` so the ``x_x`` spelling has ONE definition:
    a caller that wants to report "this class runs scalar" must not re-derive the name format.
    Returns [] for any other feature (including the hand-frozen points, which claim both classes).
    """
    if not feature.startswith(_V3P_PREFIX):
        return []
    parts = feature[len(_V3P_PREFIX) :].split("_")
    if len(parts) != 2 * len(_V3P_CLASS_ORDER) + 1:
        return []
    return [cls for i, cls in enumerate(_V3P_CLASS_ORDER) if parts[2 * i] == "x"]
