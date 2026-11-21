"""On-demand contraction block feature declarations and claimed-op reporting."""

from __future__ import annotations

from merlin.compiler.optimizations.accumulator_residency import (
    _accumulator_resident_v3_pipeline,
    _tile_epilogue_hygiene,
)
from merlin.compiler.optimizations.contraction_loops import (
    _accumulator_resident_v3_kblocked_pre_schedule,
    _accumulator_resident_v3_pre_schedule,
    _accumulator_resident_v3_scalable_pre_schedule,
    _accumulator_resident_v3_unrolled_pre_schedule,
    scalable_lanes,
)
from merlin.compiler.selection.features import ImprFeature

# ---- blocks ------------------------------------------------------------------------------------


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


# ---- perop -------------------------------------------------------------------------------------
# Per-operation block requests, sentinels and selected block schedule declarations.

PEROP_BLOCK_NAME = "perop_register_block"


def _perop_sentinel_unresolved(_passes):
    """Reached only if the sentinel survived to lowering, which is a BUG -- so raise, loudly.

    ``perop_register_block`` is a REQUEST, not a lowering edit: the block table can only be derived
    from the PREPARED module, so ``zephyr_preparation.prepare_for_lowering`` derives it, tags the IR, and
    swaps this name for the concrete ``ensure_perop_block(...)`` feature. If the sentinel is still
    enabled at ``apply_pipeline`` time, the caller skipped that step -- and the consequence is
    invisible: nothing tagged the IR, so no schedule arm matches and EVERY contraction falls to
    ``convert-linalg-to-loops`` while the build reports success and the numbers stay correct. That is
    the measured deepjscc "2.56x regression that looks like a bad block but is an untagged build".
    Failing here is the only way that gets noticed.
    """
    raise RuntimeError(
        f"{PEROP_BLOCK_NAME!r} reached the lowering pipeline unresolved. It must be consumed by "
        "runtime.backends.zephyr_model.prepare_for_lowering, which derives the per-op block table "
        "from the prepared IR, tags the contractions, and replaces this sentinel with the concrete "
        "ensure_perop_block(...) feature. Lowering with it still set would leave every contraction "
        "untagged and silently scalar."
    )


PEROP_NR_FILL_NAME = "perop_nr_fill_register"


def _perop_nr_fill_unresolved(_passes):
    """Same contract as the block sentinel: a REQUEST consumed at preparation time, never a pass."""
    raise RuntimeError(
        f"{PEROP_NR_FILL_NAME!r} reached the lowering pipeline unresolved. It must be consumed by "
        "runtime.backends.zephyr_model.prepare_for_lowering, which is where the per-op block table is "
        "derived and is therefore the only place the board's vector length can widen an N cap."
    )


PEROP_MR_FILL_NAME = "perop_mr_fill_register"


def _perop_mr_fill_unresolved(_passes):
    """Same contract as the two sentinels above: a REQUEST consumed at preparation time.

    It has to be, for the same reason: the M cap it derives is a function of each contraction's OWN
    block and element widths, and the prepared module is the first place those are readable. A
    sentinel that reaches the pipeline means preparation was skipped, and the consequence would be
    silent -- the table would be derived under the ambient cap and the build would report a lever it
    never applied.
    """
    raise RuntimeError(
        f"{PEROP_MR_FILL_NAME!r} reached the lowering pipeline unresolved. It must be consumed by "
        "runtime.backends.zephyr_model.prepare_for_lowering, which is where the per-op block table is "
        "derived and is therefore the only place the board's vector register file can size an M cap."
    )


PEROP_MR_SENTINEL_PREFIX = f"{PEROP_BLOCK_NAME}_mr"


PEROP_NR_SENTINEL_PREFIX = f"{PEROP_BLOCK_NAME}_nr"


def parse_perop_nr_sentinel(name: str) -> int | None:
    """The NR cap named by ``perop_register_block_nr<N>``, or ``None`` for another feature."""
    if not name.startswith(PEROP_NR_SENTINEL_PREFIX):
        return None
    suffix = name[len(PEROP_NR_SENTINEL_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def perop_nr_sentinel(nr_cap: int) -> str:
    """Register a preparation-time request that pins the per-op N-tile cap."""
    from merlin.compiler.optimizations.registry import known, register

    n = int(nr_cap)
    if n <= 0:
        raise ValueError(f"{PEROP_BLOCK_NAME}: NR cap must be positive, got {nr_cap!r}")
    name = f"{PEROP_NR_SENTINEL_PREFIX}{n}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"request PER-CONTRACTION register blocking with the NR cap pinned to {n}. "
                f"Identical to {PEROP_BLOCK_NAME} except the N cap is named and therefore "
                "searchable and reproducible. This exposes the model-specific tradeoff "
                "between wider panels and i32 accumulator register pressure. A sentinel "
                "resolved by prepare_for_lowering; default-off and baseline byte-identical."
            ),
            edit_pipeline=_perop_sentinel_unresolved,
            schedule_replace=True,
        )
    )
    return name


def parse_perop_mr_sentinel(name: str) -> int | None:
    """The MR cap a ``perop_register_block_mr<N>`` sentinel names, or None if ``name`` is not one.

    Deliberately strict: the suffix must be ALL DIGITS. ``ensure_perop_block`` mints names in the same
    ``perop_register_block_*`` namespace (``..._3b_128_<hash>``), so a prefix-only test would read a
    RESOLVED feature as an unresolved sentinel and re-derive the table underneath it.
    """
    if not name.startswith(PEROP_MR_SENTINEL_PREFIX):
        return None
    suffix = name[len(PEROP_MR_SENTINEL_PREFIX) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def perop_mr_sentinel(mr_cap: int) -> str:
    """Register (on demand) a per-op blocking request that PINS the MR cap to ``mr_cap``.

    Same contract as :data:`PEROP_BLOCK_NAME` -- a request consumed by
    ``zephyr_preparation.prepare_for_lowering``, never a lowering edit -- so it raises identically if it
    reaches the pipeline unresolved.

    The cap decides how much A-operand reuse each contraction can buy, and the right value is a
    property of the MODEL's shapes: the default 4 was measured at 128^3, where M is large and MR is a
    pure register-pressure question. On small_llama fp32 every contraction is M=8 with 2.0 MAC per
    byte, so the binding cost is weight TRAFFIC and a cap of 4 makes an M=8 op take two row-blocks,
    streaming the whole B matrix twice (measured K1, n=3, cos 0.9999999 on every arm: cap 4 ->
    5,762,513 ns, cap 8 -> 5,121,007 ns = 1.125x, cap 16 -> 5,153,090 ns, i.e. no further gain).
    Opposite regimes want opposite numbers, which is what makes it a search axis rather than a
    constant -- and as an env var no fork could vary it.
    """
    from merlin.compiler.optimizations.registry import known, register

    n = int(mr_cap)
    if n <= 0:
        raise ValueError(f"{PEROP_BLOCK_NAME}: MR cap must be positive, got {mr_cap!r}")
    name = f"{PEROP_MR_SENTINEL_PREFIX}{n}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"request PER-CONTRACTION register blocking with the MR cap pinned to {n}. "
                f"Identical to {PEROP_BLOCK_NAME} except the cap is NAMED rather than read from "
                f"MERLIN_PEROP_MR_CAP, so the beam can search it: the best cap is a property of "
                f"the model's shapes (measured 1.125x from 4 -> 8 on an M=8, traffic-bound "
                f"model; no further gain at 16). A sentinel, resolved by prepare_for_lowering. "
                f"Default-off; baseline byte-identical."
            ),
            edit_pipeline=_perop_sentinel_unresolved,
            schedule_replace=True,
        )
    )
    return name


def ensure_perop_block(table, kc: int, pairs: "list | tuple" = (), vec_epilogue: bool = False) -> str:
    """Register (on demand) the per-op-blocked schedule for THIS model's block table.

    The schedule text is a function of the table (one tile+vectorize arm per distinct block), so the
    feature is registered per distinct table -- named by a hash of it, the same on-demand pattern
    ``ensure_v3_perop_microkernel`` uses for the register-block knob space.
    """
    import hashlib

    from merlin.compiler.backend import providers as provided
    from merlin.compiler.optimizations.registry import known, register

    _pb = provided.component("perop_blocks")
    blocks = _pb.distinct_blocks(table)
    # The PAIR TABLE is part of the key, not just of the text. Two builds of the same model that
    # differ only in whether the requant epilogue was fused are two different schedules; keying on
    # the table alone would hand the second one the first one's already-registered feature and it
    # would build the arm it did not ask for -- silently, and with the lever reported as applied.
    pairs = [list(x) for x in (pairs or ())]
    # With NO pairs the key is the table alone, byte-for-byte as before this argument existed: an
    # existing package names its concrete feature by that hash in `compiler_features`, and changing
    # the unfused spelling would make every one of them unresolvable.
    key = hashlib.sha1(
        (repr(sorted(table.items())) if not pairs else repr((sorted(table.items()), pairs, vec_epilogue))).encode()
    ).hexdigest()[:12]
    name = f"{PEROP_BLOCK_NAME}_{len(blocks)}b_{kc}_{key}"
    if name in known():
        return name
    text = _pb.schedule_text(table, kc, pairs, vec_epilogue)
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"Per-op register blocking: {len(blocks)} distinct blocks over "
                f"{len(table)} contraction geometries, KC={kc}. Each contraction is tiled at the "
                f"widest block legal for its own extents, matched by the merlin.blk_<MR>x<NR> tag "
                f"the prepare step applies after specialization. Replaces the per-op-CLASS block, "
                f"whose smallest member otherwise clamps the whole class (measured: whisper_tiny "
                f"claims 65.9% of its MACs per class vs 100% per op)."
                + (
                    f" {len(pairs)} of them additionally carry a FUSED requantize epilogue "
                    f"(fuse_requant_into_contraction): the epilogue is tiled and the contraction "
                    f"and its accumulator fill are fused into that loop, so the i32 accumulator "
                    f"is converted and scaled in the tile that produced it and the model-sized "
                    f"i32 tensor is never built."
                    + (
                        " The epilogue tile is additionally pre-vectorized at that block."
                        if vec_epilogue
                        else " The epilogue tile keeps its loop form and is left to clang, so "
                        "the lever removes a traversal without reshaping a loop."
                    )
                    if pairs
                    else ""
                )
            ),
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=lambda _t, _text=text: _text,
            schedule_replace=True,
            # The tile-epilogue hygiene, keyed on the WIDEST matmul block in the table. This site was
            # missed when `implies` was added to the eight other v3 registration points, and it is the one
            # that mattered most: it is the whole-model per-op path, so every board build went out paying
            # for a per-tile `memref.copy %x, %x` that does nothing. MEASURED on small_llama int8, spike
            # PC histogram over the linked ELF: `memrefCopy` was 28.15% of all retired instructions --
            # more than `forward` itself at 27.09% -- while the scalar-math routines everyone (including
            # this session) had been ranking first were 1.88%, i.e. inside the board's noise band. A static
            # instruction count had put those at 16.63%; the dynamic profile is what corrected it.
            implies=_tile_epilogue_hygiene(max((mr for mr, _nr in table.values()), default=1)),
        )
    )
    return name
