"""Per-operation block requests, sentinels and selected block schedule declarations."""

from __future__ import annotations

from merlin.compiler.optimizations.hoisting import _accumulator_resident_v3_pipeline
from merlin.compiler.optimizations.resident import _tile_epilogue_hygiene
from merlin.compiler.selection.features import ImprFeature

PEROP_BLOCK_NAME = "perop_register_block"


def _perop_sentinel_unresolved(_passes):
    """Reached only if the sentinel survived to lowering, which is a BUG -- so raise, loudly.

    ``perop_register_block`` is a REQUEST, not a lowering edit: the block table can only be derived
    from the PREPARED module, so ``zephyr_model.prepare_for_lowering`` derives it, tags the IR, and
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
    ``zephyr_model.prepare_for_lowering``, never a lowering edit -- so it raises identically if it
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

    from merlin.compiler.optimizations.registry import known, register
    from merlin.llvmlower import perop_blocks as _pb

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
