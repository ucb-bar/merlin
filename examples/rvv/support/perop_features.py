"""RVV per-contraction register-block requests and concrete schedules."""

from __future__ import annotations

import hashlib

from merlin.llvmlower.feature_registry import FeatureRegistry, ImprFeature

from .active_features import _accumulator_resident_v3_pipeline
from .perop_blocks import distinct_blocks, schedule_text

PEROP_BLOCK_NAME = "perop_register_block"
PEROP_NR_FILL_NAME = "perop_nr_fill_register"
PEROP_MR_FILL_NAME = "perop_mr_fill_register"
PEROP_MR_SENTINEL_PREFIX = f"{PEROP_BLOCK_NAME}_mr"
PEROP_NR_SENTINEL_PREFIX = f"{PEROP_BLOCK_NAME}_nr"


def _unresolved(name: str):
    def fail(_passes):
        if name == PEROP_BLOCK_NAME:
            raise RuntimeError(
                f"{name!r} reached the lowering pipeline unresolved. It must be consumed by "
                "runtime.backends.zephyr_model.prepare_for_lowering, which derives the per-op block table "
                "from the prepared IR, tags the contractions, and replaces this sentinel with the concrete "
                "ensure_perop_block(...) feature. Lowering with it still set would leave every contraction "
                "untagged and silently scalar."
            )
        raise RuntimeError(
            f"{name!r} reached the lowering pipeline unresolved. It must be consumed by "
            "runtime.backends.zephyr_model.prepare_for_lowering before lowering."
        )

    return fail


def _tile_epilogue_hygiene(mr: int | None) -> frozenset[str]:
    return frozenset({"erase_self_copy"}) if (mr or 1) > 1 else frozenset()


def register_perop_features(registry: FeatureRegistry) -> None:
    registry.register(
        ImprFeature(
            name=PEROP_BLOCK_NAME,
            action_class="PASS",
            description="request PER-CONTRACTION register blocking: derive the widest block legal for each "
            "contraction's OWN extents (and its own narrowest element width) from the prepared "
            "IR, tag each contraction, and emit one tile+vectorize arm per distinct block. "
            "Resolved by prepare_for_lowering into a concrete, table-specific feature; a "
            "sentinel, so it must never reach lowering itself. Replaces the class-wide clamps "
            "(one degenerate extent in a class otherwise forces the whole class off the vector "
            "path). Default-off; baseline byte-identical.",
            edit_pipeline=_unresolved(PEROP_BLOCK_NAME),
            schedule_replace=True,
        )
    )
    descriptions = {
        PEROP_NR_FILL_NAME: "widen each contraction's per-op N cap until its NARROWEST element fills a whole "
        "vector register at the board's VLEN, instead of every op sharing one element count. "
        "MEASURED model-dependent on the K1: 1.160x faster on spectformer int8, 1.196x slower "
        "on small_llama int8 -- because the i32 accumulator is what sets LMUL, so a wider N "
        "tile can push it from m4 to m8 and spill (decoded: 0 -> 6 accumulator spill ops). A "
        "search knob, not a default. Default-off; baseline byte-identical.",
        PEROP_MR_FILL_NAME: "derive each contraction's per-op M cap from how many accumulator rows of ITS OWN "
        "block fit the board's vector register file, instead of every op in the model "
        "sharing one hand-set number. The N axis of this question has been derived per-op "
        "for a while (VLEN- and element-width-scaled); M was a single scalar, so two ops "
        "could differ only by gcd(M) clipping a shared cap. Derived from target facts only "
        "-- the VLEN built for, the op's own element triple, and the RVV register count. A "
        "bound on the architecture, not a promise about the allocator, so it is searched "
        "rather than defaulted. Default-off; baseline byte-identical.",
    }
    for name in (PEROP_NR_FILL_NAME, PEROP_MR_FILL_NAME):
        registry.register(
            ImprFeature(
                name=name,
                action_class="KNOB",
                description=descriptions[name],
                edit_pipeline=_unresolved(name),
                implies=frozenset({PEROP_BLOCK_NAME}),
            )
        )
    for nr in (2, 4, 8, 32):
        perop_nr_sentinel(registry, nr)
    for mr in (1, 2, 8, 16):
        perop_mr_sentinel(registry, mr)


def parse_perop_nr_sentinel(name: str) -> int | None:
    if not name.startswith(PEROP_NR_SENTINEL_PREFIX):
        return None
    suffix = name[len(PEROP_NR_SENTINEL_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def parse_perop_mr_sentinel(name: str) -> int | None:
    if not name.startswith(PEROP_MR_SENTINEL_PREFIX):
        return None
    suffix = name[len(PEROP_MR_SENTINEL_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def perop_nr_sentinel(registry: FeatureRegistry, nr_cap: int) -> str:
    n = int(nr_cap)
    if n <= 0:
        raise ValueError(f"{PEROP_BLOCK_NAME}: NR cap must be positive, got {nr_cap!r}")
    name = f"{PEROP_NR_SENTINEL_PREFIX}{n}"
    if name not in registry.entries:
        registry.register(
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
                edit_pipeline=_unresolved(name),
                schedule_replace=True,
            )
        )
    return name


def perop_mr_sentinel(registry: FeatureRegistry, mr_cap: int) -> str:
    n = int(mr_cap)
    if n <= 0:
        raise ValueError(f"{PEROP_BLOCK_NAME}: MR cap must be positive, got {mr_cap!r}")
    name = f"{PEROP_MR_SENTINEL_PREFIX}{n}"
    if name not in registry.entries:
        registry.register(
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
                edit_pipeline=_unresolved(name),
                schedule_replace=True,
            )
        )
    return name


def ensure_perop_block(
    registry: FeatureRegistry,
    table,
    kc: int,
    pairs: list | tuple = (),
    vec_epilogue: bool = False,
) -> str:
    blocks = distinct_blocks(table)
    pairs = [list(x) for x in (pairs or ())]
    key = hashlib.sha1(
        (
            repr(sorted(table.items()))
            if not pairs
            else repr((sorted(table.items()), pairs, vec_epilogue))
        ).encode()
    ).hexdigest()[:12]
    name = f"{PEROP_BLOCK_NAME}_{len(blocks)}b_{kc}_{key}"
    if name in registry.entries:
        return name
    text = schedule_text(table, kc, pairs, vec_epilogue)
    registry.register(
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
            edit_schedule=lambda _text, _value=text: _value,
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(
                max((mr for mr, _nr in table.values()), default=1)
            ),
        )
    )
    return name


def register_from_name(registry: FeatureRegistry, name: str) -> bool:
    mr = parse_perop_mr_sentinel(name)
    if mr is not None:
        return perop_mr_sentinel(registry, mr) == name
    nr = parse_perop_nr_sentinel(name)
    if nr is not None:
        return perop_nr_sentinel(registry, nr) == name
    return False
