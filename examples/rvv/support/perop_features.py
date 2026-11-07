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
            description="Request per-contraction RVV register blocking from prepared IR.",
            edit_pipeline=_unresolved(PEROP_BLOCK_NAME),
            schedule_replace=True,
        )
    )
    for name in (PEROP_NR_FILL_NAME, PEROP_MR_FILL_NAME):
        registry.register(
            ImprFeature(
                name=name,
                action_class="KNOB",
                description=f"Derive per-contraction RVV register cap: {name}.",
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
                description=f"Request per-contraction RVV block with NR cap {n}.",
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
                description=f"Request per-contraction RVV block with MR cap {n}.",
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
            description=f"Per-contraction RVV block over {len(table)} geometries and {len(blocks)} blocks.",
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
