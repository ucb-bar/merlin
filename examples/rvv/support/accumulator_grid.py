"""RVV accumulator-resident v1/v2 register-tile feature grids."""

from __future__ import annotations

from merlin.llvmlower.feature_registry import FeatureRegistry, ImprFeature

from .active_features import (
    _accumulator_resident_pipeline,
    _accumulator_resident_pre_schedule,
    _accumulator_resident_v2_pipeline,
    _accumulator_resident_v3_bmmpad_pipeline,
    _accumulator_resident_v3_mrpad_pipeline,
    _accumulator_resident_v3_mrpad_pre_schedule,
    _accumulator_resident_v3_pipeline,
    _accumulator_resident_v3_pre_schedule,
    _zero_attr,
)

_GRID = ((4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16))


def _tile_epilogue_hygiene(mr: int | None) -> frozenset[str]:
    return frozenset({"erase_self_copy"}) if (mr or 1) > 1 else frozenset()


def register_accumulator_grids(registry: FeatureRegistry) -> None:
    for mr, nr, kc in _GRID:
        name = (
            "accumulator_resident_microkernel"
            if (mr, nr, kc) == (4, 16, 16)
            else f"accum_resident_{mr}_{nr}_{kc}"
        )
        registry.register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"RVV post-bufferization accumulator hoist with tile {mr}x{nr}x{kc}.",
                edit_pipeline=_accumulator_resident_pipeline,
                edit_schedule=lambda _text, _mr=mr, _nr=nr, _kc=kc: (
                    _accumulator_resident_pre_schedule(_mr, _nr, _kc)
                ),
                schedule_replace=True,
            )
        )
    for name, kwargs in (
        ("accumulator_resident_ntail", {"NR_bmm": 8}),
        ("accumulator_resident_mtail", {"MR_mm": 1}),
    ):
        registry.register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"RVV accumulator-resident schedule with {name} tail clamp.",
                edit_pipeline=_accumulator_resident_pipeline,
                edit_schedule=lambda _text, _kw=kwargs: (
                    _accumulator_resident_pre_schedule(4, 16, 16, **_kw)
                ),
                schedule_replace=True,
            )
        )
    for mr, nr, kc in _GRID:
        name = (
            "accumulator_resident_v2"
            if (mr, nr, kc) == (4, 16, 16)
            else f"accum_resident_v2_{mr}_{nr}_{kc}"
        )
        registry.register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"RVV tensor-level accumulator hoist with tile {mr}x{nr}x{kc}.",
                edit_pipeline=_accumulator_resident_v2_pipeline,
                edit_schedule=lambda _text, _mr=mr, _nr=nr, _kc=kc: (
                    _accumulator_resident_pre_schedule(_mr, _nr, _kc)
                ),
                schedule_replace=True,
            )
        )
    for mr, nr, kc in _GRID:
        name = (
            "accumulator_resident_microkernel_v3"
            if (mr, nr, kc) == (4, 16, 16)
            else f"accum_resident_v3_{mr}_{nr}_{kc}"
        )
        registry.register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"RVV accumulator-resident vfmacc.vf tile {mr}x{nr}x{kc}.",
                edit_pipeline=_accumulator_resident_v3_pipeline,
                edit_schedule=lambda _text, _mr=mr, _nr=nr, _kc=kc: (
                    _accumulator_resident_v3_pre_schedule(_mr, _nr, _kc)
                ),
                schedule_replace=True,
                implies=_tile_epilogue_hygiene(mr),
            )
        )
    registry.register(
        ImprFeature(
            name="accumulator_resident_wholemodel_vf_bmmpad",
            action_class="PASS",
            description="RVV accumulator-resident vfmacc.vf with BMM tail padding.",
            edit_pipeline=_accumulator_resident_v3_bmmpad_pipeline,
            edit_schedule=lambda _text: _accumulator_resident_v3_pre_schedule(
                4, 16, 16, NR_bmm=8, MR_mm=1, pad_bmm=True
            ),
            schedule_replace=True,
        )
    )
    registry.register(
        ImprFeature(
            name="accumulator_resident_wholemodel_vf_mr4",
            action_class="PASS",
            description="RVV accumulator-resident vfmacc.vf with MR=4 register block.",
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=lambda _text: _accumulator_resident_v3_pre_schedule(
                4, 16, 16, NR_bmm=8, MR_mm=4
            ),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(4),
        )
    )
    registry.register(
        ImprFeature(
            name="accumulator_resident_wholemodel_vf_mrpad",
            action_class="PASS",
            description="RVV accumulator-resident vfmacc.vf with M-tail padding.",
            edit_pipeline=_accumulator_resident_v3_mrpad_pipeline,
            edit_schedule=lambda _text: _accumulator_resident_v3_mrpad_pre_schedule(
                4, 16, 16, NR_bmm=8
            ),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(4),
        )
    )
    for mr, nr, nb in (
        (4, 16, 8),
        (1, 16, 8),
        (1, 32, 8),
        (1, 64, 16),
        (2, 16, 8),
        (8, 16, 8),
        (4, 32, 8),
        (4, 64, 16),
        (8, 32, 16),
        (8, 64, 16),
    ):
        ensure_mrpad_for_elem_types(
            registry, "i8", "i8", "i32", MR=mr, NR=nr, NR_bmm=nb
        )


def ensure_mrpad_for_elem_types(
    registry: FeatureRegistry,
    a: str,
    b: str,
    c: str,
    MR: int = 4,
    NR: int = 16,
    NR_bmm: int = 8,
) -> str:
    types = (str(a), str(b), str(c))
    MR, NR, NR_bmm = int(MR), int(NR), int(NR_bmm)
    if min(MR, NR, NR_bmm) <= 0:
        raise ValueError(f"MR/NR/NR_bmm must be positive, got {(MR, NR, NR_bmm)}")
    name = "accumulator_resident_wholemodel_vf_mrpad_" + "_".join(types)
    name += f"_mr{MR}_nr{NR}_nb{NR_bmm}"
    if name in registry.entries:
        return name
    for t in types:
        _zero_attr(t)
    registry.register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"accumulator_resident_wholemodel_vf_mrpad at MR={MR}/NR={NR}/NR_bmm={NR_bmm} "
                f"with padding values typed for a {types[0]}x{types[1]}->{types[2]} "
                "contraction. Same recipe and same tail rule; only the pad literals differ, "
                "because a wrongly-typed pad value is a transform-interpreter error and "
                "therefore a silent whole-model scalar fallback."
            ),
            edit_pipeline=_accumulator_resident_v3_mrpad_pipeline,
            edit_schedule=lambda _text, _ty=types, _mr=MR, _nr=NR, _nb=NR_bmm: (
                _accumulator_resident_v3_mrpad_pre_schedule(
                    _mr, _nr, 16, NR_bmm=_nb, elem_types=_ty
                )
            ),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(4),
        )
    )
    return name
