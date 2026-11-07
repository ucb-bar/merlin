"""Name-derived RVV accumulator-resident register-block features."""

from __future__ import annotations

import re

from merlin.llvmlower.feature_registry import FeatureRegistry, ImprFeature

from .active_features import (
    _accumulator_resident_v3_kblocked_pre_schedule,
    _accumulator_resident_v3_pipeline,
    _accumulator_resident_v3_pre_schedule,
    _accumulator_resident_v3_scalable_pre_schedule,
    _accumulator_resident_v3_unrolled_pre_schedule,
    scalable_lanes,
)


def _hygiene(mr: int | None) -> frozenset[str]:
    return frozenset({"erase_self_copy"}) if (mr or 1) > 1 else frozenset()


def _register(
    registry: FeatureRegistry,
    name: str,
    schedule,
    mr: int | None,
) -> str:
    if name not in registry.entries:
        registry.register(
            ImprFeature(
                name=name,
                action_class="PASS",
                description=f"RVV accumulator-resident register-block point {name}.",
                edit_pipeline=_accumulator_resident_v3_pipeline,
                edit_schedule=lambda _text, _schedule=schedule: _schedule(),
                schedule_replace=True,
                implies=_hygiene(mr),
            )
        )
    return name


def ensure_v3_microkernel(registry: FeatureRegistry, mr: int, nr: int, kc: int) -> str:
    name = (
        "accumulator_resident_microkernel_v3"
        if (mr, nr, kc) == (4, 16, 16)
        else f"accum_resident_v3_{mr}_{nr}_{kc}"
    )
    return _register(
        registry, name, lambda: _accumulator_resident_v3_pre_schedule(mr, nr, kc), mr
    )


def ensure_v3_kblocked_microkernel(
    registry: FeatureRegistry, mr: int, nr: int, kc: int
) -> str:
    return _register(
        registry,
        f"accum_resident_v3kb_{mr}_{nr}_{kc}",
        lambda: _accumulator_resident_v3_kblocked_pre_schedule(mr, nr, kc),
        mr,
    )


def ensure_v3_unrolled_microkernel(
    registry: FeatureRegistry, mr: int, nr: int, kc: int
) -> str:
    return _register(
        registry,
        f"accum_resident_v3u_{mr}_{nr}_{kc}",
        lambda: _accumulator_resident_v3_unrolled_pre_schedule(mr, nr, kc),
        mr,
    )


def ensure_v3_scalable_microkernel(
    registry: FeatureRegistry, mr: int, nr: int, kc: int
) -> str:
    scalable_lanes(nr)
    return _register(
        registry,
        f"accum_resident_v3vl_{mr}_{nr}_{kc}",
        lambda: _accumulator_resident_v3_scalable_pre_schedule(mr, nr, kc),
        mr,
    )


def ensure_v3_perop_microkernel(
    registry: FeatureRegistry,
    mr_mm: int | None,
    nr_mm: int | None,
    mr_bmm: int | None,
    nr_bmm: int | None,
    kc: int,
) -> str:
    skip_mm = mr_mm is None or nr_mm is None
    skip_bmm = mr_bmm is None or nr_bmm is None
    if skip_mm and skip_bmm:
        raise ValueError(
            "ensure_v3_perop_microkernel: at least one op class must be claimed "
            "(both blocks None vectorizes nothing — use the scalar backend instead)"
        )
    tag_mm = "x_x" if skip_mm else f"{mr_mm}_{nr_mm}"
    tag_bmm = "x_x" if skip_bmm else f"{mr_bmm}_{nr_bmm}"
    name = f"accum_resident_v3p_{tag_mm}_{tag_bmm}_{kc}"
    return _register(
        registry,
        name,
        lambda: _accumulator_resident_v3_pre_schedule(
            mr_bmm or 1,
            nr_mm or 1,
            kc,
            NR_bmm=nr_bmm or 1,
            MR_mm=mr_mm or 1,
            skip_mm=skip_mm,
            skip_bmm=skip_bmm,
        ),
        None if skip_mm else mr_mm,
    )


def unclaimed_op_classes(feature: str) -> list[str]:
    if not feature.startswith("accum_resident_v3p_"):
        return []
    parts = feature[len("accum_resident_v3p_") :].split("_")
    if len(parts) != 5:
        return []
    return [
        cls
        for i, cls in enumerate(("linalg.matmul", "linalg.batch_matmul"))
        if parts[2 * i] == "x"
    ]


_POINT = re.compile(r"^accum_resident_(v3|v3kb|v3u|v3vl)_(\d+)_(\d+)_(\d+)$")


def register_from_name(registry: FeatureRegistry, name: str) -> bool:
    match = _POINT.fullmatch(name)
    if match is not None:
        kind = match.group(1)
        mr, nr, kc = (int(match.group(i)) for i in (2, 3, 4))
        make = {
            "v3": ensure_v3_microkernel,
            "v3kb": ensure_v3_kblocked_microkernel,
            "v3u": ensure_v3_unrolled_microkernel,
            "v3vl": ensure_v3_scalable_microkernel,
        }[kind]
        try:
            return make(registry, mr, nr, kc) == name
        except ValueError:
            return False
    if not name.startswith("accum_resident_v3p_"):
        return False
    parts = name[len("accum_resident_v3p_") :].split("_")
    if len(parts) != 5:
        return False
    try:
        mr_mm, nr_mm, mr_bmm, nr_bmm = (
            None if part == "x" else int(part) for part in parts[:4]
        )
        kc = int(parts[4])
        return (
            ensure_v3_perop_microkernel(registry, mr_mm, nr_mm, mr_bmm, nr_bmm, kc)
            == name
        )
    except ValueError:
        return False
