"""Apply an RVV package's codegen knobs to a workload build, via the existing build_app seam.

This is the plug-back-in point: an isolated :class:`RvvPackage` supplies its transform schedule
and cflags into ``build_app(rvv_schedule=..., cflags_override=...)`` — WITHOUT touching the
``pipeline.RVV_TRANSFORM_SCHEDULE`` / ``RVV_CFLAGS`` module defaults. ``apply_rvv_package(hand_v0)``
is therefore byte-identical to today's ``build_app(backend="rvv")`` (asserted by test_rvv_package).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..runtime.backends import zephyr_model as zm
from .registry import RvvPackage, load_rvv_package


def _shape_adapt_api():
    from ..llvmlower.backend_spi import load_vector_codegen

    return load_vector_codegen("rvv").shape_adapt_api()


def _harts_split_shapes(shapes, harts: int) -> list:
    return _shape_adapt_api()._harts_split_shapes(shapes, harts)


def shape_adapted_features(pkg: RvvPackage, model_dir: str | Path, *, harts: int = 1) -> list[str]:
    return _shape_adapt_api().shape_adapted_features(pkg, model_dir, harts=harts)


def _schedule_pinned_blocks(schedule_text: str) -> dict[str, tuple[int, int]]:
    return _shape_adapt_api()._schedule_pinned_blocks(schedule_text)


def blocking_risks(pkg: RvvPackage, model_dir: str | Path) -> list[str]:
    return _shape_adapt_api().blocking_risks(pkg, model_dir)


def _adapt_frozen_points(feats: list[str], shapes, *, target: str) -> list[str]:
    return _shape_adapt_api()._adapt_frozen_points(feats, shapes, target=target)


def apply_rvv_package(
    pkg: RvvPackage | str | Path,
    model_dir: str | Path,
    work: str | Path,
    *,
    board: str | None = None,
    harts: int = 2,
    arena_mb: int = 64,
    int8_compute: bool | None = None,
    shape_adapt: bool = False,
    **kw: Any,
) -> dict:
    """Build the Zephyr image for ``model_dir`` using THIS package's codegen knobs.

    The package owns the RVV-specific cflags; ``_CFLAGS_COMMON`` is appended here. ``dtype_strategy``
    selects an EXISTING lowering path (``int8_w8a8`` -> ``int8_compute=True`` -> passes_quant_int;
    ``bf16_f32acc`` -> the existing bf16 f32-accumulate path) — no new lowering logic. Returns the
    ``build_app`` dict (elf, app_dir, build_dir, ram_bytes, ...); ``model.o`` lives at ``work/model.o``.

    ``shape_adapt=True`` (default OFF -> byte-identical to today) resolves the package's
    ``microkernel`` block against THIS workload's contraction shapes instead of pinning it — see
    :func:`shape_adapted_features`. Opt-in because it changes the emitted schedule for workloads the
    pinned block does not fit, which is a measurement-visible change.
    """
    if board is None:
        raise ValueError("apply_rvv_package requires a board from MERLIN_BOARD_CATALOG")
    if not isinstance(pkg, RvvPackage):
        pkg = load_rvv_package(pkg)
    if int8_compute is None:
        int8_compute = pkg.is_int8
    feats = shape_adapted_features(pkg, model_dir) if shape_adapt else list(pkg.compiler_features)
    return zm.build_app(
        model_dir,
        work,
        board=board,
        backend="rvv",
        rvv_schedule=pkg.schedule_text,
        cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
        int8_compute=int8_compute,
        arena_mb=arena_mb,
        cpus=max(harts, 1),
        features=frozenset(feats) or None,
        **kw,
    )
