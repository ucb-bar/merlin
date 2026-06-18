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


def apply_rvv_package(pkg: "RvvPackage | str | Path", model_dir: str | Path, work: str | Path,
                      *, board: str = "spike_riscv64", harts: int = 2, arena_mb: int = 64,
                      int8_compute: bool | None = None, **kw: Any) -> dict:
    """Build the Zephyr image for ``model_dir`` using THIS package's codegen knobs.

    The package owns the RVV-specific cflags; ``_CFLAGS_COMMON`` is appended here. ``dtype_strategy``
    selects an EXISTING lowering path (``int8_w8a8`` -> ``int8_compute=True`` -> passes_quant_int;
    ``bf16_f32acc`` -> the existing bf16 f32-accumulate path) — no new lowering logic. Returns the
    ``build_app`` dict (elf, app_dir, build_dir, ram_bytes, ...); ``model.o`` lives at ``work/model.o``.
    """
    if not isinstance(pkg, RvvPackage):
        pkg = load_rvv_package(pkg)
    if int8_compute is None:
        int8_compute = pkg.is_int8
    return zm.build_app(
        model_dir, work, board=board, backend="rvv",
        rvv_schedule=pkg.schedule_text,
        cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
        int8_compute=int8_compute, arena_mb=arena_mb, cpus=max(harts, 1),
        features=frozenset(pkg.compiler_features) or None,
        **kw)
