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


def shape_adapted_features(pkg: "RvvPackage", model_dir: str | Path) -> list[str]:
    """``pkg``'s compiler features re-resolved against the CONTRACTIONS ``model_dir`` actually has.

    This is the seam where a package and a workload first meet, so it is where a register block
    stops being an abstract knob and becomes a claim about specific extents. A package pins one
    ``microkernel`` block; a whole model carries many matmul shapes, and a block that masks a
    parallel dim of any one of them fails to lower on the int8 path (and degrades ~34x on fp32) —
    see ``from_strategy._rvv_blocking_lowers`` for the measured predicate. Re-resolving here reads
    the pinned block as an upper BOUND and derives the largest legal one per op class.

    Returns the package's features unchanged when the model has no readable contractions, so an
    unobservable workload degrades to the shape-blind behavior rather than failing."""
    from ..kernels.shapes import contraction_shapes
    from .registry import _resolve_features
    shapes = contraction_shapes(Path(model_dir) / "model.mlir")
    if not shapes:
        return list(pkg.compiler_features)
    return _resolve_features(pkg.knobs, pkg.manifest, shapes=shapes)


def apply_rvv_package(pkg: "RvvPackage | str | Path", model_dir: str | Path, work: str | Path,
                      *, board: str = "spike_riscv64", harts: int = 2, arena_mb: int = 64,
                      int8_compute: bool | None = None, shape_adapt: bool = False,
                      **kw: Any) -> dict:
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
    if not isinstance(pkg, RvvPackage):
        pkg = load_rvv_package(pkg)
    if int8_compute is None:
        int8_compute = pkg.is_int8
    feats = shape_adapted_features(pkg, model_dir) if shape_adapt else list(pkg.compiler_features)
    return zm.build_app(
        model_dir, work, board=board, backend="rvv",
        rvv_schedule=pkg.schedule_text,
        cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
        int8_compute=int8_compute, arena_mb=arena_mb, cpus=max(harts, 1),
        features=frozenset(feats) or None,
        **kw)
