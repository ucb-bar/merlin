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
    unobservable workload degrades to the shape-blind behavior rather than failing.

    A package may pin its block one of TWO ways, and both are adapted here. A ``microkernel`` knob
    block goes through the target-agnostic space. A HAND-FROZEN named point (the champion's
    ``accumulator_resident_wholemodel_vf``) carries its block as constants inside the feature, so
    it is translated to the equivalent caps first -- otherwise the one package anyone actually
    compiles with is the one package that cannot adapt, and a workload whose extents the frozen
    tails do not fit fails to lower with a masked-parallel-dim PipelineError."""
    from ..kernels.shapes import contraction_shapes
    from .registry import _resolve_features
    shapes = contraction_shapes(Path(model_dir) / "model.mlir")
    if not shapes:
        return list(pkg.compiler_features)
    feats = _resolve_features(pkg.knobs, pkg.manifest, shapes=shapes)
    return _adapt_frozen_points(feats, shapes, target=str(pkg.manifest.get("target", "rvv")))


def _adapt_frozen_points(feats: list[str], shapes, *, target: str) -> list[str]:
    """Re-derive a hand-frozen register block ONLY for the op classes where it cannot lower.

    Deliberately minimal. Adapting a class whose frozen block already lowers would change the
    emitted code for workloads that work today -- including the model the block was tuned on,
    whose measurements other people's results depend on. So each contraction op class is checked
    against the measured predicate independently, the frozen block is kept wherever it holds, and
    only a class that would mask a parallel dim (and therefore fail to lower, not merely run slow)
    is replaced by the largest block legal for its own extents.

    Emits a per-op-class feature when anything changed, and returns the list unchanged otherwise,
    so a workload the frozen point already fits compiles byte-identically.
    """
    from ..llvmlower.frozen_blocks import frozen_block_caps, frozen_block_per_class
    from .from_strategy import _rvv_best_block, _rvv_blocking_lowers

    out: list[str] = []
    for name in feats:
        caps, frozen = frozen_block_caps(name), frozen_block_per_class(name)
        if caps is None or frozen is None:
            out.append(name)
            continue
        chosen: dict[str, tuple[int, int] | None] = {}
        changed = False
        for op, block in frozen.items():
            ext = [(s.parallel[-2], s.parallel[-1]) for s in shapes
                   if s.op == op and len(s.parallel) >= 2]
            if not ext or all(_rvv_blocking_lowers(block[0], block[1], m, n) for m, n in ext):
                chosen[op] = block
                continue
            best = _rvv_best_block(int(caps["MR"]), int(caps["NR"]), ext)
            # A 1-lane N block is not a vectorization: it buys no lanes, and vectorizing that tile
            # emits a parallel-dim-free `vector.contract` (vector<1xT> dot into a scalar) that no
            # lowering strategy matches, so the build dies late. Leave the class un-tiled instead —
            # its contractions go through convert-linalg-to-loops (scalar), which is correct, and the
            # OTHER class keeps its vectors. Reported by the caller, never silent.
            chosen[op] = None if best[1] <= 1 else best
            changed = True
        if not changed:
            out.append(name)
            continue
        from ..llvmlower.impr_features import ensure_v3_perop_microkernel
        mm, bmm = chosen["linalg.matmul"], chosen["linalg.batch_matmul"]
        if mm is None and bmm is None:      # nothing vectorizable -> keep the pinned point as-is
            out.append(name)
            continue
        out.append(ensure_v3_perop_microkernel(
            *(mm or (None, None)), *(bmm or (None, None)), int(caps["KC"])))
    seen: set[str] = set()
    return [f for f in out if not (f in seen or seen.add(f))]


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
