"""Shared Transform schedules for explicit parallel contraction dimensions.

This builder emits text only: it selects no target, runtime, provider, hardware
feature or numerical permission. Callers supply a worker count or already checked
per-operation chunks. Reductions are never divided by the class-wide schedule.
The explicit ``tile_aligned`` compatibility preset retains its historical tiles;
its sixteen-column value is a schedule choice, never an inferred hardware fact.
"""

from __future__ import annotations

from .contract import class_token, par_tag_for

PARALLEL_ENTRY = "__transform_parallel_main"

_PARALLEL_DIM_NUM_THREADS = {
    # op -> {dim: num_threads list}. A 0 means "do not tile this dim".
    "linalg.matmul": {"m": "[{n}]", "n": "[0, {n}]"},
    "linalg.batch_matmul": {"b": "[{n}]", "m": "[0, {n}]", "n": "[0, 0, {n}]"},
}


def parallel_transform_schedule(
    n_harts: int,
    *,
    matmul_dim: str = "n",
    batch_matmul_dim: str = "b",
    chunks: list | None = None,
    tile_aligned: bool = False,
) -> str:
    """Pipeline design rationale: see docs/design/llvm_pipeline_rationale.md."""
    if n_harts < 2:
        raise ValueError(f"parallel schedule needs n_harts >= 2, got {n_harts}")
    if chunks is not None:
        return _perop_parallel_schedule(chunks)
    if tile_aligned:
        body = [
            '    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 '
            ": (!transform.any_op) -> !transform.any_op\n"
            "    %matmul_loop, %matmul_tiled = transform.structured.tile_using_forall "
            "%matmul tile_sizes [0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
            '    %batch_matmul = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 '
            ": (!transform.any_op) -> !transform.any_op\n"
            "    %batch_matmul_loop, %batch_matmul_tiled = transform.structured.tile_using_forall "
            "%batch_matmul tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)",
        ]
        return (
            "module attributes {transform.with_named_sequence} {\n"
            f"  transform.named_sequence @{PARALLEL_ENTRY}"
            "(%arg0: !transform.any_op {transform.readonly}) {\n" + "\n".join(body) + "\n    transform.yield\n  }\n}\n"
        )
    body = []
    for op, dim in (("linalg.matmul", matmul_dim), ("linalg.batch_matmul", batch_matmul_dim)):
        choices = _PARALLEL_DIM_NUM_THREADS[op]
        if dim not in choices:
            raise ValueError(
                f"{op}: parallel dim {dim!r} not in {sorted(choices)} (the reduction dim K is never tileable)"
            )
        tag = op.split(".")[-1]
        body.append(
            f'    %{tag} = transform.structured.match ops{{["{op}"]}} in %arg0 '
            f": (!transform.any_op) -> !transform.any_op\n"
            f"    %{tag}_loop, %{tag}_tiled = transform.structured.tile_using_forall "
            f"%{tag} num_threads {choices[dim].format(n=n_harts)} "
            f": (!transform.any_op) -> (!transform.any_op, !transform.any_op)"
        )
    return (
        "module attributes {transform.with_named_sequence} {\n"
        f"  transform.named_sequence @{PARALLEL_ENTRY}"
        "(%arg0: !transform.any_op {transform.readonly}) {\n" + "\n".join(body) + "\n"
        "    transform.yield\n"
        "  }\n"
        "}\n"
    )


def _perop_parallel_schedule(chunks: list) -> str:
    """The PER-OP multicore library: one ``tile_using_forall`` arm per distinct (class, tile).

    Matched by the ``merlin.par_*`` attribute, not by op name, for the same reason the block arms
    match ``merlin.blk_*``: two contractions of the same class legitimately want different splits,
    and a name match cannot tell them apart. Uses ``tile_sizes`` rather than ``num_threads`` because
    the tile is the thing that has to be exact — merlin chose one that divides the extent, so every
    chunk is STATIC and equal, and the package schedule below sees the same geometry it would have
    seen at 1 hart.

    An empty ``chunks`` (a hart count was asked for, but no contraction has a dim that splits into
    equal block-legal pieces) yields a schedule with no arms: the module reaches the package schedule
    exactly as the serial build does and the image runs on one core. Said rather than silently
    produced -- see :func:`lower_to_llvm_ir`, which prints it.
    """
    body = []
    for i, (op, tiles) in enumerate(chunks):
        tiles = tuple(int(t) for t in tiles)
        h = f"p{i}{class_token(op)}"
        body.append(
            f"    %{h} = transform.structured.match attributes{{{par_tag_for(op, tiles)}}} in %arg0 "
            f": (!transform.any_op) -> !transform.any_op\n"
            f"    %{h}_loop, %{h}_tiled = transform.structured.tile_using_forall "
            f"%{h} tile_sizes [{', '.join(str(t) for t in tiles)}] "
            f": (!transform.any_op) -> (!transform.any_op, !transform.any_op)"
        )
    return (
        "module attributes {transform.with_named_sequence} {\n"
        f"  transform.named_sequence @{PARALLEL_ENTRY}"
        "(%arg0: !transform.any_op {transform.readonly}) {\n"
        + ("\n".join(body) + "\n" if body else "")
        + "    transform.yield\n"
        "  }\n"
        "}\n"
    )
