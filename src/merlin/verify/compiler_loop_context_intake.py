"""Pack full actual TILE and unroll/TILE context queries for canonical gates."""

from pathlib import Path


def prepare(buffers, *, kinds, factor, tile):
    from merlin.verify.compiler_loop_context import prepare_chain, prepare_tile_query
    from merlin.verify.compiler_loop_intake import TILE, UNROLL, PreparedCompilerLoop
    from merlin.verify.loop_transform_intake import Settings, canonical, file_ref

    if kinds == (TILE,):
        result = prepare_tile_query(*buffers, tile=tile)
        actual = (buffers[0], buffers[0], buffers[1])
    elif kinds == (UNROLL, TILE):
        result = prepare_chain(*buffers, factor=factor, tile=tile)
        actual = buffers
    else:
        raise ValueError("explicit current TILE or unroll/TILE boundary required")
    parent = Path(__file__).parent
    paths = {
        Path(__file__),
        parent / "compiler_loop_intake.py",
        parent / "loop_transform_gate.py",
        parent / "loop_transform_intake.py",
        parent / "compiler_transforms.py",
        parent / "gate_checks.py",
        parent / "gate_worker.py",
        parent / "pass_validation.py",
        parent / "lean/Merlin/GraphStateAudit.lean",
    }
    paths.update((parent.parent / "transforms").glob("*.py"))
    paths.update((parent.parent / "common").glob("owned_*.py"))
    sources = {row["path"]: row for row in result["sources"]}
    sources.update({str(path.resolve()): file_ref(path) for path in paths})
    return PreparedCompilerLoop(
        tuple(actual),
        Settings(factor=factor, tile=tile),
        kinds,
        canonical(result["capture"]),
        result["query"],
        tuple(result["roots"]),
        canonical([sources[key] for key in sorted(sources)]),
    )
