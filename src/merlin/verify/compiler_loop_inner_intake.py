"""Pack actual selected-depth proofs into canonical compiler gate custody."""

from pathlib import Path


def prepare(buffers, *, kinds, factor, tile):
    from .compiler_loop_inner import prepare_query
    from .compiler_loop_intake import TILE, UNROLL, PreparedCompilerLoop
    from .loop_transform_intake import Settings, canonical, file_ref

    result = prepare_query(tuple(buffers), kinds=kinds, factor=factor, tile=tile)
    if kinds == (UNROLL,):
        actual = (*buffers, buffers[-1])
    elif kinds == (TILE,):
        actual = (buffers[0], buffers[0], buffers[1])
    else:
        actual = tuple(buffers)
    parent = Path(__file__).parent
    paths = {
        Path(__file__),
        parent / "compiler_loop_intake.py",
        parent / "compiler_unroll_modes_intake.py",
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
