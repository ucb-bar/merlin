"""Pack current inline/tail queries into the canonical compiler consumer.

All-entry operational theorems are reused verbatim. The canonical aliases name
the actual unroll boundary, with no auxiliary segmentation in this route.
"""

from pathlib import Path

from .loop_transform_intake import ROOT, Settings, Unavailable, canonical, file_ref


def prepare(buffers, *, factor, tile):
    from . import compiler_unroll_inline, compiler_unroll_tail
    from .compiler_loop_intake import UNROLL, PreparedCompilerLoop

    selected = None
    for mode, module in (("inline", compiler_unroll_inline), ("tail", compiler_unroll_tail)):
        try:
            result = module.prepare_query(*buffers, factor=factor, index_bits=64)
        except ValueError:
            continue
        selected = mode, result
        break
    if selected is None:
        from .compiler_unroll_composition import prepare_query

        try:
            result = prepare_query(*buffers, factor=factor, tile=tile)
        except ValueError:
            return None
        selected = "entry-composition", result
    mode, result = selected
    namespace = "Merlin.GraphStatePipeline" + ("Inline" if mode == "inline" else "Tail") + "Instance"
    statement = f"""{{World Error : Type}}
    (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int)
    (external : Merlin.GraphState.Pipeline.External World Error {namespace}.registry) :
    Merlin.GraphState.Pipeline.PreservesEntries ({namespace}.M primitive external)
      {namespace}.originalEntries {namespace}.finalEntries"""
    query = (
        result["query"]
        + f"""
namespace Merlin.GraphStatePipelineInstance
theorem complete {statement} := {namespace}.complete primitive external
end Merlin.GraphStatePipelineInstance
namespace Merlin.GraphGateProfileAdjacent
theorem complete2 {statement} := Merlin.GraphStatePipelineInstance.complete primitive external
end Merlin.GraphGateProfileAdjacent
"""
    )
    if mode == "entry-composition":
        query = result["query"]
    if len(query.encode()) > 4 * 1024 * 1024:
        raise Unavailable("operational proof query byte resource budget exceeded")
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
    roots = tuple(dict.fromkeys((*result["roots"], ROOT, "Merlin.GraphGateProfileAdjacent.complete2")))
    return PreparedCompilerLoop(
        (*buffers, buffers[-1]),
        Settings(factor=factor, tile=tile),
        (UNROLL,),
        canonical(result["capture"]),
        query,
        roots,
        canonical([sources[key] for key in sorted(sources)]),
    )
