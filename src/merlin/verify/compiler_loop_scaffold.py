# ruff: noqa: E501
"""Existing four-boundary mathematics reused for a local actual loop leaf."""


def auxiliary(local, *, tile):
    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.loops import segment_loops

    module = parse_ir(local["after"])
    segment_loops(module, (tile,), index_bits=64)
    return ir_text(module)


def emit(local, universe, controls, *, tile=3):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.lean_graph_state_pipeline_emit import emit as pipeline_emit
    from merlin.verify.lean_graph_state_pipeline_emit import entry_arguments
    from merlin.verify.lean_graph_state_scalar_capture import project

    before, after, tiled = [parse_ir(local[key]) for key in ("before", "after", "auxiliary")]
    modules = (before, before, before, after, tiled)
    stages = [project(a, b, context=universe) for a, b in zip(modules, modules[1:])]
    capture = {**local["capture"], "stages": stages, "index_operations": controls, "tile": tile}
    query = pipeline_emit(capture, flexible_parameters=True)
    fi = len(stages[0]["functions"]) - 1
    base = "Merlin.GraphStatePipelineBulkInstance"
    query = query.replace("Merlin.GraphStatePipelineInstance", base)
    query += f"namespace {base}.F{fi}\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    query += "theorem context_actual {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
    query += "    (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :\n"
    query += "    evalProgram (M primitive external) S2 arguments world = evalProgram (M primitive external) S3 arguments world := by\n"
    query += entry_arguments(len(stages[0]["functions"][fi]["inputs"]), "unroll_actual")
    query += f"end {base}.F{fi}\n"
    pair = project(before, after, context=universe)
    return {
        **local,
        "capture": capture,
        "pair": pair,
        "query": query,
        "base": base,
        "proof_name": "context_actual",
        "source_stage": 2,
        "target_stage": 3,
    }
