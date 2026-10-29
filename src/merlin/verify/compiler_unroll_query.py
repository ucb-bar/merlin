"""Current-buffer preparation shared by operational inline and tail queries.

Parsing and projection are trusted mappings. No producer replay, certificate,
saved status, or diagnostic is used to establish the generated relation.
"""

from pathlib import Path


def prepare_query(before, after, *, factor, index_bits, mode):
    from xdsl.dialects import arith

    from merlin.transforms.certificates import parse_ir

    from .lean_graph_input_admission import parse_admitted
    from .lean_graph_pipeline_gate_projection import canonical, project_lean_sources
    from .lean_graph_pipeline_tail import _admit, query_roots
    from .lean_graph_state_pipeline_capture import index_operation
    from .lean_graph_state_pipeline_emit import static_controls
    from .lean_graph_state_scalar_capture import metadata, project
    from .loop_transform_intake import Unavailable, file_ref

    if type(factor) is not int or factor < 2 or type(index_bits) is not int or index_bits != 64:
        raise ValueError("current compiler unrolling requires factor at least two and index64")
    if mode not in ("inline", "tail"):
        raise ValueError("explicit inline or tail mode required")
    if any(type(raw) is not str for raw in (before, after)):
        raise ValueError("complete actual source and result text required")
    if any(len(raw.encode()) > 4 * 1024 * 1024 for raw in (before, after)):
        raise Unavailable("actual compiler buffer byte resource budget exceeded")
    if before == after:
        raise ValueError("unchanged compiler boundaries require no proof job")
    modules = [parse_admitted(raw, parse_ir, calls="external-only") for raw in (before, after)]
    for module in modules:
        if (
            sum(
                1
                + len(op.operands)
                + len(op.results)
                + sum(len(block.args) for region in op.regions for block in region.blocks)
                for op in module.walk()
            )
            > 2048
        ):
            raise Unavailable("complete scalar/SCF syntax resource budget exceeded")
        _admit(module)
        for op in module.walk():
            flags = op.properties.get("overflowFlags")
            if flags is not None and flags != arith.IntegerOverflowAttr("none"):
                raise ValueError("flagged arithmetic outside current unroll proof domain")
            if op.properties.get("fastmath") is not None:
                raise ValueError("fast-math outside current scalar integer unroll proof domain")
    pair = project(*modules, context=tuple(modules))
    descriptors = {}
    for module in modules:
        for op in module.walk():
            control = index_operation(op)
            if control is not None:
                descriptors[canonical(metadata(op))] = control
    for op in modules[0].walk():
        control = index_operation(op)
        if control is not None and control["kind"] != "index-constant":
            raise ValueError("source index arithmetic needs separate runtime range evidence")
    controls = [
        {"identity": i, **descriptors[canonical(row["descriptor"])]}
        for i, row in enumerate(pair["registry"])
        if canonical(row["descriptor"]) in descriptors
    ]
    capture = {"stages": [pair], "index_operations": controls, "factor": factor, "index_bits": index_bits}
    changed = []
    for i, function in enumerate(pair["functions"]):
        if function["declaration"] or function["before"] == function["after"]:
            continue
        loops = [row["loop"] for row in function["before"] if "loop" in row]
        if len(loops) != 1:
            raise ValueError("one changed source loop per entry required")
        if any(set(ty) != {"bits"} for ty in loops[0]["carries"]):
            raise ValueError("word-typed carried state required")
        lower, upper, step = static_controls(function["before"], capture, input_count=len(function["inputs"]))
        if upper < lower or step <= 0:
            raise ValueError("static ordered nonnegative controls required")
        count = (upper - lower + step - 1) // step
        if mode == "inline" and count > factor:
            raise ValueError("full inline or empty source loop required")
        if mode == "tail" and (count <= factor or count % factor == 0):
            raise ValueError("nonempty bulk and remainder source loop required")
        changed.append(
            {
                "entry": i,
                "lower": lower,
                "upper": upper,
                "step": step,
                "count": count,
                "carries": len(loops[0]["carries"]),
                "inputs": len(function["inputs"]),
            }
        )
    if not changed:
        raise ValueError("at least one actual changed source loop required")
    if mode == "inline":
        from .lean_graph_pipeline_inline_emit import emit
    else:
        from .lean_graph_pipeline_tail_emit import emit
    query = emit(capture, flexible_parameters=True)
    imports = [word for line in query.splitlines() if line.startswith("import ") for word in line.split()[1:]]
    parent = Path(__file__).parent
    paths = {
        Path(__file__),
        parent / f"compiler_unroll_{mode}.py",
        parent / f"lean_graph_pipeline_{mode}_emit.py",
        parent / "lean_graph_pipeline_tail.py",
        parent / "lean_graph_input_admission.py",
        parent / "compiler_loop_nested.py",
        parent / "compiler_unroll_inline_nested.py",
        parent / "lean_graph_pipeline_gate_projection.py",
    }
    paths.update(parent.glob("lean_graph_state*.py"))
    sources = [file_ref(path) for path in sorted(paths)]
    sources.extend(file_ref(source.path) for source in project_lean_sources(imports))
    return {
        "query": query,
        "roots": query_roots(query),
        "sources": sources,
        "capture": {**capture, "mode": mode, "changed": changed},
        "final_root": f"merlin_unroll_{mode}_transform",
    }
