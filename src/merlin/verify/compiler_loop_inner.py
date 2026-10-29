"""Actual selected-depth loop proofs with checked inherited capture facts.

Only complete actual buffers are accepted. Intermediate body records are exact
nested subprograms of their complete admitted projection; parent joins restore
all enclosing loops and every defined entry before the compiler may consume it.
"""

from pathlib import Path


def _changed_depth(source, target):
    """Find the first changed loop through identical actual ancestor headers."""
    from xdsl.dialects import func, scf

    from .lean_graph_state_scalar_capture import metadata

    left = list(source.body.block.ops)
    right = list(target.body.block.ops)
    if len(left) != len(right):
        raise ValueError("complete actual function roster differs")
    for function, other in zip(left, right, strict=True):
        if type(function) is not func.FuncOp or type(other) is not func.FuncOp or metadata(function) != metadata(other):
            raise ValueError("complete actual function interface differs")
        if not function.body.blocks:
            continue
        block, actual = function.body.block, other.body.block
        context = {block: actual}
        context.update(zip(block.args, actual.args, strict=True))
        depth = 0
        while True:
            source_ops, target_ops = list(block.ops), list(actual.ops)
            changed = False
            for i, operation in enumerate(source_ops):
                if i >= len(target_ops):
                    if type(operation) is scf.ForOp:
                        return depth
                    raise ValueError("actual enclosing region truncated")
                candidate = target_ops[i]
                trial = dict(context)
                if operation.is_structurally_equivalent(candidate, trial):
                    context = trial
                    continue
                if type(operation) is not scf.ForOp:
                    raise ValueError("actual non-loop prefix changed")
                retained = (
                    type(candidate) is scf.ForOp
                    and metadata(operation) == metadata(candidate)
                    and len(operation.operands) == len(candidate.operands)
                    and all(
                        context.get(value) == result
                        for value, result in zip(operation.operands, candidate.operands, strict=True)
                    )
                )
                if not retained:
                    return depth
                context.update(zip(operation.results, candidate.results, strict=True))
                context[operation.body.block] = candidate.body.block
                context.update(zip(operation.body.block.args, candidate.body.block.args, strict=True))
                block, actual = operation.body.block, candidate.body.block
                depth += 1
                changed = True
                break
            if changed:
                continue
            if len(source_ops) != len(target_ops):
                raise ValueError("actual enclosing region grew without a selected source loop")
            break
    raise ValueError("no actual selected loop change found")


def _capture(stage):
    """Keep complete buffers once and omit repeated model/live-SSA objects."""
    from .compiler_loop_inner_stages import code_type

    def scope(row):
        return {
            **{
                key: value
                for key, value in row.items()
                if key not in ("frame", "projection", "local", "before", "after")
            },
            "actual_frame_types": [code_type(value.type) for value in row["frame"]],
        }

    leaf = stage["leaf"]
    local = leaf["local"]
    local_keys = (
        "before",
        "after",
        "mode",
        "controls",
        "count",
        "wrapper_name",
        "auxiliary",
        "source_stage",
        "target_stage",
        "proof_name",
    )
    decoded = {key: local[key] for key in local_keys if key in local}
    decoded["proof_factor"] = local["capture"]["factor"]
    return {
        **{key: value for key, value in stage.items() if key not in ("leaf", "ancestors")},
        "leaf": {**scope(leaf), "local": decoded},
        "ancestors": [scope(row) for row in stage["ancestors"]],
    }


def prepare_query(buffers, *, kinds, factor=2, tile=3, index_bits=64):
    from merlin.transforms.certificates import parse_ir

    from . import (
        compiler_loop_context_emit,
        compiler_loop_inner_build,
        compiler_loop_inner_emit,
        compiler_loop_inner_stages,
    )
    from .compiler_loop_intake import TILE, UNROLL
    from .lean_graph_input_admission import parse_admitted
    from .lean_graph_pipeline_gate_profile_prepare import _admit
    from .lean_graph_pipeline_gate_projection import project_lean_sources
    from .lean_graph_state_pipeline_capture import index_operation
    from .loop_transform_intake import ROOT, Unavailable, file_ref

    if kinds not in ((UNROLL,), (TILE,), (UNROLL, TILE)):
        raise ValueError("explicit actual compiler loop boundary roster required")
    if (
        type(factor) is not int
        or factor < 2
        or type(tile) is not int
        or tile < 2
        or type(index_bits) is not int
        or index_bits != 64
    ):
        raise ValueError("explicit index64 and integer factor/tile at least two required")
    if type(buffers) is not tuple or len(buffers) != len(kinds) + 1 or any(type(raw) is not str for raw in buffers):
        raise ValueError("complete actual adjacent compiler buffers required")
    if any(len(raw.encode()) > 4 * 1024 * 1024 for raw in buffers):
        raise Unavailable("actual compiler buffer byte resource budget exceeded")
    if any(a == b for a, b in zip(buffers, buffers[1:])):
        raise ValueError("every actual requested compiler boundary must change")
    modules = [parse_admitted(raw, parse_ir, calls="external-only") for raw in buffers]
    for module in modules:
        work = sum(
            1
            + len(op.operands)
            + len(op.results)
            + sum(len(block.args) for region in op.regions for block in region.blocks)
            for op in module.walk()
        )
        if work > 2048:
            raise Unavailable("complete scalar/SCF syntax resource budget exceeded")
    _admit(modules)
    from .lean_graph_state_scalar_capture import project

    for before, after in zip(modules, modules[1:]):
        project(before, after, context=tuple(modules))
    for op in modules[0].walk():
        control = index_operation(op)
        if control is not None and control["kind"] != "index-constant":
            raise ValueError("source index arithmetic needs separate runtime range evidence")
    stages = []
    depths = []
    for i, kind in enumerate(kinds):
        depth = _changed_depth(modules[i], modules[i + 1])
        current = compiler_loop_inner_stages.split(
            buffers[i],
            buffers[i + 1],
            factor=factor,
            depth=depth,
            transform="tile" if kind == TILE else "unroll",
            tile=tile,
        )
        for row in current:
            row["boundary"] = 3 if kind == TILE else 2
        stages.extend(current)
        depths.append({"kind": kind, "actual_selected_depth": depth + 1})
    result = compiler_loop_inner_build.build(
        stages,
        buffers[0],
        buffers[-1],
        compiler_loop_inner_stages,
        compiler_loop_context_emit,
        compiler_loop_inner_emit,
        middle=buffers[1] if len(kinds) == 2 else None,
        transform="tile" if kinds == (TILE,) else "unroll",
    )
    query = result["query"]
    if len(query.encode()) > 4 * 1024 * 1024:
        raise Unavailable("operational proof query byte resource budget exceeded")
    imports = [word for line in query.splitlines() if line.startswith("import ") for word in line.split()[1:]]
    parent = Path(__file__).parent
    names = (
        "compiler_loop_inner.py",
        "compiler_loop_inner_stages.py",
        "compiler_loop_inner_build.py",
        "compiler_loop_inner_emit.py",
        "compiler_loop_context.py",
        "compiler_loop_context_emit.py",
        "compiler_loop_context_chain.py",
        "compiler_loop_scaffold.py",
        "compiler_loop_tile_emit.py",
        "compiler_loop_atomic.py",
        "compiler_loop_nested.py",
        "compiler_loop_resources.py",
        "compiler_unroll_inline_nested.py",
        "lean_graph_pipeline_inline_emit.py",
        "lean_graph_pipeline_tail_emit.py",
        "lean_graph_pipeline_tail.py",
        "lean_graph_input_admission.py",
        "lean_graph_pipeline_gate_profile_prepare.py",
        "lean_graph_pipeline_gate_projection.py",
    )
    paths = {parent / name for name in names}
    paths.update(parent.glob("lean_graph_state*.py"))
    sources = [file_ref(path) for path in sorted(paths)]
    sources.extend(file_ref(ref.path) for ref in project_lean_sources(imports))
    capture = {
        "mode": "actual-selected-depth-context",
        "depths": depths,
        "stages": [_capture(stage) for stage in stages],
        "projection": result["projection"],
    }
    from .loop_transform_intake import canonical

    if len(canonical(capture).encode()) > 32 * 1024 * 1024:
        raise Unavailable("complete actual context capture byte resource budget exceeded")
    return {"query": query, "roots": result["roots"], "sources": sources, "final_root": ROOT, "capture": capture}
