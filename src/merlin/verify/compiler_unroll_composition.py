"""Compose genuine per-entry unroll proofs over one complete actual module pair.

Fragment generation can hide other definitions from a local proof obligation;
the final theorem restores every actual defined entry in its original order.
All fragments share the complete module's descriptor registry and index model.
"""

from copy import deepcopy
from pathlib import Path


def _fragment(capture, index, mode):
    from .lean_graph_pipeline_gate_profile_prepare import _adjacent_rosters
    from .lean_graph_pipeline_inline_emit import emit as inline_emit
    from .lean_graph_pipeline_tail_emit import emit as tail_emit
    from .lean_graph_state_pipeline_emit import emit as bulk_emit

    local = deepcopy(capture)
    for stage in local["stages"]:
        for i, function in enumerate(stage["functions"]):
            if i != index:
                function["declaration"] = True
    base = f"Merlin.CompilerUnrollComposition.Fragment{index}"
    if mode in ("inline", "empty", "unchanged"):
        query = inline_emit({**local, "stages": [local["stages"][2]]}, flexible_parameters=True)
        query = query.replace("Merlin.GraphStatePipelineInlineInstance", base)
        query = query.replace("merlin_unroll_inline_transform", f"merlin_unroll_inline_fragment_{index}")
        before_stage, after_stage, proof = 0, 1, f"{base}.F{index}.actual"
    elif mode == "tail":
        query = tail_emit({**local, "stages": [local["stages"][2]]}, flexible_parameters=True)
        query = query.replace("Merlin.GraphStatePipelineTailInstance", base)
        query = query.replace("merlin_unroll_tail_transform", f"merlin_unroll_tail_fragment_{index}")
        before_stage, after_stage, proof = 2, 3, f"{base}.F{index}.actual"
    else:
        query = bulk_emit(local, flexible_parameters=True)
        adjacent, _, _ = _adjacent_rosters({"pipeline": local}, "Merlin.GraphStatePipelineInstance.", (index,))
        query += adjacent
        query = query.replace("Merlin.GraphStatePipelineInstance", base)
        adjacent_base = f"Merlin.CompilerUnrollComposition.Adjacent{index}"
        query = query.replace("Merlin.GraphGateProfileAdjacent", adjacent_base)
        before_stage, after_stage, proof = 2, 3, f"{adjacent_base}.step2F{index}"
    return query, {"base": base, "before_stage": before_stage, "after_stage": after_stage, "proof": proof}


def prepare_query(before, after, *, factor, tile=3, index_bits=64):
    from merlin.transforms.certificates import ir_text, parse_ir
    from merlin.transforms.loops import segment_loops

    from .lean_graph_input_admission import parse_admitted
    from .lean_graph_pipeline_gate_profile_prepare import _admit
    from .lean_graph_pipeline_gate_projection import project_lean_sources
    from .lean_graph_pipeline_tail import query_roots
    from .lean_graph_state_pipeline_capture import index_operation, project_modules
    from .lean_graph_state_pipeline_emit import static_controls
    from .loop_transform_intake import ROOT, Unavailable, file_ref

    if type(index_bits) is not int or index_bits != 64 or type(factor) is not int or factor < 2:
        raise ValueError("current compiler unrolling requires index64 and integer factor at least two")
    if type(tile) is not int or tile < 2:
        raise ValueError("auxiliary proof scaffold requires integer tile at least two")
    if any(type(raw) is not str for raw in (before, after)):
        raise ValueError("complete actual module text required")
    if max(len(before.encode()), len(after.encode())) > 4 * 1024 * 1024:
        raise Unavailable("actual compiler buffer byte resource budget exceeded")
    if before == after:
        raise ValueError("actual changed compiler boundary required")
    source, target = [parse_admitted(raw, parse_ir, calls="external-only") for raw in (before, after)]
    auxiliary = target.clone()
    segment_loops(auxiliary, (tile,), index_bits=index_bits)
    modules = [source, source, source, target, auxiliary]
    for module in (source, target, auxiliary):
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
    _admit(modules)
    stages, controls = project_modules(modules)
    capture = {"stages": stages, "index_operations": controls, "factor": factor, "tile": tile, "index_bits": index_bits}
    functions = stages[2]["functions"]
    fragments, texts = [], []
    changed = 0
    for index, function in enumerate(functions):
        if function["declaration"]:
            continue
        mode = "unchanged"
        facts = {}
        if function["before"] != function["after"]:
            changed += 1
            loops = [row["loop"] for row in function["before"] if "loop" in row]
            if len(loops) != 1:
                raise ValueError("one changed source loop per entry required")
            if any(set(ty) != {"bits"} for ty in loops[0]["carries"]):
                raise ValueError("word-typed carried state required")
            lower, upper, step = static_controls(function["before"], capture, input_count=len(function["inputs"]))
            if step < 1 or upper < lower:
                raise ValueError("ordered nonnegative static source controls required")
            trips = (upper - lower + step - 1) // step
            mode = "empty" if trips == 0 else "inline" if trips <= factor else "tail" if trips % factor else "bulk"
            facts = {
                "lower": lower,
                "upper": upper,
                "step": step,
                "count": trips,
                "inputs": len(function["inputs"]),
                "carries": len(loops[0]["carries"]),
            }
        fragment, references = _fragment(capture, index, mode)
        fragments.append({"entry": index, "name": function["name"], "mode": mode, **facts, **references})
        texts.append(fragment)
    if not changed:
        raise ValueError("at least one changed source entry required")
    # Unchanged contexts can contain their own controls. Changed source bodies
    # retain the same runtime range limitation as the existing leaf bridges.
    from xdsl.dialects import func

    for fragment in fragments:
        if fragment["mode"] == "unchanged":
            continue
        actual_function = next(
            op for op in source.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == fragment["name"]
        )
        for op in actual_function.walk():
            control = index_operation(op)
            if control is not None and control["kind"] != "index-constant":
                raise ValueError("source index arithmetic needs separate runtime range evidence")
    imports = sorted(
        {
            word
            for text in texts
            for line in text.splitlines()
            if line.startswith("import ")
            for word in line.split()[1:]
        }
    )
    query = "".join(f"import {name}\n" for name in imports)
    query += (
        "\n".join("\n".join(line for line in text.splitlines() if not line.startswith("import ")) for text in texts)
        + "\n"
    )
    first = fragments[0]["base"]
    query += "namespace Merlin.GraphStatePipelineInstance\n"
    query += "open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    query += f"abbrev registry := {first}.registry\n"
    query += "def M {World Error : Type} (primitive : Prim → List Int → Int)\n"
    query += "    (external : External World Error registry) : Model Int World Error registry :=\n"
    query += f"  {first}.M primitive external\n"
    for side, stage in (("original", "before_stage"), ("final", "after_stage")):
        entries = [
            "{inputs := "
            + row["base"]
            + f".F{row['entry']}.inputs, outputs := "
            + row["base"]
            + f".F{row['entry']}.outputs, body := "
            + row["base"]
            + f".F{row['entry']}.S{row[stage]}"
            + "}"
            for row in fragments
        ]
        query += f"def {side}Entries : List Entry := [" + ",".join(entries) + "]\n"
        query += f"theorem {side}EntriesChecked : checkEntries registry {side}Entries = true := by decide\n"
    proof = "entries_nil (M primitive external)"
    for row in reversed(fragments):
        proof = f"entries_cons (M primitive external) rfl rfl ({row['proof']} primitive external) ({proof})"
    query += "theorem complete {World Error : Type} (primitive : Prim → List Int → Int)\n"
    query += "    (external : External World Error registry) : "
    query += "PreservesEntries (M primitive external) originalEntries finalEntries :=\n"
    query += "  " + proof + "\nend Merlin.GraphStatePipelineInstance\n"
    query += "namespace Merlin.GraphGateProfileAdjacent\n"
    query += "theorem complete2 {World Error : Type}\n"
    query += "    (primitive : Merlin.GraphState.Scalar.Prim → List Int → Int)\n"
    query += "    (external : Merlin.GraphState.Pipeline.External World Error "
    query += "Merlin.GraphStatePipelineInstance.registry) :\n"
    query += (
        "    Merlin.GraphState.Pipeline.PreservesEntries (Merlin.GraphStatePipelineInstance.M primitive external)\n"
    )
    query += (
        "      Merlin.GraphStatePipelineInstance.originalEntries Merlin.GraphStatePipelineInstance.finalEntries :=\n"
    )
    query += "  Merlin.GraphStatePipelineInstance.complete primitive external\nend Merlin.GraphGateProfileAdjacent\n"
    if len(query.encode()) > 4 * 1024 * 1024:
        raise Unavailable("operational proof query byte resource budget exceeded")
    parent = Path(__file__).parent
    paths = {
        Path(__file__),
        parent / "lean_graph_pipeline_inline_emit.py",
        parent / "lean_graph_pipeline_tail_emit.py",
        parent / "lean_graph_pipeline_tail.py",
        parent / "lean_graph_pipeline_gate_profile_prepare.py",
        parent / "lean_graph_pipeline_gate_projection.py",
        parent / "lean_graph_input_admission.py",
        parent / "compiler_loop_resources.py",
        parent / "compiler_loop_nested.py",
        parent / "compiler_unroll_inline_nested.py",
    }
    paths.update(parent.glob("lean_graph_state*.py"))
    sources = [file_ref(path) for path in sorted(paths)]
    sources.extend(file_ref(source.path) for source in project_lean_sources(imports))
    return {
        "query": query,
        "roots": query_roots(query),
        "sources": sources,
        "final_root": ROOT,
        "capture": {
            "mode": "entry-composition",
            "pipeline": capture,
            "fragments": fragments,
            "auxiliary_buffer": ir_text(auxiliary),
            "auxiliary_delivered": False,
        },
    }
