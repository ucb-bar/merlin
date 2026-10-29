"""Emit exact finite source-command/expanded-target binding obligations.

This is SDK-free preparation. Neither an expansion hint nor declaration text
proves its operational simulation. The final theorem is computed-expanded→actual,
not original-source→actual; flags deliberately distinguish those edges.
"""

from __future__ import annotations

from .lean_graph_loops_memory_scf_after_expansion import prepare_expansion
from .lean_graph_loops_shape_after_declarations import separate_audits
from .lean_graph_loops_shape_epoch_model import obligation as source_obligation
from .lean_graph_loops_shape_epoch_model import uniform_literal
from .lean_graph_loops_shape_epoch_plan import command, private_view
from .lean_graph_loops_shape_model import SHAPE_ROOTS, binding
from .lean_graph_loops_unroll import _list, _nat, _string


def _tiled(program):
    fields = [uniform_literal(program["uniform"])]
    fields.extend(binding(program[key]) for key in ("tile", "tileIV", "sum", "bound"))
    fields.extend(_string(program[key]) for key in ("addKey", "minKey"))
    return "⟨" + ", ".join(fields) + "⟩"


def _expansion(site, step, plan, observers):
    if site["source_position"] != step["position"] or site["instruction"] != step["instruction"]:
        raise ValueError("expansion must retain its exact source position/instruction")
    tag = site["kind"]
    views = {item["object"]: item for item in plan["privateViews"]}
    if tag == "alias":
        return ".alias " + " ".join((
            _string(step["atom"]["key"]), private_view(views[site["object"]]), private_view(views[site["owner"]]),
        ))
    if tag == "empty-release":
        return ".dropRelease " + _string(step["atom"]["key"]) + " " + private_view(views[step["instruction"]["object"]])
    if tag == "tiled-initializer":
        return ".tiled link" + str(step["initializer"]) + " " + _tiled(site["initializer"])
    if tag not in ("retained", "retained-allocation", "retained-release"):
        raise ValueError("unsupported finite expansion constructor")
    return ".keep (" + command(step, plan, observers) + ")"


def prepare_plan(adapter, source_frame, raw_final, *, parser, index_bits, max_nodes):
    prepared = prepare_expansion(
        adapter, source_frame, raw_final, parser=parser, index_bits=index_bits, max_nodes=max_nodes,
    )
    source, roots = source_obligation(adapter)
    roots = tuple(dict.fromkeys((*SHAPE_ROOTS, *roots)))
    declarations = separate_audits(source, roots, final_root="GraphShapeEpochObligation.complete_chain")
    plan = adapter["step_binding"]
    observers = {row["function"]: row for row in adapter["observers"]}
    entries = [
        _expansion(site, step, plan, observers)
        for site, step in zip(prepared["expansion"]["sites"], plan["steps"], strict=True)
    ]
    indices = _list(
        prepared["actual_final"]["model"]["index_opcodes"],
        lambda pair: "(" + _string(pair[0]) + ", ." + pair[1] + ")",
    )
    materialization = prepared["query"]
    prefix = "import Merlin.GraphLoopsShapeAfter\n"
    if not materialization.startswith(prefix):
        raise ValueError("expected exact materialization declaration header")
    source = (
        "import Merlin.GraphLoopsMemorySCFAfterPlan\n" + declarations["query"]
        + "\n" + materialization.removeprefix(prefix)
    )
    source += f"""
namespace GraphMemoryAfterExpansion
open Merlin Merlin.GraphLoopsShape GraphShapeEpochObligation
open Merlin.GraphLoopsMemorySCF.After
def indices : IndexOpcodes := {indices}
def expansions : List Expansion := [{", ".join(entries)}]
theorem expansion_checked :
    decide (ExpansionBinding {_nat(index_bits)} plan kernel indices expansions) = true := by decide
theorem all_source_commands : expansions.map Expansion.source = kernel.commands :=
  expansion_source_exact {_nat(index_bits)} plan kernel indices expansions expansion_checked
theorem computed_target : expandedModule plan kernel expansions =
    GraphMemoryAfterMaterialization.expandedTarget := by decide
theorem expanded_to_actual {{Value World : Type}} (meaning : Interpretation Value World)
    (name : String) (inputs : List (Runtime Value)) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    complete meaning (expandedModule plan kernel expansions) false name inputs before outputs after ↔
      complete meaning GraphMemoryAfterMaterialization.actualFinal false name inputs before outputs after := by
  rw [computed_target]
  exact GraphMemoryAfterMaterialization.expanded_to_actual meaning name inputs before outputs after
end GraphMemoryAfterExpansion
"""
    final_root = "GraphMemoryAfterExpansion.expanded_to_actual"
    supports = [
        *roots,
        "GraphMemoryAfterMaterialization.accepted",
        "GraphMemoryAfterMaterialization.actual_final_syntax",
        "GraphMemoryAfterExpansion.expansion_checked",
        "GraphMemoryAfterExpansion.all_source_commands",
        "GraphMemoryAfterExpansion.computed_target",
    ]
    return dict(
        prepared,
        query=source,
        final_root=final_root,
        support_roots=list(dict.fromkeys(supports)),
        finite_expansion_binding_emitted=True,
        finite_expansion_binding_kernel_checked=False,
        source_plan_semantics_proved=False,
        kernel_checked=False,
        qualified=False,
        complete_memory_bridge_proved=False,
        pipeline_receipt_issued=False,
    )
