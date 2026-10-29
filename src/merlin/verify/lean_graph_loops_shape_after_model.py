"""Serialize separate actual before/after modules for the new Lean pair check.

Emission is not acceptance. The proof candidates remain uncompiled until native
runtime authorization; a saved generated query is never a pipeline receipt.
"""

from .lean_graph_loops_shape_epoch_model import module_literal
from .lean_graph_loops_unroll import _list, _nat, _string

ROOTS = (
    "Merlin.GraphLoopsShape.After.swap_twice",
    "Merlin.GraphLoopsShape.After.swap_injective",
    "Merlin.GraphLoopsShape.After.put_swapped",
    "Merlin.GraphLoopsShape.After.assign_swapped",
    "Merlin.GraphLoopsShape.After.eval_swapped",
    "Merlin.GraphLoopsShape.After.function_swapped",
    "Merlin.GraphLoopsShape.After.invoke_swapped",
    "Merlin.GraphLoopsShape.After.complete_permuted",
    "Merlin.GraphLoopsShape.After.check_sound",
    "Merlin.GraphLoopsShape.After.checked_complete",
    "Merlin.GraphLoopsShape.After.checked_context",
    "GraphShapeAfterObligation.accepted",
    "GraphShapeAfterObligation.final_syntax",
    "GraphShapeAfterObligation.complete",
    "GraphShapeAfterObligation.contextual",
)


def obligation(pair):
    swaps = _list(pair["swaps"], lambda item: "(" + _nat(item[0]) + ", " + _nat(item[1]) + ")")
    source = f"""import Merlin.GraphLoopsShapeAfter
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 8000000
open Merlin Merlin.GraphLoopsShape
namespace GraphShapeAfterObligation
def actualBefore : Module := {module_literal(pair["before"])}
def actualAfter : Module := {module_literal(pair["after"])}
def actual : After.Projection :=
  ⟨{_nat(pair["bits"])}, actualBefore, actualAfter, {swaps},
    {_list(pair["beforeFrame"], _string)}, {_list(pair["afterFrame"], _string)}⟩
theorem accepted : After.check actual = true := by decide
theorem final_syntax : After.permuteModule actual.swaps actualAfter = Epoch.normalizedModule actualBefore :=
  (After.check_sound accepted).2.2.2
theorem complete {{Value World : Type}} (meaning : Interpretation Value World)
    (name : String) (inputs : List (Runtime Value)) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    GraphLoopsShape.complete meaning actualBefore false name inputs before outputs after ↔
      GraphLoopsShape.complete meaning actualAfter false name inputs before outputs after :=
  After.checked_complete accepted meaning name inputs before outputs after
theorem contextual {{Value World Observation : Type}} (meaning : Interpretation Value World)
    (context : (String → RegionMeaning Value World) → Observation) :
    context (GraphLoopsShape.complete meaning actualBefore false) =
      context (GraphLoopsShape.complete meaning actualAfter false) := After.checked_context accepted meaning context
end GraphShapeAfterObligation
"""
    return source + "\n".join("#print axioms " + root for root in ROOTS) + "\n", ROOTS
