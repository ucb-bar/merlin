"""Kernel obligation for actual operation-level reuse→segmentation composition."""

from __future__ import annotations

from .lean_graph_loops_reuse import REUSE_ROOTS
from .lean_graph_loops_reuse import obligation as reuse_obligation
from .lean_graph_loops_unroll import _list, _nat, _string

MEMORY_SCF_ROOTS = REUSE_ROOTS + (
    "GraphMemorySCFObligation.valid",
    "GraphMemorySCFObligation.accepted",
    "GraphMemorySCFObligation.complete",
    "GraphMemorySCFObligation.chain",
    "GraphMemorySCFObligation.contextual",
    "Merlin.GraphLoopsMemorySCF.EvalPure.frame",
    "Merlin.GraphLoopsMemorySCF.Run.covers",
    "Merlin.GraphLoopsMemorySCF.Run.materializes",
    "Merlin.GraphLoopsMemorySCF.heap_run_iff_log",
    "Merlin.GraphLoopsMemorySCF.segmentation_exact_heaps",
    "Merlin.GraphLoopsMemorySCF.materializes_in_bounds",
    "Merlin.GraphLoopsMemorySCF.related_initialize_logs",
    "Merlin.GraphLoopsMemorySCF.initialization_induced",
    "Merlin.GraphLoopsMemorySCF.step_forward",
    "Merlin.GraphLoopsMemorySCF.step_backward",
    "Merlin.GraphLoopsMemorySCF.execution_forward",
    "Merlin.GraphLoopsMemorySCF.execution_backward",
    "Merlin.GraphLoopsMemorySCF.checked_complete_observations",
    "Merlin.GraphLoopsMemorySCF.execution_modes",
    "Merlin.GraphLoopsMemorySCF.checked_reuse_then_segment",
    "Merlin.GraphLoopsMemorySCF.checked_induced_atomic",
    "Merlin.GraphLoopsMemorySCF.Model.check_sound",
    "Merlin.GraphLoopsMemorySCF.Model.checked_chain",
    "Merlin.GraphLoopsMemorySCF.Model.checked_complete",
    "Merlin.GraphLoopsMemorySCF.Model.checked_context",
    "Merlin.GraphLoopsMemorySCF.Witness.uniform_heap",
    "Merlin.GraphLoopsMemorySCF.Witness.uniform_run_exists",
    "Merlin.GraphLoopsMemorySCF.Witness.uniform_run_values",
    "Merlin.GraphLoopsMemorySCF.Witness.writes_uniform",
    "Merlin.GraphLoopsMemorySCF.Model.checked_outer_control",
    "Merlin.GraphLoopsMemorySCF.Model.checked_inner_control",
    "Merlin.GraphLoopsMemorySCF.Model.checked_initializer_byte_offset",
)


def pure_op(row):
    if set(row) != {"key", "operands", "results"}:
        raise ValueError("complete explicit pure operation required")
    return "⟨" + ", ".join((_string(row["key"]), _list(row["operands"], _nat), _list(row["results"], _nat))) + "⟩"


def configuration(row):
    program, leaf = row["program"], row["program"]["leaf"]
    if set(program) != {"shape", "ivs", "outerPrefix", "leaf"} or set(leaf) != {"ops", "value", "indices"}:
        raise ValueError("complete scoped initializer program required")
    leaf_literal = (
        "⟨" + ", ".join((_list(leaf["ops"], pure_op), _nat(leaf["value"]), _list(leaf["indices"], _nat))) + "⟩"
    )
    program_literal = (
        "⟨"
        + ", ".join(
            (
                _list(program["shape"], _nat),
                _list(program["ivs"], _nat),
                _list(program["outerPrefix"], pure_op),
                leaf_literal,
            )
        )
        + "⟩"
    )
    return "⟨" + ", ".join((_nat(row["object"]), _string(row["key"]), program_literal, _nat(row["tile"]))) + "⟩"


def segment_literal(row):
    strings = {"sourceBody", "targetBody", "sourceContext", "targetContext"}
    string_lists = {"carryTypes", "sourceResults", "outerResults", "innerResults", "outerArgumentTypes"}
    naturals = {
        "targetLower",
        "targetUpper",
        "targetStride",
        "targetInnerStep",
        "outerIV",
        "strideValue",
        "upperValue",
        "originalStepValue",
        "addResult",
        "minResult",
        "sourceYieldArity",
    }
    natural_lists = {
        "sourceInitial",
        "outerInitial",
        "outerCarryArguments",
        "addOperands",
        "minOperands",
        "innerOperands",
        "innerResultValues",
        "outerYield",
    }
    if set(row) != strings | string_lists | naturals | natural_lists | {"bounds"}:
        raise ValueError("complete exact segmentation projection required")
    bounds = row["bounds"]
    if set(bounds) != {"bits", "lower", "upper", "step", "tile"}:
        raise ValueError("complete signed control bounds required")
    fields = [
        "bounds := ⟨" + ", ".join(_nat(bounds[name]) for name in ("bits", "lower", "upper", "step", "tile")) + "⟩"
    ]
    for name, render in [(name, _string) for name in sorted(strings)] + [(name, _nat) for name in sorted(naturals)]:
        fields.append(name + " := " + render(row[name]))
    for names, render in ((string_lists, _string), (natural_lists, _nat)):
        fields.extend(name + " := " + _list(row[name], render) for name in sorted(names))
    return "{ " + ", ".join(fields) + " }"


def obligation(projection):
    old = reuse_obligation(projection["reuse"])
    old = old.partition("\n")[2]  # replace only its import; new module imports this exact old root
    boundaries = []
    for row in projection["configurations"]:
        key, middle = _string(row["key"]), _string(row["middleInitializer"])
        boundaries.append(
            "⟨" + ", ".join((configuration(row), segment_literal(row["segment"]), key, middle, middle)) + "⟩"
        )
    literal = "{ reuse := GraphReuseObligation.actual, boundaries := [" + ", ".join(boundaries) + "]"
    literal += (
        ", reuseMiddle := "
        + _string(projection["middleBinding"])
        + ", segmentationMiddle := "
        + _string(projection["middleBinding"])
        + " }"
    )
    roots = "\n".join("#print axioms " + root for root in MEMORY_SCF_ROOTS[len(REUSE_ROOTS) :])
    return f"""import Merlin.GraphLoopsMemorySCFBounds
section OldReuse
{old}
end OldReuse
namespace GraphMemorySCFObligation
open Merlin Merlin.GraphLoopsMemorySCF
set_option maxHeartbeats 2000000
set_option maxRecDepth 16384
def actual : Model.Projection := {literal}
theorem valid : Model.Valid actual := by
  simp [Model.Valid, Model.Boundary.Valid, Model.Bound, Model.configs,
    Configuration.Valid, Initializer.WellFormed, Initializer.One, Initializer.Two,
    Initializer.rows, Initializer.columns, Initializer.outerIV, Initializer.innerIV,
    definitions, GraphLoopsSegment.Model.Numeric, GraphLoopsSegment.Model.Structural,
    GraphLoopsSegment.Bounds.Valid, actual, GraphReuseObligation.actual,
    GraphLoopsReuse.objectAt]
  all_goals decide
theorem accepted : Model.check actual = true := by
  apply Bool.and_eq_true_iff.mpr
  exact ⟨by simp only [valid, decide_true], GraphReuseObligation.accepted⟩
theorem complete {{Value Public : Type}} (meaning : Interpretation Value Public)
    (source middle : GraphLoopsReuse.Memory Value Public) (same : source.observed = middle.observed)
    (outcome : Public) :
    outcomes actual.reuse.objects (Model.configs actual) meaning false false actual.reuse.instructions source outcome ↔
    outcomes actual.reuse.objects (Model.configs actual) meaning true true actual.reuse.instructions middle outcome :=
  Model.checked_complete accepted meaning source middle same outcome
theorem chain {{Value Public : Type}} (meaning : Interpretation Value Public)
    (source middle : GraphLoopsReuse.Memory Value Public) (same : source.observed = middle.observed)
    (outcome : Public) :
    (outcomes actual.reuse.objects (Model.configs actual) meaning false false actual.reuse.instructions source outcome ↔
      outcomes actual.reuse.objects (Model.configs actual) meaning true false 
        actual.reuse.instructions middle outcome) ∧
    (outcomes actual.reuse.objects (Model.configs actual) meaning true false actual.reuse.instructions middle outcome ↔
      outcomes actual.reuse.objects (Model.configs actual) meaning true true 
        actual.reuse.instructions middle outcome) ∧
    (outcomes actual.reuse.objects (Model.configs actual) meaning false false actual.reuse.instructions source outcome ↔
      outcomes actual.reuse.objects (Model.configs actual) meaning true true 
        actual.reuse.instructions middle outcome) :=
  Model.checked_chain accepted meaning source middle same outcome
theorem contextual {{Value Public Outcome : Type}} (meaning : Interpretation Value Public)
    (source middle : GraphLoopsReuse.Memory Value Public) (same : source.observed = middle.observed)
    (context : (Public → Prop) → Outcome) :
    context (outcomes actual.reuse.objects (Model.configs actual) meaning false false 
      actual.reuse.instructions source) =
      context (outcomes actual.reuse.objects (Model.configs actual) meaning true true 
        actual.reuse.instructions middle) :=
  Model.checked_context accepted meaning source middle same context
end GraphMemorySCFObligation
{roots}
"""
