"""Bind the independent actual shape middle to the existing epoch adapter.

The resulting query checks an ordinary actual-after term under bijective SSA
renaming. The terminal memory-SCF program is still matched by the old Python
projector: this module never advertises that final program as independently
interpreted by Lean. Preparation does not launch Lean or issue a receipt.
"""

from __future__ import annotations

from .lean_graph_input_admission import parse_admitted
from .lean_graph_loops_reuse_capture import export as memory_export
from .lean_graph_loops_shape_after import prepare_pair
from .lean_graph_loops_shape_after_model import obligation as pair_obligation
from .lean_graph_loops_shape_epoch_model import obligation as epoch_obligation
from .lean_graph_loops_shape_epoch_observer import bind_observers
from .lean_graph_loops_shape_epoch_projection import bind_memory, project_adapter
from .lean_graph_loops_shape_epoch_steps import bind_steps
from .lean_graph_loops_shape_model import SHAPE_ROOTS

ROOTS = (
    "Merlin.GraphLoopsShape.After.actual_middle_complete",
    "Merlin.GraphLoopsShape.After.actual_middle_observations",
    "Merlin.GraphLoopsShape.After.actual_middle_epoch_chain",
    "Merlin.GraphLoopsShape.After.checked_source_epoch_chain",
    "GraphShapeAfterEpochObligation.source_binding",
    "GraphShapeAfterEpochObligation.middle_binding",
    "GraphShapeAfterEpochObligation.actual_middle_complete",
    "GraphShapeAfterEpochObligation.actual_middle_epoch_blueprint",
    "GraphShapeAfterEpochObligation.actual_source_epoch_blueprint",
)


def obligation(adapter, pair):
    """Old plan generation is untrusted; actual syntax equalities are goals."""
    old, old_roots = epoch_obligation(adapter)
    actual, pair_roots = pair_obligation(pair)
    # All imports precede all commands. Keep old query/theorem meanings intact.
    actual = actual.removeprefix("import Merlin.GraphLoopsShapeAfter\n")
    extra = """
namespace GraphShapeAfterEpochObligation
open Merlin Merlin.GraphLoopsShape GraphShapeEpochObligation
theorem source_binding : GraphShapeAfterObligation.actualBefore = GraphShapeObligation.actual.module := by rfl
theorem middle_binding : plan.module = After.permuteModule
    GraphShapeAfterObligation.actual.swaps GraphShapeAfterObligation.actualAfter := by
  rw [(After.check_sound GraphShapeAfterObligation.accepted).2.2.2, source_binding]
  exact normalized_syntax
theorem actual_middle_complete {Value User : Type} (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : GraphLoopsReuse.Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    complete (Epoch.graphMeaning plan.opcodes scalar) GraphShapeAfterObligation.actualAfter false
      kernel.function.name inputs ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
        kernel.instructions (Epoch.entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = Epoch.clockWorld plan.objects.length (Epoch.finalClock 0 kernel.commands) after :=
  After.actual_middle_complete GraphShapeAfterObligation.actualAfter GraphShapeAfterObligation.actual.swaps
    plan (of_decide_eq_true plan_valid) middle_binding kernel (of_decide_eq_true kernel_facts)
    scalar inputs typed callers heap user outputs world
theorem actual_middle_epoch_blueprint {Value User : Type} (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap candidateHeap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar)
      GraphShapeAfterObligation.actualAfter kernel.function.name inputs heap user observed ↔
      Epoch.epochOutcomes plan kernel scalar true true
        ⟨(Epoch.entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ observed :=
  After.actual_middle_epoch_chain GraphShapeAfterObligation.actualAfter GraphShapeAfterObligation.actual.swaps
    plan (of_decide_eq_true plan_valid) middle_binding kernel (of_decide_eq_true kernel_facts)
    GraphMemorySCFObligation.actual (of_decide_eq_true memory_binding) GraphMemorySCFObligation.accepted
    scalar inputs typed callers heap candidateHeap user observed
theorem actual_source_epoch_blueprint {Value User : Type} (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap candidateHeap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar)
      GraphShapeAfterObligation.actualBefore kernel.function.name inputs heap user observed ↔
      Epoch.epochOutcomes plan kernel scalar true true
        ⟨(Epoch.entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ observed :=
  After.checked_source_epoch_chain GraphShapeAfterObligation.actual GraphShapeAfterObligation.accepted
    plan (of_decide_eq_true plan_valid) middle_binding kernel (of_decide_eq_true kernel_facts)
    GraphMemorySCFObligation.actual (of_decide_eq_true memory_binding) GraphMemorySCFObligation.accepted
    scalar inputs typed callers heap candidateHeap user observed
end GraphShapeAfterEpochObligation
"""
    query = "import Merlin.GraphLoopsShapeAfterEpoch\n" + old + "\n" + actual + extra
    query += "\n".join("#print axioms " + root for root in ROOTS) + "\n"
    return query, tuple(dict.fromkeys((*SHAPE_ROOTS, *old_roots, *pair_roots, *ROOTS)))


def prepare_chain(source, normalized, reused, segmented, *, parser, index_bits, max_nodes):
    pair = prepare_pair(source, normalized, parser=parser, index_bits=index_bits, max_nodes=max_nodes)
    modules = [parse_admitted(raw, parser, calls="shape-direct") for raw in (normalized, reused, segmented)]
    memory = {
        key: memory_export(module, calls="shape-direct")
        for key, module in zip(("before_export", "middle_export", "after_export"), modules, strict=True)
    }
    adapter = project_adapter(pair["before_export"], pair["after_export"], index_bits=index_bits, max_nodes=max_nodes)
    adapter = bind_memory(adapter, pair["after_export"], memory, index_bits=index_bits, max_nodes=max_nodes)
    adapter = bind_steps(bind_observers(adapter), pair["after_export"])
    query, roots = obligation(adapter, pair["model"])
    return dict(
        adapter=adapter,
        pair=pair["model"],
        query=query,
        expected_roots=list(roots),
        actual_shape_after_independently_decoded=True,
        actual_memory_final_independently_interpreted=False,
        memory_final_python_matcher_is_trusted=True,
        lean_invoked=False,
        lean_accepted=False,
        qualified=False,
        pipeline_receipt_issued=False,
    )
