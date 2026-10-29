"""Prepare the new complete before/actual-final memory graph obligation.

Both boundary modules are independently decoded from supplied raw IR. The old
source plan remains an untrusted hint and is checked by explicit term equality.
This function emits UNCOMPILED candidates, never a proof/qualification receipt.
"""

from __future__ import annotations

import hashlib

from .lean_graph_input_admission import parse_admitted
from .lean_graph_loops_memory_scf_after_plan import prepare_plan
from .lean_graph_loops_shape_after_projection import decode
from .lean_graph_loops_shape_capture import export
from .lean_graph_loops_shape_epoch_model import module_literal
from .lean_graph_loops_unroll import _list, _nat, _string


def prepare_whole(adapter, source_frame, raw_source, raw_final, *, parser, index_bits, max_nodes):
    prepared = prepare_plan(
        adapter, source_frame, raw_final, parser=parser, index_bits=index_bits, max_nodes=max_nodes,
    )
    source_export = export(parse_admitted(raw_source, parser, calls="shape-direct"))
    source = decode(source_export, index_bits=index_bits, max_nodes=max_nodes)
    query = "import Merlin.GraphLoopsMemorySCFAfterChain\n" + prepared["query"]
    query += f"""
namespace GraphMemoryAfterWhole
open Merlin Merlin.GraphLoopsShape
open Merlin.GraphLoopsMemorySCF.After
open GraphShapeEpochObligation GraphMemoryAfterExpansion
def independentSource : Module := {module_literal(source["module"])}
def sourceFrame : List String := {_list(source["frame"], _string)}
theorem source_term_exact : independentSource = GraphShapeObligation.actual.module := by decide
theorem source_frame_exact : sourceFrame = GraphMemoryAfterMaterialization.actual.beforeFrame := by decide
theorem source_valid : independentSource.Valid {_nat(index_bits)} := by
  rw [source_term_exact]
  exact GraphShapeObligation.valid.1
theorem source_normalized : plan.module = Epoch.normalizedModule independentSource := by
  rw [source_term_exact]
  exact GraphShapeEpochObligation.normalized_syntax
theorem source_index_absent : SourceIndexAbsent indices independentSource := by decide
theorem boundary_facts : BoundaryFacts plan kernel expansions := by decide
def actual : After.Projection :=
  {{ GraphMemoryAfterMaterialization.actual with before := expandedModule plan kernel expansions }}
theorem actual_materialized : After.check actual = true := by
  unfold actual
  rw [computed_target]
  exact GraphMemoryAfterMaterialization.accepted
theorem complete_actual_chain {{Value User : Type}} (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (finalMeaning {_nat(index_bits)} indices plan.opcodes scalar)
      independentSource kernel.function.name inputs heap user observed ↔
    Epoch.graphOutcomes plan.objects.length (finalMeaning {_nat(index_bits)} indices plan.opcodes scalar)
      GraphMemoryAfterMaterialization.actualFinal kernel.function.name inputs heap user observed :=
  checked_actual_chain independentSource {_nat(index_bits)} source_valid indices source_index_absent
    plan (of_decide_eq_true plan_valid) source_normalized kernel (of_decide_eq_true kernel_facts)
    GraphMemorySCFObligation.actual (of_decide_eq_true memory_binding) GraphMemorySCFObligation.accepted
    expansions (of_decide_eq_true expansion_checked) boundary_facts
    GraphMemoryAfterMaterialization.actualFinal actual.swaps actual.beforeFrame actual.afterFrame actual_materialized
    scalar inputs typed callers heap user observed
end GraphMemoryAfterWhole
"""
    roots = [
        *prepared["support_roots"],
        prepared["final_root"],
        *["GraphMemoryAfterWhole." + name for name in (
            "source_term_exact", "source_frame_exact", "source_valid", "source_normalized",
            "source_index_absent", "boundary_facts", "actual_materialized",
        )],
    ]
    return dict(
        prepared,
        query=query,
        final_root="GraphMemoryAfterWhole.complete_actual_chain",
        support_roots=list(dict.fromkeys(roots)),
        independent_source=source,
        source_export=source_export,
        raw_source_sha256=hashlib.sha256(raw_source.encode()).hexdigest(),
        raw_final_sha256=hashlib.sha256(raw_final.encode()).hexdigest(),
        whole_boundary_candidate_emitted=True,
        common_interpretation_candidate_emitted=True,
        local_simulation_is_premise=False,
        whole_run_equality_is_premise=False,
        lean_status="UNCOMPILED",
        source_plan_semantics_proved=False,
        complete_memory_bridge_proved=False,
        kernel_checked=False,
        qualified=False,
        pipeline_receipt_issued=False,
    )
