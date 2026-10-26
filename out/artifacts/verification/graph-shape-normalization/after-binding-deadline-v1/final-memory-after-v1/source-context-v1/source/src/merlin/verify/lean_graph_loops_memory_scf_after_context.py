"""Prepare complete actual one-kernel-call context/roster candidate queries."""

import hashlib

from .lean_graph_loops_memory_scf_after_context_data import contexts, extend_hint
from .lean_graph_loops_memory_scf_after_context_witness import witness
from .lean_graph_loops_memory_scf_after_entries import _header
from .lean_graph_loops_memory_scf_after_whole import prepare_whole
from .lean_graph_loops_shape_model import bindings, code, function
from .lean_graph_loops_unroll import _list, _string


def wrapper_literal(row):
    return "⟨" + ", ".join((function(row["function"]), _string(row["key"]), _string(row["callee"]),
                            bindings(row["arguments"]), bindings(row["results"]), code(row["suffix"]))) + "⟩"


def _row(row):
    route = row["route"]
    if route == "caller":
        value = "(.caller wrapper" + str(row["wrapper"]) + ")"
    elif route == "unchanged":
        value = "(.unchanged " + _list(row["reachable"], _string) + ")"
    else:
        value = ".kernel"
    return "⟨" + _string(row["name"]) + ", " + value + "⟩"


def obligation(data, bits):
    wrappers = "\n".join(f"def wrapper{index} : CallerContext := {wrapper_literal(row)}" for index, row in enumerate(data["wrappers"]))
    text = f"""
namespace GraphMemoryAfterContext
open Merlin Merlin.GraphLoopsShape Merlin.GraphLoopsMemorySCF.After
open GraphShapeEpochObligation GraphMemoryAfterExpansion GraphMemoryAfterWhole
{wrappers}
def literals : LiteralOpcodes := {_list(data['literals'], lambda row: '(' + _string(row[0]) + ', ' + _string(row[1]) + ')')}
def reachable : List String := {_list(data['kernel_reachable'], _string)}
def boundary : ContextBoundary :=
  ⟨independentSource, {bits}, indices, plan, kernel, GraphMemorySCFObligation.actual, expansions,
    GraphMemoryAfterMaterialization.actualFinal, actual.swaps, actual.beforeFrame, actual.afterFrame, literals, reachable⟩
theorem boundary_facts : boundary.Facts := by
  refine ⟨source_valid, source_index_absent, of_decide_eq_true plan_valid, source_normalized,
    of_decide_eq_true kernel_facts, of_decide_eq_true memory_binding, GraphMemorySCFObligation.accepted,
    of_decide_eq_true expansion_checked, GraphMemoryAfterWhole.boundary_facts, actual_materialized, ?_, ?_, ?_⟩
  all_goals decide
def sourceHeaders : List EntryHeader := {_list(data['source_headers'], _header)}
def finalHeaders : List EntryHeader := {_list(data['final_headers'], _header)}
def rows : List ContextRow := {_list(data['rows'], _row)}
theorem roster_facts : ContextRoster boundary sourceHeaders finalHeaders rows := by decide
theorem complete_roster {{Value User : Type}} (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (row : ContextRow) (member : row ∈ rows) (inputs : List (Runtime Value)) (domain : row.Domain boundary inputs)
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (boundary.meaning scalar literal) independentSource row.name inputs heap user observed ↔
    Epoch.graphOutcomes plan.objects.length (boundary.meaning scalar literal) boundary.actualFinal row.name inputs heap user observed :=
  checked_context_roster boundary boundary_facts sourceHeaders finalHeaders rows roster_facts
    scalar literal row member inputs domain heap user observed

theorem every_definition {{Value User : Type}} (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (function : Function) (defined : function ∈ independentSource.functions) :
    ∃ row ∈ rows, row.name = function.name ∧
      ∀ inputs : List (Runtime Value), row.Domain boundary inputs →
      ∀ (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User),
      (Epoch.graphOutcomes plan.objects.length (boundary.meaning scalar literal) independentSource row.name inputs heap user observed ↔
       Epoch.graphOutcomes plan.objects.length (boundary.meaning scalar literal) boundary.actualFinal row.name inputs heap user observed) := by
  obtain ⟨row, member, name⟩ := context_every_definition boundary sourceHeaders finalHeaders rows roster_facts function defined
  exact ⟨row, member, name, fun inputs domain heap user observed =>
    complete_roster scalar literal row member inputs domain heap user observed⟩
"""
    roots = ["boundary_facts", "roster_facts", "complete_roster"]
    for index, _ in enumerate(data["wrappers"]):
        text += f"""
theorem wrapper{index}_checked : wrapper{index}.Checked boundary := by decide
def wrapper{index}SampleInputs : List (Runtime Unit) := sampleInputs plan.objects.length wrapper{index}.function.parameters
theorem wrapper{index}_sample_typed : ValuesHold wrapper{index}.function.parameters wrapper{index}SampleInputs :=
  sample_typed plan.objects.length wrapper{index}.function.parameters (by decide)
theorem wrapper{index}_sample_domain : wrapper{index}.Domain plan kernel wrapper{index}SampleInputs := by
  refine ⟨wrapper{index}_sample_typed, sample_external plan.objects.length wrapper{index}.function.parameters, ?_⟩
  intro caller member
  simp only [plan, List.mem_cons, List.not_mem_nil, or_false] at member
  subst caller
  exact ⟨plan.objects.length, Nat.lt_irrefl _, rfl⟩
theorem wrapper{index}_complete {{Value User : Type}} (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (inputs : List (Runtime Value)) (domain : wrapper{index}.Domain plan kernel inputs)
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (boundary.meaning scalar literal) independentSource wrapper{index}.function.name
      inputs heap user observed ↔
    Epoch.graphOutcomes plan.objects.length (boundary.meaning scalar literal) boundary.actualFinal wrapper{index}.function.name
      inputs heap user observed :=
  wrapper{index}.checked_boundary boundary boundary_facts wrapper{index}_checked scalar literal inputs domain heap user observed
"""
        roots += [f"wrapper{index}_" + name for name in ("checked", "sample_typed", "sample_domain", "complete")]
    return text + "\nend GraphMemoryAfterContext\n", ["GraphMemoryAfterContext." + name for name in roots]


def prepare_context(adapter, raw_source, raw_final, *, parser, index_bits, max_nodes):
    hint, source, _ = extend_hint(adapter, raw_source, parser=parser, index_bits=index_bits, max_nodes=max_nodes)
    whole = prepare_whole(hint, source["frame"], raw_source, raw_final,
                          parser=parser, index_bits=index_bits, max_nodes=max_nodes)
    data = contexts(whole, hint)
    text, roots = obligation(data, index_bits)
    witness_text, witness_roots, witness_status = witness(hint, data)
    query = "import Merlin.GraphLoopsMemorySCFAfterContextRoster\n" + whole["query"] + text + witness_text
    return dict(whole, query=query, context=data, extended_hint=hint, witness=witness_status,
                final_root="GraphMemoryAfterContext.every_definition",
                support_roots=list(dict.fromkeys([*whole["support_roots"], whole["final_root"], *roots, *witness_roots])),
                query_sha256=hashlib.sha256(query.encode()).hexdigest(), context_proved=False,
                whole_world_equality_assumed=False, repeated_kernel_calls_supported=False,
                source_extension_is_original_capture=False)
