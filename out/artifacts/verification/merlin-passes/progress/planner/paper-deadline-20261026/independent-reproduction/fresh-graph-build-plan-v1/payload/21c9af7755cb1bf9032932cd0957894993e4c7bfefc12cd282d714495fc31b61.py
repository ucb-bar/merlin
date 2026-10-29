"""Emit an all-supported-entry successor to the selected-entry graph candidate.

Both raw boundary modules supply complete rosters. Each unchanged entry requires
finite call-closed correspondence, not merely an equal local body. All emitted
Lean remains UNCOMPILED; parser/projection and native fidelity are separate.
"""

import hashlib

from .lean_graph_loops_memory_scf_after_entries_data import roster
from .lean_graph_loops_memory_scf_after_entries_witness import literal_witnesses
from .lean_graph_loops_memory_scf_after_whole import prepare_whole
from .lean_graph_loops_shape_model import function, kind
from .lean_graph_loops_unroll import _list, _string


def _header(row):
    return "⟨" + ", ".join((
        _string(row["name"]), _string(row["visibility"]), str(row["declaration"]).lower(),
        _list(row["parameters"], kind), _list(row["returned"], kind),
    )) + "⟩"


def _row(row):
    route = ".kernel" if row["route"] == "kernel" else "(.unchanged " + _list(row["reachable"], _string) + ")"
    return "⟨" + _string(row["name"]) + ", " + route + "⟩"


def _definitions(data):
    return f"""
namespace GraphMemoryAfterEntries
open Merlin Merlin.GraphLoopsShape Merlin.GraphLoopsMemorySCF.After
open GraphShapeEpochObligation GraphMemoryAfterExpansion GraphMemoryAfterWhole
def sourceHeaders : List EntryHeader := {_list(data['source_headers'], _header)}
def finalHeaders : List EntryHeader := {_list(data['final_headers'], _header)}
def rows : List EntryRow := {_list(data['rows'], _row)}
def literals : LiteralOpcodes := {_list(data['literals'], lambda row: '(' + _string(row[0]) + ', ' + _string(row[1]) + ')')}
def kernelReachable : List String := {_list(data['kernel_reachable'], _string)}
theorem roster_facts : EntryRosterFacts independentSource GraphMemoryAfterMaterialization.actualFinal
    sourceHeaders finalHeaders plan kernel expansions rows := by decide
theorem literal_source : LiteralModuleFacts literals independentSource := by decide
theorem literal_final : LiteralModuleFacts literals GraphMemoryAfterMaterialization.actualFinal := by decide
theorem kernel_support : KernelLiteralSupport independentSource GraphMemoryAfterMaterialization.actualFinal
    kernel kernelReachable literals := by decide
"""


def _theorems(bits):
    return f"""
theorem complete_roster {{Value User : Type}} (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (row : EntryRow) (member : row ∈ rows) (inputs : List (Runtime Value))
    (domain : EntryDomain independentSource plan kernel row inputs)
    (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (entryMeaning {bits} indices plan.opcodes scalar literals literal)
      independentSource row.name inputs heap user observed ↔
    Epoch.graphOutcomes plan.objects.length (entryMeaning {bits} indices plan.opcodes scalar literals literal)
      GraphMemoryAfterMaterialization.actualFinal row.name inputs heap user observed :=
  checked_entry_roster independentSource {bits} source_valid indices source_index_absent
    plan (of_decide_eq_true plan_valid) source_normalized kernel (of_decide_eq_true kernel_facts)
    GraphMemorySCFObligation.actual (of_decide_eq_true memory_binding) GraphMemorySCFObligation.accepted
    expansions (of_decide_eq_true expansion_checked) boundary_facts
    GraphMemoryAfterMaterialization.actualFinal actual.swaps actual.beforeFrame actual.afterFrame actual_materialized
    sourceHeaders finalHeaders rows roster_facts literals kernelReachable literal_source literal_final kernel_support
    scalar literal row member inputs domain heap user observed

theorem every_definition {{Value User : Type}} (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (function : Function) (defined : function ∈ independentSource.functions) :
    ∃ row ∈ rows, row.name = function.name ∧
      ∀ inputs : List (Runtime Value), EntryDomain independentSource plan kernel row inputs →
      ∀ (heap : Heap Value) (user : User) (observed : Epoch.Observation Value User),
      (Epoch.graphOutcomes plan.objects.length (entryMeaning {bits} indices plan.opcodes scalar literals literal)
        independentSource row.name inputs heap user observed ↔
       Epoch.graphOutcomes plan.objects.length (entryMeaning {bits} indices plan.opcodes scalar literals literal)
        GraphMemoryAfterMaterialization.actualFinal row.name inputs heap user observed) := by
  obtain ⟨row, member, name⟩ := roster_has_every_definition independentSource GraphMemoryAfterMaterialization.actualFinal
    sourceHeaders finalHeaders plan kernel expansions rows roster_facts function defined
  exact ⟨row, member, name, fun inputs domain heap user observed =>
    complete_roster scalar literal row member inputs domain heap user observed⟩
"""


def _domains(source, data):
    text, roots = [], []
    for number, (body, row) in enumerate(zip(source["functions"], data["rows"], strict=True)):
        name = "entry" + str(number)
        text.append(f"""
def {name} : Function := {function(body)}
def {name}Row : EntryRow := {_row(row)}
def {name}Inputs : List (Runtime Unit) := sampleInputs plan.objects.length {name}.parameters
theorem {name}_found : functionAt independentSource {_string(body['name'])} = some {name} := by decide
theorem {name}_member : {name}Row ∈ rows := by decide
theorem {name}_plain : Plain {name}.parameters := by decide
theorem {name}_typed : ValuesHold {name}.parameters {name}Inputs :=
  sample_typed plan.objects.length {name}.parameters {name}_plain
theorem {name}_wrong_arity {{Value World : Type}} (meaning : Interpretation Value World)
    (inputs : List (Runtime Value)) (wrong : inputs.length ≠ {name}.parameters.length)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    ¬ complete meaning independentSource false {name}.name inputs before outputs after :=
  wrong_arity_no_success meaning independentSource {name}.name {name} {name}_found inputs wrong before outputs after
theorem {name}_external : ∀ input ∈ {name}Inputs, ExternalInput plan.objects.length input :=
  sample_external plan.objects.length {name}.parameters
""")
        if row["route"] == "kernel":
            text.append(f"""
theorem {name}_kernel_parameters : {name}.parameters = kernel.function.parameters := by decide
theorem {name}_callers : Epoch.CallersValid plan.objects.length plan.callers
    (assign emptyEnv kernel.function.parameters {name}Inputs) := by
  intro caller member
  simp only [plan, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl
  exact ⟨plan.objects.length, Nat.lt_irrefl _, rfl⟩
theorem {name}_domain : EntryDomain independentSource plan kernel {name}Row {name}Inputs := by
  refine ⟨{name}_external, ?_, {name}_callers⟩
  rw [← {name}_kernel_parameters]
  exact {name}_typed
""")
            roots += [name + "_kernel_parameters", name + "_callers"]
        else:
            text.append(f"""
theorem {name}_domain : EntryDomain independentSource plan kernel {name}Row {name}Inputs :=
  ⟨{name}_external, {name}, {name}_found, {name}_typed⟩
""")
        roots += [name + suffix for suffix in ("_found", "_member", "_plain", "_typed", "_wrong_arity", "_external", "_domain")]
    return "".join(text), roots


def prepare_entries(adapter, source_frame, raw_source, raw_final, *, parser, index_bits, max_nodes):
    whole = prepare_whole(adapter, source_frame, raw_source, raw_final,
                          parser=parser, index_bits=index_bits, max_nodes=max_nodes)
    source = whole["independent_source"]["module"]
    final = whole["actual_final"]["model"]["module"]
    data = roster(whole["source_export"], whole["actual_final"]["actual_export"], source, final,
                  adapter["step_binding"]["function"])
    # Declarations retain full data/refusal; never silently zip them out of the witness roster.
    domains, domain_roots = ("", []) if any(row["declaration"] for row in data["source_headers"]) else _domains(source, data)
    witnesses, witness_roots, witness_refusals = ("", [], []) if not domain_roots else literal_witnesses(source, index_bits)
    query = "import Merlin.GraphLoopsMemorySCFAfterEntriesWitness\n" + whole["query"]
    query += _definitions(data) + _theorems(index_bits) + domains + witnesses + "\nend GraphMemoryAfterEntries\n"
    roots = [*whole["support_roots"], whole["final_root"], *[
        "GraphMemoryAfterEntries." + name for name in (
            "roster_facts", "literal_source", "literal_final", "kernel_support", "complete_roster", *domain_roots, *witness_roots,
        )
    ]]
    return dict(
        whole, query=query, entry_roster=data, support_roots=list(dict.fromkeys(roots)),
        final_root="GraphMemoryAfterEntries.every_definition",
        query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        domain_witnesses_emitted=bool(domain_roots), execution_witness_roots=witness_roots,
        execution_witness_refusals=witness_refusals, all_entries_proved=False,
        native_failure_preservation_claimed=False, divergence_preservation_claimed=False,
        scalar_literal_model="shared partial effect-free relation; numeric values/native fidelity unproved",
    )
