import WholeFinal
import Merlin.GraphLoopsMemorySCFAfterEntriesWitness
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 8000000
open Merlin Merlin.GraphLoopsShape
namespace GraphMemoryAfterEntries
open Merlin Merlin.GraphLoopsShape Merlin.GraphLoopsMemorySCF.After
open GraphShapeEpochObligation GraphMemoryAfterExpansion GraphMemoryAfterWhole
def sourceHeaders : List EntryHeader := [⟨"kernel", "<default-public>", false, [(.scalar "i32"), (.scalar "i32"), (.view ⟨[1], 0, [1], "i32", 0⟩)], [(.scalar "i32"), .index]⟩, ⟨"touch", "<default-public>", false, [(.scalar "i32"), (.view ⟨[1], 0, [1], "i32", 0⟩)], [(.scalar "i32")]⟩, ⟨"untouched", "<default-public>", false, [(.scalar "i32")], [(.scalar "i32")]⟩]
def finalHeaders : List EntryHeader := [⟨"kernel", "<default-public>", false, [(.scalar "i32"), (.scalar "i32"), (.view ⟨[1], 0, [1], "i32", 0⟩)], [(.scalar "i32"), .index]⟩, ⟨"touch", "<default-public>", false, [(.scalar "i32"), (.view ⟨[1], 0, [1], "i32", 0⟩)], [(.scalar "i32")]⟩, ⟨"untouched", "<default-public>", false, [(.scalar "i32")], [(.scalar "i32")]⟩]
def rows : List EntryRow := [⟨"kernel", .kernel⟩, ⟨"touch", (.unchanged ["touch"])⟩, ⟨"untouched", (.unchanged ["untouched"])⟩]
def literals : LiteralOpcodes := [("{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operand_arity\":0,\"properties\":[[\"value\",\"1 : i32\"]],\"result_types\":[\"i32\"]}", "i32")]
def kernelReachable : List String := ["kernel", "touch"]
theorem roster_facts : EntryRosterFacts independentSource GraphMemoryAfterMaterialization.actualFinal
    sourceHeaders finalHeaders plan kernel expansions rows := by decide
theorem literal_source : LiteralModuleFacts literals independentSource := by decide
theorem literal_final : LiteralModuleFacts literals GraphMemoryAfterMaterialization.actualFinal := by decide
theorem kernel_support : KernelLiteralSupport independentSource GraphMemoryAfterMaterialization.actualFinal
    kernel kernelReachable literals := by decide

theorem complete_roster {Value User : Type} (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (row : EntryRow) (member : row ∈ rows) (inputs : List (Runtime Value))
    (domain : EntryDomain independentSource plan kernel row inputs)
    (heap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (entryMeaning 64 indices plan.opcodes scalar literals literal)
      independentSource row.name inputs heap user observed ↔
    Epoch.graphOutcomes plan.objects.length (entryMeaning 64 indices plan.opcodes scalar literals literal)
      GraphMemoryAfterMaterialization.actualFinal row.name inputs heap user observed :=
  checked_entry_roster independentSource 64 source_valid indices source_index_absent
    plan (of_decide_eq_true plan_valid) source_normalized kernel (of_decide_eq_true kernel_facts)
    GraphMemorySCFObligation.actual (of_decide_eq_true memory_binding) GraphMemorySCFObligation.accepted
    expansions (of_decide_eq_true expansion_checked) boundary_facts
    GraphMemoryAfterMaterialization.actualFinal actual.swaps actual.beforeFrame actual.afterFrame actual_materialized
    sourceHeaders finalHeaders rows roster_facts literals kernelReachable literal_source literal_final kernel_support
    scalar literal row member inputs domain heap user observed

theorem every_definition {Value User : Type} (scalar : Epoch.BinaryMeaning Value) (literal : LiteralMeaning Value)
    (function : Function) (defined : function ∈ independentSource.functions) :
    ∃ row ∈ rows, row.name = function.name ∧
      ∀ inputs : List (Runtime Value), EntryDomain independentSource plan kernel row inputs →
      ∀ (heap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User),
      (Epoch.graphOutcomes plan.objects.length (entryMeaning 64 indices plan.opcodes scalar literals literal)
        independentSource row.name inputs heap user observed ↔
       Epoch.graphOutcomes plan.objects.length (entryMeaning 64 indices plan.opcodes scalar literals literal)
        GraphMemoryAfterMaterialization.actualFinal row.name inputs heap user observed) := by
  obtain ⟨row, member, name⟩ := roster_has_every_definition independentSource GraphMemoryAfterMaterialization.actualFinal
    sourceHeaders finalHeaders plan kernel expansions rows roster_facts function defined
  exact ⟨row, member, name, fun inputs domain heap user observed =>
    complete_roster scalar literal row member inputs domain heap user observed⟩

def entry0 : Function := ⟨"kernel", [⟨0, (.scalar "i32"), none⟩, ⟨1, (.scalar "i32"), none⟩, ⟨2, (.view ⟨[1], 0, [1], "i32", 0⟩), none⟩], [(.scalar "i32"), .index], (.constant ⟨3, .index, (some 0)⟩ 0 (.constant ⟨4, .index, (some 1)⟩ 1 (.constant ⟨5, .index, (some 2)⟩ 2 (.constant ⟨6, .index, (some 3)⟩ 3 (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.alloc\",\"operand_arity\":0,\"properties\":[[\"alignment\",\"16 : i64\"],[\"operandSegmentSizes\",\"array<i32: 0, 0>\"]],\"result_types\":[\"memref<2x3xi32>\"]}" [] [⟨7, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩] (.cast ⟨7, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨8, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ (.cast ⟨8, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨9, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ (.dimension ⟨9, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨3, .index, (some 0)⟩ ⟨10, .index, (some 2)⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ 0 2 (.dimension ⟨9, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨4, .index, (some 1)⟩ ⟨11, .index, (some 3)⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ 1 3 (.region "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"scf.for\",\"operand_arity\":3,\"properties\":[],\"result_types\":[]}" [⟨3, .index, (some 0)⟩, ⟨5, .index, (some 2)⟩, ⟨4, .index, (some 1)⟩] [⟨12, .index, none⟩] [] (.region "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"scf.for\",\"operand_arity\":3,\"properties\":[],\"result_types\":[]}" [⟨3, .index, (some 0)⟩, ⟨6, .index, (some 3)⟩, ⟨4, .index, (some 1)⟩] [⟨13, .index, none⟩] [] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.store\",\"operand_arity\":4,\"properties\":[],\"result_types\":[]}" [⟨0, (.scalar "i32"), none⟩, ⟨9, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨12, .index, none⟩, ⟨13, .index, none⟩] [] (.returned [])) (.returned [])) (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.load\",\"operand_arity\":3,\"properties\":[],\"result_types\":[\"i32\"]}" [⟨9, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨3, .index, (some 0)⟩, ⟨3, .index, (some 0)⟩] [⟨14, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.addi\",\"operand_arity\":2,\"properties\":[[\"overflowFlags\",\"#arith.overflow<none>\"]],\"result_types\":[\"i32\"]}" [⟨14, (.scalar "i32"), none⟩, ⟨0, (.scalar "i32"), none⟩] [⟨15, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.store\",\"operand_arity\":4,\"properties\":[],\"result_types\":[]}" [⟨15, (.scalar "i32"), none⟩, ⟨9, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨3, .index, (some 0)⟩, ⟨3, .index, (some 0)⟩] [] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.load\",\"operand_arity\":3,\"properties\":[],\"result_types\":[\"i32\"]}" [⟨9, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨3, .index, (some 0)⟩, ⟨3, .index, (some 0)⟩] [⟨16, (.scalar "i32"), none⟩] (.dimension ⟨9, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨3, .index, (some 0)⟩ ⟨17, .index, (some 2)⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ 0 2 (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.call\",\"operand_arity\":2,\"properties\":[[\"callee\",\"@touch\"]],\"result_types\":[\"i32\"]}" [⟨16, (.scalar "i32"), none⟩, ⟨2, (.view ⟨[1], 0, [1], "i32", 0⟩), none⟩] [⟨18, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.dealloc\",\"operand_arity\":1,\"properties\":[],\"result_types\":[]}" [⟨7, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩] [] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.alloc\",\"operand_arity\":0,\"properties\":[[\"alignment\",\"16 : i64\"],[\"operandSegmentSizes\",\"array<i32: 0, 0>\"]],\"result_types\":[\"memref<2x3xi32>\"]}" [] [⟨19, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩] (.cast ⟨19, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨20, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ (.cast ⟨20, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨21, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ (.dimension ⟨21, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨3, .index, (some 0)⟩ ⟨22, .index, (some 2)⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ 0 2 (.dimension ⟨21, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨4, .index, (some 1)⟩ ⟨23, .index, (some 3)⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ 1 3 (.region "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"scf.for\",\"operand_arity\":3,\"properties\":[],\"result_types\":[]}" [⟨3, .index, (some 0)⟩, ⟨5, .index, (some 2)⟩, ⟨4, .index, (some 1)⟩] [⟨24, .index, none⟩] [] (.region "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"scf.for\",\"operand_arity\":3,\"properties\":[],\"result_types\":[]}" [⟨3, .index, (some 0)⟩, ⟨6, .index, (some 3)⟩, ⟨4, .index, (some 1)⟩] [⟨25, .index, none⟩] [] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.store\",\"operand_arity\":4,\"properties\":[],\"result_types\":[]}" [⟨1, (.scalar "i32"), none⟩, ⟨21, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨24, .index, none⟩, ⟨25, .index, none⟩] [] (.returned [])) (.returned [])) (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.load\",\"operand_arity\":3,\"properties\":[],\"result_types\":[\"i32\"]}" [⟨21, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨3, .index, (some 0)⟩, ⟨3, .index, (some 0)⟩] [⟨26, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.addi\",\"operand_arity\":2,\"properties\":[[\"overflowFlags\",\"#arith.overflow<none>\"]],\"result_types\":[\"i32\"]}" [⟨26, (.scalar "i32"), none⟩, ⟨1, (.scalar "i32"), none⟩] [⟨27, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.store\",\"operand_arity\":4,\"properties\":[],\"result_types\":[]}" [⟨27, (.scalar "i32"), none⟩, ⟨21, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨3, .index, (some 0)⟩, ⟨3, .index, (some 0)⟩] [] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.load\",\"operand_arity\":3,\"properties\":[],\"result_types\":[\"i32\"]}" [⟨21, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩, ⟨3, .index, (some 0)⟩, ⟨3, .index, (some 0)⟩] [⟨28, (.scalar "i32"), none⟩] (.dimension ⟨21, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩ ⟨3, .index, (some 0)⟩ ⟨29, .index, (some 2)⟩ ⟨[2, 3], 0, [3, 1], "i32", 0⟩ 0 2 (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"func.call\",\"operand_arity\":2,\"properties\":[[\"callee\",\"@touch\"]],\"result_types\":[\"i32\"]}" [⟨28, (.scalar "i32"), none⟩, ⟨2, (.view ⟨[1], 0, [1], "i32", 0⟩), none⟩] [⟨30, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.dealloc\",\"operand_arity\":1,\"properties\":[],\"result_types\":[]}" [⟨19, (.view ⟨[2, 3], 0, [3, 1], "i32", 0⟩), none⟩] [] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.addi\",\"operand_arity\":2,\"properties\":[[\"overflowFlags\",\"#arith.overflow<none>\"]],\"result_types\":[\"i32\"]}" [⟨18, (.scalar "i32"), none⟩, ⟨30, (.scalar "i32"), none⟩] [⟨31, (.scalar "i32"), none⟩] (.returned [⟨31, (.scalar "i32"), none⟩, ⟨29, .index, (some 2)⟩]))))))))))))))))))))))))))))))))⟩
def entry0Row : EntryRow := ⟨"kernel", .kernel⟩
def entry0Inputs : List (Runtime Unit) := sampleInputs plan.objects.length entry0.parameters
theorem entry0_found : functionAt independentSource "kernel" = some entry0 := by decide
theorem entry0_member : entry0Row ∈ rows := by decide
theorem entry0_plain : Plain entry0.parameters := by decide
theorem entry0_typed : ValuesHold entry0.parameters entry0Inputs :=
  sample_typed plan.objects.length entry0.parameters entry0_plain
theorem entry0_wrong_arity {Value World : Type} (meaning : Interpretation Value World)
    (inputs : List (Runtime Value)) (wrong : inputs.length ≠ entry0.parameters.length)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    ¬ complete meaning independentSource false entry0.name inputs before outputs after :=
  wrong_arity_no_success meaning independentSource entry0.name entry0 entry0_found inputs wrong before outputs after
theorem entry0_external : ∀ input ∈ entry0Inputs, ExternalInput plan.objects.length input :=
  sample_external plan.objects.length entry0.parameters

theorem entry0_kernel_parameters : entry0.parameters = kernel.function.parameters := by decide
theorem entry0_callers : Epoch.CallersValid plan.objects.length plan.callers
    (assign emptyEnv kernel.function.parameters entry0Inputs) := by
  intro caller member
  simp only [plan, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl
  exact ⟨plan.objects.length, Nat.lt_irrefl _, rfl⟩
theorem entry0_domain : EntryDomain independentSource plan kernel entry0Row entry0Inputs := by
  refine ⟨entry0_external, ?_, entry0_callers⟩
  rw [← entry0_kernel_parameters]
  exact entry0_typed

def entry1 : Function := ⟨"touch", [⟨32, (.scalar "i32"), none⟩, ⟨33, (.view ⟨[1], 0, [1], "i32", 0⟩), none⟩], [(.scalar "i32")], (.constant ⟨34, .index, (some 0)⟩ 0 (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.load\",\"operand_arity\":2,\"properties\":[],\"result_types\":[\"i32\"]}" [⟨33, (.view ⟨[1], 0, [1], "i32", 0⟩), none⟩, ⟨34, .index, (some 0)⟩] [⟨35, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.addi\",\"operand_arity\":2,\"properties\":[[\"overflowFlags\",\"#arith.overflow<none>\"]],\"result_types\":[\"i32\"]}" [⟨35, (.scalar "i32"), none⟩, ⟨32, (.scalar "i32"), none⟩] [⟨36, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"memref.store\",\"operand_arity\":3,\"properties\":[],\"result_types\":[]}" [⟨36, (.scalar "i32"), none⟩, ⟨33, (.view ⟨[1], 0, [1], "i32", 0⟩), none⟩, ⟨34, .index, (some 0)⟩] [] (.returned [⟨36, (.scalar "i32"), none⟩])))))⟩
def entry1Row : EntryRow := ⟨"touch", (.unchanged ["touch"])⟩
def entry1Inputs : List (Runtime Unit) := sampleInputs plan.objects.length entry1.parameters
theorem entry1_found : functionAt independentSource "touch" = some entry1 := by decide
theorem entry1_member : entry1Row ∈ rows := by decide
theorem entry1_plain : Plain entry1.parameters := by decide
theorem entry1_typed : ValuesHold entry1.parameters entry1Inputs :=
  sample_typed plan.objects.length entry1.parameters entry1_plain
theorem entry1_wrong_arity {Value World : Type} (meaning : Interpretation Value World)
    (inputs : List (Runtime Value)) (wrong : inputs.length ≠ entry1.parameters.length)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    ¬ complete meaning independentSource false entry1.name inputs before outputs after :=
  wrong_arity_no_success meaning independentSource entry1.name entry1 entry1_found inputs wrong before outputs after
theorem entry1_external : ∀ input ∈ entry1Inputs, ExternalInput plan.objects.length input :=
  sample_external plan.objects.length entry1.parameters

theorem entry1_domain : EntryDomain independentSource plan kernel entry1Row entry1Inputs :=
  ⟨entry1_external, entry1, entry1_found, entry1_typed⟩

def entry2 : Function := ⟨"untouched", [⟨37, (.scalar "i32"), none⟩], [(.scalar "i32")], (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operand_arity\":0,\"properties\":[[\"value\",\"1 : i32\"]],\"result_types\":[\"i32\"]}" [] [⟨38, (.scalar "i32"), none⟩] (.action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.addi\",\"operand_arity\":2,\"properties\":[[\"overflowFlags\",\"#arith.overflow<none>\"]],\"result_types\":[\"i32\"]}" [⟨37, (.scalar "i32"), none⟩, ⟨38, (.scalar "i32"), none⟩] [⟨39, (.scalar "i32"), none⟩] (.returned [⟨39, (.scalar "i32"), none⟩])))⟩
def entry2Row : EntryRow := ⟨"untouched", (.unchanged ["untouched"])⟩
def entry2Inputs : List (Runtime Unit) := sampleInputs plan.objects.length entry2.parameters
theorem entry2_found : functionAt independentSource "untouched" = some entry2 := by decide
theorem entry2_member : entry2Row ∈ rows := by decide
theorem entry2_plain : Plain entry2.parameters := by decide
theorem entry2_typed : ValuesHold entry2.parameters entry2Inputs :=
  sample_typed plan.objects.length entry2.parameters entry2_plain
theorem entry2_wrong_arity {Value World : Type} (meaning : Interpretation Value World)
    (inputs : List (Runtime Value)) (wrong : inputs.length ≠ entry2.parameters.length)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    ¬ complete meaning independentSource false entry2.name inputs before outputs after :=
  wrong_arity_no_success meaning independentSource entry2.name entry2 entry2_found inputs wrong before outputs after
theorem entry2_external : ∀ input ∈ entry2Inputs, ExternalInput plan.objects.length input :=
  sample_external plan.objects.length entry2.parameters

theorem entry2_domain : EntryDomain independentSource plan kernel entry2Row entry2Inputs :=
  ⟨entry2_external, entry2, entry2_found, entry2_typed⟩

theorem entry2_literal_selection :
    (withCalls (entryMeaning (Value := Unit) (User := Unit) 64 indices plan.opcodes (fun _ _ _ _ => True) literals (fun _ _ => True)) independentSource
      (Merlin.GraphLoopsShape.invoke (entryMeaning (Value := Unit) (User := Unit) 64 indices plan.opcodes (fun _ _ _ _ => True) literals (fun _ _ => True)) independentSource false 0)).action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operand_arity\":0,\"properties\":[[\"value\",\"1 : i32\"]],\"result_types\":[\"i32\"]}" =
    LiteralAction "i32" "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operand_arity\":0,\"properties\":[[\"value\",\"1 : i32\"]],\"result_types\":[\"i32\"]}" (fun _ _ => True) := by rfl
theorem entry2_binary_selection :
    (withCalls (entryMeaning (Value := Unit) (User := Unit) 64 indices plan.opcodes (fun _ _ _ _ => True) literals (fun _ _ => True)) independentSource
      (Merlin.GraphLoopsShape.invoke (entryMeaning (Value := Unit) (User := Unit) 64 indices plan.opcodes (fun _ _ _ _ => True) literals (fun _ _ => True)) independentSource false 0)).action "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.addi\",\"operand_arity\":2,\"properties\":[[\"overflowFlags\",\"#arith.overflow<none>\"]],\"result_types\":[\"i32\"]}" =
    Epoch.Binary "i32" (fun _ _ _ => True) := by rfl
theorem entry2_source_execution :
    complete (entryMeaning (Value := Unit) (User := Unit) 64 indices plan.opcodes (fun _ _ _ _ => True) literals (fun _ _ => True)) independentSource false entry2.name
      [.scalar "i32" ()] ⟨⟨0, ()⟩, sampleHeap⟩ [.scalar "i32" ()] ⟨⟨0, ()⟩, sampleHeap⟩ :=
  literal_binary_complete (entryMeaning (Value := Unit) (User := Unit) 64 indices plan.opcodes (fun _ _ _ _ => True) literals (fun _ _ => True)) independentSource entry2 entry2_found
    "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.constant\",\"operand_arity\":0,\"properties\":[[\"value\",\"1 : i32\"]],\"result_types\":[\"i32\"]}" "{\"attributes\":[],\"index_constant\":null,\"location\":\"loc(unknown)\",\"name\":\"arith.addi\",\"operand_arity\":2,\"properties\":[[\"overflowFlags\",\"#arith.overflow<none>\"]],\"result_types\":[\"i32\"]}" "i32" ⟨37, (.scalar "i32"), none⟩ ⟨38, (.scalar "i32"), none⟩ ⟨39, (.scalar "i32"), none⟩
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    entry2_literal_selection entry2_binary_selection ⟨⟨0, ()⟩, sampleHeap⟩
theorem entry2_source_outcome :
    Epoch.graphOutcomes plan.objects.length (entryMeaning (Value := Unit) (User := Unit) 64 indices plan.opcodes (fun _ _ _ _ => True) literals (fun _ _ => True)) independentSource entry2.name
      entry2Inputs sampleHeap ()
      (Epoch.observeWorld plan.objects.length [.scalar "i32" ()] ⟨⟨0, ()⟩, sampleHeap⟩) :=
  ⟨[.scalar "i32" ()], ⟨⟨0, ()⟩, sampleHeap⟩, entry2_source_execution, rfl⟩
theorem entry2_final_outcome :
    Epoch.graphOutcomes plan.objects.length (entryMeaning (Value := Unit) (User := Unit) 64 indices plan.opcodes (fun _ _ _ _ => True) literals (fun _ _ => True)) GraphMemoryAfterMaterialization.actualFinal entry2.name
      entry2Inputs sampleHeap ()
      (Epoch.observeWorld plan.objects.length [.scalar "i32" ()] ⟨⟨0, ()⟩, sampleHeap⟩) :=
  (complete_roster (fun _ _ _ _ => True) (fun _ _ => True) entry2Row entry2_member entry2Inputs entry2_domain
    sampleHeap () _).mp entry2_source_outcome

end GraphMemoryAfterEntries
