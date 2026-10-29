/- UNCOMPILED. Actual expanded entry/call/return → frozen candidate epoch
execution. Local simulations are DERIVED by bound_local_simulations; no whole-run
equality is assumed. Complete means possible finite success, not equal failures,
termination, native pointer identity or allocator resource consumption.
-/
import Merlin.GraphLoopsMemorySCFAfterChecked

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem expanded_depth_equivalent {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (expansions : List Expansion) (binding : ExpansionBinding bits plan kernel indices expansions)
    (boundary : BoundaryFacts plan kernel expansions) (scalar : Epoch.BinaryMeaning Value) (depth : Nat)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (GraphLoopsShape.assign emptyEnv kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar) (expandedModule plan kernel expansions) false
      (depth + 2) kernel.function.name inputs ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) true true
        kernel.instructions (Epoch.entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = Epoch.clockWorld plan.objects.length (physicalFinalClock 0 expansions) after := by
  simp only [GraphLoopsShape.invoke, boundary.2.2.2.2, Function.invoke, typed, true_and]
  rw [Epoch.eval_aliases _ _ _ []]
  have localLaw := bound_local_simulations (User := User) bits indices plan planValid kernel facts expansions binding boundary scalar depth
  have commands : Epoch.CommandsValid plan kernel.function.parameters 0 (expansions.map Expansion.source) := by
    rw [binding.1]; exact facts.2.2.1
  have returnBound : AllBound (Epoch.finalContext kernel.function.parameters (expansions.map Expansion.source)) kernel.returned := by
    rw [binding.1]; exact boundary.2.2.1
  have whole := expanded_block plan scalar (expandedMeaning bits indices plan kernel expansions scalar depth)
    expansions kernel.function.parameters 0 0 commands localLaw kernel.returnKey kernel.returned facts.2.2.2.1
    returnBound binding.2.2.2.1 (Epoch.entryMemory plan kernel.function inputs heap user)
    (GraphLoopsShape.assign emptyEnv kernel.function.parameters inputs)
    (Epoch.entry_registers plan kernel.function facts.2.2.2.2.1 inputs typed heap user)
    (Epoch.entry_callers plan kernel.function facts.2.2.2.2.1 inputs callers heap user)
    (entry_related plan kernel.function facts.2.2.2.2.1 boundary.2.1 inputs typed callers heap user) outputs world
  rw [Epoch.entry_world, binding.1] at whole
  exact whole

theorem fragments_refused {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World) (fragments : List Fragment)
    (fragment : Fragment) (member : fragment ∈ fragments)
    (refused : ∀ env before next after, ¬ fragment.step meaning env before next after)
    (actual next : GraphLoopsShape.Env Value) (before after : World) :
    ¬ FragmentsRun meaning fragments actual before next after := by
  intro run
  induction run with
  | done => cases member
  | more first suffix induction =>
    rcases List.mem_cons.mp member with rfl | later
    · exact refused _ _ _ _ first
    · exact induction later

theorem observer_fragment (expansion : Expansion) (key operation name : String)
    (input buffer result : Binding) (program : Epoch.Touch)
    (source : expansion.source = .observer key operation name input buffer result program) :
    Fragment.atom (.action operation [input, buffer] [result]) ∈ expansion.fragments := by
  cases expansion <;> simp_all [Expansion.source, Expansion.fragments, Epoch.Command.atom]

theorem expanded_depth_one_refused {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (expansions : List Expansion) (binding : ExpansionBinding bits plan kernel indices expansions)
    (boundary : BoundaryFacts plan kernel expansions) (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (before after : MemoryWorld Value (Epoch.AllocationPublic User))
    (outputs : List (Runtime Value)) :
    ¬ GraphLoopsShape.invoke (finalMeaning bits indices plan.opcodes scalar) (expandedModule plan kernel expansions) false
      1 kernel.function.name inputs before outputs after := by
  simp only [GraphLoopsShape.invoke, boundary.2.2.2.2]
  rintro ⟨_, run⟩
  have plain := (Epoch.eval_aliases _ _ _ [] _ _ _ _).mp run
  obtain ⟨next, fragments, _⟩ := (fragments_complete _ _ _ _ _ _ _).mp plain
  obtain ⟨command, member, call⟩ := facts.2.2.2.2.2
  have expandedMember : command ∈ expansions.map Expansion.source := by rw [binding.1]; exact member
  obtain ⟨expansion, expansionMember, sourceEqual⟩ := List.mem_map.mp expandedMember
  cases command with
  | observer key operation name input buffer result program =>
    have target := call.1
    have found := boundary.2.2.2.1 _ member
    have fragmentMember := observer_fragment expansion key operation name input buffer result program sourceEqual
    have wholeMember : Fragment.atom (.action operation [input, buffer] [result]) ∈ expansions.flatMap Expansion.fragments :=
      List.mem_flatMap.mpr ⟨expansion, expansionMember, fragmentMember⟩
    apply fragments_refused _ _ (.atom (.action operation [input, buffer] [result])) wholeMember _ _ next before after fragments
    intro actual world next final
    exact Epoch.zero_call_refused _ (expandedModule plan kernel expansions) operation name target
      (by rw [found]; rfl) [input, buffer] [result] actual next world final
  | constant | binary | allocate | release | «initialize» | load | store => exact False.elim call

theorem expanded_complete_equivalent {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (expansions : List Expansion) (binding : ExpansionBinding bits plan kernel indices expansions)
    (boundary : BoundaryFacts plan kernel expansions) (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (GraphLoopsShape.assign emptyEnv kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    complete (finalMeaning bits indices plan.opcodes scalar) (expandedModule plan kernel expansions) false
      kernel.function.name inputs ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) true true
        kernel.instructions (Epoch.entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = Epoch.clockWorld plan.objects.length (physicalFinalClock 0 expansions) after := by
  constructor
  · rintro ⟨depth, run⟩
    cases depth with
    | zero => exact False.elim run
    | succ depth =>
      cases depth with
      | zero => exact False.elim (expanded_depth_one_refused bits indices plan kernel facts expansions binding boundary
          scalar inputs _ _ outputs run)
      | succ depth => exact (expanded_depth_equivalent bits indices plan planValid kernel facts expansions binding boundary
          scalar depth inputs typed callers heap user outputs world).mp run
  · intro run
    exact ⟨2, (expanded_depth_equivalent bits indices plan planValid kernel facts expansions binding boundary
      scalar 0 inputs typed callers heap user outputs world).mpr run⟩

/-- The ACTUAL independently decoded final module is used here. Exact complete
materialization is checked after deriving the source-plan→expanded execution.
This root still needs actual compilation/audit and a complete finite obligation.
-/
theorem actual_final_equivalent {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (plan : Epoch.PlanContext) (planValid : plan.Valid) (kernel : Epoch.Kernel) (facts : kernel.Facts plan)
    (expansions : List Expansion) (binding : ExpansionBinding bits plan kernel indices expansions)
    (boundary : BoundaryFacts plan kernel expansions)
    (actualFinal : Module) (swaps : List (Nat × Nat)) (sourceFrame finalFrame : List String)
    (materialized : GraphLoopsShape.After.check
      ⟨bits, expandedModule plan kernel expansions, actualFinal, swaps, sourceFrame, finalFrame⟩ = true)
    (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (GraphLoopsShape.assign emptyEnv kernel.function.parameters inputs))
    (heap : Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    complete (finalMeaning bits indices plan.opcodes scalar) actualFinal false kernel.function.name
      inputs ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) true true
        kernel.instructions (Epoch.entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = Epoch.clockWorld plan.objects.length (physicalFinalClock 0 expansions) after :=
  (checked_materialization bits plan kernel expansions actualFinal swaps sourceFrame finalFrame materialized
    (finalMeaning bits indices plan.opcodes scalar) kernel.function.name inputs _ outputs world).symm.trans
    (expanded_complete_equivalent bits indices plan planValid kernel facts expansions binding boundary
      scalar inputs typed callers heap user outputs world)

end Merlin.GraphLoopsMemorySCF.After
