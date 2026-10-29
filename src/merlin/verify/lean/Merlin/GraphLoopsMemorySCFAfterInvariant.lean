/- UNCOMPILED. Candidate epoch execution preserves the old scoped logical
register/caller invariants, even though private heap addresses use physical
slots. This is derived from each frozen instruction constructor, not from an
assumed source run on the candidate heap.
-/
import Merlin.GraphLoopsMemorySCFAfterPlan
import Merlin.GraphLoopsMemorySCFAfterMemorySteps

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

theorem candidate_retained_iff {Value User : Type} (bound : Nat)
    (specifications : Epoch.Specifications) (scalar : Epoch.BinaryMeaning Value) (mapping : Aliases)
    (key : String) (spec : Epoch.Spec) (selected : Epoch.specAt specifications key = some spec)
    (objects : GraphLoopsReuse.Objects) (configs : List Configuration)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User)) :
    GraphLoopsMemorySCF.Step objects configs (Epoch.interpretation bound specifications mapping scalar) true true
      (.retained key) before after ↔
      ∃ observed, Epoch.Retained bound scalar spec before.observed observed ∧ after = ⟨observed, before.heap⟩ := by
  constructor
  · intro run
    cases run with
    | ordinary _ run =>
      cases run with
      | retained key initial observed action =>
        obtain ⟨actual, found, action⟩ := action
        have same := Option.some.inj (found.symm.trans selected)
        subst actual
        exact ⟨observed, action, rfl⟩
  · rintro ⟨observed, action, rfl⟩
    exact .ordinary (by trivial) (.retained key before observed ⟨spec, selected, action⟩)

theorem candidate_initializer_public {Value Public : Type} (objects : GraphLoopsReuse.Objects)
    (configs : List Configuration) (meaning : GraphLoopsMemorySCF.Interpretation Value Public)
    (object : Nat) (key : String) (before after : GraphLoopsReuse.Memory Value Public)
    (run : GraphLoopsMemorySCF.Step objects configs meaning true true (.initialize object key) before after) :
    after.observed = before.observed := by
  generalize instructionEq : GraphLoopsReuse.Instruction.initialize object key = instruction at run
  cases run with
  | ordinary ordinary run =>
    rw [← instructionEq] at ordinary
    exact False.elim ordinary
  | initialized => rfl

theorem candidate_assignment {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock : Nat) (command : Epoch.Command)
    (checked : command.Valid plan context sourceClock) (scalar : Epoch.BinaryMeaning Value)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (registers : Epoch.RegistersValid context plan.views before.observed.registers)
    (run : GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
      command.instruction before after) :
    ∃ values, ValuesHold command.results values ∧
      after.observed.registers = GraphLoopsShape.assign before.observed.registers command.results values := by
  cases command with
  | constant key result value =>
    obtain ⟨observed, equal, rfl⟩ := (candidate_retained_iff plan.objects.length plan.specifications scalar plan.mapping
      key (.constant result value) checked.1.1 plan.objects plan.configs before after).mp run
    change observed = Epoch.writeRegister before.observed result.name (.index value) at equal
    subst observed
    exact ⟨[.index value], .cons (result.holds_index value checked.1.2.1 checked.1.2.2) .nil, rfl⟩
  | binary key operation left right result element =>
    obtain ⟨observed, action, rfl⟩ := (candidate_retained_iff plan.objects.length plan.specifications scalar plan.mapping
      key (.binary operation left right result element) checked.1.1 plan.objects plan.configs before after).mp run
    obtain ⟨a, b, value, _, _, _, holds, rfl⟩ := action
    exact ⟨[.scalar element value], .cons holds .nil, rfl⟩
  | observer key operation name input buffer result program =>
    obtain ⟨observed, action, rfl⟩ := (candidate_retained_iff plan.objects.length plan.specifications scalar plan.mapping
      key (.observer program.binaryKey input buffer result program.element) checked.1.1
      plan.objects plan.configs before after).mp run
    obtain ⟨object, value, loaded, updated, _, _, _, _, _, holds, rfl⟩ := action
    exact ⟨[.scalar program.element updated], .cons holds .nil, rfl⟩
  | allocate operation item retained =>
    have equal := (candidate_allocate_iff plan.objects plan.configs (plan.oldMeaning scalar)
      item.object retained before after).mp run
    have observed : after.observed = before.observed := by
      rw [equal]
      cases retained <;> rfl
    rw [observed]
    have validItem := checked.1.1
    have pinned := registers.2 item validItem.1
    refine ⟨[Epoch.view item.object item.shape item.element],
      .cons (Binding.holds_plain item.binding _ validItem.2.2.2.1.symm validItem.2.2.2.2) .nil, ?_⟩
    exact (Epoch.put_existing _ _ _ pinned).symm
  | release operation item retained =>
    have equal := (candidate_release_iff plan.objects plan.configs (plan.oldMeaning scalar)
      item.object retained before after).mp run
    refine ⟨[], .nil, ?_⟩
    rw [equal]
    cases retained <;> rfl
  | «initialize» link =>
    have equal := candidate_initializer_public plan.objects plan.configs (plan.oldMeaning scalar)
      link.config.object link.config.key before after run
    exact ⟨[], .nil, congrArg Epoch.PublicState.registers equal⟩
  | initializeLine link =>
    have equal := candidate_initializer_public plan.objects plan.configs (plan.oldMeaning scalar)
      link.config.object link.config.key before after run
    exact ⟨[], .nil, congrArg Epoch.PublicState.registers equal⟩
  | load key operation access result =>
    have validAccess := checked.1.2.1
    obtain ⟨coordinates, value, _, _, holds, rfl⟩ :=
      (candidate_load_iff plan.objects.length plan.specifications scalar plan.mapping key access result checked.1.1
        plan.objects plan.configs validAccess.1.2.2.1 before after).mp run
    exact ⟨[.scalar access.element value], .cons holds .nil, rfl⟩
  | store key operation access operand =>
    have validAccess := checked.1.2.1
    obtain ⟨coordinates, value, _, _, rfl⟩ :=
      (candidate_store_iff plan.objects.length plan.specifications scalar plan.mapping key access operand checked.1.1
        plan.objects plan.configs validAccess.1.2.2.1 before after).mp run
    exact ⟨[], .nil, rfl⟩

theorem candidate_pins {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock : Nat) (command : Epoch.Command)
    (checked : command.Valid plan context sourceClock) (scalar : Epoch.BinaryMeaning Value)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (registers : Epoch.RegistersValid context plan.views before.observed.registers)
    (run : GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
      command.instruction before after) : Epoch.Pinned plan.views after.observed.registers := by
  obtain ⟨values, typed, assigned⟩ := candidate_assignment plan context sourceClock command checked scalar before after registers run
  cases command with
  | allocate operation item retained =>
    have equal := (candidate_allocate_iff plan.objects plan.configs (plan.oldMeaning scalar)
      item.object retained before after).mp run
    rw [equal]
    cases retained <;> exact registers.2
  | constant | binary | release | «initialize» | initializeLine | load | store | observer =>
    rw [assigned]
    exact Epoch.pinned_assign plan.views before.observed.registers registers.2 _ values checked.2.2.2.1

theorem candidate_preserves {Value User : Type} (plan : Epoch.PlanContext)
    (context : Context) (sourceClock : Nat) (command : Epoch.Command)
    (checked : command.Valid plan context sourceClock) (scalar : Epoch.BinaryMeaning Value)
    (before after : GraphLoopsReuse.Memory Value (Epoch.PublicState Value User))
    (registers : Epoch.RegistersValid context plan.views before.observed.registers)
    (callers : Epoch.CallersValid plan.objects.length plan.callers before.observed.registers)
    (run : GraphLoopsMemorySCF.Step plan.objects plan.configs (plan.oldMeaning scalar) true true
      command.instruction before after) :
    Epoch.RegistersValid (extendContext context command.results) plan.views after.observed.registers ∧
      Epoch.CallersValid plan.objects.length plan.callers after.observed.registers := by
  obtain ⟨values, typed, assigned⟩ := candidate_assignment plan context sourceClock command checked scalar before after registers run
  refine ⟨⟨?_, candidate_pins plan context sourceClock command checked scalar before after registers run⟩, ?_⟩
  · rw [assigned]
    exact Epoch.typed_assign context before.observed.registers registers.1 command.results checked.2.2.1 values typed
  · rw [assigned]
    exact Epoch.callers_assign plan.objects.length context plan.callers before.observed.registers callers checked.2.2.2.2
      command.results checked.2.2.1 values

end Merlin.GraphLoopsMemorySCF.After
