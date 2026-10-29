/- UNCOMPILED. Conservative addition of generated index operations to ONE
interpretation. A finite key check on every source function/callee proves the
source is unaffected; no whole-call equality is assumed as a premise.
-/
import Merlin.GraphLoopsMemorySCFAfterFunction

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def actionKeys : Code → List String
  | .returned _ => []
  | .cast _ _ _ _ rest | .dimension _ _ _ _ _ _ rest | .constant _ _ rest => actionKeys rest
  | .action key _ _ rest => key :: actionKeys rest
  | .region _ _ _ _ body rest => actionKeys body ++ actionKeys rest

def SourceIndexAbsent (indices : IndexOpcodes) (module : Module) : Prop :=
  ∀ function ∈ module.functions, ∀ key ∈ actionKeys function.code, indexOpcodeAt indices key = none

instance (indices : IndexOpcodes) (module : Module) : Decidable (SourceIndexAbsent indices module) := by
  unfold SourceIndexAbsent
  infer_instance

theorem eval_action_agreement {Value World : Type} (first second : Interpretation Value World)
    (code : Code) (actions : ∀ key ∈ actionKeys code, first.action key = second.action key)
    (regions : first.region = second.region) (aliases : Aliases) (env : GraphLoopsShape.Env Value)
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    Eval first false aliases code env before outputs after ↔ Eval second false aliases code env before outputs after := by
  induction code generalizing aliases env before outputs after with
  | returned values => rfl
  | cast origin result source target rest induction =>
    simp only [Eval, Bool.false_eq_true, ↓reduceIte]
    apply exists_congr; intro value
    exact and_congr_right (fun _ => induction actions _ _ _ _ _)
  | dimension buffer axis result type axisIndex value rest induction =>
    simp only [Eval, Bool.false_eq_true, ↓reduceIte]
    apply exists_congr; intro actual
    exact and_congr_right (fun _ => induction actions _ _ _ _ _)
  | constant result value rest induction => exact induction actions _ _ _ _ _
  | action key operands results rest induction =>
    have action := actions key List.mem_cons_self
    have tail : ∀ other ∈ actionKeys rest, first.action other = second.action other :=
      fun other member => actions other (List.mem_cons_of_mem _ member)
    simp only [Eval, action]
    apply exists_congr; intro values
    apply exists_congr; intro middle
    apply and_congr_right; intro _
    exact and_congr_right (fun _ => induction tail _ _ _ _ _)
  | region key operands parameters results body rest bodyInduction restInduction =>
    have bodyActions : ∀ other ∈ actionKeys body, first.action other = second.action other :=
      fun other member => actions other (List.mem_append_left _ member)
    have restActions : ∀ other ∈ actionKeys rest, first.action other = second.action other :=
      fun other member => actions other (List.mem_append_right _ member)
    have bodySame :
        (fun inputs initial returned final => ValuesHold parameters inputs ∧
          Eval first false (extendAliases aliases parameters) body (assign env parameters inputs) initial returned final) =
        (fun inputs initial returned final => ValuesHold parameters inputs ∧
          Eval second false (extendAliases aliases parameters) body (assign env parameters inputs) initial returned final) := by
      funext inputs initial returned final
      apply propext
      exact and_congr_right (fun _ => bodyInduction bodyActions _ _ _ _ _)
    simp only [Eval, regions, bodySame]
    apply exists_congr; intro values
    apply exists_congr; intro middle
    apply and_congr_right; intro _
    exact and_congr_right (fun _ => restInduction restActions _ _ _ _ _)

theorem invoke_action_agreement {Value World : Type} (first second : Interpretation Value World)
    (module : Module)
    (actions : ∀ function ∈ module.functions, ∀ key ∈ actionKeys function.code,
      first.action key = second.action key)
    (regions : first.region = second.region) (depth : Nat) :
    GraphLoopsShape.invoke first module false depth = GraphLoopsShape.invoke second module false depth := by
  induction depth with
  | zero => rfl
  | succ depth induction =>
    funext name inputs before outputs after
    apply propext
    unfold GraphLoopsShape.invoke
    split
    · rfl
    · rename_i function found
      have member := List.mem_of_find?_eq_some found
      apply and_congr_right
      intro typed
      apply eval_action_agreement
      · intro key used
        simp only [withCalls, induction]
        cases target : callTarget module.calls key with
        | none => exact actions function member key used
        | some target =>
          split
          · rfl
          · exact actions function member key used
      · exact regions

theorem source_complete_extension {Value User : Type} (bits : Nat) (indices : IndexOpcodes)
    (opcodes : Epoch.Opcodes) (scalar : Epoch.BinaryMeaning Value) (module : Module)
    (absent : SourceIndexAbsent indices module) (name : String)
    (inputs : List (Runtime Value)) (before : MemoryWorld Value (Epoch.AllocationPublic User))
    (outputs : List (Runtime Value)) (after : MemoryWorld Value (Epoch.AllocationPublic User)) :
    complete (Epoch.graphMeaning opcodes scalar) module false name inputs before outputs after ↔
      complete (finalMeaning bits indices opcodes scalar) module false name inputs before outputs after := by
  have agreement (depth : Nat) := invoke_action_agreement (Epoch.graphMeaning opcodes scalar)
    (finalMeaning (User := User) bits indices opcodes scalar) module
    (fun function member key used => (unchanged_action bits indices opcodes scalar key (absent function member key used)).symm)
    (unchanged_regions bits indices opcodes scalar).symm depth
  unfold complete
  apply exists_congr
  intro depth
  rw [agreement depth]

end Merlin.GraphLoopsMemorySCF.After
