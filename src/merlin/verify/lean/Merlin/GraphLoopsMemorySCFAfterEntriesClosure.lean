/- UNCOMPILED successor. Finite call-closed sets justify unchanged entry
preservation and conservative primitive extension. Body equality alone is not
enough: every internal target, including absent/declaration names, is covered.
-/
import Merlin.GraphLoopsMemorySCFAfterChain

set_option autoImplicit false

namespace Merlin.GraphLoopsMemorySCF.After

open GraphLoopsShape

def TargetClosed (calls : List (String × String)) (names : List String) (key : String) : Prop :=
  match callTarget calls key with
  | none => True
  | some target => target ∈ names

instance (calls : List (String × String)) (names : List String) (key : String) : Decidable (TargetClosed calls names key) := by
  unfold TargetClosed
  split <;> infer_instance

def ClosedEntry (module : Module) (names : List String) (name : String) : Prop :=
  match functionAt module name with
  | none => True
  | some function => ∀ key ∈ actionKeys function.code, TargetClosed module.calls names key

instance (module : Module) (names : List String) (name : String) : Decidable (ClosedEntry module names name) := by
  unfold ClosedEntry
  split
  · infer_instance
  · infer_instance

def CallClosed (module : Module) (names : List String) : Prop :=
  ∀ name ∈ names, ClosedEntry module names name

instance (module : Module) (names : List String) : Decidable (CallClosed module names) := by
  unfold CallClosed
  infer_instance

theorem closed_target (module : Module) (names : List String) (closed : CallClosed module names)
    (name : String) (member : name ∈ names) (function : Function) (found : functionAt module name = some function)
    (key : String) (used : key ∈ actionKeys function.code) (target : String)
    (called : callTarget module.calls key = some target) : target ∈ names := by
  have entry := closed name member
  simp only [ClosedEntry, found] at entry
  have result := entry key used
  simpa only [TargetClosed, called] using result

def ClosedAgreement (first second : Module) (names : List String) : Prop :=
  first.calls = second.calls ∧ CallClosed first names ∧
    ∀ name ∈ names, functionAt first name = functionAt second name

instance (first second : Module) (names : List String) : Decidable (ClosedAgreement first second names) := by
  unfold ClosedAgreement
  infer_instance

/-- Each recursive call is justified by finite reachability. An unchanged
wrapper that calls a changed kernel fails ClosedAgreement rather than inheriting
an unproved complete-world/contextual congruence from a public-outcome theorem.
-/
theorem closed_invoke_agreement {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (first second : Module) (names : List String) (agreement : ClosedAgreement first second names)
    (depth : Nat) (name : String) (member : name ∈ names) :
    GraphLoopsShape.invoke meaning first false depth name = GraphLoopsShape.invoke meaning second false depth name := by
  induction depth generalizing name with
  | zero => rfl
  | succ depth induction =>
    funext inputs before outputs after
    apply propext
    simp only [GraphLoopsShape.invoke, ← agreement.2.2 name member]
    cases found : functionAt first name with
    | none => rfl
    | some function =>
      apply and_congr_right; intro typed
      apply eval_action_agreement
      · intro key used
        have tables := agreement.1
        simp only [withCalls, ← tables]
        cases called : callTarget first.calls key with
        | none => rfl
        | some target =>
          have targetMember := closed_target first names agreement.2.1 name member function found key used target called
          simp only [← agreement.2.2 target targetMember, induction target targetMember]
      · rfl

theorem closed_complete_agreement {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World)
    (first second : Module) (names : List String) (agreement : ClosedAgreement first second names)
    (name : String) (member : name ∈ names) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    complete meaning first false name inputs before outputs after ↔ complete meaning second false name inputs before outputs after := by
  unfold complete
  apply exists_congr; intro depth
  rw [closed_invoke_agreement meaning first second names agreement depth name member]

theorem closed_invoke_extension {Value World : Type} (first second : GraphLoopsShape.Interpretation Value World)
    (module : Module) (names : List String) (closed : CallClosed module names)
    (actions : ∀ name ∈ names, ∀ function, functionAt module name = some function →
      ∀ key ∈ actionKeys function.code, first.action key = second.action key)
    (regions : first.region = second.region) (depth : Nat) (name : String) (member : name ∈ names) :
    GraphLoopsShape.invoke first module false depth name = GraphLoopsShape.invoke second module false depth name := by
  induction depth generalizing name with
  | zero => rfl
  | succ depth induction =>
    funext inputs before outputs after
    apply propext
    simp only [GraphLoopsShape.invoke]
    cases found : functionAt module name with
    | none => rfl
    | some function =>
      apply and_congr_right; intro typed
      apply eval_action_agreement
      · intro key used
        simp only [withCalls]
        cases called : callTarget module.calls key with
        | none => exact actions name member function found key used
        | some target =>
          have targetMember := closed_target module names closed name member function found key used target called
          simp only [induction target targetMember, actions name member function found key used]
      · exact regions

theorem absent_entry_no_success {Value World : Type} (meaning : GraphLoopsShape.Interpretation Value World) (module : Module)
    (name : String) (absent : functionAt module name = none) (inputs : List (Runtime Value))
    (before : World) (outputs : List (Runtime Value)) (after : World) :
    ¬ complete meaning module false name inputs before outputs after := by
  rintro ⟨depth, run⟩
  cases depth with
  | zero => exact run
  | succ depth => simpa only [GraphLoopsShape.invoke, absent] using run

end Merlin.GraphLoopsMemorySCF.After
