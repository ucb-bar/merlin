/- A shared fuel induction closes independently proved body equalities across all actual
module functions. Calls, declarations, missing symbols and recursive fuel use the same relation. -/
import MerlinVerify.Check.ParametricModule

namespace MerlinVerify.Check.ModuleBodyEquality
open MerlinVerify

structure Bodies (sem : OpSem) (contexts : Ctx → Prop) where
  check : Region → Region → Bool
  sound : ∀ {left right}, check left right = true → ∀ ctx, contexts ctx → ∀ call env,
    evalRegion sem ctx call env left = evalRegion sem ctx call env right

variable {sem : OpSem} {contexts : Ctx → Prop}

def FunctionRel (sem : OpSem) (contexts : Ctx → Prop) (left right : Func) : Prop :=
  left.name = right.name ∧ left.isDecl = right.isDecl ∧
    ∀ ctx, contexts ctx → ∀ call, evalRegion sem ctx call [] left.body = evalRegion sem ctx call [] right.body

def checkFunctions (bodies : Bodies sem contexts) : List Func → List Func → Bool
  | [], [] => true
  | left :: rest, right :: tail =>
    decide (left.name = right.name ∧ left.isPrivate = right.isPrivate ∧ left.isDecl = right.isDecl) &&
      bodies.check left.body right.body && checkFunctions bodies rest tail
  | _, _ => false

theorem checkFunctions_sound (bodies : Bodies sem contexts) {left right : List Func}
    (checked : checkFunctions bodies left right = true) : List.Forall₂ (FunctionRel sem contexts) left right := by
  induction left generalizing right with
  | nil => cases right <;> simp_all [checkFunctions]
  | cons head rest ih =>
    cases right with
    | nil => cases checked
    | cons other tail =>
      simp only [checkFunctions, Bool.and_eq_true, decide_eq_true_eq] at checked
      exact .cons ⟨checked.1.1.1, checked.1.1.2.2, fun ctx permitted call => bodies.sound checked.1.2 ctx permitted call []⟩ (ih checked.2)

theorem lookup_match {left right : List Func} (matched : List.Forall₂ (FunctionRel sem contexts) left right) (name : String) :
    (left.find? (·.name == name) = none ∧ right.find? (·.name == name) = none) ∨
    ∃ a b, left.find? (·.name == name) = some a ∧ right.find? (·.name == name) = some b ∧ FunctionRel sem contexts a b := by
  induction matched with
  | nil => exact .inl ⟨rfl, rfl⟩
  | @cons a b left right same rest ih =>
    by_cases named : a.name == name
    · have other : b.name == name := by simpa [same.1] using named
      exact .inr ⟨a, b, by simp [List.find?, named], by simp [List.find?, other], same⟩
    · have other : (b.name == name) = false := by simpa [same.1] using named
      simpa [List.find?, named, other] using ih

theorem evalFunc_equal {left right : Module} (matched : List.Forall₂ (FunctionRel sem contexts) left.funcs right.funcs)
    (ctx : Ctx) (permitted : contexts ctx) : ∀ fuel name args, evalFunc sem ctx left fuel name args = evalFunc sem ctx right fuel name args := by
  intro fuel
  induction fuel with
  | zero => intros; rfl
  | succ fuel ih =>
    intro name args
    rcases lookup_match matched name with missing | ⟨a, b, foundLeft, foundRight, same⟩
    · change left.func? name = none ∧ right.func? name = none at missing
      simp [evalFunc, missing.1, missing.2]
    · change left.func? name = some a at foundLeft
      change right.func? name = some b at foundRight
      simp only [evalFunc, foundLeft, foundRight, same.2.1]
      split
      · rfl
      · have calls : evalFunc sem ctx left fuel = evalFunc sem ctx right fuel := by
          funext called values
          exact ih called values
        rw [calls]
        exact congrFun (same.2.2 ctx permitted (evalFunc sem ctx right fuel)) args

def check (bodies : Bodies sem contexts) (left right : Module) : Bool :=
  ParametricModule.functionShapes left && ParametricModule.functionShapes right &&
    checkFunctions bodies left.funcs right.funcs

theorem check_sound (bodies : Bodies sem contexts) {left right : Module} (checked : check bodies left right = true) :
    ∀ ctx, contexts ctx → ∀ fn args, denote sem ctx left fn args = denote sem ctx right fn args := by
  simp only [check, Bool.and_eq_true] at checked
  have functions := checkFunctions_sound bodies checked.2
  have length := List.Forall₂.length_eq functions
  intro ctx permitted fn args
  simp only [denote, length]
  exact evalFunc_equal functions ctx permitted _ fn args

#print axioms check_sound
end MerlinVerify.Check.ModuleBodyEquality
