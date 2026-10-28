/- Complete-module SSA matching under explicit semantic laws. All function bodies are checked;
calls use a common fuel induction, including internal calls and unknown symbols. Metadata and
non-function top-level operations are compared separately without discarding attributes. -/
import MerlinVerify.Check.ParametricAlpha
import MerlinVerify.Check.ExactSyntax

namespace MerlinVerify.Check.ParametricModule

open MerlinVerify

def FunctionMatch (left right : Func) : Prop :=
  left.name = right.name ∧ left.isPrivate = right.isPrivate ∧ left.isDecl = right.isDecl ∧
    alphaRegion [] left.body right.body = true

instance (left right : Func) : Decidable (FunctionMatch left right) := by
  unfold FunctionMatch
  infer_instance

def checkFunctions : List Func → List Func → Bool
  | [], [] => true
  | left :: rest, right :: tail => decide (FunctionMatch left right) && checkFunctions rest tail
  | _, _ => false

theorem checkFunctions_sound {left right : List Func} (checked : checkFunctions left right = true) :
    List.Forall₂ FunctionMatch left right := by
  induction left generalizing right with
  | nil => cases right <;> simp_all [checkFunctions]
  | cons first rest ih =>
    cases right with
    | nil => simp [checkFunctions] at checked
    | cons second tail =>
      simp only [checkFunctions, Bool.and_eq_true, decide_eq_true_eq] at checked
      exact .cons checked.1 (ih checked.2)

theorem lookup_match {left right : List Func} (matched : List.Forall₂ FunctionMatch left right)
    (name : String) :
    (left.find? (·.name == name) = none ∧ right.find? (·.name == name) = none) ∨
    ∃ a b, left.find? (·.name == name) = some a ∧ right.find? (·.name == name) = some b ∧
      FunctionMatch a b := by
  induction matched with
  | nil => exact .inl ⟨rfl, rfl⟩
  | @cons a b left right same rest ih =>
    by_cases named : a.name == name
    · have other : b.name == name := by simpa [same.1] using named
      exact .inr ⟨a, b, by simp [List.find?, named], by simp [List.find?, other], same⟩
    · have other : (b.name == name) = false := by simpa [same.1] using named
      simpa [List.find?, named, other] using ih

theorem evalFunc_equal {sem : OpSem} (laws : ParametricAlpha.Laws sem) {left right : Module}
    (matched : List.Forall₂ FunctionMatch left.funcs right.funcs) (ctx : Ctx) :
    ∀ fuel name args, evalFunc sem ctx left fuel name args = evalFunc sem ctx right fuel name args := by
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
      simp only [evalFunc, foundLeft, foundRight, same.2.2.1]
      split
      · rfl
      · have calls : CallAgree (evalFunc sem ctx left fuel) (evalFunc sem ctx right fuel)
            (calleesBlocks a.body.blocks) := fun called _ => funext (ih called)
        exact congrFun (ParametricAlpha.alphaRegion_sound laws _ _ [] a.body b.body ctx [] []
          same.2.2.2 Corr.nil calls) args

/-- Only function bodies may differ in the outer skeleton. Non-function operations and every
module/function attribute stay visible to complete-AST comparison. -/
def skeleton (m : Module) : Module :=
  { m with ops := m.ops.map fun op => match op with
      | .mk name results operands attrs props regions successors =>
        .mk name results operands attrs props (if name == "func.func" then [] else regions) successors }

/-- The function view reads at most one body region. Reject extra regions before erasing the
body from the outer comparison, even when they came from malformed structured input. -/
def functionShapes (m : Module) : Bool :=
  m.ops.all fun op => op.name != "func.func" ||
    op.asFunc?.isSome && decide (op.regions.length ≤ 1)

def check (left right : Module) : Bool :=
  functionShapes left && functionShapes right &&
    ExactSyntax.modules (skeleton left) (skeleton right) && checkFunctions left.funcs right.funcs

/-- Accepted complete modules have equal executions for every context, input and function,
including calls. Successful/initialized execution remains an independent obligation. -/
theorem check_sound {sem : OpSem} (laws : ParametricAlpha.Laws sem) {left right : Module}
    (checked : check left right = true) :
    ∀ ctx name args, denote sem ctx left name args = denote sem ctx right name args := by
  simp only [check, Bool.and_eq_true] at checked
  have functions := checkFunctions_sound checked.2
  have length := List.Forall₂.length_eq functions
  intro ctx name args
  simp only [denote, length]
  exact evalFunc_equal laws functions ctx _ name args

#print axioms evalFunc_equal
#print axioms check_sound

end MerlinVerify.Check.ParametricModule
