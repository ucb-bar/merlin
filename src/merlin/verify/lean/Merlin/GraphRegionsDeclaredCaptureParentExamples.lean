/- Mathematical complete-parent examples for the NEW pure development context.
The actualParent AST was independently written in CaptureExamples; these are
NOT authoritative compiler exports, actual-pass tests or native comparisons.
They demonstrate inhabited checked syntax/initialized source execution and exact
full graph projection without assuming source success or whole-program equality. -/
import Merlin.GraphRegionsDeclaredCaptureRefusals
import Merlin.GraphRegionsDeclaredCaptureExamples

namespace Merlin.GraphRegionsDeclared.Capture.Examples
open MerlinVerify Merlin.GraphRegions Merlin.GraphRegionsDeclared

def completeParent : GraphParent :=
  ⟨"captured_add", none, 1, [2], 8,
    [.literal 1 8 6, .empty 2 [2] 8], [⟨literalAddSite, literalAddReferences⟩], [3]⟩

theorem complete_parent_valid : completeParent.Valid := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, ?_, ?_, by decide, by decide⟩
  · intro site member
    have equal : site = (⟨literalAddSite, literalAddReferences⟩ : Site) := by
      simpa only [completeParent, List.mem_cons, List.not_mem_nil, or_false] using member
    subst site
    exact ⟨literalAdd_region, by decide⟩
  · intro index site found
    cases index with
    | zero =>
      simp only [completeParent, List.getElem?_cons_zero, Option.some.injEq] at found
      subst site
      exact ⟨rfl, rfl, rfl, by decide, rfl, by decide, 1, rfl⟩
    | succ index => simp [completeParent] at found

/-- Complete independently written AST equality, full existing graph projection,
and exact prefix read/return privacy are kernel obligations on this example. -/
theorem complete_parent_binding : FullSourceBinding completeParent actualParent actualParent := by
  exact ⟨⟨rfl, rfl, complete_parent_valid⟩, rfl, by decide⟩

/-- A deliberately mathematical common interpretation. Unsupported operations
map to zero here; this is NOT a model of native wrapping integer arithmetic. -/
def exampleMeaning : Primitive → List Nat → Nat
  | .constant _ value, _ => value.toNat
  | .binary _ .add, arguments => (arguments[0]?).getD 0 + (arguments[1]?).getD 0
  | _, _ => 0

def exampleArguments : List (Tensor 2 Nat) := [fun index => index.val + 3]
def exampleInitial : StateEffects.State (Tensor 2 Nat) :=
  fun slot index => if slot = 0 then index.val + 3 else 0

theorem initialized_nonzero_elements :
    tensorEvaluate literalAdd exampleMeaning exampleArguments 0 = 9 ∧
    tensorEvaluate literalAdd exampleMeaning exampleArguments 1 = 10 := by
  exact ⟨rfl, rfl⟩

/-- Actual prefix execution supplies the captured six for arbitrary initial
scalar cells; the static empty destination remains unread. No source-success
or global-typing hypothesis appears. This is the restricted additive model. -/
theorem complete_source_execution (initialCells : Prefix.Cells Nat) :
    GraphParentExecution exampleMeaning actualParent exampleInitial
      [tensorEvaluate literalAdd exampleMeaning exampleArguments] := by
  have actual := GraphParentExecution.evaluated (meaning := exampleMeaning)
    completeParent complete_parent_valid initialCells exampleInitial rfl
  have same := executeSource_eq_stages exampleMeaning initialCells
    (show Prefix.Matches (completeParent.leading.map Prefix.Instruction.emit) completeParent.leading from rfl)
    complete_parent_valid.2.2.2.2.1 completeParent.sites complete_parent_valid.2.2.2.2.2.2.1 exampleInitial
  have observed := congrArg (fun final => completeParent.observations.map final) same
  rw [observed] at actual
  exact actual

/-- Returning tensor.empty is a distinct refused domain, even when its type
matches. Compiler validity is unmeasured; this is a mathematical AST refusal. -/
def emptyEscapes : GraphParent := { completeParent with observations := [2] }

theorem returned_empty_refused : checkPrefixPrivate emptyEscapes = false := by decide

end Merlin.GraphRegionsDeclared.Capture.Examples
