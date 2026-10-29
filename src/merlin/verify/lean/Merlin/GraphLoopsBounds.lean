/- Signed index admission for the existing counted-loop evaluator. No allocation
bound follows from an integer-width bound; GraphLoopsBound checks that separately. -/
import Merlin.GraphLoops

set_option autoImplicit false

namespace Merlin.GraphLoops

theorem evaluate_fits (bits : Nat) (loopVar : String) (iteration : Nat) :
    ∀ (expression : Expr) (value : Int), evaluate bits loopVar iteration expression = some value → Fits bits value := by
  intro expression
  induction expression with
  | constant literal =>
    intro value checked
    simp only [evaluate] at checked
    split at checked
    next bound => cases checked; exact bound
    next => cases checked
  | «variable» name =>
    intro value checked
    simp only [evaluate] at checked
    split at checked
    next bound => cases checked; exact bound.2
    next => cases checked
  | add left right _ _ =>
    intro value checked
    simp only [evaluate] at checked
    cases first : evaluate bits loopVar iteration left with
    | none => simp [first] at checked
    | some a =>
      cases second : evaluate bits loopVar iteration right with
      | none => simp [first, second] at checked
      | some b =>
        simp only [first, second, bind, Option.bind] at checked
        split at checked
        next bound => cases checked; exact bound
        next => cases checked
  | scaled expression factor _ =>
    intro value checked
    simp only [evaluate] at checked
    cases selected : evaluate bits loopVar iteration expression with
    | none => simp [selected] at checked
    | some result =>
      simp only [selected, bind, Option.bind] at checked
      split at checked
      next bound => cases checked; exact bound.2
      next => cases checked
  | select name equals yes no yesBound noBound =>
    intro value checked
    simp only [evaluate] at checked
    split at checked
    next =>
      split at checked
      · exact yesBound _ checked
      · exact noBound _ checked
    next => cases checked

def ClosedOperand.Fits (bits : Nat) : ClosedOperand → Prop
  | .integer value => GraphLoops.Fits bits value
  | .pointer _ offset => GraphLoops.Fits bits offset
  | .null => True

theorem closed_operand_fits (loop : Counted) (iteration : Nat) :
    ∀ operand result, closeOperand loop iteration operand = some result → ClosedOperand.Fits loop.bits result := by
  intro operand result checked
  cases operand with
  | integer expression =>
    simp only [closeOperand] at checked
    cases selected : evaluate loop.bits loop.loopVar iteration expression with
    | none => simp [selected] at checked
    | some value =>
      simp only [selected, Functor.map, Option.map, Option.some.injEq] at checked
      subst result
      exact evaluate_fits _ _ _ _ _ selected
  | pointer space offset =>
    simp only [closeOperand] at checked
    split at checked
    next => cases checked
    next =>
      cases selected : evaluate loop.bits loop.loopVar iteration offset with
      | none => simp [selected] at checked
      | some value =>
        simp only [selected, bind, Option.bind, pure, Option.some.injEq] at checked
        subst result
        exact evaluate_fits _ _ _ _ _ selected
  | null => cases checked; trivial

private theorem mapM_every {α β : Type} (f : α → Option β) (property : β → Prop)
    (localBound : ∀ source result, f source = some result → property result) :
    ∀ (source : List α) (result : List β), source.mapM f = some result → ∀ entry ∈ result, property entry := by
  intro source
  induction source with
  | nil => intro result checked; cases checked; simp
  | cons head tail induction =>
    intro result checked
    simp only [List.mapM_cons] at checked
    cases first : f head with
    | none => simp [first] at checked
    | some value =>
      cases rest : tail.mapM f with
      | none => simp [first, rest] at checked
      | some values =>
        simp only [first, rest, bind, Option.bind, pure, Option.some.injEq] at checked
        subst result
        intro entry member
        rcases List.mem_cons.mp member with rfl | later
        · exact localBound _ _ first
        · exact induction _ rest _ later

def ClosedCall.Fits (bits : Nat) (call : ClosedCall) : Prop :=
  ∀ argument ∈ call.arguments, ClosedOperand.Fits bits argument.2

theorem closed_call_fits (loop : Counted) (iteration : Nat) (call : Call) (result : ClosedCall)
    (checked : closeCall loop iteration call = some result) : ClosedCall.Fits loop.bits result := by
  let convert := fun (entry : String × Operand) => do
    if entry.1 = "" then none else return (entry.1, ← closeOperand loop iteration entry.2)
  have allArguments : ∀ source result, convert source = some result → ClosedOperand.Fits loop.bits result.2 := by
    intro source result converted
    simp only [convert] at converted
    split at converted
    next => cases converted
    next =>
      cases selected : closeOperand loop iteration source.2 with
      | none => simp [selected] at converted
      | some value =>
        simp only [selected, bind, Option.bind, pure, Option.some.injEq] at converted
        subst result
        exact closed_operand_fits _ _ _ _ selected
  simp only [closeCall] at checked
  split at checked
  next => cases checked
  next =>
    change (do return ⟨call.instruction, ← call.arguments.mapM convert⟩) = some result at checked
    cases converted : call.arguments.mapM convert with
    | none => simp [converted] at checked
    | some arguments =>
      simp only [converted, bind, Option.bind, pure, Option.some.injEq] at checked
      subst result
      exact mapM_every _ _ allArguments _ _ converted

/-- Width bounds hold for every operand of every actual call, not just a
sampled iteration. Allocation capacity remains an independent obligation. -/
theorem checked_call_bounds {loop : Counted} {actual : List ClosedCall} (accepted : check loop actual = true) :
    ∀ call ∈ actual, ClosedCall.Fits loop.bits call := by
  obtain ⟨blocks, expanded, rfl⟩ := check_sound accepted
  have everyBlock : ∀ iteration block, loop.body.mapM (closeCall loop iteration) = some block →
      ∀ call ∈ block, ClosedCall.Fits loop.bits call := by
    intro iteration block converted
    exact mapM_every _ _ (closed_call_fits loop iteration) _ _ converted
  have allBlocks : ∀ block ∈ blocks, ∀ call ∈ block, ClosedCall.Fits loop.bits call := by
    unfold expand at expanded
    split at expanded
    · exact mapM_every _ _ everyBlock _ _ expanded
    · cases expanded
  intro call member
  obtain ⟨block, present, inside⟩ := List.mem_flatten.mp member
  exact allBlocks _ present _ inside

end Merlin.GraphLoops
