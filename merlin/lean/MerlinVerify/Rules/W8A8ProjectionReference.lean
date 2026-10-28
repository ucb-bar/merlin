/- Recognize source projection contractions without consulting a proposed output. The
reference front end supplies its existing source type, zero-initializer and scalar-body
recognizers; geometry and grouping remain independent of target metadata. -/
import MerlinVerify.Rules.W8A8ProjectionLayout
import MerlinVerify.Rules.W8A8Emission

namespace MerlinVerify.Rules.W8A8ProjectionReference
open MerlinVerify Sem.Ops.Linalg Check

structure View where
  layout : W8A8ProjectionLayout.Layout
  left : Nat
  right : Nat
  leftElt : Ty
  rightElt : Ty
  result : Value

def inspect (types : Nat → Option Ty) (zero : Nat → Bool) (body : Ty → Ty → Bool → Region)
    (op : Op) : Option View := do
  if op.name != "linalg.generic" then none
  let [left, right, destination] := op.operands | none
  let [result] := op.results | none
  let some (leftShape, leftElt) := (← types left).staticTensor? | none
  let some (rightShape, rightElt) := (← types right).staticTensor? | none
  let some (outputShape, .float .f32) := result.ty.staticTensor? | none
  if !(leftElt == .float .f32 || leftElt == .int 8) ||
      !(rightElt == .float .f32 || rightElt == .int 8) ||
      !(leftElt == .float .f32 || rightElt == .float .f32) then none
  if (← types destination) != result.ty || !zero destination then none
  if Sem.Ops.unmodelledAttr? op.blind != none || !op.successors.isEmpty then none
  if splitInsOuts op op.operands != some ([left, right], [destination]) then none
  let [actualBody] := op.regions | none
  if !(alphaRegion [] (body leftElt rightElt false) actualBody ||
      alphaRegion [] (body leftElt rightElt true) actualBody) then none
  let maps ← attrAffineMaps? op "indexing_maps"
  let iters ← iterKinds? op
  let layout ← W8A8ProjectionLayout.inspect leftShape rightShape outputShape maps iters
  return ⟨layout, left, right, leftElt, rightElt, result⟩

theorem inspect_valid {types : Nat → Option Ty} {zero : Nat → Bool} {body : Ty → Ty → Bool → Region}
    {op : Op} {view : View} (selected : inspect types zero body op = some view) :
    W8A8ProjectionLayout.Valid view.layout := by
  unfold inspect at selected
  repeat' first
    | simp only [bind, Option.bind, pure, Pure.pure] at selected
    | split at selected
    | contradiction
  all_goals cases selected; exact W8A8ProjectionLayout.inspect_sound (by assumption)

#print axioms inspect_valid
end MerlinVerify.Rules.W8A8ProjectionReference
