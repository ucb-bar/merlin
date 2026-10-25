/- Initialized exponent facts for an actual ordered floating contraction. Geometry, the
zero initializer, operand bounds and scalar-region identity are independent premises. -/
import MerlinVerify.Rules.W8A8OrderedTensor
import MerlinVerify.Rules.W8A8OrderedLayout
import MerlinVerify.Check.PolicyW8A8RotaryDequant
import MerlinVerify.Check.DefinedElementwise

namespace MerlinVerify.Check.PolicyW8A8OrderedRefined

open MerlinVerify Rules Definedness Sem.Ops.Linalg PolicyW8A8RotaryFacts

def inputFact (shape : List Nat) (info : Info) : Refined.Fact Info :=
  ⟨⟨.tensor shape (.float .f32), true⟩, info⟩

def outputFact (shape : List Nat) (cap : Nat) : Refined.Fact Info :=
  inputFact shape (PolicyW8A8RotaryDequant.boundedInfo cap)

/-- This total observation is used only after proving that the actual mapped read is an
initialized f32. Its default is unreachable at every executed point. -/
def word (value : RVal) (map : AffineMap) (point : Idx) : FBits .f32 :=
  match readAt value map point with
  | .ok value => (Sem.Ops.Dequant.float32 value).getD 0#32
  | _ => 0#32

private theorem scalar_bits {ib : Nat} {value : Val} {info : Info}
    (holds : (⟨⟨.scalar (.float .f32), true⟩, info⟩ : Refined.Fact Info).Holds model ib
      (.scalar value)) : ∃ bits, value = W8A8OrderedBody.value bits := by
  obtain ⟨⟨typed, initialized⟩, _⟩ := holds
  cases value with
  | poison => exact (initialized rfl rfl).elim
  | some scalar =>
    cases scalar with
    | bits _ _ => simp [RVal.HasTy, Val.HasTy, Scalar.conforms] at typed
    | flt kind bits =>
      cases kind <;> simp_all [RVal.HasTy, Val.HasTy, Scalar.conforms, W8A8OrderedBody.value]

theorem read_bounded {ib cap : Nat} {ranges point shape : List Nat} {value : RVal}
    {info : Info} {map : AffineMap}
    (bounded : upper info = some cap) (holds : (inputFact shape info).Holds model ib value)
    (mapped : mapInBounds ranges map shape = true) (bounds : Idx.InBounds ranges point) :
    readAt value map point = .ok (W8A8OrderedBody.value (word value map point)) ∧
      W8A8IEEEBounds.ExponentLE cap (word value map point) := by
  obtain ⟨result, read, valid⟩ := Refined.readAt_refined elementwise bounds holds mapped
  obtain ⟨bits, same⟩ := scalar_bits valid
  subst result
  have decoded : word value map point = bits := by
    simp only [word, read, W8A8OrderedBody.value, Sem.Ops.Dequant.float32]
    rfl
  rw [decoded]
  exact ⟨read, upper_sound bounded valid⟩

/-- Numerical and definedness refinement for the complete actual generic execution.
The explicit arithmetic interpretation is required; the older abstract laws alone do not
justify the new ordered sum bound. -/
theorem generic_refined {ib : Nat} {ctx : Ctx} {actual : Region}
    {layout : W8A8OrderedLayout.Layout} {left right : RVal} {initial : Tensor}
    {leftInfo rightInfo initialInfo : Info} {leftCap rightCap : Nat}
    (geometry : W8A8OrderedLayout.Valid layout)
    (accepted : PolicyW8A8OrderedBody.check actual = true)
    (call : CallSem) (env : Env) (fallback : FloatModel)
    (interpretation : ctx.M = W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback))
    (leftBound : upper leftInfo = some leftCap) (rightBound : upper rightInfo = some rightCap)
    (zero : initialInfo.prior.prior.floating.literal = some 0#32)
    (safe : W8A8OrderedDot.admissible leftCap rightCap layout.reductionSize = true)
    (leftHolds : (inputFact layout.leftShape leftInfo).Holds model ib left)
    (rightHolds : (inputFact layout.rightShape rightInfo).Holds model ib right)
    (initialHolds : (inputFact layout.outputShape initialInfo).Holds model ib (.tensor initial)) :
    ∃ results,
      genericSem ctx layout.maps layout.iterators [left, right] [.tensor initial]
        (fun c => evalRegion W8A8SourceSemantics.opSem c call env actual) = .ok results ∧
      Refined.FactsHold model ib results
        [outputFact layout.outputShape (W8A8OrderedDot.cap leftCap rightCap layout.reductionSize)] := by
  cases geometry.1.inBounds
  rename_i leftMap remaining
  cases remaining
  rename_i rightMap _remaining
  have leftRead := fun point bounds => read_bounded leftBound leftHolds leftMap
    (point := point) bounds
  have rightRead := fun point bounds => read_bounded rightBound rightHolds rightMap
    (point := point) bounds
  have zeroValues : ∀ cell, Idx.InBounds layout.outputShape cell →
      initial.get cell = W8A8OrderedBody.value 0#32 := by
    have literal := initialHolds.2.1.1.1.2
    change FloatFacts.Literal initialInfo.prior.prior.floating.literal (.tensor initial) at literal
    simp only [zero, FloatFacts.Literal] at literal
    intro cell bounds
    exact literal cell (initialHolds.1.1.1 ▸ bounds)
  have actualInitial : W8A8TensorTrace.pack layout.outputShape (.float .f32) id initial.get = initial := by
    have initialShape := initialHolds.1.1.1
    have initialElt := initialHolds.1.1.2.1
    cases initial with
    | mk shape elt get =>
      change shape = layout.outputShape at initialShape
      change elt = .float .f32 at initialElt
      subst shape
      subst elt
      rfl
  obtain ⟨result, executed, shape, elt, cells⟩ := W8A8OrderedTensor.projected_bounded
    geometry.2.1 (by simpa only [W8A8OrderedLayout.types, W8A8ProjectionLayout.Layout.maps,
      geometry.2.2] using geometry.1)
    leftHolds.1.1 rightHolds.1.1 accepted call env fallback interpretation
    (word left layout.leftMap) (word right layout.rightMap) initial.get
    (by simpa only [← geometry.2.2] using zeroValues)
    (fun point bounds => (leftRead point bounds).1) (fun point bounds => (rightRead point bounds).1)
    (fun point bounds => ⟨(leftRead point bounds).2, (rightRead point bounds).2⟩) safe
  rw [← geometry.2.2] at executed shape cells
  rw [actualInitial] at executed
  refine ⟨[.tensor result], executed, .cons ⟨⟨⟨shape, elt, ?_⟩, ?_⟩, ?_⟩ .nil⟩
  · intro cell bounds
    obtain ⟨bits, actual, _⟩ := cells cell bounds
    rw [actual]
    simp [W8A8OrderedBody.value, Val.HasTy, Scalar.conforms]
  · intro _ cell bounds
    obtain ⟨bits, actual, _⟩ := cells cell (shape ▸ bounds)
    simp [actual, W8A8OrderedBody.value]
  · apply PolicyW8A8RotaryDequant.boundedInfo_sound
    intro cell bounds
    obtain ⟨bits, actual, bounded⟩ := cells cell (shape ▸ bounds)
    simp only [pointCheck, actual, W8A8OrderedBody.value, Sem.Ops.Dequant.float32]
    exact decide_eq_true bounded

#print axioms read_bounded
#print axioms generic_refined

end MerlinVerify.Check.PolicyW8A8OrderedRefined
