/- Canonical index literal spelling is chosen from actual decoded bits, independently of hints.
Both sides use the same signed encoding at the checked index width. -/
import MerlinVerify.Check.FactNormalize
import MerlinVerify.Check.DenseExtract

namespace MerlinVerify.Rules.IndexConstant

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def indexResult (s : Op) : Bool :=
  match s.results with
  | [{ ty := .index, .. }] => true
  | _ => false

/-- Preserve result IDs and types. The candidate is replayed before it can replace the source. -/
def candidate (s : Op) {w : Nat} (value : BitVec w) : Op :=
  .mk "arith.constant" s.results [] [("value", .int value.toInt .index)] [] [] []

def replacement? (ib : Nat) (s : Op) : Option Op := do
  if !indexResult s then none else
    let value ← DenseExtract.scalar? ib s
    let .some (.bits _ bits) := value | none
    let v := candidate s bits
    if DenseExtract.scalar? ib v = some value then some v else none

theorem replacement_spec {ib : Nat} {s v : Op} (h : replacement? ib s = some v) :
    v.results = s.results ∧ ∃ value,
      ConstantFacts.value? ib s = some (.scalar value) ∧
      ConstantFacts.value? ib v = some (.scalar value) := by
  unfold replacement? at h
  split at h
  · cases h
  · cases hs : DenseExtract.scalar? ib s with
    | none => simp [hs] at h
    | some value =>
      simp only [hs, bind, Option.bind] at h
      split at h
      · split at h
        · next hv =>
          cases h
          exact ⟨rfl, _, DenseExtract.scalar_spec hs, DenseExtract.scalar_spec hv⟩
        · cases h
      · cases h

/-- Exact equality includes the entire environment, with no freshness or initialized-input
assumption. Unbound inputs, poison elsewhere, and shadowed result IDs are preserved. -/
theorem replacement_sound {ib : Nat} {s v : Op} (h : replacement? ib s = some v) :
    v.isTerminator = s.isTerminator ∧ v.results = s.results ∧
    ∀ ctx call env, ctx.indexBits = ib →
      stepOp opSem ctx call env v = stepOp opSem ctx call env s := by
  obtain ⟨hr,value,hs,hv⟩ := replacement_spec h
  have source := (ConstantFacts.value_spec hs).1
  have target := (ConstantFacts.value_spec hv).1
  refine ⟨?_,hr,?_⟩
  · simp [Op.isTerminator, terminatorNames, source.1, target.1]
  · intro ctx call env hi
    obtain ⟨r,hrs,hstep⟩ := ConstantFacts.value_exec hs hi call env
    obtain ⟨r',hrv,htarget⟩ := ConstantFacts.value_exec hv hi call env
    have : r' = r := by simpa [hr,hrs] using hrv.symm
    subst r'
    exact htarget.trans hstep.symm

/-- This family needs no producer facts and never reads opposite-side hints. -/
def family : FactNormalize.Family ConstantFacts.Fact ConstantFacts.model where
  rewrite ib _ _ _ s := (replacement? ib s).getD s
  rewrite_sound ib _ _ _ s := by
    cases h : replacement? ib s with
    | some v =>
      obtain ⟨ht,hr,hs⟩ := replacement_sound h
      exact ⟨ht,hr,fun ctx call env _ _ hi => hs ctx call env hi⟩
    | none => exact ⟨rfl,rfl,by intros; rfl⟩
  advance _ _ _ _ := []
  advance_sound _ _ _ _ _ _ _ _ _ _ _ _ := by intro f hf; cases hf

end MerlinVerify.Rules.IndexConstant
