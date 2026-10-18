/- Scalar conversion totality under every FloatModel. These rules establish initialized result
bits from actual initialized typed inputs; they assert no IEEE or numerical conversion laws. -/
import MerlinVerify.Check.DefinedRules

namespace MerlinVerify.Check.DefinedConversions

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops MerlinVerify.Check.Definedness

/-- Promise-bearing or explicitly rounded conversions require their own qualification. -/
def DefaultAttrs (op : Op) : Prop :=
  Arith.fastmathKey op = "0" ∧ hasUnit op "nonNeg" = false ∧
    overflowFlags op = [] ∧ op.attr? "roundingmode" = none

instance (op : Op) : Decidable (DefaultAttrs op) := by
  unfold DefaultAttrs; infer_instance

def floatFormatRule : Definedness.Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.float k), true⟩], [r] =>
    match r.ty with
    | .float k' =>
      if dialectOf op.name = "arith" ∧
          ((op.name = "arith.extf" ∧ k.width < k'.width) ∨
           (op.name = "arith.truncf" ∧ k'.width < k.width)) ∧ DefaultAttrs op then
        some [⟨.scalar (.float k'), true⟩]
      else none
    | _ => none
  | _, _ => none

theorem floatFormatRule_sound (ib : Nat) : Definedness.RuleSound ib floatFormatRule := by
  intro op fs rs hm h ctx call args _ ha
  unfold floatFormatRule at h
  split at h
  · next k r hr =>
    split at h
    · next k' hty =>
      split at h
      · next hc =>
        obtain ⟨hd, hn, _⟩ := hc
        cases h
        obtain _ | ⟨hx, hnil⟩ := ha
        cases hnil
        obtain ⟨x, rfl⟩ := initialized_float hx
        have hresult : Arith.resultElt op.blind = some (.float k') := by
          simp [Arith.resultElt, Op.blind_results, hr, hty]
        rw [opSem_modelled hm]
        simp only [dispatch, Op.blind_name, hd]
        rcases hn with ⟨hn, _⟩ | ⟨hn, _⟩
        all_goals
          simp only [Arith.sem, Op.blind_name, hn, hresult, Arith.unOp, Arith.pureN]
          rw [show [.scalar (.some (.flt k x))] =
            ([.some (.flt k x)] : List Val).map RVal.scalar from rfl, liftEager_scalars]
          exact ⟨_, rfl, .cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
            by simp [RVal.Defined]⟩ .nil⟩
      · cases h
    all_goals cases h
  · cases h

/-- The source width is witnessed by the actual input bits. Direct index operands are excluded,
matching the pinned upstream verifier's fixed-width integer domain. -/
def integerRule : Definedness.Rule := fun op fs =>
  match fs, op.results with
  | [⟨.scalar (.int w), true⟩], [r] =>
    match r.ty with
    | .float k =>
      if dialectOf op.name = "arith" ∧
          (op.name = "arith.sitofp" ∨ op.name = "arith.uitofp") ∧
          0 < w ∧ DefaultAttrs op then some [⟨.scalar (.float k), true⟩]
      else none
    | _ => none
  | _, _ => none

theorem integerRule_sound (ib : Nat) : Definedness.RuleSound ib integerRule := by
  intro op fs rs hm h ctx call args _ ha
  unfold integerRule at h
  split at h
  · next w r hr =>
    split at h
    · next k hty =>
      split at h
      · next hc =>
        obtain ⟨hd, hn, _, _, hnneg, _, _⟩ := hc
        cases h
        obtain _ | ⟨hx, hnil⟩ := ha
        cases hnil
        obtain ⟨x, rfl⟩ := initialized_int hx
        have hresult : Arith.resultElt op.blind = some (.float k) := by
          simp [Arith.resultElt, Op.blind_results, hr, hty]
        have hnneg' : hasUnit op.blind "nonNeg" = false := by simpa [hasUnit] using hnneg
        rw [opSem_modelled hm]
        simp only [dispatch, Op.blind_name, hd]
        rcases hn with hn | hn
        all_goals
          simp only [Arith.sem, Op.blind_name, hn, hresult, Arith.unOp, Arith.pureN, hnneg']
          rw [show [.scalar (.some (.bits w x))] =
            ([.some (.bits w x)] : List Val).map RVal.scalar from rfl, liftEager_scalars]
          simp only [Bool.and_false, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
          exact ⟨_, rfl, .cons ⟨by simp [RVal.HasTy, Val.HasTy, Scalar.conforms],
            by simp [RVal.Defined]⟩ .nil⟩
      · cases h
    all_goals cases h
  · cases h

def rules : Definedness.Rule := fun op fs =>
  (floatFormatRule op fs).orElse fun _ => integerRule op fs

theorem rules_sound (ib : Nat) : Definedness.RuleSound ib rules := by
  intro op fs rs hm h
  cases hf : floatFormatRule op fs with
  | some ys =>
    simp only [rules, hf, Option.orElse_some, Option.some.injEq] at h
    subst h
    exact floatFormatRule_sound ib op fs ys hm hf
  | none =>
    simp only [rules, hf, Option.orElse_none] at h
    exact integerRule_sound ib op fs rs hm h

/-- An additive leaf book for scalar source-definedness, independent of all float models. -/
def book (ib : Nat) : Definedness.Book opSem ib where
  rule _ op fs := if unmodelledAttr? op.blind = none ∧ op.regions = [] then rules op fs else none
  sound Γ op fs rs h ctx call env args hib _ ha := by
    split at h
    · next hc =>
      obtain ⟨hm, hregs⟩ := hc
      simpa only [hregs, evalRegions] using rules_sound ib op fs rs hm h ctx call args hib ha
    · cases h

end MerlinVerify.Check.DefinedConversions
