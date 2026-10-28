/- Exact raw-bit scalar reinterpretation for the domains already modelled by Arith.sem.
Float-to-float bitcasts are deliberately excluded: the existing dispatcher does not model them. -/
import MerlinVerify.Check.RulesArith

namespace MerlinVerify.Rules.ScalarBitcast

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

def Compatible : Ty → Ty → Prop
  | .int w, .int w' => w = w' ∧ 0 < w
  | .int w, .float k => w = k.width
  | .float k, .int w => w = k.width
  | _, _ => False

instance (a b : Ty) : Decidable (Compatible a b) := by
  unfold Compatible; split <;> infer_instance

/-- Numeric promises are not part of the raw-bit contract. -/
def Plain (s : Op) : Prop :=
  Arith.fastmathKey s = "0" ∧ overflowFlags s = [] ∧
    hasUnit s "nonNeg" = false ∧ s.attr? "roundingmode" = none

instance (s : Op) : Decidable (Plain s) := by unfold Plain; infer_instance

theorem resultElt_eq {s : Op} {r : Value} {src dst : Ty}
    (hr : s.results = [r]) (hty : r.ty = dst) (hc : Compatible src dst) :
    Arith.resultElt s.blind = some dst := by
  cases src <;> cases dst <;> simp only [Compatible] at hc
  all_goals simp [Arith.resultElt, Op.blind_results, hr, hty]

/-- Successful reinterpretation preserves typing and cannot introduce poison. The input may
itself be poison, so this same kernel supports progress and initialized-source proofs. -/
theorem meaning {s : Op} {src dst : Ty} (hn : s.name = "arith.bitcast")
    (hm : unmodelledAttr? s.blind = none) (hr : Arith.resultElt s.blind = some dst)
    (hc : Compatible src dst) (ctx : Ctx) (call : CallSem) (regions : List (Ctx → RegionSem))
    {value : Val} (ht : Val.HasTy ctx.indexBits src value) :
    ∃ target, opSem ctx call s [.scalar value] regions = .ok [.scalar target] ∧
      Val.HasTy ctx.indexBits dst target ∧ (value ≠ .poison → target ≠ .poison) := by
  have hd : dialectOf s.name = "arith" := by rw [hn]; cbv
  rw [opSem_modelled hm]
  simp only [dispatch, Op.blind_name, hd]
  cases src <;> cases dst <;> simp only [Compatible] at hc
  · next w w' =>
    obtain ⟨rfl, _⟩ := hc
    rcases hasTy_intWidth (t := .int w) rfl ht with rfl | ⟨x, rfl⟩
    all_goals
      simp only [Arith.sem, Op.blind_name, hn, hr, Arith.unOp, Arith.pureN]
      rw [show [.scalar _] = ([_] : List Val).map RVal.scalar from rfl, liftEager_scalars]
      simp only [↓reduceDIte, Outcome.bind_ok]
      exact ⟨_, rfl, by simp [Arith.bv, Val.HasTy, Scalar.conforms], by simp [Arith.bv]⟩
  · next w k =>
    subst w
    rcases hasTy_intWidth (t := .int k.width) rfl ht with rfl | ⟨x, rfl⟩
    all_goals
      simp only [Arith.sem, Op.blind_name, hn, hr, Arith.unOp, Arith.pureN]
      rw [show [.scalar _] = ([_] : List Val).map RVal.scalar from rfl, liftEager_scalars]
      simp only [↓reduceDIte, Outcome.bind_ok]
      exact ⟨_, rfl, by simp [Val.HasTy, Scalar.conforms], by simp⟩
  · next k w =>
    subst w
    rcases hasTy_float ht with rfl | ⟨x, rfl⟩
    all_goals
      simp only [Arith.sem, Op.blind_name, hn, hr, Arith.unOp, Arith.pureN]
      rw [show [.scalar _] = ([_] : List Val).map RVal.scalar from rfl, liftEager_scalars]
      simp only [↓reduceDIte, Outcome.bind_ok]
      exact ⟨_, rfl, by simp [Arith.bv, Val.HasTy, Scalar.conforms], by simp [Arith.bv]⟩

end MerlinVerify.Rules.ScalarBitcast
