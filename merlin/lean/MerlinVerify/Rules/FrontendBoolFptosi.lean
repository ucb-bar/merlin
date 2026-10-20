import MerlinVerify.Spec.FrontendIntent

/-! Soundness of the bool repair against the independently authored A2 frontend intent. -/

namespace MerlinVerify.Rules.FrontendBoolFptosi

open MerlinVerify
open MerlinVerify.Spec.FrontendIntent

/-- The default-attribute `arith.cmpf une` key emitted by the pass. This exact spelling is the
frozen Arith semantics' key for predicate 13 and no fast-math flags. -/
def uneKey : FloatOpKey :=
  { op := "arith.cmpf", modifiers := "fm1:0;rm4:none;pred13:attr 13 : i64" }

/-- The explicit target comparison obligation needed to connect parametric `FloatModel.cmp` to
the bit-level A2 source contract. The generic float model intentionally provides no such law. -/
def UneCorrect (M : FloatModel) : Prop :=
  ∀ x : FBits .f32, M.cmp uneKey .f32 x 0 =
    if x.toNat % 2^31 = 0 then false else true

/-- The minimal syntax facts extracted from a candidate repaired MLIR region. `cmpKey` includes
the predicate and fast-math flags; `zeroBits` is the exact floating constant bit pattern. -/
structure Candidate where
  sourceId : Nat
  lhsId : Nat
  zeroBits : Nat
  cmpKey : FloatOpKey
  zeroHasSourceType : Bool
  resultIsI1 : Bool
  oldCastRemoved : Bool
  deriving Repr

/-- Accept only a full, wired replacement with the reviewed `une` key and positive zero. -/
def check (c : Candidate) : Bool :=
  decide (c.lhsId = c.sourceId ∧ c.zeroBits = 0 ∧ c.cmpKey = uneKey) &&
    c.zeroHasSourceType && c.resultIsI1 && c.oldCastRemoved

private def good : Candidate :=
  { sourceId := 7, lhsId := 7, zeroBits := 0, cmpKey := uneKey,
    zeroHasSourceType := true, resultIsI1 := true, oldCastRemoved := true }

#guard check good
#guard !check { good with oldCastRemoved := false }
#guard !check { good with lhsId := 8 }
#guard !check { good with zeroBits := 1 }
#guard !check { good with cmpKey := { uneKey with modifiers := "ordered-ne" } }
#guard !check { good with zeroHasSourceType := false }
#guard !check { good with resultIsI1 := false }

/-- Evaluation of the repaired expression under an arbitrary float model and SSA environment. -/
def repaired (M : FloatModel) (c : Candidate) (k : FKind)
    (env : Nat → FBits k) : Val :=
  .some (.bits 1 (if M.cmp c.cmpKey k (env c.lhsId)
    (BitVec.ofNat k.width c.zeroBits) then 1 else 0))

/-- For every float model, input bit pattern, and supported source format, the repaired scalar
result is exactly the independently stated frontend intent. No arithmetic or IEEE axiom is used;
all value behavior is delegated to the same target comparison model. -/
theorem check_sound (c : Candidate) (h : check c = true) (M : FloatModel)
    (hune : UneCorrect M) (env : Nat → FBits .f32) :
    repaired M c .f32 env = f32ToBool (some (env c.sourceId)) := by
  simp only [check, Bool.and_eq_true] at h
  rcases h with ⟨⟨⟨hp, _⟩, _⟩, _⟩
  simp only [decide_eq_true_eq] at hp
  rcases hp with ⟨hlhs, hzero, hkey⟩
  have hcmp := hune (env c.sourceId)
  by_cases hz : (env c.sourceId).toNat % 2^31 = 0
  · simp [hz] at hcmp
    simpa [repaired, f32ToBool, hlhs, hzero, hkey, hz] using hcmp
  · simp [hz] at hcmp
    simpa [repaired, f32ToBool, hlhs, hzero, hkey, hz] using hcmp

#print axioms check_sound

end MerlinVerify.Rules.FrontendBoolFptosi
