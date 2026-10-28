/- Recompute exact arange region syntax; width and actual loop position remain
explicit execution conditions rather than certificate-provided numerical facts. -/
import MerlinVerify.Rules.W8A8AttentionRamp

namespace MerlinVerify.Check.PolicyW8A8AttentionRamp

open MerlinVerify Rules W8A8AttentionRamp

def Correct (actual : Region) : Prop :=
  ∀ (ctx : Ctx) (call : CallSem) (env : Env) (column : Nat) (destination : Val),
    ctx.indexBits = 64 → ctx.loopIndex[0]? = some column →
    evalRegion W8A8SourceSemantics.opSem ctx call env actual [.scalar destination] =
      .ok [.scalar (W8A8AttentionMaskBody.integer (BitVec.ofNat 64 column))]

def check (actual : Region) : Bool := alphaRegion [] region actual

theorem check_sound {actual : Region} (checked : check actual = true) : Correct actual := by
  intro ctx call env column destination width position
  have same := ParametricAlpha.alphaRegion_sound W8A8SourceAlpha.laws call call [] region actual ctx env env checked
    (by intro x y missing; simp [Rho.find, List.lookup] at missing) (fun _ _ => rfl)
  rw [← same]
  exact region_sound ctx call env column destination width position

end MerlinVerify.Check.PolicyW8A8AttentionRamp
