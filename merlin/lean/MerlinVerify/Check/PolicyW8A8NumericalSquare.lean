/- A square receives a nonnegative postcondition only after checking its complete scalar
region, including duplicate SSA reads and the actual yielded result. -/
import MerlinVerify.Check.PolicyW8A8NumericalScalar
import MerlinVerify.Rules.W8A8SquareBody
import MerlinVerify.Rules.W8A8SourceFree

namespace MerlinVerify.Check.PolicyW8A8NumericalSquare

open MerlinVerify Rules Definedness PolicyW8A8NumericalFacts PolicyW8A8NumericalScalar

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def infer (_ : Refined.FEnv Info) (actual : Region) (inputs : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match inputs with
  | [⟨⟨.scalar (.float .f32), true⟩, input⟩, ⟨⟨.scalar (.float .f32), _⟩, _⟩] =>
    if accepts input .nonNaN && W8A8SourceFree.region actual && W8A8SquareBody.check actual then
      some [fact .nonnegative] else none
  | _ => none

theorem infer_sound (ib : Nat) : ∀ Γ actual inputs outputs,
    infer Γ actual inputs = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args inputs → ∃ values,
      evalRegion sourceSem ctx call env actual args = .ok values ∧
      Refined.FactsHold model ib values outputs ∧ ResultsDefined values := by
  intro Γ actual inputs outputs checked ctx call env args _width permitted _valid holds
  unfold infer at checked
  split at checked
  · next input initialized initialInfo =>
    split at checked
    · next accepted =>
      cases checked
      simp only [Bool.and_eq_true] at accepted
      obtain ⟨⟨numeric, ordinary⟩, matched⟩ := accepted
      obtain _ | ⟨inputHead, tail⟩ := holds
      obtain _ | ⟨initialHead, tail⟩ := tail
      cases tail
      obtain ⟨x, rfl⟩ := initialized_float inputHead.1
      have inputNN : W8A8IEEE32.NonNaN x := by simpa [bitsCheck] using requirement numeric inputHead
      rename_i initial
      cases initial with
      | tensor _ => cases initialHead.1.1
      | scalar initial =>
        obtain ⟨base, _laws, agreement, interpretation⟩ := permitted
        have numerical := W8A8IEEE32.square_nonnegative inputNN
        have executed := W8A8SquareBody.check_sound matched ctx call env x initial
        rw [interpretation, W8A8ByteModel.multiply_eq, agreement.multiply32 _ _ numerical.1] at executed
        exact ⟨_, (W8A8SourceFree.region_eq ordinary ctx call env _).trans executed,
          .cons (result (by simpa [bitsCheck] using numerical)) .nil,
          by simp [ResultsDefined, RVal.Defined]⟩
    · cases checked
  · cases checked

def contracts (ib : Nat) : Contextual.Regions model sourceSem ib allowed where
  infer := infer
  sound := infer_sound ib

#print axioms infer_sound
#print axioms contracts

end MerlinVerify.Check.PolicyW8A8NumericalSquare
