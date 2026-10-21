/- Numerical postconditions for the actual ordered abs/max and reciprocal regions. Shape,
mapping and overwrite/reduction obligations are discharged by the enclosing structured checker. -/
import MerlinVerify.Check.DefinedRegions
import MerlinVerify.Check.FloatScalarFacts
import MerlinVerify.Check.PolicyW8A8DynamicContext
import MerlinVerify.Check.PolicyW8A8Scalar

namespace MerlinVerify.Check.PolicyW8A8DynamicBodies

open MerlinVerify Rules Definedness FloatFacts

local notation "allowed" => PolicyW8A8DynamicContext.allowed

def absMax (_ : Refined.FEnv Info) (actual : Region) (inputs : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match inputs with
  | [⟨⟨.scalar (.float .f32), true⟩, input⟩, ⟨⟨.scalar (.float .f32), true⟩, initial⟩] =>
    if Scalar.finite input && Scalar.finite initial && PolicyW8A8Scalar.checkAbsMax actual then
      some [Scalar.fact (.float .f32) .finite] else none
  | _ => none

theorem absMax_sound (ib : Nat) : ∀ Γ actual inputs outputs,
    absMax Γ actual inputs = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args inputs → ∃ values,
      evalRegion Sem.Ops.opSem ctx call env actual args = .ok values ∧
      Refined.FactsHold model ib values outputs ∧ ResultsDefined values := by
  intro Γ actual inputs outputs checked ctx call env args _width permitted _valid holds
  unfold absMax at checked
  split at checked
  · next input initial =>
    split at checked
    · next accepted =>
      cases checked
      simp only [Bool.and_eq_true] at accepted
      obtain ⟨⟨finiteInput, finiteInitial⟩, matched⟩ := accepted
      obtain _ | ⟨inputHead, tail⟩ := holds
      obtain _ | ⟨initialHead, tail⟩ := tail
      cases tail
      obtain ⟨x, rfl, finiteX⟩ := Scalar.finite_value finiteInput inputHead
      obtain ⟨acc, rfl, finiteAcc⟩ := Scalar.finite_value finiteInitial initialHead
      obtain ⟨base, laws, interpretation⟩ := permitted
      let output := (W8A8ByteModel.extend base).op (W8A8FiniteLaws.key "arith.maximumf") .f32
        [(W8A8ByteModel.extend base).op (W8A8FiniteLaws.key "math.absf") .f32 [x], acc]
      have executed := PolicyW8A8Scalar.checkAbsMax_sound matched ctx call env (some x) (some acc)
      rw [interpretation] at executed
      change evalRegion Sem.Ops.opSem ctx call env actual _ =
        .ok [.scalar (.some (.flt .f32 output))] at executed
      exact ⟨_, executed, .cons (Scalar.finite_result (W8A8DynamicLaws.absmax_finite laws finiteX finiteAcc)) .nil,
        by simp [ResultsDefined, RVal.Defined]⟩
    · cases checked
  · cases checked

def reciprocal (_ : Refined.FEnv Info) (actual : Region) (inputs : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match inputs with
  | [⟨⟨.scalar (.float .f32), true⟩, input⟩, ⟨⟨.scalar (.float .f32), _⟩, _⟩] =>
    if input.numeric = .normal ∧ PolicyW8A8Scalar.checkReciprocal actual = true then
      some [Scalar.fact (.float .f32) .finite] else none
  | _ => none

theorem reciprocal_sound (ib : Nat) : ∀ Γ actual inputs outputs,
    reciprocal Γ actual inputs = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args inputs → ∃ values,
      evalRegion Sem.Ops.opSem ctx call env actual args = .ok values ∧
      Refined.FactsHold model ib values outputs ∧ ResultsDefined values := by
  intro Γ actual inputs outputs checked ctx call env args _width permitted _valid holds
  unfold reciprocal at checked
  split at checked
  · next input init destInfo =>
    split at checked
    · next accepted =>
      cases checked
      obtain ⟨normalInput, matched⟩ := accepted
      obtain _ | ⟨inputHead, tail⟩ := holds
      obtain _ | ⟨initialHead, tail⟩ := tail
      cases tail
      obtain ⟨scale, rfl, normalScale⟩ := Scalar.normal_value normalInput inputHead
      rename_i initial
      cases initial with
      | tensor _ => cases initialHead.1.1
      | scalar initial =>
        obtain ⟨base, laws, interpretation⟩ := permitted
        let output := (W8A8ByteModel.extend base).op (W8A8FiniteLaws.key "arith.divf") .f32 [0x3f800000#32, scale]
        have executed := PolicyW8A8Scalar.checkReciprocal_sound matched ctx call env (some scale) initial
        rw [interpretation] at executed
        change evalRegion Sem.Ops.opSem ctx call env actual _ =
          .ok [.scalar (.some (.flt .f32 output))] at executed
        exact ⟨_, executed, .cons (Scalar.finite_result (W8A8DynamicLaws.reciprocal_finite laws normalScale)) .nil,
          by simp [ResultsDefined, RVal.Defined]⟩
    · cases checked
  · cases checked

def absMaxBook (ib : Nat) : Contextual.Regions model Sem.Ops.opSem ib allowed where
  infer := absMax
  sound := absMax_sound ib

def reciprocalBook (ib : Nat) : Contextual.Regions model Sem.Ops.opSem ib allowed where
  infer := reciprocal
  sound := reciprocal_sound ib

#print axioms absMax_sound
#print axioms reciprocal_sound
#print axioms absMaxBook
#print axioms reciprocalBook

end MerlinVerify.Check.PolicyW8A8DynamicBodies
