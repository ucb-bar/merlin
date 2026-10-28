/- Scale and byte-conversion region contracts use literal facts from actual enclosing SSA
bindings. Candidate captures are restricted to operand ids of the small matched region. -/
import MerlinVerify.Check.PolicyW8A8DynamicBodies

namespace MerlinVerify.Check.PolicyW8A8DynamicCaptured

open MerlinVerify Rules Definedness FloatFacts

local notation "allowed" => PolicyW8A8DynamicContext.allowed

def operandIds : Region → List ValueId
  | .mk [.mk _ ops] => ops.flatMap Op.operands
  | _ => []

def captures (Γ : Refined.FEnv Info) (actual : Region) : List ConstantFacts.Fact :=
  FloatFacts.known Γ (operandIds actual)

def scale (Γ : Refined.FEnv Info) (actual : Region) (inputs : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match inputs with
  | [⟨⟨.scalar (.float .f32), true⟩, input⟩, ⟨⟨.scalar (.float .f32), _⟩, _⟩] =>
    if Scalar.finite input then
      match (captures Γ actual).find? fun fact =>
          PolicyW8A8Scalar.checkScaleKnown (captures Γ actual) fact.id actual with
      | some _ => some [Scalar.fact (.float .f32) .normal]
      | none => none
    else none
  | _ => none

theorem scale_sound (ib : Nat) : ∀ Γ actual inputs outputs,
    scale Γ actual inputs = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args inputs → ∃ values,
      evalRegion Sem.Ops.opSem ctx call env actual args = .ok values ∧
      Refined.FactsHold model ib values outputs ∧ ResultsDefined values := by
  intro Γ actual inputs outputs checked ctx call env args _width permitted valid holds
  unfold scale at checked
  split at checked
  · next input init destInfo =>
    split at checked
    · next finiteInput =>
      split at checked
      · next capture found =>
        cases checked
        have matched := List.find?_some found
        obtain _ | ⟨inputHead, tail⟩ := holds
        obtain _ | ⟨initialHead, tail⟩ := tail
        cases tail
        obtain ⟨amax, rfl, finiteAmax⟩ := Scalar.finite_value finiteInput inputHead
        rename_i initial
        cases initial with
        | tensor _ => cases initialHead.1.1
        | scalar initial =>
          obtain ⟨base, laws, interpretation⟩ := permitted
          let output := W8A8Scale.guarded (W8A8ByteModel.extend base) amax
          have executed := PolicyW8A8Scalar.checkScaleKnown_sound matched
            (FloatFacts.known_sound _ valid) ctx call (some amax) initial
          rw [interpretation] at executed
          change evalRegion Sem.Ops.opSem ctx call env actual _ =
            .ok [.scalar (.some (.flt .f32 output))] at executed
          exact ⟨_, executed, .cons (Scalar.normal_result (W8A8DynamicLaws.scale_normal laws finiteAmax)) .nil,
            by simp [ResultsDefined, RVal.Defined]⟩
      · cases checked
    · cases checked
  · cases checked

def quantize (Γ : Refined.FEnv Info) (actual : Region) (inputs : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  match inputs with
  | [⟨⟨.scalar (.float .f32), true⟩, input⟩, ⟨⟨.scalar (.float .f32), true⟩, inverse⟩,
      ⟨⟨.scalar (.int 8), _⟩, _⟩] =>
    if Scalar.finite input && Scalar.finite inverse then
      let known := captures Γ actual
      match (known.flatMap fun upper => known.map fun lower => (upper.id, lower.id)).find? fun (upper, lower) =>
          PolicyW8A8Scalar.checkQuantizeKnown known upper lower actual with
      | some _ => some [Scalar.fact (.int 8) .any]
      | none => none
    else none
  | _ => none

theorem quantize_sound (ib : Nat) : ∀ Γ actual inputs outputs,
    quantize Γ actual inputs = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args inputs → ∃ values,
      evalRegion Sem.Ops.opSem ctx call env actual args = .ok values ∧
      Refined.FactsHold model ib values outputs ∧ ResultsDefined values := by
  intro Γ actual inputs outputs checked ctx call env args _width permitted valid holds
  unfold quantize at checked
  split at checked
  · next input inverse init destInfo =>
    split at checked
    · next finiteInputs =>
      dsimp only at checked
      split at checked
      · next capture found =>
        cases checked
        have matched := List.find?_some found
        simp only [Bool.and_eq_true] at finiteInputs
        obtain _ | ⟨inputHead, tail⟩ := holds
        obtain _ | ⟨inverseHead, tail⟩ := tail
        obtain _ | ⟨initialHead, tail⟩ := tail
        cases tail
        obtain ⟨x, rfl, finiteX⟩ := Scalar.finite_value finiteInputs.1 inputHead
        obtain ⟨inv, rfl, finiteInv⟩ := Scalar.finite_value finiteInputs.2 inverseHead
        rename_i initial
        cases initial with
        | tensor _ => cases initialHead.1.1
        | scalar initial =>
          obtain ⟨base, laws, interpretation⟩ := permitted
          obtain ⟨byte, converted⟩ := W8A8ByteModel.quantize_result base x inv
            (W8A8DynamicLaws.rounded_nonNaN laws finiteX finiteInv)
          have executed := PolicyW8A8Scalar.checkQuantizeKnown_sound matched
            (FloatFacts.known_sound _ valid) ctx call (some x) (some inv) initial
          rw [interpretation] at executed
          simp only [PolicyW8A8Scalar.upper, PolicyW8A8Scalar.lower] at executed
          rw [converted] at executed
          exact ⟨_, executed, .cons (Scalar.byte_result byte) .nil, by simp [ResultsDefined, RVal.Defined]⟩
      · cases checked
    · cases checked
  · cases checked

def scaleBook (ib : Nat) : Contextual.Regions model Sem.Ops.opSem ib allowed where
  infer := scale
  sound := scale_sound ib

def quantizeBook (ib : Nat) : Contextual.Regions model Sem.Ops.opSem ib allowed where
  infer := quantize
  sound := quantize_sound ib

#print axioms scale_sound
#print axioms quantize_sound
#print axioms scaleBook
#print axioms quantizeBook

end MerlinVerify.Check.PolicyW8A8DynamicCaptured
