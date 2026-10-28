/- Bounded scalar transfers for actual W8A8 source operations. Exact operation
keys and live SSA facts are checked before using the concrete arithmetic lemmas. -/
import MerlinVerify.Check.PolicyW8A8RotaryFacts
import MerlinVerify.Check.PolicyW8A8NumericalOps
import MerlinVerify.Rules.W8A8IntegerFloatBounds

namespace MerlinVerify.Check.PolicyW8A8RotaryScalar

open MerlinVerify Rules Definedness Sem.Ops PolicyW8A8RotaryFacts

local notation "sourceSem" => W8A8SourceSemantics.opSem
local notation "allowed" => PolicyW8A8NumericalContext.allowed

def fact (cap : Nat) : Refined.Fact Info :=
  ⟨⟨.scalar (.float .f32), true⟩, ⟨(PolicyW8A8NumericalScalar.fact .finite).info, some cap⟩⟩

theorem result {ib cap : Nat} {bits : FBits .f32} (bounded : W8A8IEEEBounds.ExponentLE cap bits) :
    (fact cap).Holds model ib (.scalar (.some (.flt .f32 bits))) := by
  have previous := PolicyW8A8NumericalScalar.result (ib := ib) (cls := .finite)
    (bits := bits) (show PolicyW8A8NumericalFacts.bitsCheck .finite bits = true from decide_eq_true bounded.1)
  exact ⟨previous.1, previous.2, by simpa [fact, Holds, pointCheck, Sem.Ops.Dequant.float32] using bounded⟩

def multiplyRule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ op.name = "arith.mulf" ∧
      op.regions = [] ∧ Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name then
    match facts, Arith.resultElt op.blind with
    | [⟨⟨.scalar (.float .f32), true⟩, left⟩, ⟨⟨.scalar (.float .f32), true⟩, right⟩], some (.float .f32) => do
      let leftCap ← upper left
      let rightCap ← upper right
      if max leftCap 1 + max rightCap 1 ≤ 379 then
        some [fact (W8A8IEEEBounds.mulCap leftCap rightCap)] else none
    | _, _ => none
  else none

theorem multiplyRule_sound (ib : Nat) : ∀ Γ op facts outputs,
    multiplyRule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold multiplyRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions, key⟩ := header
    split at checked
    · next left right resultType =>
      cases leftFound : upper left with
      | none => simp [leftFound] at checked
      | some leftCap =>
        cases rightFound : upper right with
        | none => simp [leftFound, rightFound] at checked
        | some rightCap =>
          simp only [leftFound, rightFound, bind, Option.bind] at checked
          split at checked
          · next safe =>
            cases checked
            obtain _ | ⟨leftHolds, tail⟩ := holds
            obtain _ | ⟨rightHolds, tail⟩ := tail
            cases tail
            obtain ⟨x, rfl⟩ := initialized_float leftHolds.1
            obtain ⟨y, rfl⟩ := initialized_float rightHolds.1
            have bounded := W8A8IEEEBounds.mul_bounded (upper_sound leftFound leftHolds)
              (upper_sound rightFound rightHolds) safe
            have execution := PolicyW8A8NumericalOps.binary_exec op .multiply modelled dialect named regions
              resultType key ctx call env x y
            obtain ⟨base, _laws, agreement, interpretation⟩ := permitted
            have multiplication : (W8A8ByteModel.extend base).op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, y] =
                base.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, y] := W8A8ByteModel.multiply_eq base x y
            simp only [PolicyW8A8NumericalScalar.BinaryKind.name] at execution
            rw [interpretation, multiplication,
              agreement.multiply32 _ _ (W8A8IEEE32.finite_notNaN bounded.1)] at execution
            exact ⟨_, execution, .cons (result bounded) .nil⟩
          · cases checked
    · cases checked
  · cases checked

def zeroLiteral (info : Info) : Bool :=
  info.prior.prior.floating.literal.any fun bits => W8A8ByteFloat.magnitudeWord bits == 0

theorem zeroLiteral_sound {info : Info} {bits : FBits .f32} (checked : zeroLiteral info = true)
    (holds : model.holds info (.scalar (.some (.flt .f32 bits)))) :
    ∃ sign, bits = W8A8IEEE32.pack (.zero sign) := by
  cases found : info.prior.prior.floating.literal with
  | none => simp [zeroLiteral, found] at checked
  | some expected =>
    rw [literal_bits found holds]
    apply W8A8IEEEBounds.zero_representation
    simpa [zeroLiteral, found] using checked

def addCap (left right : Info) : Option Nat :=
  if zeroLiteral left then (upper right).map (max · 1)
  else if zeroLiteral right then (upper left).map (max · 1) else none

theorem addCap_sound {ib cap : Nat} {left right : Info} {x y : FBits .f32}
    (checked : addCap left right = some cap)
    (leftHolds : (⟨⟨.scalar (.float .f32), true⟩, left⟩ : Refined.Fact Info).Holds model ib
      (.scalar (.some (.flt .f32 x))))
    (rightHolds : (⟨⟨.scalar (.float .f32), true⟩, right⟩ : Refined.Fact Info).Holds model ib
      (.scalar (.some (.flt .f32 y)))) : W8A8IEEEBounds.ExponentLE cap (W8A8IEEE32.add x y) := by
  unfold addCap at checked
  split at checked
  · next zero =>
    cases bounded : upper right with
    | none => simp [bounded] at checked
    | some bound =>
      simp only [bounded, Option.map_some, Option.some.injEq] at checked
      subst cap
      obtain ⟨sign, equal⟩ := zeroLiteral_sound zero leftHolds.2
      rw [equal]
      exact W8A8IEEEBounds.add_zero_left_bounded (upper_sound bounded rightHolds) sign
  · split at checked
    · next zero =>
      cases bounded : upper left with
      | none => simp [bounded] at checked
      | some bound =>
        simp only [bounded, Option.map_some, Option.some.injEq] at checked
        subst cap
        obtain ⟨sign, equal⟩ := zeroLiteral_sound zero rightHolds.2
        rw [equal]
        exact W8A8IEEEBounds.add_zero_bounded (upper_sound bounded leftHolds) sign
    · cases checked

def addRule (_ : Refined.FEnv Info) (op : Op) (facts : List (Refined.Fact Info)) :
    Option (List (Refined.Fact Info)) :=
  if unmodelledAttr? op.blind = none ∧ dialectOf op.name = "arith" ∧ op.name = "arith.addf" ∧
      op.regions = [] ∧ Arith.floatKey op.blind = Spec.Dequant.defaultKey op.name then
    match facts, Arith.resultElt op.blind with
    | [⟨⟨.scalar (.float .f32), true⟩, left⟩, ⟨⟨.scalar (.float .f32), true⟩, right⟩], some (.float .f32) =>
      (addCap left right).map fun cap => [fact cap]
    | _, _ => none
  else none

theorem addRule_sound (ib : Nat) : ∀ Γ op facts outputs,
    addRule Γ op facts = some outputs → ∀ ctx call env args,
    ctx.indexBits = ib → allowed ctx call → Refined.EnvHolds model ib Γ env →
    Refined.FactsHold model ib args facts → ∃ values,
      sourceSem ctx call op args (evalRegions sourceSem call env op.regions) = .ok values ∧
      Refined.FactsHold model ib values outputs := by
  intro Γ op facts outputs checked ctx call env args _width permitted _valid holds
  unfold addRule at checked
  split at checked
  · next header =>
    obtain ⟨modelled, dialect, named, regions, key⟩ := header
    split at checked
    · next left right resultType =>
      cases selected : addCap left right with
      | none => simp [selected] at checked
      | some cap =>
        simp only [selected, Option.map_some, Option.some.injEq] at checked
        subst outputs
        obtain _ | ⟨leftHolds, tail⟩ := holds
        obtain _ | ⟨rightHolds, tail⟩ := tail
        cases tail
        obtain ⟨x, rfl⟩ := initialized_float leftHolds.1
        obtain ⟨y, rfl⟩ := initialized_float rightHolds.1
        have bounded := addCap_sound selected leftHolds rightHolds
        have execution := PolicyW8A8NumericalOps.binary_exec op .add modelled dialect named regions
          resultType key ctx call env x y
        obtain ⟨base, _laws, agreement, interpretation⟩ := permitted
        have addition : (W8A8ByteModel.extend base).op (Spec.Dequant.defaultKey "arith.addf") .f32 [x, y] =
            base.op (Spec.Dequant.defaultKey "arith.addf") .f32 [x, y] := by
          simp [W8A8ByteModel.extend, Spec.Dequant.defaultKey]
        simp only [PolicyW8A8NumericalScalar.BinaryKind.name] at execution
        rw [interpretation, addition, agreement.add32 _ _ (W8A8IEEE32.finite_notNaN bounded.1)] at execution
        exact ⟨_, execution, .cons (result bounded) .nil⟩
    · cases checked
  · cases checked

def book (ib : Nat) : Contextual.Book model sourceSem ib allowed :=
  ({ rule := multiplyRule, sound := multiplyRule_sound ib } : Contextual.Book model sourceSem ib allowed).orElse
    { rule := addRule, sound := addRule_sound ib }

#print axioms multiplyRule_sound
#print axioms addRule_sound

end MerlinVerify.Check.PolicyW8A8RotaryScalar
