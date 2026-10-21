/- Proposed concrete f32 comparison instance; pending human specification review.
   Uses Lean 4.35's existing IEEE unpack/compare definitions, without a host Float conversion.
   It does not change the frozen FloatModel or the differential adapter. -/
import MerlinVerify.Rules.FrontendBoolProgram
import Init.Data.Float.Model.Unpacked.Operations.Compare
import Init.Data.Float.Model.Unpacked.Pack.Basic
import Lean.Elab.Tactic.Omega

namespace MerlinVerify.Rules.FrontendF32Comparison

open MerlinVerify FrontendBoolFptosi
open Float.Model (UnpackedFloat Format)

/-- IEEE unordered-or-not-equal is the complement of ordered equality, including all NaNs. -/
def une (x y : FBits .f32) : Bool :=
  !(UnpackedFloat.beq (UnpackedFloat.unpack Format.binary32 x)
    (UnpackedFloat.unpack Format.binary32 y))

theorem fields_zero (x : FBits .f32) :
    (x.extractLsb' 23 8 = 0 ∧ x.extractLsb' 0 23 = 0) ↔ x.toNat % 2^31 = 0 := by
  simp [BitVec.toNat_eq, BitVec.extractLsb'_toNat, Nat.shiftRight_eq_div_pow] at *
  omega

theorem une_zero (x : FBits .f32) :
    une x 0 = if x.toNat % 2^31 = 0 then false else true := by
  change BitVec 32 at x
  have hz := fields_zero x
  dsimp [FBits, FKind.width] at hz
  have hzero : UnpackedFloat.unpack Format.binary32 (0 : FBits .f32) =
      .zero .positive := by rfl
  unfold une
  rw [hzero]
  clear hzero
  simp only [UnpackedFloat.unpack, UnpackedFloat.unpackExponent,
    UnpackedFloat.unpackMantissa, UnpackedFloat.unpackSign, Format.binary32]
  dsimp only [Format.numBits] at *
  simp only [FKind.width, Nat.reduceAdd] at *
  by_cases he : x.extractLsb' 23 8 = -1#8
  all_goals by_cases hez : x.extractLsb' 23 8 = 0#8
  all_goals by_cases hm : x.extractLsb' 0 23 = 0#23
  all_goals cases hs : UnpackedFloat.Sign.ofBitVec (x.extractLsb' 31 1)
  all_goals simp only [he, hez, hm, ↓reduceIte]
  all_goals simp_all [UnpackedFloat.beq, UnpackedFloat.compare]

/-- Proposed instance override at one exact key and format. All other interpretations are kept. -/
def model (base : FloatModel) : FloatModel :=
  { base with cmp := fun key k x y =>
      if key = uneKey then
        match k with
        | .f32 => une x y
        | k => base.cmp key k x y
      else base.cmp key k x y }

theorem model_correct (base : FloatModel) : UneCorrect (model base) := by
  intro x
  simpa [model] using une_zero x

/-- Poison is handled by opSem, outside the comparison interpretation. -/
theorem model_site (base : FloatModel) (ctx : Ctx) (call : CallSem) (op : Op)
    (hm : Sem.Ops.unmodelledAttr? op.blind = none)
    (hd : Sem.Ops.dialectOf op.name = "arith") (hn : op.name = "arith.cmpf")
    (hr : Sem.Ops.Arith.resultElt op.blind = some (.int 1))
    (hk : Sem.Ops.Arith.floatKey op.blind = uneKey) (x : Option (FBits .f32)) :
    Sem.Ops.opSem { ctx with M := model base } call op
      [.scalar (FrontendBoolExecution.sourceVal x), .scalar (.some (.flt .f32 0))] [] =
      .ok [.scalar (Spec.FrontendIntent.f32ToBool x)] :=
  FrontendBoolExecution.cmp_sem_checked_option _ call op hm hd hn hr hk (model_correct base) x

/-- The accepted actual-boundary site theorem with its float premise discharged by the proposed
IEEE instance. The remaining premises identify the occurrence and current source in the evaluator;
they are not Python candidate flags. This does not approve the policy specification. -/
theorem accepted_site_exec {src tgt intended : Module}
    (hi : FrontendBoolProgram.intent? src = some intended)
    (h : FrontendBoolProgram.check src tgt = true)
    (base : FloatModel) (ctx : Ctx) (call : CallSem) (env : Env) (tail : List Op)
    (o : Op) (source z : Nat) (result : Value) (x : Option (FBits .f32))
    (hocc : FrontendBoolProgram.Occurs
      (FrontendBoolProgram.replacementCmp o source result z) intended.ops)
    (hmark : FrontendBoolProgram.repairMarked
      (FrontendBoolProgram.replacementCmp o source result z) = true)
    (hty : result.ty = .int 1)
    (hsource : env.get? source = some (.scalar (FrontendBoolExecution.sourceVal x)))
    (hfresh : source ≠ z) :
    evalOps Sem.Ops.opSem { ctx with M := model base } call env
      (FrontendBoolProgram.replacement o source result z ++ tail) =
      evalOps Sem.Ops.opSem { ctx with M := model base } call
        ((result.id, .scalar (Spec.FrontendIntent.f32ToBool x)) ::
         (z, .scalar (.some (.flt .f32 0))) :: env) tail :=
  FrontendBoolProgram.accepted_site_exec hi h _ call env tail o source z result x
    hocc hmark hty (model_correct base) hsource hfresh

/-- Every tensor coordinate obeys A2, including independently poison coordinates. -/
theorem model_tensor (base : FloatModel) (shape : List Nat)
    (x : Idx → Option (FBits .f32)) :
    FrontendBoolExecution.pointwiseResult (model base) shape x =
      Spec.FrontendIntent.f32ToBoolTensor shape x :=
  FrontendBoolExecution.pointwise_a2 _ (model_correct base) shape x

#print axioms accepted_site_exec
#print axioms model_tensor
#print axioms model_correct
#print axioms model_site

end MerlinVerify.Rules.FrontendF32Comparison
