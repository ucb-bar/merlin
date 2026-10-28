/- Checked actual softmax map bodies, unused poisoned destinations and genuine
nonempty tensor execution. This is not admission of the captured mask/SSA chain. -/
import MerlinVerify.Rules.W8A8AttentionMapTensor
import MerlinVerify.Rules.W8A8MaskedMaximum

open MerlinVerify Rules W8A8AttentionMapBody Sem.Ops.Linalg

namespace W8A8AttentionMapControls

def wrongBinary (kind : BinaryKind) (flags : String) (operands : List Nat) : Region :=
  .mk [.mk [{ id := 0, ty := .float .f32 }, { id := 1, ty := .float .f32 }, { id := 2, ty := .float .f32 }]
    [.mk kind.name [{ id := 3, ty := .float .f32 }] operands
       [("fastmath", .enum "arith" "fastmath" flags)] [] [] [],
     .mk "linalg.yield" [] [3] [] [] [] []]]

theorem exact_body_acceptances :
    Check.PolicyW8A8AttentionMapBody.checkBinary .subtract (binaryRegion .subtract) = true ∧
    Check.PolicyW8A8AttentionMapBody.checkBinary .divide (binaryRegion .divide) = true ∧
    Check.PolicyW8A8AttentionMapBody.checkExponential exponentialRegion = true := by cbv

theorem valid_binary_mutation_refusals :
    Check.PolicyW8A8AttentionMapBody.checkBinary .subtract (wrongBinary .subtract "none" [1, 0]) = false ∧
    Check.PolicyW8A8AttentionMapBody.checkBinary .divide (wrongBinary .divide "none" [1, 1]) = false ∧
    Check.PolicyW8A8AttentionMapBody.checkBinary .subtract (wrongBinary .subtract "fast" [0, 1]) = false ∧
    Check.PolicyW8A8AttentionMapBody.checkBinary .divide (binaryRegion .subtract) = false := by cbv

theorem unused_destination_can_be_poison (ctx : Ctx) (call : CallSem) (env : Env) (base : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model base) (left right : FBits .f32) :
    evalRegion W8A8SourceSemantics.opSem ctx call env (binaryRegion .subtract)
      [.scalar (W8A8OrderedBody.value left), .scalar (W8A8OrderedBody.value right), .scalar .poison] =
      .ok [.scalar (W8A8OrderedBody.value (W8A8ArithmeticModel.subtract32 left right))] :=
  Check.PolicyW8A8AttentionMapBody.checkBinary_sound (kind := .subtract) (actual := binaryRegion .subtract)
    (by cbv) ctx call env base interpretation left right .poison

theorem actual_exponential_zero (ctx : Ctx) (call : CallSem) (env : Env) (base : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model base) :
    evalRegion W8A8SourceSemantics.opSem ctx call env exponentialRegion
      [.scalar (W8A8OrderedBody.value 0), .scalar .poison] =
      .ok [.scalar (W8A8OrderedBody.value 0x3f800000)] := by
  have executed := Check.PolicyW8A8AttentionMapBody.checkExponential_sound (actual := exponentialRegion) (by cbv)
    ctx call env base interpretation 0 .poison
  simpa only [W8A8RoundedReal.exponential_zero] using executed

theorem actual_masked_exponential_zero (ctx : Ctx) (call : CallSem) (env : Env) (base : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model base) :
    evalRegion W8A8SourceSemantics.opSem ctx call env exponentialRegion
      [.scalar (W8A8OrderedBody.value W8A8MaskedMaximum.negativeInfinity), .scalar .poison] =
      .ok [.scalar (W8A8OrderedBody.value 0)] := by
  have executed := Check.PolicyW8A8AttentionMapBody.checkExponential_sound (actual := exponentialRegion) (by cbv)
    ctx call env base interpretation W8A8MaskedMaximum.negativeInfinity .poison
  simpa only [W8A8RoundedReal.exponential_negative_infinity] using executed

def ones : Idx → Val := (Tensor.canon [1] (.float .f32) (fun _ => W8A8OrderedBody.value 0x3f800000)).get

theorem complete_nonempty_tensor_division (ctx : Ctx) (call : CallSem) (env : Env) (base : FloatModel)
    (interpretation : ctx.M = W8A8RoundedReal.model base) :
    genericSem ctx [identityMap 1, identityMap 1, identityMap 1] [.parallel]
      [.tensor ⟨[1], .float .f32, ones⟩, .tensor ⟨[1], .float .f32, ones⟩]
      [.tensor ⟨[1], .float .f32, fun _ => .poison⟩]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env (binaryRegion .divide)) =
      .ok [.tensor (Tensor.canon [1] (.float .f32) (fun _ => W8A8OrderedBody.value 0x3f800000))] := by
  have geometry : Check.StructuredOk [identityMap 1, identityMap 1, identityMap 1] 1
      [.tensor [1] (.float .f32), .tensor [1] (.float .f32), .tensor [1] (.float .f32)] [1] := by
    apply Check.structuredRanges?_spec
    decide
  have initialized : ∀ cell, Idx.InBounds [1] cell → ones cell = W8A8OrderedBody.value 0x3f800000 := by
    intro cell bounds
    simp only [ones, Tensor.canon, bounds, ↓reduceIte]
  have executed := W8A8AttentionMapTensor.binary_sound (kind := .divide) (actual := binaryRegion .divide) (by cbv)
    ctx call env base interpretation [1] [1] [1] (identityMap 1) (identityMap 1) [.parallel]
    geometry ones ones (fun _ => .poison) (fun _ => 0x3f800000) (fun _ => 0x3f800000) initialized initialized
  exact executed

#print axioms exact_body_acceptances
#print axioms valid_binary_mutation_refusals
#print axioms unused_destination_can_be_poison
#print axioms actual_exponential_zero
#print axioms actual_masked_exponential_zero
#print axioms complete_nonempty_tensor_division

end W8A8AttentionMapControls
