/- The existing untrusted-input tensor checker also establishes nonpoison quantization under
the explicit concrete-byte model and declared non-NaN data domain. No new acceptance flag,
float axiom or weakening of the whole-policy checker is introduced. -/
import MerlinVerify.Check.PolicyW8A8Pointwise
import MerlinVerify.Rules.W8A8ByteTensor

namespace MerlinVerify.Check.PolicyW8A8ByteDefined

open MerlinVerify MerlinVerify.Sem.Ops.Linalg Rules PolicyW8A8Geometry

def QuantizeDefined (known : List ConstantFacts.Fact) (shape dims : List Nat)
    (maps : List AffineMap) (iters : List IterKind) (actual : Region) : Prop :=
  ∀ env, BindingFacts.Valid ConstantFacts.model known env →
    ∀ (ctx : Ctx) (call : CallSem) (get inverse : Idx → FBits .f32) (initial : Idx → Val),
    W8A8ByteTensor.Domain ctx.M shape (parameterShape shape dims)
      (parameterMap shape.length dims) get inverse →
    ∃ output, genericSem { ctx with M := W8A8ByteModel.extend ctx.M } maps iters
      [(W8A8FloatTensor.floatInput shape (fun p => some (get p))).value,
       (W8A8FloatTensor.floatInput (parameterShape shape dims) (fun p => some (inverse p))).value]
      [.tensor ⟨shape, .int 8, initial⟩] (fun c => evalRegion Sem.Ops.opSem c call env actual) =
        .ok [.tensor output] ∧ (RVal.tensor output).Defined

theorem checkQuantize_defined_sound {known : List ConstantFacts.Fact} {shape dims : List Nat}
    {hi lo : Nat} {maps : List AffineMap} {iters : List IterKind} {actual : Region}
    (checked : PolicyW8A8Pointwise.checkQuantize known shape dims hi lo maps iters actual = true) :
    QuantizeDefined known shape dims maps iters actual := by
  intro env valid ctx call get inverse initial nonNaN
  refine ⟨W8A8ByteTensor.result ctx.M shape (parameterShape shape dims)
    (parameterMap shape.length dims) get inverse, ?_, W8A8ByteTensor.result_defined nonNaN⟩
  exact PolicyW8A8Pointwise.checkQuantize_sound checked env valid
    { ctx with M := W8A8ByteModel.extend ctx.M } call (fun p => some (get p))
      (fun p => some (inverse p)) initial

#print axioms checkQuantize_defined_sound

end MerlinVerify.Check.PolicyW8A8ByteDefined
