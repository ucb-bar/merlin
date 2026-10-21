/- Explicit local obstructions to full-model definedness; no additional numeric law is assumed. -/
import MerlinVerify.Rules.FrontendRuntimeChain

namespace MerlinVerify.Rules.FrontendRuntimeObligations
open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops

/-- This is an allowed frozen FloatModel. Approving f32 UNE does not constrain toInt. -/
def absentConversions (base : FloatModel) : FloatModel :=
  {base with toInt := fun _ _ _ _ => none}

def cast8 : Op := .mk "arith.fptosi" [{id := 0, ty := .int 8}] [1] [] [] [] []

/-- An accepted actual i8 cast produces poison in an allowed interpretation, even from an
initialized f32 operand. This is a local conversion obstruction, not a full-model counterexample. -/
theorem cast8_poison {actual : Op} (h : opHeadEq cast8 actual = true)
    (limit : Nat) (base : FloatModel) (ctx : Ctx) (call : CallSem) (x : FBits .f32) :
    FrontendMaskRuntime.sem limit {ctx with M := FrontendF32Comparison.model (absentConversions base)}
      call actual [.scalar (.some (.flt .f32 x))] [] = .ok [.scalar .poison] := by
  rw [FrontendMaskRuntime.sem_blind (blind_of_opHeadEq h).symm]
  simp [FrontendMaskRuntime.sem, cast8, opSem, dispatch, show dialectOf "arith.fptosi" = "arith" by cbv, Outcome.ofOption, Outcome.bind, unmodelledAttr?, interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors, Arith.sem, Arith.widthOf,
    Arith.resultElt, Ty.bitWidth?, Arith.unOp, Arith.pureN, liftEager,
    FrontendF32Comparison.model, absentConversions]

/-- BF16 is merely the stored element type here. No BF16 arithmetic is interpreted or proved. -/
def tokenExtract : Op :=
  .mk "tensor.extract" [{id := 0, ty := .float .bf16}] [1,2,3] [] [] [] []

/-- Initialization alone does not bound token indices. An index at the table's extent is UB
under the existing tensor interpretation. The candidate retains that interpretation unchanged. -/
theorem token_upper_ub {actual : Op} (h : opHeadEq tokenExtract actual = true)
    (limit : Nat) (ctx : Ctx) (call : CallSem) (t : Tensor) (hs : t.shape = [49280,960]) :
    FrontendMaskRuntime.sem limit ctx call actual
      [.tensor t, FrontendMaskProgram.index 49280, FrontendMaskProgram.index 0] [] = .ub := by
  rw [FrontendMaskRuntime.sem_blind (blind_of_opHeadEq h).symm]
  simp [FrontendMaskRuntime.sem, tokenExtract, opSem, dispatch, show dialectOf "tensor.extract" = "tensor" by cbv, unmodelledAttr?, interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors, Tensor.sem,
    FrontendMaskProgram.index, Tensor.indexVal?, hs]

/-- A negative initialized token is the second independent missing domain restriction. -/
theorem token_negative_ub {actual : Op} (h : opHeadEq tokenExtract actual = true)
    (limit : Nat) (ctx : Ctx) (call : CallSem) (t : Tensor) (hs : t.shape = [49280,960]) :
    FrontendMaskRuntime.sem limit ctx call actual
      [.tensor t, .scalar (.some (.bits 64 (-1))), FrontendMaskProgram.index 0] [] = .ub := by
  rw [FrontendMaskRuntime.sem_blind (blind_of_opHeadEq h).symm]
  simp [FrontendMaskRuntime.sem, tokenExtract, opSem, dispatch, show dialectOf "tensor.extract" = "tensor" by cbv, unmodelledAttr?, interpretedAttrs,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors, Tensor.sem,
    FrontendMaskProgram.index, Tensor.indexVal?, hs]

#print axioms cast8_poison
#print axioms token_upper_ub
#print axioms token_negative_ub
end MerlinVerify.Rules.FrontendRuntimeObligations
