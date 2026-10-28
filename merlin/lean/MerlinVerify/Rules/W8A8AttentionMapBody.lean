/- Exact scalar bodies for the actual softmax subtract/exp/div maps. Their output
destination is unused, so arbitrary destination poison is not treated as input data. -/
import MerlinVerify.Rules.W8A8TranscendentalModel
import MerlinVerify.Rules.W8A8SourceAlpha
import MerlinVerify.Rules.W8A8OrderedBody

namespace MerlinVerify.Rules.W8A8AttentionMapBody

open MerlinVerify

inductive BinaryKind where
  | subtract | divide
  deriving DecidableEq, Repr

def BinaryKind.name : BinaryKind → String
  | .subtract => "arith.subf"
  | .divide => "arith.divf"

def binaryBody (kind : BinaryKind) : List Op :=
  [.mk kind.name [{ id := 3, ty := .float .f32 }] [0, 1]
    [("fastmath", .enum "arith" "fastmath" "none")] [] [] [],
   .mk "linalg.yield" [] [3] [] [] [] []]

def binaryRegion (kind : BinaryKind) : Region :=
  .mk [.mk [{ id := 0, ty := .float .f32 }, { id := 1, ty := .float .f32 },
    { id := 2, ty := .float .f32 }] (binaryBody kind)]

def exponentialBody : List Op :=
  [.mk "math.exp" [{ id := 2, ty := .float .f32 }] [0]
    [("fastmath", .enum "arith" "fastmath" "none")] [] [] [],
   .mk "linalg.yield" [] [2] [] [] [] []]

def exponentialRegion : Region :=
  .mk [.mk [{ id := 0, ty := .float .f32 }, { id := 1, ty := .float .f32 }] exponentialBody]

def binary : BinaryKind → FBits .f32 → FBits .f32 → FBits .f32
  | .subtract => W8A8ArithmeticModel.subtract32
  | .divide => W8A8IEEE32.div

private theorem floating_key (name : String) (id : Nat) (arguments : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty := .float .f32 }] arguments
      [("fastmath", .enum "arith" "fastmath" "none")] [] [] []) = Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem dialect_subtract : Sem.Ops.dialectOf "arith.subf" = "arith" := by cbv
private theorem dialect_divide : Sem.Ops.dialectOf "arith.divf" = "arith" := by cbv
private theorem dialect_exponential : Sem.Ops.dialectOf "math.exp" = "math" := by cbv

theorem binary_region_sound (kind : BinaryKind) (ctx : Ctx) (call : CallSem) (env : Env)
    (left right : FBits .f32) (destination : Val) :
    evalRegion W8A8SourceSemantics.opSem ctx call env (binaryRegion kind)
      [.scalar (W8A8OrderedBody.value left), .scalar (W8A8OrderedBody.value right), .scalar destination] =
      .ok [.scalar (W8A8OrderedBody.value
        (ctx.M.op (Spec.Dequant.defaultKey kind.name) .f32 [left, right]))] := by
  cases kind <;>
    simp [evalRegion, evalBlocks, evalBlock, evalOps, binaryRegion, binaryBody, BinaryKind.name,
      W8A8OrderedBody.value, Env.bindAll, Env.getAll, Env.get?, List.lookup,
      Outcome.mapM, Outcome.ofOption, W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource,
      W8A8SourceContract.name, Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs,
      Sem.Ops.dispatch, Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      dialect_subtract, dialect_divide, Sem.Ops.Arith.sem, Sem.Ops.Arith.fltBin,
      Sem.Ops.Arith.resultElt, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, liftEager,
      floating_key, Op.isTerminator, terminatorNames, FKind.width]

theorem exponential_region_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (input : FBits .f32) (destination : Val) :
    evalRegion W8A8SourceSemantics.opSem ctx call env exponentialRegion
      [.scalar (W8A8OrderedBody.value input), .scalar destination] =
      .ok [.scalar (W8A8OrderedBody.value (ctx.M.op (Spec.Dequant.defaultKey "math.exp") .f32 [input]))] := by
  simp [evalRegion, evalBlocks, evalBlock, evalOps, exponentialRegion, exponentialBody,
    W8A8OrderedBody.value, Env.bindAll, Env.getAll, Env.get?, List.lookup,
    Outcome.mapM, Outcome.ofOption, W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource,
    W8A8SourceContract.name, Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs,
    Sem.Ops.dispatch, Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_exponential, Sem.Ops.Arith.sem, Sem.Ops.Arith.fltN,
    Sem.Ops.Arith.resultElt, Sem.Ops.Arith.pureN, liftEager,
    floating_key, Op.isTerminator, terminatorNames, FKind.width]

theorem binary_model (kind : BinaryKind) (fallback : FloatModel) (left right : FBits .f32) :
    (W8A8RoundedReal.model fallback).op (Spec.Dequant.defaultKey kind.name) .f32 [left, right] =
      binary kind left right := by
  cases kind <;> rfl

end MerlinVerify.Rules.W8A8AttentionMapBody
