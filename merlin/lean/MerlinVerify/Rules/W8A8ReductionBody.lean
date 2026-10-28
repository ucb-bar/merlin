/- Exact execution of initialized f32 max/sum reduction regions. Operand order,
default numerical keys and SSA are explicit; no associativity or native claim. -/
import MerlinVerify.Rules.W8A8SourceAlpha
import MerlinVerify.Rules.W8A8OrderedCells

namespace MerlinVerify.Rules.W8A8ReductionBody

open MerlinVerify

inductive Kind where
  | maximum | sum
  deriving DecidableEq, Repr

def Kind.name : Kind → String
  | .maximum => "arith.maximumf"
  | .sum => "arith.addf"

def body (kind : Kind) : List Op :=
  [.mk kind.name [{ id := 2, ty := .float .f32 }] [0, 1]
    [("fastmath", .enum "arith" "fastmath" "none")] [] [] [],
   .mk "linalg.yield" [] [2] [] [] [] []]

def region (kind : Kind) : Region :=
  .mk [.mk [{ id := 0, ty := .float .f32 }, { id := 1, ty := .float .f32 }] (body kind)]

def combine : Kind → FBits .f32 → FBits .f32 → FBits .f32
  | .maximum => W8A8ByteFloat.maximum
  | .sum => W8A8IEEE32.add

def rawResult (kind : Kind) (model : FloatModel) (input : FBits .f32) : Option (FBits .f32) → Val
  | none => .poison
  | some accumulator => W8A8OrderedBody.value (model.op (Spec.Dequant.defaultKey kind.name) .f32 [input, accumulator])

private theorem floating_key (name : String) (id : Nat) (arguments : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty := .float .f32 }] arguments
      [("fastmath", .enum "arith" "fastmath" "none")] [] [] []) = Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem dialect_maximum : Sem.Ops.dialectOf "arith.maximumf" = "arith" := by cbv
private theorem dialect_sum : Sem.Ops.dialectOf "arith.addf" = "arith" := by cbv

theorem region_optional_sound (kind : Kind) (ctx : Ctx) (call : CallSem) (env : Env)
    (input : FBits .f32) (accumulator : Option (FBits .f32)) :
    evalRegion W8A8SourceSemantics.opSem ctx call env (region kind)
      [.scalar (W8A8OrderedBody.value input), .scalar (W8A8OrderedBody.accumulatorValue accumulator)] =
      .ok [.scalar (rawResult kind ctx.M input accumulator)] := by
  cases accumulator <;> cases kind <;>
    simp [evalRegion, evalBlocks, evalBlock, evalOps, region, body, Kind.name,
      W8A8OrderedBody.value, W8A8OrderedBody.accumulatorValue, rawResult,
      Env.bindAll, Env.getAll, Env.get?, List.lookup, Outcome.mapM, Outcome.ofOption,
      W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
      Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
      Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      dialect_maximum, dialect_sum, Sem.Ops.Arith.sem, Sem.Ops.Arith.fltBin,
      Sem.Ops.Arith.resultElt, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, liftEager,
      floating_key, Op.isTerminator, terminatorNames, FKind.width]

theorem model_combine (kind : Kind) (fallback : FloatModel) (input accumulator : FBits .f32) :
    (W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback)).op
      (Spec.Dequant.defaultKey kind.name) .f32 [input, accumulator] = combine kind input accumulator := by
  cases kind <;> rfl

end MerlinVerify.Rules.W8A8ReductionBody
