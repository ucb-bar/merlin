/- Exact scalar-region execution for ordered floating contraction. Actual SSA, attributes
and tensor geometry are checked separately; this file does not define a quantization policy. -/
import MerlinVerify.Rules.W8A8OrderedDot
import MerlinVerify.Rules.W8A8SourceAlpha
import MerlinVerify.Rules.W8A8IEEEPrimitives

namespace MerlinVerify.Rules.W8A8OrderedBody

open MerlinVerify

def body (accumulatorFirst : Bool) : List Op :=
  [.mk "arith.mulf" [{ id := 3, ty := .float .f32 }] [0, 1]
     [("fastmath", .enum "arith" "fastmath" "none")] [] [] [],
   .mk "arith.addf" [{ id := 4, ty := .float .f32 }] (if accumulatorFirst then [2, 3] else [3, 2])
     [("fastmath", .enum "arith" "fastmath" "none")] [] [] [],
   .mk "linalg.yield" [] [4] [] [] [] []]

def region (accumulatorFirst : Bool) : Region :=
  .mk [.mk [{ id := 0, ty := .float .f32 }, { id := 1, ty := .float .f32 },
    { id := 2, ty := .float .f32 }] (body accumulatorFirst)]

def value (bits : FBits .f32) : Val := .some (.flt .f32 bits)

def rawStep (accumulatorFirst : Bool) (model : FloatModel) (a b accumulator : FBits .f32) : FBits .f32 :=
  let product := model.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [a, b]
  model.op (Spec.Dequant.defaultKey "arith.addf") .f32
    (if accumulatorFirst then [accumulator, product] else [product, accumulator])

def accumulatorValue : Option (FBits .f32) → Val
  | none => .poison
  | some bits => value bits

def rawResult (accumulatorFirst : Bool) (model : FloatModel) (a b : FBits .f32) :
    Option (FBits .f32) → Val
  | none => .poison
  | some accumulator => value (rawStep accumulatorFirst model a b accumulator)

private theorem dialect_mul : Sem.Ops.dialectOf "arith.mulf" = "arith" := by cbv
private theorem dialect_add : Sem.Ops.dialectOf "arith.addf" = "arith" := by cbv

private theorem floating_key (name : String) (id : Nat) (arguments : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty := .float .f32 }] arguments
      [("fastmath", .enum "arith" "fastmath" "none")] [] [] []) = Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

set_option maxRecDepth 4096 in
theorem region_optional_sound (accumulatorFirst : Bool) (ctx : Ctx) (call : CallSem) (env : Env)
    (a b : FBits .f32) (accumulator : Option (FBits .f32)) :
    evalRegion W8A8SourceSemantics.opSem ctx call env (region accumulatorFirst)
      [.scalar (value a), .scalar (value b), .scalar (accumulatorValue accumulator)] =
      .ok [.scalar (rawResult accumulatorFirst ctx.M a b accumulator)] := by
  cases accumulator <;> cases accumulatorFirst <;>
    simp [evalRegion, evalBlocks, evalBlock, evalOps, region, body, rawStep, value,
      accumulatorValue, rawResult,
      Env.bindAll, Env.getAll, Env.get?, List.lookup, Outcome.mapM, Outcome.ofOption,
      W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
      Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
      Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      dialect_mul, dialect_add, Sem.Ops.Arith.sem, Sem.Ops.Arith.fltBin,
      Sem.Ops.Arith.resultElt, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, liftEager,
      floating_key, Op.isTerminator,
      terminatorNames, FKind.width]

theorem region_sound (accumulatorFirst : Bool) (ctx : Ctx) (call : CallSem) (env : Env)
    (a b accumulator : FBits .f32) :
    evalRegion W8A8SourceSemantics.opSem ctx call env (region accumulatorFirst)
      [.scalar (value a), .scalar (value b), .scalar (value accumulator)] =
      .ok [.scalar (value (rawStep accumulatorFirst ctx.M a b accumulator))] :=
  region_optional_sound accumulatorFirst ctx call env a b (some accumulator)

theorem rawStep_explicit (accumulatorFirst : Bool) (fallback : FloatModel)
    (a b accumulator : FBits .f32) :
    rawStep accumulatorFirst (W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback)) a b accumulator =
      W8A8OrderedDot.step accumulator (a, b) := by
  cases accumulatorFirst
  · change W8A8IEEE32.add (W8A8IEEE32.mul a b) accumulator =
      W8A8IEEE32.add accumulator (W8A8IEEE32.mul a b)
    exact W8A8OrderedMagnitude.add_comm _ _
  · rfl

#print axioms region_sound
#print axioms region_optional_sound
#print axioms rawStep_explicit

end MerlinVerify.Rules.W8A8OrderedBody
