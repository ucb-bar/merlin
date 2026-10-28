import MerlinVerify.Spec.Quantize
import MerlinVerify.Rules.DequantScalar
import MerlinVerify.Sem.OutcomeLaws

namespace MerlinVerify.Rules.QuantizeScalar

open MerlinVerify
open MerlinVerify.Rules.DequantScalar

def floating (name : String) (id : Nat) (kind : FKind) (args : List Nat) : Op :=
  .mk name [⟨id, .float kind⟩] args
    [("fastmath", .enum "arith" "fastmath" "none")] [] [] []

def inverseBody : List Op :=
  [.mk "arith.constant" [⟨2, .float .f64⟩] []
     [("value", .float 0x3ff0000000000000 (.float .f64))] [] [] [],
   op "arith.extf" 3 (.float .f64) [0],
   floating "arith.divf" 4 .f64 [2, 3],
   op "arith.truncf" 5 (.float .f32) [4], yield 5]

def body : List Op :=
  [floating "arith.mulf" 4 .f32 [0, 1], floating "math.roundeven" 5 .f32 [4],
   op "arith.sitofp" 6 (.float .f32) [2], floating "arith.addf" 7 .f32 [5, 6],
   floating "arith.maximumf" 8 .f32 [7, 100], floating "arith.minimumf" 9 .f32 [8, 101],
   op "arith.fptosi" 10 (.int 8) [9], yield 10]

def bodyValue (w : Nat) (M : FloatModel) (x inv : Option (FBits .f32))
    (zero : Option (BitVec w)) (lower upper : FBits .f32) : Val :=
  match x, inv, zero with
  | some x, some inv, some zero =>
    match Spec.Quantize.withInverse w M x inv zero lower upper with
    | some q => .some (.bits 8 q)
    | none => .poison
  | _, _, _ => .poison

private theorem floatKey_plain (name : String) (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [⟨id, ty⟩] args [] [] [] []) =
      Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem floatKey_floating (name : String) (id : Nat) (kind : FKind) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [⟨id, .float kind⟩] args
      [("fastmath", .enum "arith" "fastmath" "none")] [] [] []) = Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem d_constant : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
private theorem d_extf : Sem.Ops.dialectOf "arith.extf" = "arith" := by cbv
private theorem d_divf : Sem.Ops.dialectOf "arith.divf" = "arith" := by cbv
private theorem d_truncf : Sem.Ops.dialectOf "arith.truncf" = "arith" := by cbv
private theorem d_mulf : Sem.Ops.dialectOf "arith.mulf" = "arith" := by cbv
private theorem d_addf : Sem.Ops.dialectOf "arith.addf" = "arith" := by cbv
private theorem d_sitofp : Sem.Ops.dialectOf "arith.sitofp" = "arith" := by cbv
private theorem d_maximumf : Sem.Ops.dialectOf "arith.maximumf" = "arith" := by cbv
private theorem d_minimumf : Sem.Ops.dialectOf "arith.minimumf" = "arith" := by cbv
private theorem d_fptosi : Sem.Ops.dialectOf "arith.fptosi" = "arith" := by cbv
private theorem d_roundeven : Sem.Ops.dialectOf "math.roundeven" = "math" := by cbv
private theorem p_roundeven : "math.roundeven".startsWith "math." = true := by cbv

macro "quant_scalar_simp" : tactic => `(tactic|
  simp [evalOps, inverseBody, body, floating, op, yield, bodyValue,
    Spec.Quantize.inverse, Spec.Quantize.withInverse, floatValue, intValue,
    Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM, Outcome.ofOption,
    Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    d_constant, d_extf, d_divf, d_truncf, d_mulf, d_addf, d_sitofp,
    d_maximumf, d_minimumf, d_fptosi, d_roundeven, p_roundeven,
    evalRegions, Sem.Ops.Arith.sem,
    Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN,
    Sem.Ops.Arith.resultElt, Sem.Ops.Arith.fltN, Sem.Ops.Arith.fltBin, liftEager,
    Sem.Ops.Arith.widthOf, Sem.Ops.Arith.bv, Ty.bitWidth?,
    floatKey_plain, floatKey_floating, Op.isTerminator, terminatorNames,
    scalarOfAttr, hasUnit, Op.attr?, FKind.width, RVal.canonize,
    List.range_succ, Function.comp_def, List.mapM_cons, List.mapM_nil])

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
theorem inverse_sound (ctx : Ctx) (call : CallSem) (scale : Option (FBits .f32)) (old : Val) :
    evalOps Sem.Ops.opSem ctx call [(0, .scalar (floatValue scale)), (1, .scalar old)] inverseBody =
      .ok [.scalar (floatValue (scale.map (Spec.Quantize.inverse ctx.M)))] := by
  cases scale <;> quant_scalar_simp

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
theorem body_sound (w : Nat) (ctx : Ctx) (call : CallSem)
    (x inv : Option (FBits .f32)) (zero : Option (BitVec w)) (lower upper : FBits .f32) (old : Val) :
    evalOps Sem.Ops.opSem ctx call
      [(0, .scalar (floatValue x)), (1, .scalar (floatValue inv)),
       (2, .scalar (intValue w zero)), (3, .scalar old),
       (100, .scalar (.some (.flt .f32 lower))), (101, .scalar (.some (.flt .f32 upper)))] body =
      .ok [.scalar (bodyValue w ctx.M x inv zero lower upper)] := by
  cases x <;> cases inv <;> cases zero <;> quant_scalar_simp
  split <;> simp_all

theorem compose (w : Nat) (M : FloatModel) (x scale : Option (FBits .f32))
    (zero : Option (BitVec w)) (lower upper : FBits .f32) :
    bodyValue w M x (scale.map (Spec.Quantize.inverse M)) zero lower upper =
      Spec.Quantize.scalarValue w M x scale zero lower upper := by
  cases x <;> cases scale <;> cases zero <;> rfl

#print axioms inverse_sound
#print axioms body_sound
#print axioms compose

end MerlinVerify.Rules.QuantizeScalar
