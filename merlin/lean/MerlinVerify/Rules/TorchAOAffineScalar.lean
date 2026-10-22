/- Actual scalar-region checking for the provisional TorchAO affine contract.
No acceptance result in this module verifies the enclosing tensor reduction or captured calls. -/
import MerlinVerify.Rules.TorchAOAffineContract
import MerlinVerify.Check.ProgramLemmas
import MerlinVerify.Sem.OutcomeLaws

namespace MerlinVerify.Rules.TorchAOAffineScalar
open MerlinVerify MerlinVerify.Check TorchAOAffineContract
open DequantScalar (op floatOp yield)

def constant (id : Nat) (bits : FBits .f32) : Op :=
  .mk "arith.constant" [{id, ty := .float .f32}] []
    [("value", .float bits.toNat (.float .f32))] [] [] []

def body : List Op :=
  [constant 3 0x3f800000, constant 4 0xc2fe0000, constant 5 0x42fe0000,
   floatOp "arith.divf" 6 [3, 1], floatOp "arith.mulf" 7 [0, 6],
   floatOp "math.roundeven" 8 [7],
   floatOp "arith.maximumf" 9 [8, 4], floatOp "arith.minimumf" 10 [9, 5],
   op "arith.fptosi" 11 (.int 8) [10], yield 11]

def inputs (x scale : FBits .f32) (init : Val) : Env :=
  [(0, .scalar (.some (.flt .f32 x))), (1, .scalar (.some (.flt .f32 scale))), (2, .scalar init)]

private theorem key_default (name : String) (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{id, ty}] args [] [] [] []) = key name := by
  apply congrArg (fun modifiers => ({op := name, modifiers} : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey, Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem key_none (name : String) (id : Nat) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{id, ty := .float .f32}] args
      [("fastmath", .enum "arith" "fastmath" "none")] [] [] []) = key name := by
  apply congrArg (fun modifiers => ({op := name, modifiers} : FloatOpKey))
  simp only [floatOp, Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey, Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem dialect_arith_constant : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
private theorem dialect_arith_divf : Sem.Ops.dialectOf "arith.divf" = "arith" := by cbv
private theorem dialect_arith_mulf : Sem.Ops.dialectOf "arith.mulf" = "arith" := by cbv
private theorem dialect_math_roundeven : Sem.Ops.dialectOf "math.roundeven" = "math" := by cbv
private theorem dialect_arith_maximumf : Sem.Ops.dialectOf "arith.maximumf" = "arith" := by cbv
private theorem dialect_arith_minimumf : Sem.Ops.dialectOf "arith.minimumf" = "arith" := by cbv
private theorem dialect_arith_fptosi : Sem.Ops.dialectOf "arith.fptosi" = "arith" := by cbv

private theorem roundeven_prefix : "math.roundeven".startsWith "math." = true := by cbv

set_option maxRecDepth 8192 in
set_option maxHeartbeats 2000000 in
/-- No float law: the actual scalar evaluator computes exactly this ordered expression,
including a poison result when the conversion interpretation rejects the clamped float. -/
theorem body_sound (ctx : Ctx) (call : CallSem) (x scale : FBits .f32) (init : Val) :
    evalOps Sem.Ops.opSem ctx call (inputs x scale init) body =
      .ok [.scalar (quantizeWithoutZero ctx.M x scale)] := by
  simp (maxSteps := 1000000) [evalOps, body, constant, inputs, op, floatOp, yield,
    Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM, Outcome.ofOption,
    Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.attr?, Op.name, Op.results, Op.attrs, Op.props, Op.successors, roundeven_prefix, dialect_arith_constant, dialect_arith_divf, dialect_arith_mulf, dialect_math_roundeven, dialect_arith_maximumf, dialect_arith_minimumf, dialect_arith_fptosi, Sem.Ops.Arith.sem,
    Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt,
    Sem.Ops.Arith.fltBin, Sem.Ops.Arith.fltN, Sem.Ops.Arith.widthOf, Ty.bitWidth?,
    scalarOfAttr, liftEager, key_default, key_none, quantizeWithoutZero, clamp, asInt, f,
    Sem.Ops.Arith.bv, Op.isTerminator, terminatorNames, FKind.width, hasUnit, RVal.canonize,
    List.range_succ, Function.comp_def, List.mapM_cons, List.mapM_nil]
  all_goals split <;> simp_all

def checkBody (args : List Value) (ops : List Op) : Bool :=
  decide (args.map (·.ty) = [.float .f32, .float .f32, .int 8]) &&
    freshIds [] (args.map (·.id)) && alphaOps ([0, 1, 2].zip (args.map (·.id))) body ops

/-- The body certificate supplies no arithmetic facts. The actual IR is alpha checked against
the proved scalar program, including attributes, SSA freshness, constants and returned use. -/
theorem checkBody_sound {args : List Value} {ops : List Op} (h : checkBody args ops = true)
    (ctx : Ctx) (call : CallSem) (x scale : FBits .f32) (init : Val)
    (env : Env) (hc : Corr ([0, 1, 2].zip (args.map (·.id))) (inputs x scale init) env) :
    evalOps Sem.Ops.opSem ctx call env ops = .ok [.scalar (quantizeWithoutZero ctx.M x scale)] := by
  simp only [checkBody, Bool.and_eq_true] at h
  rw [← alphaOps_sound call call _ _ _ ctx _ _ h.2 hc (fun _ _ => rfl)]
  exact body_sound ctx call x scale init

/-- Conditional connection to the versioned reference: the omitted zero-point addition
is visible in the theorem's remaining hypothesis, never admitted as an axiom. -/
theorem checkBody_reference {args : List Value} {ops : List Op} (h : checkBody args ops = true)
    (ctx : Ctx) (hz : ZeroPointElisionCorrect ctx.M) (call : CallSem)
    (x scale : FBits .f32) (init : Val) (env : Env)
    (hc : Corr ([0, 1, 2].zip (args.map (·.id))) (inputs x scale init) env) :
    evalOps Sem.Ops.opSem ctx call env ops = .ok [.scalar (quantize ctx.M x scale)] := by
  rw [checkBody_sound h ctx call x scale init env hc, hz]

#print axioms checkBody_reference
end MerlinVerify.Rules.TorchAOAffineScalar
