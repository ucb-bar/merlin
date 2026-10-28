/-
Scalar lowering correctness, the first obligation of the quant_ext → linalg.generic rule.
The statements relate execution by the actual opSem to the independently written scalar contract.
They quantify over every FloatModel, input bit pattern, external-call interpretation, context and
unused destination value. They do not establish tensor indexing or a C1 site rule yet.
-/
import MerlinVerify.Sem.Ops
import MerlinVerify.Spec.Dequant
import MerlinVerify.Check.Alpha
import MerlinVerify.Check.AlphaFast

namespace MerlinVerify.Rules.DequantScalar

open MerlinVerify

def op (name : String) (id : Nat) (ty : Ty) (args : List Nat) : Op :=
  .mk name [{ id, ty }] args [] [] [] []

def yield (id : Nat) : Op := .mk "linalg.yield" [] [id] [] [] [] []

/-- MLIR materializes the default fast-math property in the exported attribute dictionary. -/
def floatOp (name : String) (id : Nat) (args : List Nat) : Op :=
  .mk name [{ id, ty := .float .f32 }] args
    [("fastmath", .enum "arith" "fastmath" "none")] [] [] []

def intSub (id : Nat) (args : List Nat) : Op :=
  .mk "arith.subi" [{ id, ty := .int 32 }] args
    [("overflowFlags", .enum "arith" "overflow" "none")] [] [] []

/-- Block arguments: q, scale, zero point, unused destination element. -/
def perTensorBody : List Op :=
  [op "arith.sitofp" 4 (.float .f32) [0],
   op "arith.sitofp" 5 (.float .f32) [2],
   floatOp "arith.subf" 6 [4, 5], floatOp "arith.mulf" 7 [6, 1], yield 7]

/-- The integer-subtraction replacement required by the per-channel source contract. -/
def perChannelBody : List Op :=
  [op "arith.extsi" 4 (.int 32) [0], intSub 5 [4, 2],
   op "arith.sitofp" 6 (.float .f32) [5],
   floatOp "arith.mulf" 7 [6, 1], yield 7]

def inputs (q : BitVec 8) (z : BitVec 32) (s : FBits .f32) (init : Val) : Env :=
  [(0, .scalar (.some (.bits 8 q))), (1, .scalar (.some (.flt .f32 s))),
   (2, .scalar (.some (.bits 32 z))), (3, .scalar init)]

private theorem dialect_sitofp : Sem.Ops.dialectOf "arith.sitofp" = "arith" := by cbv
private theorem dialect_subf : Sem.Ops.dialectOf "arith.subf" = "arith" := by cbv
private theorem dialect_mulf : Sem.Ops.dialectOf "arith.mulf" = "arith" := by cbv
private theorem dialect_extsi : Sem.Ops.dialectOf "arith.extsi" = "arith" := by cbv
private theorem dialect_subi : Sem.Ops.dialectOf "arith.subi" = "arith" := by cbv

private theorem floatKey_default (name : String) (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty }] args [] [] [] []) = Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey, Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem floatKey_explicitNone (name : String) (id : Nat) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty := .float .f32 }] args
      [("fastmath", .enum "arith" "fastmath" "none")] [] [] []) = Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem overflow_none (id : Nat) (args : List Nat) :
    overflowFlags (.mk "arith.subi" [{ id, ty := .int 32 }] args
      [("overflowFlags", .enum "arith" "overflow" "none")] [] [] []) = [] := by
  dsimp [overflowFlags, attrEnum?, Op.attr?, Op.props, Op.attrs]
  cbv

set_option maxRecDepth 4096 in
/-- Exact scalar-body correctness for per-tensor dequantization, with no float assumptions. -/
theorem perTensor_body_sound (ctx : Ctx) (call : CallSem)
    (q : BitVec 8) (z : BitVec 32) (s : FBits .f32) (init : Val) :
    evalOps Sem.Ops.opSem ctx call (inputs q z s init) perTensorBody =
      .ok [.scalar (.some (.flt .f32 (Spec.Dequant.perTensor ctx.M q z s)))] := by
  simp [evalOps, perTensorBody, op, floatOp, yield, inputs, Env.getAll, Env.get?, Env.bindAll,
    List.lookup, Outcome.mapM, Outcome.ofOption, Sem.Ops.opSem, Sem.Ops.unmodelledAttr?,
    Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_sitofp, dialect_subf, dialect_mulf, Sem.Ops.Arith.sem,
    Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt,
    liftEager, Sem.Ops.Arith.fltBin, floatKey_default, floatKey_explicitNone,
    Spec.Dequant.perTensor, Op.isTerminator, terminatorNames]

set_option maxRecDepth 4096 in
/-- Exact scalar-body correctness for the integer-subtraction per-channel replacement. -/
theorem perChannel_body_sound (ctx : Ctx) (call : CallSem)
    (q : BitVec 8) (z : BitVec 32) (s : FBits .f32) (init : Val) :
    evalOps Sem.Ops.opSem ctx call (inputs q z s init) perChannelBody =
      .ok [.scalar (.some (.flt .f32 (Spec.Dequant.perChannel ctx.M q z s)))] := by
  simp [evalOps, perChannelBody, op, floatOp, intSub, yield, inputs, Env.getAll, Env.get?, Env.bindAll,
    List.lookup, Outcome.mapM, Outcome.ofOption, Sem.Ops.opSem, Sem.Ops.unmodelledAttr?,
    Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_extsi, dialect_subi, dialect_sitofp, dialect_mulf, Sem.Ops.Arith.sem,
    Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt,
    liftEager, Sem.Ops.Arith.fltBin, floatKey_default, floatKey_explicitNone,
    Sem.Ops.Arith.intUn, Sem.Ops.Arith.intBin, Sem.Ops.Arith.arithWithFlags,
    Sem.Ops.Arith.widthOf, Ty.bitWidth?, Sem.Ops.Arith.bv, overflow_none,
    Spec.Dequant.perChannel, Op.isTerminator, terminatorNames]

/-- A typed integer or float operand, with `none` representing poison. -/
def intValue (w : Nat) (x : Option (BitVec w)) : Val :=
  match x with | none => .poison | some x => .some (.bits w x)

def floatValue (x : Option (FBits .f32)) : Val :=
  match x with | none => .poison | some x => .some (.flt .f32 x)

def poisonInputs (q : Option (BitVec 8)) (z : Option (BitVec 32))
    (s : Option (FBits .f32)) (init : Val) : Env :=
  [(0, .scalar (intValue 8 q)), (1, .scalar (floatValue s)),
   (2, .scalar (intValue 32 z)), (3, .scalar init)]

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
/-- Exact execution, including all eight combinations of defined/poison arithmetic operands. -/
theorem body_poison_sound (channel : Bool) (ctx : Ctx) (call : CallSem)
    (q : Option (BitVec 8)) (z : Option (BitVec 32)) (s : Option (FBits .f32)) (init : Val) :
    evalOps Sem.Ops.opSem ctx call (poisonInputs q z s init)
      (if channel then perChannelBody else perTensorBody) =
      .ok [.scalar (Spec.Dequant.scalarValue channel ctx.M q z s)] := by
  cases channel <;> cases q <;> cases z <;> cases s
  all_goals first
    | simpa only [poisonInputs, intValue, floatValue, inputs, Spec.Dequant.scalarValue,
        Spec.Dequant.scalar, Bool.false_eq_true, ↓reduceIte] using
        (perTensor_body_sound ctx call _ _ _ init)
    | simpa only [poisonInputs, intValue, floatValue, inputs, Spec.Dequant.scalarValue,
        Spec.Dequant.scalar, ↓reduceIte] using
        (perChannel_body_sound ctx call _ _ _ init)
    | simp [poisonInputs, intValue, floatValue, Spec.Dequant.scalarValue,
      Spec.Dequant.scalar,
      evalOps, perTensorBody, perChannelBody, op, floatOp, intSub, yield,
      Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM, Outcome.ofOption,
      Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
      Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      dialect_extsi, dialect_subi, dialect_sitofp, dialect_subf, dialect_mulf,
      Sem.Ops.Arith.sem, Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp,
      Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt, liftEager, Sem.Ops.Arith.fltBin,
      floatKey_default, floatKey_explicitNone, Sem.Ops.Arith.intUn,
      Sem.Ops.Arith.intBin, Sem.Ops.Arith.arithWithFlags, Sem.Ops.Arith.widthOf,
      Ty.bitWidth?, Sem.Ops.Arith.bv, overflow_none, Spec.Dequant.perTensor,
      Spec.Dequant.perChannel, Op.isTerminator, terminatorNames]

def body (channel : Bool) : List Op := if channel then perChannelBody else perTensorBody

def argMap (args : List Value) : Check.Rho := [0, 1, 2, 3].zip (args.map (·.id))

/-- Diagnostic scalar-body checker only. It does not check the enclosing tensor operation. -/
def checkBody (channel : Bool) (args : List Value) (ops : List Op) : Bool :=
  (decide (args.map (·.ty) = [.int 8, .float .f32, .int 32, .float .f32]) &&
    Check.freshIds [] (args.map (·.id))) && Check.alphaOps (argMap args) (body channel) ops

/-- An accepted, consistently renamed scalar body executes the specified ordered arithmetic. -/
theorem checkBody_sound {channel : Bool} {args : List Value} {ops : List Op}
    (h : checkBody channel args ops = true) (ctx : Ctx) (call : CallSem)
    (q : BitVec 8) (z : BitVec 32) (s : FBits .f32) (init : Val) (env : Env)
    (hc : Check.Corr (argMap args) (inputs q z s init) env) :
    evalOps Sem.Ops.opSem ctx call env ops =
      .ok [.scalar (.some (.flt .f32 (Spec.Dequant.scalar channel ctx.M q z s)))] := by
  simp only [checkBody, Bool.and_eq_true] at h
  have ha := h.2
  have he := Check.alphaOps_sound call call (argMap args) (body channel) ops ctx
    (inputs q z s init) env ha hc (fun _ _ => rfl)
  rw [← he]
  cases channel <;> simp [body, Spec.Dequant.scalar, perTensor_body_sound, perChannel_body_sound]

/-- The accepted body has the source contract even when any arithmetic operands are poison. -/
theorem checkBody_poison_sound {channel : Bool} {args : List Value} {ops : List Op}
    (h : checkBody channel args ops = true) (ctx : Ctx) (call : CallSem)
    (q : Option (BitVec 8)) (z : Option (BitVec 32)) (s : Option (FBits .f32))
    (init : Val) (env : Env)
    (hc : Check.Corr (argMap args) (poisonInputs q z s init) env) :
    evalOps Sem.Ops.opSem ctx call env ops =
      .ok [.scalar (Spec.Dequant.scalarValue channel ctx.M q z s)] := by
  simp only [checkBody, Bool.and_eq_true] at h
  have he := Check.alphaOps_sound call call (argMap args) (body channel) ops ctx
    (poisonInputs q z s init) env h.2 hc (fun _ _ => rfl)
  rw [← he]
  exact body_poison_sound channel ctx call q z s init

end MerlinVerify.Rules.DequantScalar
