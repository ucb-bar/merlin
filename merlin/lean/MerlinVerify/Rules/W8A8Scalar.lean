/-
Ordered scalar execution for Merlin's W8A8 quantization pipeline. These are implementation
component contracts, not the independently reviewed whole-program policy specification.
All statements quantify over every FloatModel; no numerical identity is assumed.
-/
import MerlinVerify.Sem.Ops
import MerlinVerify.Sem.OutcomeLaws
import MerlinVerify.Spec.Dequant
import MerlinVerify.Rules.W8A8Scale

namespace MerlinVerify.Rules.W8A8Scalar

open MerlinVerify

def floatValue : Option (FBits .f32) → Val
  | none => .poison
  | some x => .some (.flt .f32 x)

def intValue : Option (BitVec 32) → Val
  | none => .poison
  | some x => .some (.bits 32 x)

def absMax (M : FloatModel) (x acc : Option (FBits .f32)) : Val :=
  floatValue (do
    let x ← x
    let acc ← acc
    pure (M.op (Spec.Dequant.defaultKey "arith.maximumf") .f32
      [M.op (Spec.Dequant.defaultKey "math.absf") .f32 [x], acc]))

def reciprocal (M : FloatModel) (scale : Option (FBits .f32)) : Val :=
  floatValue (scale.map fun s =>
    M.op (Spec.Dequant.defaultKey "arith.divf") .f32 [0x3f800000#32, s])

/-- The implemented scale, including its explicit zero-row branch. -/
def scale (M : FloatModel) (amax : Option (FBits .f32)) : Val :=
  floatValue (amax.map (W8A8Scale.guarded M))

/-- The destination element is unused; conversion may yield poison in the abstract float model. -/
def quantize (M : FloatModel) (x inv : Option (FBits .f32)) (upper lower : FBits .f32) : Val :=
  match x, inv with
  | some x, some inv =>
    let scaled := M.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [x, inv]
    let rounded := M.op (Spec.Dequant.defaultKey "math.roundeven") .f32 [scaled]
    let hi := M.op (Spec.Dequant.defaultKey "arith.minimumf") .f32 [rounded, upper]
    let lo := M.op (Spec.Dequant.defaultKey "arith.maximumf") .f32 [hi, lower]
    match M.toInt (Spec.Dequant.defaultKey "arith.fptosi") .f32 8 lo with
    | some q => .some (.bits 8 q)
    | none => .poison
  | _, _ => .poison

private def multiplyScale (M : FloatModel) (v : Val) (s : Option (FBits .f32)) : Val :=
  match v, s with
  | .some (.flt .f32 v), some s =>
    .some (.flt .f32 (M.op (Spec.Dequant.defaultKey "arith.mulf") .f32 [v, s]))
  | _, _ => .poison

/-- Convert once, then apply scales left-to-right. Reassociation is not permitted. -/
def requantize (M : FloatModel) (acc : Option (BitVec 32))
    (scales : List (Option (FBits .f32))) : Val :=
  let converted := floatValue (acc.map fun a =>
    M.ofInt (Spec.Dequant.defaultKey "arith.sitofp") 32 .f32 a)
  scales.foldl (multiplyScale M) converted

private def plain (name : String) (id : Nat) (ty : Ty) (args : List Nat) : Op :=
  .mk name [{ id, ty }] args [] [] [] []

private def floating (name : String) (id : Nat) (args : List Nat) : Op :=
  .mk name [{ id, ty := .float .f32 }] args
    [("fastmath", .enum "arith" "fastmath" "none")] [] [] []

private def constant (id : Nat) (bits : BitVec 32) : Op :=
  .mk "arith.constant" [{ id, ty := .float .f32 }] []
    [("value", .float bits.toNat (.float .f32))] [] [] []

private def yield (id : Nat) : Op := .mk "linalg.yield" [] [id] [] [] [] []

private def compareZero (id : Nat) (args : List Nat) : Op :=
  .mk "arith.cmpf" [{ id, ty := .int 1 }] args
    [("fastmath", .enum "arith" "fastmath" "none"), ("predicate", .int 1 (.int 64))] [] [] []

def absMaxBody : List Op :=
  [floating "math.absf" 2 [0], floating "arith.maximumf" 3 [2, 1], yield 3]

def reciprocalBody : List Op :=
  [constant 2 0x3f800000#32, floating "arith.divf" 3 [2, 0], yield 3]

def scaleBody : List Op :=
  [floating "arith.divf" 2 [0, 100], constant 3 W8A8Scale.zero, constant 4 W8A8Scale.floor,
    floating "arith.maximumf" 5 [2, 4], compareZero 6 [0, 3],
    plain "arith.select" 7 (.float .f32) [6, 4, 5], yield 7]

/-- Captured upper/lower bounds use ids 100/101, distinct from all block-local ids. -/
def quantizeBody : List Op :=
  [floating "arith.mulf" 3 [0, 1], floating "math.roundeven" 4 [3],
   floating "arith.minimumf" 5 [4, 100], floating "arith.maximumf" 6 [5, 101],
   plain "arith.fptosi" 7 (.int 8) [6], yield 7]

def requantizeBody (count : Nat) : List Op :=
  [plain "arith.sitofp" 4 (.float .f32) [0]] ++
  (match count with
   | 0 => [yield 4]
   | 1 => [floating "arith.mulf" 5 [4, 1], yield 5]
   | 2 => [floating "arith.mulf" 5 [4, 1], floating "arith.mulf" 6 [5, 2], yield 6]
   | _ => [])

def floatArgs (n : Nat) : List Value :=
  (List.range n).map fun id => { id, ty := .float .f32 }

def absMaxRegion : Region := .mk [.mk (floatArgs 2) absMaxBody]
def reciprocalRegion : Region := .mk [.mk (floatArgs 2) reciprocalBody]
def scaleRegion : Region := .mk [.mk (floatArgs 2) scaleBody]
def quantizeRegion : Region :=
  .mk [.mk (floatArgs 2 ++ [{ id := 2, ty := .int 8 }]) quantizeBody]
def requantizeRegion (count : Nat) : Region :=
  .mk [.mk ([{ id := 0, ty := .int 32 }] ++
    (List.range (count + 1)).map (fun i => { id := i + 1, ty := .float .f32 }))
    (requantizeBody count)]

def quantizeCaptures (upper lower : FBits .f32) (env : Env) : Env :=
  [(100, .scalar (.some (.flt .f32 upper))), (101, .scalar (.some (.flt .f32 lower)))] ++ env

def scaleCaptures (env : Env) : Env :=
  (100, .scalar (.some (.flt .f32 W8A8Scale.denominator))) :: env

private theorem dialect_absf : Sem.Ops.dialectOf "math.absf" = "math" := by cbv
private theorem dialect_roundeven : Sem.Ops.dialectOf "math.roundeven" = "math" := by cbv
private theorem dialect_maximumf : Sem.Ops.dialectOf "arith.maximumf" = "arith" := by cbv
private theorem dialect_minimumf : Sem.Ops.dialectOf "arith.minimumf" = "arith" := by cbv
private theorem dialect_mulf : Sem.Ops.dialectOf "arith.mulf" = "arith" := by cbv
private theorem dialect_divf : Sem.Ops.dialectOf "arith.divf" = "arith" := by cbv
private theorem dialect_constant : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
private theorem dialect_fptosi : Sem.Ops.dialectOf "arith.fptosi" = "arith" := by cbv
private theorem dialect_sitofp : Sem.Ops.dialectOf "arith.sitofp" = "arith" := by cbv
private theorem dialect_cmpf : Sem.Ops.dialectOf "arith.cmpf" = "arith" := by cbv
private theorem dialect_select : Sem.Ops.dialectOf "arith.select" = "arith" := by cbv
private theorem math_absf_prefix : "math.absf".startsWith "math." = true := by cbv
private theorem math_roundeven_prefix : "math.roundeven".startsWith "math." = true := by cbv

private theorem floatKey_plain (name : String) (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty }] args [] [] [] []) =
      Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem floatKey_floating (name : String) (id : Nat) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk name [{ id, ty := .float .f32 }] args
      [("fastmath", .enum "arith" "fastmath" "none")] [] [] []) =
      Spec.Dequant.defaultKey name := by
  apply congrArg (fun modifiers => ({ op := name, modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

private theorem floatKey_compare (id : Nat) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk "arith.cmpf" [{ id, ty := .int 1 }] args
      [("fastmath", .enum "arith" "fastmath" "none"), ("predicate", .int 1 (.int 64))]
      [] [] []) = W8A8Scale.equalKey := by
  apply congrArg (fun modifiers => ({ op := "arith.cmpf", modifiers } : FloatOpKey))
  simp only [Sem.Ops.Arith.fastmathKey, Sem.Ops.Arith.attrKey,
    Op.attr?, Op.props, Op.attrs, List.lookup]
  cbv

-- Share only semantic reduction plumbing. The arithmetic contracts above remain explicit.
macro "w8_scalar_simp" : tactic => `(tactic|
  simp [*, evalRegion, evalBlocks, evalBlock, evalOps, absMaxRegion, reciprocalRegion,
    quantizeRegion, requantizeRegion, scaleRegion, floatArgs, absMaxBody, reciprocalBody,
    quantizeBody, requantizeBody, scaleBody, plain, floating, constant, yield, compareZero,
    quantizeCaptures, scaleCaptures, absMax, reciprocal, quantize, requantize, scale,
    W8A8Scale.guarded, W8A8Scale.declared, W8A8Scale.denominator, W8A8Scale.floor,
    W8A8Scale.zero, multiplyScale, floatValue, intValue,
    Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM, Outcome.ofOption,
    Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_absf, dialect_roundeven, dialect_maximumf, dialect_minimumf, dialect_mulf,
    dialect_divf, dialect_constant, dialect_fptosi, dialect_sitofp, dialect_cmpf, dialect_select,
    math_absf_prefix, math_roundeven_prefix, Sem.Ops.Arith.sem,
    Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt,
    Sem.Ops.Arith.fltN, Sem.Ops.Arith.fltBin, liftEager,
    Sem.Ops.Arith.widthOf, Sem.Ops.Arith.bv, Sem.Ops.Arith.boolVal, Ty.bitWidth?,
    floatKey_plain, floatKey_floating, floatKey_compare, Op.isTerminator, terminatorNames,
    scalarOfAttr, hasUnit, Op.attr?, FKind.width, RVal.canonize, List.range_succ, Function.comp_def,
    List.mapM_cons, List.mapM_nil])

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
theorem absMax_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (x acc : Option (FBits .f32)) :
    evalRegion Sem.Ops.opSem ctx call env absMaxRegion
      [.scalar (floatValue x), .scalar (floatValue acc)] = .ok [.scalar (absMax ctx.M x acc)] := by
  cases x <;> cases acc <;> w8_scalar_simp

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
theorem reciprocal_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (s : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call env reciprocalRegion
      [.scalar (floatValue s), .scalar init] = .ok [.scalar (reciprocal ctx.M s)] := by
  cases s <;> w8_scalar_simp

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
theorem scale_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (amax : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call (scaleCaptures env) scaleRegion
      [.scalar (floatValue amax), .scalar init] = .ok [.scalar (scale ctx.M amax)] := by
  cases amax with
  | none => w8_scalar_simp
  | some amax =>
    cases heq : ctx.M.cmp W8A8Scale.equalKey .f32 amax W8A8Scale.zero
    all_goals simp only [W8A8Scale.zero] at heq
    all_goals w8_scalar_simp

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
theorem quantize_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (x inv : Option (FBits .f32)) (init : Val) (upper lower : FBits .f32) :
    evalRegion Sem.Ops.opSem ctx call (quantizeCaptures upper lower env) quantizeRegion
      [.scalar (floatValue x), .scalar (floatValue inv), .scalar init] =
      .ok [.scalar (quantize ctx.M x inv upper lower)] := by
  cases x <;> cases inv <;> w8_scalar_simp
  all_goals split <;> simp_all

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
theorem requantize_sound (count : Nat) (hc : count ≤ 2)
    (ctx : Ctx) (call : CallSem) (env : Env) (acc : Option (BitVec 32))
    (left right : Option (FBits .f32)) (init : Val) :
    evalRegion Sem.Ops.opSem ctx call env (requantizeRegion count)
      ([.scalar (intValue acc)] ++
        ([.scalar (floatValue left), .scalar (floatValue right)].take count) ++ [.scalar init]) =
      .ok [.scalar (requantize ctx.M acc ([left, right].take count))] := by
  have hcases : count = 0 ∨ count = 1 ∨ count = 2 := by omega
  rcases hcases with rfl | rfl | rfl
  all_goals cases acc <;> cases left <;> cases right <;> w8_scalar_simp

#print axioms absMax_sound
#print axioms reciprocal_sound
#print axioms scale_sound
#print axioms quantize_sound
#print axioms requantize_sound

end MerlinVerify.Rules.W8A8Scalar
