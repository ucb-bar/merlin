/-
Scalar regions admitted by the f32-accumulation contract, with their exact execution under the
frozen semantics. Actual regions are related to these templates only by `alphaRegion`.
-/
import MerlinVerify.Rules.Bf16F32AccContract

namespace MerlinVerify.Rules.Bf16F32AccBody

open MerlinVerify Bf16F32AccContract

private def scalar (name : String) (id : Nat) (args : List Nat) (attrs : List (String × Attr)) : Op :=
  .mk name [{ id, ty := .float .f32 }] args attrs [] [] []

/-- `^bb0(%a: ka, %b: kb, %acc: f32)`: extend both operands, multiply, add to the accumulator. -/
def accBody (license : License) : List Op :=
  [scalar "arith.extf" 3 [0] [], scalar "arith.extf" 4 [1] [],
   scalar "arith.mulf" 5 [3, 4] (fastmath license.mul),
   scalar "arith.addf" 6 [2, 5] (fastmath license.add),
   .mk "linalg.yield" [] [6] [] [] [] []]

def accRegion (ka kb : FKind) (license : License) : Region :=
  .mk [.mk [{ id := 0, ty := .float ka }, { id := 1, ty := .float kb }, { id := 2, ty := .float .f32 }]
    (accBody license)]

/-- `^bb0(%acc: f32, %old: kc)`: one `arith.truncf`; the old destination element is unused. -/
def truncBody (kc : FKind) : List Op :=
  [.mk "arith.truncf" [{ id := 2, ty := .float kc }] [0] [] [] [] [],
   .mk "linalg.yield" [] [2] [] [] [] []]

def truncRegion (kc : FKind) : Region :=
  .mk [.mk [{ id := 0, ty := .float .f32 }, { id := 1, ty := .float kc }] (truncBody kc)]

private theorem dialect_extf : Sem.Ops.dialectOf "arith.extf" = "arith" := by cbv
private theorem dialect_truncf : Sem.Ops.dialectOf "arith.truncf" = "arith" := by cbv
private theorem dialect_mulf : Sem.Ops.dialectOf "arith.mulf" = "arith" := by cbv
private theorem dialect_addf : Sem.Ops.dialectOf "arith.addf" = "arith" := by cbv
private theorem extf_prefix : "arith.extf".startsWith "math." = false := by cbv

private theorem key_ext (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk "arith.extf" [{ id, ty }] args [] [] [] []) = extKey := rfl
private theorem key_trunc (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk "arith.truncf" [{ id, ty }] args [] [] [] []) = truncKey := rfl
private theorem key_mul (f : Flag) (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk "arith.mulf" [{ id, ty }] args
      [("fastmath", .enum "arith" "fastmath" f.text)] [] [] []) = mulKey f := rfl
private theorem key_add (f : Flag) (id : Nat) (ty : Ty) (args : List Nat) :
    Sem.Ops.Arith.floatKey (.mk "arith.addf" [{ id, ty }] args
      [("fastmath", .enum "arith" "fastmath" f.text)] [] [] []) = addKey f := rfl

macro "bf16_body_simp" : tactic => `(tactic|
  simp [evalRegion, evalBlocks, evalBlock, evalOps, accRegion, accBody, truncRegion, truncBody,
    scalar, value, stepOpt, step, ext, trunc, fastmath,
    Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM, Outcome.ofOption,
    Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_extf, dialect_truncf, dialect_mulf, dialect_addf, extf_prefix,
    Sem.Ops.Arith.sem, Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp, Sem.Ops.Arith.resultElt,
    Sem.Ops.Arith.fltBin, Sem.Ops.Arith.pureN, liftEager, key_ext, key_trunc, key_mul, key_add,
    Op.isTerminator, terminatorNames, Op.attr?, Spec.Annotations.semanticsFree])

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
/-- The accumulation region computes the contract's step, and poison in any operand is poison. -/
theorem accRegion_sound (ctx : Ctx) (call : CallSem) (env : Env) (ka kb : FKind) (license : License)
    (a : Option (FBits ka)) (b : Option (FBits kb)) (acc : Option (FBits .f32)) :
    evalRegion Sem.Ops.opSem ctx call env (accRegion ka kb license)
      [.scalar (value ka a), .scalar (value kb b), .scalar (value .f32 acc)] =
      .ok [.scalar (value .f32 (stepOpt ctx.M license a b acc))] := by
  cases a <;> cases b <;> cases acc <;> bf16_body_simp

set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
/-- The final-conversion region truncates its accumulator and ignores the old element. -/
theorem truncRegion_sound (ctx : Ctx) (call : CallSem) (env : Env) (kc : FKind)
    (x : Option (FBits .f32)) (old : Val) :
    evalRegion Sem.Ops.opSem ctx call env (truncRegion kc) [.scalar (value .f32 x), .scalar old] =
      .ok [.scalar (value kc (x.map (trunc ctx.M kc)))] := by
  cases x <;> bf16_body_simp

#print axioms accRegion_sound
#print axioms truncRegion_sound

end MerlinVerify.Rules.Bf16F32AccBody
