/-
Execution of Merlin's signed i8 × i8 → i32 multiply-and-accumulate region.
This is a component of the W8A8 checker, not a specification of the quantization policy.
-/
import MerlinVerify.Sem.Ops
import MerlinVerify.Rules.W8A8Accumulation

namespace MerlinVerify.Rules.W8A8IntegerBody

open MerlinVerify

private def extend (id arg : Nat) : Op :=
  .mk "arith.extsi" [{ id, ty := .int 32 }] [arg] [] [] [] []

private def binary (name : String) (id : Nat) (args : List Nat) : Op :=
  .mk name [{ id, ty := .int 32 }] args
    [("overflowFlags", .enum "arith" "overflow" "none")] [] [] []

def body : List Op :=
  [extend 3 0, extend 4 1, binary "arith.muli" 5 [3, 4],
   binary "arith.addi" 6 [5, 2], .mk "linalg.yield" [] [6] [] [] [] []]

def region : Region :=
  .mk [.mk [{ id := 0, ty := .int 8 }, { id := 1, ty := .int 8 },
    { id := 2, ty := .int 32 }] body]

/-- Optional bit patterns represent initialized values or poison, with no floating-point laws. -/
def value (w : Nat) : Option (BitVec w) → Val
  | none => .poison
  | some x => .some (.bits w x)

def result (a b : Option (BitVec 8)) (acc : Option (BitVec 32)) : Val :=
  match a, b, acc with
  | some a, some b, some acc => .some (.bits 32 (W8A8Accumulation.step acc (a, b)))
  | _, _, _ => .poison

private theorem dialect_extsi : Sem.Ops.dialectOf "arith.extsi" = "arith" := by cbv
private theorem dialect_muli : Sem.Ops.dialectOf "arith.muli" = "arith" := by cbv
private theorem dialect_addi : Sem.Ops.dialectOf "arith.addi" = "arith" := by cbv

private theorem flagList_none : flagList "none" = ["none"] := by cbv

set_option maxRecDepth 4096 in
/-- The canonical MAC body executes the actual modular operation, including poison propagation. -/
theorem body_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (a b : Option (BitVec 8)) (acc : Option (BitVec 32)) :
    evalOps Sem.Ops.opSem ctx call
      ([(0, .scalar (value 8 a)), (1, .scalar (value 8 b)),
        (2, .scalar (value 32 acc))] ++ env) body =
      .ok [.scalar (result a b acc)] := by
  cases a <;> cases b <;> cases acc
  all_goals simp [evalOps, body, extend, binary, value, result, W8A8Accumulation.step,
    Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM, Outcome.ofOption,
    Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_extsi, dialect_muli, dialect_addi, Sem.Ops.Arith.sem,
    Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt,
    liftEager, Sem.Ops.Arith.intUn, Sem.Ops.Arith.intBin, Sem.Ops.Arith.arithWithFlags,
    Sem.Ops.Arith.widthOf, Ty.bitWidth?, Sem.Ops.Arith.bv,
    overflowFlags, flagList_none, attrEnum?, Op.attr?, Op.isTerminator, terminatorNames]

/-- Region binding shadows all external SSA ids; no facts about the enclosing environment are assumed. -/
theorem region_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (a b : Option (BitVec 8)) (acc : Option (BitVec 32)) :
    evalRegion Sem.Ops.opSem ctx call env region
      [.scalar (value 8 a), .scalar (value 8 b), .scalar (value 32 acc)] =
      .ok [.scalar (result a b acc)] := by
  simpa [evalRegion, evalBlocks, evalBlock, region, Env.bindAll] using
    body_sound ctx call env a b acc

#print axioms body_sound
#print axioms region_sound

end MerlinVerify.Rules.W8A8IntegerBody
