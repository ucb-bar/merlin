/- Actual scalar body, for every float model and surrounding SSA environment. -/
import MerlinVerify.Rules.OverrankIntent
import MerlinVerify.Check.Alpha
import MerlinVerify.Check.AlphaFast
namespace MerlinVerify.Rules.OverrankBody
open MerlinVerify OverrankIntent
def flags : List (String × Attr) := [("fastmath", .enum "arith" "fastmath" "none")]
def primitive (name : String) (kind : FKind) (result : Nat) (args : List Nat) : Op :=
  .mk name [⟨result, .float kind⟩] args flags [] [] []
def region (kind : FKind) : Region := .mk [.mk
  [⟨0, .float kind⟩, ⟨1, .float kind⟩, ⟨2, .float kind⟩]
  [primitive "arith.mulf" kind 3 [0,1], primitive "arith.addf" kind 4 [2,3],
   .mk "linalg.yield" [] [4] [] [] [] []]]
/-- Default named matmul body. Its accumulator operand order differs from the repair intent;
no equality between these two floating expressions is asserted. -/
def namedRegion (kind : FKind) : Region := .mk [.mk
  [⟨0, .float kind⟩, ⟨1, .float kind⟩, ⟨2, .float kind⟩]
  [primitive "arith.mulf" kind 3 [0,1], primitive "arith.addf" kind 4 [3,2],
   .mk "linalg.yield" [] [4] [] [] [] []]]

private theorem dialect_mul : Sem.Ops.dialectOf "arith.mulf" = "arith" := by cbv
private theorem dialect_add : Sem.Ops.dialectOf "arith.addf" = "arith" := by cbv
set_option maxHeartbeats 2000000 in
set_option maxRecDepth 4096 in
/-- Actual successful evaluation, including poison propagation. -/
theorem region_sound (ctx : Ctx) (call : CallSem) (env : Env) (kind : FKind)
    (a b acc : Option (FBits kind)) :
    evalRegion Sem.Ops.opSem ctx call env (region kind)
      [.scalar (value kind a), .scalar (value kind b), .scalar (value kind acc)] =
      .ok [.scalar (value kind (stepOpt ctx.M kind a b acc))] := by
  cases a <;> cases b <;> cases acc <;>
    simp [evalRegion, evalBlocks, evalBlock, evalOps, region, primitive, flags,
      value, stepOpt, step, mulKey, addKey,
      Env.getAll, Env.get?, Env.bindAll, List.lookup, Outcome.mapM, Outcome.ofOption,
      Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
      Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      dialect_mul, dialect_add, Sem.Ops.Arith.sem, Sem.Ops.Arith.binOp,
      Sem.Ops.Arith.resultElt, Sem.Ops.Arith.fltBin, Sem.Ops.Arith.pureN, liftEager,
      Op.isTerminator, terminatorNames, Op.attr?, Spec.Annotations.semanticsFree]
  all_goals rfl
theorem region_transfer (call : CallSem) (env : Env) {kind : FKind} {actual : Region}
    (h : Check.alphaRegion [] (region kind) actual = true) (ctx : Ctx) :
    evalRegion Sem.Ops.opSem ctx call env actual = evalRegion Sem.Ops.opSem ctx call env (region kind) :=
  (Check.alphaRegion_sound call call [] _ _ ctx env env h
    (by intro x y hxy; simp [Check.Rho.find, List.lookup] at hxy) (fun _ _ => rfl)).symm
end MerlinVerify.Rules.OverrankBody
