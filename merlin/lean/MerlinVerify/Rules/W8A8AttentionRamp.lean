/- Exact initialized arange body for the captured i64 row/column ramps. Index
width is explicit; no numerical input or mask property is assumed. -/
import MerlinVerify.Rules.W8A8AttentionExecution
import MerlinVerify.Rules.W8A8AttentionMaskBody

namespace MerlinVerify.Rules.W8A8AttentionRamp

open MerlinVerify

def arithmetic (name : String) (id : Nat) (inputs : List Nat) : Op :=
  .mk name [⟨id, .int 64⟩] inputs [("overflowFlags", .enum "arith" "overflow" "none")] [] [] []

def region : Region :=
  .mk [.mk [⟨0, .int 64⟩]
    [.mk "linalg.index" [⟨1, .index⟩] [] [("dim", .int 0 (.int 64))] [] [] [],
     .mk "arith.index_cast" [⟨2, .int 64⟩] [1] [] [] [] [],
     .mk "arith.constant" [⟨3, .int 64⟩] [] [("value", .int 1 (.int 64))] [] [] [],
     arithmetic "arith.muli" 4 [2, 3],
     .mk "arith.constant" [⟨5, .int 64⟩] [] [("value", .int 0 (.int 64))] [] [] [],
     arithmetic "arith.addi" 6 [5, 4], .mk "linalg.yield" [] [6] [] [] [] []]]

private theorem dialect_index : Sem.Ops.dialectOf "linalg.index" = "linalg" := by cbv
private theorem dialect_cast : Sem.Ops.dialectOf "arith.index_cast" = "arith" := by cbv
private theorem dialect_constant : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
private theorem dialect_multiply : Sem.Ops.dialectOf "arith.muli" = "arith" := by cbv
private theorem dialect_add : Sem.Ops.dialectOf "arith.addi" = "arith" := by cbv
private theorem flagList_none : flagList "none" = ["none"] := by cbv

theorem region_sound (ctx : Ctx) (call : CallSem) (env : Env) (column : Nat) (destination : Val)
    (width : ctx.indexBits = 64) (position : ctx.loopIndex[0]? = some column) :
    evalRegion W8A8SourceSemantics.opSem ctx call env region [.scalar destination] =
      .ok [.scalar (W8A8AttentionMaskBody.integer (BitVec.ofNat 64 column))] := by
  simp [evalRegion, evalBlocks, evalBlock, evalOps, region, arithmetic, W8A8AttentionMaskBody.integer,
    Env.bindAll, Env.getAll, Env.get?, List.lookup, Outcome.mapM, Outcome.ofOption,
    W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
    Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
    Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
    dialect_index, dialect_cast, dialect_constant, dialect_multiply, dialect_add,
    Sem.Ops.Linalg.sem, Sem.Ops.Arith.sem, Sem.Ops.Arith.unOp, Sem.Ops.Arith.binOp,
    Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt, Sem.Ops.Arith.intUn, Sem.Ops.Arith.intBin,
    Sem.Ops.Arith.widthOf, Sem.Ops.Arith.arithWithFlags, Sem.Ops.Arith.bv, liftEager,
    scalarOfAttr, attrInt?, Op.attr?, Ty.bitWidth?, overflowFlags, attrEnum?, flagList_none,
    Op.isTerminator, terminatorNames, width, position]

end MerlinVerify.Rules.W8A8AttentionRamp
