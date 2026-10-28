/- Exact scalar semantics of the captured mask's comparisons, boolean operations
and f32 selection. Destination values may be poison; no floating-point law is used. -/
import MerlinVerify.Rules.W8A8SourceAlpha
import MerlinVerify.Rules.W8A8OrderedBody

namespace MerlinVerify.Rules.W8A8AttentionMaskBody

open MerlinVerify

inductive Comparison where
  | equal | lessEqual
  deriving DecidableEq, Repr

def predicate : Comparison → Int
  | .equal => 0
  | .lessEqual => 3

def compare (kind : Comparison) (left right : BitVec 64) : Bool :=
  match kind with
  | .equal => left == right
  | .lessEqual => left.sle right

def comparisonRegion (kind : Comparison) : Region :=
  .mk [.mk [⟨0, .int 64⟩, ⟨1, .int 64⟩, ⟨2, .int 1⟩]
    [.mk "arith.cmpi" [⟨3, .int 1⟩] [0, 1] [("predicate", .int (predicate kind) (.int 64))] [] [] [],
     .mk "linalg.yield" [] [3] [] [] [] []]]

def conjunctionRegion : Region :=
  .mk [.mk [⟨0, .int 1⟩, ⟨1, .int 1⟩, ⟨2, .int 1⟩]
    [.mk "arith.andi" [⟨3, .int 1⟩] [0, 1] [] [] [] [], .mk "linalg.yield" [] [3] [] [] [] []]]

def inversionRegion : Region :=
  .mk [.mk [⟨0, .int 1⟩, ⟨1, .int 1⟩]
    [.mk "arith.constant" [⟨2, .int 1⟩] [] [("value", .int 1 (.int 1))] [] [] [],
     .mk "arith.xori" [⟨3, .int 1⟩] [0, 2] [] [] [] [], .mk "linalg.yield" [] [3] [] [] [] []]]

def copyRegion : Region :=
  .mk [.mk [⟨0, .int 1⟩, ⟨1, .int 1⟩] [.mk "linalg.yield" [] [0] [] [] [] []]]

def selectRegion : Region :=
  .mk [.mk [⟨0, .int 1⟩, ⟨1, .float .f32⟩, ⟨2, .float .f32⟩, ⟨3, .float .f32⟩]
    [.mk "arith.select" [⟨4, .float .f32⟩] [0, 1, 2] [] [] [] [], .mk "linalg.yield" [] [4] [] [] [] []]]

def integer (bits : BitVec 64) : Val := .some (.bits 64 bits)
def boolean (value : Bool) : Val := Sem.Ops.Arith.boolVal value

private theorem dialect_compare : Sem.Ops.dialectOf "arith.cmpi" = "arith" := by cbv
private theorem dialect_and : Sem.Ops.dialectOf "arith.andi" = "arith" := by cbv
private theorem dialect_xor : Sem.Ops.dialectOf "arith.xori" = "arith" := by cbv
private theorem dialect_constant : Sem.Ops.dialectOf "arith.constant" = "arith" := by cbv
private theorem dialect_select : Sem.Ops.dialectOf "arith.select" = "arith" := by cbv

theorem comparison_sound (kind : Comparison) (ctx : Ctx) (call : CallSem) (env : Env)
    (left right : BitVec 64) (destination : Val) :
    evalRegion W8A8SourceSemantics.opSem ctx call env (comparisonRegion kind)
      [.scalar (integer left), .scalar (integer right), .scalar destination] =
      .ok [.scalar (boolean (compare kind left right))] := by
  cases kind <;>
    simp [evalRegion, evalBlocks, evalBlock, evalOps, comparisonRegion, predicate, compare,
      integer, boolean, Env.bindAll, Env.getAll, Env.get?, List.lookup, Outcome.mapM, Outcome.ofOption,
      W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
      Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
      Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors, Op.attr?,
      dialect_compare, Sem.Ops.Arith.sem, attrInt?, Sem.Ops.Arith.cmpiPred, Sem.Ops.Arith.binOp,
      Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt, liftEager, Sem.Ops.Arith.intBin, Op.isTerminator, terminatorNames]

theorem conjunction_sound (ctx : Ctx) (call : CallSem) (env : Env)
    (left right : Bool) (destination : Val) :
    evalRegion W8A8SourceSemantics.opSem ctx call env conjunctionRegion
      [.scalar (boolean left), .scalar (boolean right), .scalar destination] =
      .ok [.scalar (boolean (left && right))] := by
  cases left <;> cases right <;>
    simp [evalRegion, evalBlocks, evalBlock, evalOps, conjunctionRegion, boolean, Sem.Ops.Arith.boolVal,
      Env.bindAll, Env.getAll, Env.get?, List.lookup, Outcome.mapM, Outcome.ofOption,
      W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
      Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
      Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      dialect_and, Sem.Ops.Arith.sem, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt, liftEager,
      Sem.Ops.Arith.intBin, Sem.Ops.Arith.bv, Op.isTerminator, terminatorNames]

theorem inversion_sound (ctx : Ctx) (call : CallSem) (env : Env) (input : Bool) (destination : Val) :
    evalRegion W8A8SourceSemantics.opSem ctx call env inversionRegion
      [.scalar (boolean input), .scalar destination] = .ok [.scalar (boolean (!input))] := by
  cases input <;>
    simp [evalRegion, evalBlocks, evalBlock, evalOps, inversionRegion, boolean, Sem.Ops.Arith.boolVal,
      Env.bindAll, Env.getAll, Env.get?, List.lookup, Outcome.mapM, Outcome.ofOption,
      W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
      Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
      Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors, Op.attr?,
      dialect_xor, dialect_constant, Sem.Ops.Arith.sem, Sem.Ops.Arith.binOp, Sem.Ops.Arith.pureN, Sem.Ops.Arith.resultElt, liftEager,
      Sem.Ops.Arith.intBin, Sem.Ops.Arith.bv, scalarOfAttr, Op.isTerminator, terminatorNames]

theorem copy_sound (ctx : Ctx) (call : CallSem) (env : Env) (input destination : Val) :
    evalRegion W8A8SourceSemantics.opSem ctx call env copyRegion [.scalar input, .scalar destination] =
      .ok [.scalar input] := by
  simp [evalRegion, evalBlocks, evalBlock, evalOps, copyRegion, Env.bindAll,
    Env.getAll, Env.get?, List.lookup, Outcome.mapM, Outcome.ofOption,
    Op.name, Op.isTerminator, terminatorNames]

theorem select_sound (ctx : Ctx) (call : CallSem) (env : Env) (condition : Bool)
    (left right : FBits .f32) (destination : Val) :
    evalRegion W8A8SourceSemantics.opSem ctx call env selectRegion
      [.scalar (boolean condition), .scalar (W8A8OrderedBody.value left),
        .scalar (W8A8OrderedBody.value right), .scalar destination] =
      .ok [.scalar (W8A8OrderedBody.value (if condition then left else right))] := by
  cases condition <;>
    simp [evalRegion, evalBlocks, evalBlock, evalOps, selectRegion, boolean, Sem.Ops.Arith.boolVal,
      W8A8OrderedBody.value, Env.bindAll, Env.getAll, Env.get?, List.lookup, Outcome.mapM, Outcome.ofOption,
      W8A8SourceSemantics.opSem, W8A8SourceSemantics.isSource, W8A8SourceContract.name,
      Sem.Ops.opSem, Sem.Ops.unmodelledAttr?, Sem.Ops.interpretedAttrs, Sem.Ops.dispatch,
      Op.blind, Op.name, Op.results, Op.attrs, Op.props, Op.successors,
      dialect_select, Sem.Ops.Arith.sem, RVal.canonize, Op.isTerminator, terminatorNames]

end MerlinVerify.Rules.W8A8AttentionMaskBody
