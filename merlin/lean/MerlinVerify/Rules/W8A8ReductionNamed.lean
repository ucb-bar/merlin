/- The actual named reduction dispatch uses the same checked scalar region and
ordered generic loop. Operand splitting, dimensions and result arity are explicit. -/
import MerlinVerify.Rules.W8A8ReductionTensor

namespace MerlinVerify.Rules.W8A8ReductionNamed

open MerlinVerify Check Sem.Ops Sem.Ops.Linalg

def outputMap (rank : Nat) (dimensions : List Nat) : AffineMap :=
  LinalgNamed.mapOf rank ((List.range rank).filter (fun d => !(dimensions.contains d)))

def iterators (rank : Nat) (dimensions : List Nat) : List IterKind :=
  (List.range rank).map fun d => if dimensions.contains d then .reduction else .parallel

theorem execute {ib : Nat} {ctx : Ctx} {op : Op} {args : List RVal}
    {input : RVal} {initial result : Tensor} {inputShape dimensions : List Nat} {body : Region}
    (modelled : unmodelledAttr? op.blind = none) (dialect : dialectOf op.name = "linalg")
    (named : op.name = "linalg.reduce") (regions : op.regions = [body])
    (dims : LinalgNamed.natArray? op "dimensions" = some dimensions)
    (split : reductionSplit op args = ([input], [.tensor initial]))
    (arity : op.results.length = 1)
    (typedInput : RVal.HasTy ib input (.tensor inputShape (.float .f32)))
    (call : CallSem) (env : Env)
    (executed : genericSem ctx [identityMap inputShape.length, outputMap inputShape.length dimensions]
      (iterators inputShape.length dimensions) [input] [.tensor initial]
      (fun c => evalRegion W8A8SourceSemantics.opSem c call env body) = .ok [.tensor result]) :
    W8A8SourceSemantics.opSem ctx call op args
      (evalRegions W8A8SourceSemantics.opSem call env op.regions) = .ok [.tensor result] := by
  have ordinary : W8A8SourceSemantics.isSource op.name = false := by
    simp [W8A8SourceSemantics.isSource, named, W8A8SourceContract.name]
  rw [W8A8SourceSemantics.unchanged ordinary, named_sem_of modelled dialect (by simp [named])]
  dsimp only [reductionSplit] at split
  simp only [regions, evalRegions, LinalgNamed.sem, Op.blind_name, named, natArray?_blind,
    dims, splitInsOuts_blind, split, shapeOf_typed typedInput, List.map_cons, List.map_nil,
    List.nil_append, List.cons_append, VTy.shape, LinalgNamed.run, outputMap, iterators] at executed ⊢
  rw [executed]
  simp only [Outcome.bind_ok, Op.blind_results, List.length_map, arity, List.length_cons, List.length_nil,
    Nat.reduceAdd, ↓reduceIte]

end MerlinVerify.Rules.W8A8ReductionNamed
