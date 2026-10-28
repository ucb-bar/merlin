/- Pointwise refinement for eliminating an identity tensor loop only at function return.
The canonical loop result and the raw input need not be equal outside their index space. -/
import MerlinVerify.Rules.ParallelFusion
import MerlinVerify.Sem.Ops

namespace MerlinVerify.Rules.IdentityReturn

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

/-- The actual single-block region returns its input argument, independently of its closure
and old output. No operation semantics or floating-point law is used by this body. -/
theorem region_copy (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env)
    (input old : Value) (term : Op) (hn : term.name = "linalg.yield")
    (ho : term.operands = [input.id]) (value previous : Val) :
    evalRegion sem ctx call env (.mk [.mk [input, old] [term]])
      [.scalar value, .scalar previous] = .ok [.scalar value] := by
  cases term with
  | mk name results operands attrs props regions successors =>
    simp only [Op.name, Op.operands] at hn ho
    subst name operands
    simp [evalRegion, evalBlocks, evalBlock, Env.bindAll, evalOps, Op.isTerminator,
      terminatorNames, Op.name, Env.getAll, Outcome.mapM, Outcome.ofOption, Env.get?, List.lookup]

/-- Identity geometry and actual input/output shapes make the loop total, even when the
input contains poison or the iteration space is empty. Its result remains canonical. -/
theorem copy_loop (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat)
    (input init : Tensor) (arg old : Value) (term : Op)
    (hi : input.shape = shape) (ho : init.shape = shape)
    (hn : term.name = "linalg.yield") (ht : term.operands = [arg.id]) :
    genericSem ctx [identityMap shape.length, identityMap shape.length]
      (List.replicate shape.length .parallel) [.tensor input] [.tensor init]
      (fun nested => evalRegion Sem.Ops.opSem nested call env (.mk [.mk [arg, old] [term]])) =
      .ok [.tensor (Tensor.canon shape init.elt input.get)] := by
  apply ParallelFusion.identity_generic ctx shape [input] init
  · intro t h
    simp only [List.singleton_append, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with h | h <;> rw [h] <;> assumption
  · intro pt hp
    have hri : readAt (.tensor input) (identityMap shape.length) pt = .ok (input.get pt) := by
      simpa only [hi] using readAt_identity (t := input) (hi.symm ▸ hp)
    have hro : readAt (.tensor init) (identityMap shape.length) pt = .ok (init.get pt) := by
      simpa only [ho] using readAt_identity (t := init) (ho.symm ▸ hp)
    simp [bodyAt, hri, hro, region_copy Sem.Ops.opSem _ call env arg old term hn ht]

/-- Returning the raw input refines returning its canonical copy. This theorem deliberately
does not identify the two tensors as functional values outside their bounds. -/
theorem input_refines_copy (input : Tensor) (shape : List Nat) (elt : Ty)
    (hs : input.shape = shape) (he : input.elt = elt) :
    TensorRefines input (Tensor.canon shape elt input.get) := by
  refine ⟨hs, he, ?_⟩
  intro pt hp
  change Idx.InBounds shape pt at hp
  exact Or.inr (by simp [Tensor.canon, hp])

end MerlinVerify.Rules.IdentityReturn
