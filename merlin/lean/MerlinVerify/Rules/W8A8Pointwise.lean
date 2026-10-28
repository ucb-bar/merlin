/- Tensor lifting for the pointwise W8A8 stages. The checked scalar contract ignores the old
destination element; complete overwrites work even when tensor.empty supplied poison. -/
import MerlinVerify.Rules.W8A8TensorOperands

namespace MerlinVerify.Rules.W8A8Pointwise

open MerlinVerify MerlinVerify.Sem.Ops.Linalg W8A8TensorOperands

/-- A checked pointwise stage computes its scalar contract at every actual projected input
coordinate. All bounds are derived from the checked geometry, including rank-zero tensors. -/
theorem generic_sound (ctx : Ctx) (call : CallSem) (env : Env) (inputs : List Input)
    (output : Tensor) (maps : List AffineMap) (iters : List IterKind) (actual : Region)
    (geometry : Check.StructuredOk maps iters.length
      (inputs.map Input.type ++ [.tensor output.shape output.elt]) output.shape)
    (outMap : maps.drop inputs.length = [identityMap output.shape.length])
    (result : Idx → Val)
    (scalar : ∀ point, Idx.InBounds output.shape point → ∀ old : Val,
      evalRegion Sem.Ops.opSem { ctx with loopIndex := point } call env actual
        ((values inputs (maps.take inputs.length) point).map RVal.scalar ++ [.scalar old]) =
          .ok [.scalar (result point)]) :
    genericSem ctx maps iters (inputs.map Input.value) [.tensor output]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon output.shape output.elt result)] := by
  obtain ⟨hlen, hdims, hrank, hranges⟩ := runtime inputs output geometry
  have inMaps : Check.MapsInBounds output.shape (maps.take inputs.length)
      (inputs.map Input.type) := by
    have ht := List.forall₂_take inputs.length geometry.inBounds
    simpa using ht
  apply genericSem_parallel1 ctx maps iters _ output _ output.shape result
    (by simpa using hlen) hdims hrank hranges rfl (by simpa using outMap)
  intro point hp
  have reads := reads_sound point hp inputs _ inMaps
  have old := readAt_identity (t := output) hp
  have body := scalar point hp (output.get point)
  simp only [bodyAt, List.length_map, reads, List.zip_cons_cons, List.zip_nil_right,
    List.mapM_cons, List.mapM_nil, old, Outcome.bindOp_eq, Outcome.bind_ok, Outcome.pure_eq,
    List.map_append, List.map_cons, List.map_nil, body, List.length_singleton, ne_eq,
    not_true_eq_false, ↓reduceIte]

#print axioms generic_sound

end MerlinVerify.Rules.W8A8Pointwise
