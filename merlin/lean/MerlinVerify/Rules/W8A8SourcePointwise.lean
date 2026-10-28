/- Reuse the frozen parallel-loop execution theorem with a caller-selected source
semantics. Initialization is checked only at actual mapped input coordinates. -/
import MerlinVerify.Rules.W8A8TensorOperands

namespace MerlinVerify.Rules.W8A8SourcePointwise

open MerlinVerify Sem.Ops.Linalg W8A8TensorOperands

theorem coordinate_inBounds {shape ranges : List Nat} {map : AffineMap} {point : Idx}
    (mapping : Check.mapInBounds ranges map shape = true) (bounds : Idx.InBounds ranges point) :
    Idx.InBounds shape (coordinate shape map point) := by
  obtain ⟨cell, mapped, valid⟩ := Check.mapInBounds_sound mapping bounds
  simpa only [coordinate, mapped, Option.getD_some] using valid

theorem generic_sound (sem : OpSem) (ctx : Ctx) (call : CallSem) (env : Env) (inputs : List Input)
    (output : Tensor) (maps : List AffineMap) (iters : List IterKind) (actual : Region)
    (geometry : Check.StructuredOk maps iters.length
      (inputs.map Input.type ++ [.tensor output.shape output.elt]) output.shape)
    (outMap : maps.drop inputs.length = [identityMap output.shape.length])
    (result : Idx → Val)
    (scalar : ∀ point, Idx.InBounds output.shape point → ∀ old : Val,
      evalRegion sem { ctx with loopIndex := point } call env actual
        ((values inputs (maps.take inputs.length) point).map RVal.scalar ++ [.scalar old]) =
          .ok [.scalar (result point)]) :
    genericSem ctx maps iters (inputs.map Input.value) [.tensor output]
      (fun c => evalRegion sem c call env actual) =
      .ok [.tensor (Tensor.canon output.shape output.elt result)] := by
  obtain ⟨length, dimensions, ranks, ranges⟩ := runtime inputs output geometry
  have inMaps : Check.MapsInBounds output.shape (maps.take inputs.length) (inputs.map Input.type) := by
    have bounded := List.forall₂_take inputs.length geometry.inBounds
    simpa using bounded
  apply genericSem_parallel1 ctx maps iters _ output _ output.shape result
    (by simpa using length) dimensions ranks ranges rfl (by simpa using outMap)
  intro point bounds
  have reads := reads_sound point bounds inputs _ inMaps
  have old := readAt_identity (t := output) bounds
  have executed := scalar point bounds (output.get point)
  simp only [bodyAt, List.length_map, reads, List.zip_cons_cons, List.zip_nil_right,
    List.mapM_cons, List.mapM_nil, old, Outcome.bindOp_eq, Outcome.bind_ok, Outcome.pure_eq,
    List.map_append, List.map_cons, List.map_nil, executed, List.length_singleton, ne_eq,
    not_true_eq_false, ↓reduceIte]

end MerlinVerify.Rules.W8A8SourcePointwise
