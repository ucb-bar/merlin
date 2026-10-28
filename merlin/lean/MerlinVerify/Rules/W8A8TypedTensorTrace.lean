/- Ordered tensor execution with a state invariant. Unlike an unrestricted state encoding,
this admits arbitrary out-of-bounds runtime contents while constraining actual accessed cells. -/
import MerlinVerify.Rules.W8A8TensorTrace

namespace MerlinVerify.Rules.W8A8TypedTensorTrace

open MerlinVerify Sem.Ops.Linalg W8A8TensorTrace

variable {α : Type}

private theorem pack_update (shape : List Nat) (elt : Ty) (encode : α → Val)
    (coordinate : Idx → Idx) (step : Idx → α → α) (state : Idx → α) (point : Idx) :
    (pack shape elt encode state).set (coordinate point) (encode (step point (state (coordinate point)))) =
      pack shape elt encode (update coordinate step state point) := by
  unfold pack Tensor.set update
  congr 1
  funext cell
  split
  · next equal => simp [equal]
  · rfl

theorem fold_sound (ctx : Ctx) (ins : List RVal) (inMaps : List AffineMap)
    (outMap : AffineMap) (body : Ctx → RegionSem) (shape : List Nat) (elt : Ty)
    (encode : α → Val) (coordinate : Idx → Idx) (step : Idx → α → α)
    (valid : (Idx → α) → Prop) :
    ∀ (points : List Idx),
      (∀ point ∈ points, (evalMap outMap point >>= toIdx shape) = some (coordinate point)) →
      (∀ point ∈ points, ∀ state, valid state → valid (update coordinate step state point)) →
      (∀ point ∈ points, ∀ state, valid state →
        bodyAt ctx ins inMaps [outMap] [pack shape elt encode state] body point =
          .ok [encode (step point (state (coordinate point)))]) →
      ∀ initial, valid initial →
        points.foldlM (loopStep ctx ins inMaps [outMap] body) [pack shape elt encode initial] =
          .ok [pack shape elt encode (accumulate coordinate step initial points)] := by
  intro points
  induction points with
  | nil => intro _ _ _ initial _; rfl
  | cons point rest ih =>
    intro indices preserves scalar initial permitted
    have member : point ∈ point :: rest := List.mem_cons_self ..
    have index := indices point member
    have executed := scalar point member initial permitted
    have written : writeAt (pack shape elt encode initial) outMap point
        (encode (step point (initial (coordinate point)))) =
        .ok (pack shape elt encode (update coordinate step initial point)) := by
      simp only [writeAt, pack, index]
      congr 1
      exact pack_update shape elt encode coordinate step initial point
    have first : loopStep ctx ins inMaps [outMap] body [pack shape elt encode initial] point =
        .ok [pack shape elt encode (update coordinate step initial point)] := by
      unfold loopStep
      rw [executed]
      simp only [Outcome.bindOp_eq, Outcome.bind_ok, List.zip_cons_cons, List.zip_nil_right,
        List.mapM_cons, List.mapM_nil, written, Outcome.pure_eq]
    rw [List.foldlM_cons, first]
    simp only [Outcome.bindOp_eq, Outcome.bind_ok]
    exact ih (fun p hp => indices p (List.mem_cons_of_mem _ hp))
      (fun p hp => preserves p (List.mem_cons_of_mem _ hp))
      (fun p hp => scalar p (List.mem_cons_of_mem _ hp)) _ (preserves point member initial permitted)

theorem generic_sound {ib : Nat} {ctx : Ctx} {maps : List AffineMap}
    {iters : List IterKind} {inTys : List Check.VTy} {ranges shape : List Nat}
    {ins : List RVal} {elt : Ty} {outMap : AffineMap}
    (geometry : Check.StructuredOk maps iters.length (inTys ++ [.tensor shape elt]) ranges)
    (typedIns : Check.HasTys ib ins inTys) (encode : α → Val) (initial : Idx → α)
    (typedOut : Check.RVal.HasTy ib (.tensor (pack shape elt encode initial)) (.tensor shape elt))
    (output : maps.drop ins.length = [outMap]) (body : Ctx → RegionSem)
    (coordinate : Idx → Idx) (step : Idx → α → α) (valid : (Idx → α) → Prop)
    (initialValid : valid initial)
    (indices : ∀ point, Idx.InBounds ranges point →
      (evalMap outMap point >>= toIdx shape) = some (coordinate point))
    (preserves : ∀ point, Idx.InBounds ranges point → ∀ state,
      valid state → valid (update coordinate step state point))
    (scalar : ∀ point, Idx.InBounds ranges point → ∀ state, valid state →
      bodyAt ctx ins (maps.take ins.length) [outMap] [pack shape elt encode state] body point =
        .ok [encode (step point (state (coordinate point)))]) :
    genericSem ctx maps iters ins [.tensor (pack shape elt encode initial)] body =
      .ok [.tensor (Tensor.canon shape elt
        (fun cell => encode (accumulate coordinate step initial (allPoints ranges) cell)))] := by
  have outs : Check.HasTys ib ([pack shape elt encode initial].map RVal.tensor) [.tensor shape elt] :=
    .cons typedOut .nil
  rw [show [.tensor (pack shape elt encode initial)] = [pack shape elt encode initial].map RVal.tensor from rfl]
  rw [Check.genericSem_execute geometry typedIns outs, output]
  rw [fold_sound ctx ins (maps.take ins.length) outMap body shape elt encode coordinate step valid
    (allPoints ranges) (fun p hp => indices p ((mem_allPoints _ _).1 hp))
    (fun p hp => preserves p ((mem_allPoints _ _).1 hp))
    (fun p hp => scalar p ((mem_allPoints _ _).1 hp)) initial initialValid]
  rfl

#print axioms fold_sound
#print axioms generic_sound

end MerlinVerify.Rules.W8A8TypedTensorTrace
