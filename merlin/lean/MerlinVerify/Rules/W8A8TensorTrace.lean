/- Ordered tensor accumulation for W8A8. This lifts scalar contracts through actual writes,
including repeated output coordinates and empty iteration spaces. No whole-policy claim. -/
import MerlinVerify.Check.StructuredRuntime
import MerlinVerify.Sem.Facts

namespace MerlinVerify.Rules.W8A8TensorTrace

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

variable {α : Type}

def update (coordinate : Idx → Idx) (step : Idx → α → α)
    (state : Idx → α) (point : Idx) : Idx → α :=
  fun cell => if cell = coordinate point then step point (state cell) else state cell

def accumulate (coordinate : Idx → Idx) (step : Idx → α → α)
    (initial : Idx → α) (points : List Idx) : Idx → α :=
  points.foldl (update coordinate step) initial

def pack (shape : List Nat) (elt : Ty) (encode : α → Val) (state : Idx → α) : Tensor :=
  { shape, elt, get := fun cell => encode (state cell) }

/-- Every output cell sees only its own updates, in the original lexicographic order. -/
theorem accumulate_cell (coordinate : Idx → Idx) (step : Idx → α → α)
    (initial : Idx → α) (points : List Idx) (cell : Idx) :
    accumulate coordinate step initial points cell =
      points.foldl (fun acc point =>
        if cell = coordinate point then step point acc else acc) (initial cell) := by
  induction points generalizing initial with
  | nil => rfl
  | cons point rest ih =>
    simp only [accumulate, List.foldl_cons] at *
    rw [ih]
    rfl

private theorem pack_update (shape : List Nat) (elt : Ty) (encode : α → Val)
    (coordinate : Idx → Idx) (step : Idx → α → α) (state : Idx → α) (point : Idx) :
    (pack shape elt encode state).set (coordinate point)
        (encode (step point (state (coordinate point)))) =
      pack shape elt encode (update coordinate step state point) := by
  unfold pack Tensor.set update
  congr 1
  funext cell
  split
  · next h => simp [h]
  · rfl

/-- Lift a scalar step through the complete reduction trace. Bounds and the body contract are
explicit obligations; a later checked site must discharge them from its actual operands/maps. -/
theorem fold_sound (ctx : Ctx) (ins : List RVal) (inMaps : List AffineMap)
    (outMap : AffineMap) (body : Ctx → RegionSem) (shape : List Nat) (elt : Ty)
    (encode : α → Val) (coordinate : Idx → Idx) (step : Idx → α → α) :
    ∀ (points : List Idx),
      (∀ point ∈ points, (evalMap outMap point >>= toIdx shape) = some (coordinate point)) →
      (∀ point ∈ points, ∀ state : Idx → α,
        bodyAt ctx ins inMaps [outMap] [pack shape elt encode state] body point =
          .ok [encode (step point (state (coordinate point)))]) →
      ∀ initial : Idx → α,
        points.foldlM (loopStep ctx ins inMaps [outMap] body) [pack shape elt encode initial] =
          .ok [pack shape elt encode (accumulate coordinate step initial points)] := by
  intro points
  induction points with
  | nil => intro _ _ initial; rfl
  | cons point rest ih =>
    intro hindex hbody initial
    have hi := hindex point (List.mem_cons_self ..)
    have hb := hbody point (List.mem_cons_self ..) initial
    have hs : loopStep ctx ins inMaps [outMap] body [pack shape elt encode initial] point =
        .ok [pack shape elt encode (update coordinate step initial point)] := by
      have hw : writeAt (pack shape elt encode initial) outMap point
          (encode (step point (initial (coordinate point)))) =
          .ok (pack shape elt encode (update coordinate step initial point)) := by
        simp only [writeAt, pack, hi]
        congr 1
        exact pack_update shape elt encode coordinate step initial point
      unfold loopStep
      rw [hb]
      simp only [Outcome.bindOp_eq, Outcome.bind_ok, List.zip_cons_cons, List.zip_nil_right,
        List.mapM_cons, List.mapM_nil, hw, Outcome.pure_eq]
    rw [List.foldlM_cons, hs]
    simp only [Outcome.bindOp_eq, Outcome.bind_ok]
    exact ih (fun p hp => hindex p (List.mem_cons_of_mem _ hp))
      (fun p hp => hbody p (List.mem_cons_of_mem _ hp)) _

/-- Independently checked geometry plus the proved scalar step determines the entire tensor,
including canonical out-of-bounds poison. Iterator order is retained, never reassociated. -/
theorem generic_sound {ib : Nat} {ctx : Ctx} {maps : List AffineMap}
    {iters : List IterKind} {inTys : List Check.VTy} {ranges shape : List Nat}
    {ins : List RVal} {elt : Ty} {outMap : AffineMap}
    (geometry : Check.StructuredOk maps iters.length (inTys ++ [.tensor shape elt]) ranges)
    (typedIns : Check.HasTys ib ins inTys)
    (encode : α → Val) (initial : Idx → α)
    (typedOut : Check.RVal.HasTy ib (.tensor (pack shape elt encode initial)) (.tensor shape elt))
    (output : maps.drop ins.length = [outMap]) (body : Ctx → RegionSem)
    (coordinate : Idx → Idx) (step : Idx → α → α)
    (index : ∀ point, Idx.InBounds ranges point →
      (evalMap outMap point >>= toIdx shape) = some (coordinate point))
    (scalar : ∀ point, Idx.InBounds ranges point → ∀ state : Idx → α,
      bodyAt ctx ins (maps.take ins.length) [outMap] [pack shape elt encode state] body point =
        .ok [encode (step point (state (coordinate point)))]) :
    genericSem ctx maps iters ins [.tensor (pack shape elt encode initial)] body =
      .ok [.tensor (Tensor.canon shape elt
        (fun cell => encode (accumulate coordinate step initial (allPoints ranges) cell)))] := by
  have ho : Check.HasTys ib ([pack shape elt encode initial].map RVal.tensor)
      [.tensor shape elt] := .cons typedOut .nil
  rw [show [.tensor (pack shape elt encode initial)] =
      [pack shape elt encode initial].map RVal.tensor from rfl]
  rw [Check.genericSem_execute geometry typedIns ho, output]
  rw [fold_sound ctx ins (maps.take ins.length) outMap body shape elt encode coordinate step
    (allPoints ranges) (fun p hp => index p ((mem_allPoints _ _).1 hp))
    (fun p hp => scalar p ((mem_allPoints _ _).1 hp)) initial]
  rfl

#print axioms accumulate_cell
#print axioms fold_sound
#print axioms generic_sound

end MerlinVerify.Rules.W8A8TensorTrace
