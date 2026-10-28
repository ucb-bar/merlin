/- Tensor lifting of the accepted signed byte MAC. Geometry and source correspondence are
separate obligations; this file preserves every actual mapped read and repeated output write. -/
import MerlinVerify.Rules.W8A8TensorTrace
import MerlinVerify.Check.PolicyW8A8Body

namespace MerlinVerify.Rules.W8A8IntegerTensor

open MerlinVerify MerlinVerify.Sem.Ops.Linalg
open W8A8IntegerBody

def step (left right : Idx → Option (BitVec 8)) (point : Idx)
    (acc : Option (BitVec 32)) : Option (BitVec 32) := do
  let a ← left point
  let b ← right point
  let z ← acc
  pure (W8A8Accumulation.step z (a, b))

theorem step_value (a b : Option (BitVec 8)) (z : Option (BitVec 32)) :
    value 32 (step (fun _ => a) (fun _ => b) [] z) = result a b z := by
  cases a <;> cases b <;> cases z <;> rfl

def tensor (w : Nat) (shape : List Nat) (get : Idx → Option (BitVec w)) : Tensor :=
  W8A8TensorTrace.pack shape (.int w) (value w) get

def pairs (coordinate : Idx → Idx) (left right : Idx → BitVec 8)
    (points : List Idx) (cell : Idx) : List W8A8Accumulation.Pair :=
  (points.filter (fun p => decide (cell = coordinate p))).map (fun p => (left p, right p))

private theorem initialized_fold (left right : Idx → BitVec 8) (points : List Idx)
    (initial : BitVec 32) :
    points.foldl (fun acc p => step (some ∘ left) (some ∘ right) p acc) (some initial) =
      some (W8A8Accumulation.accumulate (points.map (fun p => (left p, right p))) initial) := by
  induction points generalizing initial with
  | nil => rfl
  | cons point rest ih =>
    simp only [List.foldl_cons, step, Function.comp_def, List.map_cons,
      W8A8Accumulation.accumulate] at *
    exact ih (W8A8Accumulation.step initial (left point, right point))

/-- A cell's full tensor trace is exactly its selected dot-product trace, with no cross-cell
updates. This also covers a zero-length reduction, which retains the initial accumulator. -/
theorem initialized_cell (coordinate : Idx → Idx) (left right : Idx → BitVec 8)
    (points : List Idx) (initial : Idx → BitVec 32) (cell : Idx) :
    W8A8TensorTrace.accumulate coordinate (step (some ∘ left) (some ∘ right))
      (some ∘ initial) points cell =
      some (W8A8Accumulation.accumulate (pairs coordinate left right points cell) (initial cell)) := by
  rw [W8A8TensorTrace.accumulate_cell]
  simpa [pairs, List.foldl_filter] using initialized_fold left right
    (points.filter (fun p => decide (cell = coordinate p))) (initial cell)

/-- Exact integer meaning of a zero-initialized cell. The selected trace is derived from its
actual coordinates; the length bound must still be discharged from checked geometry. -/
theorem exact_cell (coordinate : Idx → Idx) (left right : Idx → BitVec 8)
    (points : List Idx) (cell : Idx)
    (bound : ((pairs coordinate left right points cell).length : Int) * 16384 < 2147483648) :
    (W8A8TensorTrace.accumulate coordinate (step (some ∘ left) (some ∘ right))
      (fun _ => some 0) points cell).map BitVec.toInt =
      some (W8A8Accumulation.dot (pairs coordinate left right points cell)) := by
  rw [show (fun _ : Idx => some (0 : BitVec 32)) = some ∘ (fun _ => 0) from rfl,
    initialized_cell]
  simp only [Option.map_some, W8A8Accumulation.exact_of_length _ bound]

theorem tensor_typed (ib w : Nat) (shape : List Nat) (get : Idx → Option (BitVec w)) :
    Check.RVal.HasTy ib (.tensor (tensor w shape get)) (.tensor shape (.int w)) := by
  refine ⟨rfl, rfl, ?_⟩
  intro cell _
  cases h : get cell <;> simp [tensor, W8A8TensorTrace.pack, value, Check.Val.HasTy,
    Scalar.conforms, h]

/-- The actual body checker closes the scalar obligation at every loop point and accumulator.
Input-coordinate selection is explicit; callers cannot replace it with a provenance assertion. -/
theorem bodyAt_sound {actual : Region} (accepted : Check.PolicyW8A8Body.check actual = true)
    (ctx : Ctx) (call : CallSem) (env : Env) (left right : RVal)
    (leftMap rightMap outMap : AffineMap) (shape : List Nat)
    (a b : Idx → Option (BitVec 8)) (coordinate : Idx → Idx) (point : Idx)
    (ha : readAt left leftMap point = .ok (value 8 (a point)))
    (hb : readAt right rightMap point = .ok (value 8 (b point)))
    (ho : (evalMap outMap point >>= toIdx shape) = some (coordinate point))
    (state : Idx → Option (BitVec 32)) :
    bodyAt ctx [left, right] [leftMap, rightMap] [outMap] [tensor 32 shape state]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) point =
      .ok [value 32 (step a b point (state (coordinate point)))] := by
  have hout : readAt (.tensor (tensor 32 shape state)) outMap point =
      .ok (value 32 (state (coordinate point))) := by
    simp only [readAt, tensor, W8A8TensorTrace.pack, ho]
  have hs := Check.PolicyW8A8Body.check_sound accepted
    { ctx with loopIndex := point } call env (a point) (b point) (state (coordinate point))
  have hv : value 32 (step a b point (state (coordinate point))) =
      result (a point) (b point) (state (coordinate point)) :=
    step_value (a point) (b point) (state (coordinate point))
  simp [bodyAt, ha, hb, hout, hs, hv]

/-- Complete generic execution for checked integer bodies, including poison and empty loops.
All layout and access obligations remain visible until discharged by the enclosing site checker. -/
theorem generic_sound {ib : Nat} {ctx : Ctx} {maps : List AffineMap}
    {iters : List IterKind} {leftShape rightShape outShape ranges : List Nat}
    {left right : RVal} {leftMap rightMap outMap : AffineMap} {actual : Region}
    (geometry : Check.StructuredOk maps iters.length
      [.tensor leftShape (.int 8), .tensor rightShape (.int 8), .tensor outShape (.int 32)] ranges)
    (hl : Check.RVal.HasTy ib left (.tensor leftShape (.int 8)))
    (hr : Check.RVal.HasTy ib right (.tensor rightShape (.int 8)))
    (hm : maps = [leftMap, rightMap, outMap])
    (accepted : Check.PolicyW8A8Body.check actual = true)
    (call : CallSem) (env : Env) (a b : Idx → Option (BitVec 8))
    (initial : Idx → Option (BitVec 32)) (coordinate : Idx → Idx)
    (ha : ∀ point, Idx.InBounds ranges point →
      readAt left leftMap point = .ok (value 8 (a point)))
    (hb : ∀ point, Idx.InBounds ranges point →
      readAt right rightMap point = .ok (value 8 (b point)))
    (ho : ∀ point, Idx.InBounds ranges point →
      (evalMap outMap point >>= toIdx outShape) = some (coordinate point)) :
    genericSem ctx maps iters [left, right] [.tensor (tensor 32 outShape initial)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon outShape (.int 32) (fun cell => value 32
        (W8A8TensorTrace.accumulate coordinate (step a b) initial (allPoints ranges) cell)))] := by
  apply W8A8TensorTrace.generic_sound (inTys :=
    [.tensor leftShape (.int 8), .tensor rightShape (.int 8)]) geometry (.cons hl (.cons hr .nil))
    (value 32) initial (tensor_typed ib 32 outShape initial)
    (by simp [hm]) _ coordinate (step a b) ho
  intro point hp state
  simpa [hm, tensor] using bodyAt_sound accepted ctx call env left right leftMap rightMap outMap
    outShape a b coordinate point (ha point hp) (hb point hp) (ho point hp) state

/-- A total coordinate function for the reference trace. The fallback is unreachable whenever
the independently checked geometry permits the access. -/
def coordinate (shape : List Nat) (map : AffineMap) (point : Idx) : Idx :=
  (evalMap map point >>= toIdx shape).getD []

theorem coordinate_sound {shape ranges : List Nat} {map : AffineMap} {point : Idx}
    (hm : Check.mapInBounds ranges map shape = true) (hp : Idx.InBounds ranges point) :
    (evalMap map point >>= toIdx shape) = some (coordinate shape map point) := by
  obtain ⟨index, hi, _⟩ := Check.mapInBounds_sound hm hp
  simp [coordinate, hi]

theorem read_sound {w : Nat} {shape ranges : List Nat} {map : AffineMap} {point : Idx}
    (hm : Check.mapInBounds ranges map shape = true) (hp : Idx.InBounds ranges point)
    (get : Idx → Option (BitVec w)) :
    readAt (.tensor (tensor w shape get)) map point =
      .ok (value w (get (coordinate shape map point))) := by
  have hi := coordinate_sound hm hp
  simp only [readAt, tensor, W8A8TensorTrace.pack]
  rw [hi]

/-- Geometry discharges every mapped-read/write hypothesis. The theorem covers arbitrary
signed-byte tensors, poison elements, any initial i32 tensor, and every checked iteration box. -/
theorem checked_generic_sound {ib : Nat} {ctx : Ctx} {iters : List IterKind}
    {leftShape rightShape outShape ranges : List Nat}
    {leftMap rightMap outMap : AffineMap} {actual : Region}
    (geometry : Check.StructuredOk [leftMap, rightMap, outMap] iters.length
      [.tensor leftShape (.int 8), .tensor rightShape (.int 8), .tensor outShape (.int 32)] ranges)
    (accepted : Check.PolicyW8A8Body.check actual = true)
    (call : CallSem) (env : Env) (left right : Idx → Option (BitVec 8))
    (initial : Idx → Option (BitVec 32)) :
    genericSem ctx [leftMap, rightMap, outMap] iters
      [.tensor (tensor 8 leftShape left), .tensor (tensor 8 rightShape right)]
      [.tensor (tensor 32 outShape initial)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon outShape (.int 32) (fun cell => value 32
        (W8A8TensorTrace.accumulate (coordinate outShape outMap)
          (step (fun p => left (coordinate leftShape leftMap p))
            (fun p => right (coordinate rightShape rightMap p))) initial
          (allPoints ranges) cell)))] := by
  have hb := geometry.inBounds
  cases hb with
  | cons hl rest =>
    cases rest with
    | cons hr rest =>
      cases rest with
      | cons ho _ =>
        exact generic_sound geometry (tensor_typed ib 8 leftShape left)
          (tensor_typed ib 8 rightShape right) rfl accepted call env _ _ initial _
          (fun p hp => read_sound hl hp left) (fun p hp => read_sound hr hp right)
          (fun p hp => coordinate_sound ho hp)

#print axioms bodyAt_sound
#print axioms generic_sound
#print axioms checked_generic_sound
#print axioms initialized_cell
#print axioms exact_cell

end MerlinVerify.Rules.W8A8IntegerTensor
