/-
Tensor meaning of the f32-accumulation contract's structured operations. Geometry is a checked
premise (`Check.StructuredOk`, produced by `Check.structuredRanges?`); regions are related to the
contract's templates only by `alphaRegion`. Iteration order is the frozen lexicographic order.
-/
import MerlinVerify.Rules.Bf16F32AccBody
import MerlinVerify.Rules.W8A8Pointwise
import MerlinVerify.Rules.W8A8Fill

namespace MerlinVerify.Rules.Bf16F32AccTensor

open MerlinVerify MerlinVerify.Sem.Ops.Linalg Bf16F32AccContract Bf16F32AccBody
open MerlinVerify.Check (alphaRegion)

/-- A tensor whose in-bounds elements are given by `get` (`none` is poison). -/
def floats (k : FKind) (shape : List Nat) (get : Idx → Option (FBits k)) : Tensor :=
  ⟨shape, .float k, fun cell => value k (get cell)⟩

theorem floats_typed (ib : Nat) (k : FKind) (shape : List Nat) (get : Idx → Option (FBits k)) :
    Check.RVal.HasTy ib (.tensor (floats k shape get)) (.tensor shape (.float k)) := by
  refine ⟨rfl, rfl, ?_⟩
  intro cell _
  cases h : get cell <;> simp [floats, value, Check.Val.HasTy, Scalar.conforms, h]

private theorem region_transfer (call : CallSem) (env : Env) {template actual : Region}
    (matched : alphaRegion [] template actual = true) (ctx : Ctx) :
    evalRegion Sem.Ops.opSem ctx call env actual = evalRegion Sem.Ops.opSem ctx call env template :=
  (Check.alphaRegion_sound call call [] template actual ctx env env matched
    (by intro x y hxy; simp [Check.Rho.find, List.lookup] at hxy) (fun _ _ => rfl)).symm

/-- One loop point: the operands read through their maps, the accumulator read through the
output map, and the actual region executing the contract's step. -/
theorem bodyAt_sound (ctx : Ctx) (call : CallSem) (env : Env) {ka kb : FKind} (license : License)
    {actual : Region} (matched : alphaRegion [] (accRegion ka kb license) actual = true)
    (left right : RVal) (leftMap rightMap outMap : AffineMap) (shape : List Nat)
    (a : Option (FBits ka)) (b : Option (FBits kb)) (point : Idx)
    (state : Idx → Option (FBits .f32)) (cell : Idx)
    (ha : readAt left leftMap point = .ok (value ka a))
    (hb : readAt right rightMap point = .ok (value kb b))
    (ho : (evalMap outMap point >>= toIdx shape) = some cell) :
    bodyAt ctx [left, right] [leftMap, rightMap] [outMap]
      [W8A8TensorTrace.pack shape (.float .f32) (value .f32) state]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) point =
      .ok [value .f32 (stepOpt ctx.M license a b (state cell))] := by
  have hout : readAt (.tensor (W8A8TensorTrace.pack shape (.float .f32) (value .f32) state)) outMap
      point = .ok (value .f32 (state cell)) := by
    simp only [readAt, W8A8TensorTrace.pack, ho]
  have hs := accRegion_sound { ctx with loopIndex := point } call env ka kb license a b (state cell)
  simp [bodyAt, ha, hb, hout, region_transfer call env matched, hs]

/-- The accumulating generic computes clauses 1–4 of the contract on every cell: typed operands,
any loop extents the checked geometry derives, any poison, and an actual region alpha-equivalent
to the contract's template. The destination is the `+0.0` fill. -/
theorem accumulate_sound {ib : Nat} (ctx : Ctx) (call : CallSem) (env : Env)
    (iters : List IterKind) {ka kb : FKind} (license : License) (g : Geometry)
    (geometry : Check.StructuredOk [g.leftMap, g.rightMap, g.outMap] iters.length
      [.tensor g.left (.float ka), .tensor g.right (.float kb), .tensor g.out (.float .f32)] g.ranges)
    {actual : Region} (matched : alphaRegion [] (accRegion ka kb license) actual = true)
    (A : Idx → Option (FBits ka)) (B : Idx → Option (FBits kb)) :
    genericSem ctx [g.leftMap, g.rightMap, g.outMap] iters
      [.tensor (floats ka g.left A), .tensor (floats kb g.right B)]
      [.tensor (Tensor.full g.out (.float .f32) (value .f32 (some zero)))]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon g.out (.float .f32)
        (fun cell => value .f32 (accumulate ctx.M license g A B cell)))] := by
  have hin := geometry.inBounds
  obtain ⟨hl, hr, ho⟩ : Check.mapInBounds g.ranges g.leftMap g.left = true ∧
      Check.mapInBounds g.ranges g.rightMap g.right = true ∧
      Check.mapInBounds g.ranges g.outMap g.out = true := by
    cases hin with
    | cons hl rest => cases rest with
      | cons hr rest => cases rest with
        | cons ho _ => exact ⟨hl, hr, ho⟩
  let initial : Idx → Option (FBits .f32) :=
    fun cell => if Idx.InBounds g.out cell then some zero else Option.none
  have hfull : Tensor.full g.out (.float .f32) (value .f32 (some zero)) =
      W8A8TensorTrace.pack g.out (.float .f32) (value .f32) initial := by
    simp only [Tensor.full, Tensor.canon, W8A8TensorTrace.pack, initial]
    congr 1
    funext cell
    split <;> rfl
  have typedOut : Check.RVal.HasTy ib
      (.tensor (W8A8TensorTrace.pack g.out (.float .f32) (value .f32) initial))
      (.tensor g.out (.float .f32)) := floats_typed ib .f32 g.out initial
  rw [hfull, W8A8TensorTrace.generic_sound (inTys := [.tensor g.left (.float ka), .tensor g.right (.float kb)])
    geometry (.cons (floats_typed ib ka g.left A) (.cons (floats_typed ib kb g.right B) .nil))
    (value .f32) initial typedOut rfl _ (Geometry.coordinate g.out g.outMap)
    (fun point acc => stepOpt ctx.M license (A (Geometry.coordinate g.left g.leftMap point))
      (B (Geometry.coordinate g.right g.rightMap point)) acc)
    (fun point hp => W8A8TensorOperands.coordinate_sound ho hp)]
  · have hcanon : Tensor.canon g.out (.float .f32) (fun cell => value .f32
        (W8A8TensorTrace.accumulate (Geometry.coordinate g.out g.outMap)
          (fun point acc => stepOpt ctx.M license (A (Geometry.coordinate g.left g.leftMap point))
            (B (Geometry.coordinate g.right g.rightMap point)) acc) initial (allPoints g.ranges) cell)) =
        Tensor.canon g.out (.float .f32) (fun cell => value .f32 (accumulate ctx.M license g A B cell)) := by
      apply Tensor.canon_ext
      intro cell hc
      simp only [accumulate, W8A8TensorTrace.accumulate_cell, initial, if_pos hc]
    rw [hcanon]
  · intro point hp state
    have ra := W8A8TensorOperands.read_sound ⟨g.left, .float ka, fun p => value ka (A p)⟩
      g.leftMap point hl hp
    have rb := W8A8TensorOperands.read_sound ⟨g.right, .float kb, fun p => value kb (B p)⟩
      g.rightMap point hr hp
    exact bodyAt_sound ctx call env license matched _ _ g.leftMap g.rightMap g.outMap g.out
      _ _ point state _ ra rb (W8A8TensorOperands.coordinate_sound ho hp)

/-- The final-conversion generic truncates every in-bounds accumulator into the result format
(clause 5). The destination may be `tensor.empty`, i.e. poison everywhere. -/
theorem truncate_sound (ctx : Ctx) (call : CallSem) (env : Env) (iters : List IterKind)
    (kc : FKind) (shape : List Nat) (accumulated : Idx → Option (FBits .f32))
    (geometry : Check.StructuredOk [identityMap shape.length, identityMap shape.length]
      iters.length [.tensor shape (.float .f32), .tensor shape (.float kc)] shape)
    {actual : Region} (matched : alphaRegion [] (truncRegion kc) actual = true) (initial : Idx → Val) :
    genericSem ctx [identityMap shape.length, identityMap shape.length] iters
      [.tensor (Tensor.canon shape (.float .f32) (fun cell => value .f32 (accumulated cell)))]
      [.tensor ⟨shape, .float kc, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon shape (.float kc)
        (fun cell => value kc ((accumulated cell).map (trunc ctx.M kc))))] := by
  let input : W8A8TensorOperands.Input :=
    ⟨shape, .float .f32, fun cell => value .f32 (if Idx.InBounds shape cell then accumulated cell
      else Option.none)⟩
  have hinput : RVal.tensor (Tensor.canon shape (.float .f32) (fun cell => value .f32 (accumulated cell))) =
      input.value := by
    simp only [W8A8TensorOperands.Input.value, input, Tensor.canon]
    congr 2
    funext cell
    split <;> rfl
  rw [hinput]
  have result := W8A8Pointwise.generic_sound ctx call env [input] ⟨shape, .float kc, initial⟩
    [identityMap shape.length, identityMap shape.length] iters actual geometry rfl
    (fun cell => value kc ((if Idx.InBounds shape cell then accumulated cell else Option.none).map
      (trunc ctx.M kc)))
    (by
      intro point hp old
      have hp' : Idx.InBounds shape point := hp
      have body := truncRegion_sound { ctx with loopIndex := point } call env kc
        (if Idx.InBounds shape point then accumulated point else Option.none) old
      simpa [W8A8TensorOperands.values, W8A8TensorOperands.Input.at,
        W8A8TensorOperands.coordinate_identity hp', input, region_transfer call env matched] using body)
  rw [show [input].map W8A8TensorOperands.Input.value = [input.value] from rfl] at result
  rw [result]
  have hcanon : Tensor.canon shape (.float kc) (fun cell => value kc
      ((if Idx.InBounds shape cell then accumulated cell else Option.none).map (trunc ctx.M kc))) =
      Tensor.canon shape (.float kc) (fun cell => value kc ((accumulated cell).map (trunc ctx.M kc))) := by
    apply Tensor.canon_ext
    intro cell hc
    simp only [hc, ite_true]
  rw [hcanon]

/-- The `+0.0` fill of the f32 destination (clause 4), from any initial contents. -/
theorem zero_fill_sound (ctx : Ctx) (call : CallSem) (env : Env) (shape : List Nat)
    {actual : Region} (matched : alphaRegion [] (W8A8Fill.fillRegion (.float .f32)) actual = true)
    (initial : Idx → Val) :
    genericSem ctx [W8A8Fill.scalarMap shape.length, identityMap shape.length]
      (List.replicate shape.length .parallel) [.scalar (value .f32 (some zero))]
      [.tensor ⟨shape, .float .f32, initial⟩]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.full shape (.float .f32) (value .f32 (some zero)))] := by
  simp only [region_transfer call env matched]
  exact W8A8Fill.fill_sound ctx call env shape (.float .f32) _ initial

#print axioms accumulate_sound
#print axioms truncate_sound
#print axioms zero_fill_sound

end MerlinVerify.Rules.Bf16F32AccTensor
