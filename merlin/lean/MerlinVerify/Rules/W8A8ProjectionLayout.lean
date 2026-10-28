/- Source-derived geometry for projection contractions. The actual maps, operand shapes
and iterator kinds determine grouping and the per-output reduction bound. -/
import MerlinVerify.Rules.W8A8ProjectionTrace
import MerlinVerify.Check.PolicyW8A8Geometry

namespace MerlinVerify.Rules.W8A8ProjectionLayout
open MerlinVerify Sem.Ops.Linalg Check W8A8ProjectionTrace

structure Layout where
  leftShape : List Nat
  rightShape : List Nat
  outputShape : List Nat
  leftMap : AffineMap
  rightMap : AffineMap
  iterators : List IterKind
  ranges : List Nat

def Layout.mask (layout : Layout) : List Bool := layout.iterators.map (· == .parallel)
def Layout.maps (layout : Layout) : List AffineMap := [layout.leftMap, layout.rightMap, outputMap layout.mask]
def Layout.types (layout : Layout) : List VTy :=
  [.tensor layout.leftShape (.int 8), .tensor layout.rightShape (.int 8), .tensor layout.outputShape (.int 32)]
def Layout.reductionSize (layout : Layout) : Nat := (select (layout.mask.map not) layout.ranges).prod

def dimensions? (map : AffineMap) : Option (List Nat) := map.results.mapM fun
  | .dim index => some index
  | _ => none

/-- Expected scale dimensions are positions in the source operand, not iterator dimensions. -/
def kept (mask : List Bool) (dimensions : List Nat) : List Nat :=
  dimensions.zipIdx.filterMap fun (dimension, position) => if mask[dimension]?.getD false then some position else none

def outputPositions (mask : List Bool) (dimensions : List Nat) : List Nat :=
  dimensions.filterMap fun dimension =>
    if mask[dimension]?.getD false then some ((mask.take dimension).count true) else none

def Valid (layout : Layout) : Prop :=
  StructuredOk layout.maps layout.iterators.length layout.types layout.ranges ∧
  layout.mask.length = layout.ranges.length ∧
  layout.outputShape = select layout.mask layout.ranges ∧ layout.reductionSize < 131072

def check (layout : Layout) : Bool :=
  (structuredRanges? layout.maps layout.iterators.length layout.types == some layout.ranges) &&
  decide (layout.mask.length = layout.ranges.length) &&
  decide (layout.outputShape = select layout.mask layout.ranges) && decide (layout.reductionSize < 131072)

theorem check_sound {layout : Layout} (checked : check layout = true) : Valid layout := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at checked
  exact ⟨structuredRanges?_spec (eq_of_beq checked.1.1.1), checked.1.1.2, checked.1.2, checked.2⟩

def inspect (leftShape rightShape outputShape : List Nat) (actualMaps : List AffineMap)
    (iterators : List IterKind) : Option Layout := do
  let [leftMap, rightMap, actualOutput] := actualMaps | none
  let leftDimensions ← dimensions? leftMap
  let rightDimensions ← dimensions? rightMap
  if ¬ leftDimensions.Nodup ∨ ¬ rightDimensions.Nodup then none
  let parallel := iterators.map (· == .parallel)
  if actualOutput != outputMap parallel || !iterators.contains .reduction then none
  let ranges ← structuredRanges? actualMaps iterators.length
    [.tensor leftShape (.int 8), .tensor rightShape (.int 8), .tensor outputShape (.int 32)]
  let layout := Layout.mk leftShape rightShape outputShape leftMap rightMap iterators ranges
  if check layout then some layout else none

theorem inspect_sound {leftShape rightShape outputShape : List Nat} {actualMaps : List AffineMap}
    {iterators : List IterKind} {layout : Layout}
    (selected : inspect leftShape rightShape outputShape actualMaps iterators = some layout) : Valid layout := by
  unfold inspect at selected
  repeat' first
    | simp only [bind, Option.bind] at selected
    | split at selected
    | contradiction
  all_goals cases selected; exact check_sound (by assumption)

theorem every_prefix_exact {layout : Layout} (valid : Valid layout) (cell : Idx)
    (bounds : Idx.InBounds layout.outputShape cell) (left right : Idx → BitVec 8) (count : Nat) :
    let pairs := W8A8IntegerTensor.pairs
      (W8A8IntegerTensor.coordinate layout.outputShape (outputMap layout.mask)) left right
      (allPoints layout.ranges) cell
    (W8A8Accumulation.accumulate (pairs.take count)).toInt = W8A8Accumulation.dot (pairs.take count) := by
  rw [valid.2.2.1] at bounds ⊢
  exact W8A8ProjectionTrace.every_prefix_exact valid.2.1 bounds valid.2.2.2 left right count

#print axioms check_sound
#print axioms inspect_sound
#print axioms every_prefix_exact
end MerlinVerify.Rules.W8A8ProjectionLayout
