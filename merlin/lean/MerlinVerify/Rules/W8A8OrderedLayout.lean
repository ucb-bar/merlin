/- Floating contraction geometry, recomputed from actual maps and declared static shapes.
The shared layout data is reused; integer-MAC typing and integer bounds are not imported as
facts about floating execution. -/
import MerlinVerify.Rules.W8A8ProjectionLayout

namespace MerlinVerify.Rules.W8A8OrderedLayout

open MerlinVerify Check Sem.Ops.Linalg W8A8ProjectionTrace

abbrev Layout := W8A8ProjectionLayout.Layout

def types (layout : Layout) : List VTy :=
  [.tensor layout.leftShape (.float .f32), .tensor layout.rightShape (.float .f32),
    .tensor layout.outputShape (.float .f32)]

def Valid (layout : Layout) : Prop :=
  StructuredOk layout.maps layout.iterators.length (types layout) layout.ranges ∧
    layout.mask.length = layout.ranges.length ∧ layout.outputShape = select layout.mask layout.ranges

def check (layout : Layout) : Bool :=
  (structuredRanges? layout.maps layout.iterators.length (types layout) == some layout.ranges) &&
    decide (layout.mask.length = layout.ranges.length) &&
    decide (layout.outputShape = select layout.mask layout.ranges)

theorem check_sound {layout : Layout} (checked : check layout = true) : Valid layout := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq] at checked
  exact ⟨structuredRanges?_spec (eq_of_beq checked.1.1), checked.1.2, checked.2⟩

def inspect (leftShape rightShape outputShape : List Nat) (actualMaps : List AffineMap)
    (iterators : List IterKind) : Option Layout := do
  let [leftMap, rightMap, actualOutput] := actualMaps | none
  let mask := iterators.map (· == .parallel)
  if actualOutput = outputMap mask then
    let ranges ← structuredRanges? actualMaps iterators.length
      [.tensor leftShape (.float .f32), .tensor rightShape (.float .f32), .tensor outputShape (.float .f32)]
    let layout := W8A8ProjectionLayout.Layout.mk leftShape rightShape outputShape leftMap rightMap iterators ranges
    if check layout then some layout else none
  else none

theorem inspect_sound {leftShape rightShape outputShape : List Nat} {actualMaps : List AffineMap}
    {iterators : List IterKind} {layout : Layout}
    (selected : inspect leftShape rightShape outputShape actualMaps iterators = some layout) :
    Valid layout ∧ layout.leftShape = leftShape ∧ layout.rightShape = rightShape ∧
      layout.outputShape = outputShape ∧ layout.maps = actualMaps ∧ layout.iterators = iterators := by
  unfold inspect at selected
  split at selected
  · next _maps leftMap rightMap actualOutput =>
    dsimp only at selected
    split at selected
    · next output =>
      cases rangesFound : structuredRanges? [leftMap, rightMap, actualOutput] iterators.length
          [.tensor leftShape (.float .f32), .tensor rightShape (.float .f32), .tensor outputShape (.float .f32)] with
      | none => simp [rangesFound] at selected
      | some ranges =>
        simp only [rangesFound, bind, Option.bind] at selected
        split at selected
        · next checked =>
          cases selected
          refine ⟨check_sound checked, rfl, rfl, rfl, ?_, rfl⟩
          simp only [W8A8ProjectionLayout.Layout.maps, W8A8ProjectionLayout.Layout.mask]
          rw [← output]
        · cases selected
    · cases selected
  · cases selected

#print axioms check_sound
#print axioms inspect_sound

end MerlinVerify.Rules.W8A8OrderedLayout
