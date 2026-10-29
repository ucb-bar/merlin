/- Additive strengthening of the draft f32-accumulation contract.

The existing zero-initialized contract is an exact specialization. The new
initialized contract records an arbitrary f32 initial accumulator at each cell,
independent operand formats, the actual maps, ordered updates and one final
conversion. It does not broaden the compiler gate or grant numerical approval.
In particular, extending a source half initializer is an explicit conversion;
the law-free FloatModel does not identify ext(+0_half) with literal +0_f32. -/
import MerlinVerify.Rules.Bf16F32AccTensor
import MerlinVerify.Sem.Defined

namespace MerlinVerify.Rules.Bf16F32AccInitialized

open MerlinVerify MerlinVerify.Sem.Ops.Linalg
open Bf16F32AccContract Bf16F32AccTensor

def accumulateFrom (M : FloatModel) (license : License) {ka kb : FKind} (g : Geometry)
    (A : Idx → Option (FBits ka)) (B : Idx → Option (FBits kb))
    (initial : Idx → Option (FBits .f32)) : Idx → Option (FBits .f32) :=
  W8A8TensorTrace.accumulate (Geometry.coordinate g.out g.outMap)
    (fun point acc => stepOpt M license (A (Geometry.coordinate g.left g.leftMap point))
      (B (Geometry.coordinate g.right g.rightMap point)) acc) initial (allPoints g.ranges)

def resultFrom (M : FloatModel) (license : License) {ka kb : FKind} (kc : FKind) (g : Geometry)
    (A : Idx → Option (FBits ka)) (B : Idx → Option (FBits kb))
    (initial : Idx → Option (FBits .f32)) : Tensor :=
  Tensor.canon g.out (.float kc) fun cell =>
    value kc ((accumulateFrom M license g A B initial cell).map (trunc M kc))

/-- A proposed source-half initialization has one explicit extension per cell.
No zero-preservation, rounding or IEEE identity is assumed. -/
def extendInitial (M : FloatModel) (kc : FKind)
    (C : Idx → Option (FBits kc)) : Idx → Option (FBits .f32) :=
  fun cell => (C cell).map (ext M kc)

/-- The previously checked zero contract is preserved byte-for-byte in meaning. -/
theorem zero_specialization (M : FloatModel) (license : License) {ka kb : FKind}
    (kc : FKind) (g : Geometry) (A : Idx → Option (FBits ka))
    (B : Idx → Option (FBits kb)) :
    resultFrom M license kc g A B (fun _ => some zero) = result M license kc g A B := rfl

/-- Every output cell retains its exact ordered trace, including repeated writes. -/
theorem accumulateFrom_cell (M : FloatModel) (license : License) {ka kb : FKind}
    (g : Geometry) (A : Idx → Option (FBits ka)) (B : Idx → Option (FBits kb))
    (initial : Idx → Option (FBits .f32)) (cell : Idx) :
    accumulateFrom M license g A B initial cell =
      (allPoints g.ranges).foldl (fun acc point =>
        if cell = Geometry.coordinate g.out g.outMap point then
          stepOpt M license (A (Geometry.coordinate g.left g.leftMap point))
            (B (Geometry.coordinate g.right g.rightMap point)) acc
        else acc) (initial cell) := W8A8TensorTrace.accumulate_cell _ _ _ _ _

/-- Empty reduction spaces retain the initializer, followed by the declared
final conversion. This is not generally the identity in a law-free model. -/
theorem empty_trace (M : FloatModel) (license : License) {ka kb : FKind}
    (kc : FKind) (g : Geometry) (A : Idx → Option (FBits ka))
    (B : Idx → Option (FBits kb)) (initial : Idx → Option (FBits .f32))
    (empty : allPoints g.ranges = []) :
    resultFrom M license kc g A B initial =
      Tensor.canon g.out (.float kc) (fun cell => value kc ((initial cell).map (trunc M kc))) := by
  simp [resultFrom, accumulateFrom, W8A8TensorTrace.accumulate, empty]

private theorem fold_defined (M : FloatModel) (license : License) {ka kb : FKind}
    (g : Geometry) (A : Idx → Option (FBits ka)) (B : Idx → Option (FBits kb))
    (cell : Idx) (points : List Idx)
    (inputs : ∀ point ∈ points,
      (∃ a, A (Geometry.coordinate g.left g.leftMap point) = some a) ∧
      (∃ b, B (Geometry.coordinate g.right g.rightMap point) = some b))
    (initial : Option (FBits .f32)) (initialized : ∃ c, initial = some c) :
    ∃ c, points.foldl (fun acc point =>
      if cell = Geometry.coordinate g.out g.outMap point then
        stepOpt M license (A (Geometry.coordinate g.left g.leftMap point))
          (B (Geometry.coordinate g.right g.rightMap point)) acc else acc) initial = some c := by
  induction points generalizing initial with
  | nil => exact initialized
  | cons point rest ih =>
    obtain ⟨⟨a, ha⟩, ⟨b, hb⟩⟩ := inputs point (List.mem_cons_self ..)
    obtain ⟨c, rfl⟩ := initialized
    rw [List.foldl_cons]
    apply ih (fun p hp => inputs p (List.mem_cons_of_mem _ hp))
    by_cases hc : cell = Geometry.coordinate g.out g.outMap point
    · exact ⟨step M license a b c, by simp [hc, ha, hb, stepOpt]⟩
    · exact ⟨c, by simp [hc]⟩

/-- Initialized operands plus checked mapped accesses give defined results for
every FloatModel. Formats may differ; no product or reduction is reassociated.
The hypothesis concerns all input cells, not just sampled outputs. -/
theorem resultFrom_defined (M : FloatModel) (license : License) {ka kb : FKind}
    (kc : FKind) (g : Geometry) (A : Idx → Option (FBits ka))
    (B : Idx → Option (FBits kb)) (initial : Idx → Option (FBits .f32))
    (leftMap : Check.mapInBounds g.ranges g.leftMap g.left = true)
    (rightMap : Check.mapInBounds g.ranges g.rightMap g.right = true)
    (left : ∀ p, Idx.InBounds g.left p → ∃ a, A p = some a)
    (right : ∀ p, Idx.InBounds g.right p → ∃ b, B p = some b)
    (initialized : ∀ p, Idx.InBounds g.out p → ∃ c, initial p = some c) :
    (RVal.tensor (resultFrom M license kc g A B initial)).Defined := by
  intro cell hc
  change Idx.InBounds g.out cell at hc
  have inputs : ∀ point ∈ allPoints g.ranges,
      (∃ a, A (Geometry.coordinate g.left g.leftMap point) = some a) ∧
      (∃ b, B (Geometry.coordinate g.right g.rightMap point) = some b) := by
    intro point hp
    have bounds := (mem_allPoints _ _).1 hp
    obtain ⟨li, hl, hli⟩ := Check.mapInBounds_sound leftMap bounds
    obtain ⟨ri, hr, hri⟩ := Check.mapInBounds_sound rightMap bounds
    simpa [Geometry.coordinate, W8A8TensorOperands.coordinate, hl, hr] using
      And.intro (left li hli) (right ri hri)
  obtain ⟨c, h⟩ := fold_defined M license g A B cell (allPoints g.ranges) inputs
    (initial cell) (initialized cell hc)
  rw [← accumulateFrom_cell] at h
  simp [resultFrom, Tensor.canon, hc, h, value]

/-- Definedness of the existing zero-initialized contract is a corollary. -/
theorem result_defined (M : FloatModel) (license : License) {ka kb : FKind}
    (kc : FKind) (g : Geometry) (A : Idx → Option (FBits ka))
    (B : Idx → Option (FBits kb))
    (leftMap : Check.mapInBounds g.ranges g.leftMap g.left = true)
    (rightMap : Check.mapInBounds g.ranges g.rightMap g.right = true)
    (left : ∀ p, Idx.InBounds g.left p → ∃ a, A p = some a)
    (right : ∀ p, Idx.InBounds g.right p → ∃ b, B p = some b) :
    (RVal.tensor (result M license kc g A B)).Defined :=
  resultFrom_defined M license kc g A B (fun _ => some zero) leftMap rightMap left right
    (fun _ _ => ⟨zero, rfl⟩)

/-- The actual accumulating generic implements the initialized contract.
Geometry and scalar-region matching are checked premises; the initial tensor
is retained, including poison and different values in different output cells. -/
theorem accumulateFrom_sound {ib : Nat} (ctx : Ctx) (call : CallSem) (env : Env)
    (iters : List IterKind) {ka kb : FKind} (license : License) (g : Geometry)
    (geometry : Check.StructuredOk [g.leftMap, g.rightMap, g.outMap] iters.length
      [.tensor g.left (.float ka), .tensor g.right (.float kb), .tensor g.out (.float .f32)] g.ranges)
    {actual : Region}
    (matched : Check.alphaRegion [] (Bf16F32AccBody.accRegion ka kb license) actual = true)
    (A : Idx → Option (FBits ka)) (B : Idx → Option (FBits kb))
    (initial : Idx → Option (FBits .f32)) :
    genericSem ctx [g.leftMap, g.rightMap, g.outMap] iters
      [.tensor (floats ka g.left A), .tensor (floats kb g.right B)]
      [.tensor (floats .f32 g.out initial)]
      (fun c => evalRegion Sem.Ops.opSem c call env actual) =
      .ok [.tensor (Tensor.canon g.out (.float .f32)
        (fun cell => value .f32 (accumulateFrom ctx.M license g A B initial cell)))] := by
  have hin := geometry.inBounds
  obtain ⟨hl, hr, ho⟩ : Check.mapInBounds g.ranges g.leftMap g.left = true ∧
      Check.mapInBounds g.ranges g.rightMap g.right = true ∧
      Check.mapInBounds g.ranges g.outMap g.out = true := by
    cases hin with
    | cons hl rest => cases rest with
      | cons hr rest => cases rest with
        | cons ho _ => exact ⟨hl, hr, ho⟩
  apply W8A8TensorTrace.generic_sound (inTys := [.tensor g.left (.float ka), .tensor g.right (.float kb)])
    geometry (.cons (floats_typed ib ka g.left A) (.cons (floats_typed ib kb g.right B) .nil))
    (value .f32) initial (floats_typed ib .f32 g.out initial) rfl _
    (Geometry.coordinate g.out g.outMap)
    (fun point acc => stepOpt ctx.M license (A (Geometry.coordinate g.left g.leftMap point))
      (B (Geometry.coordinate g.right g.rightMap point)) acc)
    (fun point hp => W8A8TensorOperands.coordinate_sound ho hp)
  intro point hp state
  exact bodyAt_sound ctx call env license matched _ _ g.leftMap g.rightMap g.outMap g.out
    _ _ point state _ (W8A8TensorOperands.read_sound ⟨g.left, .float ka, fun p => value ka (A p)⟩
      g.leftMap point hl hp)
    (W8A8TensorOperands.read_sound ⟨g.right, .float kb, fun p => value kb (B p)⟩
      g.rightMap point hr hp) (W8A8TensorOperands.coordinate_sound ho hp)

#print axioms zero_specialization
#print axioms accumulateFrom_cell
#print axioms empty_trace
#print axioms resultFrom_defined
#print axioms result_defined
#print axioms accumulateFrom_sound

end MerlinVerify.Rules.Bf16F32AccInitialized
