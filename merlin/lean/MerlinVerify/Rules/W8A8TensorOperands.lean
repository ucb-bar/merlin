/- Shared mapped tensor reads for W8A8 stages. Operand elements remain arbitrary values;
the scalar contract, rather than a shape declaration, establishes their arithmetic meaning. -/
import MerlinVerify.Check.StructuredRuntime
import MerlinVerify.Sem.Facts

namespace MerlinVerify.Rules.W8A8TensorOperands

open MerlinVerify MerlinVerify.Sem.Ops.Linalg

structure Input where
  shape : List Nat
  elt : Ty
  get : Idx → Val

def Input.value (input : Input) : RVal := .tensor ⟨input.shape, input.elt, input.get⟩
def Input.type (input : Input) : Check.VTy := .tensor input.shape input.elt

/-- The fallback cannot be reached under the checked access bound. -/
def coordinate (shape : List Nat) (map : AffineMap) (point : Idx) : Idx :=
  (evalMap map point >>= toIdx shape).getD []

def Input.at (input : Input) (map : AffineMap) (point : Idx) : Val :=
  input.get (coordinate input.shape map point)

def values (inputs : List Input) (maps : List AffineMap) (point : Idx) : List Val :=
  (inputs.zip maps).map (fun (input, map) => input.at map point)

theorem coordinate_sound {shape ranges : List Nat} {map : AffineMap} {point : Idx}
    (hm : Check.mapInBounds ranges map shape = true) (hp : Idx.InBounds ranges point) :
    (evalMap map point >>= toIdx shape) = some (coordinate shape map point) := by
  obtain ⟨index, hi, _⟩ := Check.mapInBounds_sound hm hp
  simp [coordinate, hi]

theorem coordinate_identity {shape : List Nat} {point : Idx} (hp : Idx.InBounds shape point) :
    coordinate shape (identityMap shape.length) point = point := by
  unfold coordinate
  have hl : shape.length = point.length := hp.1.symm
  rw [hl, evalMap_identity]
  simp only [Option.bind_eq_bind, Option.bind_some, toIdx_ofNat hp, Option.getD_some]

theorem read_sound {ranges : List Nat} (input : Input) (map : AffineMap) (point : Idx)
    (hm : Check.mapInBounds ranges map input.shape = true) (hp : Idx.InBounds ranges point) :
    readAt input.value map point = .ok (input.at map point) := by
  simp only [Input.value, Input.at, readAt]
  rw [coordinate_sound hm hp]

theorem reads_sound {ranges : List Nat} (point : Idx) (hp : Idx.InBounds ranges point) :
    ∀ (inputs : List Input) (maps : List AffineMap),
      Check.MapsInBounds ranges maps (inputs.map Input.type) →
      ((inputs.map Input.value).zip maps).mapM (fun (v, map) => readAt v map point) =
        .ok (values inputs maps point) := by
  intro inputs
  induction inputs with
  | nil => intro maps h; cases h; rfl
  | cons input rest ih =>
    intro maps h
    cases h with
    | cons hm hs =>
      simp only [List.map_cons, List.zip_cons_cons, List.mapM_cons,
        read_sound input _ point hm hp, Outcome.bindOp_eq, Outcome.bind_ok, ih _ hs]
      rfl

/-- Structured header acceptance depends on actual operand shapes, not their element values. -/
theorem runtime {maps : List AffineMap} {n : Nat} {ranges : List Nat} (inputs : List Input)
    (output : Tensor)
    (geometry : Check.StructuredOk maps n
      (inputs.map Input.type ++ [Check.VTy.tensor output.shape output.elt]) ranges) :
    maps.length = inputs.length + 1 ∧
    maps.any (fun map => map.numDims ≠ n || map.numSyms ≠ 0) = false ∧
    (maps.zip (((inputs.map Input.value) ++ [RVal.tensor output]).map shapeOf)).any
      (fun pair => pair.1.results.length ≠ pair.2.length) = false ∧
    loopRanges maps (((inputs.map Input.value) ++ [RVal.tensor output]).map shapeOf) n = .ok ranges := by
  have shapes : ((inputs.map Input.value) ++ [RVal.tensor output]).map shapeOf =
      (inputs.map Input.type ++ [Check.VTy.tensor output.shape output.elt]).map Check.VTy.shape := by
    simp [Input.value, Input.type, shapeOf, Check.VTy.shape, List.map_map, Function.comp_def]
  refine ⟨by simpa using geometry.len, ?_, ?_, ?_⟩
  · rw [List.any_eq_false]
    intro map hm
    simp [(geometry.dims map hm).1, (geometry.dims map hm).2]
  · rw [shapes, List.zip_map_right, List.any_map, List.any_eq_false]
    intro pair hp
    have hin := List.forall₂_iff_zip.1 geometry.inBounds |>.2 hp
    simp only [Check.mapInBounds, Bool.and_eq_true, beq_iff_eq] at hin
    simp [hin.1]
  · rw [shapes]
    exact Check.mapM_ofOption _ _ _ _ geometry.loops

end MerlinVerify.Rules.W8A8TensorOperands
