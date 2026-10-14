/-
Op family: `tensor.*` (design §4.3), static shapes only
(`mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td`, llvm-project `a47bddcce`).

* `empty`: every element is `poison` (contents unspecified).
* `extract_slice`/`insert_slice`: static offsets/sizes/strides (the dynamic sentinel or any dynamic
  operand is `stuck`); rank-reduced slices drop size-1 dimensions. Which size-1 dimensions MLIR
  drops (`getDroppedDims`) cannot change the result, since every size-1 dimension has coordinate 0.
  A slice that leaves the source is `stuck` (the verifier rejects it).
* `collapse_shape`/`expand_shape`: contiguous reassociation groups, which preserve row-major order;
  groups are checked against both shapes.
* `extract`/`insert` outside the bounds is immediate UB (a poison index may be out of bounds).
* `pad`, `bitcast`, `reshape`, `generate` and dynamic shapes: `stuck`.
-/
import MerlinVerify.Sem.Support

namespace MerlinVerify.Sem.Ops.Tensor

open MerlinVerify

/-! ### Helpers -/

/-- Inverse of `linearIndex` for a static shape (row-major). -/
def delinearize : List Nat → Nat → Idx
  | [], _ => []
  | _ :: ds, k => let inner := shapeSize ds; (k / inner) :: delinearize ds (k % inner)

def resultTensorTy (op : Op) : Option (List Nat × Ty) :=
  match op.results with
  | [r] => r.ty.staticTensor?
  | _ => none

/-- `reassociation = [[0, 1], [2]]` as groups of indices. -/
def reassociation? (op : Op) : Option (List (List Nat)) :=
  match op.attr? "reassociation" with
  | some (.array gs) => gs.mapM fun
    | .array is => is.mapM fun | .int i _ => if i ≥ 0 then some i.toNat else none | _ => none
    | _ => none
  | _ => none

/-- Groups are contiguous, in order, cover `0..lo.length-1`, and multiply to `hi`'s extents. -/
def reassocValid (groups : List (List Nat)) (lo hi : List Nat) : Bool :=
  groups.length == hi.length &&
  groups.flatten == List.range lo.length &&
  (groups.zip hi).all fun (g, h) => shapeSize (g.map fun i => lo.getD i 0) == h

def staticArray? (op : Op) (key : String) : Option (List Nat) :=
  -- MLIR's dynamic sentinel is INT64_MIN; any negative entry is rejected.
  (attrDenseArray? op key).bind fun xs => xs.mapM fun x => if x ≥ 0 then some x.toNat else none

/-- Place non-unit coordinates `cs` into `shape`: size-1 dims get coordinate 0, the others take
`cs` in order. -/
def placeNonUnit : List Nat → List Nat → Option Idx
  | [], [] => some []
  | [], _ :: _ => none
  | s :: ss, cs => if s = 1 then (0 :: ·) <$> placeNonUnit ss cs else
      match cs with
      | c :: cs' => (c :: ·) <$> placeNonUnit ss cs'
      | [] => none

/-- Positions of a rank-reduced index `r` (shape `reduced`) inside `sizes`: every size-1 dim gets
coordinate 0, the others take the reduced index's non-unit coordinates in order. -/
def expandReduced (sizes reduced : List Nat) (r : Idx) : Option Idx :=
  placeNonUnit sizes ((r.zip reduced).filterMap fun (x, s) => if s = 1 then none else some x)

/-- Element of a concatenation along `d` at index `i`, from parts starting at offset `off`. -/
def concatPick (d : Nat) (i : Idx) : List Tensor → Nat → Val
  | [], _ => .poison
  | p :: ps, off =>
    let e := p.shape.getD d 0
    let x := i.getD d 0
    if x < off + e then p.get (i.set d (x - off)) else concatPick d i ps (off + e)

def reducedCompatible (sizes reduced : List Nat) : Bool :=
  sizes.filter (· ≠ 1) == reduced.filter (· ≠ 1) && reduced.length ≤ sizes.length

def indexVal? : Val → Option Int
  | .some (.bits _ x) => some x.toInt
  | _ => none

/-! ### The family -/

def sem : OpSem := fun ctx _ op args _ =>
  match op.name with
  | "tensor.empty" =>
    match resultTensorTy op, args with
    | some (shape, elt), [] => .ok [.tensor (Tensor.full shape elt .poison)]
    | _, _ => .stuck "tensor.empty with dynamic sizes"
  | "tensor.splat" =>
    match resultTensorTy op, args with
    | some (shape, elt), [.scalar x] => .ok [.tensor (Tensor.full shape elt x)]
    | _, _ => .stuck "malformed tensor.splat"
  | "tensor.cast" =>
    match resultTensorTy op, args with
    | some (shape, elt), [.tensor t] =>
      if t.shape == shape && t.elt == elt then .ok [.tensor t] else .stuck "tensor.cast changes shape"
    | _, _ => .stuck "malformed tensor.cast"
  | "tensor.collapse_shape" | "tensor.expand_shape" =>
    match resultTensorTy op, reassociation? op, args with
    | some (shape, elt), some groups, [.tensor t] =>
      -- collapse: groups index the source; expand: groups index the result.
      let ok := if op.name == "tensor.collapse_shape" then reassocValid groups t.shape shape
                else reassocValid groups shape t.shape
      if ok && elt == t.elt then
        .ok [.tensor { shape, elt, get := fun i => t.get (delinearize t.shape (linearIndex shape i)) }]
      else .stuck s!"{op.name}: reassociation inconsistent with shapes"
    | _, _, _ => .stuck s!"malformed {op.name} (dynamic or missing reassociation)"
  | "tensor.extract_slice" =>
    match resultTensorTy op, staticArray? op "static_offsets", staticArray? op "static_sizes",
        staticArray? op "static_strides", args with
    | some (rshape, elt), some offs, some sizes, some strides, [.tensor t] =>
      let n := t.shape.length
      if offs.length ≠ n ∨ sizes.length ≠ n ∨ strides.length ≠ n then .stuck "slice rank" else
      if !(reducedCompatible sizes rshape) then .stuck "slice result shape" else
      if (List.range n).any fun d =>
          sizes.getD d 0 ≠ 0 && offs.getD d 0 + (sizes.getD d 0 - 1) * strides.getD d 0 ≥ t.shape.getD d 0
      then .stuck "extract_slice outside the source" else
      .ok [.tensor { shape := rshape, elt, get := fun r =>
        (match expandReduced sizes rshape r with
         | some f => t.get ((List.range n).map fun d => offs.getD d 0 + f.getD d 0 * strides.getD d 0)
         | none => .poison) }]
    | _, _, _, _, _ => .stuck "extract_slice with dynamic or missing offsets/sizes/strides"
  | "tensor.insert_slice" =>
    match staticArray? op "static_offsets", staticArray? op "static_sizes",
        staticArray? op "static_strides", args with
    | some offs, some sizes, some strides, [.tensor src, .tensor dst] =>
      let n := dst.shape.length
      if offs.length ≠ n ∨ sizes.length ≠ n ∨ strides.length ≠ n then .stuck "slice rank" else
      if !(reducedCompatible sizes src.shape) then .stuck "insert_slice source shape" else
      if (List.range n).any fun d =>
          sizes.getD d 0 ≠ 0 && offs.getD d 0 + (sizes.getD d 0 - 1) * strides.getD d 0 ≥ dst.shape.getD d 0
      then .stuck "insert_slice outside the destination" else
      -- Source position of destination index `i`, if `i` lies in the slice.
      let srcPos (i : Idx) : Option Idx := do
        let f ← (List.range n).mapM fun d =>
          let x := i.getD d 0; let o := offs.getD d 0; let s := strides.getD d 0
          if x ≥ o ∧ s > 0 ∧ (x - o) % s = 0 ∧ (x - o) / s < sizes.getD d 0 then some ((x - o) / s)
          else none
        -- drop the coordinates of size-1 dims that the source does not have
        placeNonUnit src.shape ((f.zip sizes).filterMap fun (c, s) => if s = 1 then none else some c)
      .ok [.tensor { dst with get := fun i => (match srcPos i with
                                                | some j => src.get j
                                                | none => dst.get i) }]
    | _, _, _, _ => .stuck "insert_slice with dynamic or missing offsets/sizes/strides"
  | "tensor.concat" =>
    match resultTensorTy op, attrInt? op "dim", args.mapM (fun | .tensor t => some t | _ => none) with
    | some (shape, elt), some d, some (t :: ts) =>
      let d := d.toNat
      let parts := t :: ts
      if parts.any (fun p => p.shape.length ≠ shape.length) then .stuck "concat rank" else
      if parts.foldl (fun acc p => acc + p.shape.getD d 0) 0 ≠ shape.getD d 0 then .stuck "concat extent" else
      .ok [.tensor { shape, elt, get := fun i => concatPick d i parts 0 }]
    | _, _, _ => .stuck "malformed tensor.concat"
  | "tensor.extract" =>
    match args with
    | .tensor t :: idxs =>
      match idxs.mapM (fun | .scalar v => indexVal? v | _ => none) with
      | some is =>
        if is.length = t.shape.length ∧ (is.zip t.shape).all (fun (x, s) => 0 ≤ x ∧ x < s) then
          .ok [.scalar (t.get (is.map Int.toNat))]
        else .ub
      | none => .ub  -- a poison index may be out of bounds
    | _ => .stuck "malformed tensor.extract"
  | "tensor.insert" =>
    match args with
    | .scalar x :: .tensor t :: idxs =>
      match idxs.mapM (fun | .scalar v => indexVal? v | _ => none) with
      | some is =>
        if is.length = t.shape.length ∧ (is.zip t.shape).all (fun (x, s) => 0 ≤ x ∧ x < s) then
          .ok [.tensor (t.set (is.map Int.toNat) x)]
        else .ub
      | none => .ub
    | _ => .stuck "malformed tensor.insert"
  | "tensor.from_elements" =>
    match resultTensorTy op, args.mapM (fun | .scalar v => some v | _ => none) with
    | some (shape, elt), some vs =>
      if vs.length = shapeSize shape then
        let arr := vs.toArray
        .ok [.tensor { shape, elt, get := fun i => arr[linearIndex shape i]?.getD .poison }]
      else .stuck "from_elements count"
    | _, _ => .stuck "malformed tensor.from_elements"
  | name =>
    let _ := ctx
    .stuck s!"unsupported op {name}"

end MerlinVerify.Sem.Ops.Tensor
