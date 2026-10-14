/-
Op family: `linalg.generic`, `linalg.fill`, `linalg.index` (design §4.3).

`genericSem` is the loop-nest meaning MLIR gives structured ops
(`mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td`, `LinalgInterfaces.td`):
* loop ranges are read off operand shapes: loop `d` takes the extent of the first operand dimension
  whose indexing-map result is exactly `d` (MLIR's `getStaticLoopRanges`); otherwise `stuck`;
* the iteration space is visited in lexicographic order; at each point the body receives the input
  elements at their mapped indices, then the current output elements, and yields new output
  elements, written back at the output maps;
* reduction order is therefore the lexicographic loop order `convert-linalg-to-loops` materializes.
  For integers the order is provably irrelevant; for floats it is the definition (fidelity tested);
* an access outside an operand's shape is `stuck` (not modelled), never silently clamped.
Tensor semantics only: memref operands are `stuck`.
-/
import MerlinVerify.Sem.Support

namespace MerlinVerify.Sem.Ops.Linalg

open MerlinVerify

/-- Iterator kinds of a structured op (`#linalg.iterator_type<...>`). -/
inductive IterKind where
  | parallel | reduction
  deriving DecidableEq, Repr

/-! ### Affine maps -/

/-- MLIR affine semantics: `floordiv`/`mod` round toward −∞ (mod has the divisor's sign),
`ceildiv` toward +∞; division by zero is undefined (`none`). -/
def evalAffine (dims : List Nat) : AffineExpr → Option Int
  | .dim p => dims[p]?.map Int.ofNat
  | .sym _ => none
  | .const c => some c
  | .add a b => do pure ((← evalAffine dims a) + (← evalAffine dims b))
  | .mul a b => do pure ((← evalAffine dims a) * (← evalAffine dims b))
  | .mod a b => do
    let y ← evalAffine dims b
    if y = 0 then none else pure (Int.fmod (← evalAffine dims a) y)
  | .floordiv a b => do
    let y ← evalAffine dims b
    if y = 0 then none else pure (Int.fdiv (← evalAffine dims a) y)
  | .ceildiv a b => do
    let y ← evalAffine dims b
    if y = 0 then none else pure (-(Int.fdiv (-(← evalAffine dims a)) y))

def evalMap (m : AffineMap) (pt : List Nat) : Option (List Int) :=
  m.results.mapM (evalAffine pt)

/-- In-bounds multi-index from affine results, or `none`. -/
def toIdx (shape : List Nat) (xs : List Int) : Option Idx :=
  if xs.length ≠ shape.length then none else
  (xs.zip shape).mapM fun (x, s) => if 0 ≤ x ∧ x < s then some x.toNat else none

def shapeOf : RVal → List Nat
  | .scalar _ => []
  | .tensor t => t.shape

/-- The extent of loop `d`: the first operand dimension whose map result is exactly `dim d`. -/
def loopRange (maps : List AffineMap) (shapes : List (List Nat)) (d : Nat) : Option Nat :=
  ((maps.zip shapes).findSome? fun (m, s) =>
    ((m.results.zip s).find? fun (e, _) => e = .dim d).map (·.2))

/-- Element of operand `v` read through map `m` at point `pt`. -/
def readAt (v : RVal) (m : AffineMap) (pt : List Nat) : Outcome Val :=
  match v with
  | .scalar x => if m.results.isEmpty then .ok x else .stuck "scalar operand with non-empty map"
  | .tensor t =>
    match evalMap m pt >>= toIdx t.shape with
    | some i => .ok (t.get i)
    | none => .stuck "structured-op access outside operand shape"

def writeAt (t : Tensor) (m : AffineMap) (pt : List Nat) (v : Val) : Outcome Tensor :=
  match evalMap m pt >>= toIdx t.shape with
  | some i => .ok (t.set i v)
  | none => .stuck "structured-op write outside output shape"

/-- G0-FROZEN SIGNATURE. Loop-nest semantics of a structured op with explicit indexing maps
(one per input then per output operand) and iterator kinds, applied to input and output (`outs`)
values, with `body` the meaning of its single-block region (block args: one element per input, then
per output; yields one element per output). Named ops (`LinalgNamed`) are defined by calling this
with their OpDSL maps and their own region (design §4.3). Iteration is in lexicographic order of the
iteration space; `body` runs with `ctx.loopIndex` set to the current point. -/
def genericSem (ctx : Ctx) (maps : List AffineMap) (iters : List IterKind)
    (ins outs : List RVal) (body : Ctx → RegionSem) : Outcome (List RVal) := do
  let n := iters.length
  let operands := ins ++ outs
  if maps.length ≠ operands.length then .stuck "indexing map count ≠ operand count" else
  if maps.any (fun m => m.numDims ≠ n || m.numSyms ≠ 0) then .stuck "indexing map dims/symbols" else
  let shapes := operands.map shapeOf
  if (maps.zip shapes).any (fun (m, s) => m.results.length ≠ s.length) then
    .stuck "indexing map rank ≠ operand rank" else
  let outTs ← outs.mapM fun
    | .tensor t => .ok t
    | .scalar _ => .stuck "structured-op output must be a tensor"
  let ranges ← (List.range n).mapM fun d =>
    Outcome.ofOption s!"loop {d} range not determined by any operand" (loopRange maps shapes d)
  let inMaps := maps.take ins.length
  let outMaps := maps.drop ins.length
  let final ← (allPoints ranges).foldlM (init := outTs) fun cur pt => do
    let inVals ← (ins.zip inMaps).mapM fun (v, m) => readAt v m pt
    let outVals ← (cur.zip outMaps).mapM fun (t, m) => readAt (.tensor t) m pt
    let ys ← body { ctx with loopIndex := pt } ((inVals ++ outVals).map RVal.scalar)
    if ys.length ≠ cur.length then .stuck "structured-op body yields wrong arity" else
    ((ys.zip cur).zip outMaps).mapM fun ((y, t), m) =>
      match y with
      | .scalar v => writeAt t m pt v
      | .tensor _ => .stuck "structured-op body yields a tensor"
  pure (final.map RVal.tensor)

/-! ### Attribute decoding -/

def iterKinds? (op : Op) : Option (List IterKind) :=
  match op.attr? "iterator_types" with
  | some (.array xs) => xs.mapM fun
    | .enum "linalg" "iterator_type" "parallel" => some .parallel
    | .enum "linalg" "iterator_type" "reduction" => some .reduction
    | _ => none
  | _ => none

/-- Split operands into (ins, outs) with `operandSegmentSizes`. -/
def splitInsOuts {α : Type} (op : Op) (args : List α) : Option (List α × List α) :=
  match attrDenseArray? op "operandSegmentSizes" with
  | some [ni, no] =>
    if ni ≥ 0 ∧ no ≥ 0 ∧ args.length = ni.toNat + no.toNat then
      some (args.take ni.toNat, args.drop ni.toNat)
    else none
  | _ => none

def identityMap (n : Nat) : AffineMap :=
  { numDims := n, numSyms := 0, results := (List.range n).map AffineExpr.dim }

def sem : OpSem := fun ctx _ op args regions =>
  match op.name with
  | "linalg.generic" =>
    match splitInsOuts op args, attrAffineMaps? op "indexing_maps", iterKinds? op, regions with
    | some (ins, outs), some maps, some iters, [body] => do
      let rs ← genericSem ctx maps iters ins outs body
      if rs.length = op.results.length then pure rs
      else .stuck "linalg.generic: result count ≠ output count (memref form unsupported)"
    | _, _, _, _ => .stuck "malformed linalg.generic"
  | "linalg.fill" =>
    -- OpDSL `fill(value, O)`: O[...] = cast(value). The cast is in the op's own region.
    match splitInsOuts op args, regions with
    | some ([v], [.tensor t]), [body] => do
      let r := t.shape.length
      let maps := [{ numDims := r, numSyms := 0, results := [] }, identityMap r]
      let rs ← genericSem ctx maps (List.replicate r .parallel) [v] [.tensor t] body
      if rs.length = op.results.length then pure rs else .stuck "linalg.fill: result count"
    | _, _ => .stuck "malformed linalg.fill"
  | "linalg.index" =>
    match attrInt? op "dim" with
    | some d =>
      match ctx.loopIndex[d.toNat]? with
      | some i => if d ≥ 0 then .ok [.scalar (.some (.bits ctx.indexBits (BitVec.ofNat _ i)))]
                  else .stuck "linalg.index: negative dim"
      | none => .stuck "linalg.index outside a structured-op body"
    | none => .stuck "malformed linalg.index"
  | name => .stuck s!"unsupported op {name}"

end MerlinVerify.Sem.Ops.Linalg
