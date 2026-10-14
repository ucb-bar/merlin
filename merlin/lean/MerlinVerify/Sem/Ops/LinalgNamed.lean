/-
Op family: named linalg ops, DEFINED by their generic form (design §4.3): each op's indexing maps
and iterator kinds are transcribed below and passed, with the op's own region (named ops carry
their body region in generic MLIR form, including the casts OpDSL inserts), to `genericSem`.

Sources: llvm-project `a47bddcce`, `mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td`
(matmul, batch_matmul, transpose, broadcast, reduce, map) and
`mlir/python/mlir/dialects/linalg/opdsl/ops/core_named_ops.py` (quantized_matmul, copy).

| op | maps (dims) | iterators |
|---|---|---|
| matmul | A (d0,d2), B (d2,d1), C (d0,d1); or the op's `indexing_maps` | p p r |
| batch_matmul | A (d0,d1,d3), B (d0,d3,d2), C (d0,d1,d2); or `indexing_maps` | p p p r |
| quantized_matmul | A (d0,d2), B (d2,d1), AZp (), BZp (), C (d0,d1) | p p r |
| copy | I id, O id | p^n |
| transpose | in: dim p ↦ d_i where permutation[i] = p; init id | p^n |
| broadcast | in: init dims not in `dimensions`; init id | p^n |
| reduce | in id (input rank); init: input dims not in `dimensions` | r on `dimensions` |
| map | ins id, init id; body takes the inputs only | p^n |

The cross-check against `mlir-opt --linalg-generalize-named-ops` (design §4.3, §9.4) is part of the
fidelity suite; every instance in the corpus must generalize to exactly these maps.
-/
import MerlinVerify.Sem.Ops.Linalg

namespace MerlinVerify.Sem.Ops.LinalgNamed

open MerlinVerify
open MerlinVerify.Sem.Ops.Linalg (genericSem IterKind splitInsOuts identityMap shapeOf)

def mapOf (n : Nat) (dims : List Nat) : AffineMap :=
  { numDims := n, numSyms := 0, results := dims.map AffineExpr.dim }

def parallel (n : Nat) : List IterKind := List.replicate n .parallel

/-- Explicit `indexing_maps` override (matmul/batch_matmul variants), else the default maps. -/
def mapsOr (op : Op) (dflt : List AffineMap) : List AffineMap :=
  (attrAffineMaps? op "indexing_maps").getD dflt

def natArray? (op : Op) (key : String) : Option (List Nat) :=
  (attrDenseArray? op key).bind fun xs => xs.mapM fun x => if x ≥ 0 then some x.toNat else none

/-- Run `genericSem` and check the result count matches the op (tensor semantics). -/
def run (ctx : Ctx) (op : Op) (maps : List AffineMap) (iters : List IterKind)
    (ins outs : List RVal) (body : Ctx → RegionSem) : Outcome (List RVal) :=
  (genericSem ctx maps iters ins outs body).bind fun rs =>
    if rs.length = op.results.length then .ok rs else .stuck s!"{op.name}: result count"

def sem : OpSem := fun ctx _ op args regions =>
  match regions with
  | [body] =>
    match op.name with
    | "linalg.matmul" =>
      match splitInsOuts op args with
      | some ([a, b], [c]) =>
        run ctx op (mapsOr op [mapOf 3 [0, 2], mapOf 3 [2, 1], mapOf 3 [0, 1]])
          [.parallel, .parallel, .reduction] [a, b] [c] body
      | _ => .stuck "malformed linalg.matmul"
    | "linalg.batch_matmul" =>
      match splitInsOuts op args with
      | some ([a, b], [c]) =>
        run ctx op (mapsOr op [mapOf 4 [0, 1, 3], mapOf 4 [0, 3, 2], mapOf 4 [0, 1, 2]])
          [.parallel, .parallel, .parallel, .reduction] [a, b] [c] body
      | _ => .stuck "malformed linalg.batch_matmul"
    | "linalg.quantized_matmul" =>
      match splitInsOuts op args with
      | some ([a, b, azp, bzp], [c]) =>
        run ctx op [mapOf 3 [0, 2], mapOf 3 [2, 1], mapOf 3 [], mapOf 3 [], mapOf 3 [0, 1]]
          [.parallel, .parallel, .reduction] [a, b, azp, bzp] [c] body
      | _ => .stuck "malformed linalg.quantized_matmul"
    | "linalg.copy" =>
      match splitInsOuts op args with
      | some ([i], [o]) =>
        let n := (shapeOf o).length
        run ctx op [identityMap n, identityMap n] (parallel n) [i] [o] body
      | _ => .stuck "malformed linalg.copy"
    | "linalg.transpose" =>
      match args, natArray? op "permutation" with
      | [i, init], some perm =>
        let n := (shapeOf init).length
        if perm.length ≠ n ∨ (List.range n).any (fun p => !(perm.contains p)) then
          .stuck "linalg.transpose: permutation" else
        -- init[j] = input[k] with k[perm[i]] = j[i]: input dim p reads loop i where perm[i] = p.
        let inMap := mapOf n ((List.range n).map fun p => (perm.findIdx? (· = p)).getD 0)
        run ctx op [inMap, identityMap n] (parallel n) [i] [init] body
      | _, _ => .stuck "malformed linalg.transpose"
    | "linalg.broadcast" =>
      match args, natArray? op "dimensions" with
      | [i, init], some dims =>
        let n := (shapeOf init).length
        let kept := (List.range n).filter (fun d => !(dims.contains d))
        if kept.length ≠ (shapeOf i).length then .stuck "linalg.broadcast: rank" else
        run ctx op [mapOf n kept, identityMap n] (parallel n) [i] [init] body
      | _, _ => .stuck "malformed linalg.broadcast"
    | "linalg.reduce" =>
      match natArray? op "dimensions" with
      | some dims =>
        let split := (splitInsOuts op args).getD (args.take (args.length / 2), args.drop (args.length / 2))
        match split with
        | (ins@(i :: _), outs) =>
          let n := (shapeOf i).length
          let kept := (List.range n).filter (fun d => !(dims.contains d))
          let iters := (List.range n).map fun d => if dims.contains d then IterKind.reduction else .parallel
          run ctx op (ins.map (fun _ => identityMap n) ++ outs.map (fun _ => mapOf n kept))
            iters ins outs body
        | _ => .stuck "malformed linalg.reduce"
      | none => .stuck "linalg.reduce: dimensions"
    | "linalg.map" =>
      -- `linalg.map ins(...) outs(%init)`: the body takes the input elements only.
      match args.reverse with
      | init :: insRev =>
        let ins := insRev.reverse
        let n := (shapeOf init).length
        let body' : Ctx → RegionSem := fun c vals => body c (vals.take ins.length)
        run ctx op ((ins ++ [init]).map fun _ => identityMap n) (parallel n) ins [init] body'
      | [] => .stuck "malformed linalg.map"
    | name => .stuck s!"unsupported named linalg op {name}"
  | _ => .stuck s!"{op.name}: expected exactly one body region (generic form)"

end MerlinVerify.Sem.Ops.LinalgNamed
