/- Full source-derived repair of invalid named matmuls. Dedicated standalone checker support.
Only reference correspondence is claimed: the source is invalid MLIR. -/
import MerlinVerify.Rules.OverrankBody
import MerlinVerify.Check.ExactSyntax
namespace MerlinVerify.Check.OverrankReference
open MerlinVerify Rules.OverrankIntent
abbrev Scope := List (Nat × Ty)
def bind (scope : Scope) (vs : List Value) : Scope := vs.map (fun v => (v.id,v.ty)) ++ scope

def candidate (scope : Scope) (op : Op) : Bool :=
  op.name == "linalg.matmul" && (match op.results, op.operands with
    | [r], [_, b, _] => match r.ty, scope.lookup b with
      | .tensor shape _ _, some (.tensor rhs _ _) => decide (shape.length > 2) && (rhs.length == 2)
      | _, _ => false
    | _, _ => false)

structure Site where
  path : List Nat
  kind : FKind
  shape : Shape
  lhs : Nat
  rhs : Nat
  result : Value
  attrs : List (String × Attr)
  layout : layoutCheck kind shape = true

def namedMaps : List AffineMap :=
  [⟨3,0,[.dim 0,.dim 2]⟩,⟨3,0,[.dim 2,.dim 1]⟩,⟨3,0,[.dim 0,.dim 1]⟩]

def site? (scope : Scope) (path : List Nat) (op : Op) : Option Site := do
  let .mk "linalg.matmul" [result] [a,b,old] attrs [] [body] [] := op | none
  let .tensor _out (.float kind) none := result.ty | none
  if !alphaRegion [] (Rules.OverrankBody.namedRegion kind) body then none else do
  if op.attr? "indexing_maps" != some (.array (namedMaps.map Attr.affineMap)) ||
      op.attr? "operandSegmentSizes" != some (.denseArray (.int 32) [2,1]) then none else do
  let some (left, le) := (← scope.lookup a).staticTensor? | none
  let some (right, re) := (← scope.lookup b).staticTensor? | none
  let some (output, oe) := result.ty.staticTensor? | none
  if left.length ≤ 2 || right.length != 2 then none else do
  let batch := left.take (left.length-1)
  let shape : Shape := ⟨batch, left.getLast!, right[1]!⟩
  if left != shape.left || right != shape.right || output != shape.out ||
      le != .float kind || re != .float kind || oe != .float kind ||
      scope.lookup old != some result.ty then none else do
  if !attrs.all (fun p => p.1.startsWith "prov." || p.1 == "indexing_maps" || p.1 == "operandSegmentSizes") then none else do
  if h : layoutCheck kind shape = true then
    return ⟨path,kind,shape,a,b,result,attrs,h⟩
  else none

def tensorTy (shape : List Nat) (kind : FKind) : Ty := .tensor (shape.map some) (.float kind) none

def provenance (attrs : List (String × Attr)) : List (String × Attr) :=
  let kept := attrs.filter (fun p => p.1.startsWith "prov." && p.1 != "prov.transforms")
  let old := match attrs.lookup "prov.transforms" with | some (.str s) => s | _ => ""
  (kept ++ [("prov.transforms", Attr.str (if old.isEmpty then "collapse_overrank_matmul"
    else old ++ ",collapse_overrank_matmul"))]).mergeSort (fun a b => a.1 ≤ b.1)

def generic (site : Site) (dest : Nat) : Op :=
  .mk "linalg.generic" [site.result] [site.lhs,site.rhs,dest]
    ([("indexing_maps",.array ((maps site.shape).map Attr.affineMap)),
      ("iterator_types",.array ((iters site.shape).map (fun i => .enum "linalg" "iterator_type"
        (match i with | .parallel => "parallel" | .reduction => "reduction")))),
      ("operandSegmentSizes",.denseArray (.int 32) [2,1])] ++ provenance site.attrs)
    [] [Rules.OverrankBody.region site.kind] []

/-- Fresh local IDs are chosen outside all original IDs; complete alpha comparison rechecks SSA.
The fill and multiply/add regions have locally scoped binders. -/
def emission (site : Site) (fresh : Nat) : List Op :=
  let ty := tensorTy site.shape.out site.kind
  [.mk "tensor.empty" [⟨fresh,ty⟩] [] [] [] [] [],
   .mk "arith.constant" [⟨fresh+1,.float site.kind⟩] [] [("value",.float 0 (.float site.kind))] [] [] [],
   .mk "linalg.fill" [⟨fresh+2,ty⟩] [fresh+1,fresh]
     [("operandSegmentSizes",.denseArray (.int 32) [1,1])] []
     [.mk [.mk [⟨0,.float site.kind⟩,⟨1,.float site.kind⟩]
       [.mk "linalg.yield" [] [0] [] [] [] []]]] [],
   generic site (fresh+2)]

structure Body where
  ops : List Op
  sites : List Site
  valid : Bool
  next : Nat
structure Regions where
  regions : List Region
  sites : List Site
  valid : Bool
  next : Nat
structure Blocks where
  blocks : List Block
  sites : List Site
  valid : Bool
  next : Nat
mutual
  def operations (scope : Scope) (path : List Nat) (index fresh : Nat) : List Op → Body
    | [] => ⟨[],[],true,fresh⟩
    | op@(.mk name results operands attrs props rs succ) :: rest =>
      let here := path ++ [index]
      if candidate scope op then
        match site? scope here op with
        | none => ⟨[],[],false,fresh⟩
        | some site =>
          let tail := operations (bind scope results) path (index+1) (fresh+3) rest
          ⟨emission site fresh ++ tail.ops, site::tail.sites, tail.valid, tail.next⟩
      else
        let inner := regions scope here 0 fresh rs
        let tail := operations (bind scope results) path (index+1) inner.next rest
        ⟨.mk name results operands attrs props inner.regions succ :: tail.ops,
          inner.sites ++ tail.sites, inner.valid && tail.valid, tail.next⟩
    termination_by ops => sizeOf ops
    decreasing_by all_goals simp_wf; all_goals omega
  def regions (scope : Scope) (path : List Nat) (index fresh : Nat) : List Region → Regions
    | [] => ⟨[],[],true,fresh⟩
    | .mk bs :: rest =>
      let here := blocks scope (path ++ [index]) 0 fresh bs
      let tail := regions scope path (index+1) here.next rest
      ⟨.mk here.blocks::tail.regions, here.sites++tail.sites, here.valid&&tail.valid, tail.next⟩
    termination_by rs => sizeOf rs
    decreasing_by all_goals simp_wf; all_goals omega
  def blocks (scope : Scope) (path : List Nat) (index fresh : Nat) : List Block → Blocks
    | [] => ⟨[],[],true,fresh⟩
    | .mk args ops :: rest =>
      let here := operations (bind scope args) (path ++ [index]) 0 fresh ops
      let tail := blocks scope path (index+1) here.next rest
      ⟨.mk args here.ops::tail.blocks, here.sites++tail.sites, here.valid&&tail.valid, tail.next⟩
    termination_by bs => sizeOf bs
    decreasing_by all_goals simp_wf; all_goals omega
end
mutual
  def idsOps : List Op → List Nat
    | [] => []
    | .mk _ rs os _ _ regions _ :: rest => rs.map (·.id) ++ os ++ idsRegions regions ++ idsOps rest
  def idsRegions : List Region → List Nat
    | [] => []
    | .mk bs :: rest => idsBlocks bs ++ idsRegions rest
  def idsBlocks : List Block → List Nat
    | [] => []
    | .mk args ops :: rest => args.map (·.id) ++ idsOps ops ++ idsBlocks rest
end

def plan (before : Module) : Body := operations [] [] 0 ((idsOps before.ops).foldl max 0 + 1) before.ops

def reference (before : Module) : Module := {before with ops := (plan before).ops}
end MerlinVerify.Check.OverrankReference
