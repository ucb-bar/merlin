/-
Source-derived reference transformer for the f32-accumulation contract (review candidate; see
`Bf16F32AccContract`). Every in-scope source contraction must be recognized in full, or the whole
module is refused: an unrecognized 16-bit contraction never silently keeps 16-bit accumulation.

In scope (a *candidate*): a `linalg.matmul`, or a two-input `linalg.generic` with a reduction
iterator whose body contains `arith.mulf` and `arith.addf`, producing a 16-bit float tensor.
A candidate is *eligible* when, from the source alone:
* operands and result share one 16-bit format (`half`) and have static shapes;
* its body is exactly `yield(acc + a * b)` in that format (either operand order of the commutative
  pair, fast-math absent or `none`), and its attributes are only the structured-op attributes
  (semantics-free annotations aside);
* its maps are projections onto loop dimensions, the output map names no reduction dimension, and
  at least one iterator is a reduction (`linalg.matmul` uses its own maps, or the OpDSL default);
* its destination is a `tensor.splat` or `linalg.fill` of an `arith.constant` `+0.0` of that format.
The source destination is left in place; downstream dead-code elimination may remove it.

The reference replaces each eligible contraction, in place, by six operations:
`tensor.empty : f32`, `arith.constant 0.0 : f32`, `linalg.fill`, the accumulating
`linalg.generic` (the source maps and iterators, `Bf16F32AccBody.accRegion`), `tensor.empty` of
the result format, and the identity-indexed `linalg.generic` with `truncRegion`, whose result is
the source result. A per-site hint, recovered from the candidate output and never trusted, may
choose only (a) the multiply and add flags among `Flag` and (b) semantics-free annotations
(`Spec.Annotations.semanticsFree`); every other attribute must be exactly the canonical one.
-/
import MerlinVerify.Rules.Bf16F32AccTensor
import MerlinVerify.Check.PolicyModule

namespace MerlinVerify.Rules.Bf16F32AccReference

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg Bf16F32AccContract Bf16F32AccBody

/-! ### Hints: permitted license and semantics-free annotations only -/

def free (entry : String × Attr) : Bool := Spec.Annotations.semanticsFree entry.1

def semantic (attrs : List (String × Attr)) : List (String × Attr) := attrs.filter (fun e => !free e)

structure SiteHint where
  license : License
  /-- Complete attribute lists of the six emitted operations, in emission order. -/
  attrs : List (List (String × Attr))

/-- A hinted attribute list is admitted only when it differs from `canonical` by annotations. -/
def annotate (canonical hinted : List (String × Attr)) : Option (List (String × Attr)) :=
  if decide (semantic hinted = canonical) then some hinted else none

theorem annotate_semantic {canonical hinted attrs : List (String × Attr)}
    (h : annotate canonical hinted = some attrs) : attrs = hinted ∧ semantic attrs = canonical := by
  unfold annotate at h
  split at h
  · next hc => cases h; exact ⟨rfl, of_decide_eq_true hc⟩
  · cases h

/-! ### Source recognition -/

def owner? (seen : List Op) (id : Nat) : Option Op :=
  seen.find? fun op => op.results.any (fun result => result.id == id)

def type? (args : List Value) (seen : List Op) (id : Nat) : Option Ty :=
  ((seen.findSome? fun op => op.results.find? (fun value => value.id == id)).orElse
    (fun _ => args.find? (fun value => value.id == id))).map (·.ty)

def halfTensor? (ty : Ty) : Option (List Nat × FKind) :=
  match ty.staticTensor? with
  | some (shape, .float k) => if half k then some (shape, k) else none
  | _ => none

def iterAttr : IterKind → Attr
  | .parallel => .enum "linalg" "iterator_type" "parallel"
  | .reduction => .enum "linalg" "iterator_type" "reduction"

def tensorTy (shape : List Nat) (k : FKind) : Ty := .tensor (shape.map some) (.float k) none

def matmulMaps : List AffineMap :=
  [{ numDims := 3, numSyms := 0, results := [.dim 0, .dim 2] },
   { numDims := 3, numSyms := 0, results := [.dim 2, .dim 1] },
   { numDims := 3, numSyms := 0, results := [.dim 0, .dim 1] }]

def dims? (map : AffineMap) : Option (List Nat) :=
  map.results.mapM fun | .dim d => some d | _ => none

/-- Projected maps over `iters`, an output map without reduction dimensions, and a reduction. -/
def contractionMaps (maps : List AffineMap) (iters : List IterKind) : Bool := Id.run do
  let [left, right, out] := maps | return false
  let some outDims := dims? out | return false
  if (dims? left).isNone || (dims? right).isNone then return false
  return maps.all (fun m => m.numDims == iters.length && m.numSyms == 0) &&
    iters.contains .reduction &&
    outDims.all (fun d => iters[d]? == some .parallel) && outDims.Nodup

/-- The source scalar body: `yield(acc + a * b)` in format `k`. -/
def sourceRegion (k : FKind) (mulSwap addSwap : Bool) (flags : List (String × Attr)) : Region :=
  .mk [.mk [{ id := 0, ty := .float k }, { id := 1, ty := .float k }, { id := 2, ty := .float k }]
    [.mk "arith.mulf" [{ id := 3, ty := .float k }] (if mulSwap then [1, 0] else [0, 1]) flags [] [] [],
     .mk "arith.addf" [{ id := 4, ty := .float k }] (if addSwap then [3, 2] else [2, 3]) flags [] [] [],
     .mk "linalg.yield" [] [4] [] [] [] []]]

def sourceBody (k : FKind) (actual : Region) : Bool :=
  [false, true].any fun mulSwap => [false, true].any fun addSwap =>
    [[], fastmath .none].any fun flags => alphaRegion [] (sourceRegion k mulSwap addSwap flags) actual

def zeroLiteral (seen : List Op) (k : FKind) (id : Nat) : Bool :=
  match owner? seen id with
  | some (.mk "arith.constant" [result] [] attrs [] [] []) =>
    result.id == id && result.ty == .float k && semantic attrs == [("value", .float 0 (.float k))]
  | _ => false

/-- The source destination is a `+0.0` splat or fill in the source's own format. -/
def zeroDestination (seen : List Op) (k : FKind) (shape : List Nat) (id : Nat) : Bool :=
  match owner? seen id with
  | some (.mk "tensor.splat" [result] [literal] attrs [] [] []) =>
    result.ty == tensorTy shape k && semantic attrs == [] && zeroLiteral seen k literal
  | some (.mk "linalg.fill" [result] [literal, _] attrs [] [body] []) =>
    result.ty == tensorTy shape k &&
      semantic attrs == [("operandSegmentSizes", .denseArray (.int 32) [1, 1])] &&
      alphaRegion [] (W8A8Fill.fillRegion (.float k)) body && zeroLiteral seen k literal
  | _ => false

/-- Ops whose accumulation precision the contract governs. -/
def candidate (op : Op) : Bool :=
  let halfResult := match op.results with
    | [result] => (halfTensor? result.ty).isSome
    | _ => false
  halfResult && (op.name == "linalg.matmul" ||
    (op.name == "linalg.generic" &&
      (match attrDenseArray? op "operandSegmentSizes" with | some (2 :: _) => true | _ => false) &&
      ((iterKinds? op).map (·.contains .reduction)).getD false &&
      op.regions.any (fun region => region.blocks.any fun block =>
        block.ops.any (·.name == "arith.mulf") && block.ops.any (·.name == "arith.addf"))))

structure Site where
  left : Nat
  right : Nat
  result : Value
  k : FKind
  out : List Nat
  maps : List AffineMap
  iters : List IterKind

/-- Recognize an eligible site, or name the first eligibility condition it fails. -/
def siteE (args : List Value) (seen : List Op) (op : Op) : Except String Site := do
  let [left, right, destination] := op.operands | throw "not exactly two inputs and one destination"
  let [result] := op.results | throw "not exactly one result"
  let [body] := op.regions | throw "not exactly one body region"
  if !op.props.isEmpty || !op.successors.isEmpty then throw "unexpected properties or successors"
  let some (out, k) := halfTensor? result.ty | throw "result is not a static 16-bit float tensor"
  let some leftTy := type? args seen left | throw "left operand type unknown"
  let some rightTy := type? args seen right | throw "right operand type unknown"
  let some (_, kl) := halfTensor? leftTy | throw "left operand is not a static 16-bit float tensor"
  let some (_, kr) := halfTensor? rightTy | throw "right operand is not a static 16-bit float tensor"
  if kl != k || kr != k then throw "operand and result formats differ"
  if type? args seen destination != some result.ty then throw "destination type differs from the result"
  if splitInsOuts op op.operands != some ([left, right], [destination]) then
    throw "operand segments are not two inputs and one output"
  let (maps, iters, expected) ← match op.name with
    | "linalg.generic" =>
      match attrAffineMaps? op "indexing_maps", iterKinds? op with
      | some maps, some iters => pure (maps, iters, ["indexing_maps", "iterator_types", "operandSegmentSizes"])
      | _, _ => throw "indexing maps or iterator types do not decode"
    | "linalg.matmul" =>
      pure ((attrAffineMaps? op "indexing_maps").getD matmulMaps, [.parallel, .parallel, .reduction],
        ["indexing_maps", "operandSegmentSizes"])
    | _ => throw "not a linalg.generic or linalg.matmul"
  if !((semantic op.attrs).map (·.1)).all expected.contains then throw "unexpected semantic attribute"
  if !contractionMaps maps iters then throw "maps are not a projected contraction"
  if !sourceBody k body then throw "body is not yield(acc + a * b) in the source format"
  if !zeroDestination seen k out destination then
    throw "destination is not a +0.0 splat or fill in the source format"
  return ⟨left, right, result, k, out, maps, iters⟩

def site? (args : List Value) (seen : List Op) (op : Op) : Option Site := (siteE args seen op).toOption

/-! ### Emission -/

def generic (ins : Nat) (maps : List AffineMap) (iters : List IterKind) : List (String × Attr) :=
  [("indexing_maps", .array (maps.map .affineMap)), ("iterator_types", .array (iters.map iterAttr)),
   ("operandSegmentSizes", .denseArray (.int 32) [ins, 1])]

/-- The six canonical attribute lists, in emission order. -/
def canonical (site : Site) : List (List (String × Attr)) :=
  let rank := site.out.length
  [[], [("value", .float 0 (.float .f32))], [("operandSegmentSizes", .denseArray (.int 32) [1, 1])],
   generic 2 site.maps site.iters, [],
   generic 1 [identityMap rank, identityMap rank] (List.replicate rank .parallel)]

/-- Rename every value of a template region to `base + id`; the alpha check is done on use. -/
def shiftValue (base : Nat) (value : Value) : Value := { value with id := base + value.id }

def shiftRegion (base : Nat) : Region → Region
  | .mk blocks => .mk (blocks.map fun
    | .mk args ops => .mk (args.map (shiftValue base)) (ops.map fun
      | .mk name results operands attrs props _ successors =>
        .mk name (results.map (shiftValue base)) (operands.map (base + ·)) attrs props [] successors))

/-- Instantiate a template region at fresh ids, refusing unless it is alpha-equivalent. -/
def instantiate (template : Region) (base : Nat) : Option Region :=
  let region := shiftRegion base template
  if alphaRegion [] template region then some region else none

def emit (site : Site) (hint : SiteHint) (next : Nat) : Option (List Op × Nat) := do
  let [e, z, f, acc, e2, tr] := hint.attrs | none
  let [ce, cz, cf, cacc, ce2, ctr] := canonical site | none
  let e ← annotate ce e
  let z ← annotate cz z
  let f ← annotate cf f
  let acc ← annotate cacc acc
  let e2 ← annotate ce2 e2
  let tr ← annotate ctr tr
  let fillBody ← instantiate (W8A8Fill.fillRegion (.float .f32)) (next + 5)
  let accBody ← instantiate (accRegion site.k site.k hint.license) (next + 10)
  let truncBody ← instantiate (truncRegion site.k) (next + 20)
  let ops : List Op :=
    [.mk "tensor.empty" [⟨next, tensorTy site.out .f32⟩] [] e [] [] [],
     .mk "arith.constant" [⟨next + 1, .float .f32⟩] [] z [] [] [],
     .mk "linalg.fill" [⟨next + 2, tensorTy site.out .f32⟩] [next + 1, next] f [] [fillBody] [],
     .mk "linalg.generic" [⟨next + 3, tensorTy site.out .f32⟩] [site.left, site.right, next + 2] acc []
       [accBody] [],
     .mk "tensor.empty" [⟨next + 4, tensorTy site.out site.k⟩] [] e2 [] [] [],
     .mk "linalg.generic" [site.result] [next + 3, next + 4] tr [] [truncBody] []]
  return (ops, next + 30)

/-! ### Fresh identifiers -/

mutual
  def boundOps : List Op → Nat
    | [] => 0
    | .mk _ results _ _ _ regions _ :: rest =>
      max (results.foldl (fun n value => max n (value.id + 1)) 0)
        (max (boundRegions regions) (boundOps rest))
  def boundRegions : List Region → Nat
    | [] => 0
    | .mk blocks :: rest => max (boundBlocks blocks) (boundRegions rest)
  def boundBlocks : List Block → Nat
    | [] => 0
    | .mk args ops :: rest => max (args.foldl (fun n value => max n (value.id + 1)) 0)
        (max (boundOps ops) (boundBlocks rest))
end

/-! ### The transformer -/

structure State where
  next : Nat
  hints : List SiteHint
  sites : Nat := 0

def body (args : List Value) (ops : List Op) (state : State) : Option (List Op × State) := do
  let mut seen : List Op := []
  let mut emitted : List Op := []
  let mut state := state
  for op in ops do
    if candidate op then
      let site ← site? args seen op
      let hint :: rest := state.hints | none
      let (replacement, next) ← emit site hint state.next
      emitted := replacement.reverse ++ emitted
      state := { next, hints := rest, sites := state.sites + 1 }
    else emitted := op :: emitted
    seen := op :: seen
  return (emitted.reverse, state)

def function (op : Op) (state : State) : Option (Op × State) := do
  let [.mk [.mk args ops]] := op.regions | none
  let (ops', state') ← body args ops state
  return (.mk op.name op.results op.operands op.attrs op.props [.mk [.mk args ops']] op.successors,
    state')

/-- The reference program, or `none` when any candidate is ineligible or the hints do not fit. -/
def expand (hints : List SiteHint) (source : Module) : Option Module := do
  let mut state : State := ⟨boundOps source.ops, hints, 0⟩
  let mut reversed : List Op := []
  for op in source.ops do
    if op.name == "func.func" && !(op.asFunc?.map (·.isDecl)).getD true then
      let (replacement, state') ← function op state
      state := state'
      reversed := replacement :: reversed
    else reversed := op :: reversed
  if !state.hints.isEmpty then none
  return { source with ops := reversed.reverse }

def policy (hints : List SiteHint) : PolicyModule.Policy := ⟨"bf16-f32acc-contract", expand hints⟩

/-! ### Untrusted hint recovery from the candidate output -/

def flagOf? (op : Op) : Option Flag :=
  match op.attrs with
  | [("fastmath", .enum "arith" "fastmath" text)] => Flag.ofText? text
  | _ => none

def licenseOf? (acc : Op) : Option License := do
  let [.mk [.mk _ ops]] := acc.regions | none
  let [_, _, mul, add, _] := ops | none
  return ⟨← flagOf? mul, ← flagOf? add⟩

def hintsBody (sources targets : List Op) : Option (List SiteHint) :=
  match sources with
  | [] => some []
  | source :: rest =>
    if candidate source then do
      let [e, z, f, acc, e2, tr] := targets.take 6 | none
      let license ← licenseOf? acc
      let tail ← hintsBody rest (targets.drop 6)
      return ⟨license, [e.attrs, z.attrs, f.attrs, acc.attrs, e2.attrs, tr.attrs]⟩ :: tail
    else hintsBody rest (targets.drop 1)

/-- Pair functions by position; any mismatch is left for the checker to refuse. -/
def hints? (before after : Module) : Option (List SiteHint) :=
  (before.ops.zip after.ops).foldr (fun (source, target) acc => do
    let tail ← acc
    if source.name == "func.func" && !(source.asFunc?.map (·.isDecl)).getD true then
      let [.mk [.mk _ sourceOps]] := source.regions | none
      let [.mk [.mk _ targetOps]] := target.regions | none
      return (← hintsBody sourceOps targetOps) ++ tail
    else return tail) (some [])

end MerlinVerify.Rules.Bf16F32AccReference
