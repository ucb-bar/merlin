/- Source-derived draft reference transformer for W8A8. This is a review candidate, not an
independently authored or approved Spec/Policy entry. No proposed target or provenance role is
consulted. Unsupported selected contractions refuse expansion instead of silently disappearing. -/
import MerlinVerify.Rules.W8A8Emission
import MerlinVerify.Rules.W8A8ProjectionReference
import MerlinVerify.Check.PolicyModule
import MerlinVerify.Check.ConstantFacts

namespace MerlinVerify.Rules.W8A8Reference

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg W8A8Emission

inductive Scope where
  | all | captured
  deriving DecidableEq

structure Config where
  scope : Scope := .captured

def owner? (seen : List Op) (id : Nat) : Option Op :=
  seen.find? fun op => op.results.any (fun result => result.id == id)

def type? (args : List Value) (seen : List Op) (id : Nat) : Option Ty :=
  ((seen.findSome? fun op => op.results.find? (fun value => value.id == id)).orElse
    (fun _ => args.find? (fun value => value.id == id))).map (·.ty)

def dequant (op : Op) : Bool :=
  op.name == "quant_ext.dequantize_per_tensor" || op.name == "quant_ext.dequantize_per_channel"

def traces (seen : List Op) : Nat → Nat → Bool
  | 0, _ => false
  | fuel + 1, id =>
    match owner? seen id with
    | none => false
    | some op =>
      op.name.startsWith "quant_ext.dequantize" ||
        (["linalg.transpose", "tensor.cast", "tensor.collapse_shape", "tensor.expand_shape",
          "tensor.extract_slice"].contains op.name &&
          (op.operands.head?.map (traces seen fuel)).getD false)

def selected (config : Config) (seen : List Op) (inputs : List Nat) : Bool :=
  config.scope == .all || inputs.any (traces seen 5)

def floatOp (name : String) (id : Nat) (operands : List Nat) : Op :=
  .mk name [⟨id, .float .f32⟩] operands [("fastmath", .enum "arith" "fastmath" "none")] [] [] []

def sourceBody (left right : Ty) (accumulatorFirst : Bool) : Region :=
  let leftCast := if left == .int 8 then
    [.mk "arith.sitofp" [⟨3, .float .f32⟩] [0] [] [] [] []] else []
  let rightCast := if right == .int 8 then
    [.mk "arith.sitofp" [⟨4, .float .f32⟩] [1] [] [] [] []] else []
  .mk [.mk [⟨0, left⟩, ⟨1, right⟩, ⟨2, .float .f32⟩]
    (leftCast ++ rightCast ++
      [floatOp "arith.mulf" 5 [if left == .int 8 then 3 else 0, if right == .int 8 then 4 else 1],
       floatOp "arith.addf" 6 (if accumulatorFirst then [2, 5] else [5, 2]),
       .mk "linalg.yield" [] [6] [] [] [] []])]

def maps : List AffineMap :=
  [{ numDims := 3, numSyms := 0, results := [.dim 0, .dim 2] },
   { numDims := 3, numSyms := 0, results := [.dim 2, .dim 1] },
   { numDims := 3, numSyms := 0, results := [.dim 0, .dim 1] }]

def iterators : List IterKind := [.parallel, .parallel, .reduction]

def candidate (op : Op) : Bool :=
  op.name == "linalg.matmul" || (op.name == "linalg.generic" &&
    op.regions.any (fun region => region.blocks.any fun block =>
      block.ops.any (fun body => body.name == "arith.mulf") &&
      block.ops.any (fun body => body.name == "arith.addf")))

structure View where
  m : Nat
  n : Nat
  k : Nat
  left : Nat
  right : Nat
  leftElt : Ty
  rightElt : Ty
  result : Value

def zeroScalar (seen : List Op) (id : Nat) : Bool := Id.run do
  let some op := owner? seen id | return false
  let some (.scalar (.some (.flt .f32 bits))) := ConstantFacts.value? 64 op | return false
  return bits == 0#32

def zeroDestination (seen : List Op) (id : Nat) : Bool := Id.run do
  let some op := owner? seen id | return false
  if op.name != "linalg.fill" || Sem.Ops.unmodelledAttr? op.blind != none then return false
  let [value, destination] := op.operands | return false
  let [body] := op.regions | return false
  return zeroScalar seen value &&
    decide (splitInsOuts op op.operands = some ([value], [destination])) &&
    alphaRegion [] (W8A8Fill.fillRegion (.float .f32)) body

def view? (args : List Value) (seen : List Op) (op : Op) : Option View := do
  let [left, right, destination] := op.operands | none
  let [result] := op.results | none
  let some ([m, k], leftElt) := (← type? args seen left).staticTensor? | none
  let some ([k', n], rightElt) := (← type? args seen right).staticTensor? | none
  if k != k' || k >= 131072 ||
      !(leftElt == .float .f32 || leftElt == .int 8) ||
      !(rightElt == .float .f32 || rightElt == .int 8) ||
      !(leftElt == .float .f32 || rightElt == .float .f32) then none
  if result.ty != tensor [m, n] (.float .f32) ||
      (← type? args seen destination) != result.ty || !zeroDestination seen destination then none
  if Sem.Ops.unmodelledAttr? op.blind != none || op.successors != [] then none
  if splitInsOuts op op.operands != some ([left, right], [destination]) then none
  let [body] := op.regions | none
  if !(alphaRegion [] (sourceBody leftElt rightElt false) body ||
      alphaRegion [] (sourceBody leftElt rightElt true) body) then none
  if op.name == "linalg.generic" then
    if attrAffineMaps? op "indexing_maps" != some maps || iterKinds? op != some iterators then none
  else if op.name == "linalg.matmul" then
    if let some actualMaps := attrAffineMaps? op "indexing_maps" then
      if actualMaps != maps then none
  else none
  return ⟨m, n, k, left, right, leftElt, rightElt, result⟩

def isZero (value : Val) : Bool :=
  match value with | .some (.bits width bits) => width > 0 && bits.toNat == 0 | _ => false

def zeroPoint (seen : List Op) : Nat → Nat → Bool
  | 0, _ => false
  | fuel + 1, id => Id.run do
    let some op := owner? seen id | return false
    if op.name == "tensor.splat" then
      let [source] := op.operands | return false
      return zeroPoint seen fuel source
    let some value := ConstantFacts.value? 64 op | return false
    match value with
    | .scalar value => return isZero value
    | .tensor value => return (allPoints value.shape).all (fun point => isZero (value.get point))

structure Reuse where
  bytes : Nat
  scale : Nat
  scalar : Bool

/-- Replay only source-proved shape/layout operations. Unknown zero points or channel mappings
refuse calibrated reuse; they cannot be asserted by a target certificate or provenance tag. -/
def reuse (args : List Value) (seen : List Op) : Nat → List Nat → Nat → Build Reuse
  | 0, _, _ => failure
  | fuel + 1, kept, id => do
    let some op := owner? seen id | failure
    let some ty := type? args seen id | failure
    let some (shape, .float .f32) := ty.staticTensor? | failure
    if dequant op then
      let [bytes, scale, point] := op.operands | failure
      let some byteType := type? args seen bytes | failure
      if byteType != tensor shape (.int 8) || !zeroPoint seen 3 point then failure
      let scalar := op.name == "quant_ext.dequantize_per_tensor"
      let scaleShape := if scalar then [] else PolicyW8A8Geometry.parameterShape shape kept
      let some scaleType := type? args seen scale | failure
      if scaleType != tensor scaleShape (.float .f32) then failure
      if !scalar then
        let [axis] := kept | failure
        if attrInt? op "axis" != some (Int.ofNat axis) then failure
      return ⟨bytes, scale, scalar⟩
    if ["tensor.collapse_shape", "tensor.expand_shape", "tensor.cast"].contains op.name then
      let source :: rest := op.operands | failure
      let [originalResult] := op.results | failure
      let some sourceType := type? args seen source | failure
      let some (sourceShape, .float .f32) := sourceType.staticTensor? | failure
      if sourceShape.prod != shape.prod || !op.regions.isEmpty || !op.successors.isEmpty then failure
      let previous ← reuse args seen fuel [] source
      if !previous.scalar then failure
      let value ← allocate (tensor shape (.int 8))
      append (.mk op.name [value] (previous.bytes :: rest) op.attrs op.props [] [])
      if originalResult.id != id then failure
      return { previous with bytes := value.id }
    if op.name == "linalg.transpose" then
      let [source, _] := op.operands | failure
      if attrDenseArray? op "permutation" != some [1, 0] || shape.length != 2 then failure
      let previous ← reuse args seen fuel (kept.map (1 - ·)) source
      let destination ← empty shape (.int 8)
      let body ← region (W8A8Fill.fillRegion (.int 8))
      let transposed ← single "linalg.transpose" (tensor shape (.int 8)) [previous.bytes, destination]
        [("permutation", .denseArray (.int 64) [1, 0])] [body]
      return { previous with bytes := transposed }
    failure

def operand (args : List Value) (seen : List Op) (shape kept outputDims : List Nat)
    (elt : Ty) (id : Nat) : Build (Nat × Option (Nat × List Nat)) := do
  if elt == .int 8 then return (id, none)
  if traces seen (seen.length + 1) id then
    let calibrated ← reuse args seen (seen.length + 1) kept id
    return (calibrated.bytes, some (calibrated.scale, if calibrated.scalar then [] else outputDims))
  let (bytes, scale) ← dynamic shape kept id
  return (bytes, some (scale, outputDims))

def lower (args : List Value) (seen : List Op) (source : Op) (view : View) : Build Unit := do
  let (left, leftScale) ← operand args seen [view.m, view.k] [0] [0] view.leftElt view.left
  let (right, rightScale) ← operand args seen [view.k, view.n] [1] [1] view.rightElt view.right
  let accumulator ← contraction [view.m, view.n] left right maps iterators source
  let _ ← requantize [view.m, view.n] accumulator (leftScale.toList ++ rightScale.toList) source view.result

def lowerProjection (args : List Value) (seen : List Op) (source : Op)
    (view : W8A8ProjectionReference.View) : Build Unit := do
  let geometry := view.layout
  let leftDimensions ← W8A8ProjectionLayout.dimensions? geometry.leftMap
  let rightDimensions ← W8A8ProjectionLayout.dimensions? geometry.rightMap
  let (left, leftScale) ← operand args seen geometry.leftShape
    (W8A8ProjectionLayout.kept geometry.mask leftDimensions)
    (W8A8ProjectionLayout.outputPositions geometry.mask leftDimensions) view.leftElt view.left
  let (right, rightScale) ← operand args seen geometry.rightShape
    (W8A8ProjectionLayout.kept geometry.mask rightDimensions)
    (W8A8ProjectionLayout.outputPositions geometry.mask rightDimensions) view.rightElt view.right
  let accumulator ← contraction geometry.outputShape left right geometry.maps geometry.iterators source
  let _ ← requantize geometry.outputShape accumulator (leftScale.toList ++ rightScale.toList) source view.result

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

def function (config : Config) (op : Op) (next : Nat) : Option (Op × Nat) := do
  let [.mk [.mk args ops]] := op.regions | none
  let mut seen : List Op := []
  let mut state : W8A8Emission.State := ⟨next, []⟩
  for source in ops do
    if candidate source && selected config seen (source.operands.take 2) then
      let build := match view? args seen source with
        | some view => lower args seen source view
        | none => do
          let view ← W8A8ProjectionReference.inspect (type? args seen) (zeroDestination seen) sourceBody source
          lowerProjection args seen source view
      let (_, state') ← build.run state
      state := state'
    else state := { state with operations := source :: state.operations }
    seen := source :: seen
  return (.mk op.name op.results op.operands op.attrs op.props
    [.mk [.mk args state.operations.reverse]] op.successors, state.next)

def expand (config : Config) (source : Module) : Option Module := do
  let mut next := boundOps source.ops
  let mut reversed : List Op := []
  for op in source.ops do
    if op.name == "func.func" && !(op.asFunc?.map (·.isDecl)).getD true then
      let (replacement, next') ← function config op next
      next := next'
      reversed := replacement :: reversed
    else reversed := op :: reversed
  return { source with ops := reversed.reverse }

def policy (config : Config) : PolicyModule.Policy :=
  ⟨"w8a8-reference-draft", expand config⟩

end MerlinVerify.Rules.W8A8Reference
