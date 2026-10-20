/- Draft reference-program construction for the declared W8A8 recipe. This code consumes source
facts only. It is not an independently authored/reviewed policy specification and cannot be
promoted to a paper claim without that review. Frozen semantics and specifications stay unchanged. -/
import MerlinVerify.Check.AlphaNormalize
import MerlinVerify.Check.PolicyW8A8Geometry
import MerlinVerify.Check.PolicyW8A8AbsMax
import MerlinVerify.Rules.W8A8Fill
import MerlinVerify.Rules.W8A8IntegerBody
import MerlinVerify.Rules.W8A8Scalar

namespace MerlinVerify.Rules.W8A8Emission

open MerlinVerify MerlinVerify.Check MerlinVerify.Sem.Ops.Linalg
open PolicyW8A8Geometry

structure State where
  next : Nat
  operations : List Op := []

abbrev Build := StateT State Option

def allocate (ty : Ty) : Build Value := do
  let state ← get
  set { state with next := state.next + 1 }
  return ⟨state.next, ty⟩

def append (op : Op) : Build Unit := modify fun state =>
  { state with operations := op :: state.operations }

def single (name : String) (ty : Ty) (operands : List Nat)
    (attrs : List (String × Attr) := []) (regions : List Region := [])
    (result : Option Value := none) : Build Nat := do
  let value ← match result with | some value => pure value | none => allocate ty
  append (.mk name [value] operands (attrs.mergeSort fun a b => a.1 ≤ b.1) [] regions [])
  return value.id

def tensor (shape : List Nat) (elt : Ty) : Ty := W8A8Fill.tensorTy shape elt

def empty (shape : List Nat) (elt : Ty) : Build Nat := single "tensor.empty" (tensor shape elt) []

def floatConstant (bits : BitVec 32) : Build Nat :=
  single "arith.constant" (.float .f32) [] [("value", .float bits.toNat (.float .f32))]

/-- Instantiate an existing proved scalar template with fresh local names and checked capture
bindings. The alpha check makes an accidental bad instantiation refuse the reference expansion. -/
def region (template : Region) (captures : Rho := []) : Build Region := do
  let .mk [.mk args ops] := template | failure
  let state ← get
  let renamedArgs := AlphaNormalize.freshValues state.next args
  let (renamedOps, next) := AlphaNormalize.operations (RegionDepth.ops ops)
    (AlphaNormalize.extend captures args renamedArgs) (state.next + args.length) ops
  let result := Region.mk [.mk renamedArgs renamedOps]
  if !alphaRegion captures template result then failure
  set { state with next }
  return result

def fill (shape : List Nat) (elt : Ty) (value destination : Nat) : Build Nat := do
  let body ← region (W8A8Fill.fillRegion elt)
  single "linalg.fill" (tensor shape elt) [value, destination]
    [("operandSegmentSizes", .denseArray (.int 32) [1, 1])] [body]

def generic (shape : List Nat) (elt : Ty) (inputs : List Nat) (destination : Nat)
    (maps : List AffineMap) (iters : List IterKind) (body : Region)
    (annotations : List (String × Attr) := []) (result : Option Value := none) : Build Nat :=
  single "linalg.generic" (tensor shape elt) (inputs ++ [destination])
    ([("indexing_maps", .array (maps.map Attr.affineMap)),
      ("iterator_types", .array (iters.map fun kind =>
        .enum "linalg" "iterator_type" (if kind == .parallel then "parallel" else "reduction"))),
      ("operandSegmentSizes", .denseArray (.int 32) [Int.ofNat inputs.length, 1])] ++ annotations)
    [body] result

/-- A dynamic operand produces its byte tensor and its scale tensor. Group dimensions are
derived from the source contraction's input map and iterator kinds by the reference front end. -/
def dynamic (shape kept : List Nat) (source : Nat) : Build (Nat × Nat) := do
  let parameters := parameterShape shape kept
  let amxEmpty ← empty parameters (.float .f32)
  let zeroLiteral ← floatConstant 0#32
  let initialized ← fill parameters (.float .f32) zeroLiteral amxEmpty
  let amxBody ← region W8A8Scalar.absMaxRegion
  let amx ← generic parameters (.float .f32) [source] initialized
    [identityMap shape.length, parameterMap shape.length kept]
    (PolicyW8A8AbsMax.iterators shape.length kept) amxBody
  let scaleEmpty ← empty parameters (.float .f32)
  let denominator ← floatConstant W8A8Scale.denominator
  let scaleBody ← region W8A8Scalar.scaleRegion [(100, denominator)]
  let scale ← generic parameters (.float .f32) [amx] scaleEmpty
    [identityMap parameters.length, identityMap parameters.length]
    (List.replicate parameters.length .parallel) scaleBody
  let inverseEmpty ← empty parameters (.float .f32)
  let inverseBody ← region W8A8Scalar.reciprocalRegion
  let inverse ← generic parameters (.float .f32) [scale] inverseEmpty
    [identityMap parameters.length, identityMap parameters.length]
    (List.replicate parameters.length .parallel) inverseBody
  let byteEmpty ← empty shape (.int 8)
  let lower ← floatConstant 0xc2fe0000#32
  let quantizeBody ← region W8A8Scalar.quantizeRegion [(100, denominator), (101, lower)]
  let quantized ← generic shape (.int 8) [source, inverse] byteEmpty
    [identityMap shape.length, parameterMap shape.length kept, identityMap shape.length]
    (List.replicate shape.length .parallel) quantizeBody
  return (quantized, scale)

def role (source : Op) (name : String) : List (String × Attr) :=
  (source.attrs.filter fun entry => entry.1.startsWith "prov." && entry.1 != "prov.role") ++
    [("prov.role", .str name)]

def contraction (shape : List Nat) (left right : Nat) (maps : List AffineMap)
    (iters : List IterKind) (source : Op) : Build Nat := do
  let destination ← empty shape (.int 32)
  let zeroLiteral ← single "arith.constant" (.int 32) [] [("value", .int 0 (.int 32))]
  let initialized ← fill shape (.int 32) zeroLiteral destination
  let body ← region W8A8IntegerBody.region
  generic shape (.int 32) [left, right] initialized maps iters body (role source "contraction")

def requantize (shape : List Nat) (accumulator : Nat) (scales : List (Nat × List Nat))
    (source : Op) (result : Value) : Build Nat := do
  if scales.length > 2 then failure
  let destination ← empty shape (.float .f32)
  let body ← region (W8A8Scalar.requantizeRegion scales.length)
  generic shape (.float .f32) (accumulator :: scales.map (·.1)) destination
    (identityMap shape.length :: ((scales.map fun s => parameterMap shape.length s.2) ++
      [identityMap shape.length]))
    (List.replicate shape.length .parallel) body (role source "requant") (some result)

end MerlinVerify.Rules.W8A8Emission
