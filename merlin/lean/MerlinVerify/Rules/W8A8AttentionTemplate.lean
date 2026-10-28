/- A shape-parametric source execution template for the complete softmax segment.
This is a recognition target, not a new quantization policy or a success theorem. -/
import MerlinVerify.Rules.W8A8AttentionMapBody
import MerlinVerify.Rules.W8A8ReductionBody
import MerlinVerify.Rules.W8A8RowTrace
import MerlinVerify.Check.AlphaNormalize

namespace MerlinVerify.Rules.W8A8AttentionTemplate

open MerlinVerify Sem.Ops.Linalg

def tensorTy (shape : List Nat) : Ty := .tensor (shape.map some) (.float .f32) none

def value (id : Nat) (shape : List Nat) : Value := ⟨id, tensorTy shape⟩

def constant (id bits : Nat) : Op :=
  .mk "arith.constant" [⟨id, .float .f32⟩] [] [("value", .float bits (.float .f32))] [] [] []

def splat (shape : List Nat) (input result : Nat) : Op :=
  .mk "tensor.splat" [value result shape] [input] [] [] [] []

def empty (shape : List Nat) (result : Nat) : Op :=
  .mk "tensor.empty" [value result shape] [] [] [] [] []

/-- Give each closed scalar region its own fresh local namespace. The existing
normalizer is reused; the scalar body checker will still validate each result. -/
def freshRegion (start : Nat) (template : Region) : Region :=
  match template with
  | .mk [.mk arguments body] =>
    let renamed := Check.AlphaNormalize.freshValues start arguments
    let (operations, _) := Check.AlphaNormalize.operations (Check.RegionDepth.ops body)
      (Check.AlphaNormalize.extend [] arguments renamed) (start + arguments.length) body
    .mk [.mk renamed operations]
  | _ => .mk []

def reduce (shape : List Nat) (input seed result : Nat) (kind : W8A8ReductionBody.Kind) : Op :=
  .mk "linalg.reduce" [value result shape] [input, seed]
    [("dimensions", .denseArray (.int 64) [shape.length])] []
    [freshRegion (if kind == .maximum then 100 else 200) (W8A8ReductionBody.region kind)] []

def collapse (shape : List Nat) (input result : Nat) : Op :=
  .mk "tensor.collapse_shape" [value result [shapeSize shape]] [input]
    [("reassociation", .array [.array ((List.range shape.length).map fun (d : Nat) => .int (Int.ofNat d) (.int 64))])] [] [] []

def expand (shape : List Nat) (input result : Nat) : Op :=
  .mk "tensor.expand_shape" [value result shape] [input]
    [("reassociation", .array [.array ((List.range shape.length).map fun (d : Nat) => .int (Int.ofNat d) (.int 64))]),
     ("static_output_shape", .denseArray (.int 64) (shape.map Int.ofNat))] [] [] []

def broadcastMap (rank : Nat) : AffineMap :=
  ⟨rank + 1, 0, (List.range rank).map AffineExpr.dim ++ [.const 0]⟩

def generic (shape : List Nat) (inputs : List Nat) (destination result : Nat)
    (maps : List AffineMap) (body : Region) : Op :=
  .mk "linalg.generic" [value result shape] (inputs ++ [destination])
    [("indexing_maps", .array (maps.map Attr.affineMap)),
     ("iterator_types", .array (List.replicate shape.length (.enum "linalg" "iterator_type" "parallel"))),
     ("operandSegmentSizes", .denseArray (.int 32) [inputs.length, 1])] [] [body] []

def operations (shape : List Nat) (width : Nat) : List Op :=
  let full := shape ++ [width]
  let unit := shape ++ [1]
  let identity := identityMap full.length
  let broadcast := broadcastMap shape.length
  [constant 1 0xff800000,
   splat shape 1 2,
   reduce shape 0 2 3 .maximum,
   collapse shape 3 4,
   expand unit 4 5,
   empty full 6,
   generic full [0, 5] 6 7 [identity, broadcast, identity]
     (freshRegion 300 (W8A8AttentionMapBody.binaryRegion .subtract)),
   empty full 8,
   generic full [7] 8 9 [identity, identity] (freshRegion 400 W8A8AttentionMapBody.exponentialRegion),
   constant 10 0,
   splat shape 10 11,
   reduce shape 9 11 12 .sum,
   collapse shape 12 13,
   expand unit 13 14,
   empty full 15,
   generic full [9, 14] 15 16 [identity, broadcast, identity]
     (freshRegion 500 (W8A8AttentionMapBody.binaryRegion .divide))]

def terminator (output : Nat) : Op := .mk "func.return" [] [output] [] [] [] []

end MerlinVerify.Rules.W8A8AttentionTemplate
