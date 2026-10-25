/- A finite structural AST observation used ONLY to prove inequality. There is
no injectivity premise and no acceptance based on this observation. If a change
is invisible at the chosen depth, the refusal proof simply cannot be produced. -/
import MerlinVerify.Core.AST

namespace Merlin.GraphRegionsSource
open MerlinVerify

inductive SyntaxAtom where
  | text (value : String)
  | number (value : Int)
  | boolean (value : Bool)
  deriving DecidableEq

private def dims (shape : List (Option Nat)) : List SyntaxAtom :=
  shape.map fun dimension => .number (dimension.map Int.ofNat |>.getD (-1))

private def affineTrace : AffineExpr → List SyntaxAtom
  | .dim n => [.text "dim", .number n]
  | .sym n => [.text "sym", .number n]
  | .const n => [.text "const", .number n]
  | .add a b => .text "add" :: affineTrace a ++ affineTrace b
  | .mul a b => .text "mul" :: affineTrace a ++ affineTrace b
  | .mod a b => .text "mod" :: affineTrace a ++ affineTrace b
  | .floordiv a b => .text "floordiv" :: affineTrace a ++ affineTrace b
  | .ceildiv a b => .text "ceildiv" :: affineTrace a ++ affineTrace b

mutual
  def typeTrace : Nat → Ty → List SyntaxAtom
    | 0, _ => []
    | n+1, type => match type with
      | .int w => [.text "int", .number w]
      | .sint w => [.text "sint", .number w]
      | .uint w => [.text "uint", .number w]
      | .index => [.text "index"]
      | .float k => [.text "float", .number k.width, .boolean (k = .bf16)]
      | .tensor shape element encoding => .text "tensor" :: dims shape ++ typeTrace n element ++
          encoding.toList.flatMap (attrTrace n)
      | .memref shape element layout space => .text "memref" :: dims shape ++ typeTrace n element ++
          layout.toList.flatMap (attrTrace n) ++ space.toList.flatMap (attrTrace n)
      | .vector shape scalable element => .text "vector" :: shape.map (.number ∘ Int.ofNat) ++
          scalable.map SyntaxAtom.boolean ++ typeTrace n element
      | .function inputs results => .text "function" :: inputs.flatMap (typeTrace n) ++
          [.text "results"] ++ results.flatMap (typeTrace n)
      | .none => [.text "none"]
      | .opaque text => [.text "opaque", .text text]

  def attrTrace : Nat → Attr → List SyntaxAtom
    | 0, _ => []
    | n+1, attr => match attr with
      | .unit => [.text "unit"]
      | .bool b => [.text "bool", .boolean b]
      | .int i type => [.text "int", .number i] ++ typeTrace n type
      | .float bits type => [.text "float", .number bits] ++ typeTrace n type
      | .str text => [.text "str", .text text]
      | .type type => .text "type" :: typeTrace n type
      | .array values => .text "array" :: values.flatMap (attrTrace n)
      | .denseArray type values => .text "denseArray" :: typeTrace n type ++ values.map SyntaxAtom.number
      | .dense type splat bytes => [.text "dense", .boolean splat] ++ typeTrace n type ++
          bytes.map (fun byte => .number byte.toNat)
      | .affineMap map => [.text "affineMap", .number map.numDims, .number map.numSyms] ++
          map.results.flatMap affineTrace
      | .enum dialect name value => [.text "enum", .text dialect, .text name, .text value]
      | .dict entries => .text "dict" :: entries.flatMap (fun (key, value) => .text key :: attrTrace n value)
      | .symbolRef root nested => [.text "symbolRef", .text root] ++ nested.map SyntaxAtom.text
      | .opaque text => [.text "opaque", .text text]
end

def valueTrace (fuel : Nat) (value : Value) : List SyntaxAtom :=
  .number value.id :: typeTrace fuel value.ty

mutual
  def opTrace : Nat → Op → List SyntaxAtom
    | 0, _ => []
    | n+1, .mk name results operands attrs props regions successors =>
      [.text name] ++ results.flatMap (valueTrace n) ++ [.text "operands"] ++
      operands.map (.number ∘ Int.ofNat) ++ [.text "attrs"] ++
      attrs.flatMap (fun (key, value) => .text key :: attrTrace n value) ++ [.text "props"] ++
      props.flatMap (fun (key, value) => .text key :: attrTrace n value) ++ [.text "regions"] ++
      regions.flatMap (regionTrace n) ++ [.text "successors"] ++ successors.map (.number ∘ Int.ofNat)

  def regionTrace : Nat → Region → List SyntaxAtom
    | 0, _ => []
    | n+1, .mk blocks => blocks.flatMap (blockTrace n)

  def blockTrace : Nat → Block → List SyntaxAtom
    | 0, _ => []
    | n+1, .mk args ops => args.flatMap (valueTrace n) ++ [.text "body"] ++ ops.flatMap (opTrace n)
end

def sourceTrace (module : Module) : List SyntaxAtom :=
  module.ops.flatMap (opTrace 64) ++ module.attrs.flatMap (fun (key, value) => .text key :: attrTrace 64 value) ++
    module.props.flatMap (fun (key, value) => .text key :: attrTrace 64 value)

end Merlin.GraphRegionsSource
