/-
Decoder: `merlin.lean.ir.v1` JSON (`merlin/contract/schemas/lean_ir.schema.json`) → `Module`.

Total (no `partial def`): recursion is bounded by an explicit fuel, the caller passes the input's
byte length, which bounds the JSON nesting depth. Exhausted fuel is an error, never a truncation.
Every field is validated; anything not matching the schema is rejected with a path in the message.
`Lean.Json.parse` itself is part of the trusted base (design D11); the round trip (design D9)
checks decode∘export against MLIR's own printer.
-/
import MerlinVerify.Core.AST
import Lean.Data.Json

namespace MerlinVerify.Decode

open Lean

abbrev D := Except String

def field (j : Json) (k : String) : D Json :=
  match j.getObjVal? k with
  | .ok v => .ok v
  | .error _ => .error s!"missing field '{k}'"

def str (j : Json) : D String :=
  match j with | .str s => .ok s | _ => .error "expected string"

def nat (j : Json) : D Nat :=
  match j.getNat? with | .ok n => .ok n | .error _ => .error "expected natural number"

def bool (j : Json) : D Bool :=
  match j with | .bool b => .ok b | _ => .error "expected boolean"

def arr (j : Json) : D (List Json) :=
  match j with | .arr a => .ok a.toList | _ => .error "expected array"

def obj (j : Json) : D (List (String × Json)) :=
  match j with
  | .obj kvs => .ok (kvs.toArray.toList.map fun ⟨k, v⟩ => (k, v))
  | _ => .error "expected object"

def intStr (j : Json) : D Int := do
  let s ← str j
  match s.toInt? with | some v => pure v | none => throw s!"expected decimal integer, got '{s}'"

def hexDigit (c : Char) : Option Nat :=
  if '0' ≤ c ∧ c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c ∧ c ≤ 'f' then some (c.toNat - 'a'.toNat + 10)
  else none

def hexNat (s : String) : D Nat :=
  if s.isEmpty then .error "empty hex" else
  s.toList.foldlM (fun acc c => match hexDigit c with
    | some d => .ok (acc * 16 + d)
    | none => .error s!"bad hex digit '{c}'") 0

def hexBytes (s : String) : D (List UInt8) :=
  let rec go : List Char → D (List UInt8)
    | [] => .ok []
    | a :: b :: rest => match hexDigit a, hexDigit b with
      | some x, some y => (go rest).map fun bs => (x * 16 + y).toUInt8 :: bs
      | _, _ => .error "bad hex byte"
    | [_] => .error "odd-length hex"
  go s.toList

def fkind (s : String) : D FKind :=
  match s with
  | "f16" => .ok .f16 | "bf16" => .ok .bf16 | "f32" => .ok .f32 | "f64" => .ok .f64
  | _ => .error s!"unknown float format '{s}'"

def optDim (j : Json) : D (Option Nat) :=
  match j with | .null => .ok none | _ => (nat j).map some

def optField (j : Json) (k : String) : Option Json :=
  match j.getObjVal? k with
  | .ok .null => none
  | .ok v => some v
  | .error _ => none

def affineExpr : Nat → Json → D AffineExpr
  | 0, _ => .error "nesting too deep"
  | fuel + 1, j => do
    let k ← str (← field j "kind")
    match k with
    | "dim" => return .dim (← nat (← field j "pos"))
    | "symbol" => return .sym (← nat (← field j "pos"))
    | "constant" => return .const (← intStr (← field j "value"))
    | "add" | "mul" | "mod" | "floordiv" | "ceildiv" =>
      let l ← affineExpr fuel (← field j "lhs")
      let r ← affineExpr fuel (← field j "rhs")
      return match k with
        | "add" => .add l r | "mul" => .mul l r | "mod" => .mod l r
        | "floordiv" => .floordiv l r | _ => .ceildiv l r
    | _ => throw s!"unknown affine expr kind '{k}'"

mutual
  def ty : Nat → Json → D Ty
    | 0, _ => .error "nesting too deep"
    | fuel + 1, j => do
      let k ← str (← field j "kind")
      match k with
      | "int" => return .int (← nat (← field j "width"))
      | "sint" => return .sint (← nat (← field j "width"))
      | "uint" => return .uint (← nat (← field j "width"))
      | "index" => return .index
      | "none" => return .none
      | "float" => return .float (← fkind (← str (← field j "format")))
      | "tensor" =>
        let shape ← (← arr (← field j "shape")).mapM optDim
        let elt ← ty fuel (← field j "element")
        let enc ← match optField j "encoding" with
          | some e => (attr fuel e).map some
          | none => pure none
        return .tensor shape elt enc
      | "memref" =>
        let shape ← (← arr (← field j "shape")).mapM optDim
        let elt ← ty fuel (← field j "element")
        let lay ← match optField j "layout" with
          | some e => (attr fuel e).map some
          | none => pure none
        let ms ← match optField j "memory_space" with
          | some e => (attr fuel e).map some
          | none => pure none
        return .memref shape elt lay ms
      | "vector" =>
        let shape ← (← arr (← field j "shape")).mapM nat
        let sc ← (← arr (← field j "scalable")).mapM bool
        return .vector shape sc (← ty fuel (← field j "element"))
      | "function" =>
        let ins ← tys fuel (← arr (← field j "inputs"))
        let outs ← tys fuel (← arr (← field j "results"))
        return .function ins outs
      | "opaque" => return .opaque (← str (← field j "text"))
      | _ => throw s!"unknown type kind '{k}'"

  def tys : Nat → List Json → D (List Ty)
    | _, [] => .ok []
    | 0, _ :: _ => .error "nesting too deep"
    | fuel + 1, j :: js => do return (← ty fuel j) :: (← tys fuel js)

  def attr : Nat → Json → D Attr
    | 0, _ => .error "nesting too deep"
    | fuel + 1, j => do
      let k ← str (← field j "kind")
      match k with
      | "unit" => return .unit
      | "bool" => return .bool (← bool (← field j "value"))
      | "int" => return .int (← intStr (← field j "value")) (← ty fuel (← field j "type"))
      | "float" => return .float (← hexNat (← str (← field j "bits"))) (← ty fuel (← field j "type"))
      | "string" => return .str (← str (← field j "value"))
      | "type" => return .type (← ty fuel (← field j "value"))
      | "array" => return .array (← attrs fuel (← arr (← field j "elements")))
      | "dense_array" =>
        return .denseArray (← ty fuel (← field j "element")) (← (← arr (← field j "values")).mapM intStr)
      | "dense" =>
        return .dense (← ty fuel (← field j "type")) (← bool (← field j "splat"))
          (← hexBytes (← str (← field j "hex")))
      | "affine_map" =>
        let res ← (← arr (← field j "results")).mapM (affineExpr fuel)
        return .affineMap { numDims := ← nat (← field j "dims"), numSyms := ← nat (← field j "symbols"),
                            results := res }
      | "enum" =>
        return .enum (← str (← field j "dialect")) (← str (← field j "name")) (← str (← field j "value"))
      | "dict" => return .dict (← entries fuel (← obj (← field j "entries")))
      | "symbol_ref" =>
        return .symbolRef (← str (← field j "root")) (← (← arr (← field j "nested")).mapM str)
      | "opaque" => return .opaque (← str (← field j "text"))
      | _ => throw s!"unknown attribute kind '{k}'"

  def attrs : Nat → List Json → D (List Attr)
    | _, [] => .ok []
    | 0, _ :: _ => .error "nesting too deep"
    | fuel + 1, j :: js => do return (← attr fuel j) :: (← attrs fuel js)

  def entries : Nat → List (String × Json) → D (List (String × Attr))
    | _, [] => .ok []
    | 0, _ :: _ => .error "nesting too deep"
    | fuel + 1, (k, j) :: kvs => do return (k, ← attr fuel j) :: (← entries fuel kvs)
end

def value (fuel : Nat) (j : Json) : D Value := do
  return { id := ← nat (← field j "id"), ty := ← ty fuel (← field j "type") }

mutual
  def op : Nat → Json → D Op
    | 0, _ => .error "nesting too deep"
    | fuel + 1, j => do
      let name ← str (← field j "name")
      let results ← (← arr (← field j "results")).mapM (value fuel)
      let operands ← (← arr (← field j "operands")).mapM nat
      let as ← entries fuel (← obj (← field j "attrs"))
      let ps ← entries fuel (← obj (← field j "props"))
      let regs ← regions fuel (← arr (← field j "regions"))
      let succ ← (← arr (← field j "successors")).mapM nat
      return .mk name results operands as ps regs succ

  def ops : Nat → List Json → D (List Op)
    | _, [] => .ok []
    | 0, _ :: _ => .error "nesting too deep"
    | fuel + 1, j :: js => do return (← op fuel j) :: (← ops fuel js)

  def regions : Nat → List Json → D (List Region)
    | _, [] => .ok []
    | 0, _ :: _ => .error "nesting too deep"
    | fuel + 1, j :: js => do
      let bs ← blocks fuel (← arr (← field j "blocks"))
      return .mk bs :: (← regions fuel js)

  def blocks : Nat → List Json → D (List Block)
    | _, [] => .ok []
    | 0, _ :: _ => .error "nesting too deep"
    | fuel + 1, j :: js => do
      let args ← (← arr (← field j "args")).mapM (value fuel)
      let body ← ops fuel (← arr (← field j "ops"))
      return .mk args body :: (← blocks fuel js)
end

/-- Decode a whole `merlin.lean.ir.v1` document. -/
def module (text : String) : D Module := do
  let j ← match Json.parse text with
    | .ok j => pure j
    | .error e => throw s!"invalid JSON: {e}"
  let schema ← str (← field j "schema")
  if schema ≠ "merlin.lean.ir.v1" then throw s!"unexpected schema '{schema}'"
  let fuel := text.utf8ByteSize + 1
  let m ← field j "module"
  let body ← ops fuel (← arr (← field m "ops"))
  let as ← match optField m "attrs" with
    | some a => entries fuel (← obj a)
    | none => pure []
  let ps ← match optField m "props" with
    | some p => entries fuel (← obj p)
    | none => pure []
  return { ops := body, attrs := as, props := ps }

end MerlinVerify.Decode
