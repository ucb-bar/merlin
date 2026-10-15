/-
Printer: `Module` → generic-form MLIR text, for the round-trip check (design D9): the exporter's
JSON, decoded by Lean and printed here, must canonicalize (`mlir-opt --mlir-print-op-generic`) to
exactly the same bytes as the original file. Value names are `%v<id>`; MLIR renumbers on print.
-/
import MerlinVerify.Core.AST

namespace MerlinVerify.Print

def joinSep (sep : String) : List String → String
  | [] => ""
  | [x] => x
  | x :: xs => x ++ sep ++ joinSep sep xs

def hexDigits (n : Nat) (width : Nat) : String :=
  let ds := Nat.toDigits 16 n
  String.ofList (List.replicate (width - ds.length) '0' ++ ds)

def fkind : FKind → String
  | .f16 => "f16" | .bf16 => "bf16" | .f32 => "f32" | .f64 => "f64"

def escape (s : String) : String :=
  s.foldl (fun acc c =>
    if c = '"' then acc ++ "\\\"" else if c = '\\' then acc ++ "\\\\"
    else if c = '\n' then acc ++ "\\n" else if c.toNat < 32 then acc ++ "\\" ++ hexDigits c.toNat 2
    else acc.push c) ""

def affineExpr : AffineExpr → String
  | .dim p => s!"d{p}"
  | .sym p => s!"s{p}"
  | .const c => toString c
  | .add a b => s!"({affineExpr a} + {affineExpr b})"
  | .mul a b => s!"({affineExpr a} * {affineExpr b})"
  | .mod a b => s!"({affineExpr a} mod {affineExpr b})"
  | .floordiv a b => s!"({affineExpr a} floordiv {affineExpr b})"
  | .ceildiv a b => s!"({affineExpr a} ceildiv {affineExpr b})"

def affineMap (m : AffineMap) : String :=
  let dims := joinSep ", " ((List.range m.numDims).map fun i => s!"d{i}")
  let syms := if m.numSyms = 0 then "" else
    "[" ++ joinSep ", " ((List.range m.numSyms).map fun i => s!"s{i}") ++ "]"
  s!"({dims}){syms} -> ({joinSep ", " (m.results.map affineExpr)})"

/-- MLIR's `bare-id`: a letter or `_`, then letters, digits, `_`, `$` or `.`. -/
def bareId (s : String) : Bool :=
  match s.toList with
  | c :: cs => (c.isAlpha || c == '_') && cs.all fun c => c.isAlphanum || c == '_' || c == '$' || c == '.'
  | [] => false

/-- A symbol name or dictionary key: bare when MLIR's grammar allows, else a quoted string. -/
def ident (s : String) : String := if bareId s then s else s!"\"{escape s}\""

/-- Element literals nested by a static shape (row-major), as MLIR prints a dense literal. -/
def nested : List Nat → List String → String
  | [], lits => lits.headD ""
  | d :: ds, lits =>
    let inner := ds.foldl (· * ·) 1
    "[" ++ joinSep ", " ((List.range d).map fun k => nested ds ((lits.drop (k * inner)).take inner)) ++ "]"

def dim : Option Nat → String
  | some n => toString n
  | none => "?"

mutual
  def ty : Ty → String
    | .int w => s!"i{w}"
    | .sint w => s!"si{w}"
    | .uint w => s!"ui{w}"
    | .index => "index"
    | .float k => fkind k
    | .tensor shape elt enc =>
      let ds := String.join (shape.map fun d => dim d ++ "x")
      let e := match enc with | some a => ", " ++ attr a | none => ""
      s!"tensor<{ds}{ty elt}{e}>"
    | .memref shape elt lay ms =>
      let ds := String.join (shape.map fun d => dim d ++ "x")
      let l := match lay with | some a => ", " ++ attr a | none => ""
      let m := match ms with | some a => ", " ++ attr a | none => ""
      s!"memref<{ds}{ty elt}{l}{m}>"
    | .vector shape sc elt =>
      let ds := String.join ((shape.zip (sc ++ List.replicate shape.length false)).map fun (d, s) =>
        (if s then s!"[{d}]" else toString d) ++ "x")
      s!"vector<{ds}{ty elt}>"
    | .function ins outs => s!"({tys ins}) -> ({tys outs})"
    | .none => "none"
    | .opaque t => t

  def tys : List Ty → String
    | [] => ""
    | [t] => ty t
    | t :: ts => ty t ++ ", " ++ tys ts

  def attr : Attr → String
    | .unit => "unit"
    | .bool b => if b then "true" else "false"
    | .int v t => s!"{v} : {ty t}"
    | .float bits t =>
      let w := match t with | .float k => k.width / 4 | _ => 8
      s!"0x{hexDigits bits w} : {ty t}"
    | .str s => s!"\"{escape s}\""
    | .type t => ty t
    | .array xs => "[" ++ attrList xs ++ "]"
    | .denseArray elt vs =>
      if vs.isEmpty then s!"array<{ty elt}>"
      else s!"array<{ty elt}: {joinSep ", " (vs.map toString)}>"
    | .dense t splat bytes =>
      match t with
      | .tensor shape (.int 1) _ =>
        -- i1 storage is printed as literals, one per stored element, nested by the shape.
        let lits := bytes.map fun b => if b = 0 then "false" else "true"
        if splat then s!"dense<{joinSep ", " lits}> : {ty t}"
        else match shape.mapM id with
          | some sh => s!"dense<{nested sh lits}> : {ty t}"
          | none => s!"dense<[{joinSep ", " lits}]> : {ty t}"
      | _ => s!"dense<\"0x{String.join (bytes.map fun b => (hexDigits b.toNat 2).toUpper)}\"> : {ty t}"
    | .affineMap m => s!"affine_map<{affineMap m}>"
    | .enum d n v => s!"#{d}.{n}<{v}>"
    | .dict es => "{" ++ entries es ++ "}"
    | .symbolRef r ns => String.join (("@" ++ ident r) :: ns.map fun n => "::@" ++ ident n)
    | .opaque t => t

  def attrList : List Attr → String
    | [] => ""
    | [a] => attr a
    | a :: as => attr a ++ ", " ++ attrList as

  def entries : List (String × Attr) → String
    | [] => ""
    | [e] => entry e
    | e :: es => entry e ++ ", " ++ entries es

  def entry : String × Attr → String
    | (k, .unit) => ident k
    | (k, a) => ident k ++ " = " ++ attr a
end

abbrev TyEnv := List (ValueId × Ty)

def valName (id : ValueId) : String := s!"%v{id}"

def operandTy (env : TyEnv) (id : ValueId) : String :=
  match env.lookup id with
  | some t => ty t
  | none => "none"  -- an unbound operand: the round trip will fail loudly at mlir-opt

mutual
  def op (indent : String) (env : TyEnv) : Op → String × TyEnv
    | .mk name results operands as ps regions succ =>
      let lhs := if results.isEmpty then "" else
        joinSep ", " (results.map fun r => valName r.id) ++ " = "
      let args := joinSep ", " (operands.map valName)
      let succs := if succ.isEmpty then "" else "[" ++ joinSep ", " (succ.map fun s => s!"^bb{s}") ++ "]"
      let props := if ps.isEmpty then "" else " <{" ++ entries ps ++ "}>"
      let regs := if regions.isEmpty then "" else
        " (" ++ joinSep ", " (regionList (indent ++ "  ") env regions) ++ ")"
      let attrs := if as.isEmpty then "" else " {" ++ entries as ++ "}"
      let sig := s!"({joinSep ", " (operands.map (operandTy env))}) -> ({tys (results.map (·.ty))})"
      (s!"{indent}{lhs}\"{escape name}\"({args}){succs}{props}{regs}{attrs} : {sig}",
       results.reverse.map (fun r => (r.id, r.ty)) ++ env)

  def opList (indent : String) (env : TyEnv) : List Op → List String
    | [] => []
    | o :: os => let (line, env') := op indent env o; line :: opList indent env' os

  def regionList (indent : String) (env : TyEnv) : List Region → List String
    | [] => []
    | r :: rs => region indent env r :: regionList indent env rs

  def region (indent : String) (env : TyEnv) : Region → String
    | .mk bs => "{\n" ++ String.join (blockList indent env 0 bs) ++ indent ++ "}"

  def blockList (indent : String) (env : TyEnv) (i : Nat) : List Block → List String
    | [] => []
    | (.mk args body) :: bs =>
      let header := s!"{indent}^bb{i}({joinSep ", " (args.map fun a => valName a.id ++ ": " ++ ty a.ty)}):\n"
      let env' := args.reverse.map (fun a => (a.id, a.ty)) ++ env
      (header ++ String.join ((opList (indent ++ "  ") env' body).map (· ++ "\n")))
        :: blockList indent env (i + 1) bs
end

def module (m : Module) : String :=
  let body := String.join ((opList "  " [] m.ops).map (· ++ "\n"))
  let props := if m.props.isEmpty then "" else " <{" ++ entries m.props ++ "}>"
  let attrs := if m.attrs.isEmpty then "" else " {" ++ entries m.attrs ++ "}"
  "\"builtin.module\"()" ++ props ++ " ({\n^bb0:\n" ++ body ++ "})" ++ attrs ++ " : () -> ()\n"

end MerlinVerify.Print
