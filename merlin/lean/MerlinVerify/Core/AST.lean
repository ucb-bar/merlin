/-
G0-FROZEN. The abstract syntax every checker, decoder and semantics agrees on.

This mirrors the JSON interchange `merlin.lean.ir.v1`
(`merlin/contract/schemas/lean_ir.schema.json`), which the MLIR exporter emits from MLIR's own
parser (design D9). The AST is *extrinsic*: nothing here is typed beyond what MLIR printed. Typing
and support are decided by the semantics, which returns `stuck` for anything it does not model.

Changing this file after G0 requires an interface change request (design §11.2).
-/

namespace MerlinVerify

/-- Floating-point storage formats. Values of every format are bit patterns (design D5). -/
inductive FKind where
  | f16 | bf16 | f32 | f64
  deriving DecidableEq, Repr, Inhabited, Hashable

def FKind.width : FKind → Nat
  | .f16 => 16 | .bf16 => 16 | .f32 => 32 | .f64 => 64

/-- An SSA value identifier. The exporter numbers every SSA value (op results and block
arguments) uniquely within one module, in program order. -/
abbrev ValueId := Nat

/-- Quasi-affine expressions, as MLIR's `AffineExpr`. -/
inductive AffineExpr where
  | dim (pos : Nat)
  | sym (pos : Nat)
  | const (value : Int)
  | add (lhs rhs : AffineExpr)
  | mul (lhs rhs : AffineExpr)
  | mod (lhs rhs : AffineExpr)
  | floordiv (lhs rhs : AffineExpr)
  | ceildiv (lhs rhs : AffineExpr)
  deriving DecidableEq, Repr, Inhabited, Hashable

structure AffineMap where
  numDims : Nat
  numSyms : Nat
  results : List AffineExpr
  deriving DecidableEq, Repr, Inhabited, Hashable

mutual
  /-- MLIR types. Only static shapes are representable as `tensor`/`memref`; the exporter emits a
  dynamic dimension as `none`, which every semantics treats as unsupported. -/
  inductive Ty where
    /-- Signless integer `iN` (MLIR `IntegerType` with signless semantics). -/
    | int (width : Nat)
    /-- Signed/unsigned integer types (`siN`/`uiN`) are kept distinct from signless ones. -/
    | sint (width : Nat)
    | uint (width : Nat)
    | index
    | float (kind : FKind)
    | tensor (shape : List (Option Nat)) (elt : Ty) (encoding : Option Attr)
    | memref (shape : List (Option Nat)) (elt : Ty) (layout : Option Attr) (memorySpace : Option Attr)
    | vector (shape : List Nat) (scalable : List Bool) (elt : Ty)
    | function (inputs : List Ty) (results : List Ty)
    | none
    /-- Any other type, identified by its printed form only. -/
    | opaque (text : String)

  /-- MLIR attributes. Integer payloads are exact; float payloads are bit patterns; dense payloads
  are the exact little-endian storage bytes MLIR holds. -/
  inductive Attr where
    | unit
    | bool (value : Bool)
    | int (value : Int) (ty : Ty)
    /-- `FloatAttr`: the IEEE bit pattern of the value, as stored for its type. -/
    | float (bits : Nat) (ty : Ty)
    | str (value : String)
    | type (value : Ty)
    | array (elements : List Attr)
    /-- `DenseI8/16/32/64ArrayAttr`, `DenseBoolArrayAttr`, … : `elt` is the element type. -/
    | denseArray (elt : Ty) (values : List Int)
    /-- `DenseElementsAttr`/`DenseResourceElementsAttr` with materialized bytes. `splat` means
    `bytes` holds one element. -/
    | dense (ty : Ty) (splat : Bool) (bytes : List UInt8)
    | affineMap (map : AffineMap)
    /-- Dialect enum attributes with a known printed form, e.g. `#linalg.iterator_type<parallel>`
    is `enum "linalg" "iterator_type" "parallel"`; `#arith.fastmath<nnan,ninf>` is
    `enum "arith" "fastmath" "nnan,ninf"`. -/
    | enum (dialect : String) (name : String) (value : String)
    | dict (entries : List (String × Attr))
    | symbolRef (root : String) (nested : List String)
    /-- Anything else, by its printed form. An attribute the semantics must interpret but finds
    opaque makes the op unsupported. -/
    | opaque (text : String)
end

deriving instance Repr for Ty, Attr
deriving instance Inhabited for Ty
deriving instance Inhabited for Attr

structure Value where
  id : ValueId
  ty : Ty
  deriving Repr, Inhabited

mutual
  /-- One MLIR operation in generic form. `attrs` are the discardable attributes; `props` are the
  inherent attributes MLIR stores as properties (printed as `<{...}>`). `successors` are indices of
  sibling blocks in the enclosing region. -/
  inductive Op where
    | mk (name : String) (results : List Value) (operands : List ValueId)
         (attrs : List (String × Attr)) (props : List (String × Attr))
         (regions : List Region) (successors : List Nat)

  inductive Region where
    | mk (blocks : List Block)

  inductive Block where
    | mk (args : List Value) (ops : List Op)
end

deriving instance Repr for Op, Region, Block
deriving instance Inhabited for Op
deriving instance Inhabited for Region
deriving instance Inhabited for Block

namespace Op
def name : Op → String | .mk n .. => n
def results : Op → List Value | .mk _ r .. => r
def operands : Op → List ValueId | .mk _ _ o .. => o
def attrs : Op → List (String × Attr) | .mk _ _ _ a .. => a
def props : Op → List (String × Attr) | .mk _ _ _ _ p .. => p
def regions : Op → List Region | .mk _ _ _ _ _ r _ => r
def successors : Op → List Nat | .mk _ _ _ _ _ _ s => s

/-- Look an attribute up among properties first, then discardable attributes. -/
def attr? (op : Op) (key : String) : Option Attr :=
  match op.props.lookup key with
  | some a => some a
  | none => op.attrs.lookup key
end Op

def Region.blocks : Region → List Block | .mk b => b
def Block.args : Block → List Value | .mk a _ => a
def Block.ops : Block → List Op | .mk _ o => o

/-- A whole module: the ops of the top-level `builtin.module` body (normally `func.func` ops), plus
the top-level module's own discardable attributes (e.g. `prov.weights_file`) and properties (e.g.
`sym_name` of `module @name`). Module attributes are data the semantics ignores unless a reviewed
spec says otherwise; they are kept so export and round trip never drop them. -/
structure Module where
  ops : List Op
  attrs : List (String × Attr) := []
  props : List (String × Attr) := []
  deriving Repr, Inhabited

/-- A `func.func` view of a top-level op: symbol name, entry block, visibility. -/
structure Func where
  name : String
  body : Region
  isPrivate : Bool
  /-- `true` when the function has no body (an external declaration). -/
  isDecl : Bool

def Op.asFunc? (op : Op) : Option Func :=
  if op.name ≠ "func.func" then none else
  match op.attr? "sym_name" with
  | some (.str n) =>
      let priv := match op.attr? "sym_visibility" with
        | some (.str "private") => true
        | _ => false
      let body := op.regions.headD (.mk [])
      some { name := n, body := body, isPrivate := priv, isDecl := body.blocks.isEmpty }
  | _ => none

def Module.funcs (m : Module) : List Func := m.ops.filterMap Op.asFunc?

def Module.func? (m : Module) (name : String) : Option Func :=
  m.funcs.find? (·.name == name)

end MerlinVerify
