/-
Shared helpers for op semantics: attribute decoding, scalar widths, constant materialization and
elementwise lifting. Every decoder returns `Outcome` and is `stuck` on anything it does not decode
exactly (design §1.2 rule 3).

API (stable; used by every op family):
* `attrInt?`, `attrDenseArray?`, `attrAffineMaps?`, `attrEnum?`, `attrSymbol?`, `hasUnit` — attribute access.
* `Ty.bitWidth? (indexBits)` — storage width of a scalar type; `Ty.isFloat?`.
* `scalarOfAttr` — `arith.constant`-style scalar attribute → `Val`.
* `denseTensor` — `DenseElementsAttr` bytes → `Tensor` (little-endian, element storage rounded up to
  whole bytes; i1 one byte; index 8 bytes).
* `Tensor.canon`, `Tensor.canonize`, `Tensor.full`, `Tensor.set`, `allPoints`, `linearIndex` — tensor
  utilities. Every tensor an op produces is CANONICAL: `poison` outside its shape (`Tensor.canon`).
  Out-of-bounds elements are never observable, and canonical tensors that agree in bounds are equal
  (`Tensor.canon_ext`), which is what lets two constructions of one value be proved equal.
* `liftN` — apply a poison-aware scalar function elementwise over equal-shape tensors.
* `oneResult`, `expectArity` — result/arity plumbing.
-/
import MerlinVerify.Sem.Program
import MerlinVerify.Core.Eq

namespace MerlinVerify


/-! ### Attributes -/

def attrInt? (op : Op) (key : String) : Option Int :=
  match op.attr? key with
  | some (.int v _) => some v
  | _ => none

def attrDenseArray? (op : Op) (key : String) : Option (List Int) :=
  match op.attr? key with
  | some (.denseArray _ vs) => some vs
  | _ => none

def attrAffineMaps? (op : Op) (key : String) : Option (List AffineMap) :=
  match op.attr? key with
  | some (.array xs) => xs.mapM fun | .affineMap m => some m | _ => none
  | _ => none

/-- `#dialect.name<value>` enum attribute value, if the attribute is that enum. -/
def attrEnum? (op : Op) (key dialect name : String) : Option String :=
  match op.attr? key with
  | some (.enum d n v) => if d == dialect && n == name then some v else none
  | _ => none

def attrSymbol? (op : Op) (key : String) : Option String :=
  match op.attr? key with
  | some (.symbolRef root []) => some root
  | _ => none

def hasUnit (op : Op) (key : String) : Bool :=
  match op.attr? key with
  | some .unit => true
  | _ => false

/-- Comma-separated flag list of an enum attribute (`"nsw, nuw"` → `["nsw", "nuw"]`). -/
def flagList (s : String) : List String :=
  (s.splitOn ",").map (·.trimAscii.toString) |>.filter (· ≠ "")

/-- Flags of an `#arith.overflow<...>` attribute under `key` (absent means none). -/
def overflowFlags (op : Op) : List String :=
  match attrEnum? op "overflowFlags" "arith" "overflow" with
  | some s => (flagList s).filter (· ≠ "none")
  | none => []

/-! ### Types and scalars -/

/-- Storage width of a scalar type (`index` at the claim's width). -/
def Ty.bitWidth? (indexBits : Nat) : Ty → Option Nat
  | .int w => some w
  | .index => some indexBits
  | .float k => some k.width
  | _ => Option.none

def Ty.isFloat? : Ty → Option FKind
  | .float k => some k
  | _ => Option.none

/-- Is this a static, scalar-element tensor type? Returns shape and element type. -/
def Ty.staticTensor? : Ty → Option (List Nat × Ty)
  | .tensor shape elt Option.none => (shape.mapM id).map (·, elt)
  | _ => Option.none

def Scalar.width : Scalar → Nat
  | .bits w _ => w
  | .flt k _ => k.width

/-- Does a scalar runtime value have scalar type `ty`? -/
def Scalar.conforms (indexBits : Nat) (ty : Ty) (s : Scalar) : Bool :=
  match ty, s with
  | .int w, .bits w' _ => w == w'
  | .index, .bits w' _ => indexBits == w'
  | .float k, .flt k' _ => k == k'
  | _, _ => false

def Val.conforms (indexBits : Nat) (ty : Ty) : Val → Bool
  | .poison => (ty.bitWidth? indexBits).isSome
  | .some s => s.conforms indexBits ty

/-- Scalar value of an integer/index/float typed attribute. -/
def scalarOfAttr (indexBits : Nat) : Attr → Option Scalar
  | .int v (.int w) => some (.bits w (BitVec.ofInt w v))
  | .int v .index => some (.bits indexBits (BitVec.ofInt indexBits v))
  | .float bits (.float k) => some (.flt k (BitVec.ofNat k.width bits))
  | .bool b => some (.bits 1 (if b then 1 else 0))
  | _ => none

/-! ### Tensors -/

/-- Row-major linear index of `i` in `shape` (no bounds check). -/
def linearIndex : List Nat → Idx → Nat
  | [], _ => 0
  | _ :: dims, [] => 0 * dims.foldl (· * ·) 1
  | _ :: dims, x :: xs => x * dims.foldl (· * ·) 1 + linearIndex dims xs

def shapeSize (shape : List Nat) : Nat := shape.foldl (· * ·) 1

/-- Every multi-index of `shape`, in lexicographic (row-major) order. -/
def allPoints : List Nat → List Idx
  | [] => [[]]
  | d :: ds => (List.range d).flatMap fun i => (allPoints ds).map (i :: ·)

/-- The canonical tensor with elements `f` in bounds and `poison` elsewhere. -/
def Tensor.canon (shape : List Nat) (elt : Ty) (f : Idx → Val) : Tensor :=
  { shape, elt, get := fun i => if Idx.InBounds shape i then f i else .poison }

def Tensor.canonize (t : Tensor) : Tensor := Tensor.canon t.shape t.elt t.get

def RVal.canonize : RVal → RVal
  | .tensor t => .tensor t.canonize
  | v => v

theorem Tensor.canon_ext {shape : List Nat} {elt : Ty} {f g : Idx → Val}
    (h : ∀ i, Idx.InBounds shape i → f i = g i) : Tensor.canon shape elt f = Tensor.canon shape elt g := by
  unfold Tensor.canon
  congr 1
  funext i
  split
  · next hi => exact h i hi
  · rfl

def Tensor.full (shape : List Nat) (elt : Ty) (v : Val) : Tensor :=
  Tensor.canon shape elt fun _ => v

def Tensor.set (t : Tensor) (i : Idx) (v : Val) : Tensor :=
  { t with get := fun j => if j = i then v else t.get j }

/-- Little-endian unsigned value of a byte list. -/
def leNat : List UInt8 → Nat
  | [] => 0
  | b :: bs => b.toNat + 256 * leNat bs

/-- Bytes per stored element of a dense attribute (element width rounded up to whole bytes;
`index` is stored as 64-bit). -/
def denseElemBytes : Ty → Option Nat
  | .int w => some ((w + 7) / 8)
  | .index => some 8
  | .float k => some (k.width / 8)
  | _ => none

def scalarOfBits (indexBits : Nat) (elt : Ty) (n : Nat) : Option Scalar :=
  match elt with
  | .int w => some (.bits w (BitVec.ofNat w n))
  | .index => some (.bits indexBits (BitVec.ofInt indexBits (BitVec.ofNat 64 n).toInt))
  | .float k => some (.flt k (BitVec.ofNat k.width n))
  | _ => none

/-- Decode a `DenseElementsAttr` of static tensor type `ty` into a tensor. The byte count must
match exactly (one element when `splat`). -/
def denseTensor (indexBits : Nat) (ty : Ty) (splat : Bool) (bytes : List UInt8) : Outcome Tensor := do
  let some (shape, elt) := ty.staticTensor? | .stuck "dense attribute of non-static tensor type"
  let some eb := denseElemBytes elt | .stuck "dense attribute of unsupported element type"
  let n := shapeSize shape
  if eb = 0 then .stuck "zero-width dense element" else
  if splat then
    if bytes.length ≠ eb then .stuck "dense splat byte count mismatch" else
    let some s := scalarOfBits indexBits elt (leNat bytes) | .stuck "dense element decode"
    pure (Tensor.full shape elt (.some s))
  else
    if bytes.length ≠ n * eb then .stuck "dense byte count mismatch" else
    let arr := bytes.toArray
    -- Decode every element eagerly so a decode failure is `stuck`, not a lazy surprise.
    let elems ← (List.range n).mapM fun k =>
      Outcome.ofOption "dense element decode"
        (scalarOfBits indexBits elt (leNat ((arr.extract (k * eb) (k * eb + eb)).toList)))
    let elemsArr := elems.toArray
    pure (Tensor.canon shape elt fun i => match elemsArr[linearIndex shape i]? with
                                           | some s => .some s
                                           | none => .poison)

/-! ### Results and elementwise lifting -/

def oneResult (v : RVal) : Outcome (List RVal) := .ok [v]

def expectArity {α : Type} (n : Nat) (xs : List α) (what : String) : Outcome Unit :=
  if xs.length = n then .ok () else .stuck s!"{what}: expected {n}, got {xs.length}"

/-- Apply a poison-aware scalar function to scalar operands, or elementwise to tensor operands
of one common static shape. Mixed scalar/tensor operands are `stuck`. `f` must not raise UB
(ops that can raise UB are scalar-only). -/
def liftN (resultElt : Ty) (f : List Val → Val) (args : List RVal) : Outcome RVal :=
  match args.mapM (fun | .scalar v => some v | _ => none) with
  | some vs => .ok (.scalar (f vs))
  | none =>
    match args.mapM (fun | .tensor t => some t | _ => none) with
    | some (t :: ts) =>
      if ts.all (·.shape == t.shape) then
        .ok (.tensor (Tensor.canon t.shape resultElt fun i => f ((t :: ts).map (·.get i))))
      else .stuck "elementwise operands of different shapes"
    | _ => .stuck "elementwise operands are neither all scalars nor all tensors"

/-- Like `liftN`, but the scalar function may be `ub`/`stuck`, evaluated eagerly at every in-bounds
index: the op is `ub` (resp. `stuck`) if any element is, matching MLIR's elementwise rule ("the
behavior is undefined if any of its elements ..."). The result tensor is array-backed. -/
def liftEager (resultElt : Ty) (f : List Val → Outcome Val) (args : List RVal) : Outcome RVal :=
  match args.mapM (fun | .scalar v => some v | _ => none) with
  | some vs => (f vs).bind fun v => .ok (.scalar v)
  | none =>
    match args.mapM (fun | .tensor t => some t | _ => none) with
    | some (t :: ts) =>
      if ts.all (·.shape == t.shape) then
        ((allPoints t.shape).mapM fun i => f ((t :: ts).map (·.get i))).bind fun vals =>
          let arr := vals.toArray
          .ok (.tensor (Tensor.canon t.shape resultElt fun i => arr[linearIndex t.shape i]?.getD .poison))
      else .stuck "elementwise operands of different shapes"
    | _ => .stuck "elementwise operands are neither all scalars nor all tensors"

end MerlinVerify
