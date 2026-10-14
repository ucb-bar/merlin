/-
Op family: `arith.*` and `math.*` (design §4.3, D5, D6).

Integers are exact `BitVec`s with the Arith dialect's poison and UB rules
(llvm-project `a47bddcce`, `mlir/include/mlir/Dialect/Arith/IR/ArithOps.td`):
* immediate UB: division by zero (`divui/divsi/remui/remsi/ceildivui/ceildivsi/floordivsi`) and signed
  overflow `INT_MIN / -1` (`divsi/ceildivsi/floordivsi`). A poison divisor is UB (it may be zero);
  a poison dividend with divisor `-1` in a signed division is UB (it may be `INT_MIN`).
* poison: `nsw`/`nuw` overflow, shift amount ≥ width, `exact` violations, `nneg` violations,
  poison operands, poison `select` condition.
* `remsi INT_MIN, -1` is defined as `0` (the spec is silent; LLVM `srem` would be UB) — ledger item.

Floats are uninterpreted: every float operation is `ctx.M` applied to an `FloatOpKey` that encodes
the op name and its numerics attributes (fast-math mask, rounding mode, predicate), so differently
flagged ops are different functions.

Tensor operands are evaluated elementwise and eagerly: the op is UB (stuck) if any element is.
-/
import MerlinVerify.Sem.Support

namespace MerlinVerify.Sem.Ops.Arith

open MerlinVerify

/-! ### Float keys -/

def fastmathBit : String → Nat
  | "reassoc" => 1 | "nnan" => 2 | "ninf" => 4 | "nsz" => 8 | "arcp" => 16 | "contract" => 32
  | "afn" => 64 | "fast" => 127 | _ => 0

def fastmathMask (op : Op) : Nat :=
  match op.attr? "fastmath" with
  | some (.enum "arith" "fastmath" s) => (flagList s).foldl (fun m f => m ||| fastmathBit f) 0
  | _ => 0

/-- The uninterpreted-function key of a float op: name + every numerics-relevant attribute. -/
def floatKey (op : Op) : FloatOpKey :=
  let rm := match attrInt? op "roundingmode" with | some v => toString v | none => "none"
  let pred := match attrInt? op "predicate" with | some v => toString v | none => "none"
  { op := op.name, modifiers := s!"fm={fastmathMask op};rm={rm};pred={pred}" }

/-! ### Scalar helpers -/

def boolVal (b : Bool) : Val := .some (.bits 1 (if b then 1 else 0))

/-- Two same-width integer operands; any poison operand gives poison. -/
def intBin (f : (w : Nat) → BitVec w → BitVec w → Val) : Val → Val → Outcome Val
  | .some (.bits w x), .some (.bits w' y) =>
    if h : w' = w then .ok (f w x (h ▸ y)) else .stuck "integer width mismatch"
  | .some (.flt ..), _ => .stuck "integer op on float operand"
  | _, .some (.flt ..) => .stuck "integer op on float operand"
  | _, _ => .ok .poison

def intUn (f : (w : Nat) → BitVec w → Val) : Val → Outcome Val
  | .some (.bits w x) => .ok (f w x)
  | .some (.flt ..) => .stuck "integer op on float operand"
  | .poison => .ok .poison

def bv (w : Nat) (x : BitVec w) : Val := .some (.bits w x)

def signedOverflows (w : Nat) (r : Int) : Bool :=
  w = 0 || r < -(2 ^ (w - 1) : Int) || r ≥ (2 ^ (w - 1) : Int)

def unsignedOverflows (w : Nat) (r : Int) : Bool := r < 0 || r ≥ (2 ^ w : Int)

/-- add/sub/mul with overflow flags: `exact` computes the mathematical result on ℤ (signed and
unsigned readings) to decide `nsw`/`nuw` poison. -/
def arithWithFlags (flags : List String) (op : Int → Int → Int) (wrap : (w : Nat) → BitVec w → BitVec w → BitVec w)
    (w : Nat) (x y : BitVec w) : Val :=
  if flags.contains "nsw" && signedOverflows w (op x.toInt y.toInt) then .poison
  else if flags.contains "nuw" && unsignedOverflows w (op x.toNat y.toNat) then .poison
  else bv w (wrap w x y)

/-- Division-like op. `signed` additionally makes `INT_MIN / -1` UB. -/
def intDiv (signed : Bool) (f : (w : Nat) → BitVec w → BitVec w → Val) : Val → Val → Outcome Val
  | _, .poison => .ub
  | a, .some (.bits w y) =>
    if y = 0 then .ub else
    match a with
    | .poison => if signed && y = BitVec.allOnes w then .ub else .ok .poison
    | .some (.bits w' x) =>
      if h : w' = w then
        let x := h ▸ x
        if signed && x = BitVec.intMin w && y = BitVec.allOnes w && w > 1 then .ub else .ok (f w x y)
      else .stuck "integer width mismatch"
    | .some (.flt ..) => .stuck "integer op on float operand"
  | _, .some (.flt ..) => .stuck "integer op on float operand"

def floorDivInt (a b : Int) : Int := Int.fdiv a b
def ceilDivInt (a b : Int) : Int := -Int.fdiv (-a) b

/-- Shift with amount in `y` (same width); amount ≥ width is poison. -/
def shift (f : (w : Nat) → BitVec w → Nat → Val) : Val → Val → Outcome Val :=
  intBin fun w x y => if y.toNat ≥ w then .poison else f w x y.toNat

def cmpiPred : Int → Option ((w : Nat) → BitVec w → BitVec w → Bool)
  | 0 => some fun _ x y => x == y
  | 1 => some fun _ x y => x != y
  | 2 => some fun _ x y => x.slt y
  | 3 => some fun _ x y => x.sle y
  | 4 => some fun _ x y => y.slt x
  | 5 => some fun _ x y => y.sle x
  | 6 => some fun _ x y => x.ult y
  | 7 => some fun _ x y => x.ule y
  | 8 => some fun _ x y => y.ult x
  | 9 => some fun _ x y => y.ule x
  | _ => none

/-! ### Float helpers (uninterpreted) -/

def fltBin (M : FloatModel) (key : FloatOpKey) : Val → Val → Outcome Val
  | .some (.flt k x), .some (.flt k' y) =>
    if h : k' = k then .ok (.some (.flt k (M.op key k [x, h ▸ y]))) else .stuck "float kind mismatch"
  | .some (.bits ..), _ => .stuck "float op on integer operand"
  | _, .some (.bits ..) => .stuck "float op on integer operand"
  | _, _ => .ok .poison

/-- N-ary same-kind float op (math.*), any poison → poison. -/
def fltN (M : FloatModel) (key : FloatOpKey) (vs : List Val) : Outcome Val := do
  if vs.contains .poison then return .poison
  let some k := vs.head?.bind (fun | .some (.flt k _) => some k | _ => none)
    | .stuck "float op without float operands"
  let xs ← vs.mapM fun
    | .some (.flt k' x) => if h : k' = k then .ok (h ▸ x) else .stuck "float kind mismatch"
    | _ => .stuck "float op on non-float operand"
  pure (.some (.flt k (M.op key k xs)))

/-! ### The family -/

def scalarArgs (args : List RVal) : Option (List Val) :=
  args.mapM fun | .scalar v => some v | _ => none

/-- Element (or scalar) result type of result 0. -/
def resultElt (op : Op) : Option Ty :=
  match op.results with
  | r :: _ => match r.ty with
    | .tensor _ elt _ => some elt
    | t => some t
  | [] => none

/-- Apply a scalar op to scalars, or elementwise (eagerly) to equal-shape tensors. -/
def pureN (op : Op) (args : List RVal) (f : List Val → Outcome Val) : Outcome (List RVal) := do
  let some elt := resultElt op | .stuck "missing result type"
  let r ← liftEager elt f args
  pure [r]

def widthOf (indexBits : Nat) (op : Op) : Outcome Nat :=
  match resultElt op with
  | some ty => Outcome.ofOption "non-scalar result type" (ty.bitWidth? indexBits)
  | none => .stuck "missing result type"

def sem : OpSem := fun ctx _ op args _ =>
  let M := ctx.M
  let ib := ctx.indexBits
  let flags := overflowFlags op
  let exact := hasUnit op "isExact"
  let un (f : Val → Outcome Val) : Outcome (List RVal) := do
    let [a] := args | .stuck s!"{op.name}: expected 1 operand"
    pureN op [a] fun | [x] => f x | _ => .stuck "arity"
  let bin (f : Val → Val → Outcome Val) : Outcome (List RVal) := do
    let [a, b] := args | .stuck s!"{op.name}: expected 2 operands"
    pureN op [a, b] fun | [x, y] => f x y | _ => .stuck "arity"
  match op.name with
  | "arith.constant" =>
    match op.attr? "value", op.results with
    | some (.dense ty splat bytes), [_] => do
      let t ← denseTensor ib ty splat bytes
      pure [.tensor t]
    | some a, [_] =>
      match scalarOfAttr ib a with
      | some s => .ok [.scalar (.some s)]
      | none => .stuck "unsupported constant attribute"
    | _, _ => .stuck "malformed arith.constant"
  -- integer arithmetic with overflow flags
  | "arith.addi" => bin (intBin (arithWithFlags flags (· + ·) (fun _ x y => x + y)))
  | "arith.subi" => bin (intBin (arithWithFlags flags (· - ·) (fun _ x y => x - y)))
  | "arith.muli" => bin (intBin (arithWithFlags flags (· * ·) (fun _ x y => x * y)))
  | "arith.andi" => bin (intBin fun w x y => bv w (x &&& y))
  | "arith.ori" => bin (intBin fun w x y => bv w (x ||| y))
  | "arith.xori" => bin (intBin fun w x y => bv w (x ^^^ y))
  | "arith.maxsi" => bin (intBin fun w x y => bv w (if x.slt y then y else x))
  | "arith.minsi" => bin (intBin fun w x y => bv w (if x.slt y then x else y))
  | "arith.maxui" => bin (intBin fun w x y => bv w (if x.ult y then y else x))
  | "arith.minui" => bin (intBin fun w x y => bv w (if x.ult y then x else y))
  -- division family (UB-capable)
  | "arith.divui" => bin (intDiv false fun w x y =>
      if exact && x.umod y ≠ 0 then .poison else bv w (x.udiv y))
  | "arith.divsi" => bin (intDiv true fun w x y =>
      if exact && x.srem y ≠ 0 then .poison else bv w (x.sdiv y))
  | "arith.remui" => bin (intDiv false fun w x y => bv w (x.umod y))
  | "arith.remsi" =>
    -- `INT_MIN % -1` is defined (0) here: the signed-overflow UB applies only to division.
    bin fun a b => match b with
      | .some (.bits _ y) => if y = BitVec.allOnes _ then
          intBin (fun w x _ => bv w (BitVec.ofInt w (x.toInt.tmod (-1)))) a b
        else intDiv false (fun w x y => bv w (x.srem y)) a b
      | _ => intDiv false (fun w x y => bv w (x.srem y)) a b
  | "arith.ceildivui" => bin (intDiv false fun w x y =>
      bv w (BitVec.ofNat w ((x.toNat + y.toNat - 1) / y.toNat)))
  | "arith.ceildivsi" => bin (intDiv true fun w x y =>
      bv w (BitVec.ofInt w (ceilDivInt x.toInt y.toInt)))
  | "arith.floordivsi" => bin (intDiv true fun w x y =>
      bv w (BitVec.ofInt w (floorDivInt x.toInt y.toInt)))
  -- shifts
  | "arith.shli" => bin (shift fun w x n =>
      let r := x <<< n
      if flags.contains "nuw" && (r >>> n) ≠ x then .poison
      else if flags.contains "nsw" && (r.sshiftRight n) ≠ x then .poison
      else bv w r)
  | "arith.shrui" => bin (shift fun w x n =>
      if exact && (x >>> n) <<< n ≠ x then .poison else bv w (x >>> n))
  | "arith.shrsi" => bin (shift fun w x n =>
      if exact && (x.sshiftRight n) <<< n ≠ x then .poison else bv w (x.sshiftRight n))
  -- comparison and selection
  | "arith.cmpi" =>
    match attrInt? op "predicate" >>= cmpiPred with
    | some p => bin (intBin fun w x y => boolVal (p w x y))
    | none => .stuck "arith.cmpi: unknown predicate"
  | "arith.select" =>
    match args with
    | [.scalar c, t, f] =>
      match c with
      | .poison => match t with
        | .scalar _ => .ok [.scalar .poison]
        | .tensor tt => .ok [.tensor (Tensor.full tt.shape tt.elt .poison)]
      | .some (.bits 1 b) => .ok [(if b = 1 then t else f).canonize]
      | _ => .stuck "arith.select: condition is not i1"
    | [.tensor c, .tensor t, .tensor f] =>
      if c.shape == t.shape && t.shape == f.shape then
        .ok [.tensor (Tensor.canon t.shape t.elt fun i =>
          match c.get i with
          | .some (.bits 1 b) => if b = 1 then t.get i else f.get i
          | _ => .poison)]
      else .stuck "arith.select: shape mismatch"
    | _ => .stuck "arith.select: malformed operands"
  -- integer casts
  | "arith.extsi" => do
    let w' ← widthOf ib op
    un (intUn fun w x => if w' ≤ w then .poison else bv w' (x.signExtend w'))
  | "arith.extui" => do
    let w' ← widthOf ib op
    let nneg := hasUnit op "nonNeg"
    un (intUn fun w x => if w' ≤ w then .poison
      else if nneg && x.msb then .poison else bv w' (x.setWidth w'))
  | "arith.trunci" => do
    let w' ← widthOf ib op
    un (intUn fun w x =>
      if w ≤ w' then .poison else
      let r := x.setWidth w'
      if flags.contains "nuw" && r.setWidth w ≠ x then .poison
      else if flags.contains "nsw" && r.signExtend w ≠ x then .poison
      else bv w' r)
  | "arith.index_cast" => do
    let w' ← widthOf ib op
    un (intUn fun w x => bv w' (if w' ≤ w then x.setWidth w' else x.signExtend w'))
  | "arith.index_castui" => do
    let w' ← widthOf ib op
    let nneg := hasUnit op "nonNeg"
    un (intUn fun w x => if nneg && x.msb then .poison else bv w' (x.setWidth w'))
  | "arith.bitcast" =>
    match resultElt op with
    | some (.float k) => un fun
      | .some (.bits w x) => if h : w = k.width then .ok (.some (.flt k (h ▸ x)))
          else .stuck "bitcast width mismatch"
      | .poison => .ok .poison
      | _ => .stuck "bitcast operand"
    | some (.int w') => un fun
      | .some (.flt k x) => if h : k.width = w' then .ok (bv w' (h ▸ x))
          else .stuck "bitcast width mismatch"
      | .some (.bits w x) => if h : w = w' then .ok (bv w' (h ▸ x)) else .stuck "bitcast width mismatch"
      | .poison => .ok .poison
    | _ => .stuck "bitcast result type"
  -- floats (uninterpreted)
  | "arith.addf" | "arith.subf" | "arith.mulf" | "arith.divf" | "arith.remf"
  | "arith.maximumf" | "arith.minimumf" | "arith.maxnumf" | "arith.minnumf" =>
    bin (fltBin M (floatKey op))
  | "arith.negf" => un fun v => fltN M (floatKey op) [v]
  | "arith.cmpf" =>
    bin fun a b => match a, b with
      | .some (.flt k x), .some (.flt k' y) =>
        if h : k' = k then .ok (boolVal (M.cmp (floatKey op) k x (h ▸ y))) else .stuck "kind mismatch"
      | .some (.bits ..), _ => .stuck "cmpf on integer"
      | _, .some (.bits ..) => .stuck "cmpf on integer"
      | _, _ => .ok .poison
  | "arith.extf" | "arith.truncf" =>
    match resultElt op with
    | some (.float k') => un fun
      | .some (.flt k x) => .ok (.some (.flt k' (M.convert (floatKey op) k k' x)))
      | .poison => .ok .poison
      | _ => .stuck "float conversion of integer"
    | _ => .stuck "float conversion result type"
  | "arith.fptosi" | "arith.fptoui" => do
    let w' ← widthOf ib op
    un fun
      | .some (.flt k x) => .ok (match M.toInt (floatKey op) k w' x with
          | some r => bv w' r
          | none => .poison)
      | .poison => .ok .poison
      | _ => .stuck "fpto*i of integer"
  | "arith.sitofp" | "arith.uitofp" =>
    match resultElt op with
    | some (.float k) =>
      let nneg := op.name == "arith.uitofp" && hasUnit op "nonNeg"
      un fun
        | .some (.bits w x) => .ok (if nneg && x.msb then .poison
            else .some (.flt k (M.ofInt (floatKey op) w k x)))
        | .poison => .ok .poison
        | _ => .stuck "*itofp of float"
    | _ => .stuck "*itofp result type"
  | name =>
    if name.startsWith "math." then
      match resultElt op with
      | some (.float _) => pureN op args (fltN M (floatKey op))
      | some (.int _) =>
        if name == "math.absi" then un (intUn fun w x => bv w (if x.msb then -x else x))
        else .stuck s!"unsupported integer math op {name}"
      | _ => .stuck s!"unsupported math op {name}"
    else .stuck s!"unsupported op {name}"

end MerlinVerify.Sem.Ops.Arith
