/-
Executable fidelity-test adapter. This imports the actual denotation used by the proofs; it does
not reimplement integer, tensor, or structured-op semantics. The host Float/Float32 instance below
is a TEST ORACLE, outside MerlinVerify's verified namespace, not an added axiom or a proved IEEE
instance. Unsupported float operations and modifiers are rejected before evaluation.

Run with `lake env lean --run Differential.lean <request.json>` from merlin/lean. No shared checker
driver or Lake configuration needs to change to use this adapter.
-/
import MerlinVerify.Core.Json
import MerlinVerify.Sem.Ops
import MerlinVerify.Rules.FrontendMaskRuntime
import MerlinVerify.Rules.FrontendF32Comparison

open MerlinVerify Lean

namespace MerlinDifferential

def floatOps : List String :=
  ["arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf", "arith.cmpf",
   "arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui",
   "math.absf", "math.sqrt", "math.ceil", "math.floor", "math.round", "math.roundeven", "math.trunc",
   "math.fma"]

def usesFloatModel (n : String) : Bool :=
  (n.startsWith "math." && n != "math.absi") || floatOps.contains n ||
    ["arith.remf", "arith.maximumf", "arith.minimumf", "arith.maxnumf", "arith.minnumf"].contains n

-- The native observation ABI is limited to 64 bits, but the proved integer
-- denotation supports wider intermediates without a host integer conversion.
def internalScalarType : Ty → Bool
  | .int w => 0 < w
  | .index | .float .f32 | .float .f64 => true
  | _ => false

def internalValueType : Ty → Bool
  | .tensor sh elt none => sh.all Option.isSome && internalScalarType elt
  | ty => internalScalarType ty

mutual
  def validateOps (types : List (ValueId × Ty)) : List Op → Except String Unit
    | [] => pure ()
    | op@(.mk _ results _ _ _ regions _) :: rest => do
      unless results.all (fun r => internalValueType r.ty) do throw "unsupported result type in fidelity adapter"
      if usesFloatModel op.name then
        unless floatOps.contains op.name do throw s!"test float model does not implement {op.name}"
        let arity := if op.name == "math.fma" then 3
          else if ["arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.cmpf"].contains op.name then 2
          else 1
        unless op.operands.length == arity do throw "unsupported float operation arity"
        if op.name == "arith.cmpf" then
          match op.attr? "predicate" with
          | some (.int p (.int 64)) => unless 0 <= p && p <= 15 do throw "invalid cmpf predicate"
          | _ => throw "test cmpf model requires MLIR's i64 predicate attribute"
        if (op.attr? "roundingmode").isSome then throw "test float model requires default rounding"
        match op.attr? "fastmath" with
        | none => pure ()
        | some (.enum "arith" "fastmath" "none") => pure ()
        | _ => throw "test float model requires fastmath=none"
        if op.name == "arith.fptosi" || op.name == "arith.fptoui" then
          unless results.all (fun r => (r.ty.bitWidth? 64).getD 65 <= 32) do
            throw "test float-to-int model supports results up to 32 bits"
        if op.name == "arith.sitofp" || op.name == "arith.uitofp" then
          -- hostModel.ofInt uses UInt64. Check the actual operand's SSA type,
          -- including captures, before permitting that host conversion.
          let supported := match op.operands with
            | [id] => match types.lookup id with
              | some (.int w) | some (.tensor _ (.int w) none) => 0 < w && w <= 64
              | _ => false
            | _ => false
          unless supported do throw "test int-to-float model requires a 1..64-bit integer input"
      validateRegions types regions
      validateOps (results.map (fun r => (r.id, r.ty)) ++ types) rest
  def validateRegions (types : List (ValueId × Ty)) : List Region → Except String Unit
    | [] => pure ()
    | .mk bs :: rs => do
      validateBlocks types bs
      validateRegions types rs
  def validateBlocks (types : List (ValueId × Ty)) : List Block → Except String Unit
    | [] => pure ()
    | .mk args ops :: bs => do
      unless args.all (fun a => internalValueType a.ty) do throw "unsupported block argument type in fidelity adapter"
      validateOps (args.map (fun a => (a.id, a.ty)) ++ types) ops
      validateBlocks types bs
end

def roundEven (x : Float) : Float := Id.run do
  if x.isNaN || x.isInf || x.abs >= 4503599627370496.0 then return x
  let lo := x.floor
  let delta := x - lo
  let r := if delta < 0.5 then lo else if delta > 0.5 then lo + 1.0
    else if lo.abs.toUInt64 % 2 == 0 then lo else lo + 1.0
  -- Preserve negative zero, including rounding negative fractions to zero.
  return if r == 0.0 then Float.ofBits (x.toBits &&& 0x8000000000000000) else r

def trunc (x : Float) : Float := if x < 0.0 then x.ceil else x.floor

def op64 (n : String) (xs : List Float) : Float :=
  match n, xs with
  | "arith.addf", [x, y] => x + y
  | "arith.subf", [x, y] => x - y
  | "arith.mulf", [x, y] => x * y
  | "arith.divf", [x, y] => x / y
  | "arith.negf", [x] => -x
  | "math.absf", [x] => x.abs
  | "math.sqrt", [x] => x.sqrt
  | "math.ceil", [x] => x.ceil
  | "math.floor", [x] => x.floor
  | "math.round", [x] => x.round
  | "math.roundeven", [x] => roundEven x
  | "math.trunc", [x] => trunc x
  | "math.fma", [x, y, z] => x.fma y z
  | _, _ => 0.0 -- validateOps and the corpus restrict names and arities.

def op32 (n : String) (xs : List Float32) : Float32 :=
  match n, xs with
  | "arith.addf", [x, y] => x + y
  | "arith.subf", [x, y] => x - y
  | "arith.mulf", [x, y] => x * y
  | "arith.divf", [x, y] => x / y
  | "arith.negf", [x] => -x
  | "math.absf", [x] => x.abs
  | "math.sqrt", [x] => x.sqrt
  | "math.ceil", [x] => x.ceil
  | "math.floor", [x] => x.floor
  | "math.round", [x] => x.round
  | "math.roundeven", [x] => (roundEven x.toFloat).toFloat32
  | "math.trunc", [x] => (trunc x.toFloat).toFloat32
  | "math.fma", [x, y, z] => x.fma y z
  | _, _ => 0.0

def toHost (k : FKind) (x : FBits k) : Float :=
  match k with
  | .f32 => (Float32.ofBits x.toNat.toUInt32).toFloat
  | .f64 => Float.ofBits x.toNat.toUInt64
  | _ => 0.0

def ofHost (k : FKind) (x : Float) : FBits k :=
  match k with
  | .f32 => BitVec.ofNat 32 x.toFloat32.toBits.toNat
  | .f64 => BitVec.ofNat 64 x.toBits.toNat
  | _ => 0

def cmp (key : FloatOpKey) (x y : Float) : Bool := Id.run do
  -- Predicate is part of the exact float key; parse the decimal IntegerAttr literal. The request
  -- is exported by MLIR, whose cmpf predicate is an i64 attribute (0..15).
  let part := (key.modifiers.splitOn ";pred").getLast!
  let literal := (part.splitOn "attr ").getLast!
  let p := ((literal.splitOn " ").headD "").toNat?.getD 99
  let unordered := x.isNaN || y.isNaN
  return match p with
    | 0 => false
    | 1 => !unordered && x == y
    | 2 => !unordered && x > y
    | 3 => !unordered && x >= y
    | 4 => !unordered && x < y
    | 5 => !unordered && x <= y
    | 6 => !unordered && x != y
    | 7 => !unordered
    | 8 => unordered || x == y
    | 9 => unordered || x > y
    | 10 => unordered || x >= y
    | 11 => unordered || x < y
    | 12 => unordered || x <= y
    | 13 => unordered || x != y
    | 14 => unordered
    | 15 => true
    | _ => false

def hostModel : FloatModel where
  op key k xs := match k with
    | .f32 => BitVec.ofNat 32 (op32 key.op (xs.map fun x => Float32.ofBits x.toNat.toUInt32)).toBits.toNat
    | .f64 => BitVec.ofNat 64 (op64 key.op (xs.map fun x => Float.ofBits x.toNat.toUInt64)).toBits.toNat
    | _ => 0
  cmp key k x y := cmp key (toHost k x) (toHost k y)
  convert _ src dst x := ofHost dst (toHost src x)
  toInt key k w x :=
    let f := toHost k x
    if f.isNaN || f.isInf || w > 32 then none else
    let magnitude := (trunc f).abs.toUInt64.toNat
    let z : Int := if f < 0.0 then -(Int.ofNat magnitude) else Int.ofNat magnitude
    let signed := key.op == "arith.fptosi"
    let lower : Int := if signed then -(Int.ofNat (2 ^ (w - 1))) else 0
    let upper : Int := Int.ofNat (if signed then 2 ^ (w - 1) else 2 ^ w)
    if lower <= z && z < upper then some (BitVec.ofInt w z) else none
  ofInt key _w k x :=
    let z := if key.op == "arith.sitofp" then x.toInt else Int.ofNat x.toNat
    match k with
    | .f32 =>
      let f := z.natAbs.toUInt64.toFloat32
      BitVec.ofNat 32 (if z < 0 then -f else f).toBits.toNat
    | .f64 =>
      let f := z.natAbs.toUInt64.toFloat
      BitVec.ofNat 64 (if z < 0 then -f else f).toBits.toNat
    | _ => 0

def scalar (ib : Nat) (ty : Ty) (bits : String) : Except String Val := do
  let some n := bits.toNat? | throw "input bits must be unsigned decimal strings"
  let some w := ty.bitWidth? ib | throw "unsupported input element type"
  unless 0 < w && w <= 64 && n < 2 ^ w do throw "input bits exceed element width (1..64)"
  let some s := scalarOfBits ib ty n | throw "input element decode failed"
  return .some s

def input (ib : Nat) (ty : Ty) (j : Json) : Except String RVal := do
  let raw <- Decode.arr j
  match ty with
  | .tensor dims elt none =>
    let some shape := dims.mapM id | throw "dynamic tensors are outside the fidelity adapter"
    unless raw.length == shapeSize shape do throw "wrong tensor input element count"
    let vs <- raw.mapM fun x => do scalar ib elt (← Decode.str x)
    return .tensor (Tensor.canon shape elt fun i => vs.toArray[linearIndex shape i]?.getD .poison)
  | .tensor .. => throw "encoded tensors are outside the fidelity adapter"
  | _ =>
    let [x] := raw | throw "scalar input must contain exactly one element"
    return .scalar (← scalar ib ty (← Decode.str x))

def valBits : Val → Json
  | .poison => .null
  | .some (.bits _ x) => toJson (toString x.toNat)
  | .some (.flt _ x) => toJson (toString x.toNat)

def output : RVal → Json
  | .scalar v => Json.mkObj [("shape", .null), ("bits", toJson [valBits v])]
  | .tensor t => Json.mkObj [("shape", toJson t.shape),
      ("bits", toJson ((allPoints t.shape).map fun i => valBits (t.get i)))]

def outcome : Outcome (List RVal) → Json
  | .ok vs => Json.mkObj [("status", toJson "ok"), ("values", toJson (vs.map output))]
  | .ub => Json.mkObj [("status", toJson "ub")]
  | .stuck why => Json.mkObj [("status", toJson "stuck"), ("reason", toJson why)]

def run (j : Json) : Except String Json := do
  let m <- Decode.module (← Decode.str (← Decode.field j "export"))
  validateOps [] m.ops
  let fn <- Decode.str (← Decode.field j "function")
  let ib <- Decode.nat (← Decode.field j "index_bits")
  unless ib == 32 || ib == 64 do throw "fidelity adapter supports 32- or 64-bit index"
  let some op := m.ops.find? (fun o => (o.asFunc?.map (·.name)) == some fn) | throw "function missing"
  let some (.type (.function tys _)) := op.attr? "function_type" | throw "function signature missing"
  let samples <- Decode.arr (← Decode.field j "samples")
  let candidate := match j.getObjVal? "frontend_runtime" with | .ok (.bool b) => b | _ => false
  let limit ← if candidate then Decode.nat (← Decode.field j "limit") else pure 0
  let model := if candidate then MerlinVerify.Rules.FrontendF32Comparison.model hostModel else hostModel
  let sem := if candidate then MerlinVerify.Rules.FrontendMaskRuntime.sem limit else Sem.Ops.opSem
  let ctx : Ctx := { M := model, ext := fun _ _ => .stuck "no external oracle", indexBits := ib }
  let results <- samples.mapM fun sample => do
    let args <- Decode.arr sample
    unless args.length == tys.length do throw "wrong input argument count"
    let vs <- (tys.zip args).mapM fun (ty, arg) => input ib ty arg
    return outcome (MerlinVerify.denote sem ctx m fn vs)
  return toJson results

end MerlinDifferential

def main (args : List String) : IO UInt32 := do
  let [path] := args | do IO.eprintln "usage: Differential.lean <request.json>"; return 2
  let text ← IO.FS.readFile path
  match Json.parse text >>= MerlinDifferential.run with
  | .ok result => IO.println result.compress; return 0
  | .error reason => IO.eprintln reason; return 2
