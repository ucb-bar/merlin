/- Execute the actual constructed base, not an unrelated arithmetic implementation. -/
import MerlinVerify.Driver.W8A8IEEE

open Lean MerlinVerify MerlinVerify.Rules

abbrev model := W8A8IEEEPrimitives.model W8A8IEEEPrimitives.unsupported
abbrev key := Spec.Dequant.defaultKey

def compute (operation : String) (values : List Nat) : Except String Nat := do
  let width := if ["narrow", "reciprocal64", "signed64"].contains operation then 64 else 32
  unless values.all (· < 2^width) do throw "primitive input bits out of range"
  match operation, values with
  | "widen", [x] => return (model.convert (key "arith.extf") .f32 .f64 (BitVec.ofNat 32 x)).toNat
  | "narrow", [x] => return (model.convert (key "arith.truncf") .f64 .f32 (BitVec.ofNat 64 x)).toNat
  | "reciprocal64", [x] => return (model.op (key "arith.divf") .f64
      [0x3ff0000000000000, BitVec.ofNat 64 x]).toNat
  | "signed32", [x] => return (model.ofInt (key "arith.sitofp") 32 .f32 (BitVec.ofNat 32 x)).toNat
  | "signed64", [x] => return (model.ofInt (key "arith.sitofp") 64 .f32 (BitVec.ofNat 64 x)).toNat
  | "mul", [x, y] => return (model.op (key "arith.mulf") .f32 [BitVec.ofNat 32 x, BitVec.ofNat 32 y]).toNat
  | "add", [x, y] => return (model.op (key "arith.addf") .f32 [BitVec.ofNat 32 x, BitVec.ofNat 32 y]).toNat
  | "div", [x, y] => return (model.op (key "arith.divf") .f32 [BitVec.ofNat 32 x, BitVec.ofNat 32 y]).toNat
  | "rsqrt", [x] => return (model.op (key "math.rsqrt") .f32 [BitVec.ofNat 32 x]).toNat
  | "abs", [x] => return (model.op (key "math.absf") .f32 [BitVec.ofNat 32 x]).toNat
  | "roundeven", [x] => return (model.op (key "math.roundeven") .f32 [BitVec.ofNat 32 x]).toNat
  | _, _ => throw "unsupported primitive or arity"

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let request := Json.parse (← IO.FS.readFile path)
  let result := request.bind fun request => do
    Decode.exactKeys "explicit-model oracle" request ["operation", "samples"] []
    let operation ← (← request.getObjVal? "operation").getStr?
    let rows ← (← (← request.getObjVal? "samples").getArr?).toList.mapM fun row => do
      compute operation (← (← row.getArr?).toList.mapM Json.getNat?)
    return toJson rows
  match result with
  | .ok value => IO.println value.compress; return 0
  | .error error => IO.eprintln error; return 2
