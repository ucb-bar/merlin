import MerlinVerify.Driver.W8A8Numerical

open Lean MerlinVerify MerlinVerify.Rules

def compute (operation : String) (values : List Nat) : Except String (FBits .f32) := do
  unless values.all (· < 2 ^ 32) do throw "f32 bits out of range"
  match operation, values.map (BitVec.ofNat 32) with
  | "mul", [x, y] => return W8A8IEEE32.mul x y
  | "add", [x, y] => return W8A8IEEE32.add x y
  | "div", [x, y] => return W8A8IEEE32.div x y
  | "rsqrt", [x] => return W8A8IEEE32.rsqrt x
  | "inverse", [x] => return W8A8ArithmeticModel.inverse x
  | "dequant", [q, lo, hi, scale] =>
    return W8A8DequantNumerics.scalar (BitVec.ofNat 8 q.toNat)
      (BitVec.ofNat 64 (lo.toNat + hi.toNat * 2 ^ 32)) scale
  | _, _ => throw "unsupported operation or arity"

def main (paths : List String) : IO UInt32 := do
  let [path] := paths | return 2
  let request := Json.parse (← IO.FS.readFile path)
  let answer := request.bind fun request => do
    Decode.exactKeys "numerical oracle" request ["operation", "samples"] []
    let operation ← (← request.getObjVal? "operation").getStr?
    let rows ← (← (← request.getObjVal? "samples").getArr?).toList.mapM fun row => do
      let values ← (← row.getArr?).toList.mapM Json.getNat?
      return (← compute operation values).toNat
    return toJson rows
  match answer with
  | .ok result => IO.println result.compress; return 0
  | .error error => IO.eprintln error; return 2
