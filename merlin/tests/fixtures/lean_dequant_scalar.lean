/- Test adapter for the proved scalar-body checker; not a whole-pass receipt generator. -/
import MerlinVerify.Core.Json
import MerlinVerify.Rules.DequantScalar

open Lean MerlinVerify

def main (args : List String) : IO UInt32 := do
  let [path, kind] := args | return 2
  if kind != "per_tensor" && kind != "per_channel" then return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let some f := m.func? "forward" | return 2
  let .mk [.mk _ ops] := f.body | return 2
  let some generic := ops.find? (fun op => op.name == "linalg.generic") | return 2
  let [.mk [.mk blockArgs body]] := generic.regions | return 2
  let accepted := Rules.DequantScalar.checkBody (kind == "per_channel") blockArgs body
  IO.println (Json.mkObj [("scope", toJson "scalar_body_only"), ("matches", toJson accepted)]).compress
  return if accepted then 0 else 1
