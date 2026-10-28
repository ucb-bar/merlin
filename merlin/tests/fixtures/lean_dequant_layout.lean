/- Test adapter for layout only: no C1 or whole-pass verdict is issued here. -/
import MerlinVerify.Core.Json
import MerlinVerify.Rules.DequantTensor

open Lean MerlinVerify MerlinVerify.Sem.Ops.Linalg

def main (args : List String) : IO UInt32 := do
  let [path, kind, axisText] := args | return 2
  if kind != "per_tensor" && kind != "per_channel" then return 2
  let some axis := axisText.toNat? | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let some f := m.func? "forward" | return 2
  let .mk [.mk [q, s, z] ops] := f.body | return 2
  let some generic := ops.find? (fun op => op.name == "linalg.generic") | return 2
  let [r] := generic.results | return 2
  let some (shape, .int 8) := q.ty.staticTensor? | return 2
  let some (paramShape, .float .f32) := s.ty.staticTensor? | return 2
  let some (zeroShape, .int 32) := z.ty.staticTensor? | return 2
  let some (outShape, .float .f32) := r.ty.staticTensor? | return 2
  let some maps := attrAffineMaps? generic "indexing_maps" | return 2
  let some iters := iterKinds? generic | return 2
  let accepted := shape == outShape && paramShape == zeroShape &&
    Rules.DequantTensor.checkLayout (kind == "per_channel") axis shape paramShape maps iters
  IO.println (Json.mkObj [("scope", toJson "tensor_layout_only"), ("matches", toJson accepted)]).compress
  return if accepted then 0 else 1
