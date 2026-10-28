/- Diagnostic scalar-component adapter, not a W8A8 whole-boundary receipt generator. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Scalar
import MerlinVerify.Check.PolicyW8A8Body

open Lean MerlinVerify
open MerlinVerify.Check.PolicyW8A8Scalar

def main (args : List String) : IO UInt32 := do
  let path :: rest := args | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let some f := m.func? "forward" | return 2
  let .mk [.mk _ ops] := f.body | return 2
  if rest == ["inspect"] then
    let mut sites : Array Json := #[]
    let mut refused := 0
    for (op, index) in ops.zipIdx do
      let some (.str role) := op.attr? "prov.role" | continue
      if role != "contraction" && role != "requant" then continue
      let [region] := op.regions | return 2
      let .mk [.mk blockArgs _] := region | return 2
      let accepted := if role == "contraction" then Check.PolicyW8A8Body.check region
        else if blockArgs.length ≥ 2 then checkRequantize (blockArgs.length - 2) region else false
      if !accepted then refused := refused + 1
      sites := sites.push (Json.mkObj [("index", toJson index), ("role_hint", toJson role),
        ("body_matches", toJson accepted)])
    IO.println (Json.mkObj [("scope", toJson "hinted_integer_and_requant_bodies_only"),
      ("whole_pass_verified", toJson false), ("refused", toJson refused), ("sites", toJson sites)]).compress
    return if refused == 0 then 0 else 1
  let selected :: kind :: captures := rest | return 2
  let some index := selected.toNat? | return 2
  let some ids := captures.mapM String.toNat? | return 2
  let some op := ops[index]? | return 2
  if op.name != "linalg.generic" then return 2
  let [region] := op.regions | return 2
  let accepted := match kind, ids with
    | "absmax", [] => checkAbsMax region
    | "reciprocal", [] => checkReciprocal region
    | "scale", [denom] => checkScale 64 (ops.take index) denom region
    | "quantize", [hi, lo] => checkQuantize 64 (ops.take index) hi lo region
    | "requantize0", [] => checkRequantize 0 region
    | "requantize1", [] => checkRequantize 1 region
    | "requantize2", [] => checkRequantize 2 region
    | _, _ => false
  IO.println (Json.mkObj [("scope", toJson "w8a8_scalar_only"), ("matches", toJson accepted)]).compress
  return if accepted then 0 else 1
