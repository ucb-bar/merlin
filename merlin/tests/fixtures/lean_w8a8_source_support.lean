/- Whole-function typed support under the provisional source interpretation. This is distinct
from initialized execution, numerical-domain qualification and a full verified-pass receipt. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Support
import MerlinVerify.Check.TypedModule

open Lean MerlinVerify MerlinVerify.Check

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok source := Decode.module (← IO.FS.readFile path) | return 2
  let names := Module.definedNames source
  let results := names.map fun fn =>
    let supported := match TypedModule.signature? 64 source fn with
      | none => false
      | some (inputs, outputs) => PolicyW8A8Support.checkFunction 64 source fn inputs outputs
    (supported, Json.mkObj [("function", toJson fn), ("supported", toJson supported)])
  let supported := !names.isEmpty && results.all Prod.fst
  IO.println (Json.mkObj [("scope", toJson "provisional_source_typed_support"),
    ("whole_pass_verified", toJson false), ("source_defined", toJson false),
    ("supported", toJson supported), ("functions", toJson (results.map Prod.snd)),
    ("domain", toJson "all_declared_typed_inputs_poison_permitted"),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8Support.checkFunction_sound")]).compress
  return if supported then 0 else 1
