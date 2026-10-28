/- Report the proved interpretation bridge. Equality can include poison, UB and stuck outcomes;
it is independent of support/definedness and cannot issue a verified-pass receipt. -/
import MerlinVerify.Core.Json
import MerlinVerify.Rules.W8A8SourceFree

open Lean MerlinVerify MerlinVerify.Rules

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok source := Decode.module (← IO.FS.readFile path) | return 2
  let checked := W8A8SourceFree.moduleCheck source
  let functions := source.funcs.map fun fn => Json.mkObj [
    ("function", toJson fn.name), ("source_free", toJson (W8A8SourceFree.region fn.body))]
  IO.println (Json.mkObj [("scope", toJson "provisional_frozen_interpretation_agreement"),
    ("interpretations_agree", toJson checked), ("functions", toJson functions),
    ("whole_pass_verified", toJson false), ("initialized_execution_verified", toJson false),
    ("theorem", toJson "MerlinVerify.Rules.W8A8SourceFree.denote_eq")]).compress
  return if checked then 0 else 1
