/- Inspect calibration at actual top-level sites using the proved complete-prefix traversal.
The report remains diagnostic: successful execution and numerical domains are not inferred. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8SourcePrefix

open Lean MerlinVerify MerlinVerify.Check

def inspectBlock (args : List Value) (ops : List Op) : List Json := Id.run do
  let mut types := args.map fun value => (value.id, value.ty)
  let mut facts : List ConstantFacts.Fact := []
  let mut sites := []
  for actual in ops do
    if actual.name.startsWith "quant_ext." then
      sites := sites ++ [Json.mkObj [("op", toJson actual.name),
        ("results", toJson (actual.results.map (·.id))),
        ("calibration_parameters_checked", toJson
          (PolicyW8A8SourcePrefix.checkAt 64 types facts [] actual))]]
    types := actual.results.map (fun value => (value.id, value.ty)) ++ types
    facts := PolicyW8A8SourceFacts.advance 64 facts actual
  return sites

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok source := Decode.module (← IO.FS.readFile path) | return 2
  let sites := source.ops.flatMap fun op => match op.asFunc? with
    | some fn => match fn.body with
      | .mk [.mk inputs operations] => inspectBlock inputs operations
      | _ => []
    | none => []
  let segments := source.ops.flatMap fun op => match op.asFunc? with
      | some fn => match fn.body with
        | .mk [.mk inputs operations] => [PolicyW8A8SourcePrefix.checkOps 64
            (inputs.map fun value => (value.id, value.ty)) [] operations]
        | _ => []
      | none => []
  IO.println (Json.mkObj [("scope", toJson "source_prefix_calibration_only"),
    ("whole_pass_verified", toJson false), ("sites", toJson sites),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8SourcePrefix.checkAt_sound"),
    ("segments", toJson segments)]).compress
  return 0
