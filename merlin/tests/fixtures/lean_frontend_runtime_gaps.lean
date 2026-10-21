/- Actual-export qualification of frozen-runtime refusals. No runtime extension is installed. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Rules.FrontendBoolRuntimeGaps

open MerlinVerify Lean
open MerlinVerify.Rules.FrontendBoolRuntimeGaps

private def witnessJson (pair : Gap × Op) : Json :=
  Json.mkObj [
    ("operation", toJson pair.2.name),
    ("operands", toJson pair.2.operands),
    ("results", toJson (pair.2.results.map (·.id))),
    ("result_types", toJson (pair.2.results.map fun v => Print.ty v.ty)),
    ("reason", toJson pair.1.reason)]

def main (args : List String) : IO UInt32 := do
  let [path, name] := args | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let witnesses := collectOps m.ops
  IO.println (Json.mkObj [
    ("function", toJson name),
    ("successful_execution", toJson (if neverSucceeds m name then "proved impossible" else "unmeasured")),
    ("success_theorem", toJson "MerlinVerify.Rules.FrontendBoolRuntimeGaps.neverSucceeds_sound"),
    ("witness_theorem", toJson "MerlinVerify.Rules.FrontendBoolRuntimeGaps.collected_stuck"),
    ("witnesses", Json.arr (witnesses.map witnessJson).toArray),
    ("scope", toJson "Frozen-model refusals. Each witness is stuck if stepped; unconditional block scan proves no ok result. This does not assert every run reaches a witness: earlier UB is possible.")]).compress
  return 0
