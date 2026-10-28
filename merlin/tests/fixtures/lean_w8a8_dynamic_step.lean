/- Diagnose the actual suffix checks on identical typed function signatures. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8DynamicChain

open Lean MerlinVerify MerlinVerify.Check

def main (args : List String) : IO UInt32 := do
  let [beforePath, afterPath] := args | return 2
  let .ok before := Decode.module (← IO.FS.readFile beforePath) | return 2
  let .ok after := Decode.module (← IO.FS.readFile afterPath) | return 2
  let checks := (Module.definedNames before).map fun fn =>
    let (frozen, conditional) := match TypedModule.signature? 64 before fn with
      | none => (false, false)
      | some (inputs, outputs) => (PolicyW8A8Chain.stepCheck 64 inputs outputs fn before after,
          PolicyW8A8DynamicChain.stepCheck 64 inputs outputs fn before after)
    Json.mkObj [("function", toJson fn), ("frozen", toJson frozen), ("conditional", toJson conditional)]
  IO.println (Json.mkObj [("source_free", toJson (Rules.W8A8SourceFree.moduleCheck before)),
    ("target_free", toJson (Rules.W8A8SourceFree.moduleCheck after)), ("functions", toJson checks)]).compress
  return 0
