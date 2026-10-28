/- Provisional initialized-execution qualification. Refusal is distinct from a counterexample;
acceptance carries universal theorems but cannot approve the draft recipe or frozen semantics. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Qualified

open Lean MerlinVerify MerlinVerify.Check

def main (args : List String) : IO UInt32 := do
  let [beforePath, afterPath, reach] := args | return 2
  let .ok before := Decode.module (← IO.FS.readFile beforePath) | return 2
  let .ok after := Decode.module (← IO.FS.readFile afterPath) | return 2
  if reach != "all" && reach != "captured" then return 2
  let scope := if reach == "all" then Rules.W8A8Reference.Scope.all else .captured
  let config : Rules.W8A8Reference.Config := ⟨scope⟩
  let initialized := PolicyW8A8Qualified.check config 64 before after
  let functions := (Module.definedNames before).map fun fn =>
    let defined := match TypedModule.signature? 64 before fn with
      | none => false
      | some (inputs, outputs) => PolicyW8A8Defined.check 64 before fn inputs outputs
    Json.mkObj [("function", toJson fn), ("source_initialized", toJson defined)]
  IO.println (Json.mkObj [("scope", toJson "provisional_initialized_policy_refinement"),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("initialized_policy_refinement", toJson initialized), ("functions", toJson functions),
    ("domain", toJson "all_declared_typed_initialized_inputs"),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8Qualified.check_sound")]).compress
  return if initialized then 0 else 1
