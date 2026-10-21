/- Typed strict policy refinement with separately checked source/reference support. This
provisional driver cannot report initialized execution or publish a full-pass receipt. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Supported

open Lean MerlinVerify MerlinVerify.Check

def main (args : List String) : IO UInt32 := do
  let [beforePath, afterPath, reach] := args | return 2
  let .ok before := Decode.module (← IO.FS.readFile beforePath) | return 2
  let .ok after := Decode.module (← IO.FS.readFile afterPath) | return 2
  if reach != "all" && reach != "captured" then return 2
  let scope := if reach == "all" then Rules.W8A8Reference.Scope.all else .captured
  let config : Rules.W8A8Reference.Config := ⟨scope⟩
  let checked := PolicyW8A8Supported.check config 64 before after
  IO.println (Json.mkObj [("scope", toJson "provisional_typed_policy_refinement"),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("policy_refinement", toJson checked), ("initialized_execution_verified", toJson false),
    ("domain", toJson "all_declared_typed_inputs_poison_permitted"),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8Supported.check_sound")]).compress
  return if checked then 0 else 1
