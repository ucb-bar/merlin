/- Integration driver for the source-derived W8A8 reference draft. This emits no reviewed
whole-pass receipt; the conditional module theorem and policy review status stay explicit. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Rules.W8A8Reference

open Lean MerlinVerify MerlinVerify.Check

def main (args : List String) : IO UInt32 := do
  let [beforePath, afterPath, reach, referencePath] := args | return 2
  let .ok before := Decode.module (← IO.FS.readFile beforePath) | return 2
  let .ok after := Decode.module (← IO.FS.readFile afterPath) | return 2
  let scope := if reach == "all" then Rules.W8A8Reference.Scope.all else .captured
  if reach != "all" && reach != "captured" then return 2
  let policy := Rules.W8A8Reference.policy ⟨scope⟩
  let reference := policy.expand before
  let mut errors : List String := []
  let mut accepted := false
  if let some expected := reference then
    IO.FS.writeFile referencePath (Print.module expected)
    for fn in Module.definedNames expected do
      if !c1Check (synthC1 expected after fn) then
        errors := errors ++ [(explainC1 (synthC1 expected after fn)).getD "refused"]
    accepted := PolicyModule.check policy before after
  else errors := ["source outside the current W8A8 reference draft"]
  IO.println (Json.mkObj [("scope", toJson "w8a8_source_derived_reference_draft"),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("conditional_match", toJson accepted), ("expanded", toJson reference.isSome),
    ("theorem", toJson "MerlinVerify.Check.PolicyModule.check_sound"),
    ("errors", toJson errors)]).compress
  return if accepted then 0 else 1
