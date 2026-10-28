/- Compare actual before/after modules under the provisional source interpretation. This
reports equality with a source-derived policy; it cannot issue a full qualification receipt. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyParametric
import MerlinVerify.Rules.W8A8Reference
import MerlinVerify.Rules.W8A8SourceAlpha

open Lean MerlinVerify MerlinVerify.Check

def main (args : List String) : IO UInt32 := do
  let [beforePath, afterPath, reach] := args | return 2
  let .ok before := Decode.module (← IO.FS.readFile beforePath) | return 2
  let .ok after := Decode.module (← IO.FS.readFile afterPath) | return 2
  if reach != "all" && reach != "captured" then return 2
  let scope := if reach == "all" then Rules.W8A8Reference.Scope.all else .captured
  let policy := Rules.W8A8Reference.policy ⟨scope⟩
  let matched := PolicyParametric.check policy before after
  IO.println (Json.mkObj [("scope", toJson "w8a8_provisional_source_semantics_match"),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("match", toJson matched), ("expanded", toJson (policy.expand before).isSome),
    ("theorem", toJson "MerlinVerify.Check.PolicyParametric.check_sound"),
    ("semantic_laws", toJson "MerlinVerify.Rules.W8A8SourceAlpha.laws")]).compress
  return if matched then 0 else 1
