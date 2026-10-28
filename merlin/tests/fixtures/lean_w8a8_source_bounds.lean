/- Diagnostic extraction through the actual source recognizer and its reduction-bound theorem.
This evaluates no tensors, installs no policy and cannot issue a whole-pass receipt. -/
import MerlinVerify.Core.Json
import MerlinVerify.Rules.W8A8ReferenceBounds

open Lean MerlinVerify MerlinVerify.Rules

def main (args : List String) : IO UInt32 := do
  let [sourcePath, reach] := args | return 2
  if reach != "all" && reach != "captured" then return 2
  let .ok source := Decode.module (← IO.FS.readFile sourcePath) | return 2
  let config : W8A8Reference.Config := ⟨if reach == "all" then .all else .captured⟩
  let mut sites : List Json := []
  let mut refusals : List String := []
  for op in source.ops do
    if op.name == "func.func" && !(op.asFunc?.map (·.isDecl)).getD true then
      let fn := (op.asFunc?.map (·.name)).getD ""
      match op.regions with
      | [.mk [.mk inputs operations]] =>
        let mut seen : List Op := []
        for actual in operations do
          if W8A8Reference.candidate actual && W8A8Reference.selected config seen (actual.operands.take 2) then
            match checked : W8A8Reference.view? inputs seen actual with
            | none => refusals := refusals ++ [s!"{fn}: source contraction outside the reference domain"]
            | some view =>
              let extent : { k : Nat // (k : Int) * 16384 < 2147483648 } :=
                ⟨view.k, W8A8ReferenceBounds.view_bound checked⟩
              sites := sites ++ [Json.mkObj [("function", toJson fn), ("result", toJson view.result.id),
                ("m", toJson view.m), ("n", toJson view.n), ("k", toJson extent.val)]]
          seen := actual :: seen
      | _ => refusals := refusals ++ [s!"{fn}: expected one region and one block"]
  IO.println (Json.mkObj [("scope", toJson "source_reduction_bounds_only"),
    ("whole_pass_verified", toJson false), ("sites", toJson sites), ("refusals", toJson refusals),
    ("theorem", toJson "MerlinVerify.Rules.W8A8ReferenceBounds.mapped_prefix_exact")]).compress
  return if refusals.isEmpty then 0 else 1
