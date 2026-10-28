/- Actual initialized source/reference/output qualification under separate draft dynamic laws.
Passing this driver never approves the policy, primitive interpretation or public receipts. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8DynamicQualified

open Lean MerlinVerify MerlinVerify.Check

def main (args : List String) : IO UInt32 := do
  let [beforePath, afterPath, reach] := args | return 2
  let .ok before := Decode.module (← IO.FS.readFile beforePath) | return 2
  let .ok after := Decode.module (← IO.FS.readFile afterPath) | return 2
  if reach != "all" && reach != "captured" then return 2
  let config : Rules.W8A8Reference.Config := ⟨if reach == "all" then .all else .captured⟩
  let reference := Rules.W8A8Reference.expand config before
  let checked := PolicyW8A8DynamicQualified.check config 64 before after
  let functions := (Module.definedNames before).map fun fn =>
    let sourceInitialized := match TypedModule.signature? 64 before fn with
      | none => false
      | some (inputs, outputs) => PolicyW8A8DynamicQualified.initializedCheck 64 before fn inputs outputs
    let referenceInitialized := match reference with
      | none => false
      | some program => match TypedModule.signature? 64 program fn with
        | none => false
        | some (inputs, outputs) => PolicyW8A8DynamicQualified.initializedCheck 64 program fn inputs outputs
    Json.mkObj [("function", toJson fn), ("source_initialized", toJson sourceInitialized),
      ("reference_initialized", toJson referenceInitialized)]
  IO.println (Json.mkObj [("scope", toJson "provisional_dynamic_w8a8_initialized_policy_refinement"),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false),
    ("input_domain", toJson "typed_initialized_inputs_with_finite_f32_values"),
    ("floating_domain", toJson "byte_extension_with_explicit_source_and_dynamic_primitive_laws"),
    ("additional_dynamic_primitive_laws", toJson (3 : Nat)),
    ("conditional_initialized_policy_refinement", toJson checked), ("functions", toJson functions),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8DynamicQualified.check_sound")]).compress
  return if checked then 0 else 1
