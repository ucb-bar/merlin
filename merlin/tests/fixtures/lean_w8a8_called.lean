/- Inspect checked internal-call contracts. Numerical model premises and the recipe remain
provisional. Only contracts inferred from actual module bodies can qualify this report. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Called

open Lean MerlinVerify MerlinVerify.Check MerlinVerify.Check.Definedness

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok source := Decode.module (← IO.FS.readFile path) | return 2
  let names := Module.definedNames source
  let fuel := source.funcs.length + 1
  let known := PolicyW8A8Called.stage 64 source fuel
  let checks := names.map fun fn => match TypedModule.signature? 64 source fn with
    | none => false
    | some (inputs, outputs) => PolicyW8A8Called.checkKnown known fn inputs outputs
  let functions := (names.zip checks).map fun (fn, checked) =>
    let classes : List String := match Calls.find? known fn with
      | none => []
      | some contract => contract.outputs.map fun (fact : Refined.Fact FiniteFacts.Class) => reprStr fact.info
    Json.mkObj [("function", toJson fn), ("initialized", toJson checked), ("output_classes", toJson classes)]
  let checked := !names.isEmpty && checks.all id
  IO.println (Json.mkObj [("scope", toJson "provisional_finite_input_initialized_internal_calls"),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false),
    ("input_domain", toJson "typed_initialized_inputs_with_finite_f32_values"),
    ("floating_domain", toJson "byte_extension_of_base_satisfying_explicit_primitive_laws"),
    ("module_fuel", toJson fuel), ("conditional_source_initialized", toJson checked),
    ("functions", toJson functions), ("theorem", toJson "MerlinVerify.Check.PolicyW8A8Called.check_sound")]).compress
  return if checked then 0 else 1
