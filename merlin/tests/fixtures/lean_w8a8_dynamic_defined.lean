/- Conditional dynamic initialization only. Neither this report nor the synthetic witness
establishes a concrete floating implementation or approves the numerical policy. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8DynamicDefined
import MerlinVerify.Check.PolicyW8A8DynamicCalled
import MerlinVerify.Check.PolicyW8A8FiniteQualified
import MerlinVerify.Check.TypedModule
import MerlinVerify.Tests.W8A8DynamicLaws

open Lean MerlinVerify MerlinVerify.Check MerlinVerify.Rules

example : ∃ ctx, PolicyW8A8DynamicContext.ContextOk ctx := by
  obtain ⟨base, laws⟩ := MerlinVerify.Tests.W8A8DynamicLaws.domain_inhabited
  exact ⟨{ M := W8A8ByteModel.extend base, ext := fun _ _ => .stuck "no external contract", indexBits := 64 },
    base, laws, rfl⟩

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok source := Decode.module (← IO.FS.readFile path) | return 2
  let names := Module.definedNames source
  let checks := names.map fun fn => match TypedModule.signature? 64 source fn with
    | none => false
    | some (inputs, outputs) => PolicyW8A8DynamicDefined.check 64 source fn inputs outputs
  let checked := !names.isEmpty && checks.all id
  let functions := (names.zip checks).map fun (fn, initialized) =>
    let called := match TypedModule.signature? 64 source fn with
      | none => false
      | some (inputs, outputs) => PolicyW8A8DynamicCalled.check 64 source fn inputs outputs
    let sourceOnly := match TypedModule.signature? 64 source fn with
      | none => false
      | some (inputs, outputs) => PolicyW8A8FiniteQualified.initializedCheck 64 source fn inputs outputs
    Json.mkObj [("function", toJson fn), ("initialized", toJson initialized),
      ("called_initialized", toJson called), ("source_laws_only_initialized", toJson sourceOnly)]
  IO.println (Json.mkObj [("scope", toJson "provisional_dynamic_w8a8_initialized_functions"),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false), ("synthetic_model_domain_inhabited", toJson true),
    ("input_domain", toJson "typed_initialized_inputs_with_finite_f32_values"),
    ("floating_domain", toJson "byte_extension_with_explicit_source_and_dynamic_primitive_laws"),
    ("additional_dynamic_primitive_laws", toJson (3 : Nat)),
    ("conditional_initialized", toJson checked), ("functions", toJson functions),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8DynamicDefined.check_sound")]).compress
  return if checked then 0 else 1
