/- Conditional whole-function qualification, separate from the unconditional checker and public
receipts. A synthetic law witness proves the model domain is inhabited, not IEEE fidelity. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8FiniteDefined
import MerlinVerify.Check.TypedModule

open Lean MerlinVerify MerlinVerify.Check MerlinVerify.Rules

def synthetic : FloatModel where
  op _ _ _ := 0
  cmp _ _ _ _ := false
  convert _ _ dst _ := match dst with
    | .f64 => 0x3ff0000000000000#64
    | _ => 0
  ofInt _ _ _ _ := 0
  toInt _ _ _ _ := none

theorem synthetic_laws : W8A8FiniteLaws.Laws synthetic where
  widen _ _ := by
    change W8A8FiniteLaws.WideScale 0x3ff0000000000000#64
    unfold W8A8FiniteLaws.WideScale
    decide
  reciprocal _ _ := by
    change W8A8FiniteLaws.Narrowable 0
    unfold W8A8FiniteLaws.Narrowable
    decide
  narrow _ _ := by change W8A8FiniteLaws.Finite 0; decide
  integer _ _ _ := by change W8A8FiniteLaws.Finite 0; decide
  multiply _ _ _ _ := by change W8A8ByteFloat.isNaN 0 = false; decide
  round _ _ := by change W8A8ByteFloat.isNaN 0 = false; decide
  add _ _ _ _ := by change W8A8ByteFloat.isNaN 0 = false; decide

example : ∃ ctx, PolicyW8A8FiniteQuantize.ContextOk ctx :=
  ⟨{ M := W8A8ByteModel.extend synthetic, ext := fun _ _ => .stuck "no external contract", indexBits := 64 },
    synthetic, synthetic_laws, rfl⟩

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok source := Decode.module (← IO.FS.readFile path) | return 2
  let names := Module.definedNames source
  let checks := names.map fun fn => match TypedModule.signature? 64 source fn with
    | none => false
    | some (inputs, outputs) => PolicyW8A8FiniteDefined.check 64 source fn inputs outputs
  let functions := (names.zip checks).map fun (fn, checked) =>
    let unconditional := match TypedModule.signature? 64 source fn with
      | none => false
      | some (inputs, outputs) => PolicyW8A8Defined.check 64 source fn inputs outputs
    Json.mkObj [("function", toJson fn), ("conditional_source_initialized", toJson checked),
      ("unconditional_source_initialized", toJson unconditional)]
  let checked := !names.isEmpty && checks.all id
  IO.println (Json.mkObj [("scope", toJson "provisional_finite_input_initialized_functions"),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false), ("synthetic_model_domain_inhabited", toJson true),
    ("input_domain", toJson "typed_initialized_inputs_with_finite_f32_values"),
    ("floating_domain", toJson "byte_extension_of_base_satisfying_explicit_primitive_laws"),
    ("conditional_source_initialized", toJson checked), ("functions", toJson functions),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8FiniteDefined.check_sound")]).compress
  return if checked then 0 else 1
