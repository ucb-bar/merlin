/- Public diagnostic evidence for conditional initialized W8A8 qualification. This records
the stronger context separately from the older draft claim and never approves its premises. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8DynamicChain

namespace MerlinVerify.Driver.W8A8Dynamic

open MerlinVerify MerlinVerify.Check Lean

def report (config : Rules.W8A8Reference.Config) (ib : Nat)
    (before after : Module) (suffix : Option (List Module)) : Json := Id.run do
  let boundaryChecked := PolicyW8A8DynamicQualified.check config ib before after
  let checked := match suffix with
    | none => boundaryChecked
    | some rest => PolicyW8A8DynamicChain.check config ib before after rest
  let mut details : List Json := []
  let mut links : List Json := []
  let mut suffixMatched := true
  match Rules.W8A8Reference.expand config before with
  | none =>
    suffixMatched := false
    links := (suffix.getD []).zipIdx.map fun (_, index) => Json.mkObj [("index", toJson index),
      ("conditional_match", .null), ("status", toJson "unmeasured")]
  | some reference =>
    let functions := Module.definedNames reference
    for fn in functions do
      let (sourceInitialized, referenceInitialized) :=
        match TypedModule.signature? ib before fn, TypedModule.signature? ib reference fn with
        | some (inputs, outputs), some (otherInputs, otherOutputs) =>
          if inputs = otherInputs ∧ outputs = otherOutputs then
            (PolicyW8A8DynamicQualified.initializedCheck ib before fn inputs outputs,
              PolicyW8A8DynamicQualified.initializedCheck ib reference fn inputs outputs)
          else (false, false)
        | _, _ => (false, false)
      details := details ++ [Json.mkObj [("function", toJson fn),
        ("source_initialized", toJson sourceInitialized),
        ("reference_initialized", toJson referenceInitialized)]]
    let mut previous := after
    for next in suffix.getD [] do
      if !boundaryChecked then
        links := links ++ [Json.mkObj [("index", toJson links.length),
          ("conditional_match", .null), ("status", toJson "unmeasured")]]
        continue
      let matched := functions.all (fun fn =>
        match TypedModule.signature? ib reference fn with
        | none => false
        | some (inputs, outputs) => PolicyW8A8DynamicChain.stepCheck ib inputs outputs fn previous next) &&
        (outsideChanges previous next).unverified.isEmpty
      suffixMatched := suffixMatched && matched
      links := links ++ [Json.mkObj [("index", toJson links.length),
        ("conditional_match", toJson matched), ("status", toJson (if matched then "checked" else "refused"))]]
      previous := next
  return Json.mkObj [
    ("checker", toJson (if suffix.isSome then "MerlinVerify.Check.PolicyW8A8DynamicChain.check"
      else "MerlinVerify.Check.PolicyW8A8DynamicQualified.check")),
    ("theorem", toJson (if suffix.isSome then "MerlinVerify.Check.PolicyW8A8DynamicChain.check_sound"
      else "MerlinVerify.Check.PolicyW8A8DynamicQualified.check_sound")),
    ("index_bits", toJson ib),
    ("policy_scope", toJson (if config.scope == .captured then "captured" else "all")),
    ("input_domain", toJson "typed_initialized_inputs_with_finite_f32_values"),
    ("floating_domain", toJson "byte_extension_with_explicit_source_and_dynamic_primitive_laws"),
    ("source_primitive_laws", toJson (7 : Nat)), ("additional_dynamic_primitive_laws", toJson (3 : Nat)),
    ("policy_reviewed", toJson false), ("primitive_model_laws_verified", toJson false),
    ("whole_pass_verified", toJson false),
    ("boundary_qualification_passed", toJson boundaryChecked),
    ("conditional_initialized_policy_refinement", toJson checked),
    ("functions", toJson details), ("suffix_checked", toJson boundaryChecked),
    ("suffix_match", if boundaryChecked then toJson suffixMatched else .null),
    ("suffix_boundaries", toJson links)]

end MerlinVerify.Driver.W8A8Dynamic
