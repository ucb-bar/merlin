/- A separately named modular policy claim. Complete-program outcome correspondence and
successful/initialized execution are reported independently; every receipt remains draft. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Correspondence

namespace MerlinVerify.Driver.W8A8Correspondence

open Lean MerlinVerify MerlinVerify.Check

def report (config : Rules.W8A8Reference.Config) (ib : Nat)
    (before after : Module) (suffix : List Module) : Json := Id.run do
  let reference := Rules.W8A8Reference.expand config before
  let boundary := PolicyParametric.check (Rules.W8A8Reference.policy config) before after
  let checked := PolicyW8A8Correspondence.check config before after suffix
  let mut previous := after
  let mut links : List Json := []
  for next in suffix do
    let matched := ParametricModule.check previous next
    links := links ++ [Json.mkObj [("index", toJson links.length), ("matched", toJson matched)]]
    previous := next
  return Json.mkObj [
    ("schema", toJson "merlin.w8a8.correspondence-candidate.v1"),
    ("checker", toJson "MerlinVerify.Check.PolicyW8A8Correspondence.check"),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8Correspondence.check_sound"),
    ("successful_reference_theorem", toJson "MerlinVerify.Check.PolicyW8A8Correspondence.successful_reference"),
    ("defined_reference_theorem", toJson "MerlinVerify.Check.PolicyW8A8Correspondence.defined_reference"),
    ("index_bits", toJson ib),
    ("policy_scope", toJson (if config.scope == .captured then "captured" else "all")),
    ("semantic_interpretation", toJson "MerlinVerify.Rules.W8A8SourceSemantics.opSem"),
    ("correspondence_domain", toJson "all_contexts_functions_and_inputs_in_this_operational_semantics"),
    ("reference_expanded", toJson reference.isSome),
    ("source_functions", toJson (Module.definedNames before)),
    ("reference_functions", toJson ((reference.map Module.definedNames).getD [])),
    ("target_functions", toJson (Module.definedNames after)),
    ("actual_boundary_changed", toJson (!ExactSyntax.modules before after)),
    ("complete_boundary_matched", toJson boundary),
    ("complete_correspondence_passed", toJson checked),
    ("suffix_boundaries", toJson links),
    ("numerical_premises_for_correspondence", toJson (0 : Nat)),
    ("original_source_definedness_required_for_correspondence", toJson false),
    ("successful_reference_required_for_success_transfer", toJson true),
    ("whole_model_initialized_execution_established", toJson false),
    ("original_source_equivalence_claimed", toJson false),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false), ("concrete_context_inhabited", toJson false),
    ("verdict", toJson "abstracted")]

def run (beforePath afterPath width scope : String) (suffixPaths : List String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | IO.eprintln "index_bits must be positive"; return 2
  if scope != "all" && scope != "captured" then
    IO.eprintln "policy scope must be all or captured"
    return 2
  let config : Rules.W8A8Reference.Config := ⟨if scope == "all" then .all else .captured⟩
  let decoded ← (beforePath :: afterPath :: suffixPaths).mapM fun path => do
    return Decode.module (← IO.FS.readFile path)
  match decoded.mapM id with
  | .ok (before :: after :: suffix) =>
    IO.println (report config ib before after suffix).compress
    return 1
  | .error error => IO.eprintln s!"decode error: {error}"; return 2
  | _ => IO.eprintln "missing policy boundary"; return 2

end MerlinVerify.Driver.W8A8Correspondence
