/- Explicit-model specialization of unchanged numerical and prefix checkers. Embedded generic
diagnostics retain their original hypothesis labels; the new audited roots discharge those
numerical premises for the named model family. Draft admission flags remain unchanged. -/
import MerlinVerify.Driver.W8A8Rotary
import MerlinVerify.Check.PolicyW8A8IEEEPrefix

namespace MerlinVerify.Driver.W8A8IEEE

open Lean MerlinVerify Check Definedness

def report (config : Rules.W8A8Reference.Config) (ib : Nat) (contracts : W8A8Bounded.Contracts)
    (caps : W8A8Rotary.Caps) (before after : Module) : Json :=
  let intervals := W8A8Bounded.intervals contracts
  let names := Module.definedNames before
  let inhabited := !names.isEmpty && names.all fun fn =>
    match TypedModule.signature? ib before fn, caps.lookup fn with
    | some (types, _), some contract =>
      PolicyW8A8RotaryDomain.check ib types (intervals fn) contract.caps
    | _, _ => false
  Json.mkObj [
    ("schema", toJson "merlin.w8a8.ieee-context-candidate.v1"),
    ("checker", toJson "MerlinVerify.Check.PolicyW8A8IEEEQualified.check"),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8IEEEQualified.check_sound"),
    ("bound_theorem", toJson "MerlinVerify.Check.PolicyW8A8IEEEPrefix.entry_bound"),
    ("joint_domain_theorem", toJson "MerlinVerify.Check.PolicyW8A8IEEEQualified.domain_inhabited"),
    ("context_witness_theorem", toJson "MerlinVerify.Check.PolicyW8A8IEEEContext.inhabited"),
    ("logical_model", toJson "W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback)"),
    ("fallback_scope", toJson "arbitrary shared operations outside the constrained slice; witness uses NaN/poison sentinels"),
    ("rsqrt_recipe", toJson "binary32 division of one by the binary32 rounded square root"),
    ("source_laws_proved", toJson (7 : Nat)), ("dynamic_laws_proved", toJson (3 : Nat)),
    ("arithmetic_agreement_fields_proved", toJson (9 : Nat)),
    ("additional_numerical_hypotheses", toJson (0 : Nat)),
    ("logical_context_inhabitation_proved", toJson true),
    ("joint_input_domain_checked", toJson inhabited),
    ("complete_boundary_checked", toJson (PolicyW8A8IEEEQualified.check config ib intervals before after [])),
    ("native_primitive_fidelity_proved", toJson false),
    ("frontend_composite_identified", toJson false),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false), ("concrete_context_inhabited", toJson false),
    ("index_bits", toJson ib),
    ("policy_scope", toJson (if config.scope == .captured then "captured" else "all")),
    ("generic_rotary", W8A8Rotary.report config ib contracts caps before after),
    ("generic_numerical", W8A8Numerical.report config ib contracts before after []),
    ("verdict", toJson "abstracted")]

def run (beforePath afterPath contractsPath capsPath width scope : String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | IO.eprintln "index_bits must be positive"; return 2
  if scope != "all" && scope != "captured" then IO.eprintln "invalid policy scope"; return 2
  let before := Decode.module (← IO.FS.readFile beforePath)
  let after := Decode.module (← IO.FS.readFile afterPath)
  let contracts := (Json.parse (← IO.FS.readFile contractsPath)).bind W8A8Bounded.decodeContracts
  let caps := (Json.parse (← IO.FS.readFile capsPath)).bind W8A8Rotary.decodeCaps
  match before, after, contracts, caps with
  | .ok before, .ok after, .ok contracts, .ok caps =>
    let names := Module.definedNames before
    if contracts.length != names.length || caps.length != names.length ||
        !names.all (fun name => (contracts.lookup name).isSome && (caps.lookup name).isSome) then
      IO.eprintln "both contracts must name every source function exactly once"; return 2
    IO.println (report ⟨if scope == "captured" then .captured else .all⟩ ib contracts caps before after).compress
    return 1
  | .error error, _, _, _ | _, .error error, _, _ | _, _, .error error, _ | _, _, _, .error error =>
    IO.eprintln s!"decode error: {error}"; return 2

end MerlinVerify.Driver.W8A8IEEE
