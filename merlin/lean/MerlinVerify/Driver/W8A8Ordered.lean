/- Separate ordered-contraction diagnostics. Prefix requests are independently rechecked;
unsupported attention numerics do not become complete-boundary verification. -/
import MerlinVerify.Driver.W8A8OrderedContracts
import MerlinVerify.Check.PolicyW8A8OrderedPrefix

namespace MerlinVerify.Driver.W8A8Ordered

open Lean MerlinVerify Check Definedness PolicyW8A8RotaryFacts

def scan (ib : Nat) (Γ : Refined.FEnv Info) : List Op → Nat → W8A8Numerical.Scan
  | [], count => ⟨count, none⟩
  | op :: rest, count =>
    if op.isTerminator then ⟨count, none⟩ else
      match op.operands.mapM (Γ.lookup ·) with
      | none => ⟨count, some op⟩
      | some inputs => match (PolicyW8A8OrderedPrefix.book ib).rule Γ op inputs with
        | none => ⟨count, some op⟩
        | some outputs =>
          if outputs.length == op.results.length && op.successors.isEmpty then
            scan ib ((op.results.map (·.id)).zip outputs ++ Γ) rest (count + 1)
          else ⟨count, some op⟩

def prefixReport (ib : Nat) (m : Module) (fn : String) (types : List VTy)
    (ranges : List (Option IndexRanges.Interval)) (caps : List (Option Nat))
    (required : List (ValueId × Nat)) : Json := Id.run do
  match m.func? fn, PolicyW8A8RotaryDomain.inputFacts types ranges caps with
  | some function, some facts =>
    match function.body with
    | .mk [.mk arguments ops] =>
      let found := scan ib ((arguments.map (·.id)).zip facts) ops 0
      let certified := PolicyW8A8OrderedPrefix.entry ib m fn types ranges caps found.count
      let final := certified.getD []
      let tracked := (arguments ++ (ops.take found.count).flatMap (·.results)).filterMap fun result =>
        (final.lookup result.id).map (W8A8Rotary.infoJson result.id)
      let refusal := found.stopped.map fun op => Json.mkObj [
        ("operation", toJson op.name), ("results", toJson (op.results.map (·.id))),
        ("operands", toJson op.operands), ("input_facts", toJson (op.operands.map fun id =>
          (final.lookup id).map (W8A8Rotary.infoJson id) |>.getD .null))]
      return Json.mkObj [("checked", toJson certified.isSome), ("operation_count", toJson found.count),
        ("tracked_facts", toJson (if certified.isSome then tracked else [])),
        ("required_bounds", toJson (required.map fun (id, cap) => Json.mkObj [
          ("value", toJson id), ("upper", toJson cap),
          ("checked", toJson (certified.isSome && PolicyW8A8OrderedPrefix.bound final id cap))])),
        ("all_required_bounds_checked", toJson (certified.isSome &&
          required.all (fun (id, cap) => PolicyW8A8OrderedPrefix.bound final id cap))),
        ("first_refusal", refusal.getD .null)]
    | _ => return .null
  | _, _ => return .null

def report (config : Rules.W8A8Reference.Config) (ib : Nat) (contracts : W8A8Bounded.Contracts)
    (caps : W8A8OrderedContracts.Contracts) (before after : Module) : Json :=
  let reference := Rules.W8A8Reference.expand config before
  let functions := (Module.definedNames before).map fun fn =>
    match TypedModule.signature? ib before fn with
    | none => Json.mkObj [("function", toJson fn), ("domain_inhabited", toJson false),
        ("source_prefix", .null), ("reference_prefix", .null)]
    | some (types, _) =>
      let ranges := (contracts.lookup fn).getD []
      let contract : W8A8OrderedContracts.Contract := (caps.lookup fn).getD ⟨[], [], []⟩
      Json.mkObj [("function", toJson fn),
        ("domain_inhabited", toJson (PolicyW8A8RotaryDomain.check ib types ranges contract.caps)),
        ("source_prefix", prefixReport ib before fn types ranges contract.caps contract.sourceRequired),
        ("reference_prefix", (reference.map (fun actual =>
          prefixReport ib actual fn types ranges contract.caps contract.referenceRequired)).getD .null)]
  Json.mkObj [("schema", toJson "merlin.w8a8.ordered-contractions.v1"), ("verdict", toJson "abstracted"),
    ("index_bits", toJson ib), ("policy_scope", toJson (if config.scope == .captured then "captured" else "all")),
    ("input_domain", toJson "candidate_finite_nonzero_f32_with_per_argument_exponent_caps"),
    ("logical_model", toJson "W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback)"),
    ("prefix_theorem", toJson "MerlinVerify.Check.PolicyW8A8OrderedPrefix.entry_sound"),
    ("bound_theorem", toJson "MerlinVerify.Check.PolicyW8A8OrderedPrefix.entry_bound"),
    ("rule_theorem", toJson "MerlinVerify.Check.PolicyW8A8OrderedRule.rule_sound"),
    ("nonvacuity_theorem", toJson "MerlinVerify.Check.PolicyW8A8IEEEQualified.domain_inhabited"),
    ("additional_numerical_hypotheses", toJson (0 : Nat)),
    ("transcendental_bounds_assumed", toJson false),
    ("native_primitive_fidelity_proved", toJson false),
    ("whole_pass_verified", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false), ("concrete_context_inhabited", toJson false),
    ("complete_boundary_checked", toJson (PolicyW8A8IEEEQualified.check config ib
      (W8A8Bounded.intervals contracts) before after [])),
    ("structural_match", toJson (PolicyParametric.check (Rules.W8A8Reference.policy config) before after)),
    ("functions", toJson functions)]

def run (beforePath afterPath contractsPath capsPath width scope : String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | IO.eprintln "index_bits must be positive"; return 2
  if scope != "all" && scope != "captured" then IO.eprintln "invalid policy scope"; return 2
  let before := Decode.module (← IO.FS.readFile beforePath)
  let after := Decode.module (← IO.FS.readFile afterPath)
  let contracts := (Json.parse (← IO.FS.readFile contractsPath)).bind W8A8Bounded.decodeContracts
  let caps := (Json.parse (← IO.FS.readFile capsPath)).bind W8A8OrderedContracts.decode
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

end MerlinVerify.Driver.W8A8Ordered
