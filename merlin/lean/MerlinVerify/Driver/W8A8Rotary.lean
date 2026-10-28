/- Separate per-argument-bound candidate diagnostics. An accepted prefix is
rechecked against the actual entry/operations. Complete policy qualification
and native-model fidelity remain separate open obligations. -/
import MerlinVerify.Driver.W8A8Numerical
import MerlinVerify.Check.PolicyW8A8RotaryPrefix

namespace MerlinVerify.Driver.W8A8Rotary

open Lean MerlinVerify Check Definedness PolicyW8A8RotaryFacts

structure Contract where
  caps : List (Option Nat)
  required : List (ValueId × Nat)

abbrev Caps := List (String × Contract)

def decodeCaps (value : Json) : Except String Caps := do
  let entries ← (← value.getArr?).toList.mapM fun entry => do
    Decode.exactKeys "exponent contract" entry ["function", "exponent_caps", "required_bounds"] []
    let fn ← (← entry.getObjVal? "function").getStr?
    let caps ← (← (← entry.getObjVal? "exponent_caps").getArr?).toList.mapM fun cap =>
      if cap == .null then pure none else (cap.getNat?).map some
    let required ← (← (← entry.getObjVal? "required_bounds").getArr?).toList.mapM fun item => do
      Decode.exactKeys "required bound" item ["value", "upper"] []
      return (← (← item.getObjVal? "value").getNat?, ← (← item.getObjVal? "upper").getNat?)
    unless (required.map (·.1)).eraseDups.length == required.length do
      throw "required bounds contain duplicate SSA values"
    return (fn, ⟨caps, required⟩)
  unless (entries.map (·.1)).eraseDups.length == entries.length do
    throw "exponent contracts contain duplicate function names"
  return entries

def infoJson (id : ValueId) (fact : Refined.Fact Info) : Json := Json.mkObj [
  ("id", toJson id), ("initialized", toJson fact.base.initialized),
  ("class", toJson (W8A8Numerical.className fact.info.prior.numeric)),
  ("exponent_upper", (upper fact.info).map toJson |>.getD .null),
  ("integer_interval", W8A8Bounded.intervalJson fact.info.prior.prior.indices)]

def scan (ib : Nat) (Γ : Refined.FEnv Info) : List Op → Nat → W8A8Numerical.Scan
  | [], count => ⟨count, none⟩
  | op :: rest, count =>
    if op.isTerminator then ⟨count, none⟩ else
      match op.operands.mapM (Γ.lookup ·) with
      | none => ⟨count, some op⟩
      | some inputs => match (PolicyW8A8RotaryDefined.book ib).rule Γ op inputs with
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
      let certified := PolicyW8A8RotaryPrefix.entry ib m fn types ranges caps found.count
      let final := certified.getD []
      let tracked := (arguments ++ (ops.take found.count).flatMap (·.results)).filterMap fun result =>
        (final.lookup result.id).map (infoJson result.id)
      let refusal := found.stopped.map fun op => Json.mkObj [
        ("operation", toJson op.name), ("results", toJson (op.results.map (·.id))),
        ("operands", toJson op.operands), ("input_facts", toJson (op.operands.map fun id =>
          (final.lookup id).map (infoJson id) |>.getD .null))]
      return Json.mkObj [("checked", toJson certified.isSome), ("operation_count", toJson found.count),
        ("tracked_facts", toJson (if certified.isSome then tracked else [])),
        ("required_bounds", toJson (required.map fun (id, cap) => Json.mkObj [
          ("value", toJson id), ("upper", toJson cap),
          ("checked", toJson (certified.isSome && PolicyW8A8RotaryPrefix.bound final id cap))])),
        ("all_required_bounds_checked", toJson (certified.isSome &&
          required.all (fun (id, cap) => PolicyW8A8RotaryPrefix.bound final id cap))),
        ("first_refusal", refusal.getD .null)]
    | _ => return .null
  | _, _ => return .null

def report (config : Rules.W8A8Reference.Config) (ib : Nat) (contracts : W8A8Bounded.Contracts)
    (caps : Caps) (before after : Module) : Json :=
  let reference := Rules.W8A8Reference.expand config before
  let functions := (Module.definedNames before).map fun fn =>
    match TypedModule.signature? ib before fn with
    | none => Json.mkObj [("function", toJson fn), ("domain_inhabited", toJson false),
        ("source_prefix", .null), ("reference_prefix", .null)]
    | some (types, _) =>
      let ranges := (contracts.lookup fn).getD []
      let contract : Contract := (caps.lookup fn).getD ⟨[], []⟩
      let bounds := contract.caps
      Json.mkObj [("function", toJson fn), ("exponent_caps", toJson (bounds.map (fun (cap : Option Nat) =>
          (cap.map toJson).getD .null))),
        ("domain_inhabited", toJson (PolicyW8A8RotaryDomain.check ib types ranges bounds)),
        ("source_prefix", prefixReport ib before fn types ranges bounds contract.required),
        ("reference_prefix", (reference.map (fun actual => prefixReport ib actual fn types ranges bounds contract.required)).getD .null)]
  Json.mkObj [("schema", toJson "merlin.w8a8.rotary-candidate.v1"), ("verdict", toJson "abstracted"),
    ("index_bits", toJson ib), ("policy_scope", toJson (if config.scope == .captured then "captured" else "all")),
    ("input_domain", toJson "candidate_finite_nonzero_f32_with_per_argument_exponent_caps"),
    ("prefix_theorem", toJson "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry_sound"),
    ("bound_theorem", toJson "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry_bound"),
    ("nonvacuity_theorem", toJson "MerlinVerify.Check.PolicyW8A8RotaryDomain.check_inhabited"),
    ("source_primitive_laws", toJson (7 : Nat)), ("additional_dynamic_primitive_laws", toJson (3 : Nat)),
    ("arithmetic_agreement_assumed", toJson true), ("transcendental_bounds_assumed", toJson false),
    ("concrete_context_inhabited", toJson false), ("policy_reviewed", toJson false),
    ("primitive_model_laws_verified", toJson false), ("whole_pass_verified", toJson false),
    ("structural_match", toJson (PolicyParametric.check (Rules.W8A8Reference.policy config) before after)),
    ("functions", toJson functions)]

def run (beforePath afterPath contractsPath capsPath width scope : String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | IO.eprintln "index_bits must be positive"; return 2
  if scope != "all" && scope != "captured" then IO.eprintln "invalid policy scope"; return 2
  let before := Decode.module (← IO.FS.readFile beforePath)
  let after := Decode.module (← IO.FS.readFile afterPath)
  let contracts := (Json.parse (← IO.FS.readFile contractsPath)).bind W8A8Bounded.decodeContracts
  let caps := (Json.parse (← IO.FS.readFile capsPath)).bind decodeCaps
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

end MerlinVerify.Driver.W8A8Rotary
