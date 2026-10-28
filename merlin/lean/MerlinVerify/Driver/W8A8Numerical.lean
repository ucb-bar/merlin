/- Standalone numerical-candidate diagnostics. Prefix certificates and complete-boundary
qualification are distinct results; neither approves the input domain or numerical premises. -/
import MerlinVerify.Driver.W8A8Bounded
import MerlinVerify.Check.PolicyW8A8NumericalChain
import MerlinVerify.Check.PolicyW8A8NumericalPrefix

namespace MerlinVerify.Driver.W8A8Numerical

open Lean MerlinVerify MerlinVerify.Check Definedness PolicyW8A8NumericalFacts

def className : Class → String
  | .any => "any" | .finite => "finite" | .nonNaN => "non_nan"
  | .nonnegative => "nonnegative" | .nonzeroFinite => "finite_nonzero"

def infoJson (fact : Refined.Fact Info) : Json := Json.mkObj [
  ("class", toJson (className fact.info.numeric)), ("initialized", toJson fact.base.initialized),
  ("literal_f32", (fact.info.prior.floating.literal.map (fun bits => toJson bits.toNat)).getD .null)]

structure Scan where
  count : Nat
  stopped : Option Op

/-- Search chooses a prefix length only. `prefixReport` rechecks that prefix independently. -/
def scan (ib : Nat) (Γ : Refined.FEnv Info) : List Op → Nat → Scan
  | [], count => ⟨count, none⟩
  | op :: rest, count =>
    if op.isTerminator then ⟨count, none⟩ else
      match op.operands.mapM (fun id => Γ.lookup id) with
      | none => ⟨count, some op⟩
      | some inputs => match (PolicyW8A8NumericalDefined.book ib).rule Γ op inputs with
        | none => ⟨count, some op⟩
        | some outputs =>
          if outputs.length == op.results.length && op.successors.isEmpty then
            scan ib ((op.results.map (·.id)).zip outputs ++ Γ) rest (count + 1)
          else ⟨count, some op⟩

def prefixReport (ib : Nat) (m : Module) (fn : String) (types : List VTy)
    (ranges : List (Option IndexRanges.Interval)) : Json := Id.run do
  match m.func? fn, PolicyW8A8NumericalDefined.inputFacts types ranges with
  | some function, some facts =>
    match function.body with
    | .mk [.mk arguments ops] =>
      let found := scan ib ((arguments.map (·.id)).zip facts) ops 0
      let certified := PolicyW8A8NumericalPrefix.entry ib m fn types ranges found.count
      let mut quantizers := []
      let mut finiteDequantizers := []
      if let some final := certified then
        for op in ops.take found.count do
          if op.name == Rules.W8A8SourceContract.name .quantize then
            for result in op.results do
              if let some fact := final.lookup result.id then
                if fact.base.initialized then quantizers := quantizers ++ [toJson result.id]
          if op.name == Rules.W8A8SourceContract.name .dequantize then
            for result in op.results do
              if let some fact := final.lookup result.id then
                if accepts fact.info .finite then finiteDequantizers := finiteDequantizers ++ [toJson result.id]
      let refusal := found.stopped.map fun op => Json.mkObj [
        ("operation", toJson op.name), ("results", toJson (op.results.map (·.id))),
        ("operands", toJson op.operands),
        ("input_facts", toJson (op.operands.map fun id =>
          ((certified.getD []).lookup id).map infoJson |>.getD .null))]
      return Json.mkObj [("checked", toJson certified.isSome), ("operation_count", toJson found.count),
        ("initialized_source_quantizers", toJson quantizers),
        ("finite_source_dequantizers", toJson finiteDequantizers), ("first_refusal", refusal.getD .null)]
    | _ => return .null
  | _, _ => return .null

def report (config : Rules.W8A8Reference.Config) (ib : Nat) (contracts : W8A8Bounded.Contracts)
    (before after : Module) (suffix : List Module) : Json := Id.run do
  let bounds := W8A8Bounded.intervals contracts
  let boundary := PolicyW8A8NumericalQualified.check config ib bounds before after
  let checked := PolicyW8A8NumericalChain.check config ib bounds before after suffix
  let reference := Rules.W8A8Reference.expand config before
  let details := (Module.definedNames before).map fun fn =>
    match TypedModule.signature? ib before fn with
    | none => Json.mkObj [("function", toJson fn), ("input_intervals", toJson ((bounds fn).map W8A8Bounded.intervalJson)),
        ("domain_inhabited", toJson false), ("source_initialized", toJson false),
        ("reference_initialized", toJson false), ("source_prefix", .null), ("reference_prefix", .null)]
    | some (inputs, outputs) =>
      let sourceInitialized := PolicyW8A8NumericalCalled.check ib bounds before fn inputs outputs
      let referenceInitialized := reference.any fun actual =>
        PolicyW8A8NumericalCalled.check ib bounds actual fn inputs outputs
      Json.mkObj [("function", toJson fn), ("input_intervals", toJson ((bounds fn).map W8A8Bounded.intervalJson)),
        ("domain_inhabited", toJson (PolicyW8A8NumericalInhabited.check ib inputs (bounds fn))),
        ("source_initialized", toJson sourceInitialized), ("reference_initialized", toJson referenceInitialized),
        ("source_prefix", prefixReport ib before fn inputs (bounds fn)),
        ("reference_prefix", (reference.map (fun actual => prefixReport ib actual fn inputs (bounds fn))).getD .null)]
  let mut links := []
  let mut previous := after
  for next in suffix do
    let matched := boundary && (Module.definedNames before).all (fun fn =>
      match TypedModule.signature? ib before fn with
      | none => false
      | some (inputs, outputs) => (PolicyW8A8NumericalChain.steps ib bounds).check inputs outputs fn previous next) &&
        (outsideChanges previous next).unverified.isEmpty
    links := links ++ [Json.mkObj [("index", toJson links.length),
      ("checked", toJson matched), ("status", toJson (if !boundary then "unmeasured" else if matched then "checked" else "refused"))]]
    previous := next
  return Json.mkObj [
    ("schema", toJson "merlin.w8a8.numerical-candidate.v1"),
    ("checker", toJson "MerlinVerify.Check.PolicyW8A8NumericalChain.check"),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8NumericalChain.check_sound"),
    ("prefix_theorem", toJson "MerlinVerify.Check.PolicyW8A8NumericalPrefix.entry_tracked_defined"),
    ("prefix_facts_theorem", toJson "MerlinVerify.Check.PolicyW8A8NumericalPrefix.entry_sound"),
    ("nonvacuity_theorem", toJson "MerlinVerify.Check.PolicyW8A8NumericalInhabited.check_sound"),
    ("verdict", toJson "abstracted"), ("index_bits", toJson ib),
    ("policy_scope", toJson (if config.scope == .captured then "captured" else "all")),
    ("input_domain", toJson "candidate_finite_nonzero_f32_and_declared_integer_intervals_per_function"),
    ("floating_domain", toJson "byte_extension_with_source_dynamic_laws_and_explicit_arithmetic_agreement"),
    ("source_primitive_laws", toJson (7 : Nat)), ("additional_dynamic_primitive_laws", toJson (3 : Nat)),
    ("arithmetic_agreement_assumed", toJson true), ("concrete_context_inhabited", toJson false),
    ("policy_reviewed", toJson false), ("primitive_model_laws_verified", toJson false),
    ("whole_pass_verified", toJson false),
    ("structural_match", toJson (PolicyParametric.check (Rules.W8A8Reference.policy config) before after)),
    ("boundary_qualification_passed", toJson boundary),
    ("conditional_initialized_policy_refinement", toJson checked),
    ("functions", toJson details), ("suffix_boundaries", toJson links)]

def run (beforePath afterPath contractsPath width scope : String) (suffixPaths : List String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | IO.eprintln "index_bits must be positive"; return 2
  if scope != "all" && scope != "captured" then
    IO.eprintln "policy scope must be all or captured"
    return 2
  let config : Rules.W8A8Reference.Config := ⟨if scope == "all" then .all else .captured⟩
  let decoded ← (beforePath :: afterPath :: suffixPaths).mapM fun path => do
    return Decode.module (← IO.FS.readFile path)
  let contracts := (Json.parse (← IO.FS.readFile contractsPath)).bind W8A8Bounded.decodeContracts
  match decoded.mapM id, contracts with
  | .ok (before :: after :: rest), .ok contracts =>
    let names := Module.definedNames before
    if contracts.length != names.length || !names.all (fun name => (contracts.lookup name).isSome) then
      IO.eprintln "input contracts must name every source function exactly once"
      return 2
    IO.println (report config ib contracts before after rest).compress
    return 1
  | .error error, _ | _, .error error => IO.eprintln s!"decode error: {error}"; return 2
  | _, _ => IO.eprintln "missing policy boundary"; return 2

end MerlinVerify.Driver.W8A8Numerical
