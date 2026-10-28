/- Bounded-domain diagnostics bind the declared intervals to actual function names and
carry them through every suffix boundary. No diagnostic approves the provisional policy. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8BoundedChain

namespace MerlinVerify.Driver.W8A8Bounded

open MerlinVerify MerlinVerify.Check Lean

abbrev Contracts := List (String × List (Option Definedness.IndexRanges.Interval))

def decodeInterval (value : Json) : Except String (Option Definedness.IndexRanges.Interval) := do
  if value == Json.null then return none
  Decode.exactKeys "integer interval" value ["lower", "upper"] []
  return some ⟨← (← value.getObjVal? "lower").getNat?, ← (← value.getObjVal? "upper").getNat?⟩

def decodeContracts (value : Json) : Except String Contracts := do
  let entries ← (← value.getArr?).toList.mapM fun entry => do
    Decode.exactKeys "input contract" entry ["function", "intervals"] []
    let fn ← (← entry.getObjVal? "function").getStr?
    let intervals ← (← (← entry.getObjVal? "intervals").getArr?).toList.mapM decodeInterval
    return (fn, intervals)
  unless (entries.map (·.1)).eraseDups.length == entries.length do
    throw "input contracts contain duplicate function names"
  return entries

def intervals (contracts : Contracts) : PolicyW8A8BoundedQualified.InputIntervals :=
  fun fn => (contracts.lookup fn).getD []

def intervalJson : Option Definedness.IndexRanges.Interval → Json
  | none => .null
  | some range => Json.mkObj [("lower", toJson range.lower), ("upper", toJson range.upper)]

def report (config : Rules.W8A8Reference.Config) (ib : Nat) (contracts : Contracts)
    (before after : Module) (suffix : Option (List Module)) : Json := Id.run do
  let bounds := intervals contracts
  let boundaryChecked := PolicyW8A8BoundedQualified.check config ib bounds before after
  let checked := match suffix with
    | none => boundaryChecked
    | some rest => PolicyW8A8BoundedChain.check config ib bounds before after rest
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
            (PolicyW8A8BoundedCalled.check ib bounds before fn inputs outputs,
              PolicyW8A8BoundedCalled.check ib bounds reference fn inputs outputs)
          else (false, false)
        | _, _ => (false, false)
      details := details ++ [Json.mkObj [("function", toJson fn),
        ("input_intervals", toJson ((bounds fn).map intervalJson)),
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
        | some (inputs, outputs) => (PolicyW8A8BoundedChain.steps ib bounds).check inputs outputs fn previous next) &&
        (outsideChanges previous next).unverified.isEmpty
      suffixMatched := suffixMatched && matched
      links := links ++ [Json.mkObj [("index", toJson links.length),
        ("conditional_match", toJson matched), ("status", toJson (if matched then "checked" else "refused"))]]
      previous := next
  return Json.mkObj [
    ("checker", toJson (if suffix.isSome then "MerlinVerify.Check.PolicyW8A8BoundedChain.check"
      else "MerlinVerify.Check.PolicyW8A8BoundedQualified.check")),
    ("theorem", toJson (if suffix.isSome then "MerlinVerify.Check.PolicyW8A8BoundedChain.check_sound"
      else "MerlinVerify.Check.PolicyW8A8BoundedQualified.check_sound")),
    ("verdict", toJson "abstracted"), ("index_bits", toJson ib),
    ("policy_scope", toJson (if config.scope == .captured then "captured" else "all")),
    ("input_domain", toJson "finite_f32_and_declared_integer_intervals_per_function"),
    ("floating_domain", toJson "byte_extension_with_explicit_source_and_dynamic_primitive_laws"),
    ("source_primitive_laws", toJson (7 : Nat)), ("additional_dynamic_primitive_laws", toJson (3 : Nat)),
    ("policy_reviewed", toJson false), ("primitive_model_laws_verified", toJson false),
    ("whole_pass_verified", toJson false), ("boundary_qualification_passed", toJson boundaryChecked),
    ("conditional_initialized_policy_refinement", toJson checked), ("functions", toJson details),
    ("suffix_checked", toJson boundaryChecked),
    ("suffix_match", if boundaryChecked then toJson suffixMatched else .null),
    ("suffix_boundaries", toJson links)]

def run (beforePath afterPath contractsPath width scope : String) (suffixPaths : List String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | do
    IO.eprintln "index_bits must be one positive integer"
    return 2
  if scope != "all" && scope != "captured" then
    IO.eprintln "policy scope must be all or captured"
    return 2
  let config : Rules.W8A8Reference.Config := ⟨if scope == "all" then .all else .captured⟩
  let decoded ← (beforePath :: afterPath :: suffixPaths).mapM fun path => do
    return Decode.module (← IO.FS.readFile path)
  let contracts := (Json.parse (← IO.FS.readFile contractsPath)).bind decodeContracts
  match decoded.mapM id, contracts with
  | .ok (before :: after :: rest), .ok contracts =>
    let names := Module.definedNames before
    if contracts.length != names.length || !names.all (fun name => (contracts.lookup name).isSome) then
      IO.eprintln "input contracts must name every actual source function exactly once"
      return 2
    IO.println (report config ib contracts before after (some rest)).compress
    -- Provisional policy/domain review remains a hard refusal even if every check passes.
    return 1
  | .error error, _ | _, .error error => IO.eprintln s!"decode error: {error}"; return 2
  | _, _ => IO.eprintln "missing policy boundary"; return 2

end MerlinVerify.Driver.W8A8Bounded
