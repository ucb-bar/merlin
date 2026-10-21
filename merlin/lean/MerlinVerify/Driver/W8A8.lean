/- Reporting adapter for W8A8 draft integration. Independent policy review and statement pins
are intentionally not represented by a user-supplied success flag. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Module

namespace MerlinVerify.Driver.W8A8

open MerlinVerify MerlinVerify.Check Lean

def report (config : Rules.W8A8Reference.Config) (ib : Nat) (before after : Module) : Json := Id.run do
  let mut reasons := ["W8A8 reference policy is a draft: independent specification review and statement pins are required"]
  let mut functions : List String := []
  let mut abi : List String := []
  let mut details : List Json := []
  let matched := PolicyW8A8Module.check config before after
  let qualified := PolicyW8A8Module.qualify config ib before after
  match Rules.W8A8Reference.expand config before with
  | none => reasons := reasons ++ ["source outside the current source-derived W8A8 reference"]
  | some reference =>
    let identical := ExactSyntax.modules reference after
    functions := Module.definedNames reference
    let outside := outsideChanges reference after
    abi := outside.abi
    reasons := reasons ++ outside.unverified
    for fn in functions do
      let functionMatch := identical || c1Check (synthC1 reference after fn)
      let (sourceDefined, referenceSupported, why) :=
        match TypedModule.signature? ib before fn, TypedModule.signature? ib reference fn with
        | some (a, r), some (a', r') =>
          if a = a' ∧ r = r' then
            let defined := DefinedExtended.check ib before fn a r
            let supported := supportCheck (merlinBook ib) reference fn a r
            let refusal := if supported then "" else
              (diagnoseFunc (merlinBook ib) reference fn a r).getD "reference support refused"
            (defined, supported, refusal)
          else (false, false, "policy reference changes the declared input/result domain")
        | _, _ => (false, false, "missing or unsupported function signature")
      if !functionMatch then reasons := reasons ++
        [s!"@{fn}: " ++ (explainC1 (synthC1 reference after fn)).getD "function does not match the policy reference"]
      if !sourceDefined then reasons := reasons ++ [s!"@{fn}: source definedness has not been established"]
      if !referenceSupported then reasons := reasons ++ [s!"@{fn}: {why}"]
      details := details ++ [Json.mkObj [("function", toJson fn),
        ("conditional_match", toJson functionMatch), ("source_defined", toJson sourceDefined),
        ("reference_supported", toJson referenceSupported), ("first_refusal", toJson why)]]
  return Json.mkObj [("checker", toJson "w8a8-draft"),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8Module.qualify_sound"),
    ("claim_type", toJson "policy"), ("policy", toJson "w8a8-reference-draft"),
    ("policy_scope", toJson (if config.scope == .captured then "captured" else "all")),
    ("index_bits", toJson ib), ("functions", toJson functions), ("functions_verified", toJson qualified),
    ("conditional_match", toJson matched), ("qualification_passed", toJson qualified),
    ("policy_reviewed", toJson false), ("whole_pass_verified", toJson false),
    ("function_obligations", toJson details), ("abi_changes", toJson abi),
    ("unverified_changes", toJson reasons), ("reason", toJson ("; ".intercalate reasons)),
    ("verdict", toJson "abstracted")]

def run (beforePath afterPath width scope : String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | do
    IO.eprintln "index_bits must be one positive integer"
    return 2
  if scope != "all" && scope != "captured" then
    IO.eprintln "policy scope must be all or captured"
    return 2
  let config : Rules.W8A8Reference.Config := ⟨if scope == "all" then .all else .captured⟩
  match Decode.module (← IO.FS.readFile beforePath), Decode.module (← IO.FS.readFile afterPath) with
  | .ok before, .ok after =>
    IO.println (report config ib before after).compress
    return 1
  | .error error, _ | _, .error error =>
    IO.eprintln s!"decode error: {error}"
    return 2

end MerlinVerify.Driver.W8A8
