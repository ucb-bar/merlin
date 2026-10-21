/- Executable integration for the proved W8A8 policy/suffix composition. Draft review status
remains a hard refusal, independently of all checked operational obligations. -/
import MerlinVerify.Driver.W8A8
import MerlinVerify.Check.PolicyW8A8Chain

namespace MerlinVerify.Driver.W8A8Chain

open MerlinVerify MerlinVerify.Check Lean

def report (config : Rules.W8A8Reference.Config) (ib : Nat)
    (before after : Module) (rest : List Module) : Json := Id.run do
  let boundaryReport := W8A8.report config ib before after
  let mut reasons : List String := []
  let mut abi : List String := []
  let mut links : List Json := []
  let mut suffixMatched := true
  match Rules.W8A8Reference.expand config before with
  | none => suffixMatched := false
  | some reference =>
    let functions := Module.definedNames reference
    let mut previous := after
    for next in rest do
      let outside := outsideChanges previous next
      let matched := functions.all (fun fn =>
        match TypedModule.signature? ib reference fn with
        | some (args, results) => PolicyW8A8Chain.stepCheck ib args results fn previous next
        | none => false) &&
        outside.unverified.isEmpty
      suffixMatched := suffixMatched && matched
      abi := abi ++ outside.abi
      reasons := reasons ++ outside.unverified
      if !matched then reasons := reasons ++ [s!"suffix boundary {links.length}: refinement refused"]
      links := links ++ [Json.mkObj [("index", toJson links.length),
        ("conditional_match", toJson matched), ("functions", toJson functions)]]
      previous := next
  let qualified := PolicyW8A8Chain.check config ib before after rest
  match boundaryReport with
  | .obj fields =>
    let fields := fields.insert "checker" (toJson "w8a8-chain-draft")
      |>.insert "theorem" (toJson "MerlinVerify.Check.PolicyW8A8Chain.check_sound")
      |>.insert "functions_verified" (toJson qualified)
      |>.insert "qualification_passed" (toJson qualified)
      |>.insert "suffix_match" (toJson suffixMatched)
      |>.insert "suffix_boundaries" (toJson links)
      |>.insert "suffix_abi_changes" (toJson abi)
      |>.insert "suffix_refusals" (toJson reasons)
    let reason := ((boundaryReport.getObjValAs? String "reason").toOption.getD "draft policy review required")
    return .obj (fields.insert "reason" (toJson ("; ".intercalate (reason :: reasons))))
  | _ => boundaryReport

def run (beforePath afterPath width scope : String) (suffixPaths : List String) : IO UInt32 := do
  let some ib := width.toNat?.filter (· > 0) | do
    IO.eprintln "index_bits must be one positive integer"
    return 2
  if scope != "all" && scope != "captured" then
    IO.eprintln "policy scope must be all or captured"
    return 2
  let config : Rules.W8A8Reference.Config := ⟨if scope == "all" then .all else .captured⟩
  let decoded ← (beforePath :: afterPath :: suffixPaths).mapM fun path => do
    return Decode.module (← IO.FS.readFile path)
  match decoded.mapM id with
  | .ok (before :: after :: rest) =>
    IO.println (report config ib before after rest).compress
    return 1
  | .ok _ => IO.eprintln "missing policy boundary"; return 2
  | .error error => IO.eprintln s!"decode error: {error}"; return 2

end MerlinVerify.Driver.W8A8Chain
