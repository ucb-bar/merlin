/- Inspect actual source syntax and SSA declarations. This reports contract eligibility only;
it does not execute complete functions or issue source-definedness/policy receipts. -/
import MerlinVerify.Core.Json
import MerlinVerify.Rules.W8A8SourceContract
import MerlinVerify.Check.PolicyW8A8SourceFacts

open Lean MerlinVerify MerlinVerify.Rules
open MerlinVerify.Check

mutual
  def inspectOps (types : List (ValueId × Ty)) (facts : List ConstantFacts.Fact) : List Op → List Json × List Json
    | [] => ([], [])
    | actual@(.mk _ results _ _ _ regions _) :: rest => Id.run do
      let mut sites : List Json := []
      let mut refusals : List Json := []
      if actual.name.startsWith "quant_ext." then
        match W8A8SourceContract.inspect types actual with
        | none => refusals := [Json.mkObj [("op", toJson actual.name),
            ("results", toJson (actual.results.map (·.id)))]]
        | some c =>
          sites := [Json.mkObj [("op", toJson actual.name), ("results", toJson (actual.results.map (·.id))),
            ("shape", toJson c.shape), ("zero_width", toJson c.zeroWidth),
            ("calibration_parameters_checked", toJson (PolicyW8A8SourceFacts.checkParameters types facts actual))]]
      let nested := inspectRegions types facts regions
      let next := inspectOps (results.map (fun value => (value.id, value.ty)) ++ types)
        (PolicyW8A8SourceFacts.advance 64 facts actual) rest
      return (sites ++ nested.1 ++ next.1, refusals ++ nested.2 ++ next.2)
  termination_by operations => sizeOf operations
  def inspectRegions (types : List (ValueId × Ty)) (facts : List ConstantFacts.Fact) : List Region → List Json × List Json
    | [] => ([], [])
    | .mk blocks :: rest =>
      let current := inspectBlocks types facts blocks
      let next := inspectRegions types facts rest
      (current.1 ++ next.1, current.2 ++ next.2)
  termination_by regions => sizeOf regions
  def inspectBlocks (types : List (ValueId × Ty)) (facts : List ConstantFacts.Fact) : List Block → List Json × List Json
    | [] => ([], [])
    | .mk args ops :: rest =>
      let current := inspectOps (args.map (fun value => (value.id, value.ty)) ++ types)
        (BindingFacts.forget ConstantFacts.model (args.map (·.id)) facts) ops
      let next := inspectBlocks types facts rest
      (current.1 ++ next.1, current.2 ++ next.2)
  termination_by blocks => sizeOf blocks
end

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok source := Decode.module (← IO.FS.readFile path) | return 2
  let (sites, refusals) := inspectOps [] [] source.ops
  IO.println (Json.mkObj [("scope", toJson "source_contract_eligibility_only"),
    ("whole_pass_verified", toJson false), ("sites", toJson sites), ("refusals", toJson refusals),
    ("theorem", toJson "MerlinVerify.Rules.W8A8SourceContract.inspect_sound"),
    ("parameter_theorem", toJson "MerlinVerify.Check.PolicyW8A8SourceFacts.checkParameters_sound")]).compress
  return if refusals.isEmpty then 0 else 1
