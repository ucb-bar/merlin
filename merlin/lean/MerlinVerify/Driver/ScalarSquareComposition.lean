/- Isolated executable report for provisional program claims. No approval or numerical-law
discharge can be supplied by a certificate or inferred from a structural match. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Rules.ScalarSquareClaims

open MerlinVerify
open Lean (Json toJson)

namespace ScalarSquareProgramReport

mutual
  def valuesOps : List Op → List Value
    | [] => []
    | (.mk _ results _ _ _ rs _) :: rest => results ++ valuesRegions rs ++ valuesOps rest
    termination_by ops => sizeOf ops
    decreasing_by all_goals simp_wf; all_goals omega
  def valuesRegions : List Region → List Value
    | [] => []
    | .mk bs :: rest => valuesBlocks bs ++ valuesRegions rest
    termination_by rs => sizeOf rs
    decreasing_by all_goals simp_wf; all_goals omega
  def valuesBlocks : List Block → List Value
    | [] => []
    | .mk args ops :: rest => args ++ valuesOps ops ++ valuesBlocks rest
    termination_by bs => sizeOf bs
    decreasing_by all_goals simp_wf; all_goals omega
end

def report (before after : Module) : Json := Id.run do
  let ib := 64
  let unsupported := (valuesOps before.ops).filter (fun v => (Check.vtyOf ib v.ty).isNone)
  let functions := before.funcs.map fun f => Json.mkObj [
    ("name", toJson f.name), ("declaration", toJson f.isDecl),
    ("entry_argument_types_supported", toJson (match f.body.blocks with
      | [block] => block.args.all fun v => (Check.vtyOf ib v.ty).isSome
      | _ => f.isDecl))]
  let checked := Check.ScalarSquareProgram.check before after
  return Json.mkObj [
    ("schema", toJson "merlin.lean.scalar-square.program-report.v1"),
    ("program_structure_checked", toJson checked),
    ("boundary_checked", toJson (Check.ScalarSquare.check before after)),
    ("function_inventory_checked", toJson (Check.ScalarSquareProgram.functionsCheck before.funcs after.funcs)),
    ("functions", toJson functions), ("call_fuel", toJson (before.funcs.length + 1)),
    ("covered_sites", toJson (Check.ScalarSquare.sites before).length),
    ("index_bits", toJson ib), ("unsupported_runtime_types", toJson unsupported.length),
    ("unsupported_type_examples", toJson (unsupported.take 8 |>.map fun v => Json.mkObj [
      ("id", toJson v.id), ("type", toJson (Print.ty v.ty))])),
    ("source_equality_theorem", toJson "MerlinVerify.Rules.ScalarSquareClaims.source_equal"),
    ("strict_source_claim", Json.mkObj [
      ("theorem", toJson "MerlinVerify.Rules.ScalarSquareClaims.strict_source_preservation"),
      ("instantiated", toJson false),
      ("entry_domain_inhabited", toJson "not_discharged"),
      ("context_domain", toJson "not_discharged"),
      ("numerical_law", toJson "undischarged_refuted_for_tested_native_implementation"),
      ("source_typing", toJson "not_discharged"),
      ("source_definedness", toJson "not_discharged")]),
    ("multiplication_policy_claim", Json.mkObj [
      ("theorem", toJson "MerlinVerify.Rules.ScalarSquareClaims.approved_multiplication_policy"),
      ("instantiated", toJson false), ("approved", toJson false),
      ("entry_domain_inhabited", toJson "not_discharged"),
      ("context_domain", toJson "not_discharged"),
      ("policy_definedness", toJson "not_discharged"),
      ("requires_pow_square_law", toJson false),
      ("original_source_equivalence_claimed", toJson false)]),
    ("verdict", toJson "abstracted")]

end ScalarSquareProgramReport

def main (args : List String) : IO UInt32 := do
  match args with
  | [before, after] =>
    let .ok source := Decode.module (← IO.FS.readFile before)
      | throw (IO.userError "source export did not decode")
    let .ok target := Decode.module (← IO.FS.readFile after)
      | throw (IO.userError "target export did not decode")
    IO.println (ScalarSquareProgramReport.report source target).compress
    return 0
  | _ =>
    IO.eprintln "usage: ScalarSquareComposition.lean before.json after.json"
    return 2
