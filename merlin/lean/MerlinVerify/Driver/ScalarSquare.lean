/- Standalone driver: deliberately does not modify Main, the shared CLI or the locked lakefile. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Check.ScalarSquare

open MerlinVerify
open Lean (Json toJson)

def scalarSquareReport (before after : Module) : Json := Id.run do
  let plan := Check.ScalarSquare.operations [] [] 0 before.ops
  let reference : Module := { before with ops := plan.ops }
  let expected := Check.ScalarSquare.candidates before
  let coverage := decide (plan.sites.map (·.path) = expected)
  let output := Check.ExactSyntax.modules reference after
  let pairs := plan.sites.all (Check.ScalarSquare.pairCheck before after)
  let details := plan.sites.map fun site => Json.mkObj [
    ("path", toJson site.path), ("base", toJson site.base), ("exponent", toJson site.exponent),
    ("result", toJson site.result.id), ("format", toJson (reprStr site.kind)),
    ("pow_key", toJson (reprStr (Sem.Ops.Arith.floatKey site.source))),
    ("mul_key", toJson (reprStr (Sem.Ops.Arith.floatKey site.target))),
    ("exponent_bits", toJson (Rules.ScalarSquareContract.two site.kind).toNat)]
  return Json.mkObj [
    ("checker", toJson "MerlinVerify.Check.ScalarSquare.check"),
    ("theorem", toJson "MerlinVerify.Check.ScalarSquare.check_sound"),
    ("checked", toJson (coverage && output && pairs)), ("complete_output_match", toJson output),
    ("actual_pairs_checked", toJson pairs),
    ("coverage_passed", toJson coverage), ("candidate_paths", toJson expected),
    ("covered_sites", toJson plan.sites.length), ("sites", toJson details),
    ("numerical_law", toJson "MerlinVerify.Rules.ScalarSquareContract.Law"),
    ("numerical_law_status", toJson "provisional_not_discharged"),
    ("verdict", toJson "abstracted"), ("whole_capture_definedness_claimed", toJson false)]

def main (args : List String) : IO UInt32 := do
  match args with
  | ["print", input, output] =>
    let .ok source := Decode.module (← IO.FS.readFile input)
      | throw (IO.userError "export did not decode")
    IO.FS.writeFile output (Print.module source)
    return 0
  | [before, after] =>
    let .ok source := Decode.module (← IO.FS.readFile before)
      | throw (IO.userError "source export did not decode")
    let .ok target := Decode.module (← IO.FS.readFile after)
      | throw (IO.userError "target export did not decode")
    let report := scalarSquareReport source target
    IO.println report.compress
    return 0
  | _ =>
    IO.eprintln "usage: ScalarSquare.lean before.json after.json"
    return 2
