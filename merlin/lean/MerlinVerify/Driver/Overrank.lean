/- Standalone Overrank driver. No Main/CLI/lakefile/SPEC_LOCK changes. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Check.Overrank
import MerlinVerify.Rules.OverrankIntake
open MerlinVerify
open Lean (Json toJson)

def report (before after : Module) : Json :=
  let p := Check.OverrankReference.plan before
  Json.mkObj [
    ("checker",toJson "MerlinVerify.Check.Overrank.check"),
    ("theorem",toJson "MerlinVerify.Check.Overrank.check_sound"),
    ("checked",toJson (Check.Overrank.check before after)),
    ("complete_output_match",toJson (Check.Overrank.completeCheck (Check.OverrankReference.reference before) after)),
    ("coverage_passed",toJson p.valid), ("covered_sites",toJson p.sites.length),
    ("sites",toJson (p.sites.map fun s => Json.mkObj [
      ("path",toJson s.path),("format",toJson (reprStr s.kind)),
      ("batch",toJson s.shape.batch),("k",toJson s.shape.k),("n",toJson s.shape.n),
      ("layout_checked",toJson (Rules.OverrankIntent.layoutCheck s.kind s.shape)),
      ("complete_kernel_execution_checked",toJson (Check.Overrank.kernelCheck s.kind s.shape after))])),
    ("intent_status",toJson "provisional_pending_human_approval"),
    ("literal_source_equivalence_claimed",toJson false),
    ("whole_capture_definedness_claimed",toJson false)]

def main (args : List String) : IO UInt32 := do
  match args with
  | ["print",input,output] =>
    let .ok m := Decode.module (← IO.FS.readFile input) | throw (IO.userError "invalid JSON export")
    IO.FS.writeFile output (Print.module m)
    return 0
  | [before,after] =>
    let .ok s := Decode.module (← IO.FS.readFile before) | throw (IO.userError "invalid before JSON")
    let .ok t := Decode.module (← IO.FS.readFile after) | throw (IO.userError "invalid after JSON")
    IO.println (report s t).compress
    return 0
  | _ => IO.eprintln "usage: Overrank.lean [print in out | before.json after.json]"; return 2
