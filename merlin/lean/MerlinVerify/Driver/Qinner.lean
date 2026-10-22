/- Isolated Qinner driver. Does not alter the shared driver, CLI, or frozen specification. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Check.Qinner
import MerlinVerify.Check.QinnerPropagation
import MerlinVerify.Check.QinnerRuntime
import MerlinVerify.Rules.QinnerExecution
open MerlinVerify
open Lean (Json toJson)
def report (s t : Module) : Json :=
  let sites := Check.Qinner.sitesOps s.ops
  Json.mkObj [
    ("checked",toJson (Check.Qinner.check s t)),
    ("covered_sites",toJson sites.length),
    ("boundary_kind",toJson (if sites.isEmpty then "unchanged" else "lifting")),
    ("plan_checked",toJson (Check.Qinner.planCheck s t)),
    ("sites",toJson (sites.map fun site => Json.mkObj [
      ("key",toJson site.key),("source_value",toJson site.result.id),
      ("type",toJson (Print.ty site.result.ty))])),
    ("slots",toJson ((Check.Qinner.slots s t).map fun slot => Json.mkObj [
      ("key",toJson slot.key),("target_argument",toJson slot.arg.id),
      ("type",toJson (Print.ty slot.arg.ty))])),
    ("source_interpretation",toJson "explicit_named_runtime_bindings"),
    ("approval",toJson "pending_human_review"),
    ("original_poison_source_equivalence_claimed",toJson false),
    ("whole_model_definedness_claimed",toJson false)]
def blob (fuel : Nat) (j : Json) : Except String Check.QinnerRuntime.Blob := do
  return ⟨← Decode.str (← Decode.field j "key"), ← Decode.ty fuel (← Decode.field j "type"),
    ← Decode.hexBytes (← Decode.str (← Decode.field j "hex"))⟩
def main (args : List String) : IO UInt32 := do
  match args with
  | ["print",input,output] =>
    let .ok m := Decode.module (← IO.FS.readFile input) | throw (IO.userError "invalid JSON export")
    IO.FS.writeFile output (Print.module m)
    return 0
  | ["binding",before,after,certificate] =>
    let .ok s := Decode.module (← IO.FS.readFile before) | throw (IO.userError "invalid source JSON")
    let .ok t := Decode.module (← IO.FS.readFile after) | throw (IO.userError "invalid target JSON")
    let input ← IO.FS.readFile certificate
    let .ok j := Json.parse input | throw (IO.userError "invalid binding JSON")
    let decoded : Except String (List Check.QinnerRuntime.Blob × List Check.QinnerRuntime.Blob) := do
      let archive ← (← Decode.arr (← Decode.field j "archive")).mapM (blob input.length)
      let delivered ← (← Decode.arr (← Decode.field j "delivered")).mapM (blob input.length)
      return (archive,delivered)
    let .ok (archive,delivered) := decoded | throw (IO.userError "invalid binding records")
    IO.println (Json.mkObj [("checked",toJson (Check.QinnerRuntime.check archive (Check.Qinner.slots s t) delivered)),
      ("full_bytes_compared",toJson true), ("archive_authenticity_proved",toJson false)]).compress
    return 0
  | [mode,before,after] =>
    let .ok s := Decode.module (← IO.FS.readFile before) | throw (IO.userError "invalid source JSON")
    let .ok t := Decode.module (← IO.FS.readFile after) | throw (IO.userError "invalid target JSON")
    if mode == "propagate" then
      IO.println (Json.mkObj [("checked",toJson (Check.QinnerPropagation.check s t)),
        ("metadata_only",toJson (Rules.QinnerPropagation.check s t)),
        ("runtime_irrelevance_claimed",toJson false)]).compress
    else if mode == "lift" then IO.println (report s t).compress
    else return 2
    return 0
  | _ => IO.eprintln "usage: Qinner.lean [print in out | source.json target.json]"; return 2
