/- Exercise the exact module-level dead-producer checker on authoritative exports. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.DeadModule

open Lean MerlinVerify MerlinVerify.Check

def main (args : List String) : IO UInt32 := do
  let [beforePath, afterPath, width] := args | return 2
  let some ib := width.toNat? | return 2
  let .ok before := Decode.module (← IO.FS.readFile beforePath) | return 2
  let .ok after := Decode.module (← IO.FS.readFile afterPath) | return 2
  let checked := DeadModule.check ib before after
  IO.println (Json.mkObj [("checker", toJson "MerlinVerify.Check.DeadModule.check"),
    ("theorem", toJson "MerlinVerify.Check.DeadModule.check_sound"),
    ("exact_module_execution_equality", toJson checked), ("index_bits", toJson ib)]).compress
  return if checked then 0 else 1
