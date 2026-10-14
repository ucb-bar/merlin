import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Sem.Ops
import MerlinVerify.Check.Types

open MerlinVerify

/-- `merlin-lean`: checker driver (design §8.3).
* `decode <in.json>` — validate a `merlin.lean.ir.v1` document and print a summary.
* `print <in.json> <out.mlir>` — decode and print generic MLIR (round trip, design D9). -/
def main (args : List String) : IO UInt32 := do
  match args with
  | ["decode", inp] =>
    match Decode.module (← IO.FS.readFile inp) with
    | .ok m =>
      IO.println s!"ok: {m.ops.length} top-level ops, {m.funcs.length} functions"
      return 0
    | .error e => IO.eprintln s!"decode error: {e}"; return 1
  | ["print", inp, out] =>
    match Decode.module (← IO.FS.readFile inp) with
    | .ok m => IO.FS.writeFile out (Print.module m); return 0
    | .error e => IO.eprintln s!"decode error: {e}"; return 1
  | _ =>
    IO.eprintln "usage: merlin-lean decode <in.json> | print <in.json> <out.mlir>"
    return 2
