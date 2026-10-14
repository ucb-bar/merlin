import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Sem.Ops
import MerlinVerify.Check.Types
import MerlinVerify.Check.Synth
import MerlinVerify.Check.Module

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
  | ["check-c1", beforeJ, afterJ, fn] =>
    -- Verdict of C1 (rule-instance checker) for one function; claim: `Check.c1Check_sound`.
    match Decode.module (← IO.FS.readFile beforeJ), Decode.module (← IO.FS.readFile afterJ) with
    | .ok src, .ok tgt =>
      let input := Check.synthC1 src tgt fn
      let sites := (input.script.filter fun | .site .. => true | .same => false).length
      let ok := Check.c1Check input
      IO.println s!"\{\"checker\": \"C1\", \"theorem\": \"MerlinVerify.Check.c1Check_sound\", \"claim_type\": \"refine\", \"function\": \"{fn}\", \"verdict\": \"{if ok then "verified" else "abstracted"}\", \"steps\": {input.script.length}, \"rule_sites\": {sites}}"
      return (if ok then 0 else 1)
    | .error e, _ | _, .error e => IO.eprintln s!"decode error: {e}"; return 2
  | ["check-module", beforeJ, afterJ] =>
    -- Every defined function refines (`Check.moduleCheck_sound`), plus the outside-claim accounting.
    match Decode.module (← IO.FS.readFile beforeJ), Decode.module (← IO.FS.readFile afterJ) with
    | .ok src, .ok tgt =>
      let ok := Check.moduleCheck src tgt
      let out := Check.outsideChanges src tgt
      let verified := ok && out.unverified.isEmpty
      let strs (xs : List String) := "[" ++ ", ".intercalate (xs.map fun x => "\"" ++ x ++ "\"") ++ "]"
      IO.println s!"\{\"checker\": \"module\", \"theorem\": \"MerlinVerify.Check.moduleCheck_sound\", \"claim_type\": \"refine\", \"functions\": {strs (Check.Module.definedNames src)}, \"functions_verified\": {ok}, \"abi_changes\": {strs out.abi}, \"unverified_changes\": {strs out.unverified}, \"verdict\": \"{if verified then "verified" else "abstracted"}\"}"
      return (if verified then 0 else 1)
    | .error e, _ | _, .error e => IO.eprintln s!"decode error: {e}"; return 2
  | _ =>
    IO.eprintln "usage: merlin-lean decode <in.json> | print <in.json> <out.mlir> | check-c1 <before.json> <after.json> <fn> | check-module <before.json> <after.json>"
    return 2
