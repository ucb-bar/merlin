import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Sem.Ops
import MerlinVerify.Check.Types
import MerlinVerify.Check.Synth
import MerlinVerify.Check.Module
import MerlinVerify.Check.Rulebook

open MerlinVerify

/-- A JSON string literal. -/
def jsonStr (s : String) : String :=
  "\"" ++ (s.toList.map fun c => match c with
    | '"' => "\\\"" | '\\' => "\\\\" | '\n' => "\\n" | c => c.toString).foldl (· ++ ·) "" ++ "\""

/-- Argument and result value types of function `fn`, from its `function_type` at index width `ib`. -/
def funcSig? (ib : Nat) (m : Module) (fn : String) : Option (List Check.VTy × List Check.VTy) :=
  m.ops.findSome? fun op =>
    match op.asFunc?, op.attr? "function_type" with
    | some f, some (.type (.function ins outs)) =>
      if f.name == fn then
        match ins.mapM (Check.vtyOf ib), outs.mapM (Check.vtyOf ib) with
        | some a, some r => some (a, r)
        | _, _ => none
      else none
    | _, _ => none

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
  | "check-support" :: inp :: fn :: rest =>
    -- The capture is inside the proved fragment (`Check.supportCheck_sound`): never stuck on typed
    -- inputs at index width `ib` (default 64).
    let ib := (rest.head? >>= String.toNat?).getD 64
    match Decode.module (← IO.FS.readFile inp) with
    | .ok m =>
      let (ok, why) := match funcSig? ib m fn with
        | none => (false, s!"@{fn}: no function with a modelled signature")
        | some (argTys, resTys) =>
          let R := Check.merlinBook ib
          if Check.supportCheck R m fn argTys resTys then (true, "")
          else (false, (Check.diagnoseFunc R m fn argTys resTys).getD "refused")
      IO.println s!"\{\"checker\": \"support\", \"theorem\": \"MerlinVerify.Check.supportCheck_sound\", \"claim_type\": \"not_stuck\", \"function\": {jsonStr fn}, \"index_bits\": {ib}, \"verdict\": \"{if ok then "verified" else "unsupported"}\", \"first_refusal\": {if ok then "null" else jsonStr why}}"
      return (if ok then 0 else 1)
    | .error e => IO.eprintln s!"decode error: {e}"; return 2
  | _ =>
    IO.eprintln "usage: merlin-lean decode <in.json> | print <in.json> <out.mlir> | check-c1 <before.json> <after.json> <fn> | check-module <before.json> <after.json> | check-support <in.json> <fn> [index_bits]"
    return 2
