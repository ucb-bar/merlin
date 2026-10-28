import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Sem.Ops
import MerlinVerify.Check.Types
import MerlinVerify.Check.Synth
import MerlinVerify.Check.Module
import MerlinVerify.Check.Rulebook
import MerlinVerify.Check.DefinedLinalg
import MerlinVerify.Check.DefinedExtended
import MerlinVerify.Check.TypedModule
import MerlinVerify.Driver.W8A8
import MerlinVerify.Driver.W8A8Chain
import MerlinVerify.Driver.Bf16F32Acc

open MerlinVerify
open Lean (Json toJson)

/-- Argument and result value types of function `fn`, from its `function_type` at index width `ib`. -/
def funcSig? (ib : Nat) (m : MerlinVerify.Module) (fn : String) : Option (List Check.VTy × List Check.VTy) :=
  Check.TypedModule.signature? ib m fn

/-- What a per-boundary verdict leaves open: its theorem is conditional refinement
(`FuncRefinesC`), which says nothing where the source is not modelled. Strict refinement needs the
capture's support (`Check.supportCheck_sound`) and the composed chain (`Check.chain_refines`). -/
def conditionalObligation : String :=
  "strict refinement needs source support: MerlinVerify.Check.supportCheck_sound on the capture, " ++
    "composed by MerlinVerify.Check.chain_refines"

def verdictOf (ok : Bool) : String := if ok then "verified" else "abstracted"

/-- Print one result object and return its exit code. -/
def emit (fields : List (String × Json)) (ok : Bool) : IO UInt32 := do
  IO.println (Json.mkObj fields).compress
  return (if ok then 0 else 1)

/-- `merlin-lean`: checker driver (design §8.3). Every result is one JSON object on stdout.
* `decode <in.json>` — validate a `merlin.lean.ir.v1` document and print a summary.
* `print <in.json> <out.mlir>` — decode and print generic MLIR (round trip, design D9).
* `check-c1 <before.json> <after.json> <fn>` — C1 on one function (`Check.c1Check_sound`).
* `check-module <before.json> <after.json>` — every defined function, plus outside-function changes
  (`Check.moduleCheck_sound`).
* `check-module-typed <before.json> <after.json> <index_bits>` — typed normalization followed by
  C1, on each function's declared input domain (`Check.TypedModule.check_sound`).
* `check-w8a8-draft <before.json> <after.json> <index_bits> <all|captured>` — source-derived
  policy matching plus separate definedness/support checks; remains abstracted pending review.
* `check-w8a8-chain-draft <before.json> <after.json> <index_bits> <all|captured> <suffix.json>...`
  — compose the draft policy with checked downstream boundaries on the same input domain.
* `check-bf16-f32acc <before.json> <after.json> <index_bits>` — the f32-accumulation contract
  (review candidate) for `lower_bf16_matmul_f32acc`; remains abstracted pending review.
* `check-bf16-f32acc-chain <before.json> <after.json> <index_bits> <suffix.json>...` — that
  boundary composed with checked downstream refinement boundaries on the typed input domain.
* `check-support <in.json> <fn> [index_bits]` — the function is inside the proved fragment
  (`Check.supportCheck_sound`).
* `check-definedness <in.json> <fn> [index_bits]` — successful, nonpoison results on typed,
  nonpoison inputs (`Check.Definedness.definedCheck_sound`).
* `check-definedness-extended <in.json> <fn> [index_bits]` — the same domain with independently
  proved value-dependent bounds (`Check.DefinedExtended.check_sound`).
The first two checks prove conditional refinement only; their output names the open obligation. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | ["check-w8a8-draft", beforeJ, afterJ, width, scope] =>
    Driver.W8A8.run beforeJ afterJ width scope
  | "check-w8a8-chain-draft" :: beforeJ :: afterJ :: width :: scope :: suffix =>
    Driver.W8A8Chain.run beforeJ afterJ width scope suffix
  | ["check-bf16-f32acc", beforeJ, afterJ, width] =>
    Driver.Bf16F32Acc.run beforeJ afterJ width
  | "check-bf16-f32acc-chain" :: beforeJ :: afterJ :: width :: suffix =>
    Driver.Bf16F32Acc.runChain beforeJ afterJ width suffix
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
    match Decode.module (← IO.FS.readFile beforeJ), Decode.module (← IO.FS.readFile afterJ) with
    | .ok src, .ok tgt =>
      let input := Check.synthC1 src tgt fn
      let sites := (input.script.filter fun | .site .. => true | .same => false).length
      let ok := Check.c1Check input
      emit ([("checker", toJson "C1"), ("theorem", toJson "MerlinVerify.Check.c1Check_sound"),
        ("claim_type", toJson "refine_conditional"), ("obligation", toJson conditionalObligation),
        ("function", toJson fn), ("verdict", toJson (verdictOf ok)),
        ("steps", toJson input.script.length), ("rule_sites", toJson sites)] ++
        (if ok then [] else [("reason", toJson ((Check.explainC1 input).getD "refused"))]))
        ok
    | .error e, _ | _, .error e => IO.eprintln s!"decode error: {e}"; return 2
  | ["check-module", beforeJ, afterJ] =>
    match Decode.module (← IO.FS.readFile beforeJ), Decode.module (← IO.FS.readFile afterJ) with
    | .ok src, .ok tgt =>
      let ok := Check.moduleCheck src tgt
      let out := Check.outsideChanges src tgt
      let verified := ok && out.unverified.isEmpty
      let reasons := (if ok then [] else ["a function body did not verify"]) ++ out.unverified
      emit ([("checker", toJson "module"), ("theorem", toJson "MerlinVerify.Check.moduleCheck_sound"),
        ("claim_type", toJson "refine_conditional"), ("obligation", toJson conditionalObligation),
        ("functions", toJson (Check.Module.definedNames src)), ("functions_verified", toJson ok),
        ("abi_changes", toJson out.abi), ("unverified_changes", toJson out.unverified),
        ("verdict", toJson (verdictOf verified))] ++
        (if verified then [] else [("reason", toJson ("; ".intercalate reasons))]))
        verified
    | .error e, _ | _, .error e => IO.eprintln s!"decode error: {e}"; return 2
  | ["check-module-typed", beforeJ, afterJ, width] =>
    let some ib := width.toNat?.filter (· > 0) | do
      IO.eprintln "index_bits must be one positive integer"
      return 2
    match Decode.module (← IO.FS.readFile beforeJ), Decode.module (← IO.FS.readFile afterJ) with
    | .ok src, .ok tgt =>
      let ok := Check.TypedModule.check ib src tgt
      let out := Check.outsideChanges src tgt
      let verified := ok && out.unverified.isEmpty
      let reasons := (if ok then [] else ["a function body did not verify on its typed input domain"]) ++ out.unverified
      emit ([("checker", toJson "module-typed"),
        ("theorem", toJson "MerlinVerify.Check.TypedModule.check_sound"),
        ("claim_type", toJson "refine_conditional"), ("index_bits", toJson ib),
        ("functions", toJson (Check.Module.definedNames src)), ("functions_verified", toJson ok),
        ("abi_changes", toJson out.abi), ("unverified_changes", toJson out.unverified),
        ("verdict", toJson (verdictOf verified))] ++
        (if verified then [] else [("reason", toJson ("; ".intercalate reasons))])) verified
    | .error e, _ | _, .error e => IO.eprintln s!"decode error: {e}"; return 2
  | command :: inp :: fn :: rest =>
    if command != "check-support" && command != "check-definedness" &&
        command != "check-definedness-extended" then
      IO.eprintln s!"unknown command: {command}"
      return 2
    let extended := command == "check-definedness-extended"
    let definedness := command == "check-definedness" || extended
    let width := match rest with
      | [] => some 64
      | [s] => s.toNat?.filter (· > 0)
      | _ => none
    let some ib := width | do
      IO.eprintln "index_bits must be one positive integer"
      return 2
    match Decode.module (← IO.FS.readFile inp) with
    | .ok m =>
      let (ok, why) := match funcSig? ib m fn with
        | none => (false, s!"@{fn}: no function with a modelled signature")
        | some (argTys, resTys) =>
          if definedness then
            let accepted := if extended then Check.DefinedExtended.check ib m fn argTys resTys
              else Check.Definedness.check (Check.Definedness.book ib) m fn argTys resTys
            if accepted then (true, "")
            else (false, "outside the proved definedness fragment or an output may be uninitialized")
          else
            let R := Check.merlinBook ib
            if Check.supportCheck R m fn argTys resTys then (true, "")
            else (false, (Check.diagnoseFunc R m fn argTys resTys).getD "refused")
      emit [("checker", toJson (if extended then "definedness-extended"
          else if definedness then "definedness" else "support")),
        ("theorem", toJson (if extended then "MerlinVerify.Check.DefinedExtended.check_sound"
          else if definedness then "MerlinVerify.Check.Definedness.definedCheck_sound"
          else "MerlinVerify.Check.supportCheck_sound")),
        ("claim_type", toJson (if definedness then "defined" else "not_stuck")),
        ("function", toJson fn), ("index_bits", toJson ib),
        ("verdict", toJson (if ok then "verified" else "unsupported")),
        ("first_refusal", if ok then Json.null else toJson why)] ok
    | .error e => IO.eprintln s!"decode error: {e}"; return 2
  | _ =>
    IO.eprintln "usage: merlin-lean decode <in.json> | print <in.json> <out.mlir> | check-c1 <before.json> <after.json> <fn> | check-module <before.json> <after.json> | check-module-typed <before.json> <after.json> <index_bits> | check-support <in.json> <fn> [index_bits] | check-definedness <in.json> <fn> [index_bits] | check-definedness-extended <in.json> <fn> [index_bits]"
    return 2
