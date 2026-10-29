/- Source-only admission diagnostics. This adapter grants no boundary proof,
policy approval, saved-receipt authority or compiler delivery authority. -/
import MerlinVerify.Driver.Bf16F32Acc
import MerlinVerify.Check.Rulebook
import MerlinVerify.Check.TypedModule
import MerlinVerify.Check.Module

namespace MerlinVerify.Driver.SourceProfile

open MerlinVerify Lean

def tensorRefusal (ib : Nat) (source : Module) : Option String := Id.run do
  let names := Check.Module.definedNames source
  if names.isEmpty then return some "no defined function in the source model"
  for name in names do
    let some (args, results) := Check.TypedModule.signature? ib source name
      | return some s!"@{name}: signature outside the typed model"
    let book := Check.merlinBook ib
    if !Check.supportCheck book source name args results then
      return some s!"@{name}: {(Check.diagnoseFunc book source name args results).getD "source support refused"}"
  return none

def maccRefusal (source : Module) : Option String := Id.run do
  if Bf16F32Acc.candidates source == 0 then return some "no top-level half-format contraction"
  if let some reason := Bf16F32Acc.ineligible? source then return some reason
  if (Bf16F32Acc.siteFormats source).length != 1 then
    return some "one operand format is required by the compiler contract profile"
  return none

def run (kind input width : String) : IO UInt32 := do
  if kind != "tensor" && kind != "macc" then
    IO.eprintln "source profile must be tensor or macc"
    return 2
  let some ib := width.toNat?.filter (· == 64) | do
    IO.eprintln "compiler source profiles require index64"
    return 2
  match Decode.module (← IO.FS.readFile input) with
  | .error reason => IO.eprintln s!"decode error: {reason}"; return 2
  | .ok source =>
    let reason := if kind == "tensor" then tensorRefusal ib source else maccRefusal source
    IO.println (Json.mkObj [
      ("schema", toJson "merlin.compiler-source-profile.v1"),
      ("profile", toJson kind), ("index_bits", toJson ib),
      ("supported", toJson reason.isNone), ("reason", toJson reason),
      ("boundary_proof", toJson false), ("policy_approval", toJson false)]).compress
    return if reason.isNone then 0 else 1

end MerlinVerify.Driver.SourceProfile
