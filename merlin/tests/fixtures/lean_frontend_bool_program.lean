/- Execution qualification of the actual source-policy and actual after program.
This is a test adapter; the universal theorem is FrontendBoolSimulation.check_sound_ieee. -/
import MerlinVerify.Core.Json
import MerlinVerify.Rules.FrontendBoolQualified

open MerlinVerify Lean
open MerlinVerify.Rules

private def ctx : Ctx := {
  M := FrontendF32Comparison.model FrontendBoolFptosi.alwaysFalse
  ext := fun _ _ => .stuck "test has no external calls"
  indexBits := 64 }

private def scalar? (sample : String) : Option Val :=
  if sample = "poison" then some .poison else
    sample.toNat?.map fun n => .some (.flt .f32 (BitVec.ofNat 32 n))

private def observeVal : Val → Json
  | .poison => Json.null
  | .some (.bits _ b) => toJson b.toNat
  | .some (.flt _ _) => toJson "unexpected floating result"

private def observe : Outcome (List RVal) → Json
  | .ub => toJson "ub"
  | .stuck reason => Json.mkObj [("stuck", toJson reason)]
  | .ok vs => Json.arr ((vs.flatMap fun
      | .scalar v => [observeVal v]
      | .tensor t => (allPoints t.shape).map fun p => observeVal (t.get p)).toArray)

def main (args : List String) : IO UInt32 := do
  let before :: after :: mode :: samples := args | return 2
  let .ok src := Decode.module (← IO.FS.readFile before) | return 2
  let .ok tgt := Decode.module (← IO.FS.readFile after) | return 2
  let some values := samples.mapM scalar? | return 2
  let matched := FrontendBoolSimulation.check src tgt
  if !matched then return 1
  let inputs := if mode == "scalar" then values.map fun v => [RVal.scalar v]
    else [[RVal.tensor (Tensor.canon [values.length] (.float .f32)
      fun p => values[p.headD 0]?.getD .poison)]]
  let source := inputs.map fun xs => observe (FrontendBoolSimulation.policyDenote src ctx "forward" xs)
  let target := inputs.map fun xs => observe (denote Sem.Ops.opSem ctx tgt "forward" xs)
  IO.println (Json.mkObj [
    ("checker", toJson "frontend-source-policy-execution"), ("matched", toJson matched),
    ("source", toJson source), ("target", toJson target),
    ("model", toJson "proposed IEEE f32 comparison instance"),
    ("qualification", toJson "execution samples; not a universal proof")]).compress
  return 0
