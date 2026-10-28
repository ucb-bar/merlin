/- Test adapter for the proved integer-body component, never a whole-pass policy receipt. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Body

open Lean MerlinVerify

-- Kernel-evaluated boundaries and adversarial byte patterns; no native proof oracle.
#guard (131071 : Int) * 16384 < 2147483648
#guard ¬ ((131072 : Int) * 16384 < 2147483648)
#guard Rules.W8A8Accumulation.dot [(128#8, 128#8)] = 16384
#guard (Rules.W8A8Accumulation.accumulate [(128#8, 128#8)]).toInt = 16384
#guard (Rules.W8A8Accumulation.accumulate [(128#8, 127#8), (127#8, 128#8)]).toInt = -32512
#guard (Rules.W8A8Accumulation.accumulate [(1#8, 1#8)] 2147483647#32).toInt = -2147483648

-- Integer execution never consults floating-point operations.
private def integerCtx : Ctx := {
  M := {
    op := fun _ k _ => 0#k.width
    cmp := fun _ _ _ _ => false
    convert := fun _ _ k _ => 0#k.width
    toInt := fun _ _ _ _ => none
    ofInt := fun _ _ k _ => 0#k.width },
  ext := fun _ _ => .stuck "no externals", indexBits := 64 }

def main (args : List String) : IO UInt32 := do
  let path :: selected :: samples := args | return 2
  let some index := selected.toNat? | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let some f := m.func? "forward" | return 2
  let .mk [.mk _ ops] := f.body | return 2
  let some generic := ops[index]? | return 2
  if generic.name != "linalg.generic" then return 2
  let [region] := generic.regions | return 2
  let accepted := Check.PolicyW8A8Body.check region
  let mut fields := [("scope", toJson "integer_body_only"), ("matches", toJson accepted)]
  if accepted && !samples.isEmpty then
    let [a, b, acc] := samples | return 2
    let some a := a.toInt? | return 2
    let some b := b.toInt? | return 2
    let some acc := acc.toInt? | return 2
    let actual := evalRegion Sem.Ops.opSem integerCtx (fun _ _ => .stuck "no calls") [] region
      [.scalar (.some (.bits 8 (BitVec.ofInt 8 a))),
        .scalar (.some (.bits 8 (BitVec.ofInt 8 b))),
        .scalar (.some (.bits 32 (BitVec.ofInt 32 acc)))]
    let .ok [.scalar (.some (.bits 32 result))] := actual | return 2
    fields := fields ++ [("value", toJson (toString result.toInt))]
  IO.println (Json.mkObj fields).compress
  return if accepted then 0 else 1
