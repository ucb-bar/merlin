/- Concrete integer-only differential-test driver. The arbitrary float model is NOT an IEEE oracle. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.Rulebook
import MerlinVerify.Tests.Semantics

open Lean MerlinVerify

def scalarJson : Val → Json
  | .poison => Json.mkObj [("kind", toJson "poison")]
  | .some (.bits w x) => Json.mkObj [("kind", toJson "integer"), ("width", toJson w),
      ("unsigned", toJson x.toNat)]
  | .some (.flt ..) => Json.mkObj [("kind", toJson "unmodelled_float")]

def valueJson : RVal → Json
  | .scalar s => scalarJson s
  | .tensor t => Json.mkObj [("kind", toJson "tensor"), ("shape", toJson t.shape),
      ("values", Json.arr ((allPoints t.shape).map (fun i => scalarJson (t.get i))).toArray)]

def outcomeJson : Outcome (List RVal) → Json
  | .ub => Json.mkObj [("tag", toJson "ub")]
  | .stuck why => Json.mkObj [("tag", toJson "stuck"), ("reason", toJson why)]
  | .ok vs => Json.mkObj [("tag", toJson "ok"), ("values", Json.arr (vs.map valueJson).toArray)]

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let mut results : Array Json := #[]
  for op in m.ops do
    let some fn := op.asFunc? | continue
    if fn.isDecl then continue
    results := results.push (Json.mkObj [("function", toJson fn.name),
      ("outcome", outcomeJson (Sem.Ops.denote Tests.Semantics.ctx m fn.name []))])
  IO.println (Json.arr results).compress
  return 0
