/- Check actual emitted integer contraction bodies and maps, independently of provenance.
This is the generic-loop component theorem, not a whole-pass or zero-producer claim. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Projection

open Lean MerlinVerify MerlinVerify.Check MerlinVerify.Rules

def component (types : List (Nat × Ty)) (op : Op) : Option Json := do
  if op.name != "linalg.generic" || Sem.Ops.unmodelledAttr? op.blind != none || !op.successors.isEmpty then none
  let [left, right, destination] := op.operands | none
  let [result] := op.results | none
  let some (outputShape, .int 32) := result.ty.staticTensor? | none
  let some (leftShape, .int 8) := (← types.lookup left).staticTensor? | none
  let some (rightShape, .int 8) := (← types.lookup right).staticTensor? | none
  if (← types.lookup destination) != result.ty then none
  if Sem.Ops.Linalg.splitInsOuts op op.operands != some ([left, right], [destination]) then none
  let [body] := op.regions | none
  let maps ← attrAffineMaps? op "indexing_maps"
  let iters ← Sem.Ops.Linalg.iterKinds? op
  let layout ← W8A8ProjectionLayout.inspect leftShape rightShape outputShape maps iters
  return Json.mkObj [("checked", toJson (PolicyW8A8Projection.check layout body)),
    ("ranges", toJson layout.ranges), ("output_shape", toJson outputShape),
    ("reduction_size", toJson layout.reductionSize)]

def components (program : MerlinVerify.Module) : List Json := Id.run do
  let mut result := []
  for function in program.funcs do
    let .mk [.mk args ops] := function.body | continue
    let mut types := args.map fun value => (value.id, value.ty)
    for op in ops do
      if let some report := component types op then result := result ++ [report]
      types := (op.results.map fun value => (value.id, value.ty)) ++ types
  return result

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok program := Decode.module (← IO.FS.readFile path) | return 2
  IO.println (Json.mkObj [("checker", toJson "MerlinVerify.Check.PolicyW8A8Projection.check"),
    ("theorem", toJson "MerlinVerify.Check.PolicyW8A8Projection.check_sound"),
    ("components", toJson (components program))]).compress
  return 0
