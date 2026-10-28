/- Diagnostic adapter for actual floating initialization and reduction components. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8InitializedAbsMax

open Lean MerlinVerify MerlinVerify.Check

private def checkActual (pre : List Op) (types : List (Nat × Ty)) (op : Op)
    (kind : String) (kept : List Nat) : Bool := Id.run do
  if pre.length < 3 then return false
  let initializers := pre.drop (pre.length - 3)
  if kind == "fill" then
    if op.name != "func.return" then return false
    let [destination] := op.operands | return false
    let some ty := types.lookup destination | return false
    let some (shape, .float .f32) := ty.staticTensor? | return false
    return PolicyW8A8InitializedAbsMax.checkZero shape initializers destination
  if kind != "absmax" || op.name != "linalg.generic" then return false
  let [result] := op.results | return false
  let some (outShape, .float .f32) := result.ty.staticTensor? | return false
  let [source, destination] := op.operands | return false
  let some sourceTy := types.lookup source | return false
  let some (shape, .float .f32) := sourceTy.staticTensor? | return false
  if outShape != PolicyW8A8Geometry.parameterShape shape kept then return false
  let some outputTy := types.lookup destination | return false
  if outputTy != result.ty then return false
  if Sem.Ops.Linalg.splitInsOuts op op.operands != some ([source], [destination]) then return false
  let some maps := attrAffineMaps? op "indexing_maps" | return false
  let some iters := Sem.Ops.Linalg.iterKinds? op | return false
  let [region] := op.regions | return false
  return PolicyW8A8InitializedAbsMax.checkZero outShape initializers destination &&
    PolicyW8A8AbsMax.check shape kept maps iters region

def main (args : List String) : IO UInt32 := do
  let [path, selection, kind, dimensions] := args | return 2
  let some kept := if dimensions == "-" then some [] else
    (dimensions.splitOn ",").mapM String.toNat? | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let some f := m.func? "forward" | return 2
  let .mk [.mk blockArgs ops] := f.body | return 2
  let mut types := blockArgs.map (fun value => (value.id, value.ty))
  let mut sites : Array Json := #[]
  let mut refused := 0
  for (op, index) in ops.zipIdx do
    if selection.toNat? == some index then
      let accepted := checkActual (ops.take index) types op kind kept
      if !accepted then refused := refused + 1
      sites := sites.push (Json.mkObj [("index", toJson index), ("matches", toJson accepted)])
    types := op.results.map (fun value => (value.id, value.ty)) ++ types
  if sites.isEmpty then return 2
  IO.println (Json.mkObj [("scope", toJson "w8a8_initialized_absmax_components_only"),
    ("whole_pass_verified", toJson false), ("refused", toJson refused), ("sites", toJson sites)]).compress
  return if refused == 0 then 0 else 1
