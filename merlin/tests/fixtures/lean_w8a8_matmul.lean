/- Diagnostic adapter for the tensor-contraction component; never emits a whole-pass receipt. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Matmul

open Lean MerlinVerify

private def shapeOfByte : Ty → Option (Nat × Nat)
  | .tensor [some m, some k] (.int 8) none => some (m, k)
  | _ => none

private def checkActual (types : List (Nat × Ty)) (op : Op) : Bool := Id.run do
  if op.name != "linalg.generic" || op.operands.length != 3 then return false
  let some tys := op.operands.mapM (fun id => types.lookup id) | return false
  let [ta, tb, tc] := tys | return false
  let some (m, k) := shapeOfByte ta | return false
  let some (kb, n) := shapeOfByte tb | return false
  let out := Ty.tensor [some m, some n] (.int 32) none
  if kb != k || tc != out || op.results.map (·.ty) != [out] then return false
  let some maps := attrAffineMaps? op "indexing_maps" | return false
  let some iters := Sem.Ops.Linalg.iterKinds? op | return false
  let [region] := op.regions | return false
  return Check.PolicyW8A8Matmul.check m n k maps iters region

def main (args : List String) : IO UInt32 := do
  let [path, selection] := args | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let some f := m.func? "forward" | return 2
  let .mk [.mk blockArgs ops] := f.body | return 2
  let mut types := blockArgs.map (fun value => (value.id, value.ty))
  let mut sites : Array Json := #[]
  let mut refused := 0
  for (op, index) in ops.zipIdx do
    let selected := if selection == "inspect" then
      op.attr? "prov.role" == some (.str "contraction") else selection.toNat? == some index
    if selected then
      let accepted := checkActual types op
      if !accepted then refused := refused + 1
      sites := sites.push (Json.mkObj [("index", toJson index), ("matches", toJson accepted)])
    types := op.results.map (fun value => (value.id, value.ty)) ++ types
  if sites.isEmpty then return 2
  IO.println (Json.mkObj [("scope", toJson "w8a8_integer_tensor_component_only"),
    ("whole_pass_verified", toJson false), ("refused", toJson refused), ("sites", toJson sites)]).compress
  return if refused == 0 then 0 else 1
