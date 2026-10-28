/- Tensor-stage diagnostic adapter. Expected group dimensions are explicit contract inputs;
candidate maps never choose their own reference axis. No full policy receipt is emitted. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Pointwise
import MerlinVerify.Check.PolicyW8A8AbsMax

open Lean MerlinVerify
open MerlinVerify.Check
open PolicyW8A8Geometry

private def dimensions (text : String) : Option (List Nat) :=
  if text == "-" then some [] else (text.splitOn ",").mapM String.toNat?

private def checkActual (known : List ConstantFacts.Fact) (types : List (Nat × Ty))
    (kind : String) (leftDims rightDims : List Nat) (op : Op) : Bool := Id.run do
  if op.name != "linalg.generic" then return false
  let [result] := op.results | return false
  let some (.tensor shape elt) := vtyOf 64 result.ty | return false
  let some rawTypes := op.operands.mapM (fun id => types.lookup id) | return false
  let some argTypes := rawTypes.mapM (vtyOf 64) | return false
  let some (inputs, outputs) := Sem.Ops.Linalg.splitInsOuts op argTypes | return false
  if outputs != [.tensor shape elt] then return false
  let some maps := attrAffineMaps? op "indexing_maps" | return false
  let some iters := Sem.Ops.Linalg.iterKinds? op | return false
  let [region] := op.regions | return false
  let .mk [.mk _ body] := region | return false
  if kind == "absmax" then
    let [.tensor inputShape (.float .f32)] := inputs | return false
    if elt != .float .f32 || shape != parameterShape inputShape leftDims then return false
    return PolicyW8A8AbsMax.check inputShape leftDims maps iters region
  if kind == "scale" || kind == "reciprocal" then
    if elt != .float .f32 || inputs != [.tensor shape (.float .f32)] then return false
    if kind == "scale" then
      let some first := body[0]? | return false
      let [_, denom] := first.operands | return false
      return PolicyW8A8Pointwise.checkUnary known shape .scale denom maps iters region
    return PolicyW8A8Pointwise.checkUnary known shape .reciprocal 0 maps iters region
  if kind == "quantize" then
    if elt != .int 8 || inputs != [.tensor shape (.float .f32),
      .tensor (parameterShape shape leftDims) (.float .f32)] then return false
    let some hiOp := body[2]? | return false
    let some loOp := body[3]? | return false
    let [_, hi] := hiOp.operands | return false
    let [_, lo] := loOp.operands | return false
    return PolicyW8A8Pointwise.checkQuantize known shape leftDims hi lo maps iters region
  let some count := (kind.drop "requantize".length).toString.toNat? | return false
  if kind != "requantize" ++ toString count then return false
  if elt != .float .f32 || inputs != ([.tensor shape (.int 32)] ++
    ([.tensor (parameterShape shape leftDims) (.float .f32),
      .tensor (parameterShape shape rightDims) (.float .f32)].take count)) then return false
  return PolicyW8A8Pointwise.checkRequantize count shape leftDims rightDims maps iters region

def main (args : List String) : IO UInt32 := do
  let [path, selection, kind, left, right] := args | return 2
  let some leftDims := dimensions left | return 2
  let some rightDims := dimensions right | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let some f := m.func? "forward" | return 2
  let .mk [.mk blockArgs ops] := f.body | return 2
  let mut types := blockArgs.map (fun value => (value.id, value.ty))
  let mut known : List ConstantFacts.Fact := []
  let mut sites : Array Json := #[]
  let mut refused := 0
  for (op, index) in ops.zipIdx do
    let selected := if selection == "inspect" then
      op.attr? "prov.role" == some (.str "requant") else selection.toNat? == some index
    if selected then
      let accepted := checkActual known types kind leftDims rightDims op
      if !accepted then refused := refused + 1
      sites := sites.push (Json.mkObj [("index", toJson index), ("matches", toJson accepted)])
    known := ConstantFacts.advance 64 known op
    types := op.results.map (fun value => (value.id, value.ty)) ++ types
  if sites.isEmpty then return 2
  IO.println (Json.mkObj [("scope", toJson "w8a8_tensor_stages_only"), ("kind", toJson kind),
    ("whole_pass_verified", toJson false), ("refused", toJson refused), ("sites", toJson sites)]).compress
  return if refused == 0 then 0 else 1
