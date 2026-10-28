/- Diagnostic initializer adapter; not a whole-policy receipt. -/
import MerlinVerify.Core.Json
import MerlinVerify.Check.PolicyW8A8Zero

open Lean MerlinVerify

private def checkActual (pre : List Op) (op : Op) : Bool := Id.run do
  let [{ ty := .tensor [some m, some n] (.int 32) none, .. }] := op.results | return false
  let [_, _, destination] := op.operands | return false
  if pre.length < 3 then return false
  return Check.PolicyW8A8Zero.check m n (pre.drop (pre.length - 3)) destination

def main (args : List String) : IO UInt32 := do
  let [path, selection] := args | return 2
  let .ok m := Decode.module (← IO.FS.readFile path) | return 2
  let some f := m.func? "forward" | return 2
  let .mk [.mk _ ops] := f.body | return 2
  let mut sites : Array Json := #[]
  let mut refused := 0
  for (op, index) in ops.zipIdx do
    let selected := if selection == "inspect" then
      op.attr? "prov.role" == some (.str "contraction") else selection.toNat? == some index
    if selected then
      let accepted := checkActual (ops.take index) op
      if !accepted then refused := refused + 1
      sites := sites.push (Json.mkObj [("index", toJson index), ("matches", toJson accepted)])
  if sites.isEmpty then return 2
  IO.println (Json.mkObj [("scope", toJson "w8a8_zero_initializer_only"),
    ("whole_pass_verified", toJson false), ("refused", toJson refused), ("sites", toJson sites)]).compress
  return if refused == 0 then 0 else 1
