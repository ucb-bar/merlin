/- Evaluate actual exported tensor programs in the explicit model used by the new proof.
This adapter supplies inputs and serializes outcomes; it reuses the proved source evaluator. -/
import MerlinVerify.Driver.W8A8Ordered

open Lean MerlinVerify MerlinVerify.Rules

def input (ty : Ty) (raw : Json) : Except String RVal := do
  let .tensor dims (.float .f32) none := ty | throw "ordered oracle expects unencoded f32 tensors"
  let some shape := dims.mapM id | throw "ordered oracle expects static shapes"
  let words ← (← raw.getArr?).toList.mapM fun value => do
    let text ← value.getStr?
    let some word := text.toNat? | throw "input bits must be unsigned decimal strings"
    unless word < 2^32 do throw "input bits exceed f32 width"
    return W8A8OrderedBody.value (BitVec.ofNat 32 word)
  unless words.length = shape.prod do throw "wrong tensor element count"
  return .tensor (Tensor.canon shape (.float .f32)
    fun point => words.toArray[linearIndex shape point]?.getD .poison)

def bits : Val → Json
  | .some (.flt .f32 word) => toJson (toString word.toNat)
  | _ => .null

def output : RVal → Json
  | .scalar value => Json.mkObj [("shape", .null), ("bits", toJson [bits value])]
  | .tensor tensor => Json.mkObj [("shape", toJson tensor.shape),
      ("bits", toJson ((allPoints tensor.shape).map fun point => bits (tensor.get point)))]

def outcome : Outcome (List RVal) → Json
  | .ok values => Json.mkObj [("status", toJson "ok"), ("values", toJson (values.map output))]
  | .ub => Json.mkObj [("status", toJson "ub")]
  | .stuck reason => Json.mkObj [("status", toJson "stuck"), ("reason", toJson reason)]

def evaluate (document : MerlinVerify.Module) (request : Json) : Except String Json := do
  Decode.exactKeys "ordered tensor oracle" request ["function", "samples"] []
  let fn ← (← request.getObjVal? "function").getStr?
  let some function := document.func? fn | throw "unknown function"
  let .mk [.mk arguments _] := function.body | throw "expected one entry block"
  let ctx := Check.PolicyW8A8IEEEContext.context W8A8IEEEPrimitives.unsupported
    (fun name _ => .stuck s!"uninterpreted external {name}") 64
  let samples ← (← (← request.getObjVal? "samples").getArr?).toList.mapM fun sample => do
    let actual ← sample.getArr?
    unless actual.size = arguments.length do throw "wrong argument count"
    let inputs ← (arguments.zip actual.toList).mapM fun (argument, raw) => input argument.ty raw
    return outcome (MerlinVerify.denote W8A8SourceSemantics.opSem ctx document fn inputs)
  return toJson samples

def main (args : List String) : IO UInt32 := do
  let [exportPath, requestPath] := args | return 2
  let document := Decode.module (← IO.FS.readFile exportPath)
  let request := Json.parse (← IO.FS.readFile requestPath)
  match document.bind (fun document => request.bind (evaluate document)) with
  | .ok value => IO.println value.compress; return 0
  | .error reason => IO.eprintln reason; return 2
