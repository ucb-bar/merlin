/- Inactive candidate execution adapter. Native comparisons are tests, not proof facts. -/
import MerlinVerify.Core.Json
import MerlinVerify.Rules.FrontendMaskCountProgram
import MerlinVerify.Rules.FrontendMaskAlias
import MerlinVerify.Rules.FrontendMaskWhole
import MerlinVerify.Rules.FrontendMaskProjection

open MerlinVerify Lean
open MerlinVerify.Rules
namespace FrontendMaskCandidate

def baseModel : FloatModel where
  op _ _ _ := 0
  cmp _ _ _ _ := false
  convert _ _ _ _ := 0
  toInt _ _ _ _ := none
  ofInt _ _ _ _ := 0

def scalar (ib : Nat) (ty : Ty) (bits : String) : Except String Val := do
  let some n := bits.toNat? | throw "input bits must be unsigned decimal strings"
  let some w := ty.bitWidth? ib | throw "unsupported input element type"
  unless 0 < w && w <= 64 && n < 2 ^ w do throw "input bits exceed element width (1..64)"
  let some s := scalarOfBits ib ty n | throw "input element decode failed"
  return .some s

def scalarJson (ib : Nat) (ty : Ty) (j : Json) : Except String Val :=
  if j == .null then pure .poison else do scalar ib ty (← Decode.str j)

def input (ib : Nat) (ty : Ty) (j : Json) : Except String RVal := do
  let raw <- Decode.arr j
  match ty with
  | .tensor dims elt none =>
    let some shape := dims.mapM id | throw "dynamic tensors are outside the fidelity adapter"
    unless raw.length == shapeSize shape do throw "wrong tensor input element count"
    let vs <- raw.mapM fun x => do scalarJson ib elt x
    return .tensor (Tensor.canon shape elt fun i => vs.toArray[linearIndex shape i]?.getD .poison)
  | .tensor .. => throw "encoded tensors are outside the fidelity adapter"
  | _ =>
    let [x] := raw | throw "scalar input must contain exactly one element"
    return .scalar (← scalarJson ib ty x)

def valBits : Val → Json
  | .poison => .null
  | .some (.bits _ x) => toJson (toString x.toNat)
  | .some (.flt _ x) => toJson (toString x.toNat)

def output : RVal → Json
  | .scalar v => Json.mkObj [("shape", .null), ("bits", toJson [valBits v])]
  | .tensor t => Json.mkObj [("shape", toJson t.shape),
      ("bits", toJson ((allPoints t.shape).map fun i => valBits (t.get i)))]

def outcome : Outcome (List RVal) → Json
  | .ok vs => Json.mkObj [("status", toJson "ok"), ("values", toJson (vs.map output))]
  | .ub => Json.mkObj [("status", toJson "ub")]
  | .stuck why => Json.mkObj [("status", toJson "stuck"), ("reason", toJson why)]

def noFloat : Ty → Bool
  | .float _ => false
  | .tensor _ elt _ | .memref _ elt _ _ | .vector _ _ elt => noFloat elt
  | .function _ _ => false
  | _ => true

mutual
  def integerOps : List Op → Bool
    | [] => true
    | .mk _ results _ _ _ regions _ :: rest => results.all (fun v => noFloat v.ty) &&
        integerRegions regions && integerOps rest
  def integerRegions : List Region → Bool
    | [] => true
    | .mk blocks :: rest => integerBlocks blocks && integerRegions rest
  def integerBlocks : List Block → Bool
    | [] => true
    | .mk args ops :: rest => args.all (fun v => noFloat v.ty) && integerOps ops && integerBlocks rest
end

mutual
  def allOps : List Op → List Op
    | [] => []
    | op@(.mk _ _ _ _ _ regions _) :: rest => op :: (allRegions regions ++ allOps rest)
  def allRegions : List Region → List Op
    | [] => []
    | .mk blocks :: rest => allBlocks blocks ++ allRegions rest
  def allBlocks : List Block → List Op
    | [] => []
    | .mk _ ops :: rest => allOps ops ++ allBlocks rest
end

def constantNat? (ops : List Op) (id : Nat) : Option Nat := do
  let some op := ops.find? (fun op => op.results.any (fun v => v.id == id)) | none
  if op.name != "arith.constant" then none else do
    let some (.int n .index) := op.attr? "value" | none
    if 0 ≤ n then some n.toNat else none

def inspectFor (ops : List Op) (op : Op) : Json := Id.run do
  let [lower, upper, stride, _, count] := op.operands | return toJson false
  let [region] := op.regions | return toJson false
  let some n := constantNat? ops upper | return toJson false
  return Json.mkObj [
    ("operation", toJson op.name), ("results", toJson (op.results.map (·.id))),
    ("extent", toJson n),
    ("controls", toJson (constantNat? ops lower == some 0 &&
      constantNat? ops stride == some 1 && count == lower)),
    ("attributes", toJson (Sem.Ops.unmodelledAttr? op.blind == none)),
    ("gather_body", toJson (FrontendMaskProgram.checkedBody false n region)),
    ("scatter_body", toJson (FrontendMaskProgram.checkedBody true n region))]

def countCheck (ops : List Op) (loop : Op) : Bool := Id.run do
  let [_, upper, _, initial, _] := loop.operands | return false
  let some n := constantNat? ops upper | return false
  let some alloc := ops.find? (fun op => op.results.any (fun v => v.id == initial)) | return false
  let [size] := alloc.operands | return false
  let some castPosition := ops.findIdx? (fun op => op.results.any (fun v => v.id == size)) | return false
  let provider := (ops.take (castPosition + 1)).drop (castPosition + 1 - 7)
  let [_, generic, zero, splat, _, _, cast] := provider | return false
  let [mask, _] := generic.operands | return false
  let [region] := loop.regions | return false
  let some (loopMask, _, _) := FrontendMaskProgram.bodyInputs? region | return false
  if loopMask != mask then return false
  let actual : Region := .mk [.mk [] (provider ++ [.mk "func.return" [] [size] [] [] [] []])]
  return FrontendMaskCountProgram.check n mask generic.attrs generic.props
    {zero := zero.attrs, splat := splat.attrs, cast := cast.attrs} actual

def shapeChain? (ops : List Op) (result : Nat) : Nat → Option (Nat × List Op)
  | 0 => some (result, [])
  | depth+1 => do
    let some op := ops.find? (fun op => op.results.any (fun v => v.id == result)) | none
    let [source] := op.operands | none
    if op.name != "tensor.collapse_shape" && op.name != "tensor.expand_shape" then none else do
      let (root, earlier) ← shapeChain? ops source depth
      pure (root, earlier ++ [op])

def maskAlias (ops : List Op) : Bool := Id.run do
  let [gather, scatter] := ops.filter (fun op => op.name == "scf.for") | return false
  let [g] := gather.regions | return false
  let [s] := scatter.regions | return false
  let some (gm, _, _) := FrontendMaskProgram.bodyInputs? g | return false
  let some (sm, _, _) := FrontendMaskProgram.bodyInputs? s | return false
  if gm == sm then
    let r : Region := .mk [.mk [] [.mk "func.return" [] [gm] [] [] [] []]]
    return FrontendMaskAlias.check gm r r
  let some (gr, gc) := shapeChain? ops gm 3 | return false
  let some (sr, sc) := shapeChain? ops sm 3 | return false
  if gr != sr then return false
  return FrontendMaskAlias.check gr
    (.mk [.mk [] (gc ++ [.mk "func.return" [] [gm] [] [] [] []])])
    (.mk [.mk [] (sc ++ [.mk "func.return" [] [sm] [] [] [] []])])

def projectionJson (m : MerlinVerify.Module) (name : String) (start : Nat) : Json :=
  match FrontendMaskProjection.project? m name start with
  | none => Json.mkObj [("source_function", toJson name), ("start", toJson start), ("accepted", toJson false)]
  | some cut => Json.mkObj [("source_function", toJson name), ("start", toJson start),
      ("accepted", toJson (FrontendMaskWhole.check cut.program)),
      ("source_block_operations", toJson cut.block.ops.length),
      ("selected_operations", toJson cut.selected.length),
      ("input_ids", toJson (cut.args.map (·.id))), ("output_ids", toJson (cut.outputs.map (·.id))),
      ("theorem", toJson "MerlinVerify.Rules.FrontendMaskProjection.check_sound")]

def inspect (m : MerlinVerify.Module) : Json :=
  let ops := allOps m.ops
  let providers := m.funcs.flatMap fun f => f.body.blocks.flatMap fun b =>
    (b.ops.filter fun op => op.name == "scf.for").map fun op => countCheck b.ops op
  Json.mkObj [("status", toJson "inactive candidate; conditional cut-domain proof"),
    ("body_theorem", toJson "MerlinVerify.Rules.FrontendMaskProgram.checkedBody_sound"),
    ("count_theorem", toJson "MerlinVerify.Rules.FrontendMaskCountProgram.check_sound"),
    ("execution_theorem", toJson "MerlinVerify.Rules.FrontendMaskExecution.selection_execution"),
    ("count_providers", toJson providers),
    ("whole_cut", toJson (FrontendMaskWhole.check m)),
    ("whole_cut_theorem", toJson "MerlinVerify.Rules.FrontendMaskWhole.check_sound"),
    ("projections", toJson [projectionJson m "forward" 91, projectionJson m "subject" 0]),
    ("mask_alias", toJson (maskAlias ops)),
    ("mask_alias_theorem", toJson "MerlinVerify.Rules.FrontendMaskAlias.check_sound"),
    ("loops", toJson ((ops.filter (fun op => op.name == "scf.for")).map (inspectFor ops)))]

def run (j : Json) : Except String Json := do
  let m ← Decode.module (← Decode.str (← Decode.field j "export"))
  if let .ok (.bool true) := j.getObjVal? "inspect" then return inspect m
  if let .ok source := j.getObjVal? "repair_export" >>= Decode.str then
    let src ← Decode.module source
    return Json.mkObj [("accepted", toJson (FrontendBoolSimulation.check src m)),
      ("claim", toJson "f32 frontend-intent repair under inactive runtime candidate"),
      ("theorem", toJson "MerlinVerify.Rules.FrontendBoolSimulationWith.check_sound_candidate")]
  unless integerOps m.ops do throw "mask execution adapter requires integer-only cut inputs and operations"
  let fn ← Decode.str (← Decode.field j "function")
  let ib ← Decode.nat (← Decode.field j "index_bits")
  unless ib == 64 do throw "mask candidate qualification uses index64"
  let limit ← Decode.nat (← Decode.field j "limit")
  let some op := m.ops.find? (fun o => (o.asFunc?.map (·.name)) == some fn) | throw "function missing"
  let some (.type (.function tys _)) := op.attr? "function_type" | throw "signature missing"
  let samples ← Decode.arr (← Decode.field j "samples")
  let ctx : Ctx := { M := FrontendF32Comparison.model baseModel, ext := fun _ _ => .stuck "no external oracle", indexBits := ib }
  let results ← samples.mapM fun sample => do
    let args ← Decode.arr sample
    unless args.length == tys.length do throw "wrong input argument count"
    let vs ← (tys.zip args).mapM fun (ty, arg) => input ib ty arg
    return outcome (MerlinVerify.denote (FrontendMaskRuntime.sem limit) ctx m fn vs)
  return toJson results

end FrontendMaskCandidate

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  match Json.parse (← IO.FS.readFile path) >>= FrontendMaskCandidate.run with
  | .ok result => IO.println result.compress; return 0
  | .error reason => IO.eprintln reason; return 2
