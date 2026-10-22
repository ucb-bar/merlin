/- Inactive candidate driver. Acceptance decisions are executable Lean checkers.
   The census is an inventory, not a whole-function execution claim. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Rules.FrontendRuntimeEvidence
import MerlinVerify.Check.TypedModule

open MerlinVerify MerlinVerify.Check MerlinVerify.Rules Lean

namespace FrontendCheckpointDriver
mutual
  def allOps : List Op → List Op
    | [] => []
    | op@(.mk _ _ _ _ _ rs _) :: rest => op :: (regionOps rs ++ allOps rest)
  def regionOps : List Region → List Op
    | [] => []
    | .mk bs :: rest => blockOps bs ++ regionOps rest
  def blockOps : List Block → List Op
    | [] => []
    | .mk _ ops :: rest => allOps ops ++ blockOps rest
end

def site (types : FrontendBoolSimulation.Types) (op : Op) : Json :=
  Json.mkObj [("name", toJson op.name), ("results", toJson (op.results.map (·.id))),
    ("operands", toJson op.operands),
    ("operand_types", toJson (op.operands.map fun id => (types.lookup id).map MerlinVerify.Print.ty)),
    ("result_types", toJson (op.results.map fun v => MerlinVerify.Print.ty v.ty)),
    ("cast8_obstruction", toJson (opHeadEq FrontendRuntimeObligations.cast8 op)),
    ("token_obstruction", toJson (opHeadEq FrontendRuntimeObligations.tokenExtract op))]

def inventory (m : MerlinVerify.Module) : Json := Id.run do
  let ops := allOps m.ops
  let types := (FrontendBoolProgram.defsOps 64 m.ops).map FrontendBoolProgram.sourceTypes
  let mut census : List (String × Nat) := []
  for op in ops do
    census := (op.name, (census.lookup op.name).getD 0 + 1) :: census.filter (fun p => p.1 != op.name)
  let casts := ops.filter fun op => op.name == "arith.fptosi" || op.name == "arith.fptoui"
  let extracts := ops.filter fun op => op.name == "tensor.extract"
  let dynamic := ops.filter fun op => op.results.any fun v =>
    match v.ty with | .tensor shape _ _ => shape.any Option.isNone | _ => false
  return Json.mkObj [("kind", toJson "inventory; not whole-model execution proof"),
    ("operations", toJson ops.length), ("census", toJson census),
    ("float_integer_casts", toJson (casts.map (site (types.getD [])))),
    ("tensor_reads", toJson (extracts.map (site (types.getD [])))),
    ("dynamic_results", toJson (dynamic.map (site (types.getD []))))]

def readModule (path : String) : IO MerlinVerify.Module := do
  match Decode.module (← IO.FS.readFile path) with
  | .ok m => return m
  | .error why => throw (IO.userError why)

def field (j : Json) (key : String) : IO Json :=
  match Decode.field j key with
  | .ok v => pure v
  | .error why => throw (IO.userError why)

def string (j : Json) (key : String) : IO String := do
  match Decode.str (← field j key) with
  | .ok v => return v
  | .error why => throw (IO.userError why)

def natural (j : Json) (key : String) : IO Nat := do
  match Decode.nat (← field j key) with
  | .ok v => return v
  | .error why => throw (IO.userError why)

def run (j : Json) : IO Json := do
  unless (← string j "schema") == "merlin.frontend.runtime.request.v1" do
    throw (IO.userError "wrong candidate request schema")
  let mode ← string j "mode"
  let limit ← natural j "limit"
  let ib ← natural j "index_bits"
  unless ib == 64 do throw (IO.userError "candidate requires index64")
  let target ← readModule (← string j "after_export")
  if mode == "bounds_slice" then
    let name ← string j "function"
    let start ← natural j "start"
    let count ← natural j "count"
    return Json.mkObj [("accepted", toJson (FrontendRangeProjection.check target name start count limit)),
      ("whole_function_defined", toJson false), ("input_domain", toJson "typed initialized cut inputs"),
      ("theorem", toJson "MerlinVerify.Rules.FrontendRangeProjection.check_sound")]
  if mode == "bounds_defined" then
    let name ← string j "function"
    let accepted := match TypedModule.signature? 64 target name with
      | none => false
      | some (a,r) => Definedness.Refined.check (FrontendRangeDefined.candidateBook limit 64) target name a r
    return Json.mkObj [("accepted", toJson accepted),
      ("theorem", toJson "MerlinVerify.Rules.FrontendRangeDefined.check_sound")]
  if mode == "defined" then
    let name ← string j "function"
    let accepted := match TypedModule.signature? 64 target name with
      | none => false
      | some (a,r) => Definedness.check (FrontendPrefixDefined.candidateBook limit 64) target name a r
    return Json.mkObj [("accepted", toJson accepted),
      ("theorem", toJson "MerlinVerify.Rules.FrontendPrefixDefined.check_sound")]
  if mode == "inventory" then return inventory target
  let source ← readModule (← string j "before_export")
  let name ← string j "function"
  let start ← natural j "start"
  let .ok paths := Decode.arr (← field j "suffix_exports") | throw (IO.userError "invalid suffix list")
  let tail ← paths.mapM fun p => do
    let .ok path := Decode.str p | throw (IO.userError "invalid suffix path")
    readModule path
  let slices ← match j.getObjVal? "bounds_slices" with
    | .error _ => pure []
    | .ok raw => do
      let .ok rows := Decode.arr raw | throw (IO.userError "invalid bounds-slice list")
      rows.mapM fun row => do
        let start ← natural row "start"
        let count ← natural row "count"
        pure (start, count)
  let sliceResults := slices.map fun (begin, count) => Json.mkObj [
    ("start", toJson begin), ("count", toJson count),
    ("accepted", toJson (FrontendRangeProjection.check target name begin count limit)),
    ("input_domain", toJson "typed initialized cut inputs"),
    ("theorem", toJson "MerlinVerify.Rules.FrontendRangeProjection.check_sound")]
  let slicesAccepted := slices.all fun (begin, count) => FrontendRangeProjection.check target name begin count limit
  let repair := FrontendBoolSimulation.check source target
  let checkpoint := FrontendRuntimeCheckpoint.check target name start limit
  let accepted := decide (1024 ≤ limit) && repair &&
    FrontendRuntimeVerification.checkpoint source target name start limit
  let suffix := FrontendRuntimeChain.suffix target tail
  return Json.mkObj [("schema", toJson "merlin.frontend.runtime.result.v1"),
    ("active", toJson false), ("accepted", toJson (accepted && suffix && slicesAccepted)),
    ("bounds_slices", toJson sliceResults),
    ("repair", toJson repair), ("checkpoint", toJson checkpoint), ("suffix", toJson suffix),
    ("whole_function_defined", toJson false),
    ("claim", toJson "A2 f32 policy matching and defined actual initial segment"),
    ("theorems", toJson ["MerlinVerify.Rules.FrontendRuntimeVerification.check_sound",
      "MerlinVerify.Rules.FrontendRuntimeVerification.full_policy",
      "MerlinVerify.Rules.FrontendRuntimeChain.compose",
      "MerlinVerify.Rules.FrontendRangeProjection.check_sound",
      "MerlinVerify.Rules.FrontendMaskRange.check_sound"]),
    ("inventory", inventory target)]
end FrontendCheckpointDriver

def main (args : List String) : IO UInt32 := do
  let [path] := args | return 2
  let .ok request := Json.parse (← IO.FS.readFile path) | return 2
  try
    IO.println (← FrontendCheckpointDriver.run request).compress
    return 0
  catch error => IO.eprintln error.toString; return 2
