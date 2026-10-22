/- Separate provisional pass-local driver. Shared Main/CLI and frozen configuration are untouched. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Rules.LocalPassesChain
open MerlinVerify Lean
namespace LocalPassesDriver
mutual
  def allOps : List Op → List Op
    | [] => []
    | o@(.mk _ _ _ _ _ rs _) :: rest => o :: (regionOps rs ++ allOps rest)
  def regionOps : List Region → List Op
    | [] => []
    | .mk bs :: rest => blockOps bs ++ regionOps rest
  def blockOps : List Block → List Op
    | [] => []
    | .mk _ os :: rest => allOps os ++ blockOps rest
end

def readModule (path : String) : IO MerlinVerify.Module := do
  match Decode.module (← IO.FS.readFile path) with
  | .ok m => return m
  | .error why => throw (IO.userError why)

def inventory (m : MerlinVerify.Module) : Json :=
  let types := ((Rules.FrontendBoolProgram.defsOps 64 m.ops).getD []).map fun v => (v.id, v.ty)
  let casts := (allOps m.ops).filter fun o => o.name == "arith.sitofp" &&
    o.operands.any (fun x => types.lookup x == some (.int 1))
  Json.mkObj [("sites", toJson casts.length), ("formats", toJson
    (casts.map fun o => o.results.map fun r => MerlinVerify.Print.ty r.ty)),
    ("supported", toJson "scalar i1 to f32, including scalar bodies of tensor operations"),
    ("unsupported", toJson ["f16", "bf16", "f64", "direct tensor cast", "source numeric attributes"])]

def run (mode before after : String) (suffix : List String) : IO Json := do
  let src ← readModule before
  let tgt ← readModule after
  let tail ← suffix.mapM readModule
  if mode == "torchao-materialize" then
    return Json.mkObj [("mlir", toJson ((Rules.TorchAOAffineProgram.materialize? src).map MerlinVerify.Print.module))]
  if mode == "torchao-affine" then
    let bodies := (allOps tgt.ops).filterMap fun o =>
      if o.name == "linalg.generic" && o.attr? "prov.transforms" == some (.str "torchao_quantize_affine") then
        match o.regions with
        | [.mk [.mk args ops]] => some (Rules.TorchAOAffineScalar.checkBody args ops)
        | _ => some false
      else none
    return Json.mkObj [("schema", toJson "merlin.local-passes.result.v1"),
      ("accepted", toJson (Rules.LocalPassesChain.torchaoCheck src tgt tail)),
      ("suffix_count", toJson tail.length),
      ("status", toJson "materialized_contract_only"),
      ("source_call_refinement_proved", toJson false), ("contract_review", toJson "pending"),
      ("scalar_bodies", toJson bodies),
      ("theorem", toJson "MerlinVerify.Rules.LocalPassesChain.torchao_sound"),
      ("remaining", toJson ["capture version and recipe binding", "opaque-call/tensor execution bridge",
        "AbsReductionCorrect", "ZeroPointElisionCorrect", "independent specification review"])]
  unless mode == "bool-to-float" do throw (IO.userError "unsupported local pass")
  return Json.mkObj [("schema", toJson "merlin.local-passes.result.v1"),
    ("accepted", toJson (Rules.LocalPassesChain.boolCheck src tgt tail)),
    ("suffix_count", toJson tail.length),
    ("status", toJson "provisional_contract"), ("policy", toJson "frontend-bool-to-float"),
    ("ordinary_source_equivalence", toJson false), ("whole_function_defined", toJson false),
    ("theorem", toJson "MerlinVerify.Rules.LocalPassesChain.bool_sound"),
    ("inventory", inventory src)]
end LocalPassesDriver

def main (args : List String) : IO UInt32 := do
  let mode :: before :: after :: suffix := args | return 2
  try
    IO.println (← LocalPassesDriver.run mode before after suffix).compress
    return 0
  catch error => IO.eprintln error.toString; return 2
