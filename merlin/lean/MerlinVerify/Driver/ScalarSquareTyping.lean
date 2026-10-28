/- Replay driver. The diagnostic counts explain refusals; only the sound checker discharges
SourceTyping. No numerical or multiplication-policy approval is inferred. -/
import MerlinVerify.Core.Json
import MerlinVerify.Core.Print
import MerlinVerify.Rules.ScalarSquareTypingClosed

open MerlinVerify MerlinVerify.Rules.ScalarSquareTyping
open Lean (Json toJson)
namespace ScalarSquareTypingReport
mutual
  def censusOps : List Op → List Op
    | [] => []
    | op@(.mk _ _ _ _ _ rs _) :: rest => op :: censusRegions rs ++ censusOps rest
  def censusRegions : List Region → List Op
    | [] => []
    | .mk blocks :: rest => censusBlocks blocks ++ censusRegions rest
  def censusBlocks : List Block → List Op
    | [] => []
    | .mk _ ops :: rest => censusOps ops ++ censusBlocks rest
end

def report (source : Module) : Json := Id.run do
  let ib := 64
  let allOps := censusOps source.ops
  let unsupportedAttrs := allOps.filter fun op => (Sem.Ops.unmodelledAttr? op.blind).isSome && op.name != "func.func"
  let quant := allOps.filter fun op => (op.name.startsWith "quant_ext.") && (Sem.Ops.Dequant.config? op.blind).isNone
  let checked := programCheck ib source
  let closed := closedCheck ib source
  return Json.mkObj [
    ("schema", toJson "merlin.lean.scalar-square.typing-report.v1"),
    ("index_bits", toJson ib), ("functions", toJson source.funcs.length),
    ("operations_including_functions_and_terminators", toJson allOps.length),
    ("rewritten_source_sites", toJson (Check.ScalarSquare.sites source).length),
    ("program_typing_checked", toJson checked), ("closed_capture_checked", toJson closed),
    ("source_typing_discharged", toJson closed),
    ("typing_soundness", toJson "MerlinVerify.Rules.ScalarSquareTyping.closedCheck_sound"),
    ("result_preservation", toJson "MerlinVerify.Rules.ScalarSquareTyping.closed_results"),
    ("external_declarations", toJson (source.funcs.filter (·.isDecl)).length),
    ("unmodelled_attribute_operations", toJson unsupportedAttrs.length),
    ("unmodelled_attribute_examples", toJson (unsupportedAttrs.take 8 |>.map fun op => Json.mkObj [
      ("operation", toJson op.name), ("attribute", toJson ((Sem.Ops.unmodelledAttr? op.blind).getD ""))])),
    ("quantization_operations", toJson (allOps.filter fun op => op.name.startsWith "quant_ext.").length),
    ("refused_quantization_configs", toJson quant.length),
    ("source_definedness_proved", toJson false), ("successful_execution_proved", toJson false),
    ("poison_freedom_proved", toJson false), ("numerical_law_discharged", toJson false),
    ("native_numerical_law_refuted", toJson true), ("policy_approved", toJson false),
    ("original_source_equivalence_claimed", toJson false), ("verdict", toJson "typing_only")]
end ScalarSquareTypingReport

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
    let .ok source := Decode.module (← IO.FS.readFile path)
      | throw (IO.userError "source export did not decode")
    IO.println (ScalarSquareTypingReport.report source).compress
    return 0
  | _ => IO.eprintln "usage: ScalarSquareTyping.lean source.json"; return 2
