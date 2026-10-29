# ruff: noqa: E501
"""Static reuse of the reviewed generic model and complete-entry assembler.
The generated header contains actual TILE bounds; no synthetic source module.
"""

from merlin.verify.lean_graph_state_pipeline import controls_literal

from .lean_graph_generic_prepare import program, types


def header(prepared):
    capture = prepared["capture"]
    lower, upper, step, tile = (prepared["bounds"][name] for name in ("lower", "upper", "step", "tile"))
    registry = (
        "["
        + ",".join(
            "{inputs := " + types(row["inputs"]) + ",outputs := " + types(row["outputs"]) + "}"
            for row in capture["registry"]
        )
        + "]"
    )
    control_capture = {"stages": [{"registry": capture["registry"]}], "index_operations": prepared["index_operations"]}
    controls = controls_literal(control_capture)
    text = f"""import Merlin.GraphStatePipelineValueCandidate
import Merlin.GraphStatePipelineGenericContextCandidate
import Merlin.GraphStatePipelineControlRolesCandidate
import Merlin.GraphStatePipelineTail
import Merlin.GraphStatePipelineProgram
import Merlin.GraphStatePipelineLiterals
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
namespace Merlin.GraphStatePipelineInstance
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
namespace Carrier
export ValueCarrierCandidate (Codec External)
end Carrier
def registry : List Signature := {registry}
def controls : Controls := {controls}
theorem controlsChecked : checkControl registry controls = true := by decide
def rawRegistrySize : Nat := {prepared["roles"]["raw_registry_size"]}
theorem ordinaryChecked : (List.range rawRegistrySize).all
    (fun identity => decide (controlAt controls identity = none)) = true := by decide
theorem ordinary_control {{identity : Nat}} (inside : identity < rawRegistrySize) :
    controlAt controls identity = none := by
  exact of_decide_eq_true (List.all_eq_true.mp ordinaryChecked identity (List.mem_range.mpr inside))
theorem literalsChecked : checkLiterals 64 controls = true := by decide
def segmentBounds : Merlin.GraphLoopsSegment.Bounds :=
  {{bits := 64,lower := {lower},upper := {upper},step := {step},tile := {tile}}}
theorem boundsChecked : segmentBounds.Valid := by decide
theorem generated_controls {{outer : Nat}} (inside : outer < segmentBounds.upper) :
    IndexOp.eval .add [Int.ofNat outer,Int.ofNat (segmentBounds.step*segmentBounds.tile)] =
      Int.ofNat (outer+segmentBounds.step*segmentBounds.tile) ∧
    IndexOp.eval .minui [Int.ofNat (outer+segmentBounds.step*segmentBounds.tile),Int.ofNat segmentBounds.upper] =
      Int.ofNat (min (outer+segmentBounds.step*segmentBounds.tile) segmentBounds.upper) ∧
    signedRepresentative segmentBounds.bits (Int.ofNat (outer+segmentBounds.step*segmentBounds.tile)) ∧
    signedRepresentative segmentBounds.bits (Int.ofNat (min (outer+segmentBounds.step*segmentBounds.tile) segmentBounds.upper)) :=
  segment_control_meaning boundsChecked inside
def M {{Value World Error : Type}} (codec : Carrier.Codec Value) (default : Value)
    (primitive : Prim → List Value → Value) (external : Carrier.External Value World Error registry) :
    Model Value World Error registry :=
  ValueCarrierCandidate.model codec (checkControl_sound controlsChecked) default primitive external
variable {{Value World Error : Type}} (codec : Carrier.Codec Value) (default : Value)
  (primitive : Prim → List Value → Value) (external : Carrier.External Value World Error registry)
abbrev m : Model Value World Error registry := M codec default primitive external
theorem m_default : (m codec default primitive external).default = default := rfl
theorem m_index (value : Int) : (m codec default primitive external).index value = codec.encode value := rfl
theorem m_iterations (lower upper stride : Value) :
    (m codec default primitive external).iterations lower upper stride =
      indexIterations (codec.decode lower) (codec.decode upper) (codec.decode stride) := rfl
"""
    names = []
    for row in prepared["index_operations"]:
        identity, kind = row["identity"], row["kind"]
        op = (
            f"(.constant ({row['value']}))" if kind == "index-constant" else ".add" if kind == "index-add" else ".minui"
        )
        name = f"control{identity}"
        names.append(name)
        text += f"""theorem {name} (arguments : List Value) (world : World) :
    opaqueStep (m codec default primitive external) {identity} arguments world =
      .ok ([codec.encode (IndexOp.eval {op} (arguments.map codec.decode))],world) :=
  ValueCarrierCandidate.control_effect codec (checkControl_sound controlsChecked) default primitive external
    (by decide : controlAt controls {identity} = some {op}) arguments world
"""
    text += """theorem naturalRange (lower upper step : Nat) (positive : 0 < step) :
    indexIterations (Int.ofNat lower) (Int.ofNat upper) (Int.ofNat step) =
      (scfRange lower upper step).map Int.ofNat := by
  change indexIterations (lower : Int) (upper : Int) (step : Int) = _
  have pos : (0 : Int) < (step : Int) := by omega
  simp only [indexIterations,Int.natCast_nonneg,pos,and_self,↓reduceIte,Int.toNat_natCast]
"""
    return text, names


def footer(prepared):
    text = ""
    fi, capture = prepared["function"], prepared["capture"]
    count, slots = prepared["frame_count"], prepared["captures"]
    prefix = [*prepared["prefix"], {"ret": []}]
    role_plans = []
    for plan in prepared["roles"]["plans"]:
        role_plans.append(
            "["
            + ",".join(
                "{occurrence := "
                + str(mark["occurrence"])
                + ",original := "
                + str(mark["original"])
                + ",selected := "
                + str(mark["selected"])
                + "}"
                for mark in plan
            )
            + "]"
        )
    raw_rests = [stage[fi]["body"][len(prepared["prefix"]) :] for stage in prepared["roles"]["unroled_functions"]]
    text += f"""def Prefix : Program := {program(prefix)}
def SourceRest : Program := {program(prepared["rests"][0])}
def TargetRest : Program := {program(prepared["rests"][1])}
def RawSourceRest : Program := {program(raw_rests[0])}
def RawTargetRest : Program := {program(raw_rests[1])}
def sourceRoles : List ControlRolesCandidate.Mark := {role_plans[0]}
def targetRoles : List ControlRolesCandidate.Mark := {role_plans[1]}
theorem sourceRolesChecked : ControlRolesCandidate.apply sourceRoles RawSourceRest = some SourceRest := by decide
theorem targetRolesChecked : ControlRolesCandidate.apply targetRoles RawTargetRest = some TargetRest := by decide
theorem sourceRoleTypes : sourceRoles.all
    (fun mark => decide (signature registry mark.original = signature registry mark.selected)) = true := by decide
theorem targetRoleTypes : targetRoles.all
    (fun mark => decide (signature registry mark.original = signature registry mark.selected)) = true := by decide
def Captures : List Nat := [{",".join(map(str, slots))}]
theorem sourceScopeChecked : Scope.check registry (vars {count}) (Captures.map Expr.var) SourceRest MinimalSource = true := by decide
theorem targetScopeChecked : Scope.check registry (vars {count}) (Captures.map Expr.var) TargetRest MinimalTarget = true := by decide
theorem rest (environment : List Value) (world : World) (size : environment.length = {count}) :
    evalProgram (m codec default primitive external) SourceRest environment world =
      evalProgram (m codec default primitive external) TargetRest environment world := by
  rw [Context.scope_closure (m codec default primitive external) {count} Captures sourceScopeChecked environment world size,
    Context.scope_closure (m codec default primitive external) {count} Captures targetScopeChecked environment world size]
  exact minimal codec default primitive external _ world
"""
    original_entries, final_entries = [], []
    for i, (before, after) in enumerate(zip(*capture["functions"], strict=True)):
        if before["declaration"]:
            continue
        text += f"""namespace F{i}
def inputs : List Ty := {types(before["inputs"])}
def outputs : List Ty := {types(before["outputs"])}
def original : Program := {program(before["body"])}
def final : Program := {program(after["body"])}
theorem originalTyped : checkTypes registry inputs outputs original = true := by decide
theorem finalTyped : checkTypes registry inputs outputs final = true := by decide
"""
        if i == fi:
            raw_original, raw_final = [stage[i]["body"] for stage in prepared["roles"]["unroled_functions"]]
            text += f"""def rawOriginal : Program := {program(raw_original)}
def rawFinal : Program := {program(raw_final)}
theorem rawSourceJoin : Context.plug Prefix RawSourceRest = rawOriginal := rfl
theorem rawTargetJoin : Context.plug Prefix RawTargetRest = rawFinal := rfl
"""
            text += f"""theorem sourceJoin : Context.plug Prefix SourceRest = original := rfl
theorem targetJoin : Context.plug Prefix TargetRest = final := rfl
theorem growth : inputs.length + Context.frameGrowth registry Prefix = {count} := by decide
theorem actual (arguments : List Value) (world : World) (arity : arguments.length = inputs.length) :
    evalProgram (m codec default primitive external) original arguments world =
      evalProgram (m codec default primitive external) final arguments world := by
  rw [← sourceJoin,← targetJoin]
  apply Context.plug_congr (m codec default primitive external) Prefix inputs.length
  · rw [growth]
    exact rest codec default primitive external
  · exact arity
"""
        else:
            text += """theorem same : original = final := rfl
theorem actual (arguments : List Value) (world : World) (arity : arguments.length = inputs.length) :
    evalProgram (m codec default primitive external) original arguments world =
      evalProgram (m codec default primitive external) final arguments world := by rw [same]
"""
        text += f"end F{i}\n"
        original_entries.append(f"{{inputs := F{i}.inputs,outputs := F{i}.outputs,body := F{i}.original}}")
        final_entries.append(f"{{inputs := F{i}.inputs,outputs := F{i}.outputs,body := F{i}.final}}")
    text += "def originalEntries : List Entry := [" + ",".join(original_entries) + "]\n"
    text += "def finalEntries : List Entry := [" + ",".join(final_entries) + "]\n"
    text += (
        "theorem complete : PreservesEntries (m codec default primitive external) originalEntries finalEntries := by\n"
    )
    text += "  unfold originalEntries finalEntries\n"
    for i, row in enumerate(capture["functions"][0]):
        if not row["declaration"]:
            text += f"  apply entries_cons (m codec default primitive external) (first := {{inputs := F{i}.inputs,outputs := F{i}.outputs,body := F{i}.original}}) (second := {{inputs := F{i}.inputs,outputs := F{i}.outputs,body := F{i}.final}}) rfl rfl (F{i}.actual codec default primitive external)\n"
    text += "  exact entries_nil (m codec default primitive external)\n"
    text += """theorem complete3 : PreservesEntries (m codec default primitive external) originalEntries finalEntries :=
  complete codec default primitive external
theorem completeData {Payload World Error : Type}
    (primitive : Prim → List (Int ⊕ Payload) → (Int ⊕ Payload))
    (external : Carrier.External (Int ⊕ Payload) World Error registry) :
    PreservesEntries (M (ValueCarrierCandidate.sumCodec Payload) (Sum.inl 0) primitive external)
      originalEntries finalEntries :=
  complete (ValueCarrierCandidate.sumCodec Payload) (Sum.inl 0) primitive external
end Merlin.GraphStatePipelineInstance
"""
    return text
