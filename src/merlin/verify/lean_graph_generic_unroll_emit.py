# ruff: noqa: E501
# Preserve the generated literal Lean statements for source/proof review.
"""Actual complete-entry generic-carrier proof emitter, staged/uncompiled."""

from __future__ import annotations

from merlin.verify.lean_graph_state_pipeline import controls_literal

from .lean_graph_generic_prepare import program, types


def emit(prepared):
    capture = prepared["capture"]
    bounds = prepared["bounds"]
    lower, upper, step = (bounds[name] for name in ("lower", "upper", "step"))
    stride, bulk_upper, groups, trips = (bounds[name] for name in ("stride", "bulk_upper", "groups", "trips"))
    fi = prepared["function"]
    left, right = prepared["minimal"]
    source_loops = [row["loop"] for row in left if "loop" in row]
    target_loops = [row["loop"] for row in right if "loop" in row]
    if not prepared["captures"] or len(source_loops) != 1 or len(target_loops) != 2:
        raise ValueError("first concrete generic atom needs finite captures and bulk+tail")
    if any(loop["carries"] or loop["initial"] for loop in (*source_loops, *target_loops)):
        raise ValueError("first concrete generic atom has no loop carries")
    source, bulk, tail = source_loops[0], *target_loops
    source_factored = program(left).replace(program(source["body"]), "Bsource", 1)
    target_factored = (
        program(right).replace(program(bulk["body"]), "Bbulk", 1).replace(program(tail["body"]), "Btail", 1)
    )
    tick = source["body"][0]["effect"]
    tensor_slot = source["body"][0]["arguments"][1] - 4
    returned_slot = left[-1]["ret"][0] - 3
    if source["body"] != [{"effect": tick, "arguments": [0, tensor_slot + 4], "outputs": []}, {"ret": []}]:
        raise ValueError("actual source tensor-capturing partial-effect body required")
    # Every actual instruction remains in the literal definitions below; these
    # shape checks select an initial emitter, not authority over target semantics.
    if len(left) != 5 or len(right) != 8:
        raise ValueError("actual three-control setup and two-loop target required")
    tensor_value = f"(data[{tensor_slot}]?.getD default)"
    returned_value = f"(data[{returned_slot}]?.getD default)"
    tail_left = "[" + ",".join(f"(.var {i})" for i in [0, 3, 4, 5, *range(6, 6 + len(prepared["captures"]))]) + "]"
    tail_right = "[" + ",".join(f"(.var {i})" for i in range(6 + len(prepared["captures"]))) + "]"
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
import Merlin.GraphStatePipelineReachableCandidate
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
def unrollBounds : Merlin.GraphLoopsUnroll.Bounds :=
  {{bits := 64,lower := {lower},upper := {upper},step := {step},factor := 2,count := {trips},groups := {groups},remainder := 1}}
theorem boundsChecked : unrollBounds.Valid := by decide
theorem reachable_source {{index : Int}}
    (member : index ∈ indexIterations unrollBounds.lower unrollBounds.upper unrollBounds.step) :
    ∃ natural, index = Int.ofNat natural ∧ natural < unrollBounds.upper ∧
      signedRepresentative unrollBounds.bits index ∧
      signedRepresentative unrollBounds.bits (Int.ofNat (natural+unrollBounds.step)) :=
  ValueCarrierCandidate.Reachable.source_control boundsChecked member
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
    text += f"""theorem tick_effect (arguments : List Value) (world : World) :
    opaqueStep (m codec default primitive external) {tick} arguments world =
      match external.run {tick} arguments world with
      | .error (error,final) => .error (.external error,final)
      | .ok result => .ok result := by
  simp only [m,M,ValueCarrierCandidate.model,opaqueStep,ValueCarrierCandidate.effect,
    show controlAt controls {tick} = none by decide]
  rfl
@[irreducible] def Bsource : Program := {program(source["body"])}
@[irreducible] def Bbulk : Program := {program(bulk["body"])}
@[irreducible] def Btail : Program := {program(tail["body"])}
def MinimalSource : Program := {program(left)}
def MinimalTarget : Program := {program(right)}
def FactoredSource : Program := {source_factored}
def FactoredTarget : Program := {target_factored}
theorem sourceFactored : MinimalSource = FactoredSource := by
  unfold MinimalSource FactoredSource Bsource
  rfl
theorem targetFactored : MinimalTarget = FactoredTarget := by
  unfold MinimalTarget FactoredTarget Bbulk Btail
  rfl
def Esource (data : List Value) : List Value :=
  ([codec.encode {step},codec.encode {upper},codec.encode {lower}]++data)
def Ebulk (data : List Value) : List Value :=
  ([codec.encode {bulk_upper},codec.encode {stride},codec.encode {step},codec.encode {upper},codec.encode {lower}]++data)
def O (data : List Value) (index : Int) (carry : List Value) (world : World) : Outcome Value World Error :=
  evalProgram (m codec default primitive external) Bsource (codec.encode index::carry++Esource codec data) world
def U (data : List Value) (index : Int) (carry : List Value) (world : World) : Outcome Value World Error :=
  evalProgram (m codec default primitive external) Bbulk (codec.encode index::carry++Ebulk codec data) world
def T (data : List Value) (index : Int) (carry : List Value) (world : World) : Outcome Value World Error :=
  evalProgram (m codec default primitive external) Btail (codec.encode index::carry++Ebulk codec data) world
def TailLeft : List Expr := {tail_left}
def TailRight : List Expr := {tail_right}
theorem tailScopeChecked : Scope.check registry TailLeft TailRight Bsource Btail = true := by
  unfold Bsource Btail
  decide
theorem tail_same (data : List Value) (index : Int) (carry : List Value) (world : World)
    (arity : carry.length = 0) :
    T codec default primitive external data index carry world =
      O codec default primitive external data index carry world := by
  have empty : carry = [] := List.length_eq_zero_iff.mp arity
  subst carry
  simp only [T,O,Bsource,Btail,Esource,Ebulk,evalProgram,reads,m_default,List.cons_append,
    List.nil_append,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some]
"""
    simps = ",".join(
        [
            "O,U,Bsource,Bbulk,Esource,Ebulk",
            *names,
            "m_default,m_index,m_iterations,tick_effect,IndexOp.eval,evalProgram,reads",
            "List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero",
            "List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind",
            "List.length_cons,List.length_nil,codec.roundTrip,loopEval,↓reduceIte",
        ]
    )
    text += f"""theorem body_group (data : List Value) (base : Int) (carry : List Value)
    (world : World) (arity : carry.length = 0) :
    U codec default primitive external data base carry world =
      loopEval 0 (O codec default primitive external data) [base,base+{step}] carry world := by
  have empty : carry = [] := List.length_eq_zero_iff.mp arity
  subst carry
  simp only [{simps}]
  cases first : external.run {tick} [codec.encode base,{tensor_value}] world with
  | error failure => simp only [first,Except.bind]
  | ok result =>
    rcases result with ⟨values,firstWorld⟩
    have size : values.length = 0 := by
      have actual := external.arity {tick} [codec.encode base,{tensor_value}] world values firstWorld first
      simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_nil] using actual
    have empty : values = [] := List.length_eq_zero_iff.mp size
    subst values
    simp only [first,Except.bind,List.nil_append,{simps}]
    cases second : external.run {tick} [codec.encode (base+{step}),{tensor_value}] firstWorld with
    | error failure => simp only [second,Except.bind]
    | ok result =>
      rcases result with ⟨values,lastWorld⟩
      have size : values.length = 0 := by
        have actual := external.arity {tick} [codec.encode (base+{step}),{tensor_value}] firstWorld values lastWorld second
        simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_nil] using actual
      have empty : values = [] := List.length_eq_zero_iff.mp size
      subst values
      simp only [second,Except.bind,List.nil_append,{simps}]
theorem source_function (data : List Value) :
    (fun index values state => evalProgram (m codec default primitive external) Bsource
      (codec.encode index::(values++codec.encode {step}::codec.encode {upper}::codec.encode {lower}::data)) state) =
      O codec default primitive external data := by
  funext index values state
  simp only [O,Esource,List.cons_append,List.nil_append] <;> rfl
theorem bulk_function (data : List Value) :
    (fun index values state => evalProgram (m codec default primitive external) Bbulk
      (codec.encode index::(values++codec.encode {bulk_upper}::codec.encode {stride}::codec.encode {step}::codec.encode {upper}::codec.encode {lower}::data)) state) =
      U codec default primitive external data := by
  funext index values state
  simp only [U,Ebulk,List.cons_append,List.nil_append] <;> rfl
theorem tail_function (data : List Value) :
    (fun index values state => evalProgram (m codec default primitive external) Btail
      (codec.encode index::(values++codec.encode {bulk_upper}::codec.encode {stride}::codec.encode {step}::codec.encode {upper}::codec.encode {lower}::data)) state) =
      T codec default primitive external data := by
  funext index values state
  simp only [T,Ebulk,List.cons_append,List.nil_append] <;> rfl
theorem grouped (data : List Value) (world : World) :
    loopEval 0 (O codec default primitive external data) (indexIterations {lower} {upper} {step}) [] world =
      (loopEval 0 (U codec default primitive external data) (indexIterations {lower} {bulk_upper} {stride}) [] world).bind
        (fun result => loopEval 0 (O codec default primitive external data)
          (indexIterations {bulk_upper} {upper} {step}) result.1 result.2) := by
  exact Tail.ordered_bulk_tail 0 _ _ {lower} {upper} {step} 2 {groups} 1 (by decide) (by decide) (by decide)
    (by
      intro index values state arity
      have casted : Int.ofNat (index+{step}) = Int.ofNat index+{step} := by
        simpa only [Int.ofNat_eq_natCast,Int.cast_ofNat_Int] using Int.natCast_add index {step}
      change U codec default primitive external data (Int.ofNat index) values state =
        loopEval 0 (O codec default primitive external data)
          [Int.ofNat index,Int.ofNat (index+{step})] values state
      rw [casted]
      exact body_group codec default primitive external data (Int.ofNat index) values state arity)
    [] world
"""
    form_simps = ",".join(
        [
            "FactoredSource,FactoredTarget",
            *names,
            "m_default,m_index,m_iterations,IndexOp.eval,evalProgram,reads",
            "List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero",
            "List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind",
            "List.length_cons,List.length_nil,codec.roundTrip,O,U,T,Esource,Ebulk",
        ]
    )
    text += f"""theorem source_form (data : List Value) (world : World) :
    evalProgram (m codec default primitive external) MinimalSource data world =
      (loopEval 0 (O codec default primitive external data) (indexIterations {lower} {upper} {step}) [] world).bind
        (fun result => .ok ([{returned_value}],result.2)) := by
  rw [sourceFactored]
  simp only [{form_simps}]
  rw [source_function]
  cases run : loopEval 0 (O codec default primitive external data)
    (indexIterations {lower} {upper} {step}) [] world with
  | error failure => rfl
  | ok result =>
    have size := loopEval_arity 0 _ _ _ _ run
    have empty : result.1 = [] := List.length_eq_zero_iff.mp size
    simp only [Except.bind,empty,List.nil_append,reads,List.map_cons,List.map_nil,
      List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some]
theorem target_form (data : List Value) (world : World) :
    evalProgram (m codec default primitive external) MinimalTarget data world =
      ((loopEval 0 (U codec default primitive external data) (indexIterations {lower} {bulk_upper} {stride}) [] world).bind
        (fun result => loopEval 0 (O codec default primitive external data)
          (indexIterations {bulk_upper} {upper} {step}) result.1 result.2)).bind
        (fun result => .ok ([{returned_value}],result.2)) := by
  rw [targetFactored]
  have tails := loopEval_congr 0
    (T codec default primitive external data)
    (O codec default primitive external data)
    (fun index values state arity => tail_same codec default primitive external data index values state arity)
  simp only [{form_simps}]
  rw [bulk_function]
  cases bulkRun : loopEval 0 (U codec default primitive external data)
    (indexIterations {lower} {bulk_upper} {stride}) [] world with
  | error failure => rfl
  | ok bulkResult =>
    have size := loopEval_arity 0 _ _ _ _ bulkRun
    have empty : bulkResult.1 = [] := List.length_eq_zero_iff.mp size
    simp only [Except.bind]
    rw [empty]
    simp only [List.nil_append,Except.bind,{form_simps}]
    have tailEq := tails (indexIterations {bulk_upper} {upper} {step}) [] bulkResult.2
    rw [tail_function,tailEq]
    cases tailRun : loopEval 0 (O codec default primitive external data)
      (indexIterations {bulk_upper} {upper} {step}) [] bulkResult.2 with
    | error failure => rfl
    | ok result =>
      have size := loopEval_arity 0 _ _ _ _ tailRun
      have empty : result.1 = [] := List.length_eq_zero_iff.mp size
      simp only [Except.bind,empty,List.nil_append,reads,List.map_cons,List.map_nil,
        List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some]
theorem minimal (environment : List Value) (world : World) :
    evalProgram (m codec default primitive external) MinimalSource environment world =
      evalProgram (m codec default primitive external) MinimalTarget environment world := by
  rw [source_form,target_form,grouped]
"""
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
    text += """theorem complete2 : PreservesEntries (m codec default primitive external) originalEntries finalEntries :=
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
