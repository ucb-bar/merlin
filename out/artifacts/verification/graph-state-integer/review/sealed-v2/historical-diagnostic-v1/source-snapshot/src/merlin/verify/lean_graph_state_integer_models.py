# ruff: noqa: E501
"""Actual six-buffer symbolic-loop provider witness, separate from native fidelity.

The loop proof uses a structural constant-result theorem and the derived finite
range, never a literal dynamic execution trace or a source-success premise.
"""

from __future__ import annotations

from .lean_graph_state_integer_emit import emit
from .lean_graph_state_scalar import integer


def witness(capture):
    if capture["pipeline"] is None:
        raise ValueError("initialized provider witness requires the actual six-buffer chain")
    functions = capture["prefix"]["functions"]
    defined = [i for i, f in enumerate(functions) if not f["declaration"]]
    changed = [i for i in defined if functions[i]["before"] != functions[i]["after"]]
    if len(changed) != 1:
        raise ValueError("one actual changed main entry required in provider witness mode")
    index = changed[0]
    function = functions[index]
    inputs = function["inputs"]
    if len(inputs) != 2 or inputs[0] != inputs[1] or set(inputs[0]) != {"bits"} or inputs[0]["bits"] < 5:
        raise ValueError("two equal canonical words capable of nonzero provider17 required")
    suffix = capture["pipeline"]
    controls = {r["identity"]: r for r in suffix["index_operations"]}
    records = suffix["stages"][1]["functions"][index]["after"]
    loops = [r["loop"] for r in records if "loop" in r]
    if len(loops) != 1 or loops[0]["carries"] != [inputs[0]]:
        raise ValueError("single actual word-carry witness submode required")
    effects = [r for r in loops[0]["body"] if "effect" in r and r["effect"] not in controls]
    if len(effects) != 2 or effects[0]["outputs"] or effects[1]["outputs"] != [inputs[0]]:
        raise ValueError("one observer then one provider in actual loop body required")
    watch = effects[0]["effect"]
    from .lean_graph_state_pipeline_emit import static_controls

    lower, upper, stride = static_controls(records, suffix)
    count = (upper - lower + stride - 1) // stride
    if count < 1:
        raise ValueError("nonempty actual execution witness required")
    iterations = "[" + ",".join(integer(lower + stride * i) for i in range(count)) + "]"
    if count > 64:
        raise ValueError("witness range syntax budget64; graph preservation still supports huge counts")
    text = emit(capture["prefix"], pipeline_capture=suffix)
    text = "import Merlin.GraphStateIntegerModels\nimport Merlin.GraphStateIntegerCountermodels\n" + text
    text = text.removesuffix("end Merlin.GraphStateIntegerInstance\n")
    text += f"""namespace IntegerProvider
open Merlin.GraphState.Pipeline
abbrev ex (fail : Bool) : External Nat Int registry :=
  provider registry (if fail then some {watch} else none)
abbrev chosen (fail : Bool) : Model Int Nat Int registry := Suffix.M primitive (ex fail)
theorem laws (fail : Bool) : LocalTyping (chosen fail) Canonical :=
  index_model_typing (checkControl_sound Suffix.controlAccepted) (ex fail)
    (provider_typed registry (if fail then some {watch} else none))
def arguments : List Int := [3,5]
theorem entriesCanonical : Values Canonical F{index}.inputs arguments := by
  apply Values.cons (by decide : word {inputs[0]["bits"]} 3 = 3)
  apply Values.cons (by decide : word {inputs[0]["bits"]} 5 = 5)
  exact Values.nil
theorem nonzero : arguments.all (fun value => decide (value ≠ 0)) = true := by decide
theorem range : indexIterations {lower} {upper} {stride} = {iterations} :=
  indexIterations_count (lower := {lower}) (upper := {upper}) (step := {stride}) (count := {count}) (by decide) (by decide)
def body (fail : Bool) (index : Int) (carry : List Int) (world : Nat) : Outcome Int Nat Int :=
  evalProgram (chosen fail) Suffix.F{index}.B2 (index::carry++Suffix.F{index}.E2 primitive 3 5) world
theorem body_success (index : Int) (carry : List Int) (world : Nat) (arity : carry.length = 1) :
    body false index carry world = .ok ([17],world+2) := by
  obtain ⟨value,rfl⟩ := List.length_eq_one_iff.mp arity
  simp [body,chosen,Suffix.F{index}.B2,Suffix.F{index}.E2,Suffix.M,indexModel,indexEffect,
    Suffix.controls,Merlin.GraphState.Pipeline.controlAt,opaqueStep,evalProgram,reads,ex,provider,registry,Suffix.registry,
    signature,inhabit,primitive,word,Except.bind,Nat.add_assoc]
theorem body_failure (index : Int) (carry : List Int) (world : Nat) (arity : carry.length = 1) :
    body true index carry world = .error (.external 17,world+1) := by
  obtain ⟨value,rfl⟩ := List.length_eq_one_iff.mp arity
  simp [body,chosen,Suffix.F{index}.B2,Suffix.F{index}.E2,Suffix.M,indexModel,indexEffect,
    Suffix.controls,Merlin.GraphState.Pipeline.controlAt,opaqueStep,evalProgram,reads,ex,provider,registry,Suffix.registry,
    signature,inhabit,primitive,word,Except.bind,Nat.add_assoc]
theorem loop_success (world : Nat) :
    loopEval 1 (body false) (indexIterations {lower} {upper} {stride}) [3] world = .ok ([17],world+{2 * count}) := by
  rw [range]
  exact loop_constant 17 2 (body false) body_success _ _ [3] world rfl
theorem loop_failure (world : Nat) :
    loopEval 1 (body true) (indexIterations {lower} {upper} {stride}) [3] world = .error (.external 17,world+1) := by
  rw [range,loopEval]
  simp only [List.length_cons,List.length_nil,↓reduceIte,body_failure _ [3] world rfl,Except.bind]
theorem source_success (world : Nat) :
    evalProgram (chosen false) F{index}.before arguments world = .ok ([17],world+{2 * count + 1}) := by
  rw [F{index}.actual (chosen false) ⟨rfl,rfl⟩ (laws false) arguments world entriesCanonical]
  change evalProgram (Suffix.M primitive (ex false)) Suffix.F{index}.S0 [3,5] world = _
  rw [Suffix.F{index}.firstTwo Int Nat Int (chosen false) [3,5] world rfl]
  rw [Suffix.F{index}.form2]
  change (loopEval 1 (body false) (indexIterations {lower} {upper} {stride}) [3] world).bind _ = _
  rw [loop_success]
  simp [Suffix.F{index}.N2,Suffix.F{index}.E2,chosen,Suffix.M,indexModel,indexEffect,
    Suffix.controls,Merlin.GraphState.Pipeline.controlAt,opaqueStep,evalProgram,reads,ex,provider,registry,Suffix.registry,
    signature,inhabit,primitive,word,Except.bind,Nat.add_assoc]
theorem source_failure (world : Nat) :
    evalProgram (chosen true) F{index}.before arguments world = .error (.external 17,world+1) := by
  rw [F{index}.actual (chosen true) ⟨rfl,rfl⟩ (laws true) arguments world entriesCanonical]
  change evalProgram (Suffix.M primitive (ex true)) Suffix.F{index}.S0 [3,5] world = _
  rw [Suffix.F{index}.firstTwo Int Nat Int (chosen true) [3,5] world rfl]
  rw [Suffix.F{index}.form2]
  change (loopEval 1 (body true) (indexIterations {lower} {upper} {stride}) [3] world).bind _ = _
  rw [loop_failure]
  rfl
theorem actual_source_success : evalProgram (chosen false) F{index}.before arguments 7 = .ok ([17],{8 + 2 * count}) :=
  source_success 7
theorem actual_final_success : evalProgram (chosen false) Suffix.F{index}.S4 arguments 7 = .ok ([17],{8 + 2 * count}) := by
  change evalProgram (Suffix.M primitive (ex false)) Suffix.F{index}.S4 arguments 7 = _
  rw [← Suffix.F{index}.actual primitive (ex false) arguments 7 rfl]
  change evalProgram (chosen false) F{index}.after arguments 7 = _
  rw [← F{index}.actual (chosen false) ⟨rfl,rfl⟩ (laws false) arguments 7 entriesCanonical]
  exact actual_source_success
theorem actual_source_failure : evalProgram (chosen true) F{index}.before arguments 7 = .error (.external 17,8) :=
  source_failure 7
theorem actual_final_failure : evalProgram (chosen true) Suffix.F{index}.S4 arguments 7 = .error (.external 17,8) := by
  change evalProgram (Suffix.M primitive (ex true)) Suffix.F{index}.S4 arguments 7 = _
  rw [← Suffix.F{index}.actual primitive (ex true) arguments 7 rfl]
  change evalProgram (chosen true) F{index}.after arguments 7 = _
  rw [← F{index}.actual (chosen true) ⟨rfl,rfl⟩ (laws true) arguments 7 entriesCanonical]
  exact actual_source_failure
theorem source_values_reached : Reached (chosen false) Canonical F{index}.before arguments 7 :=
  (F{index}.reached (chosen false) (laws false) arguments 7 entriesCanonical).1
end IntegerProvider
end Merlin.GraphStateIntegerInstance
"""
    model_roots = [
        "Merlin.GraphStateIntegerInstance.IntegerProvider." + name
        for name in (
            "entriesCanonical",
            "nonzero",
            "body_success",
            "body_failure",
            "source_success",
            "source_failure",
            "actual_source_success",
            "actual_final_success",
            "actual_source_failure",
            "actual_final_failure",
            "source_values_reached",
        )
    ]
    model_roots += [
        "Merlin.GraphState.Integer.Countermodels." + name
        for name in (
            "noncanonical_entry_refutes_neutral_deletion",
            "entry_really_noncanonical",
            "missing_primitive_contract_refutes_deletion",
            "provider_programs_statically_typed",
            "missing_local_result_contract_refutes_deletion",
            "hidden_effect_countermodel",
        )
    ]
    return (
        text,
        model_roots,
        {
            "mode": "initialized-symbolic-provider-v1",
            "inputs": [3, 5],
            "initial_world": 7,
            "success": {"values": [17], "world": 8 + 2 * count},
            "failure": {"error": 17, "world": 8},
            "source_success_assumed": False,
            "domain": "initialized canonical Int words; no physical poison constructor",
            "external": "explicit typed constant-word17 provider; world increment1; optional first observer fault",
            "primitive_fidelity_proved": False,
            "native_fidelity_proved": False,
            "review_approved": False,
            "old_literal_trace_promoted": False,
        },
    )
