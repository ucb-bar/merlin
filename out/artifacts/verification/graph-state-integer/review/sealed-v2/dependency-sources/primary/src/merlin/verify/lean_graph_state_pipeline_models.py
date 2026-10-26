# ruff: noqa: E501
# Literal Lean equations are kept intact for readable, byte-bound proof evidence.
"""Bounded, actual-export operational witnesses under an explicit common model.

Trace authoring is untrusted data: every operation/fold equation is kernel
checked. Whole-pipeline preservation derives target execution from the proved
source witness, not from a source-success assumption or a native claim.
"""

from __future__ import annotations

from .lean_graph_state_pipeline_emit import emit
from .lean_graph_state_scalar_execution import Author, outcome
from .lean_graph_state_scalar_models import ints, model_definitions


def witness(capture, arguments, *, failure=None):
    stages = capture["stages"]
    registry = stages[0]["registry"]
    changed = [
        i
        for i, f in enumerate(stages[0]["functions"])
        if not f["declaration"] and f["before"] != stages[-1]["functions"][i]["after"]
    ]
    if len(changed) != 1:
        raise ValueError("one changed actual entry required for this finite witness mode")
    index = changed[0]
    function = stages[0]["functions"][index]
    if len(arguments) != len(function["inputs"]) or any(type(v) is not int for v in arguments):
        raise ValueError("exact initialized witness input vector required")
    if not all(
        set(t) == {"bits"} and 0 <= v < 1 << t["bits"] for v, t in zip(arguments, function["inputs"], strict=True)
    ):
        raise ValueError("witness arguments must inhabit the declared unsigned integer input types")
    controls = {r["identity"]: r for r in capture["index_operations"]}
    if failure is not None and (type(failure) is not int or failure in controls or not 0 <= failure < len(registry)):
        raise ValueError("failure must designate a retained noncontrol operation")

    def original_controls(records):
        for row in records:
            if "loop" in row:
                original_controls(row["loop"]["body"])
            elif "effect" in row and row["effect"] in controls and controls[row["effect"]]["kind"] != "index-constant":
                raise ValueError(
                    "this trace author supports source constant controls only; not a graph-proof restriction"
                )

    original_controls(function["before"])
    author = Author(registry, failure)
    proof, expected = author.execute(function["before"], arguments, [23])
    body_name = author.code(function["before"])
    model = model_definitions(registry, failure).split("def concrete :")[0]
    begin = model.index("def iterations")
    end = model.index("def selectedValue")
    model = model[:begin] + model[end:]
    text = emit(capture).removesuffix("end Merlin.GraphStatePipelineInstance\n")
    text = text.replace(
        "import Merlin.GraphStateScalar\n", "import Merlin.GraphStateScalar\nimport Merlin.GraphStateScalarSteps\n", 1
    )
    text += "namespace Witness\n" + model
    text += """def external : External (List Int) Int registry := {
  run := effect
  arity := by
    intro identity arguments world values final success
    simp only [effect] at success
    split at success
    · cases success
    · cases success; simp
}
def concrete : Model Int (List Int) Int registry := M primitive external
"""
    text += "def arguments : List Int := " + ints(arguments) + "\n"
    text += f"theorem input_arity : arguments.length = F{index}.inputs.length := by decide\n"
    text += "theorem nonzero_inputs : arguments.all (fun value => decide (value ≠ 0)) = true := by decide\n"
    text += "namespace Execution\n" + "".join(author.definitions) + "".join(author.proofs) + "end Execution\n"
    expected_lean = outcome(expected)
    text += f"""theorem actual_original_execution : evalProgram concrete F{index}.S0 arguments [23] = {expected_lean} := by
  have join : F{index}.S0 = Execution.{body_name} := rfl
  rw [join]
  exact Execution.{proof}
theorem actual_final_execution : evalProgram concrete F{index}.S4 arguments [23] = {expected_lean} := by
  rw [← F{index}.actual primitive external arguments [23] input_arity]
  exact actual_original_execution
theorem source_initialized : Initialized concrete F{index}.S0 arguments [23] :=
  typed_operands_initialized (checkTypes_sound _ _ _ F{index}.typed0) concrete arguments [23] input_arity
theorem final_initialized : Initialized concrete F{index}.S4 arguments [23] :=
  typed_operands_initialized (checkTypes_sound _ _ _ F{index}.typed4) concrete arguments [23] input_arity
end Witness
end Merlin.GraphStatePipelineInstance
"""
    roots = ["Merlin.GraphStatePipelineInstance.complete"]
    roots += [
        "Merlin.GraphStatePipelineInstance.Witness." + name
        for name in (
            "input_arity",
            "nonzero_inputs",
            "actual_original_execution",
            "actual_final_execution",
            "source_initialized",
            "final_initialized",
        )
    ]
    return (
        text,
        roots,
        {
            "status": expected[0],
            "values": expected[1],
            "world": expected[2],
            "arguments": arguments,
            "failure_identity": failure,
            "model_only": True,
            "primitive_fidelity_proved": False,
            "native_fidelity_proved": False,
            "typing_implies_source_success": False,
            "poison_freedom_proved": False,
        },
    )


def symbolic_witness(capture):
    """Independent factored provider/counter model, not the limited RFL trace.

    Ordinary wrapping primitives are instantiated WITHOUT evaluating arithmetic
    traces. Source success/failure is derived from actual body syntax and the
    explicit arity-correct provider. World is a complete abstract call counter.
    """
    from .lean_graph_state_pipeline_emit import static_controls
    from .lean_graph_state_scalar_models import MODEL

    functions = capture["stages"][0]["functions"]
    changed = [
        i
        for i, f in enumerate(functions)
        if not f["declaration"] and f["before"] != capture["stages"][-1]["functions"][i]["after"]
    ]
    if len(changed) != 1:
        raise ValueError("one changed actual entry required")
    index = changed[0]
    source2 = capture["stages"][1]["functions"][index]["after"]
    loop = next(r["loop"] for r in source2 if "loop" in r)
    if static_controls(source2, capture) != [0, 6, 1] or len(loop["carries"]) != 1:
        raise ValueError("symbolic finite provider witness submode is exactly 0..6/stride1/one carry")
    calls = [r for r in loop["body"] if "effect" in r]
    if len(calls) != 1 or len(calls[0]["outputs"]) != 1:
        raise ValueError("one single-result original-body provider required")
    declared = [*functions[index]["inputs"], *functions[index]["outputs"], *loop["carries"]]
    if any(set(ty) != {"bits"} or ty["bits"] < 5 for ty in declared):
        raise ValueError("symbolic provider input/output/carry domain must contain unsigned 3,5,17")
    provider = calls[0]["effect"]
    text = emit(capture).removesuffix("end Merlin.GraphStatePipelineInstance\n")
    text += "namespace SymbolicWitness\n" + MODEL.split("def iterations")[0]
    text += f"""def run (fail : Bool) (identity : Nat) (arguments : List Int) (world : Nat) :
    Except (Nat × Nat) (List Int × Nat) :=
  if fail && decide (identity = {provider}) && decide (arguments[1]?.getD 0 = 4) then .error (17,world+5)
  else .ok ((signature registry identity).outputs.map (fun _ => 17),world+1)
def provider (fail : Bool) : External Nat Nat registry := {{
  run := run fail
  arity := by
    intro identity arguments world values final success
    simp only [run] at success
    split at success
    · cases success
    · cases success; simp
}}
def chosen (p : Prim → List Int → Int) (fail : Bool) : Model Int Nat Nat registry := M p (provider fail)
theorem provider_step (p : Prim → List Int → Int) (fail : Bool) (x0 x1 iv value : Int) (world : Nat) :
    evalProgram (chosen p fail) F{index}.B2 (iv::value::F{index}.E2 p x0 x1) world =
      if fail && decide (iv = 4) then .error (.external 17,world+5) else .ok ([17],world+1) := by
  by_cases condition : fail = true ∧ iv = 4
  all_goals simp [F{index}.B2,F{index}.E2,chosen,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,
    IndexOp.eval,opaqueStep,evalProgram,reads,provider,run,signature,registry,Except.bind,condition]
def body (p : Prim → List Int → Int) (fail : Bool) (x0 x1 : Int) (iv : Int) (carry : List Int) (world : Nat) : Outcome Int Nat Nat :=
  evalProgram (chosen p fail) F{index}.B2 (iv::carry++F{index}.E2 p x0 x1) world
theorem staticIndices : indexIterations 0 6 1 = [0,1,2,3,4,5] :=
  indexIterations_count (lower := 0) (upper := 6) (step := 1) (count := 6) (by decide) (by decide)
theorem loop_success (p : Prim → List Int → Int) (x0 x1 value : Int) (world : Nat) :
    loopEval 1 (body p false x0 x1) (indexIterations 0 6 1) [value] world = .ok ([17],world+6) := by
  rw [staticIndices]
  simp [loopEval,body,provider_step,Nat.add_assoc,Except.bind]
theorem loop_failure (p : Prim → List Int → Int) (x0 x1 value : Int) (world : Nat) :
    loopEval 1 (body p true x0 x1) (indexIterations 0 6 1) [value] world = .error (.external 17,world+9) := by
  rw [staticIndices]
  simp [loopEval,body,provider_step,Nat.add_assoc,Except.bind]
theorem observer (p : Prim → List Int → Int) (fail : Bool) (x0 x1 value : Int) (world : Nat) :
    evalProgram (chosen p fail) F{index}.N2 (value::F{index}.E2 p x0 x1) world = .ok ([value],world+1) := by
  simp [F{index}.N2,F{index}.E2,chosen,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,
    IndexOp.eval,opaqueStep,evalProgram,reads,provider,run,signature,registry,Except.bind]
theorem source_success (p : Prim → List Int → Int) (x0 x1 : Int) (world : Nat) :
    evalProgram (chosen p false) F{index}.S0 [x0,x1] world = .ok ([17],world+7) := by
  rw [F{index}.firstTwo Int Nat Nat (chosen p false) [x0,x1] world rfl]
  change evalProgram (M p (provider false)) F{index}.S2 [x0,x1] world = _
  rw [F{index}.form2]
  change (loopEval 1 (body p false x0 x1) (indexIterations 0 6 1) [x0] world).bind _ = _
  rw [loop_success]
  simpa [chosen,Except.bind,Nat.add_assoc] using observer p false x0 x1 17 (world+6)
theorem source_failure (p : Prim → List Int → Int) (x0 x1 : Int) (world : Nat) :
    evalProgram (chosen p true) F{index}.S0 [x0,x1] world = .error (.external 17,world+9) := by
  rw [F{index}.firstTwo Int Nat Nat (chosen p true) [x0,x1] world rfl]
  change evalProgram (M p (provider true)) F{index}.S2 [x0,x1] world = _
  rw [F{index}.form2]
  change (loopEval 1 (body p true x0 x1) (indexIterations 0 6 1) [x0] world).bind _ = _
  rw [loop_failure]
  rfl
def arguments : List Int := [3,5]
theorem input_arity : arguments.length = F{index}.inputs.length := by decide
theorem input_nonzero : arguments.all (fun value => decide (value ≠ 0)) = true := by decide
theorem actual_source_success : evalProgram (chosen primitive false) F{index}.S0 arguments 7 = .ok ([17],14) :=
  source_success primitive 3 5 7
theorem actual_final_success : evalProgram (chosen primitive false) F{index}.S4 arguments 7 = .ok ([17],14) := by
  change evalProgram (M primitive (provider false)) F{index}.S4 arguments 7 = _
  rw [← F{index}.actual primitive (provider false) arguments 7 input_arity]
  exact actual_source_success
theorem actual_source_failure : evalProgram (chosen primitive true) F{index}.S0 arguments 7 = .error (.external 17,16) :=
  source_failure primitive 3 5 7
theorem actual_final_failure : evalProgram (chosen primitive true) F{index}.S4 arguments 7 = .error (.external 17,16) := by
  change evalProgram (M primitive (provider true)) F{index}.S4 arguments 7 = _
  rw [← F{index}.actual primitive (provider true) arguments 7 input_arity]
  exact actual_source_failure
theorem source_initialized : Initialized (chosen primitive false) F{index}.S0 arguments 7 :=
  typed_operands_initialized (checkTypes_sound _ _ _ F{index}.typed0) (chosen primitive false) arguments 7 input_arity
theorem final_initialized : Initialized (chosen primitive false) F{index}.S4 arguments 7 :=
  typed_operands_initialized (checkTypes_sound _ _ _ F{index}.typed4) (chosen primitive false) arguments 7 input_arity
end SymbolicWitness
end Merlin.GraphStatePipelineInstance
"""
    roots = ["Merlin.GraphStatePipelineInstance.complete"]
    roots += [
        "Merlin.GraphStatePipelineInstance.SymbolicWitness." + n
        for n in (
            "provider_step",
            "source_success",
            "source_failure",
            "input_arity",
            "input_nonzero",
            "actual_source_success",
            "actual_final_success",
            "actual_source_failure",
            "actual_final_failure",
            "source_initialized",
            "final_initialized",
        )
    ]
    return (
        text,
        roots,
        {
            "mode": "symbolic-provider-counter-v1",
            "input": [3, 5],
            "initial_world": 7,
            "success": {"values": [17], "world": 14},
            "failure": {"error": 17, "world": 16, "failure_IV": 4},
            "source_success_assumed": False,
            "primitive": "ordinary total wrapping word functions; no fidelity theorem",
            "external_contract": "explicit arity-correct constant17 provider with an optional IV4 fault and observable call counter",
            "native_fidelity_proved": False,
            "poison_freedom_proved": False,
            "limited_literal_trace_promoted": False,
        },
    )


def countermodels():
    """Necessity of control/totality/failure contracts, distinct from pass cases."""
    text = """import Merlin.GraphStatePipelineIndex
namespace Merlin.GraphStatePipelineCountermodels
open Merlin.GraphState.Scalar Merlin.GraphState.Pipeline
def first (_ : Int) (_ : List Int) (world : Nat) : Outcome Int Nat Nat := .error (.external 17,world+1)
def second (_ : Int) (_ : List Int) (world : Nat) : Outcome Int Nat Nat := .error (.external 18,world+2)
def successRelation (body : Int → List Int → Nat → Outcome Int Nat Nat)
    (index : Int) (before after : List Int × Nat) : Prop := body index before.1 before.2 = .ok after
theorem success_relations_equal : ∀ index before after, successRelation first index before after ↔ successRelation second index before after := by
  intro index before after
  simp [successRelation,first,second]
theorem failures_are_different : loopEval 1 first [0] [3] 7 ≠ loopEval 1 second [0] [3] 7 := by
  simp [loopEval,first,second,Except.bind]
theorem emptySpeculation : loopEval 1 first [] [3] 7 ≠ first 0 [3] 7 := by
  simp [loopEval,first,Except.bind]
def observed (index : Int) (_ : List Int) (world : List Int) : Outcome Int (List Int) Nat := .ok ([index],world++[index])
theorem control_world_counterexample : loopEval 1 observed (indexIterations 0 3 1) [0] [] ≠
    loopEval 1 observed (indexIterations 0 3 2) [0] [] := by
  have source : indexIterations 0 3 1 = [0,1,2] :=
    indexIterations_count (lower := 0) (upper := 3) (step := 1) (count := 3) (by decide) (by decide)
  have target : indexIterations 0 3 2 = [0,2] :=
    indexIterations_count (lower := 0) (upper := 3) (step := 2) (count := 2) (by decide) (by decide)
  rw [source,target]
  simp [loopEval,observed,Except.bind]
end Merlin.GraphStatePipelineCountermodels
"""
    roots = [
        "Merlin.GraphStatePipelineCountermodels." + n
        for n in (
            "success_relations_equal",
            "failures_are_different",
            "emptySpeculation",
            "control_world_counterexample",
        )
    ]
    return text, roots
