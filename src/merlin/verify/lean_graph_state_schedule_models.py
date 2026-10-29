# ruff: noqa: E501
"""Explicit standalone initialized provider witness on actual submitted syntax.

This mathematical word/provider interpretation is not native/runtime approval.
Success/failure is computed, not assumed; the target follows the sound checker.
"""

from __future__ import annotations

from .lean_graph_state_schedule_errors import require


def standalone(capture, text, roots):
    functions = capture["pairs"][0]["functions"]
    candidates = []
    for i, function in enumerate(functions):
        if function["declaration"] or function["inputs"] != [{"bits": 8}, {"bits": 8}] or function["outputs"] != [{"bits": 8}, {"bits": 8}]:
            continue
        source = function["before"]
        effects = [r for r in source if "effect" in r]
        if not any("loop" in r for r in source) and len(effects) == 2 and effects[0]["outputs"] == [{"bits": 8}] and effects[1]["outputs"] == []:
            candidates.append((i, effects[0]["effect"]))
    require(len(candidates) == 1, "one explicit i8 two-input provider/watcher witness entry required")
    index, provider = candidates[0]
    last = len(capture["pairs"])
    text = "import Merlin.GraphStateIntegerModels\n" + text
    text = text.removesuffix("end Merlin.GraphStateScheduleInstance\n")
    text += f"""namespace ScalarProvider
open Merlin.GraphState.Integer
def controls : Controls := List.replicate registry.length none
theorem controlsChecked : checkControl registry controls = true := by decide
def external (fail : Bool) : External Nat Int registry := provider registry (if fail then some {provider} else none)
abbrev chosen (fail : Bool) : Model Int Nat Int registry :=
  indexModel (checkControl_sound controlsChecked) primitive (external fail)
theorem laws (fail : Bool) : LocalTyping (chosen fail) Canonical :=
  index_model_typing (checkControl_sound controlsChecked) (external fail)
    (provider_typed registry (if fail then some {provider} else none))
def arguments : List Int := [3,5]
theorem canonicalArguments : Values Canonical F{index}.inputs arguments := by
  apply Values.cons (by decide : word 8 3 = 3)
  apply Values.cons (by decide : word 8 5 = 5)
  exact Values.nil
theorem nonzero : arguments.all (fun value => decide (value ≠ 0)) = true := by decide
theorem source_success : evalProgram (chosen false) F{index}.S0 arguments 7 = .ok ([17,25],9) := rfl
theorem source_failure : evalProgram (chosen true) F{index}.S0 arguments 7 = .error (.external 17,8) := rfl
"""
    for label, fail, result in (("success", "false", ".ok ([17,25],9)"), ("failure", "true", ".error (.external 17,8)")):
        text += f"theorem final_{label} : evalProgram (chosen {fail}) F{index}.S{last} arguments 7 = {result} := by\n"
        for step in reversed(range(last)):
            text += f"  rw [← F{index}.actual{step} Int Nat Int (chosen {fail}) arguments 7 rfl]\n"
        text += f"  exact source_{label}\n"
    text += f"""theorem source_reached : Reached (chosen false) Canonical F{index}.S0 arguments 7 :=
  (checked_reached F{index}.accepted0 (chosen false) (laws false) arguments 7 canonicalArguments).1
theorem final_reached : Reached (chosen false) Canonical F{index}.S{last} arguments 7 :=
  (checked_reached F{index}.accepted{last-1} (chosen false) (laws false) arguments 7 canonicalArguments).2
end ScalarProvider
end Merlin.GraphStateScheduleInstance
"""
    roots += ["Merlin.GraphStateScheduleInstance.ScalarProvider." + name for name in (
        "canonicalArguments", "nonzero", "source_success", "source_failure", "final_success", "final_failure", "source_reached", "final_reached",
    )]
    return text, roots, {
        "mode": "initialized-scalar-provider-v1", "input": [3, 5], "initial_world": 7,
        "success": {"values": [17, 25], "world": 9}, "failure": {"error": 17, "world": 8},
        "source_success_assumed": False, "runtime_reached_types_derived": True,
        "external_contract": "explicit typed canonical word17 provider; world+1; optional first provider fault",
        "primitive": "sealed exact wrapping word interpretation; no physical agreement implied",
        "native_or_poison_fidelity_proved": False, "model_or_review_approval": False,
    }
