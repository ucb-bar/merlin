"""Closed full recursive Kernel obligations; stages and Python issue no authority."""

from .configuration_elision_emit import _atom, _integer, _list, _string
from .configuration_elision_gate import Unavailable
from .configuration_hoist_emit import MODULE, ROOT
from .configuration_hoist_prefix_emit import SUPPORT as PREFIX_SUPPORT

SUPPORT = [
    *PREFIX_SUPPORT,
    *[
        "Merlin.GraphState.Configuration.PartialHoist.Recursive." + name
        for name in (
            "models_at",
            "execute_prepend",
            "repeat_fixed",
            "safe_fixed",
            "loop_prefix",
            "movable_sound",
            "repeat_congr",
            "context_congr",
            "ready_sound",
            "chain_sound",
            "check_sound",
            "checked_observation",
        )
    ],
]


def _code(value):
    if value["tag"] == "done":
        return ".done"
    if value["tag"] == "call":
        return "(.call " + _integer(value["id"]) + " " + _code(value["rest"]) + ")"
    if value["tag"] != "loop":
        raise ValueError("exact recursive Code constructor required")
    return (
        "(.loop "
        + " ".join(
            (
                _string(value["var"]),
                _integer(value["extent"]),
                _string(value["metadata"]),
                _code(value["body"]),
                _code(value["rest"]),
            )
        )
        + ")"
    )


def _context(value):
    if value["tag"] == "hole":
        return ".hole"
    if value["tag"] == "call":
        return "(.call " + _integer(value["id"]) + " " + _context(value["rest"]) + ")"
    common = [_string(value["var"]), _integer(value["extent"]), _string(value["metadata"])]
    if value["tag"] == "loopBody":
        return "(.loopBody " + " ".join([*common, _context(value["body"]), _code(value["rest"])]) + ")"
    if value["tag"] == "loopRest":
        return "(.loopRest " + " ".join([*common, _code(value["body"]), _context(value["rest"])]) + ")"
    raise ValueError("exact recursive Context constructor required")


def _move(value):
    return (
        "{ "
        + ", ".join(
            (
                "setups := " + _list(value["setups"], _integer),
                "body := " + _code(value["body"]),
                "rest := " + _code(value["rest"]),
                "var := " + _string(value["var"]),
                "extent := " + _integer(value["extent"]),
                "metadata := " + _string(value["metadata"]),
            )
        )
        + " }"
    )


def _stage(value):
    return (
        "{ after := "
        + _code(value["after"])
        + ", context := "
        + _context(value["context"])
        + ", move := "
        + _move(value["move"])
        + " }"
    )


def _program(value):
    return (
        "{ frame := "
        + _string(value["frame"])
        + ", atoms := "
        + _list(value["atoms"], _atom)
        + ", code := "
        + _code(value["code"])
        + " }"
    )


def emit(projection):
    if projection["schema"] != "merlin.configuration-hoist.recursive-projection.v1":
        raise ValueError("actual recursive Kernel projection required")
    try:
        text = """import Merlin.GraphStateConfigurationRecursiveHoist
namespace ActualConfigurationHoist
open Merlin.StateEffects Merlin.GraphState.Configuration
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 4096
"""
        text += "def fields : List Nat := " + _list(range(len(projection["fields"])), _integer) + "\n"
        text += "def before : PartialHoist.Recursive.Program := " + _program(projection["before"]) + "\n"
        text += "def after : PartialHoist.Recursive.Program := " + _program(projection["after"]) + "\n"
        text += "def stages : List PartialHoist.Recursive.Stage := " + _list(projection["stages"], _stage) + "\n"
        text += """theorem accepted : PartialHoist.Recursive.check fields before after stages = true := by decide

theorem complete_kernel {Value Error : Type} (literal : Int → Value)
    (run : Nat → PartialHoist.Recursive.Env → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Recursive.Models fields literal before.atoms run)
    (env : PartialHoist.Recursive.Env) (initial : State Value) :
    PartialHoist.Recursive.execute run env before.code initial =
      PartialHoist.Recursive.execute run env after.code initial :=
  PartialHoist.Recursive.check_sound accepted literal run models env initial

theorem complete_observation {Value Error Observation : Type} (literal : Int → Value)
    (run : Nat → PartialHoist.Recursive.Env → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Recursive.Models fields literal before.atoms run)
    (env : PartialHoist.Recursive.Env) (initial : State Value)
    (observe : PartialHoist.Outcome Value Error → Observation) :
    observe (PartialHoist.Recursive.execute run env before.code initial) =
      observe (PartialHoist.Recursive.execute run env after.code initial) :=
  PartialHoist.Recursive.checked_observation accepted literal run models env initial observe

theorem surrounding_context {Value Error : Type} (literal : Int → Value)
    (run : Nat → PartialHoist.Recursive.Env → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Recursive.Models fields literal before.atoms run)
    (context : PartialHoist.Recursive.Context) (env : PartialHoist.Recursive.Env) (initial : State Value) :
    PartialHoist.Recursive.execute run env (PartialHoist.Recursive.plug context before.code) initial =
      PartialHoist.Recursive.execute run env (PartialHoist.Recursive.plug context after.code) initial :=
  PartialHoist.Recursive.context_congr run before.code after.code
    (fun env state => PartialHoist.Recursive.check_sound accepted literal run models env state) context env initial
end ActualConfigurationHoist
"""
    except RecursionError as error:
        raise Unavailable("recursive hoist emitter exhausted its bounded Python stack") from error
    if len(text.encode()) > 4 * 1024 * 1024:
        raise Unavailable("recursive hoist query byte resource budget exceeded")
    return {
        "query": text,
        "roots": [
            *SUPPORT,
            MODULE + ".accepted",
            MODULE + ".complete_observation",
            MODULE + ".surrounding_context",
            ROOT,
        ],
        "module": MODULE,
        "final_root": ROOT,
        "kernel_outcome": "UNOBSERVED",
        "authority": False,
    }
