"""Closed actual partial-hoist source generator; no proof or approval is issued."""

from merlin.verify.configuration_elision_emit import _atom, _integer, _list, _string

MODULE = "ActualConfigurationHoist"
ROOT = MODULE + ".complete_kernel"
SUPPORT = [
    "Merlin.GraphState.Configuration.PartialHoist." + name
    for name in (
        "assign_fixed_after",
        "remove_fixed_setups",
        "hoist_nonempty_partial",
        "safeBody_frame",
        "loop_setup",
        "execute_append",
        "valid_write_fields",
        "ready_loop",
        "check_sound",
        "checked_observation",
        "context_preserved",
    )
]


def _command(command):
    if set(command) == {"call"}:
        return ".call " + _integer(command["call"])
    descriptor = command["loop"]
    return (
        ".loop { var := "
        + _string(descriptor["var"])
        + ", extent := "
        + _integer(descriptor["extent"])
        + (
            ", body := "
            + _list(descriptor["body"], _integer)
            + ", metadata := "
            + _string(descriptor["metadata"])
            + ", iterArgs := "
            + _list(descriptor["iterArgs"], lambda p: "(" + _string(p[0]) + ", " + _string(p[1]) + ")")
            + ", yields := "
            + _list(descriptor["yields"], _string)
            + ", results := "
            + _list(descriptor["results"], _string)
            + " }"
        )
    )


def _program(value):
    return (
        "{ frame := "
        + _string(value["frame"])
        + ", atoms := "
        + _list(value["atoms"], _atom)
        + (", commands := " + _list(value["commands"], _command) + " }")
    )


def _move(value):
    names = ("earlier", "setup", "body", "later", "var", "extent", "metadata")
    parts = []
    for name in names:
        item = value[name]
        term = (
            _list(item, _integer)
            if name in {"earlier", "body", "later"}
            else (_string(item) if name in {"var", "metadata"} else _integer(item))
        )
        parts.append(name + " := " + term)
    return "{ " + ", ".join(parts) + " }"


def emit(projection):
    if projection["schema"] != "merlin.configuration-hoist.projection.v1":
        raise ValueError("complete current partial-hoist projection required")
    text = (
        """import Merlin.GraphStateConfigurationHoistChecker

namespace ActualConfigurationHoist
open Merlin.StateEffects
open Merlin.GraphState.Configuration
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 4096

def fields : List Nat := """
        + _list(range(len(projection["fields"])), _integer)
        + "\n"
    )
    text += "def before : PartialHoist.Program := " + _program(projection["before"]) + "\n"
    text += "def after : PartialHoist.Program := " + _program(projection["after"]) + "\n"
    text += "def move : PartialHoist.Move := " + _move(projection["move"]) + "\n\n"
    text += """theorem accepted : PartialHoist.check fields before after move = true := by decide

theorem complete_kernel {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run) (initial : State Value) :
    PartialHoist.program run otherLoop before initial = PartialHoist.program run otherLoop after initial :=
  PartialHoist.check_sound accepted literal run otherLoop models initial

theorem complete_observation {Value Error Observation : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run) (initial : State Value)
    (observe : PartialHoist.Outcome Value Error → Observation) :
    observe (PartialHoist.program run otherLoop before initial) =
      observe (PartialHoist.program run otherLoop after initial) :=
  PartialHoist.checked_observation accepted literal run otherLoop models initial observe

theorem surrounding_context {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run)
    (earlier later : List PartialHoist.Command) (initial : State Value) :
    PartialHoist.execute run otherLoop (earlier ++ before.commands ++ later) initial =
      PartialHoist.execute run otherLoop (earlier ++ after.commands ++ later) initial :=
  PartialHoist.context_preserved accepted literal run otherLoop models earlier later initial

end ActualConfigurationHoist
"""
    roots = [*SUPPORT, MODULE + ".accepted", MODULE + ".complete_observation", MODULE + ".surrounding_context", ROOT]
    return {
        "query": text,
        "roots": roots,
        "module": MODULE,
        "final_root": ROOT,
        "kernel_outcome": "UNOBSERVED",
        "authority": False,
    }
