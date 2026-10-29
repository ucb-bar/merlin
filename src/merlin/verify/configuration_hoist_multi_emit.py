"""Closed actual full-command focus-chain obligations; no authority is issued."""

from .configuration_elision_emit import _list
from .configuration_elision_gate import Unavailable
from .configuration_hoist_emit import MODULE, ROOT, _command, _program
from .configuration_hoist_prefix_emit import SUPPORT as PREFIX_SUPPORT
from .configuration_hoist_prefix_emit import _move

SUPPORT = [
    *PREFIX_SUPPORT,
    *[
        "Merlin.GraphState.Configuration.PartialHoist.Focus." + name
        for name in (
            "move_sound",
            "ready_sound",
            "chain_sound",
            "check_sound",
            "checked_observation",
            "context_preserved",
        )
    ],
]


def _stage(value):
    focus = value["focus"]
    return (
        "{ after := "
        + _list(value["after"], _command)
        + ", focus := { earlier := "
        + _list(focus["earlier"], _command)
        + ", move := "
        + _move(focus["move"])
        + ", later := "
        + _list(focus["later"], _command)
        + " } }"
    )


def emit(projection):
    if projection["schema"] != "merlin.configuration-hoist.multi-projection.v1":
        raise ValueError("complete actual multi-loop focus projection required")
    from .configuration_elision_emit import _integer

    text = """import Merlin.GraphStateConfigurationHoistFocus
namespace ActualConfigurationHoist
open Merlin.StateEffects Merlin.GraphState.Configuration
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 4096
"""
    text += "def fields : List Nat := " + _list(range(len(projection["fields"])), _integer) + "\n"
    text += "def before : PartialHoist.Program := " + _program(projection["before"]) + "\n"
    text += "def after : PartialHoist.Program := " + _program(projection["after"]) + "\n"
    text += "def stages : List PartialHoist.Focus.Stage := " + _list(projection["stages"], _stage) + "\n"
    text += """theorem accepted : PartialHoist.Focus.check fields before after stages = true := by decide

theorem complete_kernel {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run) (initial : State Value) :
    PartialHoist.program run otherLoop before initial = PartialHoist.program run otherLoop after initial :=
  PartialHoist.Focus.check_sound accepted literal run otherLoop models initial

theorem complete_observation {Value Error Observation : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run) (initial : State Value)
    (observe : PartialHoist.Outcome Value Error → Observation) :
    observe (PartialHoist.program run otherLoop before initial) =
      observe (PartialHoist.program run otherLoop after initial) :=
  PartialHoist.Focus.checked_observation accepted literal run otherLoop models initial observe

theorem surrounding_context {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run)
    (earlier later : List PartialHoist.Command) (initial : State Value) :
    PartialHoist.execute run otherLoop (earlier ++ before.commands ++ later) initial =
      PartialHoist.execute run otherLoop (earlier ++ after.commands ++ later) initial :=
  PartialHoist.Focus.context_preserved accepted literal run otherLoop models earlier later initial
end ActualConfigurationHoist
"""
    if len(text.encode()) > 4 * 1024 * 1024:
        raise Unavailable("compiler multi-loop query byte resource budget exceeded")
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
