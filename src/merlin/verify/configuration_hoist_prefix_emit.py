"""Closed complete-prefix obligations; no compilation or authority is issued."""

from .configuration_elision_emit import _integer, _list, _string
from .configuration_hoist_emit import MODULE, ROOT, _program

SUPPORT = [
    "Merlin.GraphState.Configuration.PartialHoist.Prefix." + name
    for name in (
        "assign_append",
        "collect_sound",
        "calls_append",
        "execute_calls",
        "loop_prefix",
        "hoist_prefix",
        "atom_fixed",
        "safeFixedBody_preserves",
        "remove_fixed_prefixes",
        "hoist_nonempty_fixed",
        "hoist_prefix_fixed",
        "ready_loop",
        "check_sound",
        "checked_observation",
        "context_preserved",
    )
]


def _move(value):
    parts = []
    for name in ("earlier", "setups", "body", "later", "var", "extent", "metadata"):
        item = value[name]
        term = (
            _list(item, _integer)
            if name in {"earlier", "setups", "body", "later"}
            else (_string(item) if name in {"var", "metadata"} else _integer(item))
        )
        parts.append(name + " := " + term)
    return "{ " + ", ".join(parts) + " }"


def emit(projection):
    if projection["schema"] != "merlin.configuration-hoist.prefix-projection.v1":
        raise ValueError("complete actual pure-prefix projection required")
    text = """import Merlin.GraphStateConfigurationPrefixHoist
namespace ActualConfigurationHoist
open Merlin.StateEffects Merlin.GraphState.Configuration
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option maxRecDepth 4096
"""
    text += "def fields : List Nat := " + _list(range(len(projection["fields"])), _integer) + "\n"
    text += "def before : PartialHoist.Program := " + _program(projection["before"]) + "\n"
    text += "def after : PartialHoist.Program := " + _program(projection["after"]) + "\n"
    text += "def move : PartialHoist.Prefix.Move := " + _move(projection["move"]) + "\n"
    text += """theorem accepted : PartialHoist.Prefix.check fields before after move = true := by decide

theorem complete_kernel {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run) (initial : State Value) :
    PartialHoist.program run otherLoop before initial = PartialHoist.program run otherLoop after initial :=
  PartialHoist.Prefix.check_sound accepted literal run otherLoop models initial

theorem complete_observation {Value Error Observation : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run) (initial : State Value)
    (observe : PartialHoist.Outcome Value Error → Observation) :
    observe (PartialHoist.program run otherLoop before initial) =
      observe (PartialHoist.program run otherLoop after initial) :=
  PartialHoist.Prefix.checked_observation accepted literal run otherLoop models initial observe

theorem surrounding_context {Value Error : Type} (literal : Int → Value)
    (run : Nat → Nat → State Value → PartialHoist.Outcome Value Error)
    (otherLoop : PartialHoist.Loop → State Value → PartialHoist.Outcome Value Error)
    (models : PartialHoist.Models fields literal before.atoms run)
    (earlier later : List PartialHoist.Command) (initial : State Value) :
    PartialHoist.execute run otherLoop (earlier ++ before.commands ++ later) initial =
      PartialHoist.execute run otherLoop (earlier ++ after.commands ++ later) initial :=
  PartialHoist.Prefix.context_preserved accepted literal run otherLoop models earlier later initial
end ActualConfigurationHoist
"""
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
