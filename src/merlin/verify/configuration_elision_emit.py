"""Closed full-entry configuration obligations for the existing Elision roots.

This is only source generation. Compilation, complete type/root/std3 audit and
live binding are separate, necessary steps; no numerical/hardware approval.
"""

from __future__ import annotations

MODULE = "ConfigurationElisionQuery"
NAMESPACE = "ActualConfigurationElision"
FINAL_ROOT = NAMESPACE + ".complete_entries"
SUPPORT_ROOTS = (
    "Merlin.GraphState.Configuration.put_idempotent",
    "Merlin.GraphState.Elision.initial_unknown",
    "Merlin.GraphState.Elision.redundant_assign_identity",
    "Merlin.GraphState.Elision.advance_knowledge",
    "Merlin.GraphState.Elision.omitted_identity",
    "Merlin.GraphState.Elision.replay_sound",
    "Merlin.GraphState.Elision.check_sound",
    "Merlin.GraphState.Elision.context_preserved",
    "Merlin.GraphState.Configuration.Entries.checkEntries_sound",
    "Merlin.GraphState.Configuration.Entries.related_invoke",
    "Merlin.GraphState.Configuration.Entries.check_sound",
    "Merlin.GraphState.Configuration.Entries.checked_frame",
    "Merlin.GraphState.Configuration.Entries.checked_observation",
)


def _string(value):
    # JSON's non-ASCII text and escaped quotes/backslashes form Lean literals;
    # escape control characters with Lean's fixed-width Unicode escapes.
    if type(value) is not str:
        raise ValueError("configuration string term needs a string")
    out = '"'
    for char in value:
        if char == '"':
            out += '\\"'
        elif char == "\\":
            out += "\\\\"
        elif ord(char) < 32 or ord(char) == 127:
            out += "\\u" + format(ord(char), "04x")
        else:
            out += char
    return out + '"'


def _integer(value):
    if type(value) is not int:
        raise ValueError("configuration integer term needs an actual integer")
    return str(value) if value >= 0 else "(" + str(value) + ")"


def _list(values, render):
    return "[" + ", ".join(render(value) for value in values) + "]"


def _effect(value):
    if value is None:
        return "none"

    def assignments(rows):
        return _list(rows, lambda row: "(" + _integer(row[0]) + ", " + _integer(row[1]) + ")")

    return (
        "some { reads := "
        + assignments(value["reads"])
        + ", writes := "
        + assignments(value["writes"])
        + (
            ", clobbers := "
            + _list(value["clobbers"], _integer)
            + ", ordered := "
            + str(value["ordered"]).lower()
            + ", pureWrite := "
            + str(value["pureWrite"]).lower()
            + " }"
        )
    )


def _atom(value):
    return (
        "{ id := "
        + _integer(value["id"])
        + ", token := "
        + _string(value["token"])
        + (", effect := " + _effect(value["effect"]) + ", eraseAllowed := " + str(value["eraseAllowed"]).lower() + " }")
    )


def _entry(value):
    return (
        "{ name := "
        + _string(value["name"])
        + ", header := "
        + _string(value["header"])
        + (", code := " + _list(value["code"], _atom) + " }")
    )


def _program(value):
    return "{ frame := " + _string(value["frame"]) + ", entries := " + _list(value["entries"], _entry) + " }"


def emit(projection):
    if projection["schema"] != "merlin.configuration-elision.projection.v1":
        raise ValueError("complete current configuration projection required")
    field_ids = list(range(len(projection["fields"])))
    query = (
        """import Merlin.GraphStateConfigurationEntries

namespace ActualConfigurationElision
open Merlin.StateEffects
open Merlin.GraphState
open Merlin.GraphState.Configuration
set_option autoImplicit false
set_option maxRecDepth 4096
set_option maxHeartbeats 2000000

def fields : List Nat := """
        + _list(field_ids, _integer)
        + "\n"
    )
    query += "def before : Entries.Program := " + _program(projection["before"]) + "\n"
    query += "def after : Entries.Program := " + _program(projection["after"]) + "\n\n"
    query += "theorem accepted : Entries.check fields before after = true := by decide\n\n"
    query += """theorem complete_entries {Value Error : Type} (literal : Int → Value)
    (run : String → Nat → State Value → Except (Error × State Value) (State Value))
    (models : Entries.Admitted fields literal run before) (name : String) (initial : State Value) :
    Entries.invoke run before name initial = Entries.invoke run after name initial :=
  Entries.check_sound accepted literal run models name initial

theorem complete_frame : before.frame = after.frame := Entries.checked_frame accepted

theorem complete_observation {Value Error Observation : Type} (literal : Int → Value)
    (run : String → Nat → State Value → Except (Error × State Value) (State Value))
    (models : Entries.Admitted fields literal run before) (name : String) (initial : State Value)
    (observe : Option (Except (Error × State Value) (State Value)) → Observation) :
    observe (Entries.invoke run before name initial) = observe (Entries.invoke run after name initial) :=
  Entries.checked_observation accepted literal run models name initial observe

end ActualConfigurationElision
"""
    roots = [
        *SUPPORT_ROOTS,
        NAMESPACE + ".accepted",
        NAMESPACE + ".complete_frame",
        NAMESPACE + ".complete_observation",
        FINAL_ROOT,
    ]
    if len(set(roots)) != len(roots) or len(query.encode()) > 2 * 1024 * 1024:
        raise ValueError("bounded distinct complete configuration query/root roster required")
    return {
        "query": query,
        "roots": roots,
        "module": MODULE,
        "final_root": FINAL_ROOT,
        "entry_roster": [entry["name"] for entry in projection["before"]["entries"]],
        "source_status": "UNCOMPILED",
        "kernel_outcome": "UNOBSERVED",
        "proof_authority": False,
    }
