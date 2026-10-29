"""Kernel obligation for complete typed shape-normalization module blueprints."""

from .lean_graph_loops_unroll import _list, _nat, _string

SHAPE_ROOTS = (
    "GraphShapeObligation.valid",
    "GraphShapeObligation.accepted",
    "GraphShapeObligation.complete",
    "GraphShapeObligation.memory",
    "GraphShapeObligation.contextual",
    "GraphShapeObligation.call_depth",
    "Merlin.GraphLoopsShape.identical_cast",
    "Merlin.GraphLoopsShape.cast_descriptor",
    "Merlin.GraphLoopsShape.cast_address",
    "Merlin.GraphLoopsShape.static_dimension",
    "Merlin.GraphLoopsShape.Scope.extend",
    "Merlin.GraphLoopsShape.Agreement.define",
    "Merlin.GraphLoopsShape.Agreement.alias",
    "Merlin.GraphLoopsShape.Agreement.define_many",
    "Merlin.GraphLoopsShape.cast_result",
    "Merlin.GraphLoopsShape.dimension_result",
    "Merlin.GraphLoopsShape.preservation",
    "Merlin.GraphLoopsShape.entry_equivalence",
    "Merlin.GraphLoopsShape.Function.preservation",
    "Merlin.GraphLoopsShape.Module.checked_calls",
    "Merlin.GraphLoopsShape.Module.checked_complete",
    "Merlin.GraphLoopsShape.Module.checked_memory",
    "Merlin.GraphLoopsShape.Model.check_sound",
    "Merlin.GraphLoopsShape.Model.checked_complete",
    "Merlin.GraphLoopsShape.Model.checked_memory",
    "Merlin.GraphLoopsShape.Model.checked_context",
    "Merlin.GraphLoopsShape.zero_has_no_coordinate",
    "Merlin.GraphLoopsShape.inconsistent_dimension",
    "Merlin.GraphLoopsShape.observable_cast_countermodel",
    "Merlin.GraphLoopsShape.Countermodels.same_type_different_allocation",
    "Merlin.GraphLoopsShape.Countermodels.same_type_cannot_change_buffer",
    "Merlin.GraphLoopsShape.Countermodels.logging_deletion",
    "Merlin.GraphLoopsShape.Countermodels.trapping_deletion",
    "Merlin.GraphLoopsShape.Countermodels.inconsistent_metadata",
    "Merlin.GraphLoopsShape.Countermodels.same_type_different_base",
)


def view_type(row):
    if set(row) != {"shape", "offset", "strides", "element", "space"}:
        raise ValueError("complete descriptor type fields required")
    return (
        "⟨"
        + ", ".join(
            (
                _list(row["shape"], _nat),
                _nat(row["offset"]),
                _list(row["strides"], _nat),
                _string(row["element"]),
                _nat(row["space"]),
            )
        )
        + "⟩"
    )


def kind(row):
    if row == {"tag": "index"}:
        return ".index"
    if set(row) == {"tag", "type"} and row["tag"] == "scalar":
        return "(.scalar " + _string(row["type"]) + ")"
    if set(row) == {"tag", "type"} and row["tag"] == "view":
        return "(.view " + view_type(row["type"]) + ")"
    raise ValueError("unknown typed graph value")


def binding(row):
    if set(row) != {"name", "kind", "known"}:
        raise ValueError("complete lexical SSA binding required")
    known = "none" if row["known"] is None else "(some " + _nat(row["known"]) + ")"
    return "⟨" + ", ".join((_nat(row["name"]), kind(row["kind"]), known)) + "⟩"


def bindings(rows):
    return _list(rows, binding)


def code(row):
    tag = row["tag"]
    expected = {
        "returned": {"values"},
        "cast": {"origin", "result", "source", "target", "rest"},
        "dimension": {"view", "axis", "result", "type", "axisIndex", "value", "rest"},
        "constant": {"result", "value", "rest"},
        "action": {"key", "operands", "results", "rest"},
        "region": {"key", "operands", "parameters", "results", "body", "rest"},
    }
    if tag not in expected or set(row) != expected[tag] | {"tag"}:
        raise ValueError("unknown/incomplete program node")
    if tag == "returned":
        args = [bindings(row["values"])]
    elif tag == "cast":
        args = [binding(row["origin"]), binding(row["result"]), view_type(row["source"]), view_type(row["target"])]
    elif tag == "dimension":
        args = [binding(row[name]) for name in ("view", "axis", "result")] + [
            view_type(row["type"]),
            _nat(row["axisIndex"]),
            _nat(row["value"]),
        ]
    elif tag == "constant":
        args = [binding(row["result"]), _nat(row["value"])]
    elif tag == "action":
        args = [_string(row["key"]), bindings(row["operands"]), bindings(row["results"])]
    else:
        args = [
            _string(row["key"]),
            bindings(row["operands"]),
            bindings(row["parameters"]),
            bindings(row["results"]),
            code(row["body"]),
        ]
    if tag != "returned":
        args.append(code(row["rest"]))
    return "(." + tag + " " + " ".join(args) + ")"


def function(row):
    if set(row) != {"name", "parameters", "returned", "code"}:
        raise ValueError("complete function signature and body required")
    return (
        "⟨"
        + ", ".join(
            (_string(row["name"]), bindings(row["parameters"]), _list(row["returned"], kind), code(row["code"]))
        )
        + "⟩"
    )


def obligation(projection):
    module = projection["module"]
    if set(module) != {"functions", "calls"}:
        raise ValueError("complete function/call environment required")
    calls = _list(module["calls"], lambda pair: "(" + _string(pair[0]) + ", " + _string(pair[1]) + ")")
    literal = (
        "⟨"
        + ", ".join(
            (
                _nat(projection["bits"]),
                "⟨" + _list(module["functions"], function) + ", " + calls + "⟩",
                _string(projection["source_binding"]),
                _string(projection["target_binding"]),
            )
        )
        + "⟩"
    )
    roots = "\n".join("#print axioms " + root for root in SHAPE_ROOTS)
    return f"""import Merlin.GraphLoopsShapeCountermodels
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 4000000
open Merlin Merlin.GraphLoopsShape
namespace GraphShapeObligation
def actual : Model.Projection := {literal}
theorem valid : Model.Valid actual := by
  simp [Model.Valid, Module.Valid, Function.Valid, GraphLoopsShape.Valid, Scope,
    CastFacts, DimensionFacts, ConstantFacts, AllBound, Plain, Fresh, names,
    extendContext, extendAliases, resolve, ValidDimension, actual]
  all_goals decide
theorem accepted : Model.check actual = true := by
  simp only [Model.check, valid, decide_true]
theorem complete {{Value World : Type}} (meaning : Interpretation Value World)
    (name : String) (inputs : List (Runtime Value)) (before : World)
    (outputs : List (Runtime Value)) (after : World) :
    GraphLoopsShape.complete meaning actual.module false name inputs before outputs after ↔
      GraphLoopsShape.complete meaning actual.module true name inputs before outputs after :=
  Model.checked_complete accepted meaning name inputs before outputs after
theorem memory {{Value Public : Type}} (meaning : Interpretation Value (MemoryWorld Value Public))
    (classify : String → MemoryOpcode) (name : String) (inputs : List (Runtime Value))
    (before : MemoryWorld Value Public) (outputs : List (Runtime Value)) (after : MemoryWorld Value Public) :
    GraphLoopsShape.complete (withMemory meaning classify) actual.module false name inputs before outputs after ↔
      GraphLoopsShape.complete (withMemory meaning classify) actual.module true name inputs before outputs after :=
  Model.checked_memory accepted meaning classify name inputs before outputs after
theorem contextual {{Value World Outcome : Type}} (meaning : Interpretation Value World)
    (context : (String → RegionMeaning Value World) → Outcome) :
    context (GraphLoopsShape.complete meaning actual.module false) =
      context (GraphLoopsShape.complete meaning actual.module true) := Model.checked_context accepted meaning context
theorem call_depth {{Value World : Type}} (meaning : Interpretation Value World) (depth : Nat) :
    invoke meaning actual.module false depth = invoke meaning actual.module true depth :=
  Module.checked_calls actual.module actual.bits (Model.check_sound accepted).1 meaning depth
end GraphShapeObligation
{roots}
"""
