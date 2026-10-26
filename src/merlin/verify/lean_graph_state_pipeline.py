"""Proof-producing scalar/SCF pipeline leaf (development, not qualified).

Raw Programs, complete common registry, typed scopes and operational claims are
emitted into kernel-checked Lean. Python projection is explicitly trusted.
No compiler trace, diagnostic inventory or previous receipt grants acceptance.
"""

from __future__ import annotations

from .lean_graph_state_scalar import integer, primitive


def lean_list(values):
    return "[" + ",".join(values) + "]"


def controls_literal(capture):
    result = ["none"] * len(capture["stages"][0]["registry"])
    for row in capture["index_operations"]:
        identity = row["identity"]
        integer(identity, natural=True)
        if not 0 <= identity < len(result) or result[identity] != "none":
            raise ValueError("invalid/duplicate control identity")
        kind = row["kind"]
        if kind == "index-constant" and set(row) == {"identity", "kind", "value"}:
            op = "(.constant " + integer(row["value"]) + ")"
        elif kind in ("index-add", "index-minui") and set(row) == {"identity", "kind"}:
            op = ".add" if kind == "index-add" else ".minui"
        else:
            raise ValueError("closed actual supported control descriptor required")
        result[identity] = "(some " + op + ")"
    return lean_list(result)


def prefix_environment(records, inputs, capture):
    """Symbolic values of actual pre-loop total operations; no effect erasure."""
    env = list(inputs)
    controls = {row["identity"]: row for row in capture["index_operations"]}
    for record in records:
        if "loop" in record:
            return env
        args = [env[index] for index in record["arguments"]]
        if "pure" in record:
            value = "(primitive " + primitive(record["pure"]) + " " + lean_list(args) + ")"
        elif "effect" in record and record["effect"] in controls:
            control = controls[record["effect"]]
            if control["kind"] == "index-constant" and not args:
                value = "(" + str(control["value"]) + " : Int)"
            elif control["kind"] == "index-add" and len(args) == 2:
                value = "(" + args[0] + " + " + args[1] + ")"
            elif control["kind"] == "index-minui" and len(args) == 2:
                value = "(min " + args[0] + " " + args[1] + ")"
            else:
                raise ValueError("malformed prefix control operands")
        else:
            raise ValueError("partial/effectful prefix unsupported by this development submode")
        env.insert(0, value)
    raise ValueError("missing actual loop")


SIMPS = (
    "M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,IndexOp.eval,opaqueStep,evalProgram,reads,"
    "List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,"
    "List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,"
    "List.length_cons,List.length_nil,↓reduceIte"
)


def emit_development(capture):
    """Compatibility entry for the current proof-producing development checker."""
    from .lean_graph_state_pipeline_emit import emit

    return emit(capture)
