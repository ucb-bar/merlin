"""Actual exact-middle scalar chain in one checked descriptor/binder semantics.

Component theorems and their transitive composition are distinct from the fresh
direct original/final check. All routes live in one explicitly reported object;
this does not claim independently imported stage-object custody.
"""

from __future__ import annotations

from .lean_graph_state_scalar import emit_query, program


def emit_chain_query(projection):
    stages = projection["chain_stages"]
    if set(stages) != {"first", "second"}:
        raise ValueError("complete actual two-stage chain projection required")
    first, second = stages["first"], stages["second"]
    if any(
        projection[key] != first[key] or first[key] != second[key] for key in ("registry", "types", "module_header")
    ):
        raise ValueError("actual chain semantic/type/module contexts differ")
    if len(projection["functions"]) != len(first["functions"]) or len(first["functions"]) != len(second["functions"]):
        raise ValueError("complete function rosters differ across actual stages")
    text, roots = emit_query(projection)
    text = text.removesuffix("end Merlin.GraphStateScalarInstance\n")
    for index, (direct, left, right) in enumerate(
        zip(projection["functions"], first["functions"], second["functions"], strict=True)
    ):
        for key in ("name", "header", "inputs", "outputs", "declaration"):
            if direct[key] != left[key] or left[key] != right[key]:
                raise ValueError("actual chain changed an interface, function, or declaration")
        if left["after"] != right["before"]:
            raise ValueError("exact middle SSA/control/use-def graph differs")
        if direct["before"] != left["before"] or direct["after"] != right["after"]:
            raise ValueError("direct endpoints differ from component endpoints")
        if direct["declaration"]:
            continue
        namespace = "F" + str(index)
        text += (
            "namespace " + namespace + "\n"
            "def componentBefore : Program := " + program(left["before"]) + "\n"
            "def middleLeft : Program := " + program(left["after"]) + "\n"
            "def middleRight : Program := " + program(right["before"]) + "\n"
            "def componentAfter : Program := " + program(right["after"]) + "\n"
            "theorem source_join : componentBefore = before := by decide\n"
            "theorem middle_join : middleLeft = middleRight := by decide\n"
            "theorem final_join : componentAfter = after := by decide\n"
            "theorem first_accepted : check registry inputs outputs componentBefore middleLeft = true := by decide\n"
            "theorem second_accepted : check registry inputs outputs middleRight componentAfter = true := by decide\n"
            "theorem first_actual : Preserves registry inputs.length componentBefore middleLeft :=\n"
            "  check_sound first_accepted\n"
            "theorem second_actual : Preserves registry inputs.length middleRight componentAfter :=\n"
            "  check_sound second_accepted\n"
            "theorem middle_typed : Typed registry inputs outputs middleLeft := (check_typed first_accepted).2\n"
            "theorem independently_composed : Preserves registry inputs.length before after := by\n"
            "  have linked : Preserves registry inputs.length middleLeft componentAfter := by\n"
            "    rw [middle_join]; exact second_actual\n"
            "  simpa only [source_join,final_join] using preserves_trans first_actual linked\n"
            "end " + namespace + "\n"
        )
        roots += [
            "Merlin.GraphStateScalarInstance." + namespace + "." + name
            for name in (
                "source_join",
                "middle_join",
                "final_join",
                "first_accepted",
                "second_accepted",
                "first_actual",
                "second_actual",
                "middle_typed",
                "independently_composed",
            )
        ]
    return text + "end Merlin.GraphStateScalarInstance\n", roots
