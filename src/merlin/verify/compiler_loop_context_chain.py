# ruff: noqa: E501
"""Adjacent whole-entry roots over an exact actual three-buffer loop chain."""


def append_boundaries(query, projection, middle, proofs):
    from merlin.verify.lean_graph_state_pipeline import lean_list
    from merlin.verify.lean_graph_state_scalar import program

    if projection["registry"] != middle["registry"]:
        raise ValueError("one complete actual chain registry required")
    query += "namespace Merlin.GraphStatePipelineInstance\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    defined = []
    for i, (function, retained) in enumerate(zip(projection["functions"], middle["functions"], strict=True)):
        if function["name"] != retained["name"] or function["declaration"] != retained["declaration"]:
            raise ValueError("complete actual intermediate entry roster differs")
        if function["declaration"]:
            continue
        defined.append(i)
        if function["inputs"] != retained["inputs"] or function["outputs"] != retained["outputs"]:
            raise ValueError("complete actual intermediate signature differs")
        query += f"namespace F{i}\ndef Middle : Program := {program(retained['before'])}\n"
        query += "theorem typedMiddle : checkTypes registry inputs outputs Middle = true := by decide\n"
        for boundary, first, last in ((2, "S0", "Middle"), (3, "Middle", "S1")):
            query += f"theorem boundary{boundary} {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
            query += "    (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :\n"
            query += f"    evalProgram (M primitive external) {first} arguments world = evalProgram (M primitive external) {last} arguments world := "
            steps = proofs[boundary].get(function["name"], [])
            if not steps:
                original = function["before"] if boundary == 2 else retained["before"]
                final = retained["before"] if boundary == 2 else function["after"]
                if original != final:
                    raise ValueError("unproved actual chain boundary changed")
                query += "rfl\n"
            else:
                proof = f"({steps[-1]} primitive external arguments world arity)"
                for step in reversed(steps[:-1]):
                    proof = f"({step} primitive external arguments world arity).trans ({proof})"
                query += proof + "\n"
        query += f"end F{i}\n"
    query += (
        "def middleEntries : List Entry := "
        + lean_list([f"{{inputs := F{i}.inputs,outputs := F{i}.outputs,body := F{i}.Middle}}" for i in defined])
        + "\n"
    )
    query += "theorem middleEntriesChecked : checkEntries registry middleEntries = true := by decide\n"
    for boundary, first, last in ((2, "original", "middle"), (3, "middle", "final")):
        proof = "entries_nil (M primitive external)"
        for i in reversed(defined):
            proof = (
                f"entries_cons (M primitive external) rfl rfl (F{i}.boundary{boundary} primitive external) ({proof})"
            )
        query += f"theorem boundary{boundary} {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry) :\n"
        query += f"    PreservesEntries (M primitive external) {first}Entries {last}Entries :=\n  {proof}\n"
    return query + "end Merlin.GraphStatePipelineInstance\n"
