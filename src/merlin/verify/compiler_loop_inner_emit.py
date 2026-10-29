# ruff: noqa: E501
"""Scratch inherited lexical-body congruence through an actual enclosing loop."""


def emit(
    row,
    universe,
    atomic_module,
    inherited,
    *,
    child,
    rest_name="Merlin.InnerParentRest",
    prefix_name="Merlin.InnerParentPrefix",
    whole_name="Merlin.InnerParentWhole",
):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.lean_graph_state_pipeline import lean_list
    from merlin.verify.lean_graph_state_pipeline_emit import unpack
    from merlin.verify.lean_graph_state_scalar import program
    from merlin.verify.lean_graph_state_scalar_capture import project

    projection = row.get("projection") or project(parse_ir(row["before"]), parse_ir(row["after"]), context=universe)
    function = next(f for f in projection["functions"] if f["name"] == row["entry"])
    pos = row["position"]
    source, target = (function[side][pos]["loop"] for side in ("before", "after"))
    if (
        any(source[key] != target[key] for key in ("bounds", "initial", "carries"))
        or function["before"][pos + 1 :] != function["after"][pos + 1 :]
    ):
        raise ValueError("actual ancestor header or suffix changed")
    local = row["local"]
    n = len(local["frame"])
    c = len(source["carries"])
    base = local.get(
        "base", "Merlin.GraphStatePipeline" + ("Inline" if local["mode"] == "inline" else "Tail") + "Instance"
    )
    model = f"({base}.M primitive external)"
    substitutions = row["required_facts"]
    args = [f"x{i}" for i in range(n)]
    argtext = " ".join(args)
    vector = lean_list(args)
    conditions = " ".join(f"(control{i} : x{slot} = {value})" for i, (slot, value) in enumerate(substitutions.items()))
    text = f"namespace {rest_name}\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    text += f"abbrev registry := {base}.registry\n"
    text += f"def SourceRest : Program := {program(function['before'][pos:])}\ndef TargetRest : Program := {program(function['after'][pos:])}\n"
    text += f"def SourceBody : Program := {program(source['body'])}\ndef TargetBody : Program := {program(target['body'])}\n"
    text += f"def Suffix : Program := {program(function['before'][pos + 1 :])}\n"
    text += f"theorem sourceBodyJoin : SourceBody = {child['whole']}.S0 := by unfold {child['whole']}.S0; rfl\ntheorem targetBodyJoin : TargetBody = {child['whole']}.S1 := by unfold {child['whole']}.S1; rfl\n"
    text += "theorem actual_inputs {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
    text += ("    (" + argtext + " : Int)\n" if args else "") + f"    (world : World) {conditions} :\n"
    text += f"    evalProgram {model} SourceRest {vector} world = evalProgram {model} TargetRest {vector} world := by\n"
    first = f"(fun iv values state => evalProgram {model} SourceBody (iv::values++{vector}) state)"
    second = f"(fun iv values state => evalProgram {model} TargetBody (iv::values++{vector}) state)"
    text += f"  have bodies : ∀ iv values state, values.length = {c} →\n"
    text += f"      evalProgram {model} SourceBody (iv::values++{vector}) state = evalProgram {model} TargetBody (iv::values++{vector}) state := by\n"
    text += "    intro iv values state size\n" + unpack("values", "size", c, "c", "    ")
    text += "    rw [sourceBodyJoin,targetBodyJoin]\n"
    names = {slot: f"control{i}" for i, slot in enumerate(substitutions)}
    inherited_args = []
    for slot, value in child["inherited"].items():
        parent_slot = slot - (1 + c)
        if parent_slot not in substitutions or substitutions[parent_slot] != value:
            raise ValueError("inherited body control did not join actual ancestor frame")
        inherited_args.append(names[parent_slot])
    text += (
        f"    exact {child['whole']}.actual_inputs primitive external "
        + " ".join(["iv", *[f"c{i}" for i in range(c)], *args])
        + " state "
        + " ".join(inherited_args)
        + "\n"
    )
    bounds = " ".join(f"({vector}[{slot}]?.getD 0)" for slot in source["bounds"])
    indices = f"(indexIterations {bounds})"
    initial = f"(reads 0 {vector} " + lean_list([str(i) for i in source["initial"]]) + ")"
    observer = f"(fun result => evalProgram {model} Suffix (result.1++{vector}) result.2)"
    text += f"  have loops := loopEval_congr {c} {first} {second} bodies {indices} {initial} world\n"
    text += f"  change (loopEval {c} {first} {indices} {initial} world).bind {observer} =\n"
    text += f"    (loopEval {c} {second} {indices} {initial} world).bind {observer}\n  rw [loops]\n"
    text += f"end {rest_name}\n"
    prefixtext, info = inherited.emit_prefix_facts(row, universe, atomic_module, namespace=prefix_name)
    whole, info = inherited.emit_whole_context(
        row,
        function,
        base,
        n,
        prefixtext,
        info,
        rest_namespace=rest_name,
        suffix_namespace=rest_name,
        whole_namespace=whole_name,
    )
    return text + whole, info
