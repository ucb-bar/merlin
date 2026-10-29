# ruff: noqa: E501
"""Local segmentation proof over actual retained bodies and suffix captures.

The caller admits the original user source. For a composed chain, generated
index operations appear only after the preceding exact, proven unroll boundary.
This emitter neither grants general source arithmetic admission nor runs loops.
"""


def emit(local, universe, controls, *, tile):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.lean_graph_state_pipeline import lean_list, prefix_environment
    from merlin.verify.lean_graph_state_pipeline_emit import (
        compose_text,
        header,
        outer_text,
        scope_text,
        static_controls,
    )
    from merlin.verify.lean_graph_state_scalar import program, types
    from merlin.verify.lean_graph_state_scalar_capture import project

    pair = project(parse_ir(local["before"]), parse_ir(local["after"]), context=universe)
    capture = {**local["capture"], "stages": [pair], "index_operations": controls, "factor": 1, "tile": tile}
    text = header(capture)
    fi = len(pair["functions"]) - 1
    function = pair["functions"][fi]
    records = {2: function["before"], 3: function["before"], 4: function["after"]}
    args = [f"x{i}" for i in range(len(function["inputs"]))]
    argtext, vector = " ".join(args), lean_list(args)
    text += f"namespace F{fi}\ndef inputs : List Ty := {types(function['inputs'])}\ndef outputs : List Ty := {types(function['outputs'])}\n"
    for stage, raw in ((0, records[2]), (1, records[2]), *records.items()):
        text += f"def S{stage} : Program := {program(raw)}\ntheorem typed{stage} : checkTypes registry inputs outputs S{stage} = true := by decide\n"
    text += "theorem firstAccepted : check registry inputs outputs S0 S1 = true := by decide\ntheorem secondAccepted : check registry inputs outputs S1 S2 = true := by decide\ntheorem firstTwo : Preserves registry inputs.length S0 S2 := check_composes firstAccepted secondAccepted\n"
    loops, envs = {}, {}
    for stage, raw in records.items():
        positions = [i for i, record in enumerate(raw) if "loop" in record]
        if len(positions) != 1:
            raise ValueError("one actual local segmented loop required")
        pos = positions[0]
        loops[stage] = raw[pos]["loop"]
        envs[stage] = prefix_environment(raw, args, capture)
        text += f"def E{stage} (primitive : Prim → List Int → Int) ({argtext} : Int) : List Int := {lean_list(envs[stage])}\n"
        text += f"def B{stage} : Program := {program(loops[stage]['body'])}\ndef N{stage} : Program := {program(raw[pos + 1 :])}\n"
    bounds = static_controls(records[3], capture, input_count=len(args))
    lower, upper, step = bounds
    if step < 1 or not 0 <= lower <= upper:
        raise ValueError("static ordered nonnegative segmented controls required")
    if static_controls(records[4], capture, input_count=len(args)) != [lower, upper, step * tile]:
        raise ValueError("actual local segmented control mismatch")
    if any(loops[stage]["carries"] != loops[2]["carries"] for stage in (3, 4)):
        raise ValueError("actual ordered carried types differ")
    count = len(loops[2]["carries"])
    initial = [lean_list([envs[stage][i] for i in loops[stage]["initial"]]) for stage in (2, 3, 4)]
    if len(set(initial)) != 1:
        raise ValueError("actual ordered initial carries differ")
    text += (
        "variable {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) "
        + ("(" + argtext + " : Int)" if args else "")
        + "\n"
    )
    for stage in (2, 3, 4):
        controls_text = " ".join(envs[stage][i] for i in loops[stage]["bounds"])
        text += f"theorem form{stage} (world : World) : evalProgram (M primitive external) S{stage} {vector} world =\n"
        text += f"  (loopEval {count} (fun i carry w => evalProgram (M primitive external) B{stage} (i::carry++E{stage} primitive {argtext}) w) (indexIterations {controls_text}) {initial[0]} world).bind\n"
        text += f"    (fun result => evalProgram (M primitive external) N{stage} (result.1++E{stage} primitive {argtext}) result.2) := rfl\n"
    text += scope_text(capture, loops, envs, count, argtext)
    text += outer_text(count, upper, step, tile, argtext)
    text += compose_text(
        count, bounds, bounds, 1, (upper - lower + step - 1) // step, tile, 64, initial[0], args, identity_unroll=True
    )
    text += f"end F{fi}\nend Merlin.GraphStatePipelineInstance\n"
    base = "Merlin.GraphStatePipelineTileInstance"
    text = text.replace("Merlin.GraphStatePipelineInstance", base)
    return {
        **local,
        "capture": capture,
        "pair": pair,
        "query": text,
        "base": base,
        "source_stage": 0,
        "target_stage": 4,
    }
