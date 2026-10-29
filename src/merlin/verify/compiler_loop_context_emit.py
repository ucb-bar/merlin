# ruff: noqa: E501
"""Kernel-query joins for actual loop segments, lexical frames and entry contexts."""


def emit_suffix_bridge(row, universe):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.lean_graph_state_pipeline import lean_list
    from merlin.verify.lean_graph_state_pipeline_emit import unpack
    from merlin.verify.lean_graph_state_scalar import program
    from merlin.verify.lean_graph_state_scalar_capture import project

    local = row["local"]
    projection = project(parse_ir(row["before"]), parse_ir(row["after"]), context=universe)
    if projection["registry"] != local["capture"]["stages"][0]["registry"]:
        raise ValueError("suffix projection differs from complete registry")
    function = next(item for item in projection["functions"] if item["name"] == row["entry"])
    pos, width = row["position"], row["target_width"]
    source = function["before"][pos + 1 :]
    target = function["after"][pos + width :]
    prefix = function["after"][pos : pos + width]
    segment = [*prefix, {"ret": row["return_slots"]}]
    growth = sum(
        1 if "pure" in op else len(op["outputs"]) if "effect" in op else len(op["loop"]["carries"]) for op in prefix
    )
    size = len(local["frame"]) + growth
    left = lean_list([f"(.var {i})" for i in range(size)])
    right = lean_list([f"(.var {i})" for i in row["return_slots"]])
    base = local.get(
        "base", "Merlin.GraphStatePipeline" + ("Inline" if local["mode"] == "inline" else "Tail") + "Instance"
    )
    text = "\nnamespace Merlin.LocalUnrollSuffixBridge\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    text += f"abbrev registry := {base}.registry\n"
    text += (
        "def M {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) : Model Int World Error registry := "
        + base
        + ".M primitive external\n"
    )
    text += f"def SourceSuffix : Program := {program(source)}\ndef TargetSuffix : Program := {program(target)}\n"
    text += f"def ActualSegment : Program := {program(segment)}\n"
    text += f"def LS : List Expr := {left}\ndef RS : List Expr := {right}\n"
    text += "theorem suffixChecked : Scope.check registry LS RS TargetSuffix SourceSuffix = true := by decide\n"
    text += "theorem suffix_same {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
    text += f"    (environment : List Int) (world : World) (size : environment.length = {size}) :\n"
    text += "    evalProgram (M primitive external) TargetSuffix environment world =\n"
    text += (
        "      evalProgram (M primitive external) SourceSuffix (reads 0 environment "
        + lean_list([str(i) for i in row["return_slots"]])
        + ") world := by\n"
    )
    text += unpack("environment", "size", size, "e")
    values = lean_list([f"e{i}" for i in range(size)])
    text += f"  have checked := Scope.check_sound suffixChecked (M primitive external) {values} world\n"
    text += "  simpa only [LS,RS,evalExpr,reads,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] using checked\n"
    source_count = len(local["frame"])
    carries = len(function["before"][pos]["loop"]["carries"])
    source_segment = [function["before"][pos], {"ret": list(range(carries + source_count))}]
    text += f"def SourceSegment : Program := {program(source_segment)}\n"
    text += f"def SourceRest : Program := {program(function['before'][pos:])}\n"
    text += f"def TargetRest : Program := {program(function['after'][pos:])}\n"
    text += "theorem sourcePlug : Context.plug SourceSegment SourceSuffix = SourceRest := rfl\n"
    text += "theorem targetPlug : Context.plug ActualSegment TargetSuffix = TargetRest := rfl\n"
    text += f"theorem targetGrowth : Context.frameGrowth registry ActualSegment = {growth} := by decide\n"
    text += f"theorem sourceGrowth : Context.frameGrowth registry SourceSegment = {carries} := by decide\n"
    text += "theorem source_capture {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
    text += f"    (environment : List Int) (world : World) (size : environment.length = {carries + source_count}) :\n"
    text += "    evalProgram (M primitive external) SourceSuffix environment world =\n"
    text += (
        "      evalProgram (M primitive external) SourceSuffix (reads 0 environment "
        + lean_list([str(i) for i in range(carries + source_count)])
        + ") world := by\n"
    )
    text += unpack("environment", "size", carries + source_count, "s") + "  rfl\n"
    for side, segment_name, suffix_name, rest_name, proof_name, growth_name in (
        ("source", "SourceSegment", "SourceSuffix", "SourceRest", "source_capture", "sourceGrowth"),
        ("target", "ActualSegment", "TargetSuffix", "TargetRest", "suffix_same", "targetGrowth"),
    ):
        text += f"theorem {side}_factor {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
        text += f"    (environment : List Int) (world : World) (size : environment.length = {source_count}) :\n"
        text += f"    evalProgram (M primitive external) {rest_name} environment world =\n"
        text += f"      (evalProgram (M primitive external) {segment_name} environment world).bind\n"
        text += "        (fun result => evalProgram (M primitive external) SourceSuffix result.1 result.2) := by\n"
        text += f"  rw [← {side}Plug]\n"
        text += f"  apply Context.segment_factor (M primitive external) {segment_name} {suffix_name} SourceSuffix {source_count} environment world size\n"
        text += f"  intro final state arity\n  rw [{growth_name}] at arity\n"
        text += f"  exact {proof_name} primitive external final state arity\n"
    text += "end Merlin.LocalUnrollSuffixBridge\n"
    return "import Merlin.GraphStatePipelineContext\nimport Merlin.GraphStatePipelineScope\n" + local["query"] + text


def emit_prefix_facts(row, universe, atomic_module, *, namespace="Merlin.LocalPrefixFacts"):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.compiler_loop_nested import _run, _simulate
    from merlin.verify.compiler_loop_resources import check_lanes
    from merlin.verify.lean_graph_state_pipeline import lean_list
    from merlin.verify.lean_graph_state_pipeline_emit import unpack
    from merlin.verify.lean_graph_state_scalar_capture import project

    local = row["local"]
    projection = project(parse_ir(row["before"]), parse_ir(row["after"]), context=universe)
    function = next(item for item in projection["functions"] if item["name"] == row["entry"])
    pos = row["position"]
    if function["before"][:pos] != function["after"][:pos]:
        raise ValueError("common prefix projection must be identical")
    n = len(local["frame"])
    k = len(function["inputs"])
    args = [f"a{i}" for i in range(k)]
    prefix = [*function["before"][:pos], {"ret": list(range(n))}]
    check_lanes(local["capture"], [{"ret": []}], repetitions=0, environment=args, carry=[], continuation=prefix)
    base = local.get(
        "base", "Merlin.GraphStatePipeline" + ("Inline" if local["mode"] == "inline" else "Tail") + "Instance"
    )
    text = atomic_module.emit(
        local["capture"], prefix, prefix, arguments=args, substitutions={}, model_base=base, namespace=namespace
    )
    text = text.split("theorem actual_inputs", 1)[0]
    controls = {item["identity"]: item for item in local["capture"]["index_operations"]}
    final, calls = _simulate(prefix, args, controls)
    body = function["before"][pos]["loop"]
    substitutions = dict(zip(body["bounds"], local["controls"], strict=True))
    goals = [
        f"resultValues.length = {n}",
        *[f"(resultValues[{slot}]?.getD 0) = {value}" for slot, value in substitutions.items()],
    ]
    conjunction = " ∧ ".join(goals)
    text += "theorem facts (world : World) (resultValues : List Int) (resultWorld : World)\n"
    text += f"    (run : evalProgram (M primitive external) P2 {lean_list(args)} world = .ok (resultValues,resultWorld)) :\n"
    text += "    " + conjunction + " := by\n"
    pieces = [f"Piece2_{i}" for i in range(len(prefix))]
    nested = [f"nestedExecute2_{i}" for i, item in enumerate(prefix) if "loop" in item]
    rules = [
        "nestedRet",
        "nestedPure",
        *[f"nestedEffect{i}" for i in range(len(local["capture"]["stages"][0]["registry"]))],
    ]
    simp = "reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte"
    simp = ",".join(nested + rules + [simp])
    text += "  simp only [P2_factored," + ",".join(pieces) + "," + simp + "] at run\n"
    for call in calls:
        i = call["index"]
        pad = "  " + "  " * i
        expression = (
            _run(2, call["position"], call["environment"], call["world"])
            if "loop" in call
            else f"external.run {call['identity']} {lean_list(call['arguments'])} {call['world']}"
        )
        text += f"{pad}cases step{i} : {expression} with\n{pad}| error failure =>\n{pad}  rcases failure with ⟨error,terminal⟩\n"
        text += f"{pad}  simp only [step{i},{simp}] at run\n{pad}  cases run\n"
        text += f"{pad}| ok result =>\n{pad}  rcases result with ⟨values,w{i}⟩\n{pad}  have size : values.length = {call['outputs']} := by\n"
        if "loop" in call:
            text += f"{pad}    have shaped := step{i}\n{pad}    unfold NestedRun2_{call['position']} at shaped\n{pad}    exact loopEval_arity _ _ _ _ _ shaped\n"
        else:
            text += f"{pad}    have localArity := external.arity {call['identity']} {lean_list(call['arguments'])} {call['world']} values w{i} step{i}\n"
            text += f"{pad}    simpa only [registry,{base}.registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity\n"
        text += unpack("values", "size", call["outputs"], f"v{i}_", pad + "  ")
        text += f"{pad}  simp only [step{i},{simp}] at run\n"
    pad = "  " + "  " * len(calls)
    text += f"{pad}have returned : resultValues = {lean_list(final)} := (congrArg Prod.fst (Except.ok.inj run)).symm\n"
    text += f"{pad}rw [returned]\n{pad}exact ⟨" + ",".join("rfl" for _ in goals) + "⟩\n"
    text += f"end {namespace}\n"
    return text, {
        "prefix": prefix,
        "inputs": function["inputs"],
        "frame_count": n,
        "substitutions": substitutions,
        "base": base,
        "namespace": namespace,
        "function": function,
    }


def emit_stage(row, universe, atomic_module):
    from merlin.transforms.certificates import parse_ir
    from merlin.verify.lean_graph_state_pipeline import lean_list
    from merlin.verify.lean_graph_state_pipeline_emit import entry_arguments, unpack
    from merlin.verify.lean_graph_state_scalar import program, types
    from merlin.verify.lean_graph_state_scalar_capture import project

    local = row["local"]
    capture = local["capture"]
    projection = project(parse_ir(row["before"]), parse_ir(row["after"]), context=universe)
    function = next(item for item in projection["functions"] if item["name"] == row["entry"])
    pos, width = row["position"], row["target_width"]
    loop = function["before"][pos]["loop"]
    n = len(local["frame"])
    c = len(loop["carries"])
    args = [f"x{i}" for i in range(n)]
    argtext = " ".join(args)
    vector = lean_list(args)
    substitutions = dict(zip(loop["bounds"], local["controls"], strict=True))
    conditions = " ".join(f"(control{i} : x{slot} = {value})" for i, (slot, value) in enumerate(substitutions.items()))
    controltext = " ".join(f"control{i}" for i in range(len(substitutions)))
    source = [function["before"][pos], {"ret": list(range(c + n))}]
    target = [*function["after"][pos : pos + width], {"ret": row["return_slots"]}]
    wrapper = local.get("pair", capture["stages"][0])["functions"][-1]
    fi = len(capture["stages"][0]["functions"]) - 1
    base = local.get(
        "base", "Merlin.GraphStatePipeline" + ("Inline" if local["mode"] == "inline" else "Tail") + "Instance"
    )
    s0, s1 = (0, 1) if local["mode"] == "inline" else (2, 3)
    s0, s1 = local.get("source_stage", s0), local.get("target_stage", s1)
    text = "import Merlin.GraphStatePipelineContext\nimport Merlin.GraphStatePipelineScope\n" + local["query"]
    text += atomic_module.emit(
        capture,
        source,
        wrapper["before"],
        arguments=args,
        substitutions=substitutions,
        model_base=base,
        namespace="Merlin.LocalSourceAtomicBridge",
    )
    text += atomic_module.emit(
        capture,
        target,
        wrapper["after"],
        arguments=args,
        substitutions=substitutions,
        model_base=base,
        namespace="Merlin.LocalTargetAtomicBridge",
    )
    suffix = emit_suffix_bridge(row, universe).split("namespace Merlin.LocalUnrollSuffixBridge\n", 1)[1]
    text += "namespace Merlin.LocalUnrollSuffixBridge\n" + suffix
    model = f"({base}.M primitive external)"
    suffixbase = "Merlin.LocalUnrollSuffixBridge"
    sourceatom = "Merlin.LocalSourceAtomicBridge"
    targetatom = "Merlin.LocalTargetAtomicBridge"
    text += "namespace Merlin.LocalRestBridge\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    text += f"abbrev registry := {base}.registry\n"
    text += "theorem actual_inputs {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
    text += ("    (" + argtext + " : Int)\n" if args else "") + f"    (world : World) {conditions} :\n"
    text += f"    evalProgram {model} {suffixbase}.SourceRest {vector} world = evalProgram {model} {suffixbase}.TargetRest {vector} world := by\n"
    text += f"  have first : evalProgram {model} {suffixbase}.SourceSegment {vector} world = evalProgram {model} {base}.F{fi}.S{s0} {vector} world :=\n"
    text += f"    {sourceatom}.actual_inputs primitive external {argtext} world {controltext}\n"
    text += f"  have middle := {base}.F{fi}.{local.get('proof_name', 'actual')} primitive external {vector} world rfl\n"
    text += f"  have last : evalProgram {model} {base}.F{fi}.S{s1} {vector} world = evalProgram {model} {suffixbase}.ActualSegment {vector} world :=\n"
    text += f"    ({targetatom}.actual_inputs primitive external {argtext} world {controltext}).symm\n"
    observer = f"(fun result => evalProgram {model} {suffixbase}.SourceSuffix result.1 result.2)"
    text += "  calc\n"
    text += f"    evalProgram {model} {suffixbase}.SourceRest {vector} world =\n"
    text += f"        (evalProgram {model} {suffixbase}.SourceSegment {vector} world).bind {observer} :=\n"
    text += f"      {suffixbase}.source_factor primitive external {vector} world rfl\n"
    text += f"    _ = (evalProgram {model} {suffixbase}.ActualSegment {vector} world).bind {observer} :=\n"
    text += f"      congrArg (fun result => result.bind {observer}) (first.trans (middle.trans last))\n"
    text += f"    _ = evalProgram {model} {suffixbase}.TargetRest {vector} world :=\n"
    text += f"      ({suffixbase}.target_factor primitive external {vector} world rfl).symm\n"
    text += "end Merlin.LocalRestBridge\n"
    prefixtext, info = emit_prefix_facts(row, universe, atomic_module)
    text += prefixtext
    prefixbase = info["namespace"]
    k = len(function["inputs"])
    entryargs = [f"a{i}" for i in range(k)]
    entrytext = " ".join(entryargs)
    entryvector = lean_list(entryargs)
    text += "namespace Merlin.LocalWholeBridge\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    text += f"abbrev registry := {base}.registry\n"
    text += (
        f"def inputs : List Ty := {types(function['inputs'])}\ndef outputs : List Ty := {types(function['outputs'])}\n"
    )
    text += f"def S0 : Program := {program(function['before'])}\ndef S1 : Program := {program(function['after'])}\n"
    text += "theorem typed0 : checkTypes registry inputs outputs S0 = true := by decide\ntheorem typed1 : checkTypes registry inputs outputs S1 = true := by decide\n"
    text += f"abbrev Prefix := {prefixbase}.P2\n"
    text += f"theorem sourceJoin : Context.plug Prefix {suffixbase}.SourceRest = S0 := rfl\n"
    text += f"theorem targetJoin : Context.plug Prefix {suffixbase}.TargetRest = S1 := rfl\n"
    text += f"theorem prefixGrowth : {k} + Context.frameGrowth registry Prefix = {n} := by decide\n"
    text += f"theorem prefixReturnsAll (environment : List Int) (size : environment.length = {n}) :\n"
    text += (
        "    reads 0 environment (Context.returnSlots Prefix) = environment := by\n"
        + unpack("environment", "size", n, "e")
        + "  rfl\n"
    )
    text += (
        "variable {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry)\n"
    )
    text += f"theorem prefix_capture (observer : Program) (environment : List Int) (world : World) (size : environment.length = {n}) :\n"
    text += f"    evalProgram {model} observer environment world = evalProgram {model} observer (reads 0 environment (Context.returnSlots Prefix)) world := by\n"
    text += "  rw [prefixReturnsAll environment size]\n"
    for side, rest in (("source", "SourceRest"), ("target", "TargetRest")):
        text += f"theorem {side}_factor (arguments : List Int) (world : World) (arity : arguments.length = {k}) :\n"
        text += f"    evalProgram {model} " + ("S0" if side == "source" else "S1") + " arguments world =\n"
        text += f"      (evalProgram {model} Prefix arguments world).bind (fun result => evalProgram {model} {suffixbase}.{rest} result.1 result.2) := by\n"
        text += f"  rw [← {side}Join]\n"
        text += f"  apply Context.segment_factor {model} Prefix {suffixbase}.{rest} {suffixbase}.{rest} {k} arguments world arity\n"
        text += "  intro final state size\n  rw [prefixGrowth] at size\n"
        text += f"  exact prefix_capture primitive external {suffixbase}.{rest} final state size\n"
    text += "theorem actual_inputs " + ("(" + entrytext + " : Int) " if entryargs else "") + "(world : World) :\n"
    text += f"    evalProgram {model} S0 {entryvector} world = evalProgram {model} S1 {entryvector} world := by\n"
    text += f"  rw [source_factor primitive external {entryvector} world rfl,target_factor primitive external {entryvector} world rfl]\n"
    text += f"  cases run : evalProgram {model} Prefix {entryvector} world with\n  | error failure => rfl\n  | ok result =>\n    rcases result with ⟨values,afterPrefix⟩\n    simp only [Except.bind]\n"
    text += f"    have facts := {prefixbase}.facts primitive external {entrytext} world values afterPrefix run\n"
    text += "    rcases facts with ⟨size," + ",".join(f"control{i}" for i in range(len(substitutions))) + "⟩\n"
    text += unpack("values", "size", n, "p", "    ")
    text += (
        "    exact Merlin.LocalRestBridge.actual_inputs primitive external "
        + " ".join(f"p{i}" for i in range(n))
        + " afterPrefix "
        + controltext
        + "\n"
    )
    text += "theorem actual (arguments : List Int) (world : World) (arity : arguments.length = inputs.length) :\n"
    text += (
        f"    evalProgram {model} S0 arguments world = evalProgram {model} S1 arguments world := by\n"
        + entry_arguments(k, "actual_inputs")
    )
    text += "end Merlin.LocalWholeBridge\n"
    return text, info
