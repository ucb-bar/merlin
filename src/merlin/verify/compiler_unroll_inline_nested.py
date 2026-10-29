# ruff: noqa: E501
# Keep generated operational equations readable.
"""Candidate atomic nested-event inline proof, generated from actual programs."""

from .compiler_loop_nested import _loop_term, _model_rules, _run, _shape, _simulate
from .lean_graph_state_pipeline import lean_list
from .lean_graph_state_pipeline_emit import unpack
from .lean_graph_state_scalar import program


def emit(capture, before, after, loop, env, initial, trips, lower, upper, step, args):
    text, rules = _model_rules(capture)
    pos = next(i for i, row in enumerate(before) if "loop" in row)
    from .compiler_loop_resources import check_lanes

    check_lanes(
        capture,
        loop["body"],
        repetitions=trips,
        environment=env,
        carry=initial,
        continuation=before[pos + 1 :],
    )
    if before[:pos] != after[:pos]:
        raise ValueError("actual inline prefix must retain exact SSA context")
    text += f"def T : Program := {program(after[pos:])}\n"
    text += (
        "theorem target_form (world : World) : evalProgram (M primitive external) S1 "
        + lean_list(args)
        + " world = evalProgram (M primitive external) T "
        + lean_list(env)
        + " world := rfl\n"
    )
    controls = {row["identity"]: row for row in capture["index_operations"]}
    groups = {
        2: loop["body"],
        3: after[pos:],
        4: before[next(i for i, row in enumerate(before) if "loop" in row) + 1 :],
    }
    pieces = []
    executes = []
    for stage, records in groups.items():
        for i, row in enumerate(records):
            if "loop" not in row:
                continue
            node = row["loop"]
            n = len(node["carries"])
            text += f"@[irreducible] def NestedBody{stage}_{i} : Program := {program(node['body'])}\n"
            text += f"def NestedProgram{stage}_{i} : Program := {_loop_term(node, program(node['body']), '(.ret ' + lean_list([str(j) for j in range(n)]) + ')')}\n"
            bounds = " ".join(f"(environment[{j}]?.getD 0)" for j in node["bounds"])
            initial_slots = lean_list([str(j) for j in node["initial"]])
            text += f"@[irreducible] def NestedRun{stage}_{i} {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry) (environment : List Int) (world : World) : Outcome Int World Error :=\n"
            text += f"  loopEval {n} (fun iv values state => evalProgram (M primitive external) NestedBody{stage}_{i} (iv::values++environment) state) (indexIterations {bounds}) (reads 0 environment {initial_slots}) world\n"
            text += f"theorem nestedExecute{stage}_{i} {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry) (next : Program) (environment : List Int) (world : World) :\n"
            text += f"  evalProgram (M primitive external) {_loop_term(node, f'NestedBody{stage}_{i}', 'next')} environment world =\n"
            text += f"  (NestedRun{stage}_{i} primitive external environment world).bind (fun result => evalProgram (M primitive external) next (result.1++environment) result.2) := by unfold NestedRun{stage}_{i}; rfl\n"
            executes.append(f"nestedExecute{stage}_{i}")
        for i in reversed(range(len(records))):
            row = records[i]
            if "ret" in row:
                term = program([row])
            elif "loop" in row:
                term = _loop_term(row["loop"], f"NestedBody{stage}_{i}", f"InlinePiece{stage}_{i + 1}")
            else:
                raw = program([row, {"ret": []}])
                a, sep, b = raw.rpartition("(.ret [])")
                assert sep
                term = a + f"InlinePiece{stage}_{i + 1}" + b
            text += f"@[irreducible] def InlinePiece{stage}_{i} : Program := {term}\n"
            pieces.append(f"InlinePiece{stage}_{i}")
        name = {2: "B", 3: "T", 4: "N"}[stage]
        names = [f"InlinePiece{stage}_{i}" for i in range(len(records))] + [
            f"NestedBody{stage}_{i}" for i, row in enumerate(records) if "loop" in row
        ]
        text += (
            f"theorem {name}_factored : {name} = InlinePiece{stage}_0 := by\n  unfold {name} "
            + " ".join(names)
            + "\n  rfl\n"
        )
    values = list(initial)
    calls = []
    for lane in range(trips):
        values, more = _simulate(
            loop["body"],
            [str(lower + lane * step), *values, *env],
            controls,
            len(calls),
            "world" if not calls else f"w{len(calls) - 1}",
        )
        calls += [{**row, "stage": 2} for row in more]
    _, more = _simulate(
        groups[4], [*values, *env], controls, len(calls), "world" if not calls else f"w{len(calls) - 1}"
    )
    calls += [{**row, "stage": 4} for row in more]
    _, target_calls = _simulate(after[pos:], env, controls)
    if len(calls) != len(target_calls):
        raise ValueError("actual inline nested event count differs")
    for source, target in zip(calls, target_calls, strict=True):
        if ("loop" in source) != ("loop" in target) or source["outputs"] != target["outputs"]:
            raise ValueError("inline event geometry differs")
        if "loop" not in source:
            continue
        i = source["index"]
        logical = list(dict.fromkeys(source["environment"] + target["environment"]))
        for label, call in (("L", source), ("R", target)):
            text += (
                f"def Nested{label}{i} : List Expr := "
                + lean_list([f"(.var {logical.index(value)})" for value in call["environment"]])
                + "\n"
            )
        text += f"theorem nestedAccepted{i} : Scope.check registry NestedL{i} NestedR{i} NestedProgram{source['stage']}_{source['position']} NestedProgram3_{target['position']} = true := by decide\n"
    text += (
        "theorem actual_inputs (world : World) : evalProgram (M primitive external) S0 "
        + lean_list(args)
        + " world =\n"
    )
    text += (
        "    evalProgram (M primitive external) S1 "
        + lean_list(args)
        + " world := by\n  rw [source_form,target_form]\n"
    )
    text += f"  have indices : indexIterations ({lower} : Int) {upper} {step} = (progression {lower} {step} {trips}).map Int.ofNat :=\n"
    text += f"    indexIterations_count (lower := {lower}) (upper := {upper}) (step := {step}) (count := {trips}) (by decide) (by decide)\n  rw [indices]\n"
    simp = "reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte"
    simp = ",".join(executes + rules + [simp])
    text += (
        "  simp only [B_factored,N_factored,T_factored,E,progression," + ",".join(pieces) + "," + simp + ",loopEval]\n"
    )
    for source, target in zip(calls, target_calls, strict=True):
        i = source["index"]
        pad = "  " + "    " * i
        text += pad + "all_goals\n"
        pad += "  "
        if "loop" in source:
            stage = source["stage"]
            logical = list(dict.fromkeys(source["environment"] + target["environment"]))
            text += _shape(stage, source, pad) + _shape(3, target, pad)
            text += f"{pad}have same{i} (w : World) : {_run(3, target['position'], target['environment'], 'w')} = {_run(stage, source['position'], source['environment'], 'w')} := by\n"
            text += f"{pad}  have checked := Scope.check_sound nestedAccepted{i} (M primitive external) {lean_list(logical)} w\n"
            text += f"{pad}  simp only [NestedL{i},NestedR{i},evalExpr,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] at checked\n"
            text += f"{pad}  rw [shape{stage}_{i},shape3_{i}] at checked\n{pad}  exact checked.symm\n"
            text += f"{pad}simp only [same{i}]\n"
            expression = _run(stage, source["position"], source["environment"], source["world"])
        else:
            expression = f"external.run {source['identity']} {lean_list(source['arguments'])} {source['world']}"
        text += f"{pad}cases step{i} : {expression} with\n{pad}| error failure =>\n{pad}  rcases failure with ⟨error,terminal⟩\n"
        text += f"{pad}  simp only [step{i},{simp},loopEval]\n{pad}| ok result =>\n{pad}  rcases result with ⟨values,w{i}⟩\n"
        text += f"{pad}  have size : values.length = {source['outputs']} := by\n"
        if "loop" in source:
            text += f"{pad}    have shaped := step{i}\n{pad}    unfold NestedRun{source['stage']}_{source['position']} at shaped\n{pad}    exact loopEval_arity _ _ _ _ _ shaped\n"
        else:
            text += f"{pad}    have localArity := external.arity {source['identity']} {lean_list(source['arguments'])} {source['world']} values w{i} step{i}\n"
            text += f"{pad}    simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity\n"
        text += unpack("values", "size", source["outputs"], f"v{i}_", pad + "  ")
        text += f"{pad}  simp only [step{i},{simp},loopEval]\n"
    return text
