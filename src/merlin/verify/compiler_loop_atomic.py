# ruff: noqa: E501
"""Actual Program-to-Program atom correspondence under explicit capture facts."""

from merlin.verify.compiler_loop_nested import _loop_term, _model_rules, _run, _shape, _simulate
from merlin.verify.lean_graph_state_pipeline import lean_list
from merlin.verify.lean_graph_state_pipeline_emit import unpack
from merlin.verify.lean_graph_state_scalar import program


def emit(capture, left, right, *, arguments, substitutions, model_base, namespace):
    from .compiler_loop_resources import check_lanes

    for records in (left, right):
        check_lanes(capture, [{"ret": []}], repetitions=0, environment=arguments, carry=[], continuation=records)
    text = f"namespace {namespace}\nopen Merlin.GraphState.Scalar Merlin.GraphState.Pipeline\n"
    text += f"abbrev registry := {model_base}.registry\n"
    text += (
        "def M {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) : Model Int World Error registry := "
        + model_base
        + ".M primitive external\n"
    )
    text += (
        "variable {World Error : Type} (primitive : Prim → List Int → Int) (external : External World Error registry) "
        + ("(" + " ".join(arguments) + " : Int)" if arguments else "")
        + "\n"
    )
    rules_text, rules = _model_rules(capture)
    rules_text = rules_text.replace("M,indexModel", f"M,{model_base}.M,indexModel").replace(
        ",controls,", f",{model_base}.controls,"
    )
    text += rules_text
    controls = {row["identity"]: row for row in capture["index_operations"]}
    pieces = []
    executes = []
    for stage, records in ((2, left), (3, right)):
        text += f"def P{stage} : Program := {program(records)}\n"
        nested = []
        for i, row in enumerate(records):
            if "loop" not in row:
                continue
            nested.append(i)
            node = row["loop"]
            n = len(node["carries"])
            text += f"@[irreducible] def NestedBody{stage}_{i} : Program := {program(node['body'])}\n"
            text += f"def NestedProgram{stage}_{i} : Program := {_loop_term(node, program(node['body']), '(.ret ' + lean_list([str(j) for j in range(n)]) + ')')}\n"
            bounds = " ".join(f"(environment[{j}]?.getD 0)" for j in node["bounds"])
            initial = lean_list([str(j) for j in node["initial"]])
            text += f"@[irreducible] def NestedRun{stage}_{i} {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry) (environment : List Int) (world : World) : Outcome Int World Error :=\n"
            text += f"  loopEval {n} (fun iv values state => evalProgram (M primitive external) NestedBody{stage}_{i} (iv::values++environment) state) (indexIterations {bounds}) (reads 0 environment {initial}) world\n"
            text += f"theorem nestedExecute{stage}_{i} {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry) (next : Program) (environment : List Int) (world : World) :\n"
            text += f"  evalProgram (M primitive external) {_loop_term(node, f'NestedBody{stage}_{i}', 'next')} environment world =\n"
            text += f"  (NestedRun{stage}_{i} primitive external environment world).bind (fun result => evalProgram (M primitive external) next (result.1++environment) result.2) := by unfold NestedRun{stage}_{i}; rfl\n"
            executes.append(f"nestedExecute{stage}_{i}")
        for i in reversed(range(len(records))):
            row = records[i]
            if "ret" in row:
                term = program([row])
            elif "loop" in row:
                term = _loop_term(row["loop"], f"NestedBody{stage}_{i}", f"Piece{stage}_{i + 1}")
            else:
                raw = program([row, {"ret": []}])
                a, separator, b = raw.rpartition("(.ret [])")
                if not separator:
                    raise ValueError("actual atomic continuation missing")
                term = a + f"Piece{stage}_{i + 1}" + b
            text += f"@[irreducible] def Piece{stage}_{i} : Program := {term}\n"
            pieces.append(f"Piece{stage}_{i}")
        definitions = [f"Piece{stage}_{i}" for i in range(len(records))] + [f"NestedBody{stage}_{i}" for i in nested]
        text += (
            f"theorem P{stage}_factored : P{stage} = Piece{stage}_0 := by\n  unfold P{stage} "
            + " ".join(definitions)
            + "\n  rfl\n"
        )
    environment = [str(substitutions.get(i, arg)) for i, arg in enumerate(arguments)]
    _, calls = _simulate(left, environment, controls)
    _, targets = _simulate(right, environment, controls)
    if len(calls) != len(targets):
        raise ValueError("actual atomic event counts differ")
    for source, target in zip(calls, targets, strict=True):
        if ("loop" in source) != ("loop" in target) or source["outputs"] != target["outputs"]:
            raise ValueError("actual atomic result geometry differs")
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
        text += f"theorem nestedAccepted{i} : Scope.check registry NestedL{i} NestedR{i} NestedProgram2_{source['position']} NestedProgram3_{target['position']} = true := by decide\n"
    conditions = " ".join(
        f"(control{i} : {arguments[slot]} = {value})" for i, (slot, value) in enumerate(substitutions.items())
    )
    text += f"theorem actual_inputs (world : World) {conditions} :\n"
    text += f"  evalProgram (M primitive external) P2 {lean_list(arguments)} world = evalProgram (M primitive external) P3 {lean_list(arguments)} world := by\n"
    if substitutions:
        text += "  simp only [" + ",".join(f"control{i}" for i in range(len(substitutions))) + "]\n"
    simp = "reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte"
    simp = ",".join(executes + rules + [simp])
    text += "  simp only [P2_factored,P3_factored," + ",".join(pieces) + "," + simp + "]\n"
    for source, target in zip(calls, targets, strict=True):
        i = source["index"]
        pad = "  " + "    " * i
        text += pad + "all_goals\n"
        pad += "  "
        if "loop" in source:
            logical = list(dict.fromkeys(source["environment"] + target["environment"]))
            text += _shape(2, source, pad) + _shape(3, target, pad)
            text += f"{pad}have same{i} (w : World) : {_run(3, target['position'], target['environment'], 'w')} = {_run(2, source['position'], source['environment'], 'w')} := by\n"
            text += f"{pad}  have checked := Scope.check_sound nestedAccepted{i} (M primitive external) {lean_list(logical)} w\n"
            text += f"{pad}  simp only [NestedL{i},NestedR{i},evalExpr,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] at checked\n"
            text += (
                f"{pad}  rw [shape2_{i},shape3_{i}] at checked\n{pad}  exact checked.symm\n{pad}simp only [same{i}]\n"
            )
            expression = _run(2, source["position"], source["environment"], source["world"])
        else:
            expression = f"external.run {source['identity']} {lean_list(source['arguments'])} {source['world']}"
        text += f"{pad}cases step{i} : {expression} with\n{pad}| error failure =>\n{pad}  rcases failure with ⟨error,terminal⟩\n{pad}  simp only [step{i},{simp}]\n"
        text += f"{pad}| ok result =>\n{pad}  rcases result with ⟨values,w{i}⟩\n{pad}  have size : values.length = {source['outputs']} := by\n"
        if "loop" in source:
            text += f"{pad}    have shaped := step{i}\n{pad}    unfold NestedRun2_{source['position']} at shaped\n{pad}    exact loopEval_arity _ _ _ _ _ shaped\n"
        else:
            text += f"{pad}    have localArity := external.arity {source['identity']} {lean_list(source['arguments'])} {source['world']} values w{i} step{i}\n"
            text += f"{pad}    simpa only [registry,{model_base}.registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity\n"
        text += unpack("values", "size", source["outputs"], f"v{i}_", pad + "  ")
        text += f"{pad}  simp only [step{i},{simp}]\n"
    return text + f"end {namespace}\n"
