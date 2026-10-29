# ruff: noqa: E501 -- generated Lean equations retain their literal layout
"""Proof emission for ordered lanes retaining arbitrary nested scalar loops.

Nested loops are atomic only in the local proof script. Each atomic equation is
derived from the existing kernel-checked lexical-scope checker over the complete
actual nested programs; no loop or effect is erased from the semantic query.
"""

from .lean_graph_state_pipeline import lean_list
from .lean_graph_state_pipeline_emit import unpack
from .lean_graph_state_scalar import primitive, program, types


def _simulate(records, environment, controls, offset=0, world="world"):
    env, calls = list(environment), []
    for position, row in enumerate(records):
        if "ret" in row:
            return [env[i] for i in row["ret"]], calls
        if "loop" in row:
            outputs = len(row["loop"]["carries"])
            call = {"loop": row["loop"], "environment": list(env), "position": position}
        else:
            args = [env[i] for i in row["arguments"]]
            if "pure" in row:
                env.insert(0, "(primitive " + primitive(row["pure"]) + " " + lean_list(args) + ")")
                continue
            if row["effect"] in controls:
                op = controls[row["effect"]]
                if op["kind"] == "index-constant":
                    value = str(op["value"])
                elif op["kind"] == "index-add":
                    value = f"({args[0]}+{args[1]})"
                elif op["kind"] == "index-minui":
                    value = f"(min {args[0]} {args[1]})"
                else:
                    raise ValueError("unsupported exact nested control")
                env.insert(0, value)
                continue
            outputs = len(row["outputs"])
            call = {"identity": row["effect"], "arguments": args, "position": position}
        index = offset + len(calls)
        calls.append({**call, "index": index, "outputs": outputs, "world": world})
        env = [f"v{index}_{i}" for i in range(outputs)] + env
        world = f"w{index}"
    raise ValueError("complete nested body terminator required")


def _loop_term(loop, body, next):
    return (
        "(.loop "
        + " ".join(str(i) for i in loop["bounds"])
        + " "
        + lean_list([str(i) for i in loop["initial"]])
        + " "
        + types(loop["carries"])
        + " "
        + body
        + " "
        + next
        + ")"
    )


def _run(stage, position, env, world):
    return f"(NestedRun{stage}_{position} primitive external {lean_list(env)} {world})"


def _model_rules(capture):
    text = "theorem nestedRet (values : List Nat) (environment : List Int) (world : World) : evalProgram (M primitive external) (.ret values) environment world = .ok (reads 0 environment values,world) := rfl\n"
    text += "theorem nestedPure (op : Prim) (arguments : List Nat) (next : Program) (environment : List Int) (world : World) : evalProgram (M primitive external) (.pure op arguments next) environment world = evalProgram (M primitive external) next (primitive op (reads 0 environment arguments)::environment) world := rfl\n"
    rules = ["nestedRet", "nestedPure"]
    controls = {r["identity"]: r for r in capture["index_operations"]}
    for identity in range(len(capture["stages"][0]["registry"])):
        name = f"nestedEffect{identity}"
        rules.append(name)
        control = controls.get(identity)
        if control is not None:
            if control["kind"] == "index-constant":
                binders, operands, value = "", "[]", str(control["value"])
            else:
                binders, operands = "(left right : Nat)", "[left,right]"
                a, b = "(environment[left]?.getD 0)", "(environment[right]?.getD 0)"
                value = f"{a}+{b}" if control["kind"] == "index-add" else f"min {a} {b}"
            text += f"theorem {name} {binders} (next : Program) (environment : List Int) (world : World) : evalProgram (M primitive external) (.opaque {identity} {operands} [.index] next) environment world = evalProgram (M primitive external) next (({value})::environment) world := rfl\n"
        else:
            text += f"theorem {name} (arguments : List Nat) (outputs : List Ty) (next : Program) (environment : List Int) (world : World) : evalProgram (M primitive external) (.opaque {identity} arguments outputs next) environment world = (match external.run {identity} (reads 0 environment arguments) world with | .error (error,terminal) => .error (.external error,terminal) | .ok result => evalProgram (M primitive external) next (result.1++environment) result.2) := by\n"
            text += f"  cases executed : external.run {identity} (reads 0 environment arguments) world with\n  | error failure =>\n    rcases failure with ⟨error,terminal⟩\n    simp only [evalProgram,opaqueStep,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,executed,Except.bind]\n  | ok result =>\n    simp only [evalProgram,opaqueStep,M,indexModel,indexEffect,controls,Merlin.GraphState.Pipeline.controlAt,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,executed,Except.bind]\n"
    return text, rules


def _shape(stage, call, pad):
    position, count = call["position"], call["outputs"]
    env = lean_list(call["environment"])
    run = _run(stage, position, call["environment"], "w")
    text = f"{pad}have shape{stage}_{call['index']} (w : World) :\n"
    text += f"{pad}    evalProgram (M primitive external) NestedProgram{stage}_{position} {env} w = {run} := by\n"
    text += f"{pad}  unfold NestedProgram{stage}_{position} NestedRun{stage}_{position} NestedBody{stage}_{position}\n"
    loop = call["loop"]
    bounds = " ".join(f"({env}[{i}]?.getD 0)" for i in loop["bounds"])
    body = (
        f"(fun iv values state => evalProgram (M primitive external) {program(loop['body'])} (iv::values++{env}) state)"
    )
    initial = f"(reads 0 {env} {lean_list([str(i) for i in loop['initial']])})"
    raw = f"loopEval {count} {body} (indexIterations {bounds}) {initial} w"
    text += f"{pad}  change ({raw}).bind (fun result => .ok (reads 0 (result.1++{env}) {lean_list([str(i) for i in range(count)])}, result.2)) = ({raw})\n"
    text += f"{pad}  cases executed : {raw} with\n{pad}  | error failure => rfl\n"
    text += f"{pad}  | ok result =>\n{pad}    have size := loopEval_arity {count} {body} (indexIterations {bounds}) {initial} w executed\n"
    text += f"{pad}    rcases result with ⟨values,final⟩\n"
    text += unpack("values", "size", count, f"nested{stage}_{call['index']}_", pad + "    ")
    return text + f"{pad}    rfl\n"


def body_group_text(capture, loops, envs, count, factor, step, args):
    controls = {r["identity"]: r for r in capture["index_operations"]}
    text, rules = _model_rules(capture)
    groups = {}
    for stage in (2, 3):
        records = loops[stage]["body"]
        program(records)
        pieces, pending, nested = [], [], []
        for i, row in enumerate(records):
            pending.append(f"Piece{stage}_{i}")
            if "loop" in row:
                loop = row["loop"]
                nested.append(i)
                text += f"@[irreducible] def NestedBody{stage}_{i} : Program := {program(loop['body'])}\n"
                text += f"def NestedProgram{stage}_{i} : Program := {_loop_term(loop, program(loop['body']), '(.ret ' + lean_list([str(j) for j in range(len(loop['carries']))]) + ')')}\n"
                bounds = " ".join(f"(environment[{j}]?.getD 0)" for j in loop["bounds"])
                initial = lean_list([str(j) for j in loop["initial"]])
                n = len(loop["carries"])
                text += f"@[irreducible] def NestedRun{stage}_{i} {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry) (environment : List Int) (world : World) : Outcome Int World Error :=\n"
                text += f"  loopEval {n} (fun iv values state => evalProgram (M primitive external) NestedBody{stage}_{i} (iv::values++environment) state) (indexIterations {bounds}) (reads 0 environment {initial}) world\n"
                text += f"theorem nestedExecute{stage}_{i} {{World Error : Type}} (primitive : Prim → List Int → Int) (external : External World Error registry) (next : Program) (environment : List Int) (world : World) :\n"
                text += f"  evalProgram (M primitive external) {_loop_term(loop, f'NestedBody{stage}_{i}', 'next')} environment world =\n"
                text += f"  (NestedRun{stage}_{i} primitive external environment world).bind (fun result => evalProgram (M primitive external) next (result.1++environment) result.2) := by unfold NestedRun{stage}_{i}; rfl\n"
            atomic = "loop" in row or ("effect" in row and row["effect"] not in controls)
            if atomic or "ret" in row:
                pieces.append(pending)
                pending = []
        if pending or not pieces:
            raise ValueError("complete nested body partition required")
        groups[stage] = pieces
        for i in reversed(range(len(records))):
            row = records[i]
            if "ret" in row:
                term = program([row])
            elif "loop" in row:
                term = _loop_term(row["loop"], f"NestedBody{stage}_{i}", f"Piece{stage}_{i + 1}")
            else:
                raw = program([row, {"ret": []}])
                head, separator, tail = raw.rpartition("(.ret [])")
                if not separator:
                    raise ValueError("nested statement continuation missing")
                term = head + f"Piece{stage}_{i + 1}" + tail
            text += f"@[irreducible] def Piece{stage}_{i} : Program := {term}\n"
        names = [f"Piece{stage}_{i}" for i in range(len(records))] + [f"NestedBody{stage}_{i}" for i in nested]
        text += (
            f"theorem B{stage}_factored : B{stage} = Piece{stage}_0 := by\n  unfold B{stage} "
            + " ".join(names)
            + "\n  rfl\n"
        )
    source_calls = len(groups[2]) - 1
    if len(groups[3]) - 1 != factor * source_calls:
        raise ValueError("actual target nested/effect count differs from ordered lanes")
    values, calls = [f"c{i}" for i in range(count)], []
    for lane in range(factor):
        iv = "base" if lane == 0 else f"(base+{step * lane})"
        values, more = _simulate(
            loops[2]["body"],
            [iv, *values, *envs[2]],
            controls,
            len(calls),
            "world" if not calls else f"w{len(calls) - 1}",
        )
        calls += more
    _, target_calls = _simulate(loops[3]["body"], ["base", *[f"c{i}" for i in range(count)], *envs[3]], controls)
    for source, target in zip(calls, target_calls, strict=True):
        if ("loop" in source) != ("loop" in target) or source["outputs"] != target["outputs"]:
            raise ValueError("actual nested event shape differs from source lanes")
        if "loop" not in source:
            continue
        i = source["index"]
        logical = list(dict.fromkeys(source["environment"] + target["environment"]))
        for label, call in (("L", source), ("R", target)):
            text += (
                f"def Nested{label}{i} : List Expr := "
                + lean_list([f"(.var {logical.index(v)})" for v in call["environment"]])
                + "\n"
            )
        text += f"theorem nestedAccepted{i} : Scope.check registry NestedL{i} NestedR{i} NestedProgram2_{source['position']} NestedProgram3_{target['position']} = true := by decide\n"
    text += f"theorem body_group (base : Int) (carry : List Int) (world : World) (arity : carry.length = {count}) :\n"
    text += f"  evalProgram (M primitive external) B3 (base::carry++E3 primitive {args}) world =\n"
    text += f"  loopEval {count} (fun iv values state => evalProgram (M primitive external) B2 (iv::values++E2 primitive {args}) state) {lean_list(['base' if lane == 0 else f'(base+{step * lane})' for lane in range(factor)])} carry world := by\n"
    text += unpack("carry", "arity", count, "c")
    simp = "reads,List.map_cons,List.map_nil,List.cons_append,List.nil_append,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,Option.getD_none,Except.bind,List.length_cons,List.length_nil,↓reduceIte"
    executes = [
        f"nestedExecute{stage}_{i}" for stage in (2, 3) for i, row in enumerate(loops[stage]["body"]) if "loop" in row
    ]
    simp = ",".join(executes + rules + [simp])
    initial = ",".join(groups[2][0] + groups[3][0])
    text += f"  simp only [B2_factored,B3_factored,{initial},E2,E3,{simp},loopEval]\n"
    for source, target in zip(calls, target_calls, strict=True):
        i = source["index"]
        pad = "  " + "  " * i
        if "loop" in source:
            logical = list(dict.fromkeys(source["environment"] + target["environment"]))
            text += _shape(2, source, pad) + _shape(3, target, pad)
            text += f"{pad}have same{i} (w : World) : {_run(3, target['position'], target['environment'], 'w')} = {_run(2, source['position'], source['environment'], 'w')} := by\n"
            text += f"{pad}  have checked := Scope.check_sound nestedAccepted{i} (M primitive external) {lean_list(logical)} w\n"
            text += f"{pad}  simp only [NestedL{i},NestedR{i},evalExpr,List.map_cons,List.map_nil,List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some] at checked\n"
            text += f"{pad}  rw [shape2_{i},shape3_{i}] at checked\n{pad}  exact checked.symm\n"
            text += f"{pad}simp only [same{i}]\n"
            expression = _run(2, source["position"], source["environment"], source["world"])
        else:
            expression = f"external.run {source['identity']} {lean_list(source['arguments'])} {source['world']}"
        next_defs = groups[3][i + 1] + groups[2][(i + 1) % source_calls]
        if (i + 1) % source_calls == 0:
            next_defs += groups[2][-1]
        definitions = ",".join(next_defs)
        text += f"{pad}cases step{i} : {expression} with\n{pad}| error failure =>\n{pad}  rcases failure with ⟨error,terminal⟩\n"
        text += f"{pad}  simp only [step{i},{definitions},{simp},loopEval]\n{pad}| ok result =>\n{pad}  rcases result with ⟨values,w{i}⟩\n"
        if "loop" in source:
            text += f"{pad}  have size : values.length = {source['outputs']} := by\n"
            text += f"{pad}    have shaped := step{i}\n{pad}    unfold NestedRun2_{source['position']} at shaped\n{pad}    exact loopEval_arity _ _ _ _ _ shaped\n"
        else:
            text += f"{pad}  have size : values.length = {source['outputs']} := by\n"
            text += f"{pad}    have localArity := external.arity {source['identity']} {lean_list(source['arguments'])} {source['world']} values w{i} step{i}\n"
            text += f"{pad}    simpa only [registry,signature,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some,List.length_cons,List.length_nil] using localArity\n"
        text += unpack("values", "size", source["outputs"], f"v{i}_", pad + "  ")
        text += f"{pad}  simp only [step{i},{definitions},{simp},loopEval]\n"
    return text
