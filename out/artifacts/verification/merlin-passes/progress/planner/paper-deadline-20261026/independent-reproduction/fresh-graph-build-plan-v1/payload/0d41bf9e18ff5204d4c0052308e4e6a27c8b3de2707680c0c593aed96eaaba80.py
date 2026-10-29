# ruff: noqa: E501
# Keep literal kernel statement/proof equations readable and byte-bound.
"""Propose local rewrite proof terms for complete actual Scalar programs.

The Python normal forms are never proof authority: old SSA normalization,
every local identity, typed binder and whole-context relation is kernel checked.
"""

from __future__ import annotations

from .lean_graph_state_scalar import budget, integer, primitive, program, type_code, types


def var(index):
    return {"var": index}


def zero(width, value):
    return {"zero": {"kind": "constant", "width": width, "value": value}}


def shift(amount, expression):
    if "var" in expression:
        return var(amount + expression["var"])
    if "zero" in expression:
        return expression
    if "one" in expression:
        op, value = expression["one"]
        return {"one": [op, shift(amount, value)]}
    op, left, right = expression["two"]
    return {"two": [op, shift(amount, left), shift(amount, right)]}


def lift(count, symbols):
    return [var(i) for i in range(count)] + [shift(count, value) for value in symbols]


def normalize(records, symbols):
    symbols = list(symbols)
    for position, row in enumerate(records):
        if "ret" in row:
            if position != len(records) - 1:
                raise ValueError("unexpected early terminator")
            return {"ret": [symbols[i] for i in row["ret"]]}
        if "pure" in row:
            op = row["pure"]
            args = [symbols[i] for i in row["arguments"]]
            if len(args) > 2:
                raise ValueError("unsupported primitive arity")
            expression = {"zero": op} if not args else {"one" if len(args) == 1 else "two": [op, *args]}
            symbols.insert(0, expression)
        elif "effect" in row:
            count = len(row["outputs"])
            return {
                "effect": row["effect"],
                "arguments": [symbols[i] for i in row["arguments"]],
                "outputs": row["outputs"],
                "next": normalize(records[position + 1 :], lift(count, symbols)),
            }
        else:
            loop = row["loop"]
            return {
                "loop": {
                    "bounds": [symbols[i] for i in loop["bounds"]],
                    "initial": [symbols[i] for i in loop["initial"]],
                    "carries": loop["carries"],
                    "body": normalize(loop["body"], lift(len(loop["carries"]) + 1, symbols)),
                    "next": normalize(records[position + 1 :], lift(len(loop["carries"]), symbols)),
                }
            }
    raise ValueError("complete scalar code needs a final observation")


def expr(value):
    if "var" in value:
        return "(.var " + integer(value["var"], natural=True) + ")"
    if "zero" in value:
        return "(.zero " + primitive(value["zero"]) + ")"
    key = "one" if "one" in value else "two"
    op, *args = value[key]
    return "(." + key + " " + primitive(op) + " " + " ".join(expr(v) for v in args) + ")"


def exprs(values):
    return "[" + ",".join(expr(value) for value in values) + "]"


def code(value):
    if "ret" in value:
        return "(.ret " + exprs(value["ret"]) + ")"
    if "effect" in value:
        return (
            "(.opaque "
            + str(value["effect"])
            + " "
            + exprs(value["arguments"])
            + " "
            + types(value["outputs"])
            + " "
            + code(value["next"])
            + ")"
        )
    loop = value["loop"]
    return (
        "(.loop "
        + " ".join(expr(v) for v in loop["bounds"])
        + " "
        + exprs(loop["initial"])
        + " "
        + types(loop["carries"])
        + " "
        + code(loop["body"])
        + " "
        + code(loop["next"])
        + ")"
    )


def expr_type(value, environment):
    if "var" in value:
        return environment[value["var"]]
    op = value["zero"] if "zero" in value else value["one" if "one" in value else "two"][0]
    return {"bits": op.get("target", op.get("width"))}


def typed_expr(value, environment, expected):
    if expr_type(value, environment) != expected:
        raise ValueError("runtime representation proof has wrong static expression width/type")
    if "var" in value:
        return "(ExprTyped.var (type := " + type_code(expected) + ") (by decide))"
    if "zero" in value:
        return "(ExprTyped.zero (by decide) (by decide))"
    key = "one" if "one" in value else "two"
    op, *arguments = value[key]
    inputs = [{"bits": op["source"]}] if key == "one" else [{"bits": op["width"]}] * 2
    children = [
        typed_expr(argument, environment, declared) for argument, declared in zip(arguments, inputs, strict=True)
    ]
    binders = (
        " (type := " + type_code(inputs[0]) + ")"
        if key == "one"
        else " (first := " + type_code(inputs[0]) + ") (second := " + type_code(inputs[1]) + ")"
    )
    return "(ExprTyped." + key + binders + " (by decide) (by decide) " + " ".join(children) + ")"


def typed_exprs(values, environment, expected):
    if len(values) != len(expected):
        raise ValueError("wrong complete typed expression vector arity")
    result = "ExprsTyped.nil"
    for value, declared in reversed(list(zip(values, expected, strict=True))):
        result = "(ExprsTyped.cons " + typed_expr(value, environment, declared) + " " + result + ")"
    return result


def const(value):
    if "zero" in value and value["zero"]["kind"] == "constant":
        return value["zero"]
    return None


def simplify(value, environment):
    if "var" in value:
        return value, "(Same.refl _ _)"
    if "zero" in value:
        op = value["zero"]
        folded = op["value"] % (1 << op["width"])
        return zero(op["width"], folded), (
            f"(Same.constant _ {op['width']} {integer(op['value'])} {integer(folded)} (by decide))"
        )
    key = "one" if "one" in value else "two"
    op, *children = value[key]
    reduced = [simplify(child, environment) for child in children]
    current = {key: [op, *[r[0] for r in reduced]]}
    congruence = "(Same." + key + " " + primitive(op) + " " + " ".join(r[1] for r in reduced) + ")"
    if key == "one":
        return current, congruence
    left, right = [r[0] for r in reduced]
    lhs, rhs = const(left), const(right)
    width, kind = op["width"], op["kind"]
    if kind not in ("add", "sub", "mul"):
        raise ValueError("unsupported exact integer rewrite primitive")
    if lhs and rhs and lhs["width"] == rhs["width"] == width:
        a, b = lhs["value"], rhs["value"]
        result = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}[kind]() % (1 << width)
        reduced_value = zero(width, result)
        proof = f"(Same.fold_{kind} _ {width} {integer(a)} {integer(b)} {integer(result)} (by decide))"
    elif kind == "add" and lhs and lhs["width"] == width and lhs["value"] == 0:
        reduced_value = right
        proof = f"(Same.zero_add {width} " + typed_expr(right, environment, {"bits": width}) + ")"
    elif kind in ("add", "sub") and rhs and rhs["width"] == width and rhs["value"] == 0:
        reduced_value = left
        proof = f"(Same.{kind}_zero {width} " + typed_expr(left, environment, {"bits": width}) + ")"
    elif kind == "mul" and lhs and lhs["width"] == width and lhs["value"] == 1:
        reduced_value = right
        proof = f"(Same.one_mul {width} " + typed_expr(right, environment, {"bits": width}) + ")"
    elif kind == "mul" and rhs and rhs["width"] == width and rhs["value"] == 1:
        reduced_value = left
        proof = f"(Same.mul_one {width} " + typed_expr(left, environment, {"bits": width}) + ")"
    else:
        return current, congruence
    return reduced_value, "(Same.trans " + congruence + " " + proof + ")"


def same_expr(first, second, environment):
    left, proof = simplify(first, environment)
    right, other = simplify(second, environment)
    if left != right:
        raise ValueError("actual ordered expression lacks a supported exact local rewrite proof")
    return "(Same.trans " + proof + " (Same.symm " + other + "))"


def same_exprs(first, second, environment):
    if len(first) != len(second):
        raise ValueError("complete observation/operand vector arity differs")
    result = "(Sames.nil _)"
    for left, right in reversed(list(zip(first, second, strict=True))):
        result = "(Sames.cons " + same_expr(left, right, environment) + " " + result + ")"
    return result


def relation(first, second, environment, results, registry):
    if set(first) != set(second):
        raise ValueError("complete retained call/loop/observation skeleton differs")
    if "ret" in first:
        return (
            "(Relation.ret "
            + typed_exprs(first["ret"], environment, results)
            + " "
            + same_exprs(first["ret"], second["ret"], environment)
            + ")"
        )
    if "effect" in first:
        if first["effect"] != second["effect"] or first["outputs"] != second["outputs"]:
            raise ValueError("retained effect descriptor/result order differs")
        identity = first["effect"]
        selected = registry[identity]
        args = typed_exprs(first["arguments"], environment, selected["inputs"])
        same = same_exprs(first["arguments"], second["arguments"], environment)
        rest = relation(first["next"], second["next"], first["outputs"] + environment, results, registry)
        return f"(Relation.effect (by decide) (by decide) {args} {same} {rest})"
    left, right = first["loop"], second["loop"]
    if left["carries"] != right["carries"]:
        raise ValueError("ordered loop carried types differ")
    carries = left["carries"]
    typed = [typed_expr(value, environment, {"index": True}) for value in left["bounds"]]
    initial = typed_exprs(left["initial"], environment, carries)
    bounds = [same_expr(a, b, environment) for a, b in zip(left["bounds"], right["bounds"], strict=True)]
    same = same_exprs(left["initial"], right["initial"], environment)
    body = relation(left["body"], right["body"], [{"index": True}, *carries, *environment], carries, registry)
    rest = relation(left["next"], right["next"], [*carries, *environment], results, registry)
    return "(Relation.loop " + " ".join([*typed, initial, *bounds, same, body, rest]) + ")"


def emit(projection, *, pipeline_capture=None):
    suffix = ""
    if pipeline_capture is not None:
        from .lean_graph_state_pipeline_emit import emit as emit_pipeline

        if (
            projection["registry"] != pipeline_capture["stages"][0]["registry"]
            or projection["types"] != pipeline_capture["stages"][0]["types"]
        ):
            raise ValueError("exact prefix/suffix registry and type join required")
        suffix = emit_pipeline(pipeline_capture).replace(
            "Merlin.GraphStatePipelineInstance", "Merlin.GraphStateIntegerInstance.Suffix"
        )
        suffix = "import Merlin.GraphStateIntegerProgram\nimport Merlin.GraphStateIntegerTrace\n" + suffix
    registry = (
        "["
        + ",".join(
            "{inputs := " + types(row["inputs"]) + ",outputs := " + types(row["outputs"]) + "}"
            for row in projection["registry"]
        )
        + "]"
    )
    text = suffix or "import Merlin.GraphStateIntegerProgram\nimport Merlin.GraphStateIntegerTrace\n"
    text += (
        "set_option maxRecDepth 4096\nset_option maxHeartbeats 2000000\n"
        "namespace Merlin.GraphStateIntegerInstance\n"
        "open Merlin.GraphState.Scalar Merlin.GraphState.Integer\n"
        + (
            "abbrev registry := Suffix.registry\ntheorem originalRegistry : registry = " + registry + " := by decide\n"
            if suffix
            else "def registry : List Signature := " + registry + "\n"
        )
    )
    defined = []
    for index, function in enumerate(projection["functions"]):
        if function["declaration"]:
            continue
        defined.append(index)
        for key in ("before", "after"):
            budget(function[key], len(function["inputs"]), [0])
        symbols = [var(i) for i in range(len(function["inputs"]))]
        first, second = [normalize(function[key], symbols) for key in ("before", "after")]
        proof = relation(first, second, function["inputs"], function["outputs"], projection["registry"])
        text += f"namespace F{index}\n"
        text += "def inputs : List Ty := " + types(function["inputs"]) + "\n"
        text += "def outputs : List Ty := " + types(function["outputs"]) + "\n"
        for key in ("before", "after"):
            text += f"def {key} : Program := " + program(function[key]) + "\n"
            text += f"theorem {key}Typed : checkTypes registry inputs outputs {key} = true := by decide\n"
        for key, value, raw in (("first", first, "before"), ("second", second, "after")):
            text += f"def {key} : Code := " + code(value) + "\n"
            text += (
                f"theorem {key}Normalized : normalize registry (vars inputs.length) {raw} = some {key} := by decide\n"
            )
        text += "theorem localRewrites : Relation registry inputs outputs first second :=\n  " + proof + "\n"
        text += "theorem actual {World Error : Type} (model : Model Int World Error registry)\n"
        text += "    (exact : ExactModel model) (laws : LocalTyping model Canonical)\n"
        text += "    (values : List Int) (world : World) (entry : Values Canonical inputs values) :\n"
        text += "    evalProgram model before values world = evalProgram model after values world :=\n"
        text += (
            "  (programs_sound model exact laws firstNormalized secondNormalized localRewrites values world entry).2\n"
        )
        text += "theorem reached {World Error : Type} (model : Model Int World Error registry)\n"
        text += "    (laws : LocalTyping model Canonical) (values : List Int) (world : World)\n"
        text += "    (entry : Values Canonical inputs values) :\n"
        text += "    Reached model Canonical before values world ∧ Reached model Canonical after values world :=\n"
        text += "  ⟨typed_reached model laws (checkTypes_sound _ _ _ beforeTyped) values world entry,\n"
        text += "   typed_reached model laws (checkTypes_sound _ _ _ afterTyped) values world entry⟩\n"
        text += f"end F{index}\n"
    if not defined:
        raise ValueError("no defined entry; declarations alone cannot demonstrate a rewrite")
    for name, stage in (("originalEntries", "before"), ("middleEntries", "after")):
        text += (
            "def "
            + name
            + " : List Merlin.GraphState.Pipeline.Entry := ["
            + ",".join(f"{{inputs := F{i}.inputs, outputs := F{i}.outputs, body := F{i}.{stage}}}" for i in defined)
            + "]\n"
        )
    proof = "(Merlin.GraphState.Integer.entries_nil model)"
    for index in reversed(defined):
        proof = f"(Merlin.GraphState.Integer.entries_cons model rfl rfl (F{index}.actual model exact laws) {proof})"
    text += "theorem prefix_actual {World Error : Type} (model : Model Int World Error registry)\n"
    text += "    (exact : ExactModel model) (laws : LocalTyping model Canonical) :\n"
    text += "    PreservesEntriesCanonical model originalEntries middleEntries :=\n  " + proof + "\n"
    if suffix:
        text += "theorem exactMiddle : middleEntries = Suffix.originalEntries := by decide\n"
        text += "theorem complete {World Error : Type} (external : Merlin.GraphState.Pipeline.External World Error registry)\n"
        text += "    (typed : ∀ identity arguments world values final,\n"
        text += "      Values Canonical (signature registry identity).inputs arguments →\n"
        text += "      external.run identity arguments world = .ok (values,final) →\n"
        text += "      Values Canonical (signature registry identity).outputs values) :\n"
        text += (
            "    PreservesEntriesCanonical (Suffix.M primitive external) originalEntries Suffix.finalEntries := by\n"
        )
        text += "  apply compose_suffix (Suffix.M primitive external) (prefix_actual _ ⟨rfl,rfl⟩\n"
        text += "    (index_model_typing (Merlin.GraphState.Pipeline.checkControl_sound Suffix.controlAccepted) external typed))\n"
        text += "  rw [exactMiddle]\n  exact Suffix.complete primitive external\n"
    return text + "end Merlin.GraphStateIntegerInstance\n"
