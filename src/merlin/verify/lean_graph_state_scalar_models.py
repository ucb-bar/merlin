"""Concrete actual-roster Int models, distinct from primitive/runtime fidelity.

The kernel computes successful/failing complete execution; no source-success or
final-equality assumption supplies the witness. Private numeric interpretations
here do not discharge Merlin's numerical gates or external operation approval.
"""

from __future__ import annotations

from .lean_graph_state_scalar import emit_query, integer, program, types

MODEL = """
def wrap (width : Nat) (value : Int) : Int := value % (2^width : Int)
def primitive (op : Prim) (arguments : List Int) : Int :=
  let first := arguments[0]?.getD 0
  let second := arguments[1]?.getD 0
  match op with
  | .constant width value => wrap width value
  | .add width => wrap width (first+second)
  | .sub width => wrap width (first-second)
  | .mul width => wrap width (first*second)
  | .zext _ target | .trunc _ target => wrap target first
  | .sext source target =>
    let unsigned := wrap source first
    wrap target (if unsigned < (2^(source-1) : Int) then unsigned else unsigned-(2^source : Int))
def iterations (lower upper stride : Int) : List Int :=
  if 0 < stride then
    (List.range (((upper-lower+stride-1)/stride).toNat)).map (fun index : Nat => lower+stride*(index : Int))
  else []
"""


def evaluate(records, environment, registry, world, failure=None):
    """Expected DATA only: Lean independently checks the emitted execution."""
    environment, world = list(environment), list(world)

    def arithmetic(op, arguments):
        kind = op["kind"]
        if kind in ("constant", "add", "sub", "mul"):
            width = op["width"]
            if kind == "constant":
                value = op["value"]
            elif kind == "add":
                value = arguments[0] + arguments[1]
            elif kind == "sub":
                value = arguments[0] - arguments[1]
            else:
                value = arguments[0] * arguments[1]
        else:
            width, value = op["target"], arguments[0]
            if kind == "sext":
                value %= 1 << op["source"]
                if value >= 1 << (op["source"] - 1):
                    value -= 1 << op["source"]
        return value % (1 << width)

    for record in records:
        if "ret" in record:
            return "ok", [environment[index] for index in record["ret"]], world
        if "pure" in record:
            environment.insert(0, arithmetic(record["pure"], [environment[index] for index in record["arguments"]]))
        elif "effect" in record:
            identity = record["effect"]
            rule = registry[identity]
            operands = [environment[index] for index in record["arguments"]]
            if rule["index_constant"] is None:
                world += [identity, *operands]
            if failure == identity:
                return "error", 17, world
            value = 17 if rule["index_constant"] is None else rule["index_constant"]
            environment = [value] * len(rule["outputs"]) + environment
        else:
            loop = record["loop"]
            lower, upper, stride = [environment[index] for index in loop["bounds"]]
            carry = [environment[index] for index in loop["initial"]]
            for index in range(lower, upper, stride) if stride > 0 else []:
                if len(carry) != len(loop["carries"]):
                    return "arity", 0, world
                status, carry, world = evaluate(loop["body"], [index, *carry, *environment], registry, world, failure)
                if status != "ok":
                    return status, carry, world
            if len(carry) != len(loop["carries"]):
                return "arity", 0, world
            environment = [*carry, *environment]
    raise ValueError("missing actual complete model terminator")


def ints(values):
    return "[" + ",".join(integer(value) for value in values) + "]"


def model_definitions(registry, failure):
    cases = [
        f"| {index} => {integer(17 if row['index_constant'] is None else row['index_constant'])}"
        for index, row in enumerate(registry)
    ]
    observed = [index for index, row in enumerate(registry) if row["index_constant"] is None]
    selected_failure = "none" if failure is None else "some " + integer(failure, natural=True)
    return MODEL + (
        "def selectedValue (identity : Nat) : Int := match identity with\n  "
        + "\n  ".join(cases)
        + ("\n  " if cases else "")
        + "| _ => 17\n"
        "def observed : List Nat := " + str(observed) + "\n"
        "def failure : Option Nat := " + selected_failure + "\n"
        "def effect (identity : Nat) (arguments world : List Int) : Except (Int × List Int) (List Int × List Int) :=\n"
        "  let final := if identity ∈ observed then world ++ ((identity : Int)::arguments) else world\n"
        "  if failure = some identity then .error (17,final) else\n"
        "    .ok ((signature registry identity).outputs.map (fun _ => selectedValue identity),final)\n"
        "def concrete : Model Int (List Int) Int registry := {\n"
        "  default := 0, primitive := primitive, index := fun index => index,\n"
        "  iterations := iterations, effect := effect\n"
        "  arity := by\n"
        "    intro identity arguments world values final success\n"
        "    simp only [effect] at success\n"
        "    split at success\n"
        "    · cases success\n"
        "    · cases success; simp\n"
        "}\n"
    )


def emit_witness(projection, arguments, *, failure=None):
    if projection["functions"][0]["declaration"]:
        raise ValueError("concrete witness needs actual defined first function")
    function = projection["functions"][0]
    if len(arguments) != len(function["inputs"]):
        raise ValueError("explicit inhabited actual input arity required")
    for value, declared in zip(arguments, function["inputs"], strict=True):
        if set(declared) == {"bits"} and not 0 <= value < 1 << declared["bits"]:
            raise ValueError("model witness input must inhabit its declared unsigned bit domain")
        if set(declared) not in ({"bits"}, {"index"}):
            raise ValueError("this initialized Int witness does not model opaque input representations")
    if "chain_stages" in projection:
        from .lean_graph_state_scalar_chain import emit_chain_query

        text, roots = emit_chain_query(projection)
    else:
        text, roots = emit_query(projection)
    status, values, world = evaluate(function["before"], arguments, projection["registry"], [23], failure)
    if status == "arity":
        raise ValueError("well-typed actual witness unexpectedly has an internal arity fault")
    expected = (
        ".ok (" + ints(values) + "," + ints(world) + ")"
        if status == "ok"
        else ".error (.external 17," + ints(world) + ")"
    )
    text = text.removesuffix("end Merlin.GraphStateScalarInstance\n")
    text = text.replace(
        "import Merlin.GraphStateScalar\n", "import Merlin.GraphStateScalar\nimport Merlin.GraphStateScalarSteps\n", 1
    )
    text += "namespace Witness\n" + model_definitions(projection["registry"], failure)
    text += (
        "def arguments : List Int := " + ints(arguments) + "\n"
        "theorem input_arity : arguments.length = F0.inputs.length := by decide\n"
    )
    from .lean_graph_state_scalar_execution import emit_execution

    text += emit_execution(function, projection["registry"], arguments, failure)
    text += (
        "theorem actual_target_execution : evalProgram concrete F0.after arguments [23] = " + expected + " := by\n"
        "  rw [← F0.actual Int (List Int) Int concrete arguments [23] input_arity]\n"
        "  exact actual_original_execution\n"
    )
    if "chain_stages" in projection:
        text += (
            "theorem final_from_composition : evalProgram concrete F0.after arguments [23] = " + expected + " := by\n"
            "  rw [← F0.independently_composed Int (List Int) Int concrete arguments [23] input_arity]\n"
            "  exact actual_original_execution\n"
        )
        roots.append("Merlin.GraphStateScalarInstance.Witness.final_from_composition")
    text += "end Witness\nend Merlin.GraphStateScalarInstance\n"
    roots += [
        "Merlin.GraphStateScalarInstance.Witness." + name
        for name in ("input_arity", "actual_original_execution", "actual_target_execution")
    ]
    return (
        text,
        roots,
        {
            "status": status,
            "values": values,
            "world": world,
            "arguments": arguments,
            "source_success_fidelity_proved": False,
            "actual_primitive_fidelity_proved": False,
        },
    )


def emit_counterexample(projection, arguments, *, failure=None):
    """Typed actual source/altered-output operational countermodel, no acceptance."""
    function = projection["functions"][0]
    registry = (
        "["
        + ",".join(
            "{inputs := " + types(row["inputs"]) + ",outputs := " + types(row["outputs"]) + "}"
            for row in projection["registry"]
        )
        + "]"
    )
    text = (
        "import Merlin.GraphStateScalar\nnamespace Merlin.GraphStateScalarCounterexample\n"
        "open Merlin.GraphState.Scalar\n"
        "def registry : List Signature := " + registry + "\n"
        "def inputs : List Ty := " + types(function["inputs"]) + "\n"
        "def outputs : List Ty := " + types(function["outputs"]) + "\n"
        "def before : Program := " + program(function["before"]) + "\n"
        "def after : Program := "
        + program(function["after"])
        + "\n"
        + model_definitions(projection["registry"], failure)
        + "theorem source_typed : checkTypes registry inputs outputs before = true := by decide\n"
        "theorem target_typed : checkTypes registry inputs outputs after = true := by decide\n"
        "theorem refused : check registry inputs outputs before after = false := by decide\n"
        "def observe : Outcome Int (List Int) Int → Int × List Int × List Int\n"
        "  | .ok (values,world) => (0,values,world)\n"
        "  | .error (.external error,world) => (error+1,[],world)\n"
        "  | .error (.arity,world) => (-1,[],world)\n"
        "theorem observation_separated : observe (evalProgram concrete before " + ints(arguments) + " [23]) ≠\n"
        "    observe (evalProgram concrete after " + ints(arguments) + " [23]) := by decide\n"
        "theorem observed_difference : evalProgram concrete before " + ints(arguments) + " [23] ≠\n"
        "    evalProgram concrete after " + ints(arguments) + " [23] :=\n"
        "  fun equal => observation_separated (congrArg observe equal)\n"
        "end Merlin.GraphStateScalarCounterexample\n"
    )
    roots = [
        "Merlin.GraphStateScalarCounterexample." + name
        for name in ("source_typed", "target_typed", "refused", "observed_difference")
    ]
    return text, roots
