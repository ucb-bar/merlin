"""Kernel literals for the complete actual typed statement/epoch plan.

Finite command checks bind old instruction keys, typed operands, private/caller
views, allocation order, observer functions and initializer configurations. This
leaf does not itself assert or qualify whole-program semantic correspondence.
"""

from .lean_graph_loops_shape_epoch_observer import observer_literal
from .lean_graph_loops_shape_model import binding, bindings
from .lean_graph_loops_shape_model import function as function_literal
from .lean_graph_loops_unroll import _list, _nat, _string


def private_view(row):
    return (
        "⟨"
        + ", ".join((binding(row["binding"]), _nat(row["object"]), _list(row["shape"], _nat), _string(row["element"])))
        + "⟩"
    )


def access(row):
    return (
        "⟨"
        + ", ".join(
            (
                _nat(row["object"]),
                binding(row["buffer"]),
                bindings(row["indices"]),
                _list(row["shape"], _nat),
                _string(row["element"]),
            )
        )
        + "⟩"
    )


def opcode(row):
    tag = row["tag"]
    if tag in ("load", "store", "loop", "release"):
        return "." + tag
    if tag == "allocate":
        return ".allocate " + _list(row["shape"], _nat) + " " + _string(row["element"])
    if tag == "binary":
        return ".binary " + _string(row["element"])
    raise ValueError("unknown actual operation classifier")


def spec(row):
    tag = row["tag"]
    if tag == "constant":
        fields = (binding(row["result"]), _nat(row["value"]))
    elif tag == "binary":
        fields = (
            _string(row["operation"]),
            binding(row["left"]),
            binding(row["right"]),
            binding(row["result"]),
            _string(row["element"]),
        )
    elif tag in ("load", "store"):
        fields = (access(row["access"]), binding(row["result" if tag == "load" else "value"]))
    elif tag == "observer":
        fields = (
            _string(row["operation"]),
            binding(row["input"]),
            binding(row["buffer"]),
            binding(row["result"]),
            _string(row["element"]),
        )
    elif tag == "returned":
        fields = (bindings(row["values"]),)
    else:
        raise ValueError("unknown complete SSA specification")
    return "." + tag + " " + " ".join(fields)


def command(row, plan, observers):
    atom, instruction = row["atom"], row["instruction"]
    kind = instruction["kind"]
    key = instruction.get("key")
    selected = dict(plan["specifications"]).get(key)
    if kind == "retained":
        tag = selected["tag"]
        if tag == "constant":
            return ".constant " + " ".join((_string(key), binding(selected["result"]), _nat(selected["value"])))
        if tag == "binary":
            fields = (
                _string(key),
                _string(atom["key"]),
                binding(selected["left"]),
                binding(selected["right"]),
                binding(selected["result"]),
                _string(selected["element"]),
            )
        elif tag == "observer":
            program = observers[row["observer"]]
            fields = (
                _string(key),
                _string(atom["key"]),
                _string(row["observer"]),
                binding(selected["input"]),
                binding(selected["buffer"]),
                binding(selected["result"]),
                observer_literal(program),
            )
        else:
            raise ValueError("unsupported retained command")
        return "." + tag + " " + " ".join(fields)
    if kind in ("allocate", "release"):
        item = plan["privateViews"][instruction["object"]]
        return (
            "."
            + kind
            + " "
            + " ".join((_string(atom["key"]), private_view(item), "true" if instruction["retained"] else "false"))
        )
    if kind == "initialize":
        item = plan["privateViews"][instruction["object"]]
        constructor = ".initializeLine link" if len(item["shape"]) == 1 else ".initialize link"
        return constructor + str(row["initializer"])
    if kind in ("load", "store"):
        return (
            "."
            + kind
            + " "
            + " ".join(
                (
                    _string(key),
                    _string(atom["key"]),
                    access(selected["access"]),
                    binding(selected["result" if kind == "load" else "value"]),
                )
            )
        )
    raise ValueError("unknown complete epoch instruction")


def clauses(adapter):
    plan = adapter["step_binding"]
    function = next(row for row in adapter["normalized_candidate"]["functions"] if row["name"] == plan["function"])
    observers = {row["function"]: row for row in adapter["observers"]}
    sequence_index = next(
        index for index, row in enumerate(adapter["sequences"]) if row["function"] == plan["function"]
    )
    callers = [
        dict(binding=row, element=row["kind"]["type"]["element"])
        for row in function["parameters"]
        if row["kind"]["tag"] == "view"
    ]

    def table(rows, render):
        return _list(rows, lambda pair: "(" + _string(pair[0]) + ", " + render(pair[1]) + ")")

    mapping = _list(
        sorted((int(raw), source) for raw, source in adapter["candidate_to_source"].items()),
        lambda pair: "(" + _nat(pair[0]) + ", " + _nat(pair[1]) + ")",
    )
    literal = "{ objects := GraphMemorySCFObligation.actual.reuse.objects, "
    literal += (
        "configs := GraphLoopsMemorySCF.Model.configs GraphMemorySCFObligation.actual, module := actualNormalized, "
    )
    literal += (
        "opcodes := " + table(plan["opcodes"], opcode) + ", specifications := " + table(plan["specifications"], spec)
    )
    literal += ", mapping := " + mapping + ", views := " + _list(plan["privateViews"], private_view)
    literal += (
        ", callers := "
        + _list(callers, lambda row: "⟨" + binding(row["binding"]) + ", " + _string(row["element"]) + "⟩")
        + " }"
    )
    commands = _list(plan["steps"], lambda row: command(row, plan, observers))
    returned = dict(plan["specifications"])[plan["terminal"]["key"]]["values"]
    result = f"""
def plan : Epoch.PlanContext := {literal}
def commands : List Epoch.Command := {commands}
theorem commands_atoms : commands.map Epoch.Command.atom = sequence{sequence_index} := by rfl
theorem commands_instructions : commands.map Epoch.Command.instruction ++
    [GraphLoopsReuse.Instruction.retained {_string(plan["terminal"]["key"])}] =
    GraphMemorySCFObligation.actual.reuse.instructions := by rfl
theorem commands_checked :
    decide (Epoch.CommandsValid plan {bindings(function["parameters"])} 0 commands) = true := by decide
theorem plan_valid : decide plan.Valid = true := by decide
def kernel : Epoch.Kernel :=
    ⟨{function_literal(function)}, commands, {_string(plan["terminal"]["key"])}, {bindings(returned)}⟩
theorem kernel_facts : decide (kernel.Facts plan) = true := by decide
theorem kernel_instructions : kernel.instructions = GraphMemorySCFObligation.actual.reuse.instructions := by rfl
theorem complete_adapter {{Value User : Type}} (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap : GraphLoopsReuse.Heap Value) (user : User) (outputs : List (Runtime Value))
    (world : MemoryWorld Value (Epoch.AllocationPublic User)) :
    complete (Epoch.graphMeaning plan.opcodes scalar) plan.module false kernel.function.name inputs
      ⟨⟨0, user⟩, heap⟩ outputs world ↔
      ∃ after, GraphLoopsMemorySCF.Execution plan.objects plan.configs (plan.oldMeaning scalar) false false
        kernel.instructions (Epoch.entryMemory plan kernel.function inputs heap user) after ∧
        outputs = after.observed.returned ∧
        world = Epoch.clockWorld plan.objects.length (Epoch.finalClock 0 kernel.commands) after :=
    kernel.complete_equivalent plan (of_decide_eq_true plan_valid) (of_decide_eq_true kernel_facts)
      scalar inputs typed callers heap user outputs world
theorem memory_binding : decide (Epoch.MemoryBinding plan kernel GraphMemorySCFObligation.actual) = true := by decide
theorem complete_chain {{Value User : Type}} (scalar : Epoch.BinaryMeaning Value)
    (inputs : List (Runtime Value)) (typed : ValuesHold kernel.function.parameters inputs)
    (callers : Epoch.CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs))
    (heap candidateHeap : GraphLoopsReuse.Heap Value) (user : User) (observed : Epoch.Observation Value User) :
    Epoch.graphOutcomes plan.objects.length (Epoch.graphMeaning plan.opcodes scalar)
      GraphShapeObligation.actual.module kernel.function.name inputs heap user observed ↔
      Epoch.epochOutcomes plan kernel scalar true true
        ⟨(Epoch.entryMemory plan kernel.function inputs heap user).observed, candidateHeap⟩ observed :=
    kernel.checked_chain GraphShapeObligation.actual.module GraphShapeObligation.actual.bits
      GraphShapeObligation.valid.1 plan (of_decide_eq_true plan_valid) normalized_syntax
      (of_decide_eq_true kernel_facts) GraphMemorySCFObligation.actual
      (of_decide_eq_true memory_binding) GraphMemorySCFObligation.accepted scalar inputs typed callers
      heap candidateHeap user observed
"""
    roots = tuple(
        "GraphShapeEpochObligation." + name
        for name in (
            "commands_atoms",
            "commands_instructions",
            "commands_checked",
            "plan_valid",
            "kernel_facts",
            "kernel_instructions",
            "complete_adapter",
            "memory_binding",
            "complete_chain",
        )
    )
    return result, roots
