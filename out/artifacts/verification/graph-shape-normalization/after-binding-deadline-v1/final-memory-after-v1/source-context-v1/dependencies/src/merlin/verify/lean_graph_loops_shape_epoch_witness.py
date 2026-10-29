"""Actual-source-bound inhabited adapter witnesses under one concrete meaning.

Uniform initialization is proved symbolically in the cell count. Retained
operations read the actual SSA operands and update actual caller observations;
they are never replaced by identity summaries. Nat addition here is an explicit
inhabited interpretation, not a proof of native fixed-width arithmetic fidelity.
"""

from __future__ import annotations

from .lean_graph_loops_shape_epoch_model import obligation
from .lean_graph_loops_shape_epoch_plan import access, spec
from .lean_graph_loops_shape_model import binding, bindings
from .lean_graph_loops_unroll import _string

WITNESS_ROOTS = (
    "GraphShapeEpochWitness.epoch_source_inhabited",
    "GraphShapeEpochWitness.typed_source_inhabited",
    "GraphShapeEpochWitness.segmented_inhabited",
)


def witness_obligation(adapter):
    plan = adapter["step_binding"]
    function = next(row for row in adapter["normalized_candidate"]["functions"] if row["name"] == plan["function"])
    parameters = function["parameters"]
    if [row["kind"]["tag"] for row in parameters] != ["scalar", "scalar", "view"]:
        raise ValueError("finite witness requires two scalar seeds and one canonical caller view")
    if len(adapter["initializer_links"]) != 2 or len(plan["privateViews"]) != 2:
        raise ValueError("this inhabited witness covers two logical epochs; the theorem is parameterized")
    element = parameters[0]["kind"]["type"]
    if parameters[1]["kind"]["type"] != element:
        raise ValueError("witness scalar types differ")
    body, roots = obligation(adapter)
    registers = {parameters[0]["name"]: 17, parameters[1]["name"]: 29}
    caller_object, caller_value = len(plan["privateViews"]), 91
    cells, cell_proofs = {}, {}
    selected = dict(plan["specifications"])
    definitions, lines, steps = [], [], []
    instructions = adapter["memory"]["reuse"]["instructions"]
    holds = "by exact ⟨rfl, by intro index impossible; cases impossible⟩"
    for index, instruction in enumerate(instructions):
        prior, state = ("source" if index == 0 else f"state{index - 1}"), f"state{index}"
        kind, obj = instruction["kind"], instruction.get("object")
        key = instruction.get("key")
        if key is not None:
            definitions.append(f"def key{index} : String := {_string(key)}")
        arguments = f"plan.objects.length plan.specifications scalar plan.mapping key{index}"
        if kind == "retained":
            row = selected[key]
            definitions += [
                f"def spec{index} : Spec := {spec(row)}",
                f"theorem selected{index} : specAt plan.specifications key{index} = some spec{index} := by decide",
            ]
            tag = row["tag"]
            if tag == "returned":
                values = row["values"]
                lines.append(f"  let {state} := returnMemory {bindings(values)} {prior}")
                proof = (
                    f"(returned_step_iff {arguments} {bindings(values)} selected{index} "
                    f"plan.objects plan.configs {prior} {state}).mpr rfl"
                )
            else:
                result = row["result"]
                if tag == "constant":
                    value = row["value"]
                    runtime = f".index {value}"
                    retained_proof = "rfl"
                elif tag == "binary":
                    left, right = registers[row["left"]["name"]], registers[row["right"]["name"]]
                    value = left + right
                    runtime = f".scalar {_string(element)} {value}"
                    retained_proof = f"⟨{left}, {right}, {value}, rfl, rfl, rfl, {holds}, rfl⟩"
                elif tag == "observer":
                    operand, old = registers[row["input"]["name"]], caller_value
                    value = caller_value = old + operand
                    runtime = f".scalar {_string(element)} {value}"
                    retained_proof = (
                        f"⟨{caller_object}, {operand}, {old}, {value}, by decide, rfl, rfl, rfl, rfl, {holds}, rfl⟩"
                    )
                else:
                    raise ValueError("unsupported retained witness operation")
                registers[result["name"]] = value
                observed = f"writeRegister {prior}.observed {result['name']} ({runtime})"
                if tag == "observer":
                    observed = (
                        "{ "
                        + observed
                        + f" with callerHeap := GraphLoopsReuse.replaceCell {prior}.observed.callerHeap "
                        + f"{caller_object} 0 {value} "
                        + "}"
                    )
                lines.append(f"  let {state} : State := ⟨{observed}, {prior}.heap⟩")
                proof = (
                    f"(retained_step_iff {arguments} spec{index} selected{index} "
                    f"plan.objects plan.configs {prior} {state}).mpr ⟨_, {retained_proof}, rfl⟩"
                )
            instruction_literal = f".retained key{index}"
        elif kind in ("allocate", "release"):
            keep = str(instruction["retained"]).lower()
            lines.append(f"  let {state} := GraphLoopsReuse.clearMemory plan.objects false {obj} {keep} {prior}")
            instruction_literal = f".{kind} {obj} {keep}"
            proof = f"({kind}_step_iff plan.objects plan.configs meaning {obj} {keep} {prior} {state}).mpr rfl"
        elif kind == "initialize":
            link = next(row for row in adapter["initializer_links"] if row["config"]["object"] == obj)
            cfg = link["config"]
            rows, columns = cfg["program"]["shape"]
            outer, inner = cfg["program"]["ivs"]
            seed = cfg["program"]["leaf"]["value"]
            canonical_seed = link["canonical"]["seed"]["name"]
            value = registers[canonical_seed]
            lines.append(
                f"  obtain ⟨heap{index}, initializedRun{index}, initialized{index}⟩ := "
                "GraphLoopsMemorySCF.Witness.uniform_heap "
                f"meaning.pure {rows} {columns} {outer} {inner} {seed} {cfg['tile']} {obj} "
                f"(by decide) (by decide) (by decide) (by decide) (by decide) "
                f"(meaning.captures {prior}.observed) {value} (by rfl) {prior}.heap"
            )
            lines.append(f"  let {state} : State := ⟨{prior}.observed, heap{index}⟩")
            config_index = next(i for i, row in enumerate(adapter["memory"]["configurations"]) if row == cfg)
            cfg_expr = f"GraphMemorySCFObligation.actual.boundaries[{config_index}].config"
            proof = f".initialized (config := {cfg_expr}) (by decide) {prior} initializedRun{index}"
            instruction_literal = f".initialize {obj} key{index}"
            cells[obj], cell_proofs[obj] = value, f"initialized{index} 0 (by decide)"
        elif kind in ("load", "store"):
            row = selected[key]
            item = row["access"]
            coordinates = [registers[b["name"]] for b in item["indices"]]
            if coordinates != [0, 0]:
                raise ValueError("finite witness currently selects the initialized zero coordinate")
            definitions += [
                f"def access{index} : Access := {access(item)}",
                f"def binding{index} : Binding := {binding(row['result' if kind == 'load' else 'value'])}",
                f"theorem selected{index} : specAt plan.specifications key{index} = "
                f"some (.{kind} access{index} binding{index}) := by decide",
            ]
            rows, columns = item["shape"]
            read = (
                f"⟨rfl, rfl, by change GraphLoopsReuse.Geometry.Coordinate [{rows}, {columns}] [0, 0]; "
                "simp [GraphLoopsReuse.Geometry.Coordinate]⟩"
            )
            if kind == "load":
                value = cells[obj]
                registers[row["result"]["name"]] = value
                lines.append(
                    f"  let {state} : State := ⟨writeRegister {prior}.observed {row['result']['name']} "
                    f"(.scalar {_string(element)} {value}), {prior}.heap⟩"
                )
                tail = f"⟨[0, 0], {value}, {read}, {cell_proofs[obj]}, {holds}, rfl⟩"
            else:
                value = registers[row["value"]["name"]]
                lines.append(
                    f"  let {state} : State := ⟨{prior}.observed, "
                    f"GraphLoopsReuse.replaceCell {prior}.heap {obj} 0 {value}⟩"
                )
                lines.append(f"  have cell{index} : {state}.heap {obj} 0 = some {value} := rfl")
                cells[obj], cell_proofs[obj] = value, f"cell{index}"
                tail = f"⟨[0, 0], {value}, {read}, rfl, rfl⟩"
            proof = (
                f"({kind}_step_iff {arguments} access{index} binding{index} selected{index} "
                f"plan.objects plan.configs (by rfl) {prior} {state}).mpr {tail}"
            )
            instruction_literal = f".{kind} {obj} key{index}"
        else:
            raise ValueError("unsupported witness instruction")
        step = f"step{index}"
        steps.append(step)
        lines.append(
            f"  have {step} : GraphLoopsMemorySCF.Step plan.objects plan.configs meaning false false "
            f"({instruction_literal}) {prior} {state} := {proof}"
        )
    returned = selected[plan["terminal"]["key"]]["values"]
    outputs = (
        "["
        + ", ".join(
            (
                f".index {registers[b['name']]}"
                if b["kind"]["tag"] == "index"
                else f".scalar {_string(element)} {registers[b['name']]}"
            )
            for b in returned
        )
        + "]"
    )
    nested = ".done _"
    for step in reversed(steps):
        nested = f".more {step} ({nested})"
    lines += [f"  refine ⟨state{len(instructions) - 1}, ?_, rfl, rfl⟩", "  exact " + nested]
    code = f"""
namespace GraphShapeEpochWitness
set_option autoImplicit false
set_option maxRecDepth 16384
set_option maxHeartbeats 8000000
open Merlin Merlin.GraphLoopsShape Merlin.GraphLoopsShape.Epoch
open GraphShapeEpochObligation
abbrev State := GraphLoopsReuse.Memory Nat (PublicState Nat Unit)
def scalar : BinaryMeaning Nat := fun _ a b result => result = a + b
def inputs : List (Runtime Nat) :=
  [.scalar {_string(element)} 17, .scalar {_string(element)} 29, view {caller_object} [1] {_string(element)}]
def initialHeap : GraphLoopsReuse.Heap Nat :=
  fun object index => if object = {caller_object} ∧ index = 0 then some 91 else none
def source : State := entryMemory plan kernel.function inputs initialHeap ()
def meaning : GraphLoopsMemorySCF.Interpretation Nat (PublicState Nat Unit) := plan.oldMeaning scalar
{chr(10).join(definitions)}
theorem epoch_source_inhabited : ∃ after,
    GraphLoopsMemorySCF.Execution plan.objects plan.configs meaning false false kernel.instructions source after ∧
      after.observed.returned = {outputs} ∧ after.observed.callerHeap {caller_object} 0 = some {caller_value} := by
{chr(10).join(lines)}
theorem typed_source_inhabited : ∃ world,
    complete (graphMeaning plan.opcodes scalar) GraphShapeObligation.actual.module false kernel.function.name
      inputs ⟨⟨0, ()⟩, initialHeap⟩ {outputs} world ∧ world.heap {caller_object} 0 = some {caller_value} := by
  obtain ⟨after, run, returned, caller⟩ := epoch_source_inhabited
  have typed : ValuesHold kernel.function.parameters inputs := by
    exact .cons ⟨rfl, by intro i h; cases h⟩
      (.cons ⟨rfl, by intro i h; cases h⟩ (.cons ⟨rfl, by intro i h; cases h⟩ .nil))
  have callers : CallersValid plan.objects.length plan.callers (assign emptyEnv kernel.function.parameters inputs) := by
    intro item member
    simp only [plan, List.mem_cons, List.not_mem_nil, or_false] at member
    subst item
    exact ⟨{caller_object}, by decide, rfl⟩
  refine ⟨clockWorld plan.objects.length (finalClock 0 kernel.commands) after, ?_, ?_⟩
  · apply (GraphShapeEpochObligation.complete_plain _ _ _ _ _ _).mpr
    exact (complete_adapter scalar inputs typed callers initialHeap () _ _).mpr ⟨after, run, returned.symm, rfl⟩
  · exact (read_caller plan.objects.length {caller_object} 0
      after.heap after.observed.callerHeap (by decide)).trans caller
theorem segmented_inhabited : ∃ state,
    GraphLoopsMemorySCF.outcomes plan.objects plan.configs meaning true true kernel.instructions source state ∧
      state.returned = {outputs} ∧ state.callerHeap {caller_object} 0 = some {caller_value} := by
  obtain ⟨after, run, returned, caller⟩ := epoch_source_inhabited
  refine ⟨after.observed, ?_, returned, caller⟩
  exact (GraphMemorySCFObligation.complete meaning source source rfl after.observed).mp ⟨after, run, rfl⟩
end GraphShapeEpochWitness
"""
    return body + code + "\n".join("#print axioms " + root for root in WITNESS_ROOTS) + "\n", roots + WITNESS_ROOTS
