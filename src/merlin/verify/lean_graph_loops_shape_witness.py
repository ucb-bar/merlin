"""Generate untrusted nonzero witness terms for the actual typed memory model.

Lean rechecks every step against the full submitted source blueprint. This is an
inhabited abstract interpretation, not native execution or a replacement pass.
"""

import json

from .lean_graph_loops_shape_model import binding, obligation, view_type
from .lean_graph_loops_unroll import _list, _nat, _string

WITNESS_ROOTS = (
    "GraphShapeWitness.source_inhabited",
    "GraphShapeWitness.target_inhabited",
    "GraphShapeWitness.caller_value",
    "Merlin.GraphLoopsShape.Load.ready",
    "Merlin.GraphLoopsShape.Store.ready",
    "Merlin.GraphLoopsShape.SCF.once",
)


def witness_obligation(projection):
    function = projection["module"]["functions"][0]
    if function["name"] != "probe" or any(row["kind"]["tag"] != "view" for row in function["parameters"]):
        raise ValueError("finite descriptor witness supports the explicit view probe family")
    declarations, env = [], {}
    for index, row in enumerate(function["parameters"]):
        data = row["kind"]["type"]
        ty = view_type(data)
        fields = (
            _nat(index + 1),
            _nat(100 * (index + 1)),
            _nat(data["offset"]),
            _list(data["shape"], _nat),
            _list(data["strides"], _nat),
            _string(data["element"]),
            _nat(data["space"]),
        )
        declarations.extend(
            [
                f"def ty{index} : ViewType := {ty}",
                f"def descriptor{index} : Descriptor := ⟨" + ", ".join(fields) + "⟩",
                f"theorem valid{index} : Conforms ty{index} descriptor{index} := by decide",
                f"def arg{index} : Runtime Nat := .view ty{index} descriptor{index} valid{index}",
            ]
        )
        env[row["name"]] = dict(tag="view", index=index, type=data, allocation=index + 1, base=100 * (index + 1))
    classes = {}

    def collect(code):
        if code["tag"] == "returned":
            return
        if code["tag"] in ("action", "region"):
            name = json.loads(code["key"])["name"]
            classes[code["key"]] = {"memref.load": "load", "memref.store": "store", "scf.for": "loop"}.get(
                name, "other"
            )
            if code["tag"] == "region":
                collect(code["body"])
        collect(code["rest"])

    collect(function["code"])
    classify = ".other"
    for key, opcode in reversed(list(classes.items())):
        classify = f"if key = {_string(key)} then .{opcode} else ({classify})"
    worlds, current_cells = [], {}

    def literal(value):
        if value["tag"] == "view":
            return "arg" + str(value["index"])
        if value["tag"] == "index":
            return "(.index " + str(value["value"]) + ")"
        return "(.scalar " + _string(value["type"]) + " " + str(value["value"]) + ")"

    def typed(rows, values):
        result = "ValuesHold.nil"
        for row, value in reversed(list(zip(rows, values, strict=True))):
            result = f"(ValuesHold.cons (Binding.holds_plain {binding(row)} {literal(value)} rfl rfl) {result})"
        return result

    def evaluate(code, bindings_env, world):
        tag = code["tag"]
        if tag == "returned":
            return [bindings_env[row["name"]] for row in code["values"]], world, ["exact ⟨rfl, rfl⟩"]
        local = dict(bindings_env)
        proof = []
        if tag == "constant":
            local[code["result"]["name"]] = dict(tag="index", value=code["value"])
            proof.append("apply Eval.source_constant")
        elif tag == "cast":
            value = local[code["origin"]["name"]]
            local[code["result"]["name"]] = value
            proof.extend([f"apply Eval.source_cast {literal(value)}", "· rfl"])
        elif tag == "dimension":
            local[code["result"]["name"]] = dict(tag="index", value=code["value"])
            proof.extend([f"apply Eval.source_dimension (actual := {code['value']})", "· rfl"])
        elif tag == "action":
            opcode = classes[code["key"]]
            values = [local[row["name"]] for row in code["operands"]]
            if opcode == "load":
                view, *indices = values
                coordinates = [value["value"] for value in indices]
                address = (
                    view["base"]
                    + view["type"]["offset"]
                    + sum(a * b for a, b in zip(view["type"]["strides"], coordinates, strict=True))
                )
                stored = current_cells.get((view["allocation"], address), {1: 17, 2: 29, 3: 91}[view["allocation"]])
                returned = [dict(tag="scalar", type=view["type"]["element"], value=stored)]
                run = (
                    f"Load.ready ty{view['index']} descriptor{view['index']} valid{view['index']} "
                    f"{_list(coordinates, _nat)} {stored} {world} "
                    f"(by simp [descriptor{view['index']}, Merlin.GraphLoopsReuse.Geometry.Coordinate]) (by rfl)"
                )
                next_world = world
            elif opcode == "store":
                scalar, view, *indices = values
                coordinates = [value["value"] for value in indices]
                address = (
                    view["base"]
                    + view["type"]["offset"]
                    + sum(a * b for a, b in zip(view["type"]["strides"], coordinates, strict=True))
                )
                next_world = "world" + str(len(worlds) + 1)
                worlds.append(
                    f"def {next_world} : MemoryWorld Nat (List Nat) := ⟨{world}.observed, "
                    f"Merlin.GraphLoopsReuse.replaceCell {world}.heap "
                    f"{view['allocation']} {address} {scalar['value']}⟩"
                )
                current_cells[view["allocation"], address] = scalar["value"]
                returned = []
                run = (
                    f"Store.ready ty{view['index']} descriptor{view['index']} valid{view['index']} "
                    f"{_list(coordinates, _nat)} {scalar['value']} {world} "
                    f"(by simp [descriptor{view['index']}, Merlin.GraphLoopsReuse.Geometry.Coordinate])"
                )
            else:
                raise ValueError("finite witness has no unproved opaque operation meaning")
            values_literal = _list(returned, literal)
            proof.extend(
                [
                    f"apply Eval.source_action {values_literal} {next_world}",
                    f"· exact {run}",
                    f"· exact {typed(code['results'], returned)}",
                ]
            )
            for row, value in zip(code["results"], returned, strict=True):
                local[row["name"]] = value
            world = next_world
        elif tag == "region":
            values = [local[row["name"]] for row in code["operands"]]
            if [value.get("value") for value in values] != [0, 1, 1] or code["results"]:
                raise ValueError(
                    "finite witness supports one explicit no-carry iteration; general theorem is not bounded here"
                )
            child_env = dict(local)
            child_inputs = [dict(tag="index", value=0)]
            for row, value in zip(code["parameters"], child_inputs, strict=True):
                child_env[row["name"]] = value
            returned, next_world, child_proof = evaluate(code["body"], child_env, world)
            if returned:
                raise ValueError("finite no-carry body must yield no values")
            proof.extend(
                [
                    f"apply Eval.source_region [] {next_world}",
                    "· apply SCF.once",
                    f"  refine ⟨{typed(code['parameters'], child_inputs)}, ?_⟩",
                ]
            )
            proof.extend("  " + line for line in child_proof)
            proof.append("· exact ValuesHold.nil")
            world = next_world
        else:
            raise ValueError("unknown witness syntax")
        outputs, final, remaining = evaluate(code["rest"], local, world)
        if tag in ("cast", "dimension", "action", "region"):
            proof.extend(["· " + remaining[0], *["  " + line for line in remaining[1:]]])
        else:
            proof.extend(remaining)
        return outputs, final, proof

    outputs, after, proof = evaluate(function["code"], env, "before")
    args = [env[row["name"]] for row in function["parameters"]]
    arg_literal, out_literal = _list(args, literal), _list(outputs, literal)
    code_source = obligation(projection)
    caller_value = current_cells.get((3, 300), 91)
    roots = "\n".join("#print axioms " + root for root in WITNESS_ROOTS)
    return (
        code_source
        + f"""
namespace GraphShapeWitness
open Merlin Merlin.GraphLoopsShape
abbrev actual := GraphShapeObligation.actual
{chr(10).join(declarations)}
def classify (key : String) : MemoryOpcode := {classify}
def base : Interpretation Nat (MemoryWorld Nat (List Nat)) where
  action := fun _ _ _ _ _ => False
  region := fun _ _ _ _ _ _ => False
def meaning := withMemory base classify
def inputs : List (Runtime Nat) := {arg_literal}
def before : MemoryWorld Nat (List Nat) := ⟨[], fun object _ =>
  if object = 1 then some 17 else if object = 2 then some 29 else if object = 3 then some 91 else none⟩
{chr(10).join(worlds)}
def after : MemoryWorld Nat (List Nat) := {after}
def outputs : List (Runtime Nat) := {out_literal}
theorem source_inhabited : complete meaning actual.module false "probe" inputs before outputs after := by
  refine ⟨1, ?_⟩
  change ValuesHold actual.module.functions[0].parameters inputs ∧
    Eval (withCalls meaning actual.module (invoke meaning actual.module false 0)) false
      (extendAliases [] actual.module.functions[0].parameters) actual.module.functions[0].code
      (assign emptyEnv actual.module.functions[0].parameters inputs) before outputs after
  refine ⟨{typed(function["parameters"], args)}, ?_⟩
{chr(10).join("  " + line for line in proof)}
theorem target_inhabited : complete meaning actual.module true "probe" inputs before outputs after :=
  (GraphShapeObligation.complete meaning "probe" inputs before outputs after).mp source_inhabited
theorem caller_value : after.heap 3 300 = some {caller_value} := by rfl
end GraphShapeWitness
{roots}
"""
    )
