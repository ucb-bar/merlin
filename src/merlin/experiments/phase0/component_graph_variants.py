"""Independent bounded topology families constructed from generic DAG semantics.

The source factory is fixed infrastructure, not a workload topology or target
schedule. Selected external parameters control depth, fanout and rank-two
extents. Both representations retain the same arithmetic order and all outputs.
"""

from __future__ import annotations

import copy

from merlin.targetgen import component_program, corpus_spec

from .component_generation import digest

SCHEMA = "merlin.component_graph_family.v1"
VARIANTS = ("logical_epochs", "fresh_values")
PARAMETERS = ("M", "K", "N", "depth", "fanout")
UNAVAILABLE = "_component_source_unavailable"


def validate(declaration):
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "representations"}
        or declaration["schema"] != SCHEMA
        or declaration["representations"] != list(VARIANTS)
    ):
        raise ValueError("graph family requires the closed v1 schema and complete representation pair")
    return declaration


def validate_owners(selected):
    """Every generated computation needs a selected reviewed semantic owner."""
    from merlin.targetgen.semantic_families import from_op

    for operation in ("matmul", "copy", "add"):
        family = from_op(operation)
        if family is None or not any(
            operation in owner.get("ops", []) or family in owner.get("families", []) for owner in selected
        ):
            raise ValueError("graph family lacks a selected semantic owner for " + operation)


def parameters(entry):
    values = {key: entry.get(key) for key in PARAMETERS}
    if any(type(value) is not int or value < 1 for value in values.values()) or values["fanout"] < 2:
        raise ValueError("graph family requires explicit positive M/K/N/depth and fanout at least two")
    return values


def program(values, variant):
    """Build one checked representation only after its symbolic budget passes.

    The same initial contraction feeds a fork and ordered join in every epoch.
    Copies publish the old snapshot and the new epoch without being overwritten
    by subsequent updates. The first fork escapes while also feeding the join.
    Logical updates reuse one source storage owner; fresh values are plain SSA.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown graph representation")
    m, k, n, depth, fanout = (values[key] for key in PARAMETERS)
    nodes = [{"name": "P", "op": "matmul", "inputs": ["A", "W"]}]
    outputs, current = [], "P"
    for epoch in range(depth):
        prefix = "E" + str(epoch) + "_"

        def append(name, op, args):
            name = prefix + name
            nodes.append({"name": name, "op": op, "inputs": args})
            return name

        if variant == "logical_epochs":
            current = append("view", "alias", ["P"])
        snapshot = append("snapshot", "copy", [current])
        forks = [append("fork" + str(branch), "copy", [current]) for branch in range(fanout)]
        joined = forks[0]
        for branch in range(1, fanout):
            joined = append("join" + str(branch), "add", [joined, forks[branch]])
        current = append("next", "update" if variant == "logical_epochs" else "add", [current, joined])
        after = append("after", "copy", [current])
        outputs += [
            {"name": prefix + "old", "value": snapshot},
            {"name": prefix + "escaped", "value": forks[0]},
            {"name": prefix + "published", "value": after},
        ]
    outputs.append({"name": "Final", "value": "P" if variant == "logical_epochs" else current})
    return {
        "inputs": [
            {"name": "A", "role": "input", "shape": [m, k], "dtype": "operand"},
            {"name": "W", "role": "weight", "shape": [k, n], "dtype": "operand"},
        ],
        "nodes": nodes,
        "outputs": outputs,
    }


def _source(values, variant, operand, accumulator, palette=None):
    return {
        "kind": "component_program",
        "program": component_program.analyze(
            program(values, variant), operand_dtype=operand, accumulator_dtype=accumulator
        ),
        "input_palette": palette,
        "stimulus_range": None,
    }


def symbolic_cost(entry, *, binding, variant):
    """Derive exact affine counts from bounded source prototypes, never unroll.

    Depth adds identical stages; each extra fork adds one copy and one ordered
    add. All output counts and scalar widths remain fixed per stage. The actual
    ordinary source cost is rechecked after construction. No shaped data, leaf
    palette or numerical evaluator is invoked here.
    """
    from .component_execution_budget import measure

    values = parameters(entry)
    regime, selected = corpus_spec.entry_binding(entry, binding)
    if regime != "int":
        raise ValueError("graph family has no independently derived cost for this numerical engine")
    operand, accumulator = (
        selected.mlir_dtype(selected.operand_dtype),
        selected.mlir_dtype(selected.accum_dtype),
    )
    base = measure(_source({**values, "depth": 0, "fanout": 2}, variant, operand, accumulator))
    stage_source = _source({**values, "depth": 1, "fanout": 2}, variant, operand, accumulator)
    stage = measure(stage_source)
    extra = measure(_source({**values, "depth": 1, "fanout": 3}, variant, operand, accumulator))
    result = {"scalar_bits": stage["scalar_bits"]}
    for key in ("reference_work", "materialized_elements", "tensor_payload_bytes"):
        result[key] = base[key] + values["depth"] * (
            stage[key] - base[key] + (values["fanout"] - 2) * (extra[key] - stage[key])
        )
    if entry.get("input_palette") is not None:
        # Palette realization is a one-time leaf charge, independent of stages.
        with_palette = measure({**stage_source, "input_palette": entry["input_palette"]})
        for key in result:
            if key != "scalar_bits":
                result[key] += with_palette[key] - stage[key]
    return result


def expand_entry(entry, *, binding, policy):
    """Retain both requested variants, including unavailable source failures."""
    from .component_execution_budget import measure, source_for_entry
    from .component_execution_budget import validate as validate_budget

    declaration = validate(entry["graph_family"])
    result = []
    for variant in VARIANTS:
        value = copy.deepcopy(entry)
        value.pop("graph_family")
        try:
            if policy is None:
                raise ValueError("graph family requires an explicit v2 small-execution budget")
            validate_budget(policy)
            values = parameters(entry)
            cost = symbolic_cost(entry, binding=binding, variant=variant)
            exceeded = [key for key, amount in cost.items() if amount > policy["max_" + key]]
            if exceeded:
                raise ValueError("graph source budget exceeded before topology construction: " + ", ".join(exceeded))
            value["program"] = program(values, variant)
            if measure(source_for_entry(value, binding=binding)) != cost:
                raise ValueError("graph source symbolic cost differs from actual ordinary DAG cost")
        except ValueError as exc:
            values = {key: entry.get(key) for key in PARAMETERS}
            value[UNAVAILABLE] = str(exc)
        value["graph_variant"] = {
            "schema": SCHEMA,
            "family_sha256": digest(declaration),
            "representation": variant,
            "parameters": values,
        }
        result.append(value)
    return result
