"""Closed independent source-only plans, with no tensor or golden construction.

Hardware-derived extents stress original source types around observed memory
volume/depth. They do not assert any mapping, residence or encoded index width.
All execution and target static legality remain separately required evidence.
"""

from __future__ import annotations

from merlin.targetgen import component_program, corpus_spec
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.semantic_families import from_op

from . import component_compile_graphs as G
from . import component_graph_variants as F
from .component_semantic_basis import PROVENANCE
from .rtl_intake import RtlIntakeRefusal

SCHEMA = "merlin.component_compile_only_plan.v1"
OBLIGATIONS = ("semantic_coverage", "index_bounds", "resource_legality", "complete_output_coverage")
_EXTRA = {"input_numeric_domain", "streaming_legality", "dependency_legality", "encoded_field_bounds"}
_BUDGET = {"max_members", "max_source_bytes", "max_extent_bits", "max_scalar_bits"}


def _name(value):
    return isinstance(value, str) and bool(value) and all(c.isalnum() or c in "_-" for c in value)


def _positive(value, detail):
    if type(value) is not int or value < 1:
        raise RtlIntakeRefusal(detail + " requires a positive integer")
    return value


def validate(plan, *, hardware, software):
    """Validate the complete protected declaration before source construction."""
    if (
        not isinstance(plan, dict)
        or set(plan)
        != {"schema", "provenance", "hardware_intake_sha256", "software_intake_sha256", "budget", "members"}
        or plan["schema"] not in {SCHEMA, G.SCHEMA}
        or plan["provenance"] != PROVENANCE
        or plan["hardware_intake_sha256"] != hardware.sha256
        or plan["software_intake_sha256"] != software.sha256
    ):
        raise RtlIntakeRefusal("source-only plan needs exact independent preauthor selection and live intake bindings")
    budget, members = plan["budget"], plan["members"]
    budget_keys = _BUDGET | ({"max_nodes", "max_outputs"} if plan["schema"] == G.SCHEMA else set())
    if not isinstance(budget, dict) or set(budget) != budget_keys:
        raise RtlIntakeRefusal("source-only plan needs the complete explicit metadata budget")
    for key, value in budget.items():
        _positive(value, key)
    if not isinstance(members, list) or not members:
        raise RtlIntakeRefusal("source-only plan must retain every required member")
    names = set()
    for row in members:
        keys = {
            "name",
            "cohort",
            "expectation",
            "operation",
            "dimensions",
            "required_static_obligations",
        }
        graph = plan["schema"] == G.SCHEMA and G.is_graph(row)
        keys |= G.GRAPH_FIELDS if graph else {"operation_owner"}
        if not isinstance(row, dict) or set(row) != keys:
            raise RtlIntakeRefusal("source-only members refuse authored source, model, schedule and history metadata")
        if not _name(row["name"]) or row["name"] in names:
            raise RtlIntakeRefusal("source-only member names and semantic owners must be unique safe components")
        names.add(row["name"])
        if row["cohort"] not in {"functional_guard", "withheld_transfer"} or row["expectation"] not in {
            "compile_only",
            "static_refusal",
        }:
            raise RtlIntakeRefusal("source-only member needs its original distinct cohort and expectation")
        obligations = row["required_static_obligations"]
        if (
            not isinstance(obligations, list)
            or any(type(name) is not str for name in obligations)
            or len(set(obligations)) != len(obligations)
            or not set(OBLIGATIONS) <= set(obligations)
            or not set(obligations) <= set(OBLIGATIONS) | _EXTRA
        ):
            raise RtlIntakeRefusal("source-only members must retain all original static obligations")
        if not isinstance(row["operation"], str) or not isinstance(row["dimensions"], dict):
            raise RtlIntakeRefusal("source-only operation and dimensions require an explicit closed declaration")
        if graph:
            G.validate(row)
            if not all(_name(owner) for owner in row["operation_owners"].values()):
                raise RtlIntakeRefusal("source-only graph needs safe independent primitive owners")
            if not {"dependency_legality", "input_numeric_domain"} <= set(obligations):
                raise RtlIntakeRefusal("source-only graph must retain dependency and numerical-domain obligations")
        elif not _name(row["operation_owner"]):
            raise RtlIntakeRefusal("source-only member requires a safe selected semantic owner")
    G.members(plan)
    return plan


def dtypes(software, budget):
    semantics = software.public_facts()["numerical_semantics"]
    if semantics["model"]["engine"] != "integer_reference" or semantics.get("overflow") not in {
        "bounded_exact",
        "modular_wrap",
    }:
        raise RtlIntakeRefusal(
            "source-only primitive frontend currently requires explicit independent integer semantics"
        )
    result = {}
    for role in ("operand", "accumulator"):
        try:
            _, dtype, _, integer = corpus_spec.dtype_info(semantics[role + "_dtype"])
        except (KeyError, TypeError, ValueError) as error:
            raise RtlIntakeRefusal("source-only primitive dtype is unavailable") from error
        if not integer or not dtype.startswith("i") or int(dtype[1:]) > budget["max_scalar_bits"]:
            raise RtlIntakeRefusal("source-only scalar width exceeds its declared metadata budget")
        result[role] = dtype
    return result


def _extent(expression, *, hardware, types, budget):
    if not isinstance(expression, dict):
        raise RtlIntakeRefusal("source-only extents require explicit independent expressions")
    kind = expression.get("kind")
    if kind == "integer" and set(expression) == {"kind", "value"}:
        value = expression["value"]
        evidence = dict(expression)
    elif kind in {"memory_volume", "memory_depth"}:
        required = {"kind", "memory", "offset"}
        if kind == "memory_volume":
            required |= {"dtype_role", "fixed_elements"}
        if set(expression) != required or type(expression["offset"]) is not int:
            raise RtlIntakeRefusal("source-only memory boundary needs exactly its independently selected operands")
        index = expression["memory"]
        if type(index) is not int or index < 0:
            raise RtlIntakeRefusal("source-only boundary requires an actual selected memory ordinal")
        field = "bytes" if kind == "memory_volume" else "depth"
        fact = "memories." + str(index) + "." + field
        observed = _positive(hardware.fact(fact), "observed " + field)
        if kind == "memory_volume":
            role = expression["dtype_role"]
            if role not in types:
                raise RtlIntakeRefusal("source-only volume requires a selected semantic dtype role")
            fixed = _positive(expression["fixed_elements"], "fixed_elements")
            if fixed.bit_length() > budget["max_extent_bits"]:
                raise RtlIntakeRefusal("source-only boundary metadata exceeds the extent budget")
            bits = int(types[role][1:])
            base = observed * 8 // (bits * fixed)
            evidence = {**expression, "fact": fact, "observed": observed, "dtype": types[role], "base": base}
        else:
            base = observed
            evidence = {**expression, "fact": fact, "observed": observed, "base": base}
        value = base + expression["offset"]
    else:
        raise RtlIntakeRefusal("source-only extent expression is unsupported or ambiguous")
    _positive(value, "source-only extent")
    if value.bit_length() > budget["max_extent_bits"]:
        raise RtlIntakeRefusal("source-only extent exceeds its declared metadata budget")
    return value, {
        **evidence,
        "value": value,
        "scope": "original source extent only; target mapping/index legality unknown",
    }


def _owner(row, software, types, program, *, typed_inputs=None, result_dtype=None):
    selected = [owner for owner in software.public_facts()["operations"] if owner["id"] == row["operation_owner"]]
    family, operation = from_op(row["operation"]), row["operation"]
    if len(selected) != 1 or not (
        operation in selected[0].get("ops", [])
        or (
            not selected[0].get("ops")
            and (selected[0]["id"] in {operation, family} or family in selected[0].get("families", []))
        )
    ):
        raise RtlIntakeRefusal("source-only operation " + operation + " has no selected independent semantic owner")
    constraints = selected[0].get("signature") or {}
    result = result_dtype or (types["accumulator"] if operation == "matmul" else types["operand"])
    operands = typed_inputs or [types["operand"]] * len(program["inputs"])
    observations = {
        "ordered_operand_dtypes": operands,
        "ordered_result_dtypes": [result],
        "operand_dtypes": operands[0],
        "accumulator_dtype": types["accumulator"],
        "readout_dtype": result,
        "compute_dtypes": [result],
        "broadcasting": "none",
        "aliasing": "none",
    }
    for key, expected in constraints.items():
        if key not in observations:
            raise RtlIntakeRefusal("source-only semantic signature has an unimplemented obligation: " + key)
        actual = observations[key]
        if "dtype" in key:

            def canonical(value):
                return corpus_spec.dtype_info(value)[1]

            try:
                expected = (
                    [canonical(value) for value in expected] if isinstance(expected, list) else canonical(expected)
                )
            except (TypeError, KeyError, ValueError) as error:
                raise RtlIntakeRefusal("source-only semantic signature dtype is unknown") from error
        if actual not in expected if key == "operand_dtypes" and isinstance(expected, list) else actual != expected:
            raise RtlIntakeRefusal("source-only source disagrees with selected semantic signature: " + key)


def produce(row, *, hardware, software, budget):
    """Render and structurally reopen original types; never create shaped values."""
    operation = row["operation"]
    if G.is_graph(row):
        return _graph(row, hardware=hardware, software=software, budget=budget)
    keys = {"M", "K", "N"} if operation == "matmul" else {"M", "N"}
    if operation not in {"copy", "matmul"} or set(row["dimensions"]) != keys:
        raise RtlIntakeRefusal("source-only primitive frontend or dimension schema is unavailable")
    types = dtypes(software, budget)
    derived = {
        name: _extent(expression, hardware=hardware, types=types, budget=budget)
        for name, expression in row["dimensions"].items()
    }
    m, n = (derived[key][0] for key in ("M", "N"))
    inputs = [{"name": "A", "role": "input", "shape": [m, n], "dtype": "operand"}]
    if operation == "matmul":
        k = derived["K"][0]
        inputs[0]["shape"] = [m, k]
        inputs.append({"name": "W", "role": "weight", "shape": [k, n], "dtype": "operand"})
    program = {
        "inputs": inputs,
        "nodes": [{"name": "P", "op": operation, "inputs": [r["name"] for r in inputs]}],
        "outputs": [{"name": "Y", "value": "P"}],
    }
    _owner(row, software, types, program)
    return _render(program, types, derived, budget)


def _graph(row, *, hardware, software, budget):
    G.counts(row, budget)
    types = dtypes(software, budget)
    derived = {
        name: _extent(expression, hardware=hardware, types=types, budget=budget)
        for name, expression in row["dimensions"].items()
    }
    # A bounded original stage covers every generated primitive/type role.
    # Refuse undeclared semantics before constructing the selected full graph.
    prototype = F.program(
        {**{key: value for key, (value, _) in derived.items()}, "depth": 1, "fanout": 2}, row["graph_variant"]
    )
    _graph_owners(prototype, row, software, types)
    program = G.program(row, {key: value for key, (value, _) in derived.items()}, budget)
    typed = _graph_owners(program, row, software, types)
    source, abi, derivation = _render(program, types, derived, budget)
    derivation["graph"] = {
        "family": row["graph_family"],
        "representation": row["graph_variant"],
        "depth": row["depth"],
        "fanout": row["fanout"],
        "outputs": len(abi.outputs),
        "effects": typed["effects"],
        "logical_epochs": typed["logical_epochs"],
        "scope": "original functionalized SSA graph; physical reuse and numerical equivalence unknown",
    }
    return source, abi, derivation


def _graph_owners(program, row, software, types):
    typed = component_program.analyze(program, operand_dtype=types["operand"], accumulator_dtype=types["accumulator"])
    values = {value["name"]: value for value in typed["inputs"] + typed["nodes"]}
    for node in typed["nodes"]:
        if node["op"] == "alias":
            # This source view is removed by original SSA functionalization;
            # it grants no physical alias or target operation support.
            continue
        operation = "add" if node["op"] == "update" else node["op"]
        _owner(
            {"operation": operation, "operation_owner": row["operation_owners"][operation]},
            software,
            types,
            program,
            typed_inputs=[values[name]["dtype"] for name in node["actual_inputs"]],
            result_dtype=node["dtype"],
        )
    return typed


def _render(program, types, derived, budget):
    typed, source = component_program.render(
        program, operand_dtype=types["operand"], accumulator_dtype=types["accumulator"]
    )
    if len(source.encode()) > budget["max_source_bytes"]:
        raise RtlIntakeRefusal("source-only original source exceeds its declared metadata budget")
    parsed = parse_linalg_mlir(source)
    if (
        parsed["entry"] != "forward"
        or len(parsed["args"]) != len(typed["inputs"])
        or len(parsed["results"]) != len(typed["outputs"])
    ):
        raise RtlIntakeRefusal("source-only original IR omits or changes its complete typed ABI")
    slots = []
    for originals, observed in ((typed["inputs"], parsed["args"]), (typed["outputs"], parsed["results"])):
        records = []
        for original, actual in zip(originals, observed, strict=True):
            if actual["shape"] != original["shape"] or actual["dtype"] != original["dtype"]:
                raise RtlIntakeRefusal(
                    "source-only actual original signature differs from its independently constructed types"
                )
            records.append(CompileOnlyTensor(original["name"], tuple(actual["shape"]), actual["dtype"]))
        slots.append(tuple(records))
    abi = CompileOnlySourceAbi(*slots)
    abi.record()
    return (
        source,
        abi,
        {
            "extents": {key: evidence for key, (_, evidence) in derived.items()},
            "nodes": len(typed["nodes"]),
            "source_bytes": len(source.encode()),
            "tensor_values_allocated": False,
            "goldens_allocated": False,
        },
    )
