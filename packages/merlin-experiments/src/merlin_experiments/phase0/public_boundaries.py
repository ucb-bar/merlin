"""Ordinary public-source pressure witnesses, captured by the existing writer.

Every changed geometry is a new complete source call and fresh host reference.
The sum of logical tensor volumes is conditional pressure at a selected store's
row width. It proves no buffer placement, simultaneous allocation or lifetime.
"""

from __future__ import annotations

import copy
import hashlib
import json
from math import prod
from pathlib import Path

from . import public_boundary_semantics as S
from . import public_programs as P

SCHEMA = "merlin.phase0.public_joint_boundaries.v1"
SCHEMA_V2 = "merlin.phase0.public_joint_boundaries.v2"
SCHEMA_V3 = "merlin.phase0.public_joint_boundaries.v3"
TEMPLATE = {
    "schema": SCHEMA,
    "constructor": "original_connected_float32_cuts",
    "pressure": "logical_tensor_volume_at_selected_store_rows",
}
TEMPLATE_V2 = {**TEMPLATE, "schema": SCHEMA_V2, "constructor": "original_typed_connected_cuts"}
TEMPLATE_V3 = {
    "schema": SCHEMA_V3,
    "constructor": "original_typed_functional_connected_cuts",
    "scope": "bounded_functional_source_only",
}
_BUDGETS = {"max_members", "max_nodes", "max_logical_elements", "max_extent", "max_source_bytes"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", *_BUDGETS}
        or value.get("schema") not in (SCHEMA, SCHEMA_V2, SCHEMA_V3)
        or any(type(value[k]) is not int or value[k] < 1 for k in _BUDGETS)
    ):
        raise ValueError("public joint boundaries require an explicit version and finite source budgets")
    return copy.deepcopy(value)


def template_for(policy):
    return {SCHEMA: TEMPLATE, SCHEMA_V2: TEMPLATE_V2, SCHEMA_V3: TEMPLATE_V3}[validate(policy)["schema"]]


def render(program):
    """Unroll exact original ATen calls; no new arithmetic or integer surrogate."""
    if not program.get("inputs") or not program.get("nodes"):
        raise ValueError("public boundary source requires a closed nonempty graph")
    if any(
        not isinstance(row.get("name"), str) or not row["name"].startswith("v") or not row["name"][1:].isdigit()
        for row in [*program["inputs"], *program["nodes"]]
    ):
        raise ValueError("public boundary source must use constructor-owned value names")
    names = {row["name"] for row in program["inputs"]}
    typed = program.get("schema") == S.TYPED_SCHEMA
    from . import public_typed_semantics as T

    def literal(value):
        if isinstance(value, dict):
            if typed and set(value) == {"kind", "value"}:
                return T.literal(value)
            if set(value) != {"value"} or value["value"] not in names:
                raise ValueError("public boundary source operand is unavailable")
            return value["value"]
        if isinstance(value, list):
            return "[" + ", ".join(literal(v) for v in value) + "]"
        if value is not None and type(value) not in {int, float, bool, str}:
            raise ValueError("public boundary literal is unsupported")
        return repr(value)

    lines = [
        '"""Generated original public source cut; physical mapping and timing UNKNOWN."""',
        "import torch",
        "",
        f"# Typed shape bindings: {digest(program)}",
        "class Program(torch.nn.Module):",
        "    def forward(self, " + ", ".join(row["name"] for row in program["inputs"]) + "):",
    ]
    for row in program["nodes"]:
        if (
            row["target"] not in (T.SUPPORTED if typed else S.SUPPORTED)
            or row["dtype"] not in (T.DTYPES if typed else {"float32"})
            or row["name"] in names
        ):
            raise ValueError("public boundary call is not a supported fresh original node")
        args = [literal(v) for v in row["args"]] + [k + "=" + literal(v) for k, v in row["kwargs"].items()]
        lines.append(f"        {row['name']} = torch.ops.{row['target']}({', '.join(args)})")
        names.add(row["name"])
    if any(name not in names for name in program["outputs"]):
        raise ValueError("public boundary output is unavailable")
    lines += ["        return (" + ", ".join(program["outputs"]) + ",)", "", "def get_model_and_inputs():"]
    for ordinal, row in enumerate(program["inputs"]):
        if row["dtype"] not in (T.DTYPES if typed else {"float32"}) or any(
            type(d) is not int or d < 1 for d in row["shape"]
        ):
            raise ValueError("public boundary input type is unavailable")
        # A bounded deterministic independent input domain. No original weights,
        # source outputs or backend-dependent arithmetic enters these stimuli.
        if typed:
            # Source integer representations retain their actual signedness.
            # Float stimuli cross integer cast boundaries instead of collapsing
            # to all-zero narrowed values; eager capture still owns references.
            dtype = row["dtype"]
            expression = f"(((torch.arange({prod(row['shape'])},dtype=torch.int64)+{ordinal})%17)-8)"
            if dtype.startswith(("float", "bfloat")):
                expression += f".to(torch.{dtype}) / 4"
            elif dtype == "bool":
                expression += ".remainder(2).to(torch.bool)"
            else:
                expression += f".to(torch.{dtype})"
            order = row.get("layout_order", list(range(len(row["shape"]))))
            inverse = [order.index(i) for i in range(len(order))]
            expression = f"({expression}).reshape({[row['shape'][i] for i in order]!r})"
            if order != list(range(len(order))):
                expression += f".permute({inverse!r})"
            lines.append(f"    {row['name']} = {expression}")
            continue
        lines.append(
            f"    {row['name']} = (((torch.arange({prod(row['shape'])}, dtype=torch.int64) + {ordinal}) % 17) - 8)"
            f".to(torch.float32).reshape({row['shape']!r}) / 32"
        )
    lines.append("    return Program().eval(), (" + ", ".join(row["name"] for row in program["inputs"]) + ",)")
    return "\n".join(lines) + "\n"


def derive(original_entries, outputs, *, evidence, policy):
    """Construct bounded new programs from already bound public capture bytes."""
    if validate(policy)["schema"] == SCHEMA_V3:
        from .public_functional_sources import derive as functional

        return functional(original_entries, outputs, evidence=evidence, policy=policy)
    from .resource_boundaries import BoundaryUnavailable
    from .resource_boundaries import derive as boundary
    from .sweeps import resolve_extent

    policy = validate(policy)
    entries, products, missing = [], {}, []
    requested = 0
    facts = evidence.refreshed_facts
    quantum_values = {
        a.get("rows")
        for a in (facts.get("facts") or {}).get("arrays") or []
        if isinstance(a, dict) and type(a.get("rows")) is int and a["rows"] > 0
    }
    quantum = next(iter(quantum_values)) if len(quantum_values) == 1 else None
    stores = (facts.get("facts") or {}).get("memories") or []

    def unknown(origin, reason, **detail):
        missing.append({"origin": origin, "status": "UNKNOWN", "mandatory": True, "reason": reason, **detail})

    if type(quantum) is not int or quantum < 1 or not stores:
        unknown(None, "selected facts do not establish an extent quantum and memory capacity roster")
        return (
            entries,
            products,
            {
                "schema": policy["schema"],
                "selection": policy,
                "requested": 0,
                "constructed": 0,
                "missing": missing,
                "members_sha256": digest(entries),
                "raw_facts_sha256": evidence.raw_facts_sha256,
            },
        )
    for original in original_entries:
        binding = original["materialized_capture"]
        prefix = Path(binding["path"]).parent.as_posix()
        raw = outputs[prefix + "/frontend-trace.json"]
        receipt = json.loads(outputs[prefix + "/capture_receipt.json"])
        if hashlib.sha256(raw).hexdigest() != receipt["artifacts"]["frontend-trace.json"]["sha256"]:
            raise ValueError("public boundary source trace differs from its original capture receipt")
        trace = json.loads(raw)
        source_context = {"path": prefix + "/frontend-trace.json", "sha256": hashlib.sha256(raw).hexdigest()}
        products[source_context["path"]] = raw
        typed = policy["schema"] == SCHEMA_V2
        from . import public_typed_semantics as T

        regions, gaps = S.cuts(trace, max_nodes=policy["max_nodes"], typed=typed)
        origin = {
            key: binding[key] for key in ("capture_sha256", "receipt_sha256", "loader_sha256", "full_inventory_sha256")
        }
        origin["graph_sha256"] = trace["graphs"]["original"]["sha256"]
        for gap in gaps:
            unknown(origin, gap["reason"], source_gap=gap)
        if not regions:
            unknown(origin, "no exact connected contraction/non-contraction source cut is constructible")
        for region in regions:
            dimensions = region["dimensions"]
            if len(dimensions) < 2:
                unknown(origin, "source cut has fewer than two independent shape bindings")
                continue
            # Every independent dimension is represented. The first varies at
            # the boundary; the others jointly exercise neighboring extents.
            axis = dimensions[0]
            fixed = {d: [max(1, quantum + (1 if i % 2 else -1))] for i, d in enumerate(dimensions[1:])}
            for index, store in enumerate(stores):
                if not isinstance(store, dict) or not isinstance(store.get("name"), str):
                    unknown(origin, "selected memory has no exact store identity", memory_index=index)
                    continue
                spec = {
                    "derive": "resident_allocation_boundary",
                    "store": store["name"],
                    "capacity_fact": ["memories", index, "bytes"],
                    "reservation_facts": [],
                    "quantum": quantum,
                    "tail_offsets": [-1, 1],
                    "include_exact_capacity": True,
                    **({"logical_payload_only": True} if typed else {}),
                    "allocations": [
                        {
                            "name": r["name"],
                            "shape": r["shape"],
                            "dtype": T.DTYPES[r["dtype"]] if typed else "f32",
                            **(
                                {
                                    "logical_element_bits": 1
                                    if r["dtype"] == "bool"
                                    else int(
                                        T.DTYPES[r["dtype"]]
                                        .removeprefix("bf")
                                        .removeprefix("f")
                                        .removeprefix("i")
                                        .removeprefix("u")
                                    )
                                }
                                if typed
                                else {}
                            ),
                        }
                        for r in [*region["inputs"], *region["nodes"]]
                    ],
                }
                try:
                    values, labels, pressure = boundary(
                        spec,
                        owner="public source logical pressure",
                        axis=axis,
                        target=evidence.target,
                        tile=quantum,
                        dtype=T.DTYPES[region["inputs"][0]["dtype"]] if typed else "fp32",
                        fixed=fixed,
                        evidence=evidence,
                        resolve_extent=resolve_extent,
                    )
                except BoundaryUnavailable as exc:
                    unknown(origin, "selected source/store pressure boundary is unavailable", boundary=exc.record)
                    continue
                pressure["scope"] = (
                    "sum of logical source tensor volumes at selected store row width; "
                    "physical mapping, lifetimes and reservations UNKNOWN"
                )
                pressure["hardware_mapping"] = "UNKNOWN"
                pressure["reservation_interpretation"] = "no physical reservation claim; logical volume comparison only"
                for extent in values:
                    requested += 1
                    actual = {d: fixed[d][0] for d in fixed} | {axis: extent}
                    if requested > policy["max_members"] or max(actual.values()) > policy["max_extent"]:
                        unknown(origin, "requested boundary member exceeds finite metadata budget", dimensions=actual)
                        continue
                    program = S.instantiate(region, actual)
                    if S.logical_elements(program) > policy["max_logical_elements"]:
                        unknown(
                            origin,
                            "complete logical tensor/reference volume exceeds selected budget",
                            dimensions=actual,
                        )
                        continue
                    source = render(program).encode()
                    if len(source) > policy["max_source_bytes"]:
                        unknown(origin, "constructed source exceeds selected byte budget", dimensions=actual)
                        continue
                    point = next(p for p in pressure["points"] if p["extent"] == extent)
                    member = {
                        "schema": policy["schema"],
                        "origin": origin,
                        "source_context": source_context,
                        "constructor_node_budget": policy["max_nodes"],
                        "program": program,
                        "pressure": pressure,
                        "point": point,
                        "loader_sha256": hashlib.sha256(source).hexdigest(),
                    }
                    identity = digest(member)[:20]
                    name = "PJ_" + identity
                    path = "public-boundaries/" + name + "/loader.py"
                    purpose = (
                        "complete connected original-call source cut at logical aggregate "
                        + labels[extent]
                        + " pressure, with joint dimension tails and every original live-out result checked"
                    )
                    performance = P.performance(original, trace=trace, forms={}, facts_sha256=evidence.raw_facts_sha256)
                    performance["family"] = "public_joint_boundary"
                    performance["lever"] = purpose
                    performance["emitter"]["entry"] = "normal sealed Model2MLIR capture + write_model_capsule"
                    performance["gate"]["source_execution_basis"]["operations"] = list(
                        dict.fromkeys(n["target"] for n in program["nodes"])
                    )
                    performance["curriculum"].update(
                        purpose=purpose,
                        operation_semantics=performance["gate"]["source_execution_basis"]["operations"],
                        resource_boundary={
                            "derivation": pressure,
                            "selected_point": point,
                            "hardware_mapping": "UNKNOWN",
                        },
                        derivation={
                            **origin,
                            "constructed_program_sha256": digest(program),
                            "loader_sha256": member["loader_sha256"],
                        },
                    )
                    performance["curriculum"]["observed_families"] = {
                        key: [
                            op for op in operations if op in performance["gate"]["source_execution_basis"]["operations"]
                        ]
                        for key, operations in performance["curriculum"]["observed_families"].items()
                    }
                    performance["curriculum"]["missing_families"] = [
                        key
                        for key, operations in performance["curriculum"]["observed_families"].items()
                        if not operations
                    ]
                    performance["curriculum"]["numerical_reference"] = (
                        "all newly captured original live-out outputs; existing output accuracy gates retained"
                    )
                    performance["curriculum"]["limitations"].append(
                        "source cut uses fresh independent inputs and does not certify the complete original program"
                    )
                    entries.append(
                        {
                            "name": name,
                            "kind": "model",
                            "op": "model",
                            "cat": "_perf",
                            "label": "dev",
                            "source_role": "derived_sweep",
                            "source_reference": "receipt-bound public original call semantics and selected facts",
                            "operand_dtype": next(
                                (
                                    T.DTYPES[r["dtype"]]
                                    for r in region["inputs"]
                                    if r["dtype"].startswith(("float", "bfloat"))
                                ),
                                "fp32",
                            )
                            if typed
                            else "fp32",
                            "loader": path,
                            "public_boundary": member,
                            "performance": performance,
                        }
                    )
                    products[path] = source
    report = {
        "schema": policy["schema"],
        "selection": policy,
        "requested": requested,
        "constructed": len(entries),
        "missing": missing,
        "members_sha256": digest(entries),
        "raw_facts_sha256": evidence.raw_facts_sha256,
    }
    return entries, products, report


def selected_loader(entry, root):
    member = entry["public_boundary"]
    context = member["source_context"]
    context_path = Path(context["path"])
    if context_path.is_absolute() or ".." in context_path.parts:
        raise ValueError("public boundary original context must remain in its selected public input bundle")
    context_path = Path(root) / context_path
    if (
        context_path.is_symlink()
        or not context_path.is_file()
        or hashlib.sha256(context_path.read_bytes()).hexdigest() != context["sha256"]
    ):
        raise ValueError("public boundary original typed source context changed")
    trace = json.loads(context_path.read_bytes())
    if trace["graphs"]["original"]["sha256"] != member["origin"]["graph_sha256"]:
        raise ValueError("public boundary original source identity differs")
    functional = member.get("schema") == SCHEMA_V3
    regions, _ = S.cuts(
        trace,
        max_nodes=member["constructor_node_budget"],
        typed=member.get("schema") in (SCHEMA_V2, SCHEMA_V3),
        shared_inputs=functional,
    )
    regions = [region for region in regions if region["original_nodes"] == member["program"]["original_nodes"]]
    if (
        len(regions) != 1
        or set(regions[0]["dimensions"]) != set(member["program"]["dimension_values"])
        or S.instantiate(regions[0], member["program"]["dimension_values"]) != member["program"]
    ):
        raise ValueError("public boundary program is not a constructed original connected source cut")
    if functional:
        from .public_functional_sources import verify_member

        verify_member(member, regions[0], entry)
    path = Path(entry["loader"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("public boundary loader must be contained beside its synthesis profile")
    path = Path(root) / path
    if (
        path.is_symlink()
        or not path.is_file()
        or path.read_bytes() != render(member["program"]).encode()
        or hashlib.sha256(path.read_bytes()).hexdigest() != member["loader_sha256"]
    ):
        raise ValueError("public boundary loader differs from its constructed typed source")
    return path


def verify_written(capsule, directory):
    """Check the actual fresh capture before publication and again at freeze."""
    directory = Path(directory)
    member = capsule["public_boundary"]
    program = member["program"]
    functional = member.get("schema") == SCHEMA_V3
    basis = member["source_execution_basis"] if functional else capsule["performance"]["gate"]["source_execution_basis"]
    if functional:
        from .public_functional_sources import basis as functional_basis

        if basis != functional_basis(member):
            raise ValueError("functional source purpose differs from its full derivation")
    if (
        basis["capture_sha256"] != member["origin"]["capture_sha256"]
        or basis["receipt_sha256"] != member["origin"]["receipt_sha256"]
        or basis["operations"] != list(dict.fromkeys(n["target"] for n in program["nodes"]))
        or (
            not functional
            and capsule["performance"]["curriculum"]["derivation"]["constructed_program_sha256"] != digest(program)
        )
    ):
        raise ValueError("public boundary purpose differs from its source derivation")
    source = (directory / "capsule.pytorch.py").read_bytes()
    if (
        member.get("schema") not in (SCHEMA, SCHEMA_V2, SCHEMA_V3)
        or source != render(program).encode()
        or hashlib.sha256(source).hexdigest() != member["loader_sha256"]
    ):
        raise ValueError("public boundary source changed")
    frontend = capsule["frontend_trace"]
    raw = (directory / "frontend-trace.json").read_bytes()
    interface = (directory / "capsule.interface.mlir").read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != frontend["sha256"]
        or interface != (directory / "frontend-source.mlir").read_bytes()
        or hashlib.sha256(interface).hexdigest() != frontend["source_mlir_sha256"]
    ):
        raise ValueError("public boundary captured source changed")
    trace = json.loads(raw)
    S.original_use_def_semantics(trace)
    nodes = trace["graphs"]["original"]["nodes"]
    inputs = [n for n in nodes if n["op"] == "placeholder"]
    calls = [n for n in nodes if n["op"].startswith("call")]
    if (
        len(inputs) != len(program["inputs"])
        or len(calls) != len(program["nodes"])
        or any(n["op"] == "get_attr" for n in nodes)
    ):
        raise ValueError("public boundary capture changed the original call or input roster")
    names = {}

    def argument(value):
        if isinstance(value, dict):
            if set(value) == {"node_id", "value_id"}:
                return {"value": names[value["value_id"]]}
            if program.get("schema") == S.TYPED_SCHEMA:
                from .public_typed_semantics import literal

                literal(value)
                return value
            raise ValueError("public boundary actual argument is unsupported")
        return [argument(v) for v in value] if isinstance(value, list) else value

    for actual, expected in zip([*inputs, *calls], [*program["inputs"], *program["nodes"]], strict=True):
        if len(actual["results"]) != 1 or any(actual["results"][0].get(k) != expected[k] for k in ("shape", "dtype")):
            raise ValueError("public boundary actual tensor type differs from derived dimensions")
        if "layout_order" in expected:
            from .public_typed_semantics import layout_order

            if layout_order(actual["results"][0]) != expected["layout_order"]:
                raise ValueError("public boundary changed the original logical input layout")
        if "target" in expected and (
            actual["target"] != expected["target"]
            or trace["graphs"]["original"].get("operator_schemas", {}).get(actual["target"])
            != expected["operator_schema"]
            or argument(actual["args"]) != expected["args"]
            or {k: argument(v) for k, v in actual["kwargs"].items()} != expected["kwargs"]
        ):
            raise ValueError("public boundary actual original call semantics differ")
        names[actual["results"][0]["id"]] = expected["name"]
    published = [names[v] for v in S.references(nodes[-1]["args"])]
    if published != program["outputs"]:
        raise ValueError("public boundary capture did not preserve exactly the original live-out boundary")
    from merlin.targetgen import golden_store

    golden = golden_store.load_golden(directory)
    if golden.get("golden_source") != "host_torch_eager" or len(golden.get("outputs") or {}) != len(program["outputs"]):
        raise ValueError("public boundary lacks every fresh host eager reference output")
    import numpy as np

    results = {row["name"]: row for row in program["nodes"]}
    for name, value in zip(program["outputs"], golden["outputs"].values(), strict=True):
        if list(np.asarray(value).shape) != results[name]["shape"]:
            raise ValueError("public boundary numerical reference has a different output geometry")
    result = {"status": "fresh_source_reference_prepared", "physical_mapping": "UNKNOWN", "timing": "UNKNOWN"}
    if functional:
        result["stages"] = {"generated": True, "evaluated": True, "admitted": False, "executed": False}
    return result
