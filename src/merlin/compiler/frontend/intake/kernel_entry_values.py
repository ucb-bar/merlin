"""Independently check storage-backed source value port declarations.

The actual bound TTIR and standard payload still execute. These checks bind
the new port roster to original source types and declared storage, and grant
no numeric, alias, source-equivalence or target semantics.
"""

from xdsl.dialects import builtin
from xdsl.parser import Parser


def checked_entry_values(configuration, binding, spec, generic):
    declared = binding.get("bindings", [])
    if not isinstance(declared, list) or any(not isinstance(row, dict) for row in declared):
        raise ValueError("entry source bindings require their complete record list")
    values = [row for row in declared if row.get("kind") == "value"]
    if configuration["schema"] != "t2m.source-allocation-arguments.v3":
        if values or "result_bindings" in binding:
            raise ValueError("entry value ports require the explicit v3 source configuration")
        return
    if (set(configuration) != {"schema", "source_entry", "arguments", "results"}
            or not isinstance(configuration["results"], list)):
        raise ValueError("entry value configuration lacks its complete argument/result roster")
    from merlin.compiler.frontend.linalg_mlir import make_context

    module = Parser(make_context(), generic.decode()).parse_module()
    functions = [op for op in module.ops
                 if op.properties.get("sym_name") == builtin.StringAttr(binding["source_entry"])]
    if len(functions) != 1 or "function_type" not in functions[0].properties:
        raise ValueError("entry value ports lack their original typed source function")
    signature = functions[0].properties["function_type"]
    if len(declared) != len(signature.inputs):
        raise ValueError("entry value source arguments differ from the complete original signature")
    roots = {root["name"]: root for root in spec["buffers"]}
    scalars = {scalar["name"]: scalar for scalar in spec.get("runtime_scalars", [])}

    def port(raw, *, result):
        if (not isinstance(raw, dict) or "allocation" not in raw
                or set(raw) - {"allocation", "offset", "per_program"}):
            raise ValueError("entry value port has unknown or missing fields")
        record = {"allocation": raw["allocation"], "offset": raw.get("offset", 0),
                  "per_program": raw.get("per_program", result)}
        if (type(record["allocation"]) is not str or record["allocation"] not in roots
                or type(record["per_program"]) is not bool
                or not (type(record["offset"]) is int and -(2**63) <= record["offset"] < 2**63
                    or type(record["offset"]) is str and record["offset"] in scalars
                    and scalars[record["offset"]]["dtype"] in ("i32", "i64"))):
            raise ValueError("entry value port has no exact allocation/origin/program selection")
        return record

    def typed(record, type_, *, result):
        shaped = isinstance(type_, builtin.TensorType)
        element = type_.element_type if shaped else type_
        root = roots[record["allocation"]]
        if (str(element) != root["dtype"] or not isinstance(element, (builtin.IntegerType, builtin.AnyFloat))
                or result and root["effect"] == "read" or not result and root["effect"] == "write"
                or shaped and any(n <= 0 for n in type_.get_shape())):
            raise ValueError("entry value source type differs from its numeric storage/effect")
        return {**record, "source_type": str(type_)}

    requested = []
    for argument in configuration["arguments"]:
        if isinstance(argument, dict) and argument.get("kind") == "value":
            requested.append(port({k: v for k, v in argument.items() if k != "kind"}, result=False))
    observed = []
    for declaration, type_ in zip(declared, signature.inputs, strict=True):
        if declaration.get("kind") == "value":
            raw = {k: v for k, v in declaration.items() if k not in ("kind", "source_type")}
            expected = typed(port(raw, result=False), type_, result=False)
            if declaration != {"kind": "value", **expected}:
                raise ValueError("entry value argument differs from its actual source type")
            observed.append({k: v for k, v in expected.items() if k != "source_type"})
    if observed != requested:
        raise ValueError("entry value arguments differ from their configured source order")
    results = configuration["results"]
    if len(results) != len(signature.outputs):
        raise ValueError("entry result ports omit or add an original source result")
    expected = [typed(port(raw, result=True), type_, result=True)
                for raw, type_ in zip(results, signature.outputs, strict=True)]
    if binding.get("result_bindings", []) != expected or not values and not expected:
        raise ValueError("entry result ports differ from their original types/configuration")
    names = [row["allocation"] for row in expected]
    pointers = {row["allocation"] for row in declared if row.get("kind") == "pointer"}
    if not pointers <= roots.keys():
        raise ValueError("entry pointer binding names an undeclared storage root")
    if len(set(names)) != len(names) or set(names) & pointers:
        raise ValueError("entry result ports must retain every result apart from source pointer state")
    mutable = {name for name in pointers if roots[name]["effect"] != "read"}
    if {row["allocation"] for row in observed} & (mutable | set(names)):
        raise ValueError("entry value inputs must stay independent of mutable pointer state and result ports")
