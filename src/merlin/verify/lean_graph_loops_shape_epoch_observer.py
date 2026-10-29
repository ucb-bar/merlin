"""Bind a concrete caller-memory observer and complete top-level Code sequences.

No function-name allowlist or whole-call assumption is used: every recognized
callee must contain precisely the retained constant/load/binary/store/return
operations with the same scoped typed operands. Arithmetic stays an unchanged
partial relation. This is a new adapter leaf outside standalone qualification.
"""

from __future__ import annotations

import copy
import json

from .lean_graph_loops_shape_model import binding, bindings, code
from .lean_graph_loops_unroll import _list, _nat, _string


def sequence(function):
    atoms, current = [], function["code"]
    while current["tag"] != "returned":
        if current["tag"] not in ("constant", "action", "region"):
            raise ValueError("ordinary normalized Code retains a descriptor rewrite")
        atoms.append({key: copy.deepcopy(value) for key, value in current.items() if key != "rest"})
        current = current["rest"]
    return dict(function=function["name"], atoms=atoms, returned=current["values"])


def observer(function):
    row = sequence(function)
    if len(row["atoms"]) != 4 or len(function["parameters"]) != 2:
        raise ValueError("caller observer needs the complete constant/load/binary/store body")
    zero, load, binary, store = row["atoms"]
    if zero["tag"] != "constant" or zero["value"] != 0:
        raise ValueError("observer index is not the actual zero constant")
    for op, expected in ((load, "memref.load"), (binary, "arith.addi"), (store, "memref.store")):
        if op["tag"] != "action" or json.loads(op["key"])["name"] != expected:
            raise ValueError("observer operation or order changed")
    if len(load["results"]) != 1 or len(binary["results"]) != 1 or store["results"]:
        raise ValueError("observer result arity changed")
    argument, buffer = function["parameters"]
    loaded, result = load["results"][0], binary["results"][0]
    if (load["operands"], binary["operands"], store["operands"], row["returned"]) != (
        [buffer, zero["result"]],
        [loaded, argument],
        [result, buffer, zero["result"]],
        [result],
    ):
        raise ValueError("observer capture, caller address, arithmetic order or return changed")
    if argument["kind"]["tag"] != "scalar" or function["returned"] != [argument["kind"]]:
        raise ValueError("observer has unsupported input/result kinds")
    return dict(
        function=function["name"],
        loadKey=load["key"],
        binaryKey=binary["key"],
        storeKey=store["key"],
        input=argument,
        buffer=buffer,
        zero=zero["result"],
        loaded=loaded,
        result=result,
        element=argument["kind"]["type"],
    )


def bind_observers(adapter):
    module = adapter["normalized_candidate"]
    functions = {function["name"]: function for function in module["functions"]}
    targets = sorted({target for _, target in module["calls"]})
    if any(target not in functions for target in targets):
        raise ValueError("declared external call has no derived caller-observer body")
    return dict(
        adapter,
        observers=[observer(functions[target]) for target in targets],
        sequences=[sequence(function) for function in module["functions"]],
        full_epoch_public_adapter_proved=False,
    )


def atom_literal(atom):
    if atom["tag"] == "constant":
        return ".constant " + binding(atom["result"]) + " " + _nat(atom["value"])
    if atom["tag"] == "action":
        return ".action " + " ".join((_string(atom["key"]), bindings(atom["operands"]), bindings(atom["results"])))
    if atom["tag"] == "region":
        return ".region " + " ".join(
            (
                _string(atom["key"]),
                bindings(atom["operands"]),
                bindings(atom["parameters"]),
                bindings(atom["results"]),
                code(atom["body"]),
            )
        )
    raise ValueError("unknown complete sequence atom")


def observer_literal(row):
    fields = [_string(row[name]) for name in ("loadKey", "binaryKey", "storeKey")]
    fields.extend(binding(row[name]) for name in ("input", "buffer", "zero", "loaded", "result"))
    return "⟨" + ", ".join([*fields, _string(row["element"])]) + "⟩"


def clauses(adapter):
    result, roots = [], []
    for index, row in enumerate(adapter["sequences"]):
        name = "sequence" + str(index)
        result.append(f"""def {name} : List Epoch.Atom := {_list(row["atoms"], atom_literal)}
theorem {name}_syntax :
    (functionAt actualNormalized {_string(row["function"])}).map Function.code =
      some (Epoch.sequenceCode {name} {bindings(row["returned"])}) := by rfl
""")
        roots.append("GraphShapeEpochObligation." + name + "_syntax")
    for index, row in enumerate(adapter["observers"]):
        name = "observer" + str(index)
        result.append(f"""def {name} : Epoch.Touch := {observer_literal(row)}
theorem {name}_checked : decide {name}.Facts = true := by decide
theorem {name}_syntax : functionAt actualNormalized {_string(row["function"])} =
    some ({name}.function {_string(row["function"])}) := by rfl
theorem {name}_separate : {name}.input.name ≠ {name}.buffer.name := by decide
theorem {name}_call_keys : callTarget actualNormalized.calls {name}.loadKey = none ∧
    callTarget actualNormalized.calls {name}.binaryKey = none ∧
    callTarget actualNormalized.calls {name}.storeKey = none := by decide
""")
        roots.extend(
            "GraphShapeEpochObligation." + name + suffix
            for suffix in ("_checked", "_syntax", "_separate", "_call_keys")
        )
    return "\n".join(result), tuple(roots)
