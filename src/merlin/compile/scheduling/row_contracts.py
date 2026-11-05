"""Bind complete-row scheduling to selected numerical and resource contracts.

The source algorithm is matched, never selected from a hardware lane count.
RTL bounds storage, chunks and integer representation. Software declares each
primitive's complete operands and effects. Execution still needs independent
qualification, including the exponential and ordered rounded arithmetic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from math import lcm

from merlin.sched.ir import NULL, Call, Const
from merlin.sched.ir.kernel import DTYPE_BYTES
from merlin.targetgen.isa_spec import resolve_fact_integer

from .row_softmax import (
    PHASE_ROLES,
    ROLES,
    CausalRowMask,
    RowSoftmaxAdapter,
    RowSoftmaxNumerics,
    RowSoftmaxProblem,
    RowSoftmaxResources,
    RowSoftmaxStep,
    RowSoftmaxUnavailable,
)

READ_ROLES = ((), (0,), (0,), (), (1, 2), (2,), (1, 3))
WRITE_ROLES = ((0,), (0,), (1,), (2,), (2,), (3,), ())


def _fields(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"complete row {label}: expected exactly {sorted(keys)}")
    return value


def _tree(value, depth=0):
    if depth > 256:
        raise ValueError("complete row sum tree exceeds its structural budget")
    if type(value) is int:
        return value
    if type(value) is list and len(value) == 2:
        return tuple(_tree(child, depth + 1) for child in value)
    raise ValueError("complete row tree requires ordered binary additions")


@dataclass(frozen=True)
class SelectedRowContract:
    declarations_json: str

    def bind(self, selected, problem, numerics, resources, banks, role_banks, adapter, *,
             reservations, private_ownership):
        row = json.loads(self.declarations_json)
        if (type(problem) is not RowSoftmaxProblem or type(numerics) is not RowSoftmaxNumerics
                or type(adapter) is not RowSoftmaxAdapter):
            raise ValueError("complete row requires typed current source and execution contracts")
        algorithm = {**row["algorithm"], "tree": _tree(row["algorithm"]["tree"])}
        if (numerics != RowSoftmaxNumerics(**algorithm) or numerics.source_policy != selected.numerical_policy
                or adapter.configuration_identity != selected.hardware_identity
                or adapter.numerical_identity != numerics.identity or not adapter.qualification_identity):
            raise ValueError("complete row source or adapter disagrees with selected numerical/hardware contracts")
        expected = {"completion": "synchronous", "order": "source", "state": "addressable"}
        if any(row["support"][key] != value for key, value in expected.items()):
            raise RowSoftmaxUnavailable("selected row interface lacks synchronous source order or addressable state")
        mask = "causal_row" if problem.mask is not None else "none"
        if mask not in row["support"]["masks"]:
            raise RowSoftmaxUnavailable("selected row interface does not implement the complete source mask")
        limits = row["limits"]
        if any(value is None for value in limits.values()):
            raise RowSoftmaxUnavailable("selected row RTL chunk/granularity/index limits are unknown")
        if adapter.index_bits != limits["index_bits"]:
            raise ValueError("complete row adapter index width disagrees with selected RTL")
        declared = RowSoftmaxResources(limits["max_chunk_elements"], limits["granularity"],
                                       row["support"]["partial_chunks"])
        if resources is not None:
            if type(resources) is not RowSoftmaxResources:
                raise ValueError("complete row search controls require typed resource budgets")
            declared = RowSoftmaxResources(min(resources.max_chunk_elements, declared.max_chunk_elements),
                                           lcm(resources.granularity, declared.granularity),
                                           resources.partial_chunks and declared.partial_chunks,
                                           resources.max_static_calls)
        # The existing row allocator spells its byte unit 'byte'. Normalize
        # that explicit unit at this boundary; no word-address inference occurs.
        if banks is not None:
            if type(banks) is not tuple or any(bank.unit != "byte" for bank in banks):
                raise ValueError("complete row bank controls must use the existing byte unit")
            banks = tuple(replace(bank, unit="bytes") for bank in banks)
        banks, reservations, _windows = selected.storage.allocation_model(
            banks=banks, reservations=reservations, external_windows=None, private_ownership=private_ownership)
        banks = tuple(replace(bank, unit="byte") for bank in banks)
        roles = tuple(row["roles"][name] for name in ROLES)
        if role_banks is not None and role_banks != roles:
            raise ValueError("complete row state roles disagree with selected software physical mapping")
        external = {window.resource: window for window in selected.storage.external}
        required = problem.rows * problem.columns * DTYPE_BYTES[numerics.dtype]
        if any(external[name].start != 0 or external[name].size < required for name in row["external"].values()):
            raise RowSoftmaxUnavailable("complete row arguments exceed selected zero-based logical byte windows")
        windows = {window.resource: window for window in selected.storage.banks}
        instructions = {item["instruction"]: (kind, item) for kind, item in row["instructions"].items()}

        def project(call):
            if type(call) is not Call or call.instr not in instructions:
                raise ValueError("complete row actual instruction has no selected phase semantics")
            kind, item = instructions[call.instr]
            values = dict(call.args)
            if len(values) != len(call.args) or set(values) != set(item["arguments"]):
                raise ValueError("complete row actual operands disagree with selected ISA bindings")
            def constant(actual, expected):
                return (actual is NULL if expected is None else
                        type(actual) is Const and type(actual.value) is int and actual.value == expected)
            for name, value in item["constants"].items():
                if not constant(values[name], value):
                    raise ValueError("complete row actual instruction changed a declared constant/mode")
            bindings = item["bindings"]
            def integer(key):
                value = values[bindings[key]]
                if type(value) is not Const or type(value.value) is not int:
                    raise ValueError("complete row actual count/mask must be a constant integer")
                return value.value
            period, offset = integer("period"), integer("offset")
            if not period and offset:
                raise ValueError("complete row unmasked instruction has an offset")
            local = []
            for role, binding in zip(PHASE_ROLES[kind], item["local"], strict=True):
                selector = binding["selector"]
                if selector is not None and not constant(values[selector["argument"]], selector["value"]):
                    raise ValueError("complete row actual local-memory selector changed")
                local.append((windows[roles[role]].backing, values[binding["address"]]))
            actual = RowSoftmaxStep(kind, values[bindings["row"]], values[bindings["column"]], integer("count"),
                                    tuple(local), values[bindings["source"]] if "source" in bindings else None,
                                    values[bindings["output"]] if "output" in bindings else None,
                                    numerics.identity, CausalRowMask(period, offset) if period else None)
            if adapter.project(call) != actual:
                raise ValueError("provider row projection disagrees with selected actual operands/semantics")
            return actual

        return declared, banks, roles, reservations, replace(adapter, project=project)


def resolve_row_contract(declaration, facts, isa, storage, policy):
    row = _fields(declaration, ("schema", "algorithm", "element_bits", "limits", "support", "roles", "external",
                                "instructions"), "contract")
    if row["schema"] != "merlin.complete_row_contract.v1" or storage is None:
        raise ValueError("complete row scheduling requires its versioned contract and selected storage windows")
    algorithm = _fields(row["algorithm"], RowSoftmaxNumerics.__dataclass_fields__, "algorithm")
    numerics = RowSoftmaxNumerics(**{**algorithm, "tree": _tree(algorithm["tree"])})
    bits = resolve_fact_integer(row["element_bits"], facts, label="complete row element bits", minimum=1)
    if bits != DTYPE_BYTES[numerics.dtype] * 8 or numerics.source_policy != policy:
        raise ValueError("complete row software dtype/policy disagrees with selected RTL/numerical policy")
    limits = _fields(row["limits"], ("max_chunk_elements", "granularity", "index_bits"), "limits")
    resolved = {key: (None if value is None else resolve_fact_integer(value, facts, label="row " + key, minimum=1))
                for key, value in limits.items()}
    if resolved["index_bits"] is not None and resolved["index_bits"] not in (32, 64):
        raise ValueError("complete row requires a supported 32/64-bit index representation")
    support = _fields(row["support"], ("partial_chunks", "completion", "order", "state", "masks"), "support")
    if (type(support["partial_chunks"]) is not bool
            or any(type(support[k]) is not str or not support[k] for k in ("completion", "order", "state"))
            or type(support["masks"]) is not list or len(support["masks"]) > 2
            or any(m not in ("none", "causal_row") for m in support["masks"])):
        raise ValueError("complete row execution, tail and mask support must be explicit")
    roles = _fields(row["roles"], ROLES, "state roles")
    banks = {window.resource for window in storage.banks}
    if any(type(name) is not str or name not in banks for name in roles.values()):
        raise ValueError("complete row state requires declared private memory banks")
    external = _fields(row["external"], ("input", "output"), "arguments")
    if any(type(name) is not str or name not in {w.resource for w in storage.external} for name in external.values()):
        raise ValueError("complete row arguments require selected external memory windows")
    instructions = _fields(row["instructions"], PHASE_ROLES, "phases")
    normalized, used = {}, set()
    for ordinal, (kind, _role_ids) in enumerate(PHASE_ROLES.items()):
        item = _fields(instructions[kind], ("instruction", "bindings", "local", "constants"), "phase")
        instruction = isa.instruction(item["instruction"])
        if instruction.name in used:
            raise ValueError("complete row phases require distinct declared instruction identities")
        used.add(instruction.name)
        semantics = json.loads(instruction.semantics_json)
        if (semantics.get("primitive") != "merlin.complete_row." + kind
                or semantics.get("parameters") != {"numerical_identity": numerics.identity,
                                                   "completion": support["completion"]}):
            raise ValueError("complete row ISA primitive/algorithm/completion semantics disagree")
        required = {"row", "column", "count", "period", "offset"}
        if kind in ("maximum", "exponential"):
            required.add("source")
        if kind == "normalize":
            required.add("output")
        bindings = _fields(item["bindings"], required, "primitive bindings")
        if semantics.get("bindings") != bindings:
            raise ValueError("complete row ISA and phase operand bindings disagree")
        args = {argument.name: argument for argument in instruction.arguments()}
        encoded = {field.argument: field.width for field in instruction.fields if field.argument is not None}
        names = []
        def argument(name, *, pointer=None, narrow=False, byte_address=False):
            value = args.get(name) if type(name) is str else None
            if value is None or name not in encoded:
                raise ValueError("complete row binding must name an encoded typed ISA argument")
            bits = min(value.type.bits, encoded[name])
            if pointer is not None:
                if value.type.kind != "address" or value.type.space != external[pointer] or value.unit != "bytes":
                    raise ValueError("complete row external operand requires its typed byte address space")
            elif (value.type.kind != "integer" or not narrow and resolved["index_bits"] is not None
                  and bits < resolved["index_bits"]):
                raise ValueError("complete row integer operand is narrower than its checked index domain")
            if byte_address and value.unit != "bytes":
                raise ValueError("complete row local addresses require explicit byte units")
            if (pointer is not None or byte_address) and value.alignment != 1:
                raise RowSoftmaxUnavailable("complete row ISA operand alignment requires a separate mapping contract")
            names.append(name)
            return value
        for key, name in bindings.items():
            argument(name, pointer="input" if key == "source" else "output" if key == "output" else None)
        if type(item["local"]) is not list or len(item["local"]) != len(PHASE_ROLES[kind]):
            raise ValueError("complete row local bindings must cover each phase state operand")
        local = []
        for binding in item["local"]:
            _fields(binding, ("address", "selector"), "local operand")
            argument(binding["address"], byte_address=True)
            selector = binding["selector"]
            if selector is not None:
                _fields(selector, ("argument", "value"), "local selector")
                arg = argument(selector["argument"], narrow=True)
                number = resolve_fact_integer(selector["value"], facts, label="row memory selector")
                if number >= 1 << (min(arg.type.bits, encoded[selector["argument"]]) - int(arg.type.signed)):
                    raise ValueError("complete row memory selector exceeds its ISA representation")
                selector = {"argument": selector["argument"], "value": number}
            local.append({**binding, "selector": selector})
        if type(item["constants"]) is not dict:
            raise ValueError("complete row constant/mode bindings must be explicit")
        for name, value in item["constants"].items():
            arg = args.get(name)
            if arg is None or name not in encoded:
                raise ValueError("complete row constant must name an encoded ISA argument")
            bits = min(arg.type.bits, encoded[name])
            low = -(1 << (bits - 1)) if arg.type.signed else 0
            high = 1 << (bits - int(arg.type.signed))
            if value is None:
                if arg.type.kind != "address":
                    raise ValueError("complete row null constants require a typed address")
            elif type(value) is not int or arg.type.kind != "integer" or not low <= value < high:
                raise ValueError("complete row constant is outside its typed ISA representation")
            names.append(name)
        if len(names) != len(set(names)) or set(names) != set(args):
            raise ValueError("complete row bindings must cover every actual ISA operand exactly once")
        reads = {roles[ROLES[r]] for r in READ_ROLES[ordinal]}
        writes = {roles[ROLES[r]] for r in WRITE_ROLES[ordinal]}
        if "source" in bindings:
            reads.add(external["input"])
        if "output" in bindings:
            writes.add(external["output"])
        effects = json.loads(instruction.effects_json)
        if effects == {"status": "unknown"} or set(effects["reads"]) != reads or set(effects["writes"]) != writes:
            raise ValueError("complete row ISA effects disagree with complete phase memory effects")
        normalized[kind] = {**item, "local": local, "arguments": list(args)}
    return SelectedRowContract(json.dumps({**row, "limits": resolved, "instructions": normalized},
                                          sort_keys=True, separators=(",", ":"), allow_nan=False))
