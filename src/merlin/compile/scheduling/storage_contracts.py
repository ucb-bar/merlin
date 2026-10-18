"""Bind private allocation to RTL windows and complete software memory effects.

This is admission and actual-operand projection for the existing allocator, not
another allocation algorithm. Runtime ownership and target execution still need
independent qualification. Source addresses may be virtual before relocation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from math import prod

from merlin.sched.ir import Call, evaluate
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank
from merlin.targetgen.isa_dialect_gen import encode_instruction
from merlin.targetgen.isa_spec import resolve_fact_integer

from .program_limits import AddressWindow, EncodedAccess
from .temporary_storage import StorageAdapter, StorageEffects, StorageOptimizationUnavailable


def _fields(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"storage {label}: expected exactly {sorted(keys)}")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def binding_value(binding, values):
    """Evaluate an already resolved byte/count binding over actual operands."""
    if "product" in binding:
        factors = [binding_value(factor, values) for factor in binding["product"]]
        if any(value <= 0 for value in factors):
            raise ValueError("logical element count dimensions must be positive")
        return binding["scale"] * prod(factors)
    return values[binding["argument"]] if "argument" in binding else binding["constant"]


def resolve_operand_binding(value, instruction, *, address_spaces=None):
    """Select an encoded typed operand; never infer a pointer unit."""
    if type(value) is not dict or len(value) != 1:
        raise ValueError("storage operand binding needs one encoded argument or integer constant")
    if "constant" in value and type(value["constant"]) is int:
        return value
    args = {a.name: a for a in instruction.arguments()}
    encoded = {f.argument for f in instruction.fields if f.argument is not None}
    if "argument" in value and type(value["argument"]) is str and value["argument"] in encoded:
        argument = args[value["argument"]]
        if argument.type.kind == "integer":
            return value
        if (argument.type.kind == "address" and address_spaces == {argument.type.space}
                and argument.unit == "bytes"):
            return value
    raise ValueError("storage operand binding must name an encoded integer or compatible byte address argument")


def resolve_element_count(factors, instruction):
    """A bounded product of explicit logical dimensions, not a scalar language."""
    if type(factors) is not list or not factors or len(factors) > 16:
        raise ValueError("element count requires one to sixteen explicit integer factors")
    resolved = [resolve_operand_binding(factor, instruction) for factor in factors]
    if any("constant" in factor and factor["constant"] <= 0 for factor in resolved):
        raise ValueError("logical element count dimensions must be positive")
    return resolved


def _space(binding, values):
    if "resource" in binding:
        return binding["resource"]
    selector = values[binding["argument"]]
    for item in binding["choices"]:
        if selector == item["value"]:
            return item["resource"]
    raise ValueError("storage memory selector is outside its selected software/RTL domain")


@dataclass(frozen=True)
class StorageWindow:
    resource: str
    backing: str
    start: int
    size: int
    alignment: int
    capacity: int | None
    ownership: str


@dataclass(frozen=True)
class TemporaryStorageContract:
    banks: tuple[StorageWindow, ...]
    external: tuple[StorageWindow, ...]
    reservations: tuple[Reservation, ...]
    declarations_json: str

    def allocation_model(self, *, banks, reservations, external_windows, private_ownership):
        """Shared geometry/ownership admission, independent of instruction IR."""
        if private_ownership is not True or any(w.ownership != "invocation_private" for w in self.banks):
            raise StorageOptimizationUnavailable("selected storage lacks invocation-private runtime ownership")
        if any(w.ownership not in ("runtime_managed", "persistent") for w in self.external):
            raise StorageOptimizationUnavailable("selected storage lacks fixed external runtime ownership")
        if any(w.capacity is None for w in self.banks):
            raise StorageOptimizationUnavailable("selected storage RTL memory capacity is unknown")
        derived = tuple(StorageBank(w.resource, w.backing, w.start + w.size, "bytes", w.alignment, w.start)
                        for w in self.banks)
        if banks is None:
            banks = derived
        if (type(banks) is not tuple or any(type(b) is not StorageBank for b in banks)
                or len(banks) != len(derived) or {b.name for b in banks} != {b.name for b in derived}):
            raise ValueError("selected storage requires the declared typed memory banks")
        selected = {b.name: b for b in derived}
        for bank in banks:
            bound = selected[bank.name]
            if (bank.backing != bound.backing or bank.unit != "bytes" or bank.start < bound.start
                    or bank.capacity > bound.capacity or bank.alignment % bound.alignment):
                raise ValueError("storage bank disagrees with selected RTL window, alignment or physical alias")
        permitted = tuple(AddressWindow(w.resource, w.start, w.size) for w in self.external)
        if external_windows is None:
            external_windows = permitted
        if type(external_windows) is not tuple or any(type(w) is not AddressWindow for w in external_windows):
            raise ValueError("selected storage needs explicit immutable external address windows")
        for window in external_windows:
            if not any(w.space == window.space and w.start <= window.start
                       and window.start + window.size <= w.start + w.size for w in permitted):
                raise ValueError("runtime external window exceeds its selected software/RTL address window")
        if type(reservations) is not tuple or any(type(r) is not Reservation for r in reservations):
            raise ValueError("selected storage reservations must be typed byte intervals")
        reservations = _reservations((*self.reservations, *reservations), banks)
        return banks, reservations, external_windows

    def bind(self, isa, adapter, *, banks, reservations, external_windows, private_ownership):
        """Derive geometry and check existing runtime/allocation premises."""
        if type(adapter) is not StorageAdapter or not adapter.qualification_identity:
            raise ValueError("selected storage requires an independently qualified effect/rewrite adapter")
        banks, reservations, external_windows = self.allocation_model(
            banks=banks, reservations=reservations, external_windows=external_windows,
            private_ownership=private_ownership,
        )
        declarations = json.loads(self.declarations_json)

        def project(call):
            if type(call) is not Call or call.instr not in declarations:
                return None  # Unknown operations are conservative barriers.
            if call.unit or call.produces or call.awaits or call.stages or call.sets or call.assumes:
                return None
            try:
                values = {key: evaluate(value, {}) for key, value in call.args}
            except ValueError:
                return None
            if len(values) != len(call.args):
                raise ValueError("selected storage operands contain duplicate bindings")
            encode_instruction(isa, call.instr, values)
            declaration = declarations[call.instr]
            accesses = tuple(EncodedAccess(_space(a["space"], values), binding_value(a["start"], values),
                                           binding_value(a["size"], values), a["writes"])
                             for a in declaration["accesses"])
            completion = declaration["completion"]
            if completion == "unknown":
                return None
            token = None if declaration["token"] is None else str(binding_value(declaration["token"], values))
            waits = tuple(str(binding_value(binding, values)) for binding in declaration["waits"])
            effect = adapter.project(call)
            if type(effect) is not StorageEffects:
                return None
            if (effect.accesses != accesses or effect.completion != completion or effect.token != token
                    or effect.waits != waits or effect.exact_copy != declaration["exact_copy"]):
                raise ValueError("provider storage effects disagree with selected operands or software completion")
            if effect.elementwise is not None and declaration["nonmemory_effects"]:
                raise ValueError("selected storage cannot treat extra execution effects as pure elementwise arithmetic")
            addresses = {a["start"]["argument"] for a in declaration["accesses"] if "argument" in a["start"]}
            preserved = {key: value for key, value in values.items() if key not in addresses}
            # Every nonaddress argument and the selected operation semantics
            # survive relocation, even if a provider accidentally omits them.
            key = hashlib.sha256(_json((call.instr, declaration["semantics"], preserved,
                                       effect.semantic_key)).encode()).hexdigest()
            return replace(effect, semantic_key=key)

        return banks, reservations, external_windows, replace(adapter, project=project)


def _reservations(rows, banks):
    """Union overlapping typed reservations in canonical backing coordinates."""
    by_name = {b.name: b for b in banks}
    for r in rows:
        bank = by_name.get(r.storage)
        if bank is None or r.start < bank.start or r.start + r.extent > bank.capacity:
            raise ValueError("selected reservation exceeds its declared memory view")
    result, ends = [], {}
    for r in sorted(rows, key=lambda r: (by_name[r.storage].backing, r.start, -r.extent, r.storage)):
        backing = by_name[r.storage].backing
        start = max(r.start, ends.get(backing, r.start))
        end = r.start + r.extent
        if start < end:
            result.append(Reservation(r.storage, start, end - start))
        ends[backing] = max(end, ends.get(backing, end))
    return tuple(result)


def resolve_storage_contract(declaration, facts, isa):
    row = _fields(declaration, ("schema", "banks", "external", "reservations", "instructions"), "contract")
    if row["schema"] != "merlin.temporary_storage_contract.v1":
        raise ValueError("unknown temporary storage contract version")
    resources = {r.name: r for r in isa.resources}

    def windows(section, local):
        if type(section) is not dict or not section or len(section) > 256:
            raise ValueError("storage windows must be bounded nonempty resource mappings")
        result = []
        for name, item in section.items():
            _fields(item, ("backing", "start", "size", "ownership", "alignment") if local
                    else ("backing", "start", "size", "ownership"), "window")
            resource = resources.get(name) if type(name) is str else None
            if resource is None or resource.kind != "memory" or resource.unit != "bytes":
                raise ValueError("storage windows require ISA byte memory resources")
            if type(item["backing"]) is not str or not item["backing"]:
                raise ValueError("storage windows require canonical physical backing identities")
            if item["ownership"] not in ("invocation_private", "runtime_managed", "persistent", "unknown"):
                raise ValueError("storage ownership must be explicitly declared or unknown")
            start = resolve_fact_integer(item["start"], facts, label="storage window start")
            size = resolve_fact_integer(item["size"], facts, label="storage window size", minimum=1)
            alignment = (resolve_fact_integer(item["alignment"], facts, label="storage alignment", minimum=1)
                         if local else 1)
            if resource.capacity is not None and start + size > resource.capacity:
                raise ValueError("storage window exceeds its selected RTL resource capacity")
            result.append(StorageWindow(name, item["backing"], start, size, alignment,
                                        resource.capacity, item["ownership"]))
        return tuple(result)

    banks, external = windows(row["banks"], True), windows(row["external"], False)
    if {w.resource for w in banks} & {w.resource for w in external}:
        raise ValueError("private and external storage resources must be distinct")
    if {w.backing for w in banks} & {w.backing for w in external}:
        raise ValueError("private storage cannot alias external or persistent runtime state")
    if len({w.backing for w in external}) != len(external):
        raise ValueError("distinct external resource aliases need a qualified canonical projection before allocation")
    by_name = {w.resource: w for w in (*banks, *external)}
    if type(row["reservations"]) is not list or len(row["reservations"]) > 4096:
        raise ValueError("storage reservations must be a bounded list")
    reservations = []
    for item in row["reservations"]:
        _fields(item, ("resource", "start", "size", "purpose"), "reservation")
        if (type(item["resource"]) is not str or item["resource"] not in {w.resource for w in banks}
                or type(item["purpose"]) is not str or not item["purpose"]):
            raise ValueError("storage reservation requires a local resource and explicit purpose")
        reservations.append(Reservation(
            item["resource"], resolve_fact_integer(item["start"], facts, label="storage reservation start"),
            resolve_fact_integer(item["size"], facts, label="storage reservation size", minimum=1),
        ))
    instructions = row["instructions"]
    if type(instructions) is not dict or len(instructions) > 4096:
        raise ValueError("storage instructions must be a bounded mapping")
    normalized = {}
    for name, declaration in instructions.items():
        instruction = isa.instruction(name)
        item = _fields(declaration, ("accesses", "completion", "token", "waits", "exact_copy"), "effects")
        if (item["completion"] not in ("synchronous", "asynchronous", "wait", "unknown")
                or type(item["exact_copy"]) is not bool):
            raise ValueError("storage completion and copy semantics must be explicit")
        if (type(item["accesses"]) is not list or len(item["accesses"]) > 256
                or type(item["waits"]) is not list or len(item["waits"]) > 256):
            raise ValueError("storage accesses and completion waits must be bounded lists")

        def binding(value, *, address_spaces=None):
            return resolve_operand_binding(value, instruction, address_spaces=address_spaces)

        def space(value):
            if type(value) is dict and set(value) == {"resource"} and type(value["resource"]) is str:
                choices = [(None, value["resource"])]
                resolved = value
            else:
                _fields(value, ("argument", "choices"), "memory selector")
                binding({"argument": value["argument"]})
                if type(value["choices"]) is not list or not value["choices"] or len(value["choices"]) > 256:
                    raise ValueError("storage memory selector requires bounded explicit resource choices")
                choices = []
                for choice in value["choices"]:
                    _fields(choice, ("value", "resource"), "memory choice")
                    choices.append((resolve_fact_integer(choice["value"], facts, label="storage memory selector"),
                                    choice["resource"]))
                if len({number for number, _ in choices}) != len(choices):
                    raise ValueError("storage memory selector has duplicate encodings")
                resolved = {"argument": value["argument"],
                            "choices": [dict(value=number, resource=resource) for number, resource in choices]}
            if any(type(resource) is not str or resource not in by_name for _, resource in choices):
                raise ValueError("storage accesses require declared private or external memory windows")
            return resolved, {resource for _, resource in choices}

        accesses, coverage = [], {"reads": set(), "writes": set()}
        for access in item["accesses"]:
            _fields(access, ("space", "start", "size", "writes"), "access")
            if type(access["writes"]) is not bool:
                raise ValueError("storage access direction must be Boolean")
            resolved, names = space(access["space"])
            start_binding = binding(access["start"], address_spaces=names)
            size = access["size"]
            if type(size) is dict and set(size) == {"elements", "element_bits"}:
                bits = resolve_fact_integer(size["element_bits"], facts, label="storage element bits", minimum=1)
                if bits % 8:
                    raise ValueError("storage element widths must have a complete byte representation")
                size_binding = {"product": resolve_element_count(size["elements"], instruction), "scale": bits // 8}
            else:
                size_binding = binding(size)
            if "constant" in start_binding and names & {w.resource for w in banks}:
                raise ValueError("private storage addresses must be explicit relocatable ISA arguments")
            accesses.append({**access, "space": resolved, "start": start_binding, "size": size_binding})
            coverage["writes" if access["writes"] else "reads"].update(names)
        for value in item["waits"]:
            binding(value)
        if item["token"] is not None:
            binding(item["token"])
        completion = item["completion"]
        if ((completion == "asynchronous") != (item["token"] is not None)
                or (completion == "wait") != bool(item["waits"]) or completion == "wait" and accesses):
            raise ValueError("storage software tokens must describe actual launches or pure completion waits")
        effects, semantics = json.loads(instruction.effects_json), json.loads(instruction.semantics_json)
        if effects == {"status": "unknown"} or semantics == {"status": "unknown"}:
            raise ValueError("storage instructions require complete ISA effects and explicit operation semantics")
        for kind in ("reads", "writes"):
            if (any(resources[r].kind == "state" for r in effects[kind])
                    or {r for r in effects[kind] if resources[r].kind == "memory"} != coverage[kind]):
                raise ValueError("storage access coverage disagrees with ISA memory or persistent-state effects")
        nonmemory = any(resources[r].kind != "memory" for kind in ("reads", "writes") for r in effects[kind])
        if item["exact_copy"] and nonmemory:
            raise ValueError("storage exact copy cannot discard additional ISA execution or state effects")
        if item["exact_copy"] and (
            completion not in ("synchronous", "asynchronous") or len(accesses) != 2
            or accesses[0]["writes"] or not accesses[1]["writes"]
        ):
            raise ValueError("storage exact copy requires one read and write with explicit completion")
        normalized[name] = {**item, "accesses": accesses, "semantics": semantics, "nonmemory_effects": nonmemory}
    return TemporaryStorageContract(banks, external, tuple(reservations), _json(normalized))
