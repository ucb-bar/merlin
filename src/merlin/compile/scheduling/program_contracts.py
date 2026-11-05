"""Bind final encoded-program checks to selected structural and software facts.

The existing finalizer owns counting and safe segmentation. This module binds
its limits and reconstructs effects from actual fixed-width ISA words, rather
than trusting a provider's request stream. Qualification of the instruction
implementation and launch persistence remains a separate runtime obligation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import lcm

from merlin.sched.ir import Call, Const
from merlin.targetgen.isa_dialect_gen import decode_instruction
from merlin.targetgen.isa_spec import resolve_fact_integer

from .program_limits import (
    AddressWindow,
    BranchRule,
    Continuation,
    DecodedInstruction,
    EncodedAccess,
    EncodedBranch,
    ProgramDecoder,
    ProgramLimits,
)
from .storage_contracts import _space, binding_value


def _fields(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"final program {label}: expected exactly {sorted(keys)}")
    return value


def _names(value, label):
    if (type(value) is not list or len(value) > 4096 or any(type(v) is not str or not v for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"final program {label}: requires bounded distinct names")
    return value


@dataclass(frozen=True)
class SelectedProgramContract:
    declarations_json: str

    @property
    def physical_spaces(self):
        return tuple((name, window["backing"]) for name, window in
                     json.loads(self.declarations_json)["windows"].items())

    def published(self):
        return tuple(EncodedAccess(name, w["start"], w["size"]) for name, w in
                     json.loads(self.declarations_json)["windows"].items() if w["visibility"] != "private")

    def calls(self, selected, image):
        width = selected.isa.word_bits // 8
        if selected.isa.word_bits % 8 or type(image) is not bytes or not image or len(image) % width:
            raise ValueError("selected final program needs complete byte-aligned ISA words")
        for index in range(len(image) // width):
            name, values = decode_instruction(selected.isa, int.from_bytes(
                image[index * width:(index + 1) * width], selected.instruction_byteorder))
            yield index * width, name, values

    def bind(self, selected, limits, decoder, continuation, qualification_identity):
        """Restrict user budgets and admit explicitly qualified launch semantics."""
        row = json.loads(self.declarations_json)
        if type(decoder) is not ProgramDecoder or decoder.configuration_identity != selected.hardware_identity:
            raise ValueError("final program decoder must match selected hardware and independent qualification")
        if limits is not None and type(limits) is not ProgramLimits:
            raise ValueError("final program controls require typed byte/count/address budgets")
        maximum = row["limits"]["max_bytes"]
        if maximum is None and limits is None:
            raise ValueError("final program byte capacity is unknown; supply an explicit code-growth budget")
        alignment = row["limits"]["instruction_alignment"]
        if alignment is None:
            raise ValueError("selected final instruction alignment is unknown")
        windows = tuple(AddressWindow(name, w["start"], w["size"]) for name, w in row["windows"].items())
        rules = tuple(BranchRule(name, r["minimum"], r["maximum"], r["scale"], r["origin"])
                      for name, r in row["branches"].items() if r["origin"] != "unknown")
        count = row["limits"]["max_instructions"]
        if limits is not None:
            maximum = min(limits.max_bytes, maximum) if maximum is not None else limits.max_bytes
            alignment = lcm(alignment, limits.instruction_alignment)
            count = min(c for c in (count, limits.max_instructions) if c is not None) if (
                count is not None or limits.max_instructions is not None) else None
            declared = {w.space: w for w in windows}
            narrowed = []
            for window in limits.windows:
                bound = declared.get(window.space)
                if bound is None or window.start < bound.start or window.start + window.size > bound.start + bound.size:
                    raise ValueError("final program address control exceeds a selected RTL/software window")
                narrowed.append(window)
            windows = tuple(narrowed)
            declared_rules = {r.name: r for r in rules}
            for rule in limits.branches:
                bound = declared_rules.get(rule.name)
                if (bound is None or rule.origin != bound.origin or rule.scale != bound.scale
                        or rule.minimum < bound.minimum or rule.maximum > bound.maximum):
                    raise ValueError("final program branch control disagrees with selected ISA reach/origin")
                declared_rules[rule.name] = rule
            rules = tuple(declared_rules.values())
        effective = ProgramLimits(maximum, count, alignment, windows, rules)
        launch = row["continuation"]
        if continuation is not None and type(continuation) is not Continuation:
            raise ValueError("final continuation controls require the existing typed capability")
        if qualification_identity is not None and (
            type(qualification_identity) is not str or not qualification_identity.strip()
        ):
            raise ValueError("final continuation qualification must be independently selected and nonempty")
        if continuation is not None:
            if qualification_identity is not None and qualification_identity != continuation.identity:
                raise ValueError("final continuation qualification disagrees with the supplied runtime capability")
            qualification_identity = continuation.identity
        unavailable = None
        if launch is None:
            if continuation is not None or qualification_identity is not None:
                raise ValueError("selected software declares no state-continuation interface")
            unavailable = "no_software_continuation"
        elif qualification_identity is None:
            unavailable = "missing_independent_runtime_qualification"
        elif not launch["sequential_completion"] or not launch["starts_at_zero"]:
            unavailable = "unsupported_launch_completion_or_entry"
        if unavailable is None:
            spaces = tuple(name for name, w in row["windows"].items() if w["backing"] in launch["preserved_backings"])
            configuration = tuple(launch["preserved_configuration"])
            segments = launch["max_segments"]
            if continuation is not None:
                if (set(continuation.preserved_spaces) - set(spaces)
                        or set(continuation.preserved_configuration) - set(configuration)):
                    raise ValueError("runtime continuation cannot expand selected memory/configuration persistence")
                # Preservation is physical: aliases of one backing cannot have
                # contradictory survival semantics at a launch boundary.
                retained = {row["windows"][name]["backing"] for name in continuation.preserved_spaces}
                if any((name in continuation.preserved_spaces) != (w["backing"] in retained)
                       for name, w in row["windows"].items()):
                    raise ValueError("runtime continuation must preserve all views of a physical backing")
                spaces, configuration = continuation.preserved_spaces, continuation.preserved_configuration
                segments = min(segments, continuation.max_segments)
            continuation = Continuation(qualification_identity, spaces, True, True, segments, configuration)
        else:
            continuation = None

        storage = json.loads(selected.storage.declarations_json) if selected.storage is not None else {}
        width = selected.isa.word_bits // 8
        decoder_state = decoder.configuration_identity, decoder.qualification_identity, id(decoder.decode)
        def fresh_decoder():
            if decoder_state != (decoder.configuration_identity, decoder.qualification_identity, id(decoder.decode)):
                raise ValueError("final program decoder changed during checking; replan")
            selected.revalidate()
        def decode(image):
            fresh_decoder()
            actual = decoder.decode(image)
            fresh_decoder()
            expected = []
            for offset, name, values in self.calls(selected, image):
                instruction = row["instructions"].get(name)
                if instruction is None or instruction["control_flow"] == "unknown":
                    raise ValueError("actual final instruction has unknown selected control/effect semantics")
                effect = storage.get(name)
                accesses = (() if effect is None else tuple(
                    EncodedAccess(_space(a["space"], values), binding_value(a["start"], values),
                                  binding_value(a["size"], values), a["writes"]) for a in effect["accesses"]))
                branches = tuple(EncodedBranch(key, values[rule["displacement"]]) for key, rule in
                                 row["branches"].items() if rule["instruction"] == name)
                expected.append(DecodedInstruction(offset, width, accesses, branches,
                                                    instruction["completion"], instruction["relocatable"],
                                                    instruction["control_flow"] != "relative"))
            if type(actual) is not tuple or actual != tuple(expected):
                raise ValueError("provider final decoding disagrees with actual selected ISA effects/branches")
            return actual
        record = {
            "schema": "merlin.selected_final_program_binding.v1",
            "hardware_byte_capacity": "unknown" if row["limits"]["max_bytes"] is None else "checked",
            "hardware_instruction_capacity": "unknown" if row["limits"]["max_instructions"] is None else "checked",
            "continuation_unavailable": unavailable, "effective_limits": asdict(effective),
        }
        return effective, ProgramDecoder(decode, decoder.configuration_identity,
                                         decoder.qualification_identity), continuation, record

    def verify_configuration(self, selected, image, continuation, qualification_identity):
        """Reconstruct state fields from actual words before allowing launches."""
        adapter = selected.configuration_adapter(qualification_identity)
        resources = {r.name: r for r in selected.isa.resources}
        for _offset, name, values in self.calls(selected, image):
            effects = json.loads(selected.isa.instruction(name).effects_json)
            if any(resources[r].kind == "execution" for kind in ("reads", "writes") for r in effects[kind]):
                raise ValueError("final segmentation needs explicit continuation for nonmemory execution resources")
            if not any(resources[r].kind == "state" for kind in ("reads", "writes") for r in effects[kind]):
                continue
            actual = adapter.project(Call(name, tuple((k, Const(v)) for k, v in values.items())))
            if actual is None or not actual.ordered:
                raise ValueError("final segmentation lacks complete configuration-state projection")
            fields = {key for key in (*dict(actual.reads), *dict(actual.writes), *actual.clobbers)}
            if not fields <= set(continuation.preserved_configuration):
                raise ValueError("final segmentation lacks selected configuration-field persistence")


def resolve_program_contract(declaration, facts, isa, storage):
    row = _fields(declaration, ("schema", "limits", "windows", "branches", "instructions", "continuation"), "contract")
    if row["schema"] != "merlin.final_program_contract.v1" or isa.word_bits % 8:
        raise ValueError("selected final program requires its versioned byte-aligned ISA contract")
    limits = _fields(row["limits"], ("max_bytes", "max_instructions", "instruction_alignment"), "limits")
    limits = {key: None if value is None else resolve_fact_integer(value, facts, label="program " + key, minimum=1)
              for key, value in limits.items()}
    if type(row["windows"]) is not dict or len(row["windows"]) > 256:
        raise ValueError("final program windows must be a bounded memory mapping")
    resources = {r.name: r for r in isa.resources}
    storage_windows = {w.resource: w for w in (*storage.banks, *storage.external)} if storage is not None else {}
    windows = {}
    for name, item in row["windows"].items():
        _fields(item, ("start", "size", "backing", "visibility"), "window")
        resource = resources.get(name)
        if resource is None or resource.kind != "memory" or resource.unit != "bytes":
            raise ValueError("final program windows require ISA byte memory resources")
        start = resolve_fact_integer(item["start"], facts, label="program window start")
        size = resolve_fact_integer(item["size"], facts, label="program window size", minimum=1)
        if resource.capacity is not None and start + size > resource.capacity:
            raise ValueError("final program window exceeds selected RTL resource capacity")
        if (type(item["backing"]) is not str or not item["backing"]
                or item["visibility"] not in ("private", "external", "unknown")):
            raise ValueError("final program requires explicit canonical backing and visibility")
        prior = storage_windows.get(name)
        if prior is not None and (item["backing"] != prior.backing or (item["visibility"] == "private") != (
            prior.ownership == "invocation_private") or start < prior.start or start + size > prior.start + prior.size):
            raise ValueError("final program windows contradict selected storage aliases or ownership")
        windows[name] = {**item, "start": start, "size": size}
    for window in windows.values():
        if any(w["backing"] == window["backing"] and w["visibility"] != window["visibility"] for w in windows.values()):
            raise ValueError("final program aliases must agree on physical visibility")
    if type(row["branches"]) is not dict or len(row["branches"]) > 256:
        raise ValueError("final program branch rules must be a bounded mapping")
    branches = {}
    for name, item in row["branches"].items():
        _fields(item, ("instruction", "displacement", "scale", "origin"), "branch")
        if type(name) is not str or not name or item["origin"] not in ("instruction", "next_instruction", "unknown"):
            raise ValueError("final program branch identity and relative origin must be explicit")
        instruction = isa.instruction(item["instruction"])
        argument = next((a for a in instruction.arguments() if a.name == item["displacement"]), None)
        field = next((f for f in instruction.fields if f.argument == item["displacement"]), None)
        if argument is None or argument.type.kind != "integer" or field is None:
            raise ValueError("final branch displacement must name an encoded integer argument")
        bits = min(argument.type.bits, field.width)
        branches[name] = {**item, "minimum": -(1 << (bits - 1)) if argument.type.signed else 0,
                          "maximum": (1 << (bits - int(argument.type.signed))) - 1,
                          "scale": resolve_fact_integer(item["scale"], facts, label="branch byte scale", minimum=1)}
    if len({r["instruction"] for r in branches.values()}) != len(branches):
        raise ValueError("select one explicit relative branch rule per instruction")
    if type(row["instructions"]) is not dict or len(row["instructions"]) > 4096:
        raise ValueError("final program instructions must be a bounded mapping")
    effects = json.loads(storage.declarations_json) if storage is not None else {}
    for name, item in row["instructions"].items():
        instruction = isa.instruction(name)
        _fields(item, ("completion", "relocatable", "control_flow"), "instruction")
        if (item["completion"] not in ("synchronous", "asynchronous", "drain", "unknown")
                or type(item["relocatable"]) is not bool
                or item["control_flow"] not in ("fallthrough", "conditional_relative", "relative", "unknown")):
            raise ValueError("final program execution and control effects must be explicit")
        if (item["control_flow"] in ("relative", "conditional_relative")) != any(
            b["instruction"] == name for b in branches.values()
        ):
            raise ValueError("final program relative control flow requires exactly its selected branch binding")
        declared, semantics = json.loads(instruction.effects_json), json.loads(instruction.semantics_json)
        if declared == {"status": "unknown"} or semantics == {"status": "unknown"}:
            raise ValueError("final program requires complete ISA operation/effect declarations")
        memory = {r for k in ("reads", "writes") for r in declared[k] if resources[r].kind == "memory"}
        if memory and name not in effects:
            raise ValueError("final program memory effects require selected storage projections and windows")
        if name in effects and item["completion"] != "unknown":
            permitted = {"drain": ("wait", "synchronous"), "asynchronous": ("asynchronous",),
                         "synchronous": ("synchronous",)}[item["completion"]]
            if effects[name]["completion"] not in permitted:
                raise ValueError("final program completion contradicts selected storage software effects")
    if any(rule["instruction"] not in row["instructions"] for rule in branches.values()):
        raise ValueError("final branch has no declared program instruction semantics")
    launch = row["continuation"]
    if launch is not None:
        _fields(launch, ("sequential_completion", "starts_at_zero", "max_segments", "preserved_backings",
                        "preserved_configuration"), "continuation")
        if (any(type(launch[k]) is not bool for k in ("sequential_completion", "starts_at_zero"))
                or type(launch["max_segments"]) is not int or launch["max_segments"] <= 0):
            raise ValueError("final program continuation requires explicit launch semantics and a finite budget")
        if set(_names(launch["preserved_backings"], "preserved backings")) - {w["backing"] for w in windows.values()}:
            raise ValueError("final continuation names an undeclared physical backing")
        if type(launch["preserved_configuration"]) is not dict or len(launch["preserved_configuration"]) > 256:
            raise ValueError("final configuration persistence requires a bounded resource/field mapping")
        configuration = []
        for resource, indices in launch["preserved_configuration"].items():
            state = resources.get(resource)
            if state is None or state.kind != "state" or state.unit != "fields" or type(indices) is not list:
                raise ValueError("final configuration persistence requires typed ISA state fields")
            if state.capacity is None or len(indices) > 4096:
                raise ValueError("final configuration persistence needs known bounded RTL field capacity")
            for value in indices:
                index = resolve_fact_integer(value, facts, label="persistent configuration index")
                if index >= state.capacity:
                    raise ValueError("persistent configuration index exceeds selected RTL capacity")
                configuration.append(resource + ":" + str(index))
        launch = {**launch, "preserved_configuration": sorted(set(configuration))}
    return SelectedProgramContract(json.dumps({**row, "limits": limits, "windows": windows,
                                               "branches": branches, "continuation": launch},
                                              sort_keys=True, separators=(",", ":"), allow_nan=False))
