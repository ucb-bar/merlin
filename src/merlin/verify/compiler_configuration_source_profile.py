"""Fixed original-source predicates for bounded configuration verification.

No optimizer, proposed target or proof runner is used. Explicit model exclusions
are separate from original-selection, parser, resource and custody failures.
"""

from __future__ import annotations

import json

from xdsl.dialects import builtin

from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer
from merlin.compile.scheduling.configuration_loops import _bounded
from merlin.sched.ir import Call, Kernel, Loop, parse_kernel
from merlin.sched.ir.kernel import check_structure

from .compiler_configuration_intake import request
from .compiler_configuration_kernel import program as elision_kernel_program
from .compiler_configuration_kernel import tree
from .compiler_profile import SourceOutsideProfile
from .configuration_elision_bridge import Unsupported as ElisionUnsupported
from .configuration_elision_bridge import _program as elision_module_program
from .configuration_elision_bridge import _reserved_attributes
from .configuration_hoist_bridge import Unsupported as HoistUnsupported
from .configuration_hoist_bridge import program as static_hoist_program
from .lean_graph_input_admission import InputAdmissionError, _attributes, _fields, admit_resources, admit_text
from .memory_scalar_nan_admission import admit_memory_scalar_attributes, parse_memory_module

ELISION_ID = "configuration-straightline-integer-source-v1"
HOIST_ID = "configuration-positive-counted-kernel-source-v1-v6"
MODEL = {
    "elision": {
        "id": ELISION_ID,
        "gate": "configuration_write_elision",
        "root": "ActualConfigurationElision.complete_entries",
        "claim": "equal_modeled_complete_entry_error_and_terminal_state",
        "source": "complete straightline integer instruction entries or straightline typed Kernel",
        "bounds": {"instruction_bytes": 1024 * 1024, "calls": 4096, "fields": 256},
        "excluded": ["nested regions/CFG/internal calls", "dynamic decoded MLIR ISA operands", "floating metadata"],
        "Kernel_retained_calls": "unknown/dynamic calls remain exact barriers, never removable setters",
    },
    "hoist": {
        "id": HOIST_ID,
        "gate": "configuration_loop_hoisting",
        "root": "ActualConfigurationHoist.complete_kernel",
        "claim": "equal_conditional_complete_kernel_error_terminal_state_hoist",
        "source": "literal pure setup prefixes in positive flat or recursive typed counted Kernel",
        "bounds": {"calls": 4096, "fields": 256, "depth": 64, "default_extent": 1000000},
        "C64_extent": (1 << 63) - 1,
        "excluded": ["completion/staging", "dynamic configuration values", "unknown expression forms"],
    },
    "premises": ["original selected ISA/effects", "total exact pure setters", "complete nonaliasing state"],
    "approval": "qualification labels do not approve effects, native behavior or numerical policy",
}


def _outside(profile, code, reason):
    raise SourceOutsideProfile(profile, code, reason)


def _selected(selection, profile):
    request(selection)  # Invalid or changed original owners are not source exclusions.
    if selection.qualification_identity is None:
        _outside(profile, "effect-premise", "configuration proof needs the original explicit effect-premise selection")
    return selection.selected_contracts.isa.record(), json.loads(selection.selected_contracts.effects_json)


def _budgets(program, selection, profile):
    atoms = (
        [atom for entry in program["entries"] for atom in entry["code"]] if "entries" in program else program["atoms"]
    )
    if len(atoms) > min(4096, selection.max_operations):
        _outside(profile, "calls", "configuration source exceeds its fixed/original call budget")
    fields = set()
    for atom in atoms:
        effect = atom["effect"]
        if effect is not None:
            fields.update(key for key, _ in (*effect["reads"], *effect["writes"]))
            fields.update(effect["clobbers"])
    if len(fields) > min(256, selection.max_fields):
        _outside(profile, "fields", "configuration source exceeds its fixed/original field budget")


def admit_elision_source(raw, *, selection, representation):
    isa, effects = _selected(selection, ELISION_ID)
    if type(raw) is not str:
        raise TypeError("original complete configuration source text required")
    limit = min(selection.max_ir_bytes, 1024 * 1024) if representation == "instruction_mlir" else selection.max_ir_bytes
    if len(raw.encode()) > limit:
        _outside(ELISION_ID, "source-bytes", "configuration source exceeds its declared raw-byte model bound")
    try:
        if representation == "instruction_mlir":
            admit_text(raw)
            _reserved_attributes(raw)
            module = parse_memory_module(raw)
            admit_resources(module)
            admit_memory_scalar_attributes(module)
            if len(tuple(module.walk())) > 4096:
                _outside(ELISION_ID, "syntax-nodes", "configuration instruction source exceeds4096 operations")
            for operation in module.walk():
                for value, field in _fields(operation):
                    for attribute, _ in _attributes(value, field):
                        if isinstance(attribute, builtin.AnyFloat):
                            _outside(
                                ELISION_ID, "floating", "bounded configuration instruction profile is integer-only"
                            )
            program = elision_module_program(module, isa, effects, allow_bare=True)
        elif representation == "schedule_kernel":
            parsed = parse_kernel(raw)
            if check_structure(parsed):
                raise ValueError("original configuration Kernel is not well formed")
            if parsed.text() != raw:
                _outside(ELISION_ID, "canonical-kernel", "configuration model requires canonical complete Kernel text")
            program = elision_kernel_program(raw, isa=isa, effects=effects)
        else:
            raise ValueError("original configuration representation must be instruction_mlir or schedule_kernel")
    except InputAdmissionError as error:
        _outside(ELISION_ID, error.code, str(error))
    except ElisionUnsupported as error:
        _outside(ELISION_ID, "source-grammar", str(error))
    _budgets(program, selection, ELISION_ID)
    return ELISION_ID


def admit_hoist_source(kernel, *, selection, c_consumer=None):
    isa, effects = _selected(selection, HOIST_ID)
    if type(kernel) is not Kernel:
        raise TypeError("original typed Kernel required for configuration hoisting")
    from merlin.compile.scheduling.configuration import ConfigurationUnavailable

    try:
        _bounded(kernel, min(4096, selection.max_operations), min(256, selection.max_fields), 64)
    except ConfigurationUnavailable as error:
        _outside(HOIST_ID, "source-bounds-or-expression", str(error))
    if check_structure(kernel):
        raise ValueError("original Kernel is not well formed")
    extent_limit = 1000000
    if c_consumer is not None:
        if type(c_consumer) is not ConfigurationCConsumer:
            raise TypeError("original typed configuration C consumer required")
        c_consumer.revalidate()
        c_consumer.admit(kernel)  # Native consumer legality is not an exclusion escape.
        extent_limit = (1 << 63) - 1
    pending = list(kernel.body)
    nested = False
    for statement in kernel.body:
        if type(statement) is Loop and any(type(child) is Loop for child in statement.body):
            nested = True
    while pending:
        statement = pending.pop()
        if type(statement) is Loop:
            if not 0 < statement.extent <= extent_limit:
                _outside(HOIST_ID, "tripcount", "configuration loop count exceeds the original selected model domain")
            if statement.iter_args or statement.yields or statement.results:
                _outside(HOIST_ID, "completion", "configuration hoisting model excludes loop completion fields")
            pending.extend(statement.body)
        elif type(statement) is not Call:
            _outside(HOIST_ID, "statement", "configuration model admits only ordered calls and counted loops")
    raw = kernel.text()
    if len(raw.encode()) > selection.max_ir_bytes:
        _outside(HOIST_ID, "source-bytes", "configuration Kernel exceeds its original source-byte model bound")
    if parse_kernel(raw) != kernel:
        raise ValueError("original typed Kernel text did not roundtrip")
    try:
        if nested:
            from .compiler_configuration_hoist_recursive import decode

            program = decode(raw, isa=isa, effects=effects, analyses=[], extent_limit=extent_limit)
        else:
            original = tree(kernel)
            try:
                program, _ = static_hoist_program(original, isa, effects, extent_limit=extent_limit)
            except HoistUnsupported:
                from .compiler_configuration_hoist_indexed import program_current

                program, _ = program_current(
                    original, isa, effects, analyses=[], allow_multiple=True, extent_limit=extent_limit
                )
    except HoistUnsupported as error:
        _outside(HOIST_ID, "source-grammar", str(error))
    _budgets(program, selection, HOIST_ID)
    return HOIST_ID
