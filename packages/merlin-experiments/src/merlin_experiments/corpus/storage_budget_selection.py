"""Explicit operator storage budget from the already selected execution memory maps.

This is preparation metadata, not an ELF-fit, numerical or runtime certificate.
The ordinary complete-storage/readback renderer remains the actual allocator.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.text_tokens import source_tokens

CEILING = 256 << 20
RESERVE = 16 << 20  # Explicit software headroom; not inferred target capacity.


def _pin(row, *, limit):
    if type(row) is not dict or not {'path', 'sha256'} <= set(row):
        raise ValueError('storage selection requires original source pins')
    path = Path(row['path'])
    if (not path.is_absolute() or path.resolve() != path or path.is_symlink()
        or not path.is_file() or path.stat().st_size > limit
        or sha256_file(path) != row['sha256']):
        raise ValueError('storage selection source pin changed or exceeds its read budget')
    return path


def _unsigned_literal(value):
    if value.startswith("0x"):
        valid = bool(value[2:]) and all(char in "0123456789abcdefABCDEF" for char in value[2:])
    else:
        valid = bool(value) and all(char in "0123456789" for char in value)
    if not valid:
        raise ValueError("unsupported nonliteral memory extent")
    return int(value, 0)


def _memory_region(argument):
    parts = argument[2:].split(":")
    if len(parts) != 2:
        raise ValueError('storage selection requires one explicit Spike base:size memory region')
    return tuple(_unsigned_literal(part) for part in parts)


def _dram_parameters(text):
    """Read one FIRRTL external memory declaration at circuit indentation."""
    bodies, current = [], None
    for raw in text.splitlines():
        if raw.strip() and len(raw) - len(raw.lstrip()) <= 2:
            current = None
            if raw.partition(":")[0].split() == ["extmodule", "SimDRAM"] and ":" in raw:
                current = []
                bodies.append(current)
        elif current is not None:
            current.append(raw)
    if len(bodies) != 1:
        raise ValueError('requires one explicit SimDRAM memory declaration')
    parameters = {}
    for raw in bodies[0]:
        fields = raw.split()
        if len(fields) >= 2 and fields[:1] == ["parameter"] and fields[1] in ("MEM_BASE", "MEM_SIZE"):
            if (len(fields) != 4 or fields[2] != "=" or not fields[3].isascii()
                    or not fields[3].isdecimal() or fields[1] in parameters):
                raise ValueError('missing or ambiguous selected SimDRAM ' + fields[1])
            parameters[fields[1]] = int(fields[3])
    if set(parameters) != {"MEM_BASE", "MEM_SIZE"}:
        raise ValueError('missing selected SimDRAM memory parameters')
    return parameters["MEM_BASE"], parameters["MEM_SIZE"]


def _linker_memory(text):
    tokens = [token.text for token in source_tokens(text)]
    loads, reservations = [], []
    if any(token in {"INPUT", "GROUP", "SEARCH_DIR", "INCLUDE", "STARTUP"} for token in tokens):
        raise ValueError('unsupported selected linker load/closure declaration')
    for index, token in enumerate(tokens):
        if token != ".":
            continue
        if tokens[index + 1:index + 2] == ["="]:
            start, destination = index + 2, loads
        elif tokens[index + 1:index + 3] == ["+", "="]:
            start, destination = index + 3, reservations
        else:
            continue
        if tokens[start + 1:start + 2] != [";"]:
            raise ValueError('unsupported dynamic linker reservation or load address')
        destination.append(_unsigned_literal(tokens[start]))
    return loads, reservations


def select(contract, budget_bytes):
    if type(budget_bytes) is not int or not 0 < budget_bytes <= CEILING:
        raise ValueError('logical storage budget must be a positive integer within the 256MiB software ceiling')
    if type(contract) is not dict:
        raise ValueError('storage selection requires an original contract')
    runner = contract.get('runner', {})
    if runner.get('backend') != 'chipyard_rocc':
        raise ValueError('storage selection requires the explicit data-bound execution family')
    block = runner.get('chipyard_rocc', {})
    engines, toolchain = block.get('engines', {}), block.get('toolchain', {})
    if set(engines) != {'spike', 'gsim'}:
        raise ValueError('storage selection requires known memory maps for every selected engine')
    arguments = engines['spike'].get('argv', [])
    memory = [arg for arg in arguments if type(arg) is str and arg.startswith('-m')]
    if len(memory) != 1:
        raise ValueError('storage selection requires one explicit Spike base:size memory region')
    spike_base, spike_size = _memory_region(memory[0])
    receipt_pin = engines['gsim'].get('receipt')
    receipt_path = _pin(receipt_pin, limit=16 << 20)
    receipt = json.loads(receipt_path.read_bytes())
    fir_pin = receipt.get('artifacts', {}).get('firrtl')
    fir_path = _pin(fir_pin, limit=128 << 20)
    if (receipt.get('status') != 'complete' or receipt.get('firrtl_sha256') != fir_pin['sha256']
        or receipt.get('binary_sha256') != engines['gsim'].get('binary', {}).get('sha256')):
        raise ValueError('selected gSIM memory source does not match its completed engine receipt')
    native_base, native_size = _dram_parameters(fir_path.read_text())
    link_pin = toolchain.get('link_script')
    link_path = _pin(link_pin, limit=1 << 20)
    loads, reservations = _linker_memory(link_path.read_text())
    if len(loads) != 1 or loads[0] != toolchain.get('load_address'):
        raise ValueError('unsupported selected linker load/closure declaration')
    if sum(reservations) >= RESERVE:
        raise ValueError('selected linker reservations exhaust explicit software headroom')
    base = toolchain['load_address']
    for start, size in ((spike_base, spike_size), (native_base, native_size)):
        if (start != base or size <= 0 or start + size >= 2**64
            or budget_bytes + RESERVE > size):
            raise ValueError('logical storage plus explicit software headroom exceeds selected execution memory')
    # Reopen only the source leaves read above before publishing new selection.
    for pin, limit in ((receipt_pin, 16 << 20), (fir_pin, 128 << 20), (link_pin, 1 << 20)):
        _pin(pin, limit=limit)
    selected = copy.deepcopy(contract)
    logical = selected.setdefault('logical_harness', {})
    if type(logical) is not dict:
        raise ValueError('logical harness declaration must remain a mapping')
    previous = logical.get('storage_budget_bytes')
    if previous is not None and (type(previous) is not int or previous <= 0):
        raise ValueError('malformed original logical storage budget')
    logical['storage_budget_bytes'] = budget_bytes
    return selected, dict(schema='merlin.logical_storage_budget_selection.v1',
        storage_budget_bytes=budget_bytes, previous_budget_bytes=previous,
        software_ceiling_bytes=CEILING, software_headroom_bytes=RESERVE,
        load_address=base, spike_memory_argument=memory[0],
        gsim_memory=dict(base=native_base, bytes=native_size, firrtl=copy.deepcopy(fir_pin), receipt=copy.deepcopy(receipt_pin)),
        link_script=copy.deepcopy(link_pin),
        scope='explicit complete harness storage bound only; full output roster/readback and native ELF execution checks remain mandatory; not a linked-image fit proof')


def reconcile(original, released, preparation, selected_path):
    """Reopen the exact operator preparation before admitting a changed budget."""
    budget = (released.get('logical_harness') or {}).get('storage_budget_bytes')
    if budget == (original.get('logical_harness') or {}).get('storage_budget_bytes'):
        return original
    record = preparation.get('scaffolding', {}).get('capability_contract', {}).get('logical_storage_selection')
    if type(record) is not dict or record.get('selected_contract_sha256') != sha256_file(selected_path):
        raise ValueError('storage budget lacks its exact reviewed release preparation pin')
    selected, observed = select(original, budget)
    if record != {**observed, 'selected_contract_sha256': sha256_file(selected_path)}:
        raise ValueError('storage budget preparation differs from selected memory/layout evidence')
    return selected
