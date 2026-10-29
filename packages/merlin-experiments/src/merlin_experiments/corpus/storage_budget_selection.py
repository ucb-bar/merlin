"""Explicit operator storage budget from the already selected execution memory maps.

This is preparation metadata, not an ELF-fit, numerical or runtime certificate.
The ordinary complete-storage/readback renderer remains the actual allocator.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from merlin.common.digest import sha256_file

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
    if len(memory) != 1 or not re.fullmatch(r'-m(?:0x[0-9a-fA-F]+|[0-9]+):(?:0x[0-9a-fA-F]+|[0-9]+)', memory[0]):
        raise ValueError('storage selection requires one explicit Spike base:size memory region')
    spike_base, spike_size = (int(part, 0) for part in memory[0][2:].split(':'))
    receipt_pin = engines['gsim'].get('receipt')
    receipt_path = _pin(receipt_pin, limit=16 << 20)
    receipt = json.loads(receipt_path.read_bytes())
    fir_pin = receipt.get('artifacts', {}).get('firrtl')
    fir_path = _pin(fir_pin, limit=128 << 20)
    if (receipt.get('status') != 'complete' or receipt.get('firrtl_sha256') != fir_pin['sha256']
        or receipt.get('binary_sha256') != engines['gsim'].get('binary', {}).get('sha256')):
        raise ValueError('selected gSIM memory source does not match its completed engine receipt')
    text = fir_path.read_text()
    modules = re.findall(r'^  extmodule SimDRAM\s*:[^\n]*\n((?:    [^\n]*\n|\n)*)', text, re.MULTILINE)
    if len(modules) != 1:
        raise ValueError('requires one explicit SimDRAM memory declaration')
    def parameter(name):
        matches = re.findall(r'^    parameter ' + name + r' = ([0-9]+)\s*$', modules[0], re.MULTILINE)
        if len(matches) != 1:
            raise ValueError('missing or ambiguous selected SimDRAM ' + name)
        return int(matches[0])
    native_base, native_size = parameter('MEM_BASE'), parameter('MEM_SIZE')
    link_pin = toolchain.get('link_script')
    link_path = _pin(link_pin, limit=1 << 20)
    layout = re.sub(r'/\*.*?\*/', '', link_path.read_text(), flags=re.DOTALL)
    loads = re.findall(r'^\s*\.\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)\s*;', layout, re.MULTILINE)
    reservations = re.findall(r'\.\s*\+=\s*([^;]+);', layout)
    if any(not re.fullmatch(r'(0x[0-9a-fA-F]+|[0-9]+)', item.strip()) for item in reservations):
        raise ValueError('unsupported dynamic linker reservation')
    if (len(loads) != 1 or int(loads[0], 0) != toolchain.get('load_address')
        or re.search(r'\b(INPUT|GROUP|SEARCH_DIR|INCLUDE|STARTUP)\b', layout)):
        raise ValueError('unsupported selected linker load/closure declaration')
    if sum(int(item, 0) for item in reservations) >= RESERVE:
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
