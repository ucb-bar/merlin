"""Explicit budget preparation retains pinned memory maps and complete storage bounds."""
import copy
import json
from pathlib import Path

import pytest
import yaml

from merlin.common.digest import sha256_file
from merlin_experiments.corpus import preparation
from merlin_experiments.corpus.storage_budget_selection import select


def pin(path):
    return {'path': str(path), 'sha256': sha256_file(path)}


@pytest.fixture
def contract(tmp_path):
    fir = tmp_path / 'target.fir'
    fir.write_text('circuit Target :\n  extmodule SimDRAM :\n    parameter MEM_BASE = 2147483648\n    parameter MEM_SIZE = 17179869184\n  module Target :\n    skip\n')
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'status': 'complete', 'firrtl_sha256': sha256_file(fir),
        'binary_sha256': 'a' * 64, 'artifacts': {'firrtl': pin(fir)}}))
    linker = tmp_path / 'link.ld'
    linker.write_text('SECTIONS {\n . = 0x80000000;\n .text : { *(.text*) }\n . += 0x40000;\n}\n')
    return {'harness_abi': {'entry_symbol': 'entry', 'fence_symbol': 'done'},
        'numeric_policy': {'compare': 'exact_int'},
        'logical_harness': {'cycle_counter_asm': 'counter'},
        'runner': {'backend': 'chipyard_rocc', 'chipyard_rocc': {
            'engines': {'spike': {'argv': ['-m0x80000000:0x400000000', '{elf}']},
                        'gsim': {'receipt': pin(receipt), 'binary': {'sha256': 'a' * 64}}},
            'toolchain': {'load_address': 0x80000000, 'link_script': pin(linker)}}}}


def test_only_explicit_budget_changes_and_native_stage_records_original(contract, tmp_path):
    before = copy.deepcopy(contract)
    selected, record = select(contract, 128 << 20)
    assert contract == before
    expected = copy.deepcopy(contract)
    expected['logical_harness']['storage_budget_bytes'] = 128 << 20
    assert selected == expected
    assert record['gsim_memory']['bytes'] == 16 << 30
    assert record['software_ceiling_bytes'] == 256 << 20
    source = tmp_path / 'source.yaml'; source.write_text(yaml.safe_dump(contract))
    destination = tmp_path / 'new' / 'contract.yaml'
    staged = preparation._stage_capability_contract(source, sha256_file(source), destination, None,
        logical_storage_budget_bytes=128 << 20)
    assert yaml.safe_load(destination.read_bytes()) == expected
    assert staged['sha256'] == sha256_file(source)
    assert staged['logical_storage_selection']['selected_contract_sha256'] == sha256_file(destination)
    assert staged['logical_storage_selection']['storage_budget_bytes'] == 128 << 20


@pytest.mark.parametrize('budget', [True, False, 0, -1, (256 << 20) + 1, 1.5])
def test_budget_is_bounded_exact_integer(contract, budget):
    with pytest.raises(ValueError, match='256MiB'): select(contract, budget)


@pytest.mark.parametrize('mutation', ['changed_fir', 'changed_receipt', 'changed_linker', 'unknown_engine',
    'missing_memory', 'multi_memory', 'wrong_base', 'too_small', 'binary_mismatch',
    'receipt_incomplete', 'ambiguous_module', 'large_reservation', 'dynamic_reservation', 'indirect_linker'])
def test_changed_unknown_or_insufficient_memory_refuses(contract, mutation):
    block = contract['runner']['chipyard_rocc']; native = block['engines']['gsim']
    receipt_path = Path(native['receipt']['path']); receipt = json.loads(receipt_path.read_bytes())
    fir_path = Path(receipt['artifacts']['firrtl']['path']); link_path = Path(block['toolchain']['link_script']['path'])
    if mutation == 'changed_fir': fir_path.write_text(fir_path.read_text() + '\n')
    if mutation == 'changed_receipt': receipt_path.write_text('{}')
    if mutation == 'changed_linker': link_path.write_text('changed')
    if mutation == 'unknown_engine': block['engines']['other'] = {}
    if mutation == 'missing_memory': block['engines']['spike']['argv'] = ['{elf}']
    if mutation == 'multi_memory': block['engines']['spike']['argv'].append('-m0:1024')
    if mutation == 'wrong_base': block['toolchain']['load_address'] = 0
    if mutation == 'too_small': block['engines']['spike']['argv'] = ['-m0x80000000:0x4000000', '{elf}']
    if mutation == 'binary_mismatch': native['binary']['sha256'] = 'b' * 64
    if mutation == 'receipt_incomplete':
        receipt['status'] = 'incomplete'; receipt_path.write_text(json.dumps(receipt)); native['receipt'] = pin(receipt_path)
    if mutation == 'ambiguous_module':
        fir_path.write_text(fir_path.read_text() + '  extmodule SimDRAM :\n    parameter MEM_BASE = 0\n')
        receipt['artifacts']['firrtl'] = pin(fir_path); receipt['firrtl_sha256'] = sha256_file(fir_path)
        receipt_path.write_text(json.dumps(receipt)); native['receipt'] = pin(receipt_path)
    if mutation in ('large_reservation', 'dynamic_reservation', 'indirect_linker'):
        old = link_path.read_text()
        old = old.replace('0x40000', '0x1000000' if mutation == 'large_reservation' else 'EXTERNAL_SIZE') if mutation != 'indirect_linker' else old + '\nINCLUDE other.ld\n'
        link_path.write_text(old); block['toolchain']['link_script'] = pin(link_path)
    with pytest.raises(ValueError): select(contract, 128 << 20)


def test_unspecified_budget_preserves_historical_contract_bytes(contract, tmp_path):
    source = tmp_path / 'source.yaml'; source.write_text(yaml.safe_dump(contract))
    destination = tmp_path / 'unchanged.yaml'
    record = preparation._stage_capability_contract(source, sha256_file(source), destination, None)
    assert source.read_bytes() == destination.read_bytes()
    assert 'logical_storage_selection' not in record


@pytest.mark.parametrize('mutation', ['none', 'missing', 'budget', 'hash', 'memory', 'extra'])
def test_live_source_intake_reopens_exact_preparation_record(contract, tmp_path, mutation):
    from merlin_experiments.corpus.storage_budget_selection import reconcile
    selected, record = select(contract, 128 << 20)
    path = tmp_path / 'selected.yaml'; path.write_text(yaml.safe_dump(selected))
    record['selected_contract_sha256'] = sha256_file(path)
    prepared = {'scaffolding': {'capability_contract': {'logical_storage_selection': record}}}
    if mutation == 'missing': prepared = {}
    if mutation == 'budget': record['storage_budget_bytes'] = 256 << 20
    if mutation == 'hash': record['selected_contract_sha256'] = '0' * 64
    if mutation == 'memory': record['gsim_memory']['bytes'] -= 1
    if mutation == 'extra': record['unchecked'] = True
    if mutation == 'none': assert reconcile(contract, selected, prepared, path) == selected
    else:
        with pytest.raises(ValueError): reconcile(contract, selected, prepared, path)
