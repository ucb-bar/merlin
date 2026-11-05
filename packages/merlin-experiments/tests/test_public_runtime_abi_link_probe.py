"""Private callable-only native controls; no candidate, workload or simulator."""
import hashlib
from pathlib import Path
import shutil

import pytest
import yaml

from merlin_experiments.phase1 import public_runtime_abi as P
from merlin_experiments.phase1 import public_runtime_abi_link_probe as L


def selected():
    compiler = shutil.which('cc')
    if compiler is None: pytest.skip('C link control compiler unavailable')
    compiler = Path(compiler).resolve()
    return {'name': 'independent_target',
            'harness_abi': {'entry_symbol': 'alternate_call', 'fence_symbol': 'alternate_finish'},
            'logical_harness': {'completion_asm': 'fixed_control_barrier'},
            'runner': {'backend': 'private_control', 'private_control': {'toolchain': {
                'compiler': {'path': str(compiler), 'sha256': hashlib.sha256(compiler.read_bytes()).hexdigest()},
                'cflags': ['-O1', '-ffreestanding'], 'ldflags': []}}}}


def public(contract):
    return yaml.safe_load(P.rewrite_protocol(yaml.safe_dump({'logical_kernel_abi': {
        'symbol': 'old_entry', 'signature': 'void old_entry(void *arg_0, ..., void *arg_N)'}}).encode(), contract=contract))


def test_native_selected_link_uses_only_public_callable_definitions(tmp_path):
    contract = selected()
    result = L.probe(contract, public(contract), output=tmp_path/'private-control')
    assert result['status'] == 'pass'
    assert result['author_artifacts_delivered'] is False
    assert result['elf_sha256']


def test_native_link_rejects_old_entry_only_public_packet(tmp_path):
    contract = selected();doc = public(contract)
    doc['logical_kernel_abi']['required_callables'] = doc['logical_kernel_abi']['required_callables'][:1]
    result = L.probe(contract, doc, output=tmp_path/'old-entry-only')
    assert result['status'] == 'fail'
    assert b'alternate_finish' in (tmp_path/'old-entry-only/stderr.log').read_bytes()


def test_native_compile_rejects_public_completion_with_wrong_argument_prototype(tmp_path):
    contract = selected();doc = public(contract)
    doc['logical_kernel_abi']['required_callables'][1]['role'] = 'entry'
    doc['logical_kernel_abi']['required_callables'][1]['parameters'] = 'per_case_logical_tensor_pointers'
    result = L.probe(contract, doc, output=tmp_path/'wrong-completion-prototype')
    assert result['status'] == 'fail'


def test_link_control_refuses_changed_compiler_pin_before_execution(tmp_path):
    contract = selected();doc = public(contract)
    contract['runner']['private_control']['toolchain']['compiler']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='differs from selected pin'):
        L.probe(contract, doc, output=tmp_path/'changed-tool')
    assert not (tmp_path/'changed-tool').exists()
