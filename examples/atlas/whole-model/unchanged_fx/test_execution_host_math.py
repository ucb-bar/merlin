"""Checks for validated component text replacement in execution packaging."""
import copy
import ctypes as ct
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest

import execution_host_math as consumer
from merlin.compile.c_sources import c_source_pair


def digest(data):
    return sha256(data).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    production = tmp_path / 'application/c-package'
    execution = tmp_path / 'runtime/c-package'
    production.mkdir(parents=True)
    archive = tmp_path / 'native.a'
    archive.write_bytes(b'synthetic native supplier')
    native_archive_sha256 = digest(archive.read_bytes())
    monkeypatch.setattr(consumer, 'PRODUCTION_ARCHIVE', 'b' * 64)
    contract = {'schema': 'merlin.host_linkage_contract.v1', 'supplier': 'selected_math_archive',
                'archive_sha256': native_archive_sha256, 'symbols': ['cosf', 'sinf']}
    out = tmp_path / 'out'
    gate_path = out / 'execution-host-math/equivalence-receipt.json'
    gate_path.parent.mkdir(parents=True)
    angles = struct.pack('<II', 0, 0x3f800000) * 32768
    rv64_output = tmp_path / 'domain' / 'qualified.bin'
    rv64_output.parent.mkdir()
    rv64_output.write_bytes(b''.join(angles[i:i+4] + b'\0' * 8 for i in range(0, len(angles), 4)))
    gate = {'schema': 'rotary.execution-host-math-domain-check.v1', 'gate': 'PASS', 'rounding': 'RNE',
            'linkage': contract, 'execution_host_archive_sha256': native_archive_sha256,
            'production_rv64_archive_sha256': consumer.PRODUCTION_ARCHIVE,
            'domain': {'distinct_input_lanes': 65536, 'compared_f32_trig_words': 131072},
            'angle_mismatch_count': 0, 'angle_input_mismatch_count': 0, 'f32_mismatch_count': 0,
            'bf16_mismatch_count_after_cat': 0, 'execution_host_output_sha256': 'a' * 64,
            'rv64_output_sha256': digest(rv64_output.read_bytes()),
            'results': {op: {'f32_mismatch_count': 0, 'bf16_mismatch_count_after_cat': 0} for op in ('sin', 'cos')},
            'output_hashes': {engine: {'angle_sha256': digest(angles)} for engine in ('rv64', 'execution_host')}}
    gate['execution_host_output_sha256'] = gate['rv64_output_sha256']
    gate_path.write_text(json.dumps(gate))
    regions, headers, sources = [], {}, {}
    nodes = [{'id': 'input', 'type': {'dtype': 'f32', 'shape': [2], 'numerical_policy': 'raw-bit-pattern'}}]
    prod_contract = {**contract, 'archive_sha256': consumer.PRODUCTION_ARCHIVE}
    for index, symbol in enumerate(('sinf', 'cosf')):
        name = f'region_{index:03d}'
        member, header = f'regions/{name}/{name}.c', f'regions/{name}/{name}.h'
        headers[header] = f'int {name}(void *,const void *const *,void *const *);\n'
        sources[member] = (f'#include "{name}.h"\n#if !defined(__riscv)\n#error RV64 required\n#endif\n'
                           f'int {name}(void *c,const void *const *i,void *const *o) {{ return 7; }}\n')
        for path, text in ((member, sources[member]), (header, headers[header])):
            target = production / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        op = 'source_f32_' + symbol[:-1] + '_unqualified'
        nodes.append({'id': name, 'inputs': ['input'], 'op': op, 'type': nodes[0]['type']})
        desc = {'arithmetic': 'selected-library-source-math', 'source_intrinsic': op, 'host_isa': 'rv64gc',
                'host_abi': 'lp64d', 'numeric_policy': 'selected-libm-rne-finite-v1',
                'effects': 'enabled-fpu-rne-unobserved-flags-and-errno-v1', 'source_request_digest': 'c' * 64,
                'implementation': name, 'host_linkage': {**prod_contract, 'symbols': [symbol]}}
        regions.append({'description': desc, 'placement': 'host', 'native_programs': [], 'outputs': [name],
                        'inputs': ['input'], 'source_request_digest': 'c' * 64})
    sources['retained.c'] = 'int retained(void) { return 91; }\n'
    package = c_source_pair('combined', public_headers=headers, private_headers={}, sources=sources)
    for path, text in package.items():
        (production / path).write_text(text)
    request = production.parent / 'source-request.json'
    request.write_text(json.dumps({'nodes': nodes}))
    inventory = {'c-package/' + str(path.relative_to(production)): digest(path.read_bytes())
                 for path in production.rglob('*') if path.is_file()}
    inventory['source-request.json'] = digest(request.read_bytes())
    manifest = {'application_interface': {'source': 'combined.c', 'host_linkage': [prod_contract]},
                'regions': regions, 'stage_artifacts': inventory}
    return manifest, production, execution, gate_path, archive, rv64_output


def test_validated_component_text_replacement_preserves_production(case):
    manifest, production, execution, gate, archive, rv64_output = case
    before = {str(p): p.read_bytes() for p in production.rglob('*') if p.is_file()}
    manifest_before = copy.deepcopy(manifest)
    record = consumer.prepare_execution_host_math(*case)
    result = (execution / 'combined.c').read_text()
    assert '#error RV64 required' not in result
    assert 'int retained(void) { return 91; }' in result
    assert len(record['components']) == 2
    assert record['production_linkage'] != record['execution_linkage']
    assert record['qualified_distinct_encodings'] == 2
    assert manifest == manifest_before
    assert all(Path(path).read_bytes() == content for path, content in before.items())


def test_identical_evidence_at_explicit_paths_emits_identical_component_bytes(case, tmp_path):
    manifest, production, execution, gate, archive, rv64_output = case
    first = consumer.prepare_execution_host_math(*case)
    relocated = tmp_path / 'relocated'
    relocated.mkdir()
    other_gate, other_archive, other_output = (relocated / name for name in ('receipt.json', 'supplier.a', 'domain.bin'))
    for old, new in ((gate, other_gate), (archive, other_archive), (rv64_output, other_output)):
        new.write_bytes(old.read_bytes())
    other_execution = tmp_path / 'other-runtime'
    second = consumer.prepare_execution_host_math(manifest, production, other_execution,
                                                  other_gate, other_archive, other_output)
    assert (execution / 'combined.c').read_bytes() == (other_execution / 'combined.c').read_bytes()
    assert first['components'] == second['components']
    assert first['execution_source_sha256'] == second['execution_source_sha256']


@pytest.mark.parametrize('change', ['archive', 'receipt_archive', 'pin', 'gate', 'domain', 'component', 'request'])
def test_refuses_changed_evidence(case, change):
    manifest, production, execution, gate_path, archive, rv64_output = case
    if change == 'archive':
        archive.write_bytes(b'wrong archive')
    elif change == 'receipt_archive':
        gate = json.loads(gate_path.read_text())
        gate['execution_host_archive_sha256'] = 'e' * 64
        gate_path.write_text(json.dumps(gate))
    elif change == 'pin':
        manifest['regions'][0]['description']['host_linkage']['archive_sha256'] = 'd' * 64
    elif change == 'gate':
        gate = json.loads(gate_path.read_text()); gate['f32_mismatch_count'] = 1
        gate_path.write_text(json.dumps(gate))
    elif change == 'domain':
        rv64_output.write_bytes(b'wrong domain')
    elif change == 'component':
        (production / 'regions/region_000/region_000.c').write_text('changed')
    else:
        (production.parent / 'source-request.json').write_text('{}')
    with pytest.raises(ValueError):
        consumer.prepare_execution_host_math(*case)
    assert not (execution / 'combined.c').exists()


@pytest.mark.skipif(os.environ.get('EXECUTION_HOST_MATH_C_TEST') != '1', reason='opt-in lightweight C check')
def test_native_preflight_checks_before_writes(tmp_path):
    source = tmp_path / 'unit.c'
    source.write_text(consumer._native_source('unary', 'sinf', 2, [0, 0x3f800000]) + '''
int set_round(int mode) { return fesetround(mode ? FE_DOWNWARD : FE_TONEAREST); }
void set_mxcsr(unsigned value) { _mm_setcsr(value); }
unsigned get_mxcsr(void) { return _mm_getcsr(); }
''')
    library = tmp_path / 'unit.so'
    subprocess.run(['nice', '-n', '10', 'cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-frounding-math', '-ffp-contract=off', '-shared', '-fPIC', str(source), '-lm', '-o', str(library)], check=True)
    lib = ct.CDLL(str(library))
    lib.unary.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    lib.get_mxcsr.restype = ct.c_uint
    inp = (ct.c_uint32 * 2)(0, 0x3f800000)
    out = (ct.c_uint32 * 2)(0xa5a5a5a5, 0xa5a5a5a5)
    inputs = (ct.c_void_p * 1)(ct.addressof(inp)); outputs = (ct.c_void_p * 1)(ct.addressof(out))
    assert lib.unary(None, inputs, outputs) == 0
    for bad in (0x40000000, 0x7fc00000):
        inp[1] = bad; out[:] = [0xa5a5a5a5] * 2
        assert lib.unary(None, inputs, outputs) in (-2, -4)
        assert list(out) == [0xa5a5a5a5] * 2
    inp[1] = 0x3f800000
    assert lib.unary(None, inputs, inputs) == -1
    out[:] = [0xa5a5a5a5] * 2
    assert lib.set_round(1) == 0
    try:
        assert lib.unary(None, inputs, outputs) == -3
        assert list(out) == [0xa5a5a5a5] * 2
    finally:
        lib.set_round(0)
    original = lib.get_mxcsr()
    try:
        for bit in (1 << 6, 1 << 15, 1 << 13):
            lib.set_mxcsr(original | bit)
            assert lib.unary(None, inputs, outputs) == -3
            assert list(out) == [0xa5a5a5a5] * 2
    finally:
        lib.set_mxcsr(original)
