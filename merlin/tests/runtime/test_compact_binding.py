"""Compact callers relocate opaque bytes, with explicit ABI facts and no reuse."""
from dataclasses import asdict, replace
import ctypes
import hashlib
import json
import shutil
import subprocess

import pytest

from merlin.llvmlower.compact_abi import BaseBuffer, PointerBinding
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.compact_binding import CompactTargetABI, resolve_compact_binding
from merlin.runtime.storage_binding import StoragePrepackRequired


def fixture():
    encodings = {
        'a': GroupedAxesStorage((2, 2), 'i16', ((0,), (1,)), (2, 2), (3, 1), 6),
        'scale': GroupedAxesStorage((), 'f32', ((),), (1,), (1,), 1),
        'out': GroupedAxesStorage((2, 2), 'f32', ((1,), (0,)), (2, 2), (3, 1), 8),
    }
    cb = {'tensors': {name: {'shape': list(e.physical_shape), 'dtype': e.dtype,
                            'role': 'output' if name == 'out' else 'input'} for name, e in encodings.items()},
          'params': {'storage_encodings': {name: e.to_dict() for name, e in encodings.items()}},
          'kernel_abi': {'kind': 'whole_program', 'args': [
              {'tensor': name, 'access': 'write' if name == 'out' else 'read'} for name in encodings],
              'outputs': ['out']}}
    contract = {'schema': 'compact_pointer_entry_v1', 'original_symbol': 'original', 'compact_symbol': 'compact',
        'original_argument_count': 3, 'compact_argument_count': 2,
        'bases': [asdict(BaseBuffer('inputs', 64, 32)), asdict(BaseBuffer('outputs', 64, 32))],
        'bindings': [asdict(PointerBinding(0, 0, 0, 12)), asdict(PointerBinding(1, 0, 16, 4)),
                     asdict(PointerBinding(2, 1, 16, 32))],
        'binding_provenance': 'explicit test allocation', 'storage_reused': False,
        'alignment_or_noalias_added': False, 'whole_cfg_preserved_under_pointer_substitution': True}
    facts = CompactTargetABI((32, 32), (16, 16), (2, 4, 4), ('read', 'write'), 'test ABI facts, not inferred from host')
    payloads = {'a': b'\xff\xff\x00\x80\x01\x00\xff\x7f', 'scale': b'\x45\x23\xc1\x7f'}
    return cb, contract, payloads, facts


def resolve(data, **kwargs):
    cb, contract, payloads, facts = data
    return resolve_compact_binding(cb, contract, payloads, target_abi=facts, max_storage_bytes=128, **kwargs)


def test_opaque_words_padding_permutation_and_exact_readback():
    data = fixture()
    result = resolve(data)
    views = result.original_argument_views()
    assert bytes(views[0]) == b'\xff\xff\x00\x80\0\0\x01\x00\xff\x7f\0\0'
    assert bytes(views[1]) == data[2]['scale']
    assert views[0].readonly and not views[2].readonly
    words = [b'\x00\x00\x00\x80', b'\x45\x23\xc1\x7f', b'\xff\xff\xff\xff', b'\x01\x00\x00\x00']
    for index, offset in enumerate((0, 12, 4, 16)):
        views[2][offset:offset+4] = words[index]
    assert result.readback() == {'out': b''.join(words)}
    evidence = result.setup_evidence()
    assert evidence['packing_inside_compute_roi'] is False and not evidence['storage_reused']
    assert evidence['numerical_equivalence'] == 'UNPROVEN'
    assert result.validate_runtime_addresses([0x1000, 0x2000]) == (0x1000, 0x1010, 0x2010)
    assert not evidence['runtime_addresses_validated']
    address_evidence = result.runtime_address_evidence([0x1000, 0x2000])
    assert address_evidence['alignment_and_nonoverlap_checked']
    assert not address_evidence['live_target_allocation_verified']
    assert not result.setup_evidence()['runtime_addresses_validated']


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'extent', 'bounds', 'overlap', 'unused',
    'width', 'offset_alignment', 'base_alignment', 'access', 'reuse', 'same_symbol', 'payload',
    'missing_input', 'write_initializer', 'missing_encoding', 'budget', 'bool_index'])
def test_malformed_or_unsupported_contracts_refuse(mutation):
    cb, contract, payloads, facts = fixture()
    if mutation == 'missing': contract['bindings'].pop()
    elif mutation == 'duplicate': contract['bindings'][1]['argument_index'] = 0
    elif mutation == 'extent': contract['bindings'][0]['byte_extent'] = 8
    elif mutation == 'bounds': contract['bindings'][2]['byte_offset'] = 48
    elif mutation == 'overlap': contract['bindings'][1]['byte_offset'] = 4
    elif mutation == 'unused':
        contract['bases'].append(asdict(BaseBuffer('unused', 1, 32)))
        contract['compact_argument_count'] = 3
    elif mutation == 'width': facts = replace(facts, pointer_index_bits=(64, 64))
    elif mutation == 'offset_alignment': contract['bindings'][1]['byte_offset'] = 17
    elif mutation == 'base_alignment': facts = replace(facts, base_alignments=(2, 16))
    elif mutation == 'access': facts = replace(facts, base_access=('readwrite', 'write'))
    elif mutation == 'reuse': contract['storage_reused'] = True
    elif mutation == 'same_symbol': contract['compact_symbol'] = 'original'
    elif mutation == 'payload': payloads['a'] = payloads['a'][:-1]
    elif mutation == 'missing_input': del payloads['scale']
    elif mutation == 'write_initializer': payloads['out'] = bytes(16)
    elif mutation == 'missing_encoding': del cb['params']['storage_encodings']['scale']
    elif mutation == 'budget': contract['bases'][0]['byte_extent'] = 65
    elif mutation == 'bool_index': contract['bindings'][0]['base_index'] = False
    with pytest.raises(ValueError):
        resolve((cb, contract, payloads, facts))


def test_readonly_returned_payload_and_runtime_address_obligations():
    result = resolve(fixture())
    payloads = [bytes(view) for view in result.base_views()]
    payloads[0] = b'\0'+payloads[0][1:]
    with pytest.raises(ValueError, match='read-only'):
        result.readback(payloads)
    for addresses in ([0x1001, 0x2000], [0x1000, 0x1020], [0, 0x2000], [0x1000]):
        with pytest.raises(ValueError):
            result.validate_runtime_addresses(addresses)


def test_nontrivial_input_permutation_cannot_bypass_prepack_authority():
    data = fixture()
    data[0]['params']['storage_encodings']['a'] = GroupedAxesStorage(
        (2, 2), 'i16', ((1,), (0,)), (2, 2), (3, 1), 6).to_dict()
    with pytest.raises(StoragePrepackRequired):
        resolve(data)
    with pytest.raises(ValueError, match='exact host authorization'):
        resolve(data, prepack_authorizations={'a': {'approved': True}})


def test_host_authorization_rechecks_exact_cb_encoding_and_payload():
    from merlin.frontends.argument_identity import ArgumentIdentityBridge
    from merlin.runtime.captured_constants import CapturedConstant
    from merlin.runtime.prepack_authority import HostPrepackAuthorization
    data = fixture()
    cb, _, payloads, _ = data
    encoding = GroupedAxesStorage((2, 2), 'i16', ((1,), (0,)), (2, 2), (3, 1), 6)
    cb['params']['storage_encodings']['a'] = encoding.to_dict()
    payload = payloads['a']
    digest = lambda value: hashlib.sha256(value).hexdigest()
    # Typed host test authority, not serialized candidate metadata or a claim of
    # real capture verification. Production constructs this through capture replay.
    constant = CapturedConstant(0, 'fixture.manifest', digest(b'manifest'), 'fixture.safetensors',
        digest(b'file'), 'a', 'param', (2, 2), 'i16', 'I16', digest(b'header'), 0, digest(payload), payload)
    bridge = ArgumentIdentityBridge(digest(b'raw'), digest(b'normalized'), 'original',
                                   ('tensor<2x2xi16>',), (), ())
    grant = HostPrepackAuthorization('a', digest(json.dumps(cb, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()), encoding, constant, bridge)
    result = resolve(data, prepack_authorizations={'a': grant})
    assert bytes(result.original_argument_views()[0]) == payload[:2]+payload[4:6]+b'\0\0'+payload[2:4]+payload[6:]+b'\0\0'
    payloads['a'] = bytes(len(payload))
    with pytest.raises(ValueError, match='actual initializer'):
        resolve(data, prepack_authorizations={'a': grant})
    payloads['a'] = payload
    cb['metadata_changed'] = True
    with pytest.raises(ValueError, match='does not bind'):
        resolve(data, prepack_authorizations={'a': grant})


def test_alternate_explicit_index_profile_and_readwrite_group():
    cb, contract, payloads, facts = fixture()
    for base in contract['bases']:
        base['pointer_index_bits'] = 64
    for arg in cb['kernel_abi']['args'][:2]:
        arg['access'] = 'readwrite'
    cb['kernel_abi']['outputs'].append('a')
    facts = replace(facts, pointer_index_bits=(64, 64), base_access=('readwrite', 'write'))
    result = resolve((cb, contract, payloads, facts))
    result.original_argument_views()[0][:2] = b'\x10\x20'
    assert result.readback()['a'][:2] == b'\x10\x20'


def test_trusted_tiny_native_original_vs_compact_addresses(tmp_path):
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('native C compiler unavailable')
    # Trusted fixture code, not a candidate compiler/kernel. Only address/byte
    # equivalence is tested; neither target ABI discovery nor model execution.
    source = tmp_path / 'address.c'
    source.write_text('''#include <stddef.h>
void original(const unsigned char *a, const unsigned char *s, unsigned char *o) {
  for (size_t i=0;i<4;i++) o[i]=s[i];
  for (size_t i=0;i<8;i++) o[4+i]=a[i];
}
void compact(const unsigned char *readbase, unsigned char *writebase) {
  original(readbase, readbase+16, writebase+16);
}
''')
    library = tmp_path / 'address.so'
    built = subprocess.run([compiler, '-shared', '-fPIC', '-O0', str(source), '-o', str(library)],
                           capture_output=True, timeout=20)
    assert built.returncode == 0, built.stderr.decode()
    native = ctypes.CDLL(str(library))
    native.original.argtypes = [ctypes.c_void_p]*3
    native.compact.argtypes = [ctypes.c_void_p]*2
    native.original.restype = native.compact.restype = None
    result = resolve(fixture())
    # Actual native allocation alignment is observed and validated separately.
    owners = [ctypes.create_string_buffer(64+15) for _ in range(2)]
    addresses = [(ctypes.addressof(owner)+15)//16*16 for owner in owners]
    original_addresses = result.validate_runtime_addresses(addresses)
    for address, view in zip(addresses, result.base_views(), strict=True):
        ctypes.memmove(address, bytes(view), len(view))
    native.original(*original_addresses)
    expected = ctypes.string_at(addresses[1], 64)
    ctypes.memset(addresses[1], 0, 64)
    native.compact(*addresses)
    assert ctypes.string_at(addresses[1], 64) == expected
    output = result.readback([ctypes.string_at(address, 64) for address in addresses])
    assert output['out'][:4] == fixture()[2]['scale']
