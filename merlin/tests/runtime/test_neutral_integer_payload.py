"""Canonical payload packing preserves exact bytes and existing refusals."""

import array
import base64
import copy
from dataclasses import replace
import struct
from types import SimpleNamespace

import pytest

from merlin.common.quant_formats import get
from merlin.runtime import harness_render as H


INTEGER_DTYPES = [(f'{prefix}{bits}', bits // 8, prefix == 'i')
                  for prefix in ('i', 'u') for bits in (8, 16, 32, 64)]


def payload(value, dtype='i16', shape=(2, 3), spec=None):
    # The repository has no registered unsigned scalar formats. Exercise the
    # generic packing branch with an explicit synthetic descriptor; this grants
    # no new public dtype or renderer capability.
    descriptor = replace(get('i' + dtype[1:]), signed=False) if dtype.startswith('u') else get(dtype)
    extent = descriptor.element_bits // 8
    for dimension in shape:
        extent *= dimension
    return H._payload(value, spec or {}, shape, descriptor, extent)


@pytest.mark.parametrize('dtype,width,signed', INTEGER_DTYPES)
@pytest.mark.parametrize('container', [list, tuple])
def test_flat_native_array_bytes_match_independent_integer_words(dtype, width, signed, container):
    low = -(1 << (width * 8 - 1)) if signed else 0
    high = (1 << (width * 8 - int(signed))) - 1
    values = container([low, high, 0, 1, -1 if signed else high - 1, 17])
    before = copy.deepcopy(values)
    expected = b''.join(value.to_bytes(width, 'little', signed=signed) for value in values)
    assert base64.b64decode(payload(values, dtype)) == expected
    assert values == before
    assert payload([list(values[:3]), list(values[3:])], dtype) == payload(values, dtype)
    if signed:
        assert payload({'shape': [2, 3], 'dtype': dtype, 'values': values}, dtype) == payload(values, dtype)


@pytest.mark.parametrize('dtype,width,signed', INTEGER_DTYPES)
@pytest.mark.parametrize('boundary', ['below', 'above'])
def test_integer_range_refusal_is_unchanged(dtype, width, signed, boundary):
    low = -(1 << (width * 8 - 1)) if signed else 0
    high = (1 << (width * 8 - int(signed))) - 1
    bad = low - 1 if boundary == 'below' else high + 1
    with pytest.raises(ValueError, match='outside its storage range'):
        payload([0, bad], dtype, (2,))


class IntegerSubclass(int):
    pass


@pytest.mark.parametrize('bad', [True, False, IntegerSubclass(7), 1.0])
@pytest.mark.parametrize('nested', [False, True])
def test_exact_integer_requirement_keeps_bool_and_subclasses_out(bad, nested):
    value = [[0, bad]] if nested else [0, bad]
    with pytest.raises(ValueError, match='exact integers'):
        payload(value, shape=(1, 2))


@pytest.mark.parametrize('mutation', ['shape', 'bool_dimension', 'dtype', 'unknown_dtype', 'extra', 'missing_values'])
def test_typed_metadata_is_checked_before_packing(mutation):
    value = {'shape': [2, 3], 'dtype': 'i16', 'values': [0, 1, 2, 3, 4, 5]}
    if mutation == 'shape': value['shape'] = [3, 2]
    if mutation == 'bool_dimension': value['shape'] = [True, 6]
    if mutation == 'dtype': value['dtype'] = 'u16'
    if mutation == 'unknown_dtype': value['dtype'] = 'unknown'
    if mutation == 'extra': value['stride'] = 3
    if mutation == 'missing_values': value.pop('values')
    with pytest.raises(ValueError): payload(value)


def test_preload_bytes_extent_and_nested_fallback_remain_exact():
    values = [1, -2, 3, -4, 5, -6]
    encoded = payload(values)
    assert payload(values, spec={'preload_b64': encoded}) == encoded
    assert payload(struct.pack('<6h', *values)) == encoded
    with pytest.raises(ValueError, match='declared source bytes'):
        payload(values, spec={'preload_b64': base64.b64encode(bytes(12)).decode()})
    for bad in ([1, 2], [[1, 2], [3, 4, 5]], [[1, 2, 3], [4, 5, [6]]], bytes(11), bytes(13)):
        with pytest.raises(ValueError): payload(bad)


@pytest.mark.parametrize('dtype,code,values', [
    ('f16', 'e', [0.0, -0.0, 1.5, -2.0, 16.0, .25]),
    ('f32', 'f', [0.0, -0.0, 1.5, -2.0, 16.0, .25]),
    ('f64', 'd', [0.0, -0.0, 1.5, -2.0, 16.0, .25]),
])
def test_float_path_retains_signed_zero_and_exact_representation(dtype, code, values, monkeypatch):
    monkeypatch.setattr(H, 'array', SimpleNamespace(array=lambda *a: pytest.fail('float selected integer packing')))
    assert base64.b64decode(payload(values, dtype)) == struct.pack('<6' + code, *values)
    with pytest.raises(ValueError): payload([float('nan')] * 6, dtype)
    with pytest.raises(ValueError): payload([1] * 6, dtype)
    if dtype != 'f64':
        with pytest.raises(ValueError, match='rounding'): payload([0.1] * 6, dtype)


@pytest.mark.parametrize('dtype,width,signed', INTEGER_DTYPES)
@pytest.mark.parametrize('native_order', ['little', 'big'])
def test_native_endian_normalization_is_canonical(dtype, width, signed, native_order, monkeypatch):
    # Changing sys.byteorder alone would not emulate native arrays. This small
    # boundary fake serializes validated array values in the declared host order.
    swaps = []
    class NativeArray:
        def __init__(self, code, values=()):
            self.native = array.array(code, values)
            self.itemsize = self.native.itemsize
            self.order = native_order
            self.code = code
        def byteswap(self):
            swaps.append(self.code)
            self.order = 'big' if self.order == 'little' else 'little'
        def tobytes(self):
            return b''.join(int(v).to_bytes(self.itemsize, self.order, signed=self.code.islower())
                            for v in self.native)
    monkeypatch.setattr(H, 'array', SimpleNamespace(array=NativeArray))
    monkeypatch.setattr(H, 'sys', SimpleNamespace(byteorder=native_order))
    high = (1 << (width * 8 - int(signed))) - 1
    values = [0, 1, high, -1 if signed else 2, 17, 31]
    expected = b''.join(v.to_bytes(width, 'little', signed=signed) for v in values)
    assert base64.b64decode(payload(values, dtype)) == expected
    assert len(swaps) == (1 if native_order == 'big' else 0)


def test_platform_itemsize_mismatch_keeps_original_word_packer(monkeypatch):
    calls = []
    def wrong_width(code, *args):
        calls.append(code)
        assert not args, 'unsupported native layout received scalar payload'
        return SimpleNamespace(itemsize=7)
    monkeypatch.setattr(H, 'array', SimpleNamespace(array=wrong_width))
    values = [-(1 << 63), (1 << 63) - 1, -7, 0, 1, 123]
    assert base64.b64decode(payload(values, 'i64')) == struct.pack('<6q', *values)
    assert calls == ['q']
    with pytest.raises(ValueError, match='outside its storage range'):
        payload([1 << 64], 'i64', (1,))


def test_ordinary_renderer_consumes_identical_flat_nested_and_byte_payloads():
    from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64
    contract = {'harness_abi': {
        'version': 2, 'kind': 'logical_pointer', 'entry_symbol': 'control_entry',
        'fence_symbol': 'control_complete', 'tensor_alignment': 8,
        'byte_order': 'little', 'main_convention': 'void',
        'readback_transport': FULL_VALUES_B64}}
    cb = {'tensors': {
        'X': {'shape': [2, 3], 'dtype': 'i64', 'role': 'input'},
        'Y': {'shape': [2, 3], 'dtype': 'i64', 'role': 'output'}}}
    values = [-(1 << 63), (1 << 63) - 1, (1 << 53) + 1, 0, 7, -11]
    sources = [H.render_harness(cb, target='synthetic', inputs={'X': value}, contract=contract)
               for value in (values, [values[:3], values[3:]], struct.pack('<6q', *values))]
    assert sources[0] == sources[1] == sources[2]
    assert 'control_entry(tensor_0, tensor_1);' in sources[0]
    assert 'control_complete();' in sources[0]
    assert 'OUT_B64_BEGIN v1 Y 2 3 8 s' in sources[0]
