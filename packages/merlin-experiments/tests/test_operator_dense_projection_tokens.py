"""Exact ASCII scanner boundaries; registered typed validation remains mandatory."""
import io

import pytest
from merlin_experiments.phase0 import operator_dense_projection as P


@pytest.mark.parametrize('name', [b'0', b'123', b'x', b'_a', b'.', b'$', b'-', b'a.3_$-'])
@pytest.mark.parametrize('space', [b'', b' ', b'\t', b' \t '])
def test_original_bare_ssa_and_physical_whitespace(name, space):
    prefix = space + b'%' + name + space + b'=' + space + b'arith.constant'
    separator = space or b' '
    assert P._header(prefix + separator + b'dense<') == (prefix, None, len(prefix + separator) + 6)


@pytest.mark.parametrize('name', [b'', b'"quoted"', b'a:b', b'a/b', b'a,b', b'\xc3\xa9', b'a\x00b'])
def test_unknown_ssa_spelling_stays_original_unprojected_syntax(name):
    assert P._header(b'%' + name + b' = arith.constant dense<') is None


@pytest.mark.parametrize('suffix', [b'x', b'0', b'_', b'.foo', b'\n', b'\r', b'\v', b'\x00'])
def test_operation_spelling_and_nonhorizontal_separator_do_not_project(suffix):
    assert P._header(b'%0 = arith.constant' + suffix + b'dense<') is None


@pytest.mark.parametrize('attrs', [b'{}', b'{prov.kind = "a } dense< \\" b"}',
                                  b'{prov.info = {child = "nested"}}'])
def test_balanced_original_attribute_dictionary_is_retained(attrs):
    prefix = b'%0 = arith.constant'
    raw = prefix + b' ' + attrs + b' dense<'
    assert P._header(raw) == (prefix, attrs, len(raw))


@pytest.mark.parametrize('suffix,expected', [('', True), ('_0', True), ('_01_2', True),
    ('_', False), ('__1', False), ('_1_', False), ('_x', False), ('x', False),
    ('_\u0661', False), ('_\uff11', False), ('_1\n', False)])
def test_resource_uniquification_is_ascii_decimal_only(suffix, expected):
    assert P._resource_handle('base' + suffix, 'base') is expected


def test_header_retains_original_finite_prefix_bound():
    assert P._header(b' ' * 4096 + b'%x = arith.constant dense<') is None
    for n in (4064, 4065, 4096, 8192):
        raw = b'%x = arith.constant {prov.x = "' + b'x' * n + b'"} dense<'
        assert P._header(raw) is None


@pytest.mark.parametrize('chunk', [1, 2, 3, 7, 31, 65536])
def test_every_nonhex_byte_refuses_before_initializer_ownership(chunk):
    for value in range(256):
        if value in b'0123456789abcdefABCDEF"':
            continue
        raw = b'%x = arith.constant dense<"0x00' + bytes([value]) + b'00"> : tensor<2xi8>\n'
        with pytest.raises(ValueError):
            P.scan(io.BytesIO(raw), max_input_bytes=8192, max_compact_bytes=4096,
                   chunk_bytes=chunk)
