"""Positional naming is a bijective ABI rename, never numerical qualification."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from merlin.targetgen.capsule_runner import _bind_positional_output_name
from merlin.targetgen.package_certification import _qualifying_pointer_abi
from merlin.targetgen.owned_host_model_execution import _require_instruction_classes


def fixture():
    return {
        'operand_naming': 'positional',
        'tensors': {
            'arg0': {'role': 'input', 'shape': [2], 'dtype': 'i8'},
            'out0': {'role': 'output', 'shape': [2], 'dtype': 'i32'},
            'arg1': {'role': 'input', 'shape': [2], 'dtype': 'i8'},
        },
        'outputs': ['out0'], 'arg_order': ['arg0', 'out0', 'arg1'],
        'compiler_pointer_abi': {'version': 1, 'arguments': ['arg0', 'out0', 'arg1']},
        'commands': [{'opcode': 'COMMIT', 'operands': {'src': 'tile', 'dst': 'out0'}, 'attrs': {'n': 2}}],
    }


def test_positional_name_change_preserves_complete_pointer_order_and_data():
    cb = fixture()
    old = deepcopy(cb)
    _bind_positional_output_name(cb, {'operation': {'attributes': {'out': 'Y0'}}})
    assert list(cb['tensors']) == ['arg0', 'Y0', 'arg1']
    assert cb['tensors']['Y0'] == old['tensors']['out0']
    assert cb['outputs'] == ['Y0']
    assert cb['arg_order'] == cb['compiler_pointer_abi']['arguments'] == ['arg0', 'Y0', 'arg1']
    assert cb['commands'] == [{'opcode': 'COMMIT', 'operands': {'src': 'tile', 'dst': 'Y0'}, 'attrs': {'n': 2}}]
    backend = SimpleNamespace(kernel_abi_from_commands=lambda command: {'args': [{'tensor': n} for n in command['arg_order']]})
    assert _qualifying_pointer_abi(cb, backend)['status'] == 'pass'
    once = deepcopy(cb)
    _bind_positional_output_name(cb, {'operation': {'attributes': {'out': 'Y0'}}})
    assert cb == once


@pytest.mark.parametrize('change', ['collision', 'missing_pointer', 'duplicate_pointer', 'malformed_pointer'])
def test_positional_invalid_names_refuse_before_any_mutation(change):
    cb = fixture()
    if change == 'collision': cb['tensors']['Y0'] = {'role': 'input', 'shape': [2], 'dtype': 'i8'}
    if change == 'missing_pointer': cb['compiler_pointer_abi']['arguments'] = ['arg0', 'arg1']
    if change == 'duplicate_pointer': cb['compiler_pointer_abi']['arguments'].append('out0')
    if change == 'malformed_pointer': cb['compiler_pointer_abi'] = ['out0']
    before = deepcopy(cb)
    with pytest.raises(Exception, match='positional output'):
        _bind_positional_output_name(cb, {'operation': {'attributes': {'out': 'Y0'}}})
    assert cb == before


def test_no_flag_no_inferred_output_names_or_multioutput_reordering():
    cb = fixture(); del cb['operand_naming']; before = deepcopy(cb)
    _bind_positional_output_name(cb, {'operation': {'attributes': {'out': 'Y0'}}})
    assert cb == before
    cb['interface'] = 'linalg_positional'
    cb['tensors']['other'] = {'role': 'output', 'shape': [2], 'dtype': 'i32'}
    before = deepcopy(cb)
    _bind_positional_output_name(cb, {'operation': {'attributes': {'out': 'Y0'}}})
    assert cb == before


def test_instruction_class_failure_names_exact_declared_difference():
    with pytest.raises(ValueError, match="missing=\\['COMPUTE'\\]") as error:
        _require_instruction_classes(['LOAD', 'COMPUTE'], ['STORE', 'LOAD'])
    assert "observed=['LOAD', 'STORE']" in str(error.value)
    assert "required=['COMPUTE', 'LOAD']" in str(error.value)
    _require_instruction_classes(['LOAD'], ['COMPUTE', 'LOAD'])
