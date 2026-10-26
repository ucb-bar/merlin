"""The public resolver and runner share declaration-only shapes and pointer order."""
import copy
import json

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.compile import logical_abi as public
from merlin.llvmlower.host_builder import FnBuilder


def selected_document():
    return yaml.safe_load((repo_root()/'merlin/contract/mlir_oot_backend_contract.yaml').read_text())


def implicit_commit():
    return {'tensors': {
        'weights': {'shape': [6, 5], 'dtype': 'i8', 'role': 'weight'},
        'activations': {'shape': [3, 6], 'dtype': 'i8', 'role': 'input'}},
        'commands': [
            {'opcode': 'RES_PACK', 'operands': {'src': 'weights', 'dst': 'resident'}},
            {'opcode': 'MATMUL_RESIDENT', 'operands': {
                'lhs': 'activations', 'rhs': 'resident', 'dst': 'acc'}},
            {'opcode': 'COMMIT', 'operands': {'src': 'acc', 'dst': 'result'},
             'attributes': {'output_dtype': 'i32', 'epilogue': []}}]}


@pytest.mark.parametrize('pooled', [False, True])
def test_implicit_commit_shapes_and_declared_order_are_shared_with_runner(pooled):
    from merlin.targetgen.contract import harness_render as runner
    from merlin.targetgen.contract.interface_emit import _commit_out_shape

    selected = public.logical_abi_from_document(selected_document())
    cb = implicit_commit()
    shape = (3, 5)
    if pooled:
        cb['tensors']['activations']['shape'] = [12, 6]
        cb['commands'][-1]['attributes'].update(epilogue=['maxpool'], pool_in_dims=[3, 4],
            pool_size=[2, 2], pool_stride=[1, 1], pool_padding=[0, 0, 0, 0])
        shape = (6, 5)
    buffers = public.logical_interface(cb, selected)
    assert [(b.name, b.shape) for b in buffers] == [
        ('weights', (6, 5)), ('activations', tuple(cb['tensors']['activations']['shape'])), ('result', shape)]
    assert _commit_out_shape(cb, 'acc', cb['commands'][-1]['attributes']) == shape
    assert runner.logical_interface is public.logical_interface
    assert runner.LogicalAbi is public.LogicalAbi and runner.HarnessRenderError is public.LogicalAbiError
    builder, pointers = FnBuilder.for_logical_kernel(cb, abi=selected)
    assert list(pointers) == ['weights', 'activations', 'result']
    assert pointers['activations'] is builder.entry.args[1]


def test_derived_inputs_and_command_output_dtype_keep_the_original_single_owner():
    from merlin.runtime import commandbuffer, tensor

    cb = implicit_commit()
    cb['tensors']['packed'] = {'shape': [3, 6], 'dtype': 'i8', 'role': 'input'}
    cb['params'] = {'im2col_recipes': [{'source': 'activations', 'target': 'packed'}]}
    cb['tensors']['result'] = {'shape': [3, 5], 'dtype': 'i8', 'role': 'output'}
    cb['commands'][1]['operands']['lhs'] = 'packed'
    cb['commands'][-1]['attributes']['output_dtype'] = 'i32'
    abi = public.logical_abi_from_document(selected_document())
    buffers = public.logical_interface(cb, abi)
    assert [b.name for b in buffers] == ['weights', 'activations', 'result']
    assert buffers[-1].dtype == 'i32'
    assert commandbuffer.harness_derived_tensors is public.harness_derived_tensors
    assert commandbuffer.declared_output_dtypes is public.declared_output_dtypes
    assert commandbuffer.pool_params is public.pool_params
    assert tensor.pool_out_dims is public.pool_out_dims


def test_selected_declaration_loader_and_missing_selection_never_guess(tmp_path, monkeypatch):
    doc = selected_document()
    path = tmp_path/'selected-abi.json'
    path.write_text(json.dumps(doc))
    abi = public.logical_abi(path)
    assert abi == public.logical_abi_from_document(doc)
    monkeypatch.setenv('MERLIN_CONTRACT_DIR', str(tmp_path/'missing'))
    with pytest.raises(FileNotFoundError):
        public.kernel_arg_order(implicit_commit())
    builder, mapping = FnBuilder.for_logical_kernel(implicit_commit(), abi=abi)
    assert mapping['weights'] is builder.entry.args[0]
    broken = copy.deepcopy(doc)
    broken['logical_kernel_abi']['argument_order']['default'] = ['semantic_lhs_then_rhs']
    with pytest.raises(public.LogicalAbiError, match='does not resolve'):
        public.logical_abi_from_document(broken)
