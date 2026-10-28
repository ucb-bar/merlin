"""Shared fusion graphs route each declared runtime scale to the composition oracle."""
import copy

import pytest
from merlin.targetgen.capsule_golden import UnsupportedGoldenFormat, golden


def _graph():
    def leaf(name, dtype='i8', shape=None):
        return dict(name=name, dtype=dtype, shape=shape or [1, 1], role='input')

    scheme = dict(id='q_f32_tensor', status='resolved', unknown=[], graded=True,
                  weight_scale_granularity='tensor', activation_scale_granularity='tensor',
                  readout_scale_granularity='tensor', readout_strategy='f32_product_rne',
                  scale_value='arbitrary_f32', operand_sum_rounding='single_rounding',
                  exactness_gate=dict(definitional='f32_product_rne',
                                     requires='decoded_multiplier_bits_equal_formed_multiplier'))
    node = dict(inputs=[leaf('A'), leaf('W0')], operation=dict(op='composition_program', attributes=dict(
        M=1, K=1, N=1, case=dict(producer='contraction', exit='accumulator',
            accumulator_stages=[], readout_stages=[], multipliers=[1.0]),
        quantization_scheme=scheme, outputs=['mac0'])))
    runtime = copy.deepcopy(node)
    runtime['inputs'].append(leaf('ReadoutScale', 'f32', [1]))
    child = runtime['operation']['attributes']
    child['case'].update(exit='requant', runtime_scale_input='ReadoutScale',
                         readout_stages=['convert', 'scale', 'requant'])
    child['quantization_scheme']['scale_source'] = 'runtime_input'
    child['outputs'] = ['readout']
    cap = copy.deepcopy(node)
    cap['inputs'] = [leaf('A'), leaf('node0_W0'), leaf('node1_W0'), leaf('node1_ReadoutScale', 'f32', [1])]
    cap['operation']['attributes'].update(validated_graph=[node, runtime], validated_structure='shared_input',
        validated_output_shapes=[[1, 1], [1, 1]], outputs=['node0_mac0', 'node1_readout'])
    cap['input_palette'] = dict(schema='merlin.component_input_palette.v1', inputs=[
        dict(name=name, axis='linear', offset=0, values=[value]) for name, value in
        [('A', 2), ('node0_W0', 3), ('node1_W0', 5), ('node1_ReadoutScale', 0.5)]])
    return cap


def test_second_node_runtime_scale_has_exact_complete_golden():
    assert golden(_graph()) == {'node0_mac0': [[6]], 'node1_readout': [[5]]}


def test_unbound_float_data_operand_still_refuses():
    cap = _graph()
    cap['inputs'][1]['dtype'] = 'f32'
    with pytest.raises(UnsupportedGoldenFormat, match='node0_W0'):
        golden(cap)


def test_declared_runtime_scale_still_requires_scalar_storage():
    cap = _graph()
    cap['inputs'][-1]['shape'] = [2]
    cap['input_palette']['inputs'][-1]['values'] = [0.5, 0.25]
    with pytest.raises(ValueError, match='shape differs|f32'):
        golden(cap)
