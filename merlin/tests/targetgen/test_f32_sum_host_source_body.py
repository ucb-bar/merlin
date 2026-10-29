"""Actual sum roots and observed compiler bounds are required for host placement."""
import copy

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_f32_reductions import STATIC_F32_SUM_SOURCE_BODY_SCHEMA as SCHEMA
from merlin.targetgen.application_inventory import operation_structure
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities


def source(shape='2x3x7', *, seed='0.0', operation='arith.addf', flags='none'):
    dims = shape.split('x')
    output = 'x'.join(dims[:-1])
    tensor = ('tensor<' + output + 'xf32>') if output else 'tensor<f32>'
    module = mq.parse(f'''builtin.module {{
      func.func @f(%x: tensor<{shape}xf32>) -> {tensor} {{
        %zero = arith.constant {seed} : f32
        %init = tensor.splat %zero : {tensor}
        %out = "linalg.reduce"(%x, %init) <{{dimensions = array<i64: {len(dims)-1}>}}> ({{
          ^bb0(%value: f32, %acc: f32):
            %v = "{operation}"(%value, %acc) <{{fastmath = #arith.fastmath<{flags}>}}> : (f32, f32) -> f32
            "linalg.yield"(%v) : (f32) -> ()
        }}) {{prov.aten = "aten.sum.dim_IntList"}} : (tensor<{shape}xf32>, {tensor}) -> {tensor}
        func.return %out : {tensor}
      }}
    }}''')
    return next(mq.walk(module, 'linalg.reduce'))


def selected():
    declaration = dict(id='neutral_sum', ops=['aten.sum.dim_IntList'], placement='host',
        signature=dict(family='reduction', ordered_operand_dtypes=['f32', 'f32'], ordered_result_dtypes=['f32']),
        source_body=dict(schema=SCHEMA, operation='arith.addf'), numerical_contract=dict(status='unreviewed'))
    document = dict(schema='merlin.host_capabilities.v1', status='reviewed',
        compiler=dict(package_sha256='a'*64, dtype_strategy='int8_w8a8'), operations=[declaration], evidence={})
    validate_host_capabilities(document)
    return dict(neutral=dict(package_sha256='a'*64, capability_spec_sha256='b'*64,
                            dtype_strategy='int8_w8a8', capability_spec=document))


def context():
    return dict(selected_index_observation=dict(schema='merlin.selected-index-lowering.v1',
        compiler_requested='fixture', compiler_resolved='/fixture', compiler_sha256='c'*64,
        cross_flags=['--target=fixture'], data_layout='e-p:64:64', index_bits=64, scope='synthetic unit observation'))


def screen(ops, *, count=None, ctx=True, reference=None):
    first = reference if reference is not None else ops[0]
    structure = operation_structure(first)
    rank = len(structure['ordered_result_types'][0]['shape'])
    return admit_host_operation(selected(), dict(frontend_op='aten.sum.dim_IntList', mlir_operation='linalg.reduce',
        count=len(ops) if count is None else count, **structure),
        dict(family='reduction', ordered_operand_dtypes=['f32', 'f32'], ordered_result_dtypes=['f32'], rank=rank),
        source_operations=ops, source_context=context() if ctx else None)


@pytest.mark.parametrize('shape', ['7', '3x11', '2x5x9', '1x2x3x7'])
def test_sum_source_admission_is_shape_generic_without_numerical_credit(shape):
    one, two = source(shape), source(shape)
    result = screen((one, two))
    assert result['status'] == 'admitted'
    assert result['source_body_proof']['schema'] == SCHEMA
    assert len(result['source_body_proof']['patterns']) == 2
    assert selected()['neutral']['capability_spec']['operations'][0]['numerical_contract']['status'] == 'unreviewed'
    assert 'numerical_qualification' not in result


@pytest.mark.parametrize('changed', [dict(seed='1.0'), dict(seed='-0.0'),
                                    dict(operation='arith.mulf'), dict(flags='fast')])
def test_sum_source_body_mutations_cannot_inherit_placement(changed):
    assert screen((source(**changed),))['status'] == 'unsupported'


def test_missing_width_occurrences_alias_or_typed_shape_refuses():
    one = source()
    assert screen((one,), ctx=False)['status'] == 'unknown'
    assert screen((one,), count=2)['status'] == 'unsupported'
    assert screen((one, one))['status'] == 'unsupported'
    assert screen((one,), reference=source('3x7'))['status'] == 'unsupported'


@pytest.mark.parametrize('change', [dict(numerical_contract=dict(status='reviewed')),
    dict(ops=['aten.mean.dim']), dict(source_body=dict(schema=SCHEMA, operation='arith.mulf'))])
def test_sum_declaration_cannot_broaden_or_assert_numerics(change):
    document = copy.deepcopy(selected()['neutral']['capability_spec'])
    document['operations'][0].update(change)
    with pytest.raises(ValueError):
        validate_host_capabilities(document)
