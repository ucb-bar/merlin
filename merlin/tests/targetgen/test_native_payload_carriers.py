"""Literal and unknown payloads cannot escape across region or CFG carriers."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_unit_policy import construction_evidence


CARRIERS = [
    '''%r = scf.for %i = %lo to %hi step %step iter_args(%v = %seed) -> (tensor<2x2xi32>) {
         scf.yield %v : tensor<2x2xi32>
       }''',
    '''%r = scf.for %i = %lo to %hi step %step iter_args(%v = %x) -> (tensor<2x2xi32>) {
         scf.yield %seed : tensor<2x2xi32>
       }''',
    '''%r = scf.if %condition -> (tensor<2x2xi32>) {
         scf.yield %x : tensor<2x2xi32>
       } else { scf.yield %seed : tensor<2x2xi32> }''',
    '''%r = scf.while (%v = %seed) : (tensor<2x2xi32>) -> tensor<2x2xi32> {
         scf.condition(%condition) %v : tensor<2x2xi32>
       } do { ^bb0(%v: tensor<2x2xi32>): scf.yield %v : tensor<2x2xi32> }''',
    '''%r = scf.while (%v = %x) : (tensor<2x2xi32>) -> tensor<2x2xi32> {
         scf.condition(%condition) %v : tensor<2x2xi32>
       } do { ^bb0(%v: tensor<2x2xi32>): scf.yield %seed : tensor<2x2xi32> }''',
    '''%r = "scf.execute_region"() ({
         cf.cond_br %condition, ^left(%x : tensor<2x2xi32>), ^right(%seed : tensor<2x2xi32>)
       ^left(%a: tensor<2x2xi32>): cf.br ^join(%a : tensor<2x2xi32>)
       ^right(%b: tensor<2x2xi32>): cf.br ^join(%b : tensor<2x2xi32>)
       ^join(%v: tensor<2x2xi32>): scf.yield %v : tensor<2x2xi32>
       }) : () -> tensor<2x2xi32>''',
    '''%r = "scf.index_switch"(%lo) <{cases = array<i64: 0>}> ({ scf.yield %x : tensor<2x2xi32> },
         { scf.yield %seed : tensor<2x2xi32> }) : (index) -> tensor<2x2xi32>''',
]


def module(carrier, table):
    seed = ('%seed = arith.constant dense<[[1,2],[3,4]]> : tensor<2x2xi32>' if table else
            '%seed = tensor.cast %x : tensor<2x2xi32> to tensor<2x2xi32>')
    return parse_mlir_text('''module { func.func @f(%x: tensor<2x2xi32>, %condition: i1) -> tensor<2x2xi32> {
      %lo = arith.constant 0 : index
      %hi = arith.constant 2 : index
      %step = arith.constant 1 : index
      ''' + seed + '\n' + carrier + '\nfunc.return %r : tensor<2x2xi32> }}')


@pytest.mark.parametrize('carrier', CARRIERS)
@pytest.mark.parametrize('table', [False, True])
def test_all_carrier_inputs_and_backedges_are_screened(carrier, table):
    m = module(carrier, table)
    m.verify()
    result = construction_evidence(m)
    assert result['status'] == ('host' if table else 'eligible'), result
    assert ('nonuniform_literal_result_construction' in {e['reason'] for e in result['evidence']}) is table


@pytest.mark.parametrize('table', [False, True])
@pytest.mark.parametrize('arithmetic_after', [False, True])
def test_unknown_carrier_fails_closed_even_after_known_arithmetic(table, arithmetic_after):
    carrier = '%r = "unknown.payload"(%seed) : (tensor<2x2xi32>) -> tensor<2x2xi32>'
    if arithmetic_after:
        carrier = carrier.replace('%r =', '%u =') + '''
          %r = arith.addi %u, %x : tensor<2x2xi32>
        '''
    result = construction_evidence(module(carrier, table), excluded=[])
    assert result['status'] == 'host'
    assert any(e['reason'] == 'unproved_payload_carrier' for e in result['evidence'])


def test_uninitialized_payload_is_not_proved_to_be_nonliteral_movement():
    result = construction_evidence(module('%r = tensor.empty() : tensor<2x2xi32>', False))
    assert result['status'] == 'host'


DEQUANTIZE = '''module {func.func @f(%x: tensor<4xi8>) -> tensor<4xf32> {
      %scale = arith.constant dense<1.0> : tensor<f32>
      %zero = arith.constant dense<0> : tensor<i64>
      DATA
      %r = "quant_ext.dequantize_per_tensor"(CODES,%scale,%zero) :
        (tensor<4xi8>,tensor<f32>,tensor<i64>) -> tensor<4xf32>
      func.return %r : tensor<4xf32>
    }}'''
QUANT = DEQUANTIZE.replace('DATA\n      ', '').replace('CODES', '%x')


@pytest.mark.parametrize('table', [False, True])
def test_validated_numeric_boundary_preserves_its_data_provenance(table):
    data = ('%codes = arith.constant dense<[1,2,3,4]> : tensor<4xi8>' if table else
            '%codes = tensor.cast %x : tensor<4xi8> to tensor<4xi8>')
    text = DEQUANTIZE.replace('DATA', data).replace('CODES', '%codes')
    result = construction_evidence(parse_mlir_text(text))
    assert result['status'] == ('host' if table else 'eligible'), result
    if table:
        assert any(e['reason'] == 'nonuniform_literal_result_construction' for e in result['evidence'])


@pytest.mark.parametrize('mutation', ['unknown_boundary', 'invalid_arity', 'unproved_input', 'unknown_metadata',
                                      'channel_without_axis', 'channel_axis_out_of_range'])
def test_quant_spelling_cannot_authenticate_an_unknown_or_invalid_carrier(mutation):
    text = QUANT
    if mutation == 'unknown_boundary':
        text = text.replace('dequantize_per_tensor', 'unknown_movement')
    elif mutation == 'invalid_arity':
        text = text.replace('%x,%scale,%zero','%x,%scale').replace('tensor<4xi8>,tensor<f32>,tensor<i64>','tensor<4xi8>,tensor<f32>')
    elif mutation.startswith('channel'):
        axis = '' if mutation == 'channel_without_axis' else ' <{axis = 1 : i64}>'
        text = text.replace('per_tensor"(%x,%scale,%zero)', 'per_channel"(%x,%scale,%zero)' + axis)
    elif mutation == 'unproved_input':
        text = text.replace('%r =', '%opaque = "unknown.payload"(%x) : (tensor<4xi8>) -> tensor<4xi8>\n%r =').replace('(%x,%scale,%zero)','(%opaque,%scale,%zero)')
    else:
        text = text.replace(') :\n        (tensor<4xi8>', ') {prov.quant.unknown = "bad"} :\n        (tensor<4xi8>')
    result = construction_evidence(parse_mlir_text(text), excluded=[])
    assert result['status'] == 'host', result
    assert any(e['reason'] == 'unproved_payload_carrier' for e in result['evidence'])


@pytest.mark.parametrize('table', [False, True])
def test_reduction_region_arguments_have_explicit_input_and_seed_provenance(table):
    source = ('%seed = arith.constant dense<[1,2]> : tensor<2xi32>' if table else
              '%seed = tensor.cast %x : tensor<2xi32> to tensor<2xi32>')
    m = parse_mlir_text('''module {func.func @f(%x:tensor<2xi32>,%a:tensor<2x2xi32>) -> tensor<2xi32> {
      '''+source+'''
      %r = linalg.reduce ins(%a:tensor<2x2xi32>) outs(%seed:tensor<2xi32>) dimensions=[1]
        (%v:i32,%acc:i32) { linalg.yield %acc : i32 }
      func.return %r : tensor<2xi32>
    }}''')
    m.verify()
    result = construction_evidence(m)
    assert result['status'] == ('host' if table else 'eligible'), result
