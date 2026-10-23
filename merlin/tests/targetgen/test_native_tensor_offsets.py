"""Tensor literals and intervening casts cannot conceal biased rounding."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_unit_policy import construction_evidence


def decide(module):
    module.verify()
    return construction_evidence(module, excluded=['readout_offset_rounding'])


def source(offset, rounding, *, splat=False):
    constant = ('%s = arith.constant '+offset+' : f32\n%c = tensor.splat %s : tensor<4xf32>' if splat else
                '%c = arith.constant '+offset+' : tensor<4xf32>')
    return parse_mlir_text('''module { func.func @f(%x: tensor<4xf32>) -> tensor<4xf32> {
      ''' + constant + '''
      %e = tensor.empty() : tensor<4xf32>
      %r = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]}
        ins(%x,%c : tensor<4xf32>,tensor<4xf32>) outs(%e : tensor<4xf32>) {
        ^bb0(%a:f32,%b:f32,%unused:f32):
          %sum = arith.addf %a,%b : f32
          ''' + rounding + '''
          linalg.yield %v : f32
        } -> tensor<4xf32>
      func.return %r : tensor<4xf32>
    }}''')


ROUNDINGS = [
    '%n = arith.fptosi %sum : f32 to i32\n%v = arith.sitofp %n : i32 to f32',
    '%v = math.trunc %sum : f32',
    '%v = math.round %sum : f32',
    '%v = math.roundeven %sum : f32',
    # A narrowed value that feeds further arithmetic is not a storage boundary.
    '%n = arith.truncf %sum : f32 to f16\n%w = arith.extf %n : f16 to f32\n%v = arith.mulf %w,%a : f32',
    '%n = arith.truncf %sum : f32 to bf16\n%w = arith.extf %n : bf16 to f32\n%v = arith.mulf %w,%a : f32',
    '%wide = arith.extf %sum : f32 to f64\n%n = arith.fptosi %wide : f64 to i32\n%v = arith.sitofp %n : i32 to f32',
    '%one = arith.constant 1.0 : f32\n%p = arith.mulf %sum,%one : f32\n%v = math.floor %p : f32',
]


@pytest.mark.parametrize('rounding', ROUNDINGS)
@pytest.mark.parametrize('offset', ['dense<0.5>', 'dense<[0.0,0.5,0.0,-0.5]>'])
def test_dense_offsets_before_every_rounding_observation(rounding, offset):
    result = decide(source(offset, rounding))
    assert result['status'] == 'host', result
    assert any(e['reason'] == 'readout_offset_rounding' for e in result['evidence'])


@pytest.mark.parametrize('rounding', ROUNDINGS)
def test_scalar_splat_offset_is_not_an_escape(rounding):
    assert decide(source('0.5', rounding, splat=True))['status'] == 'host'


@pytest.mark.parametrize('rounding', ROUNDINGS)
def test_zero_tensor_offset_is_not_biased_rounding(rounding):
    assert decide(source('dense<-0.0>', rounding))['status'] == 'eligible'


# An IEEE float-to-float RNE narrowing (f32 -> f16 here) whose result is only
# stored rounds to the storage format's significand, not to an integer: it is
# the exact storage boundary of an ordinary affine result (eager f16/bf16
# add.Scalar computes x + c in f32 and rounds once to storage). Literal offsets
# before it, uniform or not, are not offset-rounding constructions. Magic-bias
# rounding through the same narrowing, (x + C) narrowed and widened, then the
# bias removed in any spelling, stays excluded.
STORAGE_RNE = '%n = arith.truncf %sum : f32 to f16\n%v = arith.extf %n : f16 to f32'


@pytest.mark.parametrize('offset', ['dense<0.5>', 'dense<[0.0,0.5,0.0,-0.5]>'])
def test_storage_rne_after_literal_offset_is_not_offset_rounding(offset):
    assert decide(source(offset, STORAGE_RNE))['status'] == 'eligible'
    assert decide(source('0.5', STORAGE_RNE, splat=True))['status'] == 'eligible'


@pytest.mark.parametrize('offset', ['dense<1536.0>', 'dense<[1536.0,3072.0,1536.0,6144.0]>'])
def test_magic_bias_through_storage_rne_is_excluded(offset):
    removal = '%w = arith.extf %n : f16 to f32\n%v = arith.subf %w,%b : f32'
    result = decide(source(offset, STORAGE_RNE.replace('%v = arith.extf %n : f16 to f32', removal)))
    assert result['status'] == 'host', result


def test_literal_affine_offset_without_rounding_remains_eligible():
    assert construction_evidence(source('dense<0.5>', '%v = arith.addf %sum,%a : f32'))['status'] == 'eligible'


def single(body):
    return parse_mlir_text('''module { func.func @f(%x: tensor<4xf32>) -> tensor<4xf32> {
      %e = tensor.empty() : tensor<4xf32>
      %r = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]}
        ins(%x : tensor<4xf32>) outs(%e : tensor<4xf32>) {
        ^bb0(%a:f32,%unused:f32):
          ''' + body + '''
          linalg.yield %v : f32
        } -> tensor<4xf32>
      func.return %r : tensor<4xf32>
    }}''')


# Round half away from zero, x + copysign(0.5, x), is a literal offset; copysign(x, 0.5) = |x|
# carries runtime magnitude, so adding it is not a literal bias.
@pytest.mark.parametrize('copysign,status', [('%h, %a', 'host'), ('%a, %h', 'eligible')])
def test_only_a_literal_magnitude_with_runtime_sign_is_an_offset(copysign, status):
    body = f'%h = arith.constant 0.5 : f32\n%b = math.copysign {copysign} : f32\n%sum = arith.addf %a,%b : f32\n'
    result = decide(single(body + '%n = arith.fptosi %sum : f32 to i32\n%v = arith.sitofp %n : i32 to f32'))
    assert result['status'] == status, result


MAGIC = '%c = arith.constant 1536.0 : f32\n%s = arith.addf %a,%c : f32\n'
NARROW = '%n = arith.truncf %s : f32 to f16\n%w = arith.extf %n : f16 to f32\n'
MAGIC_REMOVALS = {
    'subf_same_literal': MAGIC + NARROW + '%v = arith.subf %w,%c : f32',
    'addf_negated_literal': MAGIC + NARROW + '%d = arith.constant -1536.0 : f32\n%v = arith.addf %w,%d : f32',
    'subf_closed_expression': MAGIC + NARROW + '%k1 = arith.constant 1024.0 : f32\n%k2 = arith.constant 512.0 : f32\n'
    '%k = arith.addf %k1,%k2 : f32\n%v = arith.subf %w,%k : f32',
    'addf_negf_first': '%c = arith.constant 1536.0 : f32\n%s = arith.addf %c,%a : f32\n' + NARROW
    + '%m = arith.negf %c : f32\n%v = arith.addf %m,%w : f32',
    'bf16_addf_negated_literal': '%c = arith.constant 12582912.0 : f32\n%s = arith.addf %a,%c : f32\n'
    '%n = arith.truncf %s : f32 to bf16\n%w = arith.extf %n : bf16 to f32\n'
    '%d = arith.constant -12582912.0 : f32\n%v = arith.addf %w,%d : f32',
    'f64_to_f16': '%c = arith.constant 1536.0 : f64\n%x64 = arith.extf %a : f32 to f64\n%s = arith.addf %x64,%c : f64\n'
    '%n = arith.truncf %s : f64 to f16\n%w = arith.extf %n : f16 to f32\n'
    '%d = arith.constant -1536.0 : f32\n%v = arith.addf %w,%d : f32',
}


@pytest.mark.parametrize('body', MAGIC_REMOVALS.values(), ids=MAGIC_REMOVALS.keys())
def test_magic_bias_removed_after_narrowing_in_any_spelling_is_excluded(body):
    result = decide(single(body))
    assert result['status'] == 'host', result
    assert any(e['reason'] == 'readout_offset_rounding' for e in result['evidence'])


def test_narrowing_that_is_stored_and_also_reused_is_not_a_storage_boundary():
    # The stored copy does not exempt the second, arithmetic use of the same narrowed value.
    m = parse_mlir_text('''module { func.func @f(%x: tensor<4xf32>) -> (tensor<4xf16>, tensor<4xf32>) {
      %e = tensor.empty() : tensor<4xf16>
      %f = tensor.empty() : tensor<4xf32>
      %r:2 = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
                             iterator_types=["parallel"]}
        ins(%x : tensor<4xf32>) outs(%e, %f : tensor<4xf16>, tensor<4xf32>) {
        ^bb0(%a:f32,%u0:f16,%u1:f32):
          ''' + MAGIC + NARROW + '''%d = arith.constant -1536.0 : f32
          %v = arith.addf %w,%d : f32
          linalg.yield %n, %v : f16, f32
        } -> (tensor<4xf16>, tensor<4xf32>)
      func.return %r#0, %r#1 : tensor<4xf16>, tensor<4xf32>
    }}''')
    assert decide(m)['status'] == 'host'


def test_stored_narrowing_of_an_affine_result_stays_eligible():
    m = single('%c = arith.constant 0.5 : f32\n%s = arith.addf %a,%c : f32\n'
               '%n = arith.truncf %s : f32 to f16\n%v = arith.extf %n : f16 to f32')
    assert decide(m)['status'] == 'eligible'
