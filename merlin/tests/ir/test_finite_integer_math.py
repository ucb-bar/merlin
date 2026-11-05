"""Execute exact runtime preparation over all INT8 input pairs and a tail."""

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.automatic_host_features import choices
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.optional_passes import Selection, applied


def source(rows, columns, *, varying=False, strict=False, quantized=False):
    shape = f"{rows}x{columns}"
    argument = "%v" if varying else "%x"
    dtype = "i8" if quantized else "f32"
    quantization = """
            %lo=arith.constant -128.0:f32
            %hi=arith.constant 127.0:f32
            %qscale=arith.constant 64.0:f32
            %scaled=arith.mulf %out,%qscale:f32
            %low=arith.maximumf %scaled,%lo:f32
            %clamped=arith.minimumf %low,%hi:f32
            %rounded=math.roundeven %clamped:f32
            %quantized=arith.fptosi %rounded:f32 to i8
    """ if quantized else ""
    return f"""module {{
      func.func @forward(%a:tensor<{shape}xi8>,%b:tensor<{shape}xi8>)
          ->tensor<{shape}x{dtype}> attributes {{llvm.emit_c_interface{', llvm.strictfp' if strict else ''}}} {{
        %scale=arith.constant 0.03125:f32
        %upscale=arith.constant 0.015625:f32
        %one=arith.constant 1.0:f32
        %empty=tensor.empty():tensor<{shape}x{dtype}>
        %result=linalg.generic {{
          indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0,d1)>],
          iterator_types=["parallel","parallel"]}}
          ins(%a,%b:tensor<{shape}xi8>,tensor<{shape}xi8>) outs(%empty:tensor<{shape}x{dtype}>) {{
          ^bb0(%g:i8,%u:i8,%old:{dtype}):
            %gf=arith.sitofp %g:i8 to f32
            %uf=arith.sitofp %u:i8 to f32
            %x=arith.mulf %gf,%scale:f32
            %up=arith.mulf %uf,%upscale:f32
            {'%v=arith.addf %x,%up:f32' if varying else ''}
            %neg=arith.negf {argument}:f32
            %e=math.exp %neg:f32
            %den=arith.addf %e,%one:f32
            %recip=arith.divf %one,%den:f32
            %gate=arith.mulf {argument},%recip:f32
            %out=arith.mulf %gate,%up:f32
            {quantization}
            linalg.yield {'%quantized' if quantized else '%out'}:{dtype}
        }} ->tensor<{shape}x{dtype}>
        return %result:tensor<{shape}x{dtype}>
      }}
    }}"""


def invoke(result, a, b, *, quantized=False):
    output = np.empty(a.shape, dtype=np.int8 if quantized else np.float32)
    HostModel.load(str(result.host_so))([(value.ctypes.data, value.shape) for value in (a, b, output)])
    return output


@pytest.mark.parametrize("rows,columns,varying,quantized", [
    (256, 256, False, False), (3, 1027, False, False), (3, 1027, True, False),
    (256, 256, False, True),
])
def test_complete_runtime_domain_and_tail(tmp_path, rows, columns, varying, quantized):
    text = source(rows, columns, varying=varying, quantized=quantized)
    work = tmp_path / f"rows{rows}-columns{columns}-varying{int(varying)}-int8{int(quantized)}"
    if rows == columns == 256:
        a = np.broadcast_to(np.arange(256, dtype=np.uint8)[:, None], (256, 256)).copy().view(np.int8)
        b = np.broadcast_to(np.arange(256, dtype=np.uint8)[None, :], (256, 256)).copy().view(np.int8)
    else:
        rng = np.random.default_rng(134)
        a = rng.integers(-128, 128, (rows, columns), dtype=np.int8)
        b = rng.integers(-128, 128, (rows, columns), dtype=np.int8)
    with applied(Selection.parse("-finite-integer-math")):
        baseline = lower_model(text, work/'control')
    with applied(Selection.parse("finite-integer-math")):
        candidate = lower_model(text, work/'candidate')
    np.testing.assert_array_equal(invoke(candidate, a, b, quantized=quantized).view(np.uint8),
                                  invoke(baseline, a, b, quantized=quantized).view(np.uint8))
    report = candidate.stats['automatic_host_optimizations']
    assert report['rewrites']['finite_integer_math'] == (0 if varying else 1)
    assert report['hardware_cycles'] == 'not_measured'
    if not varying:
        llvm = candidate.ll_path.read_text()
        assert '@expf' in llvm
        assert '256' in llvm


def test_table_selection_is_explicit_and_can_be_disabled(monkeypatch):
    assert 'finite_integer_math' not in choices(set())
    monkeypatch.setenv('MERLIN_FINITE_INTEGER_MATH', '1')
    assert choices(set())['finite_integer_math']
    with applied(Selection.parse('-finite-integer-math')):
        assert 'finite_integer_math' not in choices(set())


def test_explicit_floating_environment_retains_original_source(tmp_path):
    with applied(Selection.parse('finite-integer-math')):
        result = lower_model(source(3, 1027, strict=True), tmp_path/'strict')
    assert result.stats['automatic_host_optimizations']['rewrites']['finite_integer_math'] == 0
