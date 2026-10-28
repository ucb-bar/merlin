"""Live floating residuals and permuted quantizers share one exact scalar loop."""

import numpy as np
import pytest
from xdsl.dialects import builtin

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.host_pointwise_fusion import fuse_live_pointwise, prepare_host_pointwise_fusion


def source():
    # The retained full-model residual has this fanout: the unquantized f32
    # value survives for a later skip, while the next device consumes NHWC i8.
    return '''module {
      func.func @forward(%x:tensor<2x3x5xf32>,%state:tensor<2x3x5xf32>)
          -> (tensor<2x3x5xf32>,tensor<2x5x3xi8>) {
        %e=tensor.empty():tensor<2x3x5xf32>
        %p=linalg.generic {indexing_maps=[affine_map<(i,j,k)->(i,j,k)>,
            affine_map<(i,j,k)->(i,j,k)>,affine_map<(i,j,k)->(i,j,k)>],
            iterator_types=["parallel","parallel","parallel"]}
            ins(%x,%state:tensor<2x3x5xf32>,tensor<2x3x5xf32>) outs(%e:tensor<2x3x5xf32>) {
          ^bb0(%a:f32,%s:f32,%unused:f32):
            %v=arith.addf %a,%s:f32
            %zero=arith.constant 0.0:f32
            %negative=arith.cmpf olt,%v,%zero:f32
            %r=arith.select %negative,%zero,%v:f32
            linalg.yield %r:f32
        } -> tensor<2x3x5xf32>
        %qe=tensor.empty():tensor<2x5x3xi8>
        %q=linalg.generic {indexing_maps=[affine_map<(i,j,k)->(i,j,k)>,
            affine_map<(i,j,k)->(i,k,j)>],iterator_types=["parallel","parallel","parallel"]}
            ins(%p:tensor<2x3x5xf32>) outs(%qe:tensor<2x5x3xi8>) {
          ^bb0(%a:f32,%unused:i8):
            %scale=arith.constant 0.5:f32
            %v=arith.mulf %a,%scale:f32
            %lo=arith.constant -128.0:f32
            %hi=arith.constant 127.0:f32
            %lower=arith.maximumf %v,%lo:f32
            %upper=arith.minimumf %lower,%hi:f32
            %r=math.roundeven %upper:f32
            %q=arith.fptosi %r:f32 to i8
            linalg.yield %q:i8
        } -> tensor<2x5x3xi8>
        return %p,%q:tensor<2x3x5xf32>,tensor<2x5x3xi8>
      }
    }'''


def test_live_output_and_distinct_affine_maps_share_exact_bodies():
    original = parse_mlir_text(source())
    module = original.clone()
    before = [op for op in original.walk() if op.name == "linalg.generic"]
    assert fuse_live_pointwise(module) == [{"shape": [2, 3, 5], "shared_elements": 30,
                                           "eliminated_read_bytes": 120}]
    fused, = [op for op in module.walk() if op.name == "linalg.generic"]
    assert len(fused.results) == 2
    assert fused.indexing_maps.data[-2:] == (
        before[0].indexing_maps.data[-1], before[1].indexing_maps.data[-1])
    assert [op.name for op in fused.body.block.ops] == [
        *(op.name for op in tuple(before[0].body.block.ops)[:-1]),
        *(op.name for op in tuple(before[1].body.block.ops)[:-1]), "linalg.yield"]
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert tuple(returned.operands) == tuple(fused.results)
    assert fuse_live_pointwise(module) == []


@pytest.mark.parametrize("fault", ["effect", "early_observer", "destination_read", "reduction", "budget"])
def test_unknown_effects_earlier_observers_recurrence_and_budget_retain_boundaries(fault):
    module = parse_mlir_text(source())
    producer, consumer = [op for op in module.walk() if op.name == "linalg.generic"]
    if fault == "effect":
        producer.parent.insert_op_before(builtin.UnregisteredOp.with_name("opaque.touch").create(), consumer)
    elif fault == "early_observer":
        producer.parent.insert_op_before(
            builtin.UnregisteredOp.with_name("opaque.observe").create(operands=[producer.results[0]]), consumer)
    elif fault == "destination_read":
        producer.body.block.last_op.operands = [producer.body.block.args[-1]]
    elif fault == "reduction":
        from xdsl.dialects.linalg.attrs import IteratorTypeAttr
        producer.properties["iterator_types"] = builtin.ArrayAttr([IteratorTypeAttr.reduction()] * 3)
    assert fuse_live_pointwise(module, max_scalar_operations=0 if fault == "budget" else 512) == []


def test_default_source_is_unchanged():
    assert prepare_host_pointwise_fusion(source())[0] == source()
    assert prepare_host_pointwise_fusion(source(), enabled=True, serial=False)[0] == source()


def test_ordinary_native_fusion_preserves_float_bits_and_complete_integer_output(tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.impr_features import FUSE_AFTER_GENERALIZE_NAME
    from merlin.llvmlower.lower import lower_model

    if not toolchain.available():
        pytest.skip("selected native lowering toolchain unavailable")
    # Ties, saturation, signed zero, cancellation and nonexact float addition.
    x = np.resize(np.array([-1000, -0., 0., 1., 3., 5., 253., 255., 1000., 2**24],
                           dtype=np.float32), 30).reshape(2, 3, 5)
    state = np.resize(np.array([0., -0., .25, -.25, 1.], dtype=np.float32), 30).reshape(2, 3, 5)
    summed = np.add(x, state, dtype=np.float32)
    expected_f = np.where(summed < 0, np.float32(0), summed)
    expected_q = np.rint(np.clip(expected_f * np.float32(.5), -128, 127)).astype(np.int8).transpose(0, 2, 1)
    loads = {}
    for arm, features in (("baseline", ()), ("fused", (FUSE_AFTER_GENERALIZE_NAME,))):
        result = lower_model(source(), tmp_path / arm, targets=("host",), features=features)
        assert result.stats["host_pointwise_fusion"]["rewrites"] == (arm == "fused")
        loads[arm] = result.ll_path.read_text().count(" = load float")
        invoke = HostModel.load(str(result.host_so),
                               image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve()))
        floating = np.empty_like(x)
        quantized = np.empty((2, 5, 3), dtype=np.int8)
        invoke([(a.ctypes.data, a.shape) for a in (x, state, floating, quantized)])
        np.testing.assert_array_equal(floating.view(np.uint32), expected_f.view(np.uint32))
        np.testing.assert_array_equal(quantized, expected_q)
    assert loads["fused"] < loads["baseline"]
