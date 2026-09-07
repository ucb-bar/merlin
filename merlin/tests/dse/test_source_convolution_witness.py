from __future__ import annotations

import numpy as np
import pytest

from merlin.perf.source_convolution_witness import extract_source_convolution, evaluate_source_convolution


def source(*, stride=1, dilation=1, padding=1, channels=3, output=5):
    kernel = 3
    spatial = (output-1)*stride + (kernel-1)*dilation+1-2*padding
    padded = spatial+2*padding
    return f'''module {{
      func.func @compute(%x: tensor<1x{channels}x{spatial}x{spatial}xi8>,
                         %w: tensor<4x{channels}x3x3xi8>) -> tensor<1x4x{output}x{output}xi32> {{
        %z = arith.constant 0 : i8
        %p = tensor.splat %z : tensor<1x{channels}x{padded}x{padded}xi8>
        %a = "tensor.insert_slice"(%x,%p) <{{static_offsets=array<i64:0,0,{padding},{padding}>,
          static_sizes=array<i64:1,{channels},{spatial},{spatial}>, static_strides=array<i64:1,1,1,1>,
          operandSegmentSizes=array<i32:1,1,0,0,0>}}> :
          (tensor<1x{channels}x{spatial}x{spatial}xi8>, tensor<1x{channels}x{padded}x{padded}xi8>)
          -> tensor<1x{channels}x{padded}x{padded}xi8>
        %zero = arith.constant 0 : i32
        %empty = tensor.empty() : tensor<1x4x{output}x{output}xi32>
        %initial = linalg.fill ins(%zero:i32) outs(%empty:tensor<1x4x{output}x{output}xi32>)
          -> tensor<1x4x{output}x{output}xi32>
        %result = linalg.generic {{indexing_maps=[
          affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d0,d4,d2*{stride}+d5*{dilation},d3*{stride}+d6*{dilation})>,
          affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d1,d4,d5,d6)>,
          affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d0,d1,d2,d3)>],
          iterator_types=["parallel","parallel","parallel","parallel","reduction","reduction","reduction"]}}
          ins(%a,%w:tensor<1x{channels}x{padded}x{padded}xi8>,tensor<4x{channels}x3x3xi8>)
          outs(%initial:tensor<1x4x{output}x{output}xi32>) attrs = {{prov.region_id="selected"}} {{
          ^bb0(%aa:i8,%bb:i8,%acc:i32):
            %ae = arith.extsi %aa : i8 to i32
            %be = arith.extsi %bb : i8 to i32
            %mul = arith.muli %ae,%be : i32
            %sum = arith.addi %mul,%acc : i32
            linalg.yield %sum : i32
          }} -> tensor<1x4x{output}x{output}xi32>
        func.return %result : tensor<1x4x{output}x{output}xi32>
      }}
    }}'''


@pytest.mark.parametrize("stride,dilation,padding", [(1,1,1),(2,1,1),(1,2,2),(2,2,1),(1,1,0)])
def test_source_geometry_and_signed_integer_reference(stride, dilation, padding):
    original = source(stride=stride, dilation=dilation, padding=padding)
    text, record = extract_source_convolution(original, 6, entry="compute")
    assert record["source_indices"] == list(range(7))
    assert record["weight_shape"][2:] == [3,3]
    assert record["stride"] == [stride,stride]
    assert record["dilation"] == [dilation,dilation]
    assert record["padding"] == [padding]*4
    assert record["output_shape"] == [1,2,2,2]
    assert "arith.extsi" in text and "tensor.insert_slice" in text and "prov.region_id" in text
    x = ((np.arange(np.prod(record["input_shape"])).reshape(record["input_shape"])*37) % 256 - 128).astype(np.int8)
    w = ((np.arange(np.prod(record["weight_shape"])).reshape(record["weight_shape"])*19) % 256 - 128).astype(np.int8)
    got = evaluate_source_convolution(record,x,w)
    expected = np.zeros(record["output_shape"], dtype=np.int32)
    for n,c,y,z in np.ndindex(*expected.shape):
        for ci,ky,kx in np.ndindex(*w.shape[1:]):
            iy,ix = y*stride+ky*dilation-padding,z*stride+kx*dilation-padding
            if 0 <= iy < x.shape[2] and 0 <= ix < x.shape[3]:
                expected[n,c,y,z] += int(x[n,ci,iy,ix])*int(w[c,ci,ky,kx])
    np.testing.assert_array_equal(got,expected)
    assert np.any(np.abs(got.astype(np.int64)) > 127)


@pytest.mark.parametrize("mutation", [
    lambda text: text.replace("arith.constant 0 : i8", "arith.constant 1 : i8"),
    lambda text: text.replace("arith.constant 0 : i32", "arith.constant 1 : i32"),
    lambda text: text.replace("arith.extsi", "arith.extui"),
    lambda text: text.replace("d0,d4,d2*1", "d4,d0,d2*1"),
])
def test_unsupported_semantics_do_not_become_a_different_probe(mutation):
    with pytest.raises(ValueError):
        extract_source_convolution(mutation(source()),6,entry="compute")


def test_budget_and_explicit_source_selection_fail_closed():
    with pytest.raises(ValueError,match="work bound"):
        extract_source_convolution(source(),6,entry="compute",max_macs=1)
    with pytest.raises(ValueError):
        extract_source_convolution(source(),5,entry="compute")
    with pytest.raises(ValueError):
        extract_source_convolution(source(),6,entry="wrong")


def test_reference_refuses_wrong_operand_width():
    _, record = extract_source_convolution(source(),6,entry="compute")
    with pytest.raises(ValueError,match="signed i8"):
        evaluate_source_convolution(record,np.zeros(record["input_shape"],dtype=np.uint8),
                                    np.zeros(record["weight_shape"],dtype=np.int8))
