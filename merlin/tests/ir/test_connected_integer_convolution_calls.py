"""Connected exact convolution calls retain separate live results and complete values."""

import hashlib

import numpy as np
import pytest
from test_integer_convolution_calls import _run, source
from test_integer_matmul_calls import EFFECTS, READOUT
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.integer_convolution_calls import (
    lower_integer_convolution_program,
    prepare_integer_convolution_calls,
)
from merlin.llvmlower.target_data_layout import _query
from merlin.llvmlower.toolchain import clang
from merlin.transforms.certificates import ir_text


def connected_source():
    stages = [parse_mlir_text(text).body.block.first_op for text in (
        source(n=1, ci=3, co=5),
        source(n=1, ci=5, co=7, kh=1, kw=1, sh=1, sw=1, dh=1, dw=1),
    )]
    inputs = [*stages[0].function_type.inputs, *tuple(stages[1].function_type.inputs)[1:]]
    block = Block(arg_types=inputs)
    function = func.FuncOp("forward", (inputs, stages[1].function_type.outputs), Region(block))
    mapper = dict(zip(stages[0].body.block.args, block.args[:3], strict=True))
    for operation in stages[0].body.block.ops:
        if type(operation) is not func.ReturnOp:
            block.add_op(operation.clone(value_mapper=mapper))
    first_result = mapper[stages[0].body.block.last_op.arguments[0]]
    mapper.update(zip(stages[1].body.block.args, (first_result, *block.args[3:]), strict=True))
    for operation in stages[1].body.block.ops:
        block.add_op(operation.clone(value_mapper=mapper))
    module = builtin.ModuleOp([function])
    module.verify()
    return ir_text(module)


@pytest.mark.parametrize("host_features", [(), ("inline_tensor_helpers_before_bufferization",), None])
def test_connected_calls_keep_live_intermediate_and_complete_modular_values(
    tmp_path, upstream_host_tools, host_features,
):
    text = connected_source()
    prepared = prepare_integer_convolution_calls(text, select=lambda _: True, borrowed_effects=EFFECTS,
                                                 output_readout=READOUT)
    assert len(prepared.calls) == 2
    assert prepared.calls[0].output_shape == prepared.calls[1].input_shape
    assert len({call.kernel_symbol for call in prepared.calls}) == 2

    def emit(call):
        b, h, w, ci = call.input_shape
        kh, kw, _, co = call.weight_shape
        _, oh, ow, _ = call.output_shape
        sh, sw = call.stride
        dh, dw = call.dilation
        path = tmp_path / f"{call.kernel_symbol}.c"
        path.write_text(f"""#include <stdint.h>
void {call.kernel_symbol}(void *aa,void *ww,void *bb,void *oo) {{
 const int8_t *a=aa,*weight=ww;const int32_t *bias=bb;int8_t *out=oo;
 for(int n=0;n<{b};n++)for(int y=0;y<{oh};y++)for(int x=0;x<{ow};x++)for(int f=0;f<{co};f++) {{
  uint32_t sum=(uint32_t)bias[f];
  for(int ky=0;ky<{kh};ky++)for(int kx=0;kx<{kw};kx++)for(int c=0;c<{ci};c++)
   sum+=(uint32_t)((int32_t)a[((n*{h}+y*{sh}+ky*{dh})*{w}+x*{sw}+kx*{dw})*{ci}+c]
      *(int32_t)weight[((ky*{kw}+kx)*{ci}+c)*{co}+f]);
  int32_t v=(int32_t)sum;out[((n*{oh}+y)*{ow}+x)*{co}+f]=v < -128 ? -128 : v > 127 ? 127 : v;
 }}
}}""")
        llvm = path.with_suffix(".ll")
        _run([clang(), "-O2", "-fPIC", "-S", "-emit-llvm", path, "-o", llvm])
        return llvm.read_text(), ((path, hashlib.sha256(path.read_bytes()).hexdigest()),)

    program = lower_integer_convolution_program(text, tmp_path / "composed", select=lambda _: True,
                                                emit_kernel=emit, borrowed_effects=EFFECTS,
                                                data_layout=_query(clang(), ("-O2", "-fPIC")), output_readout=READOUT,
                                                host_features=host_features)
    if host_features is None or host_features:
        assert "llvm.memcpy" not in program.host.ll_path.read_text()
    library = tmp_path / f"connected_convolution_{bool(host_features)}.so"
    _run([clang(), "-O2", "-fPIC", "-shared", program.host.ll_path, *program.kernels, *program.bridges, "-o", library])
    rng = np.random.default_rng(72)
    calls = prepared.calls
    image = rng.integers(-128, 128, calls[0].input_shape, dtype=np.int8)
    weights = [rng.integers(-128, 128, call.weight_shape, dtype=np.int8) for call in calls]
    extreme_bias = np.array([2147483647, -2147483648, 1, 0, -1], np.int32)
    biases = [np.resize(extreme_bias, call.weight_shape[3]) for call in calls]
    inputs = (image, weights[0], biases[0], weights[1], biases[1])
    originals = [value.copy() for value in inputs]
    expected = image
    for call, weight, bias in zip(calls, weights, biases, strict=True):
        _, oh, ow, co = call.output_shape
        kh, kw, _, _ = call.weight_shape
        sh, sw = call.stride
        dh, dw = call.dilation
        result = np.empty(call.output_shape, np.int64)
        for y in range(oh):
            for x in range(ow):
                patch = expected[:, y*sh:y*sh+dh*(kh-1)+1:dh, x*sw:x*sw+dw*(kw-1)+1:dw].astype(np.int64)
                result[:, y, x, :] = np.einsum("nyxc,yxcf->nf", patch, weight.astype(np.int64)) + bias
        expected = np.clip(result.astype(np.int32), -128, 127).astype(np.int8)
    output = np.full(calls[-1].output_shape, 73, np.int8)
    invoke = HostModel.load(str(library))
    for _ in range(2):
        invoke([(value.ctypes.data, value.shape) for value in (*inputs, output)])
        np.testing.assert_array_equal(output, expected)
        for value, original in zip(inputs, originals, strict=True):
            np.testing.assert_array_equal(value, original)
