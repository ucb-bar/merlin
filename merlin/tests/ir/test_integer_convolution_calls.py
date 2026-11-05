"""Direct convolution recovery checks actual maps, closed users and modular bias."""

import hashlib
import subprocess

import numpy as np
import pytest
from test_im2col_panel_pack import _module
from test_integer_matmul_calls import EFFECTS, READOUT

from merlin.common.cpu_pool import shared_compute_pool
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.integer_convolution_calls import (
    lower_integer_convolution_program,
    prepare_integer_convolution_calls,
)
from merlin.llvmlower.target_data_layout import _query
from merlin.llvmlower.toolchain import clang
from merlin.transforms.certificates import ir_text


def source(*, n=2, ci=3, co=5, kh=2, kw=3, oh=3, ow=5, sh=2, sw=1, dh=1, dw=2, narrow=True):
    h, w = (oh - 1) * sh + (kh - 1) * dh + 1, (ow - 1) * sw + (kw - 1) * dw + 1
    k, m, elements = ci * kh * kw, n * oh * ow, co * n * oh * ow
    image, nchw = f"tensor<{n}x{h}x{w}x{ci}xi8>", f"tensor<{n}x{ci}x{h}x{w}xi8>"
    weight, oihw = f"tensor<{kh}x{kw}x{ci}x{co}xi8>", f"tensor<{co}x{ci}x{kh}x{kw}xi8>"
    shape = f"{n}x{oh}x{ow}x{co}"
    output = f"tensor<{shape}x{'i8' if narrow else 'i32'}>"
    text = _module(c=ci, kh=kh, kw=kw, n=n, oh=oh, ow=ow, f=co, sh=sh, sw=sw, dh=dh, dw=dw)
    old_header = f"func.func @forward(%in: {nchw}, %w: tensor<{co}x{k}xi8>)\n      -> tensor<{co}x{m}xi32>"
    text = text.replace(old_header, f"func.func @forward(%image: {image}, %weight: {weight}, %bias:tensor<{co}xi32>)"
                        f" -> {output}")
    text = text.replace("    %c0 =", f"""
    %ie = tensor.empty():{nchw}
    %in = linalg.transpose ins(%image:{image}) outs(%ie:{nchw}) permutation = [0,3,1,2]
    %we = tensor.empty():{oihw}
    %wt = linalg.transpose ins(%weight:{weight}) outs(%we:{oihw}) permutation = [3,2,0,1]
    %wf = tensor.collapse_shape %wt [[0,1,2,3]]:{oihw} into tensor<{co*k}xi8>
    %w = tensor.expand_shape %wf [[0,1]] output_shape [{co},{k}]
      :tensor<{co*k}xi8> into tensor<{co}x{k}xi8>
    %c0 =""")
    cnchw, cbchw = f"tensor<{co}x{n}x{oh}x{ow}xi32>", f"tensor<{n}x{co}x{oh}x{ow}xi32>"
    nhwc = f"tensor<{shape}xi32>"
    suffix = f"""
    %rf = tensor.collapse_shape %mm [[0,1]]:tensor<{co}x{m}xi32> into tensor<{elements}xi32>
    %rx = tensor.expand_shape %rf [[0,1,2,3]] output_shape [{co},{n},{oh},{ow}]
      :tensor<{elements}xi32> into {cnchw}
    %re = tensor.empty():{cbchw}
    %rt = linalg.transpose ins(%rx:{cnchw}) outs(%re:{cbchw}) permutation = [1,0,2,3]
    %be = tensor.empty():{cbchw}
    %biased = linalg.generic {{indexing_maps=[affine_map<(a,b,c,d)->(a,b,c,d)>,
      affine_map<(a,b,c,d)->(b)>,affine_map<(a,b,c,d)->(a,b,c,d)>],
      iterator_types=["parallel","parallel","parallel","parallel"]}}
      ins(%rt,%bias:{cbchw},tensor<{co}xi32>) outs(%be:{cbchw}) {{
    ^bb0(%x:i32,%b:i32,%unused:i32):
      %plus = arith.addi %x,%b:i32
      linalg.yield %plus:i32
    }} -> {cbchw}
    %oe = tensor.empty():{nhwc}
    %ot = linalg.transpose ins(%biased:{cbchw}) outs(%oe:{nhwc}) permutation = [0,2,3,1]
"""
    if narrow:
        suffix += f"""
    %ne = tensor.empty():{output}
    %out = linalg.generic {{indexing_maps=[affine_map<(a,b,c,d)->(a,b,c,d)>,
      affine_map<(a,b,c,d)->(a,b,c,d)>],iterator_types=["parallel","parallel","parallel","parallel"]}}
      ins(%ot:{nhwc}) outs(%ne:{output}) {{
    ^bb0(%v:i32,%unused:i8):
      %lo = arith.constant -128:i32
      %hi = arith.constant 127:i32
      %low = arith.maxsi %v,%lo:i32
      %high = arith.minsi %low,%hi:i32
      %saturated = arith.trunci %high:i32 to i8
      linalg.yield %saturated:i8
    }} -> {output}
    return %out:{output}
"""
    else:
        suffix += f"    return %ot:{output}\n"
    return text.replace(f"    func.return %mm : tensor<{co}x{m}xi32>", suffix)


def prepare(text, **kwargs):
    return prepare_integer_convolution_calls(text, select=lambda c: True, borrowed_effects=EFFECTS,
                                             output_readout=READOUT, **kwargs)


@pytest.mark.parametrize("shape", [dict(), dict(n=1, ci=1, co=1, kh=1, kw=1), dict(ci=17, co=19, ow=7)])
@pytest.mark.parametrize("narrow", [True, False])
def test_semantic_geometry_bias_and_output_policy(shape, narrow):
    text = source(**shape, narrow=narrow)
    prepared = prepare(text)
    assert len(prepared.calls) == 1
    call = prepared.calls[0]
    assert call.output_dtype == ("i8" if narrow else "i32")
    assert call.input_shape[-1] == shape.get("ci", 3)
    assert call.weight_shape[-1] == shape.get("co", 5)
    assert call.stride == (2, 1) and call.dilation == (1, 2)
    assert len(prepared.readout_proofs) == int(narrow)
    assert all(proof.verified for proof in prepared.readout_proofs)
    parse_mlir_text(prepared.source).verify()


@pytest.mark.parametrize("mutation", ["unsigned", "nonzero", "bias_sub", "bias_axis", "unknown_product",
                                      "unknown_gather", "unknown_gather_yield", "extra_raw_observer", "effect"])
def test_changed_arithmetic_maps_effects_and_observers_retain_original_source(mutation):
    from xdsl.dialects import builtin, func
    from xdsl.rewriter import InsertPoint, Rewriter

    text = source()
    if mutation == "unsigned":
        text = text.replace("arith.extsi", "arith.extui")
    elif mutation == "nonzero":
        text = text.replace("constant 0 : i32", "constant 1 : i32")
    elif mutation == "bias_sub":
        text = text.replace("arith.addi %x,%b", "arith.subi %x,%b")
    elif mutation == "bias_axis":
        # Equal axis extents cannot establish a broadcast coordinate identity.
        text = source(n=5).replace("affine_map<(a,b,c,d)->(b)>", "affine_map<(a,b,c,d)->(a)>")
    else:
        module = parse_mlir_text(text)
        products = [o for o in module.walk() if o.name == "linalg.generic"]
        dot = next(o for o in products if len(o.inputs) == 2 and len(o.body.block.args) == 3
                   and any(inner.name == "arith.muli" for inner in o.body.block.ops))
        if mutation == "unknown_product":
            dot.attributes["unknown"] = builtin.UnitAttr()
        elif mutation in ("unknown_gather", "unknown_gather_yield"):
            gather = products[0]
            target = gather if mutation == "unknown_gather" else gather.body.block.last_op
            target.attributes["unknown"] = builtin.UnitAttr()
        elif mutation == "extra_raw_observer":
            function = next(o for o in module.body.block.ops if isinstance(o, func.FuncOp))
            function.properties["function_type"] = builtin.FunctionType.from_lists(
                list(function.function_type.inputs), [*function.function_type.outputs, dot.results[0].type])
            function.body.block.last_op.operands = (*function.body.block.last_op.operands, dot.results[0])
        else:
            from xdsl.ir import Region

            declaration = func.FuncOp("opaque", ([], []), Region(), visibility="private")
            module.body.block.add_op(declaration)
            Rewriter.insert_op(func.CallOp("opaque", [], []), InsertPoint.after(dot))
        text = ir_text(module)
    prepared = prepare(text)
    assert not prepared.calls and prepared.source == text


def test_non_boolean_selection_is_rejected():
    with pytest.raises(ValueError, match="Boolean"):
        prepare_integer_convolution_calls(source(), select=lambda _: 1, borrowed_effects=EFFECTS)


def _run(argv):
    with shared_compute_pool(timeout_s=90):
        return subprocess.run(list(map(str, argv)), check=True, capture_output=True, timeout=90)


@pytest.mark.parametrize("narrow", [False, True])
@pytest.mark.parametrize("changes", [dict(ci=3, co=5), dict(ci=17, co=19, kh=1, kw=1, n=1)])
def test_complete_native_modular_bias_tails_and_input_preservation(tmp_path, upstream_host_tools, changes, narrow):
    text = source(**changes, narrow=narrow)
    call = prepare(text).calls[0]
    b, h, w, ci = call.input_shape
    kh, kw, _, co = call.weight_shape
    _, oh, ow, _ = call.output_shape
    sh, sw = call.stride
    dh, dw = call.dilation
    stored = "signed_sum < -128 ? -128 : signed_sum > 127 ? 127 : signed_sum" if narrow else "signed_sum"
    c = tmp_path / "neutral.c"
    c.write_text(f"""#include <stdint.h>
void {call.kernel_symbol}(void *aa,void *ww,void *bb,void *oo) {{
 const int8_t *a=aa,*weight=ww;const int32_t *bias=bb;
 {'int8_t' if narrow else 'int32_t'} *out=oo;
 for(int n=0;n<{b};n++)for(int y=0;y<{oh};y++)for(int x=0;x<{ow};x++)for(int f=0;f<{co};f++) {{
   uint32_t sum=(uint32_t)bias[f];
   for(int ky=0;ky<{kh};ky++)for(int kx=0;kx<{kw};kx++)for(int c=0;c<{ci};c++)
     sum+=(uint32_t)((int32_t)a[((n*{h}+y*{sh}+ky*{dh})*{w}+x*{sw}+kx*{dw})*{ci}+c]
                    *(int32_t)weight[((ky*{kw}+kx)*{ci}+c)*{co}+f]);
   int32_t signed_sum=(int32_t)sum;
   out[((n*{oh}+y)*{ow}+x)*{co}+f]={stored};
 }}
}}""")
    ll = tmp_path / "neutral.ll"
    _run([clang(), "-O2", "-fPIC", "-S", "-emit-llvm", c, "-o", ll])
    pins = ((c, hashlib.sha256(c.read_bytes()).hexdigest()),)
    program = lower_integer_convolution_program(
        text, tmp_path / "program", select=lambda _: True, borrowed_effects=EFFECTS,
        emit_kernel=lambda _: (ll.read_text(), pins), output_readout=READOUT,
        data_layout=_query(clang(), ("-O2", "-fPIC")),
    )
    # A passed pytest temporary directory may be reused while dlopen still
    # retains that pathname. Give different monomorphic policies distinct names.
    lib = tmp_path / f"conv_{b}_{h}_{w}_{ci}_{co}_{kh}_{kw}_{narrow}.so"
    _run([clang(), "-O2", "-fPIC", "-shared", program.host.ll_path,
          *program.kernels, *program.bridges, "-o", lib])
    rng = np.random.default_rng(31)
    image = rng.integers(-128, 128, call.input_shape, dtype=np.int8)
    weight = rng.integers(-128, 128, call.weight_shape, dtype=np.int8)
    bias = np.resize(np.array([2147483647, -2147483648, 0, 129, -129], np.int32), co)
    originals = [value.copy() for value in (image, weight, bias)]
    expected = np.empty(call.output_shape, np.int64)
    for y in range(oh):
        for x in range(ow):
            patch = image[:, y*sh:y*sh+dh*(kh-1)+1:dh, x*sw:x*sw+dw*(kw-1)+1:dw].astype(np.int64)
            expected[:, y, x, :] = np.einsum("nyxc,yxcf->nf", patch, weight.astype(np.int64)) + bias
    expected = expected.astype(np.int32)
    if narrow:
        expected = np.clip(expected, -128, 127).astype(np.int8)
    output = np.full(call.output_shape, 73, expected.dtype)
    invoke = HostModel.load(str(lib))
    for _ in range(2):
        invoke([(value.ctypes.data, value.shape) for value in (image, weight, bias, output)])
        np.testing.assert_array_equal(output, expected)
        for value, original in zip((image, weight, bias), originals, strict=True):
            np.testing.assert_array_equal(value, original)
