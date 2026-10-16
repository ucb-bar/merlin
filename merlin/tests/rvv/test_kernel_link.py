"""Linking an external kernel body into a model before lowering (``llvmlower.kernel_link``).

The input is a real model2MLIR capture, ``data/mlp_bundles/mlp_basic_triton`` (``gen_mlp_kernel_fixtures.py``),
whose fc1+relu+fc2+relu is one ``prov.kernel = "triton_exp::fused_mlp2"`` call to a private declaration.
The link replaces that declaration with the MLIR twin of the ``@triton_op`` wrapper and merges the
kernel's ``llvm.func``; the facts it relies on are measured here on the lowered module, through both
preprocess paths (the xDSL one prints the declaration generic, the textual one pretty):

* no weight is copied before the call (the wrapper's operands are read-only buffers);
* the kernel's result buffer is allocated by ``@forward``, not by the wrapper (hoisted out-param);
* nothing is left undefined: the kernel is a definition inside ``model.ll``.
"""

from __future__ import annotations

import dataclasses

import pytest
from mlp_bundles import forward_args

from merlin.common.paths import merlin_dir
from merlin.llvmlower import kernel_link as KL
from merlin.llvmlower import kernel_table as KT
from merlin.llvmlower import toolchain

FIXTURE = merlin_dir() / "tests" / "data" / "mlp_bundles" / "mlp_basic_triton"
KERNEL_FREE = merlin_dir() / "tests" / "data" / "mlp_bundles" / "mlp_basic"
SYMBOL = "triton_exp__fused_mlp2"
ENTRY = KT.lookup("triton_exp::fused_mlp2").body.entry

needs_toolchain = pytest.mark.skipif(not toolchain.available(), reason="clang-23 / model2MLIR venv not available")


def _text(path=FIXTURE) -> str:
    return (path / "model.mlir").read_text(encoding="utf-8")


def _use(**over):
    (use,) = KL.find_kernel_uses(_text())
    return dataclasses.replace(use, **over)


def test_find_kernel_uses_mlp2():
    (use,) = KL.find_kernel_uses(_text())
    assert use.symbol == SYMBOL
    assert use.qualname == "triton_exp::fused_mlp2"
    assert use.operand_names == ("x", "w1", "b1", "w2", "b2")
    assert use.arg_types == (((4, 16), "f32"), ((32, 16), "f32"), ((32,), "f32"), ((32, 32), "f32"), ((32,), "f32"))
    assert use.result_types == (((4, 32), "f32"),)
    assert use.kernel_args == {}


def test_kernel_free_module_has_no_uses():
    assert KL.find_kernel_uses(_text(KERNEL_FREE)) == ()


def test_unknown_kernel_named_in_error():
    text = _text().replace('"triton_exp::fused_mlp2"', '"acme::nope"')
    with pytest.raises(KL.KernelLinkError) as err:
        KL.find_kernel_uses(text)
    assert "acme::nope" in str(err.value) and "triton_exp::fused_mlp2" in str(err.value)


# --------------------------------------------------------------------------- the wrapper


def test_wrapper_mlp2_text():
    src = KL.emit_wrappers([_use()])
    assert f"func.func private @{SYMBOL}(" in src
    assert src.count(" read_only : ") == 5  # every operand is promised read-only
    assert src.count("memref.alloc()") == 1  # the one result
    (call,) = [line for line in src.splitlines() if f"llvm.call @{ENTRY}(" in line]
    assert call.count("!llvm.ptr") == 6 and call.count("i32") == 7  # 6 ptr + M + 6 launch ints
    assert "%p6 = arith.constant 4 : i32" in src  # M is x's row count
    assert "%g0 = arith.constant 1 : i32" in src  # cdiv(4, 16) programs on axis 0
    assert f"llvm.func @{ENTRY}(" in src  # declared, so the module verifies alone


def test_two_shapes_two_wrappers_one_entry():
    a = _use()
    b = _use(symbol=SYMBOL + "_1", arg_types=(((20, 16), "f32"),) + a.arg_types[1:], result_types=(((20, 32), "f32"),))
    src = KL.emit_wrappers([a, b])
    assert src.count(f"func.func private @{SYMBOL}") == 2
    assert src.count(f"llvm.func @{ENTRY}(") == 1
    assert "%g0 = arith.constant 2 : i32" in src  # cdiv(20, 16)


def test_specialization_mismatch_rejected():
    bad = _use(arg_types=(((4, 24), "f32"),) + _use().arg_types[1:])
    with pytest.raises(KL.KernelLinkError) as err:
        KL.emit_wrappers([bad])
    assert "x" in str(err.value) and "16" in str(err.value) and "24" in str(err.value)


def test_operand_dtype_mismatch_rejected():
    bad = _use(arg_types=(((4, 16), "bf16"),) + _use().arg_types[1:])
    with pytest.raises(KL.KernelLinkError) as err:
        KL.emit_wrappers([bad])
    assert "bf16" in str(err.value) and "f32" in str(err.value)


def test_dynamic_shape_rejected():
    bad = _use(arg_types=(((-1, 16), "f32"),) + _use().arg_types[1:])
    with pytest.raises(KL.KernelLinkError, match="dynamic"):
        KL.emit_wrappers([bad])


def test_result_the_body_does_not_write_rejected():
    with pytest.raises(KL.KernelLinkError, match="result"):
        KL.emit_wrappers([_use(result_types=_use().result_types * 2)])


def test_kernel_args_rejected():
    with pytest.raises(KL.KernelLinkError, match="kernel_args"):
        KL.emit_wrappers([_use(kernel_args={"alpha": "5.0e-01 : f64"})])


# --------------------------------------------------------------------------- the merge


@needs_toolchain
@pytest.mark.parametrize("textual", [False, True], ids=["xdsl", "textual"])
def test_link_replaces_declaration(textual, tmp_path):
    from merlin.llvmlower.passes_xdsl import preprocess_text, preprocess_text_textual

    upstream, _ = (preprocess_text_textual if textual else preprocess_text)(_text())
    linked = KL.link_kernels(upstream, KL.find_kernel_uses(_text()), tmp_path)
    (head,) = [line for line in linked.splitlines() if f"func.func private @{SYMBOL}(" in line]
    assert head.rstrip().endswith("{")  # a definition now, not a declaration
    assert f"llvm.func @{ENTRY}(" in linked
    assert f"llvm.call @{ENTRY}(" in linked
    # The body's locations and debug info (the exporting checkout's source paths) are not carried.
    assert 'loc("' not in linked and "#loc" not in linked and "#di_" not in linked
    assert (tmp_path / "kernel_wrappers.mlir").is_file()


@needs_toolchain
def test_wrapper_that_does_not_match_the_body_refused(tmp_path, monkeypatch):
    """The kernel's own signature is the check on the table: drop the M argument and the linked
    module fails MLIR verification, by name, before the pipeline runs."""
    entry = KT.KERNEL_TABLE["triton_exp::fused_mlp2"]
    body = dataclasses.replace(entry.body, args=entry.body.args[:-1])
    monkeypatch.setitem(KT.KERNEL_TABLE, entry.qualname, dataclasses.replace(entry, body=body))
    with pytest.raises(KL.KernelLinkError) as err:
        KL.link_kernels(_text(), KL.find_kernel_uses(_text()), tmp_path)
    assert "triton_exp::fused_mlp2" in str(err.value) and "does not verify" in str(err.value)


@needs_toolchain
def test_body_symbol_already_in_module_refused(tmp_path):
    text = _text().replace(
        f"  func.func private @{SYMBOL}(", f"  func.func private @{ENTRY}()\n  func.func private @{SYMBOL}("
    )
    with pytest.raises(KL.KernelLinkError) as err:
        KL.link_kernels(text, KL.find_kernel_uses(_text()), tmp_path)
    assert ENTRY in str(err.value)


def _function(ll: str, symbol: str) -> str:
    """The body of the function ``symbol`` defines (not a call of it from another function)."""
    at = 0
    while True:
        at = ll.index("\ndefine", at) + 1
        if f" @{symbol}(" in ll[at : ll.index("\n", at)]:
            return ll[at : ll.index("\n}\n", at)]


@needs_toolchain
@pytest.mark.parametrize("textual", [False, True], ids=["xdsl", "textual"])
def test_lowered_model_is_closed(textual, tmp_path):
    from merlin.llvmlower.lower import lower_model

    res = lower_model(_text(), tmp_path, targets=(), textual=textual)
    assert res.ll_path.name == "model.ll"
    assert res.stats["kernel_uses"] == 1 and res.stats["kernels_linked"] == ["triton_exp::fused_mlp2"]
    ll = res.ll_path.read_text(encoding="utf-8")
    assert f"define void @{ENTRY}(" in ll
    assert f"define internal void @{SYMBOL}(" in ll
    assert "_mlir_ciface_triton" not in ll  # no descriptor ABI at the kernel boundary


@needs_toolchain
@pytest.mark.parametrize("textual", [False, True], ids=["xdsl", "textual"])
def test_weights_not_copied_and_result_hoisted(textual, tmp_path):
    from merlin.llvmlower.lower import lower_model

    lower_model(_text(), tmp_path, targets=(), textual=textual)
    ll = (tmp_path / "model.ll").read_text(encoding="utf-8")
    body = _function(ll, "forward")
    before_call = body[: body.index(f"@{SYMBOL}(")]
    assert "@llvm.memcpy" not in before_call
    assert "@malloc(" not in _function(ll, SYMBOL)  # the caller allocates the result


@needs_toolchain
def test_riscv_object_resolves_kernel(tmp_path):
    import subprocess

    from merlin.llvmlower.lower import lower_model

    res = lower_model(_text(), tmp_path, targets=("riscv",))
    syms = subprocess.run(
        [str(toolchain.clang().with_name("llvm-readelf")), "--syms", "-W", str(res.riscv_obj)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    undefined = {line.split()[-1] for line in syms.splitlines() if len(line.split()) >= 8 and line.split()[6] == "UND"}
    assert ENTRY not in undefined and SYMBOL not in undefined


@needs_toolchain
@pytest.mark.parametrize("textual", [False, True], ids=["xdsl", "textual"])
def test_host_model_matches_golden(textual, tmp_path):
    """The linked module builds for the host too: an x86 oracle for the whole model before any
    simulator is involved."""
    import numpy as np

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    # Own directory per preprocess path: a reused path would dlopen the earlier library.
    res = lower_model(_text(), tmp_path / ("textual" if textual else "xdsl"), targets=("host",), textual=textual)
    args = forward_args(FIXTURE)
    out = np.zeros((4, 8), dtype=np.float32)
    HostModel.load(str(res.host_so))([(a.ctypes.data, a.shape) for a in args] + [(out.ctypes.data, out.shape)])
    np.testing.assert_allclose(out, np.load(FIXTURE / "golden.npy"), rtol=1e-5, atol=1e-5)


_OPERANDS = "tensor<4x16xf32>, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>, tensor<32xf32>"
_STRIDED = (
    "builtin.module {\n"
    "  func.func @forward(%x: tensor<8x16xf32>, %w1: tensor<32x16xf32>, %b1: tensor<32xf32>,"
    " %w2: tensor<32x32xf32>, %b2: tensor<32xf32>) -> tensor<4x32xf32> {\n"
    # Rows 1, 3, 5, 7 of x, in generic form (the xDSL preprocess has no custom parser for it).
    '    %xs = "tensor.extract_slice"(%x) <{operandSegmentSizes = array<i32: 1, 0, 0, 0>,'
    " static_offsets = array<i64: 1, 0>, static_sizes = array<i64: 4, 16>,"
    " static_strides = array<i64: 2, 1>}> : (tensor<8x16xf32>) -> tensor<4x16xf32>\n"
    f"    %0 = func.call @{SYMBOL}(%xs, %w1, %b1, %w2, %b2)"
    ' {prov.kernel = "triton_exp::fused_mlp2", prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"]}'
    f" : ({_OPERANDS}) -> tensor<4x32xf32>\n"
    "    func.return %0 : tensor<4x32xf32>\n"
    "  }\n"
    f"  func.func private @{SYMBOL}({_OPERANDS}) -> tensor<4x32xf32>"
    ' attributes {prov.kernel = "triton_exp::fused_mlp2"}\n'
    "}\n"
)


@needs_toolchain
def test_strided_operand_reaches_kernel_dense(tmp_path):
    """The kernel indexes raw pointers as dense row-major. A strided view (rows 1, 3, 5, 7 of x) must
    be copied dense at the call -- the wrapper's identity-layout arguments force it -- or the kernel
    would read rows 1..4 instead."""
    import numpy as np

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    res = lower_model(_STRIDED, tmp_path, targets=("host",))
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 16), dtype=np.float32)
    w1 = rng.standard_normal((32, 16), dtype=np.float32)
    b1 = rng.standard_normal(32, dtype=np.float32)
    w2 = rng.standard_normal((32, 32), dtype=np.float32)
    b2 = rng.standard_normal(32, dtype=np.float32)
    out = np.full((4, 32), np.nan, dtype=np.float32)
    HostModel.load(str(res.host_so))([(a.ctypes.data, a.shape) for a in (x, w1, b1, w2, b2, out)])
    xs = x[1::2]
    ref = np.maximum(np.maximum(xs @ w1.T + b1, 0) @ w2.T + b2, 0)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)
