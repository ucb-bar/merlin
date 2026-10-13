"""Linking an external kernel body into a lowered model (``llvmlower.kernel_link``).

The input is a real model2MLIR capture, ``fixtures/mlp_basic_triton`` (``gen_mlp_kernel_fixtures.py``),
whose fc1+relu+fc2+relu is one ``prov.kernel = "triton_exp::fused_mlp2"`` call to a private declaration.
Two facts measured on that capture motivate the first half of these tests:

* the two whole-model preprocessing paths gave the declaration DIFFERENT ABIs — the xDSL path marked it
  ``llvm.emit_c_interface`` (call to ``_mlir_ciface_<sym>(ptr...)``), the textual path did not (call
  to ``<sym>`` with every descriptor field expanded);
* with no access annotation one-shot-bufferize copied all four weights before the call (4 of the
  module's 5 ``llvm.memcpy``).
"""
from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower import kernel_link as KL
from merlin.llvmlower import toolchain

FIXTURE = merlin_dir() / "tests" / "fixtures" / "mlp_basic_triton"
KERNEL_FREE = merlin_dir() / "tests" / "fixtures" / "mlp_basic"
SYMBOL = "triton_exp__fused_mlp2"

needs_toolchain = pytest.mark.skipif(not toolchain.available(),
                                     reason="clang-23 / model2MLIR venv not available")


def _text(path=FIXTURE) -> str:
    return (path / "model.mlir").read_text(encoding="utf-8")


def test_find_kernel_uses_mlp2():
    (use,) = KL.find_kernel_uses(_text())
    assert use.symbol == SYMBOL
    assert use.qualname == "triton_exp::fused_mlp2"
    assert use.operand_names == ("x", "w1", "b1", "w2", "b2")
    assert use.arg_types == (((4, 16), "f32"), ((32, 16), "f32"), ((32,), "f32"),
                             ((32, 32), "f32"), ((32,), "f32"))
    assert use.result_types == (((4, 32), "f32"),)
    assert use.kernel_args == {}


def test_kernel_free_module_has_no_uses():
    assert KL.find_kernel_uses(_text(KERNEL_FREE)) == ()


def test_unknown_kernel_named_in_error():
    text = _text().replace('"triton_exp::fused_mlp2"', '"acme::nope"')
    with pytest.raises(KL.KernelLinkError) as err:
        KL.find_kernel_uses(text)
    assert "acme::nope" in str(err.value) and "triton_exp::fused_mlp2" in str(err.value)


def test_prepare_marks_declaration():
    uses = KL.find_kernel_uses(_text())
    out = KL.prepare_kernel_declarations(_text(), uses)
    decl = next(line for line in out.splitlines() if f"func.func private @{SYMBOL}(" in line)
    assert decl.count('{bufferization.access = "read"}') == 5
    assert "llvm.emit_c_interface" in decl
    assert 'prov.kernel = "triton_exp::fused_mlp2"' in decl      # existing attributes kept


def test_prepare_is_idempotent():
    uses = KL.find_kernel_uses(_text())
    once = KL.prepare_kernel_declarations(_text(), uses)
    assert KL.prepare_kernel_declarations(once, uses) == once


def test_prepare_leaves_kernel_free_text_alone():
    assert KL.prepare_kernel_declarations(_text(KERNEL_FREE), ()) == _text(KERNEL_FREE)


def _forward_body(ll: str) -> str:
    start = ll.index("define void @forward(")
    return ll[start:ll.index("\n}\n", start)]


@needs_toolchain
@pytest.mark.parametrize("textual", [False, True], ids=["xdsl", "textual"])
def test_both_paths_emit_ciface_call(textual, tmp_path):
    from merlin.llvmlower.lower import lower_model
    res = lower_model(_text(), tmp_path, targets=(), textual=textual, vectorize=True)
    ll = (tmp_path / "model.ll").read_text(encoding="utf-8")
    assert f"declare void @_mlir_ciface_{SYMBOL}(ptr, ptr, ptr, ptr, ptr, ptr)" in ll
    assert res.stats["kernel_uses"] == 1


@needs_toolchain
def test_weights_not_copied_before_kernel(tmp_path):
    from merlin.llvmlower.lower import lower_model
    lower_model(_text(), tmp_path, targets=(), textual=True, vectorize=True)
    body = _forward_body((tmp_path / "model.ll").read_text(encoding="utf-8"))
    before_call = body[:body.index(f"@{SYMBOL}(")]
    assert "@llvm.memcpy" not in before_call
