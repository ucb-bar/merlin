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
from mlp_bundles import forward_args

from merlin.common.paths import merlin_dir
from merlin.llvmlower import kernel_link as KL
from merlin.llvmlower import toolchain

FIXTURE = merlin_dir() / "tests" / "fixtures" / "mlp_basic_triton"
KERNEL_FREE = merlin_dir() / "tests" / "fixtures" / "mlp_basic"
SYMBOL = "triton_exp__fused_mlp2"

needs_toolchain = pytest.mark.skipif(not toolchain.available(), reason="clang-23 / model2MLIR venv not available")


def _text(path=FIXTURE) -> str:
    return (path / "model.mlir").read_text(encoding="utf-8")


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


def test_prepare_marks_declaration():
    uses = KL.find_kernel_uses(_text())
    out = KL.prepare_kernel_declarations(_text(), uses)
    decl = next(line for line in out.splitlines() if f"func.func private @{SYMBOL}(" in line)
    assert decl.count('{bufferization.access = "read"}') == 5
    assert "llvm.emit_c_interface" in decl
    assert 'prov.kernel = "triton_exp::fused_mlp2"' in decl  # existing attributes kept


def test_prepare_is_idempotent():
    uses = KL.find_kernel_uses(_text())
    once = KL.prepare_kernel_declarations(_text(), uses)
    assert KL.prepare_kernel_declarations(once, uses) == once


def test_prepare_marks_generic_form_declaration():
    """The xDSL preprocess prints the declaration generically; arg_attrs is a property there."""
    from merlin.llvmlower.passes_xdsl import preprocess_text

    upstream, _ = preprocess_text(_text())
    uses = KL.find_kernel_uses(_text())
    once = KL.prepare_kernel_declarations(upstream, uses)
    at = once.index(f'sym_name = "{SYMBOL}"')
    props = once[once.rfind("<{", 0, at) : once.index("}>", at)]
    assert props.count('{bufferization.access = "read"}') == 5
    assert KL.prepare_kernel_declarations(once, uses) == once


def test_prepare_leaves_kernel_free_text_alone():
    assert KL.prepare_kernel_declarations(_text(KERNEL_FREE), ()) == _text(KERNEL_FREE)


def _forward_body(ll: str) -> str:
    start = ll.index("define void @forward(")
    return ll[start : ll.index("\n}\n", start)]


@needs_toolchain
@pytest.mark.parametrize("textual", [False, True], ids=["xdsl", "textual"])
def test_both_paths_emit_ciface_call(textual, tmp_path):
    from merlin.llvmlower.lower import lower_model

    res = lower_model(_text(), tmp_path, targets=(), textual=textual, vectorize=True)
    ll = (tmp_path / "model.ll").read_text(encoding="utf-8")
    assert f"declare void @_mlir_ciface_{SYMBOL}(ptr, ptr, ptr, ptr, ptr, ptr)" in ll
    assert res.stats["kernel_uses"] == 1


@needs_toolchain
@pytest.mark.parametrize("textual", [False, True], ids=["xdsl", "textual"])
def test_weights_not_copied_before_kernel(textual, tmp_path):
    """Both preprocess paths: the xDSL one prints the declaration in GENERIC form
    (``"func.func"() <{sym_name = ...}>``), which a pretty-form-only patch silently missed."""
    from merlin.llvmlower.lower import lower_model

    lower_model(_text(), tmp_path, targets=(), textual=textual, vectorize=True)
    body = _forward_body((tmp_path / "model.ll").read_text(encoding="utf-8"))
    before_call = body[: body.index(f"@{SYMBOL}(")]
    assert "@llvm.memcpy" not in before_call


# --------------------------------------------------------------------------- shim + link

import dataclasses

from merlin.llvmlower import kernel_table as KT


def _use(**over):
    (use,) = KL.find_kernel_uses(_text())
    return dataclasses.replace(use, **over)


def test_emit_shim_mlp2_text():
    src = KL.emit_shim([_use()], target="riscv")
    assert f"void _mlir_ciface_{SYMBOL}(" in src
    call = src[src.index("fused_mlp2_kernel(", src.index(f"_mlir_ciface_{SYMBOL}(")) :]
    call = call[: call.index(");")]
    assert call.count(",") == 12  # 13 arguments: 6 ptr + M + 6 launch ints
    assert "->sizes[0]" in call  # M is read from x's descriptor, not baked


def test_two_shapes_two_entries_one_body(tmp_path):
    a = _use()
    b = _use(symbol=SYMBOL + "_1", arg_types=(((8, 16), "f32"),) + a.arg_types[1:], result_types=(((8, 32), "f32"),))
    src = KL.emit_shim([a, b], target="riscv")
    assert src.count("void _mlir_ciface_") == 2
    assert src.count("void fused_mlp2_kernel(") == 1  # one prototype for the shared body


def test_specialization_mismatch_rejected():
    bad = _use(arg_types=(((4, 24), "f32"),) + _use().arg_types[1:])
    with pytest.raises(KL.KernelLinkError) as err:
        KL.emit_shim([bad], target="riscv")
    assert "x" in str(err.value) and "16" in str(err.value) and "24" in str(err.value)


def test_operand_dtype_mismatch_rejected():
    bad = _use(arg_types=(((4, 16), "bf16"),) + _use().arg_types[1:])
    with pytest.raises(KL.KernelLinkError) as err:
        KL.emit_shim([bad], target="riscv")
    assert "bf16" in str(err.value) and "f32" in str(err.value)


def test_multi_result_rejected():
    with pytest.raises(KL.KernelLinkError, match="result"):
        KL.emit_shim([_use(result_types=_use().result_types * 2)], target="riscv")


def test_kernel_args_rejected():
    with pytest.raises(KL.KernelLinkError, match="kernel_args"):
        KL.emit_shim([_use(kernel_args={"alpha": "5.0e-01 : f64"})], target="riscv")


def test_link_arch_is_derived_from_codegen_flags():
    assert KL.link_arch("riscv") == "riscv64"


@needs_toolchain
def test_nonportable_foreign_triple_refused(tmp_path, monkeypatch):
    """A body that is NOT target-neutral is only linked for the arch its triple names: the x86 export
    registered as non-portable must be refused for riscv by name, before llvm-link or clang run."""
    entry = KT.KERNEL_TABLE["triton_exp::fused_mlp2"]
    body = dataclasses.replace(entry.bodies[0], portable=False, triple="x86_64")
    monkeypatch.setitem(KT.KERNEL_TABLE, entry.qualname, dataclasses.replace(entry, bodies=(body,)))
    from merlin.llvmlower.lower import lower_model

    with pytest.raises(KL.KernelLinkError) as err:
        lower_model(_text(), tmp_path, targets=("riscv",))
    assert "mlp2.ll" in str(err.value) and "x86_64" in str(err.value)


@needs_toolchain
@pytest.mark.parametrize("form", ["llvm_ir", "llvm_mlir"])
def test_linked_object_resolves_kernel(form, tmp_path):
    import subprocess

    from merlin.llvmlower.lower import lower_model

    res = lower_model(_text(), tmp_path, targets=("riscv",), vectorize=True, kernel_form=form)
    assert res.ll_path.name == "model.linked.ll"
    assert res.stats["kernels_linked"] == [f"triton_exp::fused_mlp2:{form}"]
    syms = subprocess.run(
        [str(toolchain.clang().with_name("llvm-readelf")), "--syms", "-W", str(res.riscv_obj)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    undefined = {line.split()[-1] for line in syms.splitlines() if len(line.split()) >= 8 and line.split()[6] == "UND"}
    assert f"_mlir_ciface_{SYMBOL}" not in undefined and "fused_mlp2_kernel" not in undefined


@needs_toolchain
@pytest.mark.parametrize("form", ["llvm_ir", "llvm_mlir"])
def test_host_linked_model_matches_golden(form, tmp_path):
    """Portable bodies link for the host too, which gives an x86 oracle for the whole model before
    any simulator is involved."""
    import numpy as np

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    res = lower_model(_text(), tmp_path, targets=("host",), kernel_form=form)
    args = forward_args(FIXTURE)
    out = np.zeros((4, 8), dtype=np.float32)
    HostModel.load(str(res.host_so))([(a.ctypes.data, a.shape) for a in args] + [(out.ctypes.data, out.shape)])
    np.testing.assert_allclose(out, np.load(FIXTURE / "golden.npy"), rtol=1e-5, atol=1e-5)
