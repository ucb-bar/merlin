"""A model with an external kernel, end to end on spike rv64gcv.

``mlp_basic_triton`` computes fc1+relu+fc2+relu as one triton-cpu kernel linked in by
``llvmlower.kernel_link``; ``mlp_basic`` is the same model in plain linalg. Both must reproduce the
PyTorch golden of the kernel-free model, for both body forms, through the whole-model ``spike_model``
runner (weights blob + the C runtime + the bare-metal allocator).

``runtime.kernel_spike`` is deliberately not exercised: its harness links no allocator, so it cannot
run any whole capture (kernel-free ``mlp_basic`` fails to link on ``malloc``/``free`` too), and a kernel
shim allocates its result.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from mlp_bundles import MLP_BASIC, MLP_BASIC_TRITON, golden

from merlin.common.kernel_calls import entry_function, is_external_decl
from merlin.common.mlir_query import parse


def _spike_ready() -> bool:
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike

    return spike.available() and toolchain.available() and toolchain.llvm_link().is_file()


needs_spike = pytest.mark.skipif(not _spike_ready(), reason="spike / chipyard / clang / llvm-link missing")


def test_entry_function_skips_declarations():
    module = parse((MLP_BASIC_TRITON / "model.mlir").read_text())
    assert entry_function(module).sym_name.data == "forward"
    decls = [op for op in module.walk() if op.name == "func.func" and is_external_decl(op)]
    assert [d.sym_name.data for d in decls] == ["triton_exp__fused_mlp2"]


def _whole_model(bundle, tmp_path, **kw):
    from merlin.runtime.backends import spike_model

    model = tmp_path / "model"
    shutil.copytree(bundle, model)
    build = spike_model.build(model, tmp_path / "work", **kw)
    run = spike_model.run(build["elf"], mem_bytes=build["mem_bytes"], vlen=build.get("vlen"))
    return build, run


@needs_spike
def test_mlp_basic_whole_model_on_spike(tmp_path):
    """The kernel-free twin: the baseline the kernel build is compared against."""
    build, run = _whole_model(MLP_BASIC, tmp_path)
    assert build["kernels"] == []
    np.testing.assert_allclose(run["outputs"], golden(MLP_BASIC).ravel(), rtol=1e-5, atol=1e-5)


@needs_spike
@pytest.mark.parametrize("form", ["llvm_ir", "llvm_mlir"])
def test_mlp_basic_triton_whole_model_on_spike(form, tmp_path):
    build, run = _whole_model(MLP_BASIC_TRITON, tmp_path, kernel_form=form)
    assert build["kernels"] == [f"triton_exp::fused_mlp2:{form}"]
    np.testing.assert_allclose(run["outputs"], golden(MLP_BASIC_TRITON).ravel(), rtol=1e-5, atol=1e-5)
