"""The external kernel table: model2MLIR ``prov.kernel`` names -> kernel bodies merlin can link.

Three ways the table can rot, one test each: a key m2m emits that merlin has no entry for (the
model then fails to lower, naming it), a body file that is missing or not target-neutral, and a
launch (``args`` + ``grid``) that does not describe how the kernel is really called (wrong answers
rather than an error). The last is checked by LOWERING a call of the kernel through the generated
wrapper on the host and comparing with numpy, so the table data itself is what is tested.
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from merlin.llvmlower import kernel_table as KT
from merlin.llvmlower import toolchain


def _bodies():
    return [(e.qualname, e.body) for e in KT.KERNEL_TABLE.values()]


def test_lookup_unknown_lists_keys():
    with pytest.raises(KT.KernelTableError) as err:
        KT.lookup("nope::x")
    assert "nope::x" in str(err.value)
    assert "triton_exp::fused_mlp2" in str(err.value)


@pytest.mark.parametrize("qualname,body", _bodies(), ids=lambda v: getattr(v, "entry", v))
def test_every_body_file_exists_and_is_portable(qualname, body):
    """A body is merged into the model module and compiled for the model's target, which is only sound
    when it has nothing that belongs to one ISA: no arch intrinsic namespace or dialect, no inline asm,
    no address spaces (an NVPTX export has all three)."""
    path = KT.body_path(body)
    assert path.is_file(), f"{qualname}: {path} missing"
    text = path.read_text(encoding="utf-8")
    assert f"llvm.func @{body.entry}(" in text
    for marker in (
        '"llvm.x86.',
        '"llvm.nvvm.',
        '"llvm.amdgcn.',
        '"llvm.riscv.',
        "x86vector.",
        "nvvm.",
        "rocdl.",
        "arm_sve.",
        "arm_neon.",
        "llvm.inline_asm",
        "!llvm.ptr<",
    ):
        assert marker not in text, f"{path.name} is linked into every target but contains {marker!r}"


def test_every_result_written_once():
    for qualname, body in _bodies():
        written = sorted(a.index for a in body.args if a.kind == "result")
        assert written == list(range(len(written))), f"{qualname}: results {written}"
        assert len(body.grid) <= 3, qualname


def _capture(m: int) -> str:
    """A minimal model2MLIR-shaped module: @forward returns one call of the mlp2 kernel at M = ``m``."""
    x, out = f"tensor<{m}x16xf32>", f"tensor<{m}x32xf32>"
    operands = f"{x}, tensor<32x16xf32>, tensor<32xf32>, tensor<32x32xf32>, tensor<32xf32>"
    return (
        "builtin.module {\n"
        f"  func.func @forward(%x: {x}, %w1: tensor<32x16xf32>, %b1: tensor<32xf32>, "
        f"%w2: tensor<32x32xf32>, %b2: tensor<32xf32>) -> {out} {{\n"
        f"    %0 = func.call @triton_exp__fused_mlp2(%x, %w1, %b1, %w2, %b2) "
        '{prov.kernel = "triton_exp::fused_mlp2", prov.kernel_operands = ["x", "w1", "b1", "w2", "b2"]} '
        f": ({operands}) -> {out}\n"
        f"    func.return %0 : {out}\n"
        "  }\n"
        f"  func.func private @triton_exp__fused_mlp2({operands}) -> {out} "
        'attributes {prov.kernel = "triton_exp::fused_mlp2"}\n'
        "}\n"
    )


@pytest.mark.skipif(not toolchain.available(), reason="clang-23 / model2MLIR venv not available")
@pytest.mark.parametrize("m", [4, 16, 20])
def test_mlp2_launch_matches_numpy(m, tmp_path):
    """Lower a call of the kernel at M rows on the host and compare with the nn.Linear-layout MLP it
    stands for. M=20 is a two-program grid whose last block is partial (masking)."""
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    # A directory per M: pytest hands a passed parametrization's tmp_path name to the next one, and
    # dlopen of a path already loaded returns the earlier library, not the new file.
    res = lower_model(_capture(m), tmp_path / f"m{m}", targets=("host",))
    assert res.stats["kernels_linked"] == ["triton_exp::fused_mlp2"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((m, 16), dtype=np.float32)
    w1 = rng.standard_normal((32, 16), dtype=np.float32)
    b1 = rng.standard_normal(32, dtype=np.float32)
    w2 = rng.standard_normal((32, 32), dtype=np.float32)
    b2 = rng.standard_normal(32, dtype=np.float32)
    out = np.full((m, 32), np.nan, dtype=np.float32)
    HostModel.load(str(res.host_so))([(a.ctypes.data, a.shape) for a in (x, w1, b1, w2, b2, out)])
    ref = np.maximum(np.maximum(x @ w1.T + b1, 0) @ w2.T + b2, 0)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-4)


def test_table_keys_match_m2m_registry():
    """Every kernel model2MLIR can emit has an entry here; otherwise its captures cannot link."""
    py = toolchain.m2m_python()
    if not py.is_file():
        pytest.skip("model2MLIR venv not configured")
    got = subprocess.run(
        [
            str(py),
            "-c",
            "from m2m.ir.kernel_registry import KERNEL_REGISTRY;print('\\n'.join(sorted(KERNEL_REGISTRY)))",
        ],
        cwd=str(toolchain.m2m_dir()),
        capture_output=True,
        text=True,
    )
    if got.returncode != 0:
        pytest.skip(f"model2MLIR registry not importable: {got.stderr[-300:]}")
    m2m_keys = [k for k in got.stdout.split() if k]
    assert m2m_keys, "model2MLIR reported no registered kernels"
    missing = sorted(set(m2m_keys) - set(KT.KERNEL_TABLE))
    assert not missing, f"model2MLIR kernels with no merlin entry: {missing}"
