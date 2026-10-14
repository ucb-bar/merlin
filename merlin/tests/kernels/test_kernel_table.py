"""The external kernel table: model2MLIR ``prov.kernel`` names -> kernel bodies merlin can link.

Three ways the table can rot, one test each: a key m2m emits that merlin has no entry for (the
model then fails at link time with an undefined symbol), a body file that is missing or not what
its registration claims (target-specific IR registered as portable), and an ABI tuple that does not
describe how the body is actually called (wrong answers rather than an error). The last is checked
by DRIVING the body through its own ``abi`` on the host, so the table data itself is what is tested.
"""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.llvmlower import kernel_table as KT
from merlin.llvmlower import toolchain


def _bodies():
    return [(e.qualname, b) for e in KT.KERNEL_TABLE.values() for b in e.bodies]


def test_lookup_unknown_lists_keys():
    with pytest.raises(KT.KernelTableError) as err:
        KT.lookup("nope::x")
    assert "nope::x" in str(err.value)
    assert "triton_exp::fused_mlp2" in str(err.value)


def test_body_by_unknown_form_lists_forms():
    entry = KT.lookup("triton_exp::fused_mlp2")
    with pytest.raises(KT.KernelTableError) as err:
        entry.body("spirv")
    assert "llvm_ir" in str(err.value) and "llvm_mlir" in str(err.value)


def test_mlp2_has_both_forms():
    entry = KT.lookup("triton_exp::fused_mlp2")
    assert [b.form for b in entry.bodies] == ["llvm_ir", "llvm_mlir"]
    assert entry.body().form == "llvm_ir"  # first body is the default


@pytest.mark.parametrize("qualname,body", _bodies(), ids=lambda v: getattr(v, "form", v))
def test_every_body_file_exists_and_is_portable(qualname, body):
    """A portable body has its triple and datalayout dropped at link, which is only sound when the
    IR has nothing that belongs to one ISA: no arch intrinsic namespace, no inline asm, no address
    spaces (an NVPTX export has all three)."""
    path = KT.body_path(body)
    assert path.is_file(), f"{qualname}: {path} missing"
    text = path.read_text(encoding="utf-8")
    assert body.entry in text
    if body.portable:
        for marker in (
            "@llvm.x86.",
            "@llvm.nvvm.",
            "@llvm.amdgcn.",
            "@llvm.riscv.",
            "addrspace(",
            "inline_asm",
            " asm ",
        ):
            assert marker not in text, f"{path.name} is registered portable but contains {marker!r}"


def _host_object(body, tmp_path):
    """Compile the body for THIS host so its numerics can be checked without a simulator."""
    if not toolchain.clang().is_file():
        pytest.skip("clang-23 not available")
    src = KT.body_path(body)
    if body.form == "llvm_mlir":
        if not toolchain.mlir_translate().is_file():
            pytest.skip("mlir-translate not available")
        ll = tmp_path / "body.ll"
        subprocess.run(
            [str(toolchain.mlir_translate()), "--mlir-to-llvmir", str(src), "-o", str(ll)],
            check=True,
            capture_output=True,
        )
        src = ll
    obj, so = tmp_path / "body.o", tmp_path / "body.so"
    subprocess.run(
        [str(toolchain.clang()), "-O2", "-fPIC", "-Wno-override-module", "-c", str(src), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["cc", "-shared", str(obj), "-o", str(so)], check=True, capture_output=True)
    return ctypes.CDLL(str(so))


@pytest.mark.parametrize("body", KT.lookup("triton_exp::fused_mlp2").bodies, ids=lambda b: b.form)
@pytest.mark.parametrize("m", [4, 16, 20])
def test_mlp2_body_abi_matches_numpy(body, m, tmp_path):
    """Call the body exactly as its ``abi``/``block`` say, over the grid they imply, and compare with
    the nn.Linear-layout MLP it stands for. M=20 exercises the partial last block (masking)."""
    lib = _host_object(body, tmp_path)
    rng = np.random.default_rng(0)
    arrays = {
        "x": rng.standard_normal((m, 16), dtype=np.float32),
        "w1": rng.standard_normal((32, 16), dtype=np.float32),
        "b1": rng.standard_normal(32, dtype=np.float32),
        "w2": rng.standard_normal((32, 32), dtype=np.float32),
        "b2": rng.standard_normal(32, dtype=np.float32),
    }
    out = np.full((m, 32), np.nan, dtype=np.float32)
    operand, axis, size = body.block
    grid = (arrays[operand].shape[axis] + size - 1) // size
    fn = getattr(lib, body.entry)
    for pid in range(grid):
        args = []
        for p in body.abi:
            if p.kind == "operand":
                args.append(ctypes.c_void_p(arrays[p.name].ctypes.data))
            elif p.kind == "result":
                args.append(ctypes.c_void_p(out.ctypes.data))
            elif p.kind == "dim":
                args.append(ctypes.c_int32(arrays[p.name].shape[p.index]))
            elif p.kind == "program_id":
                args.append(ctypes.c_int32(pid if p.index == 0 else 0))
            elif p.kind == "grid":
                args.append(ctypes.c_int32(grid if p.index == 0 else 1))
            else:
                args.append(ctypes.c_void_p(0))
        fn(*args)
    x, w1, b1, w2, b2 = (arrays[k] for k in ("x", "w1", "b1", "w2", "b2"))
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
