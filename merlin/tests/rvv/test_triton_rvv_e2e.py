"""The same Triton source, compiled to RISC-V vector and executed on spike.

This is the architectural gate, not a performance one. The claim under test is that reaching a
second, structurally unrelated backend from the same kernel source cost **zero** lines under
``merlin/python/merlin/triton/`` — the frontend never learns what RVV is; it hands linalg-on-tensors
to Merlin and Merlin decides. If making this pass had required editing the frontend, the boundary
would be in the wrong place.

Outputs come back as raw bit patterns over the console, so "spike agrees with host" is exact
equality rather than a tolerance, and a difference means a real codegen divergence.
"""
from __future__ import annotations

import numpy as np
import pytest
import triton_kernels as K

from merlin.runtime import kernel_spike
from merlin.triton import source
from merlin.triton.bridge import to_linalg

pytestmark = [
    pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed"),
    pytest.mark.skipif(not kernel_spike.available(), reason="chipyard toolchain/spike not available"),
]


def bridged(spec):
    return to_linalg(source.make_ttir(spec), spec)


def host_reference(module, entry: str, inputs: list[np.ndarray], out_shape, out_dtype, workdir):
    """Run the SAME module on the host, so the comparison isolates the RISC-V backend."""
    from merlin.llvmlower.kernel_backend import compile_host, extract_kernel

    model = compile_host(extract_kernel(module, entry), workdir)
    out = np.zeros(out_shape, out_dtype)
    model([(a.ctypes.data, a.shape) for a in inputs] + [(out.ctypes.data, out.shape)])
    return out


@pytest.mark.parametrize("n", [256, 1000, 1025])
def test_vector_add_on_spike_is_bit_identical_to_the_host(n, tmp_path):
    """256 divides the block, 1000 leaves a partial tail, 1025 adds a program for one element."""
    spec = K.vector_add_spec(n=n)
    result = bridged(spec)
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    run = kernel_spike.run(result.module, [x, y], tmp_path / "spike")
    assert np.array_equal(run.outputs[0], x + y), "spike disagrees with numpy"
    host = host_reference(result.module, spec.name, [x, y], (n,), np.float32, tmp_path / "host")
    assert np.array_equal(run.outputs[0], host), "spike and host disagree bit for bit"


def test_the_emitted_object_really_contains_vector_instructions(tmp_path):
    """Evidence, not inference: a scalar loop would pass the numeric gate just as well."""
    result = bridged(K.vector_add_spec(n=1024))
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024).astype(np.float32)
    run = kernel_spike.run(result.module, [x, x], tmp_path)
    assert "vsetvli" in run.vector_ops, run.vector_ops
    assert "vfadd.vv" in run.vector_ops, (
        f"the add was not vectorized; emitted {run.vector_ops}")


def test_the_int8_matmul_runs_on_spike(tmp_path):
    """The accelerator-shaped payload still has to be correct on a plain vector CPU."""
    result = bridged(K.matmul_one_tile_spec())
    rng = np.random.default_rng(5)
    a = rng.integers(-8, 8, size=(K.TILE_M, K.TILE_K), dtype=np.int8)
    b = rng.integers(-8, 8, size=(K.TILE_K, K.TILE_N), dtype=np.int8)
    run = kernel_spike.run(result.module, [a, b], tmp_path)
    assert np.array_equal(run.outputs[0], a.astype(np.int32) @ b.astype(np.int32))


def test_reaching_rvv_needed_no_change_to_the_triton_frontend():
    """The stated architectural gate, asserted rather than claimed.

    Nothing in this arm may reach into the frontend for RVV-specific behavior: the frontend produces
    linalg-on-tensors, and everything RISC-V lives in Merlin's runtime. If this list ever has to
    grow, the abstraction has leaked.
    """
    import ast
    import pathlib

    from merlin.common.paths import merlin_dir

    package = merlin_dir() / "python" / "merlin" / "triton"
    imported: set[str] = set()
    for path in sorted(pathlib.Path(package).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
    assert not {m for m in imported if "runtime" in m or "rvv" in m or "riscv" in m}, imported
