"""`fuse_transpose_b` folds a `linalg.transpose` of a matmul's B operand INTO the matmul.

The whole-model profiler measured `linalg.transpose` at 393 ms in openvla (scalar, unvectorized);
every openvla matmul is a transposed-B addmm fed by a standalone weight transpose. This feature
rewrites `matmul(A, transpose(B, [1,0]))` into a transpose-b `linalg.matmul` (B indexing map
`(k,n)->(n,k)`, reading the un-transposed weight) and erases the dead transpose, so the scalar
transpose disappears while the matmul stays vectorized by the frozen RVV schedule.

Two levers in this repo (KC, MR under unroll_m) looked correctly wired while being inert, so the
fusion is proved on the ACTUAL rewrite output (the shipping RUNNER_PRELUDE, run in the m2m venv),
not on the schedule text. The pure-python part guards the default-off invariant (baseline lowering
byte-identical) without needing torch-mlir.
"""
from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

from merlin.llvmlower import impr_features as F
from merlin.llvmlower.impr_features import apply_pipeline, apply_schedule
from merlin.llvmlower.pipeline import (RVV_TRANSFORM_SCHEDULE, _RUNNER, build_rvv_pipeline)
from merlin.llvmlower.transpose_fuse import FEATURE, RUNNER_PRELUDE

_FEAT = "fuse_transpose_b"


def test_feature_is_registered_with_the_expected_name():
    assert FEATURE == _FEAT
    assert _FEAT in F.known()
    assert F.get(_FEAT).action_class == "PASS"


def test_default_off_lowering_is_byte_identical():
    """The rewrite is runner-side and gated by argv[5], so enabling the feature changes NEITHER the
    transform schedule NOR the pass pipeline — the frozen hand_v0 control stays byte-identical."""
    fs = frozenset({_FEAT})
    assert apply_schedule(RVV_TRANSFORM_SCHEDULE, fs) == apply_schedule(RVV_TRANSFORM_SCHEDULE,
                                                                        frozenset())
    base = build_rvv_pipeline("/tmp/s.mlir", features=frozenset())
    feat = build_rvv_pipeline("/tmp/s.mlir", features=fs)
    assert base == feat


def test_runner_wires_the_gated_fusion_hook():
    """The plain lowering runner carries the fusion prelude, the argv[5] gate, and the call — so a
    package that lists the feature actually triggers the rewrite."""
    assert "_fuse_transpose_b(module, ctx)" in _RUNNER
    assert "_FUSE_TRANSPOSE_B" in _RUNNER
    assert "sys.argv[5]" in RUNNER_PRELUDE


def test_runner_prelude_is_valid_python():
    # argv guard needs `sys`; compile in that namespace exactly as the runner splices it.
    compile("import sys\n" + RUNNER_PRELUDE, "<transpose_fuse_prelude>", "exec")


# ---- emitted-code proof (needs the model2mlir venv / torch-mlir bindings) ------------

def _m2m() -> Path | None:
    try:
        from merlin.llvmlower.toolchain import m2m_python
        p = Path(m2m_python())
        return p if p.is_file() else None
    except Exception:  # noqa: BLE001
        return None


# A minimal transposed-B matmul in the SAME shape the frontend emits: a standalone `linalg.transpose`
# of the B weight (perm [1,0]) feeding a `linalg.matmul` whose B indexing_map reads (k,n) = (d2,d1).
_TRANSPOSE_B_MODULE = textwrap.dedent("""\
    #mapA = affine_map<(d0, d1, d2) -> (d0, d2)>
    #mapB = affine_map<(d0, d1, d2) -> (d2, d1)>
    #mapC = affine_map<(d0, d1, d2) -> (d0, d1)>
    module {
      func.func @forward(%A: tensor<8x16xf32>, %W: tensor<32x16xf32>, %C0: tensor<8x32xf32>) -> tensor<8x32xf32> {
        %e = tensor.empty() : tensor<16x32xf32>
        %Bt = linalg.transpose ins(%W : tensor<32x16xf32>) outs(%e : tensor<16x32xf32>) permutation = [1, 0]
        %C = linalg.matmul indexing_maps = [#mapA, #mapB, #mapC] ins(%A, %Bt : tensor<8x16xf32>, tensor<16x32xf32>) outs(%C0 : tensor<8x32xf32>) -> tensor<8x32xf32>
        return %C : tensor<8x32xf32>
      }
    }
    """)

# Driver run INSIDE the m2m venv: parse the module, apply the shipping rewrite, report counts + the
# rewritten B map, and verify the module still type-checks.
_PROBE = "import sys\n" + RUNNER_PRELUDE + textwrap.dedent("""\
    from torch_mlir import ir
    ctx = ir.Context(); ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(open(sys.argv[1]).read(), ctx)
    n = _fuse_transpose_b(module, ctx)
    module.operation.verify()
    txt = str(module.operation)
    print("FUSED", n)
    print("TRANSPOSES", txt.count("linalg.transpose"))
    # the B map must now read (d1, d2) = (n, k) of the un-transposed weight
    print("HAS_NK_MAP", "(d0, d1, d2) -> (d1, d2)" in txt)
    """)


@pytest.mark.skipif(_m2m() is None, reason="model2mlir venv missing")
def test_rewrite_folds_the_transpose_into_the_matmul():
    """Prove on the ACTUAL shipping rewrite that the standalone transpose is folded away and the
    matmul's B access pattern becomes (n, k) — value-identical transpose-b GEMM."""
    d = Path(tempfile.mkdtemp(prefix="tr_fuse_test_"))
    (d / "m.mlir").write_text(_TRANSPOSE_B_MODULE)
    (d / "probe.py").write_text(_PROBE)
    out = subprocess.run([str(_m2m()), str(d / "probe.py"), str(d / "m.mlir")],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    kv = dict(ln.split(maxsplit=1) for ln in out.stdout.splitlines() if " " in ln)
    assert kv.get("FUSED") == "1", out.stdout
    assert kv.get("TRANSPOSES") == "0", out.stdout          # the dead transpose is erased
    assert kv.get("HAS_NK_MAP") == "True", out.stdout       # B now read (n, k)
