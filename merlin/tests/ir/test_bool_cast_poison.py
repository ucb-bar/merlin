"""float -> i1 bool cast: ``arith.fptosi`` is POISON, and poison erases whole models.

model2MLIR emits ``aten._to_copy`` to a bool tensor as ``arith.fptosi %x : f32 to i1``. Signed
``i1`` holds only ``{-1, 0}``, so every float whose truncation is not one of those two — ``1.0``
included — is poison in LLVM. That is categorically worse than the sign flip
:func:`~merlin.llvmlower.passes_xdsl.fix_bool_sitofp` repairs on the other direction: poison
propagates, and a branch on it lets ``simplifycfg`` delete every block after it.

Measured on ``smolvla_int8_consistent`` (a ~500M-parameter VLA model): the poisoned bool mask fed a
masked-select whose result count sized a ``malloc`` and bounded a data-dependent loop, LLVM folded
that loop's exit test to a self-branch, and ``forward`` compiled to 3,654 bytes — 8 ``malloc``, 2
``memset``, 2 ``roundevenf``, and no compute at all — while the link succeeded and the build
reported success. With the rewrite in place the same IR compiles to a 2,663,634-byte ``forward``.

The structural tests run everywhere xDSL is present; the numerical one compiles and executes both
forms on the host (auto-skips without the toolchain).
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def _bool_cast_src(dtype: str = "f32") -> str:
    """``out = float(bool(x))`` exactly as model2MLIR emits it: fptosi straight to i1."""
    return (
        "builtin.module { func.func @forward(%x: tensor<4x" + dtype + ">) -> tensor<4xf32> { "
        "%e = tensor.empty() : tensor<4xi1> "
        "%b = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]} "
        "ins(%x : tensor<4x" + dtype + ">) outs(%e : tensor<4xi1>) { "
        "^bb0(%xv: " + dtype + ", %o: i1): "
        "%c = arith.fptosi %xv : " + dtype + " to i1 "
        "linalg.yield %c : i1 } -> tensor<4xi1> "
        "%e2 = tensor.empty() : tensor<4xf32> "
        "%r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        "affine_map<(d0) -> (d0)>], iterator_types = [\"parallel\"]} "
        "ins(%b : tensor<4xi1>) outs(%e2 : tensor<4xf32>) { "
        "^bb0(%bv: i1, %o2: f32): "
        "%f = arith.uitofp %bv : i1 to f32 "
        "linalg.yield %f : f32 } -> tensor<4xf32> "
        "func.return %r : tensor<4xf32> } }")


def test_pass_rewrites_float_to_i1_fptosi_as_ne_zero():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi

    m = parse_mlir_text(_bool_cast_src())
    assert any(op.name == "arith.fptosi" for op in m.walk())
    assert fix_bool_fptosi(m) == 1
    m.verify()
    assert not any(op.name == "arith.fptosi" for op in m.walk())
    cmps = [op for op in m.walk() if op.name == "arith.cmpf"]
    assert len(cmps) == 1
    # `une` (unordered-or-not-equal) is MLIR cmpf predicate 13 — NaN must compare TRUE, as
    # torch's `bool(nan)` is. `one` (6, ordered) would silently turn NaN into False.
    assert int(cmps[0].properties["predicate"].value.data) == 13


def test_pass_leaves_wider_integer_conversions_alone():
    """A cast to i8/i32 is a real numeric conversion, not a bool test — it must survive."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi

    for width in ("i8", "i32"):
        m = parse_mlir_text(_bool_cast_src().replace("to i1", f"to {width}")
                            .replace("tensor<4xi1>", f"tensor<4x{width}>")
                            .replace("%o: i1", f"%o: {width}")
                            .replace("linalg.yield %c : i1", f"linalg.yield %c : {width}")
                            .replace("%bv: i1", f"%bv: {width}")
                            .replace("arith.uitofp %bv : i1", f"arith.uitofp %bv : {width}"))
        assert fix_bool_fptosi(m) == 0


def test_pass_covers_every_float_width():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi

    for dtype in ("f32", "f16", "bf16", "f64"):
        m = parse_mlir_text(_bool_cast_src(dtype))
        assert fix_bool_fptosi(m) == 1, dtype
        m.verify()


def test_prepare_wires_the_fix_into_both_paths():
    """The compiled path and the numpy interpreter must apply the SAME bool-cast semantics.

    They diverge silently otherwise: ``int(x)`` in the interpreter, poison in the object, and the
    interpreter's golden then certifies a compiled model that no longer exists.
    """
    from merlin.common.paths import merlin_dir

    root = merlin_dir() / "python" / "merlin"
    for rel in ("runtime/backends/zephyr_model.py", "runtime/dispatch_runtime.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "fix_bool_fptosi(module)" in src, rel


def _toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_bool_cast_values_match_torch_after_fix(tmp_path):
    """``bool(x)`` is ``x != 0`` for every value, not a 1-bit signed truncation."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
    from merlin.xdsl_dialects._common import text as to_text

    x = np.array([0.0, 1.0, 2.5, -3.0], np.float32)
    mod = parse_mlir_text(_bool_cast_src())
    assert fix_bool_fptosi(mod) == 1
    res = lower_model(to_text(mod), tmp_path / "fixed", targets=("host",))
    model = HostModel.load(str(res.host_so))
    out = np.zeros(4, np.float32)
    model([(x.ctypes.data, (4,)), (out.ctypes.data, (4,))])
    assert list(out) == [0.0, 1.0, 1.0, 1.0]        # torch: (x != 0).float()


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_unfixed_cast_reaches_llvm_as_a_poison_source(tmp_path):
    """The un-rewritten form really does hand LLVM a float->i1 fptosi; the rewritten one does not.

    This is the property that matters: not that the numbers differ (poison's numbers are not
    defined, so asserting on them would be asserting on a coin flip) but that the poison SOURCE is
    gone from the IR the backend receives.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
    from merlin.xdsl_dialects._common import text as to_text

    def to_i1_fptosi_count(text, tag):
        ll = lower_model(text, tmp_path / tag, targets=(), textual=True).ll_path
        n = 0
        for line in ll.read_text(encoding="utf-8").splitlines():
            head, sep, rest = line.partition(" = fptosi ")
            if sep and rest.strip().endswith(" to i1"):
                n += 1
        return n

    assert to_i1_fptosi_count(_bool_cast_src(), "unfixed") == 1
    mod = parse_mlir_text(_bool_cast_src())
    fix_bool_fptosi(mod)
    assert to_i1_fptosi_count(to_text(mod), "fixed") == 0
