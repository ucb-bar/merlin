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
        'affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} '
        "ins(%x : tensor<4x" + dtype + ">) outs(%e : tensor<4xi1>) { "
        "^bb0(%xv: " + dtype + ", %o: i1): "
        "%c = arith.fptosi %xv : " + dtype + " to i1 "
        "linalg.yield %c : i1 } -> tensor<4xi1> "
        "%e2 = tensor.empty() : tensor<4xf32> "
        "%r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, "
        'affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} '
        "ins(%b : tensor<4xi1>) outs(%e2 : tensor<4xf32>) { "
        "^bb0(%bv: i1, %o2: f32): "
        "%f = arith.uitofp %bv : i1 to f32 "
        "linalg.yield %f : f32 } -> tensor<4xf32> "
        "func.return %r : tensor<4xf32> } }"
    )


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
        m = parse_mlir_text(
            _bool_cast_src()
            .replace("to i1", f"to {width}")
            .replace("tensor<4xi1>", f"tensor<4x{width}>")
            .replace("%o: i1", f"%o: {width}")
            .replace("linalg.yield %c : i1", f"linalg.yield %c : {width}")
            .replace("%bv: i1", f"%bv: {width}")
            .replace("arith.uitofp %bv : i1", f"arith.uitofp %bv : {width}")
        )
        assert fix_bool_fptosi(m) == 0


def test_pass_covers_every_float_width():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi

    for dtype in ("f32", "f16", "bf16", "f64"):
        m = parse_mlir_text(_bool_cast_src(dtype))
        assert fix_bool_fptosi(m) == 1, dtype
        m.verify()


def _assert_frontend_intent_shape(module, source=None):
    """Small mutation oracle for the reviewed `cmpf une x, +0` repair shape."""
    import math

    ops = list(module.walk())
    assert not any(op.name in ("arith.fptosi", "arith.fptoui", "arith.trunci") for op in ops)
    cmps = [op for op in ops if op.name == "arith.cmpf"]
    assert len(cmps) == 1
    cmp = cmps[0]
    assert int(cmp.properties["predicate"].value.data) == 13
    assert not cmp.properties["fastmath"].data
    assert cmp.results[0].type.width.data == 1
    if source is not None:
        assert cmp.operands[0] is source
    zero = cmp.operands[1].owner
    assert zero.name == "arith.constant"
    assert zero.results[0].type == cmp.operands[0].type
    zero_value = zero.properties["value"].value.data
    assert zero_value == 0.0 and math.copysign(1.0, zero_value) == 1.0


def test_fptosi_mutations_are_valid_mlir_and_rejected():
    """Each defect is parsed and verified as real MLIR before the intent guard rejects it."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi

    retained = parse_mlir_text(_bool_cast_src())
    retained.verify()
    with pytest.raises(AssertionError):
        _assert_frontend_intent_shape(retained)

    from xdsl.dialects.builtin import IntegerAttr, i64

    for predicate in (8, 6):  # inverted equality, then ordered-ne (NaN becomes false)
        mutated = parse_mlir_text(_bool_cast_src())
        source = next(op for op in mutated.walk() if op.name == "arith.fptosi").operands[0]
        fix_bool_fptosi(mutated)
        cmp = next(op for op in mutated.walk() if op.name == "arith.cmpf")
        cmp.properties["predicate"] = IntegerAttr(predicate, i64)
        mutated.verify()
        with pytest.raises(AssertionError):
            _assert_frontend_intent_shape(mutated, source)

    trunc_src = _bool_cast_src().replace(
        "%c = arith.fptosi %xv : f32 to i1 ",
        "%wide = arith.fptosi %xv : f32 to i32 "
        "%c = arith.trunci %wide : i32 to i1 ",
    )
    truncated = parse_mlir_text(trunc_src)
    truncated.verify()
    with pytest.raises(AssertionError):
        _assert_frontend_intent_shape(truncated)


def test_actual_smolvla_capture_has_one_repaired_bool_boundary():
    from merlin.common.paths import merlin_dir
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi

    capture = merlin_dir() / "contract/capsules/model/SY_model_smolvla/capsule.interface.mlir"
    module = parse_mlir_text(capture.read_text(encoding="utf-8"))
    before = [
        op for op in module.walk()
        if op.name in ("arith.fptosi", "arith.fptoui")
        and getattr(op.results[0].type, "width", None) is not None
        and op.results[0].type.width.data == 1
    ]
    assert len(before) == 1
    original_operand = before[0].operands[0]
    assert fix_bool_fptosi(module) == 1
    module.verify()
    assert not any(op in module.walk() for op in before)
    replacements = [
        op for op in module.walk()
        if op.name == "arith.cmpf" and original_operand in op.operands
    ]
    assert len(replacements) == 1
    replacement = replacements[0]
    assert int(replacement.properties["predicate"].value.data) == 13
    assert replacement.operands[1].owner.name == "arith.constant"
    assert replacement.operands[1].owner.properties["value"].value.data == 0.0


def test_prepare_wires_the_fix_into_both_paths():
    """The compiled path and the numpy interpreter must apply the SAME bool-cast semantics.

    They diverge silently otherwise: ``int(x)`` in the interpreter, poison in the object, and the
    interpreter's golden then certifies a compiled model that no longer exists.
    """
    from merlin.common.paths import merlin_dir

    root = merlin_dir() / "python" / "merlin"
    for rel in ("runtime/backends/zephyr_model.py", "runtime/dispatch_runtime.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert (
            "fix_bool_fptosi(module)" in src
            or '_lean_stages.run("prepare-fix-bool-fptosi", fix_bool_fptosi, module)' in src
        ), rel


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
    assert list(out) == [0.0, 1.0, 1.0, 1.0]  # torch: (x != 0).float()


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_bool_cast_ieee_edges_match_native_and_torch(tmp_path):
    import json
    import os
    import subprocess

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.passes_xdsl import fix_bool_fptosi
    from merlin.xdsl_dialects._common import text as to_text

    values = np.array(
        [np.nan, np.inf, -np.inf, 0.0, -0.0, 0.25, -0.25, 1.0, -3.5, 2.0], dtype=np.float32
    )
    src = _bool_cast_src().replace("tensor<4x", "tensor<10x")
    module = parse_mlir_text(src)
    assert fix_bool_fptosi(module) == 1
    module.verify()
    result = lower_model(to_text(module), tmp_path / "ieee-edges", targets=("host",))
    model = HostModel.load(str(result.host_so))
    out = np.zeros(values.shape, np.float32)
    model([(values.ctypes.data, values.shape), (out.ctypes.data, out.shape)])
    # Independent transcription of A2's f32 bit-pattern intent, including NaNs and signed zero.
    expected = ((values.view(np.uint32) & np.uint32(0x7FFFFFFF)) != 0).astype(np.float32)
    np.testing.assert_array_equal(out, expected)
    torch_python = os.environ.get("MERLIN_TORCH_PYTHON")
    if torch_python:
        program = (
            "import json, sys, numpy as np, torch; "
            "bits = np.array(json.load(sys.stdin), dtype=np.uint32); "
            "print(json.dumps(torch.from_numpy(bits.view(np.float32)).to(torch.bool)"
            ".to(torch.float32).numpy().tolist()))"
        )
        oracle = subprocess.run(
            [torch_python, "-c", program], input=json.dumps(values.view(np.uint32).tolist()),
            text=True, capture_output=True, check=True, timeout=30,
        )
        np.testing.assert_array_equal(out, np.array(json.loads(oracle.stdout), np.float32))
        return
    try:
        import torch
    except ImportError:
        return
    torch_expected = torch.tensor(values).ne(0).to(torch.float32).numpy()
    np.testing.assert_array_equal(out, torch_expected)


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
