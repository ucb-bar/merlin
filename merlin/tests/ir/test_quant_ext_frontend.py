"""Tests for the opt-in quant-aware frontend (merlin.frontends.quant_ext).

Skipped when the model2MLIR checkout is not resolvable (MERLIN_M2M_DIR / MERLIN_MODEL2MLIR) — the
untyped default path still works without it.
"""
from __future__ import annotations

import pytest

from merlin.common import quant_formats as qf
from merlin.common.paths import artifacts_dir
from merlin.frontends import quant_ext as qe

pytestmark = pytest.mark.skipif(not qe.available(), reason="model2MLIR quant_ext dialect unavailable")


def _sample_int8_mlir():
    """First real int8 capture bundle carrying quant_ext.dequantize ops, if present."""
    root = artifacts_dir() / "recaptures"
    if not root.is_dir():
        return None
    for d in sorted(root.glob("*_int8_*")):
        m = d / "model.mlir"
        if m.is_file() and "quant_ext." in m.read_text(encoding="utf-8"):
            return m
    return None


def test_dialect_loads_torch_free():
    import sys

    dialect = qe.load_dialect()
    assert dialect is not None and dialect.name == "quant_ext"
    assert "torch" not in sys.modules, "importing quant_ext must not pull torch"


def test_default_context_still_leaves_quant_ops_unregistered():
    # The default frontend context must NOT register quant_ext (the int8 lowering depends on the
    # unregistered form). Only the opt-in context registers it.
    from merlin.frontends import linalg_mlir as fl

    # allow_unregistered synthesises an op class on demand, so discriminate on the *dialect*:
    # the default context has no registered quant_ext dialect; the opt-in context does.
    assert fl.make_context().get_optional_dialect("quant_ext") is None
    assert qe.make_quant_context().get_optional_dialect("quant_ext") is not None


def test_format_from_quant_types_maps_to_registry():
    from m2m.ir.quant.types import (
        MXQuantizedTensorType,
        NVFP4TensorType,
        PackedIntTensorType,
    )
    from xdsl.dialects.builtin import IntegerType

    assert qe.format_from_quant_type(NVFP4TensorType()).name == "nvfp4"
    assert qe.format_from_quant_type(MXQuantizedTensorType(6)).name == "mxfp6"
    assert qe.format_from_quant_type(MXQuantizedTensorType(4)).name == "mxfp4"
    # A packed 4-bit integer tensor resolves to an integer format of that width.
    f4 = qe.format_from_quant_type(PackedIntTensorType(4, 0, IntegerType(8)))
    assert f4 is not None and f4.element_bits == 4 and f4.kind == "int_affine"


@pytest.mark.skipif(_sample_int8_mlir() is None, reason="no int8 recapture bundle present")
def test_quantized_tensors_on_real_int8_bundle():
    module = qe.parse_quant_mlir(_sample_int8_mlir())
    tensors = qe.quantized_tensors(module)
    assert tensors, "expected quant_ext.dequantize ops in an int8 capture"
    t = tensors[0]
    assert t.op_name.startswith("quant_ext.dequantize")
    assert t.storage_dtype == "i8"
    assert t.granularity == "per_channel"
    assert t.fmt is not None and t.fmt.name == "int8"
