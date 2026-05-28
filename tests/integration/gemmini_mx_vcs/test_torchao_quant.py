"""Unit-style tests for the mxGemmini torchao integration (D7).

These run inside the merlin-dev pytest harness but only exercise the
custom-dtype Python module — no chipyard, no VCS. They verify both
Stage 6.A (stock torchao MX dtypes via ``MXDynamicActivationMXWeightConfig``)
and Stage 6.B (custom ``MxGemminiE4M4Tensor`` / ``MxGemminiE2M2Tensor``).
"""

from __future__ import annotations

import importlib
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _has_real_torch() -> bool:
    """The merlin-dev venv ships a stripped torch namespace (executorch
    leftover) that lacks ``torch.__version__`` and the MX dtypes we need.
    Skip torchao tests cleanly on those installs."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return getattr(__import__("torch"), "__version__", None) is not None


pytestmark = pytest.mark.skipif(
    not _has_real_torch(),
    reason="merlin-dev venv has stripped torch (no __version__); run this "
    "test under third_party/Understanding-PI0/.venv which has torch 2.10 + torchao 0.16.",
)


def test_e4m4_roundtrip():
    """Stage 6.B: E4M4 quantize-dequantize sanity."""
    cd = importlib.import_module("models.gemmini_mx_quant.custom_dtype")
    import torch

    torch.manual_seed(0)
    x = torch.randn(4, 64) * 5.0
    q, s = cd.quantize_to_e4m4(x, block_size=16)
    assert q.dtype == torch.uint8
    assert q.shape == x.shape
    assert s.shape == (4, 64 // 16)
    deq = cd.MxGemminiE4M4Tensor.from_float(x, block_size=16).dequantize()
    assert deq.shape == x.shape
    # Element-level codes fit in 8 bits.
    assert int(q.max()) <= 0xFF


def test_e2m2_roundtrip():
    """Stage 6.B: E2M2 quantize-dequantize sanity."""
    cd = importlib.import_module("models.gemmini_mx_quant.custom_dtype")
    import torch

    torch.manual_seed(0)
    x = torch.randn(2, 32) * 3.0
    q, s = cd.quantize_to_e2m2(x, block_size=16)
    # E2M2 codes fit in a nibble (4 bits).
    assert int(q.max()) <= 0xF
    deq = cd.MxGemminiE2M2Tensor.from_float(x, block_size=16).dequantize()
    assert deq.shape == x.shape


def test_saturation_constants():
    """Stage 6.B saturation magnitudes must match MxRequantizer.scala."""
    cd = importlib.import_module("models.gemmini_mx_quant.custom_dtype")
    assert cd.E4M4_PMAX == 448.0
    assert cd.E2M2_PMAX == 6.0


def test_stock_mx_config_block_16():
    """Stage 6.A: ``make_mxgemmini_fp8_config`` returns block_size=16."""
    config_mod = importlib.import_module("models.gemmini_mx_quant.config")
    cfg = config_mod.make_mxgemmini_fp8_config()
    assert cfg.block_size == 16


def test_stock_mx_config_fp4():
    """Stage 6.A: ``make_mxgemmini_fp4_config`` returns block_size=16
    and an FP4-flavored elem dtype if available."""
    config_mod = importlib.import_module("models.gemmini_mx_quant.config")
    cfg = config_mod.make_mxgemmini_fp4_config()
    assert cfg.block_size == 16


def test_safe_quantize_linears_stage_6b():
    """End-to-end: build a tiny module, quantize via Stage 6.B."""
    qmod = importlib.import_module("models.gemmini_mx_quant.quantize")
    cd = importlib.import_module("models.gemmini_mx_quant.custom_dtype")
    from collections import OrderedDict

    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.fc(x)

    m = M()
    plan = OrderedDict([("fc", "mx")])
    results = qmod.safe_quantize_linears_(m, plan, format="fp8", stage="6B", verbose=False)
    assert all(r.ok for r in results)
    assert isinstance(m.fc.mxgemmini_weight, cd.MxGemminiE4M4Tensor)


def test_export_rewrite_stage_6b():
    """Verify the export path replaces our quantized linear with an
    eager linear that has a high-precision weight."""
    qmod = importlib.import_module("models.gemmini_mx_quant.quantize")
    emod = importlib.import_module("models.gemmini_mx_quant.export")
    from collections import OrderedDict

    import torch
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.fc(x)

    m = M()
    qmod.safe_quantize_linears_(m, OrderedDict([("fc", "mx")]), format="fp4", stage="6B", verbose=False)
    rewritten, records = emod.clone_and_rewrite_quantized_linears_for_export(m, verbose=False)
    assert any(r.replaced for r in records)
    assert isinstance(rewritten.fc, emod.ExportableMxGemminiLinear)
    # Smoke-forward
    x = torch.randn(2, 16)
    y = rewritten(x)
    assert y.shape == (2, 16)
