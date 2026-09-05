"""A conv2d capsule must have a golden on a FLOAT datapath, not only an integer one.

The corpus's convolution perf family (the one that varies input-channel depth at a fixed window) is
declared in the shared performance template, so every target whose gate admits it gets those capsules
-- including targets whose operands are a float format. The integer engine has always had a conv2d
branch; the float engine did not, so those capsules raised ``no float golden for op 'conv2d'`` and the
family was reported as a generation ERROR on exactly the targets that admit it.

What is asserted here is the property that makes the float branch trustworthy rather than merely
present: an im2col conv is a contraction over gathered windows, so at a 1x1 window with unit stride and
no padding it must be BYTE-IDENTICAL to the matmul golden on the same operands. A branch that gathered
differently from the runtime, or reduced differently from the matmul branch, fails that.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import corpus_spec as CS

sys.path.insert(0, str(repo_root() / "merlin/contract/capsules"))
import generate_corpus as GC  # noqa: E402

TILE = 16


def _float_binding(dtype: str = "fp8_e4m3") -> CS.CorpusBinding:
    """A float-regime binding, target-agnostic: the engine under test reads formats and extents, never
    a target name."""
    return CS.CorpusBinding(target="t", tile_dim=TILE, operand_dtype=dtype, accum_dtype="bf16",
                            integer=False, tiers=["L3"], compare="tolerance_float",
                            atol=0.25, rtol=0.02)


def _specir_or_skip():
    try:
        GC._specir()
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"specir refmodel unavailable: {exc}")


def _conv_entry(**over) -> dict:
    entry = {"name": "PVtest_c16", "kind": "model_slice", "cat": "_perf",
             "source_role": "derived_sweep", "source_reference": "conv golden test",
             "op": "conv2d", "out": "Y0", "ifm": "IFM", "weight": "W",
             "Himg": 8, "Wimg": 8, "kh": 3, "kw": 3, "ci": TILE, "N": TILE}
    entry.update(over)
    return entry


def test_float_engine_materializes_a_conv2d_golden():
    """The gap itself: a conv2d entry on a float binding produces a golden at the conv's own extent."""
    _specir_or_skip()
    outputs, prov = GC._float_golden(_conv_entry(), _float_binding())
    y = outputs["Y0"]
    assert len(y) == 6 * 6 and len(y[0]) == TILE          # 8x8 image, 3x3 window, unit stride, no pad
    flat = [v for row in y for v in row]
    assert len(set(flat)) > 1, "a constant golden hides every addressing bug the capsule exists to find"
    # Both leaves are recorded, the activation at the RANK-4 shape the capsule declares for it.
    assert prov["IFM"]["shape"] == [1, 8, 8, TILE]
    assert prov["W"]["shape"] == [3 * 3 * TILE, TILE]
    assert len(prov["IFM"]["decoded"]) == 8 * 8 * TILE


def test_pointwise_conv_is_the_matmul_golden_on_the_same_operands():
    """A 1x1/stride-1/no-pad conv IS the matmul [H*W, Ci] @ [Ci, Co]. The two engines must agree exactly:
    the conv branch reduces through the same `mm`, over the runtime's own gather, on operands built at the
    same 2-D shape -- so any divergence is a real disagreement about the window or the reduction."""
    _specir_or_skip()
    binding = _float_binding()
    conv = _conv_entry(kh=1, kw=1, ci=TILE, N=TILE, Himg=4, Wimg=4,
                       stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1])
    matmul = {**_conv_entry(), "op": "matmul", "lhs": "IFM", "weight": "W",
              "M": 4 * 4, "K": TILE, "N": TILE}
    conv_out, _ = GC._float_golden(conv, binding)
    mm_out, _ = GC._float_golden(matmul, binding)
    assert conv_out["Y0"] == mm_out["Y0"]


def test_padding_changes_the_extent_and_the_values():
    """Zero-padding is applied, not ignored: it widens the output extent and the interior rows it does not
    touch stay put. A branch that dropped the pad would produce the unpadded extent."""
    _specir_or_skip()
    binding = _float_binding()
    unpadded, _ = GC._float_golden(_conv_entry(), binding)
    padded, _ = GC._float_golden(_conv_entry(padding=[1, 1, 1, 1]), binding)
    assert len(unpadded["Y0"]) == 6 * 6
    assert len(padded["Y0"]) == 8 * 8
    assert padded["Y0"] != unpadded["Y0"]


def test_declared_conv_perf_family_generates_on_a_float_binding():
    """The shipped family, not a hand-made entry: every channel depth the shared template's conv sweep
    declares must materialize a golden on a float binding."""
    _specir_or_skip()
    import yaml
    shared = yaml.safe_load(
        (repo_root() / "merlin/contract/capsules/profiles/_perf.yaml").read_text(encoding="utf-8"))
    convs = [s for s in (shared.get("sweeps") or [])
             if (s.get("base") or {}).get("op") == "conv2d"]
    assert convs, "the shared performance template declares no conv2d family"
    binding = _float_binding()
    for sweep in convs:
        base = sweep["base"]
        (axis, points), = ((k, v) for k, v in (sweep.get("axes") or {}).items())
        for point in points:
            entry = {**{k: v for k, v in base.items() if k != "performance"},
                     "name": f"{sweep['id']}_{axis}{point}", "source_reference": "template",
                     axis: GC.resolve_extent(point, binding.tile_dim)}
            outputs, _ = GC._float_golden(entry, binding)
            assert outputs and next(iter(outputs.values()))
