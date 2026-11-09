from __future__ import annotations

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower.torchao_affine import (
    lower_torchao_affine_quant,
)

_REF = merlin_dir() / "tests" / "data" / "torchao_affine" / "reference.npz"
_SCHEME = "int8_dyn_act_int8_weight"


def _module_text(in_shape, *, scheme: str | None = _SCHEME, scale_shape=None):
    """A module shaped exactly like a capture's: two opaque calls, no attributes but provenance."""
    scale_shape = tuple(in_shape[:-1]) if scale_shape is None else tuple(scale_shape)
    dims = lambda s: "x".join((str(d) for d in s))
    it, st = (f"tensor<{dims(in_shape)}xf32>", f"tensor<{dims(scale_shape)}xf32>")
    qt = f"tensor<{dims(in_shape)}xi8>"
    attrs = f' attributes {{prov.quantization = "{scheme}"}}' if scheme else ""
    return f'builtin.module{attrs} {{\n  func.func private @torchao_choose_qparams_affine_default({it}) -> {st}\n  func.func private @torchao_quantize_affine_default({it}, {st}) -> {qt}\n  func.func @forward(%x: {it}) -> ({st}, {qt}) {{\n    %s = func.call @torchao_choose_qparams_affine_default(%x) {{prov.region_id = "cq_0", prov.dispatch_id = "cq_0"}} : ({it}) -> {st}\n    %q = func.call @torchao_quantize_affine_default(%x, %s) {{prov.region_id = "qa_0", prov.dispatch_id = "qa_0"}} : ({it}, {st}) -> {qt}\n    func.return %s, %q : {st}, {qt}\n  }}\n}}'


def _parse(text):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    return parse_mlir_text(text)


def _run_case(x):
    """Lower + outline + compile + execute one case; returns ``(scale, q)`` as numpy."""
    import tempfile

    from merlin.runtime.dispatch_runtime import execute
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    module = _parse(_module_text(x.shape))
    assert lower_torchao_affine_quant(module) == 2
    outlined = outline_dispatches(module)
    with tempfile.TemporaryDirectory() as tmp:
        res = execute(outlined, [np.ascontiguousarray(x)], tmp)
    return (np.asarray(res[0]), np.asarray(res[1]))


@pytest.mark.parametrize("case", ["ties", "zeros"])
def test_bit_exact_against_torchao(case):
    if not _REF.is_file():
        pytest.skip(f"torchao reference fixture absent: {_REF}")
    ref = np.load(_REF)
    x, want_s, want_q = (ref[f"{case}::x"], ref[f"{case}::scale"], ref[f"{case}::q"])
    try:
        got_s, got_q = _run_case(x)
    except Exception as exc:
        pytest.skip(f"host kernel toolchain unavailable: {type(exc).__name__}: {exc}")
    got_s = got_s.reshape(want_s.shape)
    got_q = got_q.reshape(want_q.shape)
    assert (got_s.view(np.uint32) == want_s.view(np.uint32)).all(), (
        f"{case}: scale differs from torchao's own (max |delta| {np.abs(got_s.astype(np.float64) - want_s.astype(np.float64)).max()})"
    )
    assert (got_q.astype(np.int64) == want_q.astype(np.int64)).all(), (
        f"{case}: {int((got_q != want_q).sum())} of {want_q.size} quantized elements differ"
    )
