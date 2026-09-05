"""torchao's affine activation quant decomposes to linalg, bit-exactly.

The capture leaves ``torchao.choose_qparams_affine`` / ``torchao.quantize_affine`` as opaque calls
to externs nothing defines — a link failure on the compiled path and an ``OutlineError`` on the
interpreter path. :mod:`merlin.llvmlower.torchao_affine` decomposes them.

BIT-EXACT IS THE BAR, and the reference is torchao's OWN result, not a second transcription of the
same reading: ``merlin/tests/data/torchao_affine/reference.npz`` was produced by calling
``torchao.quantization.quant_primitives`` in the capture venv (regenerate with
``build_tools/scripts/make_torchao_affine_reference.py``; see its ``PROVENANCE.json``). An integer
op that is merely close is a model that gates at cos 0.99 and is quietly wrong.

There is a SECOND reference, and the generator asserts it agrees: handed the same
``(scale, zero_point)``, ExecuTorch's ``quantized_decomposed.quantize_per_token`` produces the
identical int8 on every case here. Their ``choose_qparams`` do NOT agree, and that is a property of
the two libraries rather than of this lowering — ``quantized_decomposed`` implements the ASYMMETRIC
XNNPACK-qd8 scheme (-128/127, ``(max_pos-min_neg)/255``, derived zero point) while our bundles carry
torchao's SYMMETRIC one. PROVENANCE.json records the measured divergence.

The cases are chosen so each arithmetic decision is OBSERVABLE — a suite that cannot fail is worse
than none. Verified by mutation against the same reference:

  * ``ties``  — values landing exactly on ``.5``; round-half-to-away instead of ``math.roundeven``
    moves 8 of 32 elements;
  * ``ulp``   — an element where ``round(x * (1/s)) != round(x / s)``; emitting the division
    instead of the reciprocal-then-multiply moves 1 element;
  * ``tiny`` / ``zeros`` — scale below the eps floor; dropping the ``clamp(min=eps)`` changes it;
  * every case  — dividing by ``(qmax-qmin)/2`` for the DTYPE range (127.5) instead of the range
    the scheme actually passes (127.0) changes every scale.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower.torchao_affine import (ACTIVATION_QUANT, TorchAOAffineError,
                                             derive_block_layout, lower_torchao_affine_quant)

_REF = merlin_dir() / "tests" / "data" / "torchao_affine" / "reference.npz"
_SCHEME = "int8_dyn_act_int8_weight"


def _module_text(in_shape, *, scheme: str | None = _SCHEME, scale_shape=None):
    """A module shaped exactly like a capture's: two opaque calls, no attributes but provenance."""
    scale_shape = tuple(in_shape[:-1]) if scale_shape is None else tuple(scale_shape)
    dims = lambda s: "x".join(str(d) for d in s)  # noqa: E731
    it, st = f"tensor<{dims(in_shape)}xf32>", f"tensor<{dims(scale_shape)}xf32>"
    qt = f"tensor<{dims(in_shape)}xi8>"
    attrs = f' attributes {{prov.quantization = "{scheme}"}}' if scheme else ""
    return f'''builtin.module{attrs} {{
  func.func private @torchao_choose_qparams_affine_default({it}) -> {st}
  func.func private @torchao_quantize_affine_default({it}, {st}) -> {qt}
  func.func @forward(%x: {it}) -> ({st}, {qt}) {{
    %s = func.call @torchao_choose_qparams_affine_default(%x) {{prov.region_id = "cq_0", prov.dispatch_id = "cq_0"}} : ({it}) -> {st}
    %q = func.call @torchao_quantize_affine_default(%x, %s) {{prov.region_id = "qa_0", prov.dispatch_id = "qa_0"}} : ({it}, {st}) -> {qt}
    func.return %s, %q : {st}, {qt}
  }}
}}'''


def _parse(text):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    return parse_mlir_text(text)


# --- derivation -------------------------------------------------------------------------------

def test_block_size_is_derived_from_the_types_not_assumed():
    """The capture drops torchao's ``block_size``; it is recovered from input vs scale shapes."""
    assert derive_block_layout((1, 2048), (1,)).block_size == (1, 2048)          # resnet50 fc
    assert derive_block_layout((1, 345, 32), (1, 345)).block_size == (1, 1, 32)  # lstmnetvit
    assert derive_block_layout((1, 345, 32), (1, 345)).granularity == "per_token"
    # keepdim=True form: ranks match, so each axis carries its own block count.
    assert derive_block_layout((4, 256), (4, 2)).block_size == (1, 128)


def test_an_underivable_block_layout_fails_closed():
    with pytest.raises(TorchAOAffineError):
        derive_block_layout((2, 3, 32), (7,))       # scale is not the leading axes
    with pytest.raises(TorchAOAffineError):
        derive_block_layout((1, 30), (1, 4))        # 30 is not divisible by 4


def test_an_unknown_scheme_is_refused_rather_than_defaulted():
    """quant_min/quant_max/eps come from the scheme, and no scheme means no defensible default."""
    with pytest.raises(TorchAOAffineError, match="ACTIVATION_QUANT"):
        lower_torchao_affine_quant(_parse(_module_text((1, 32), scheme="int4_weight_only")))
    with pytest.raises(TorchAOAffineError, match="prov.quantization"):
        lower_torchao_affine_quant(_parse(_module_text((1, 32), scheme=None)))


def test_a_module_without_these_calls_is_untouched():
    """The frozen-baseline invariant: adding this pass cannot perturb a bundle that never had them."""
    from merlin.xdsl_dialects._common import text as to_text
    plain = '''builtin.module {
  func.func @forward(%x: tensor<4xf32>) -> tensor<4xf32> {
    func.return %x : tensor<4xf32>
  }
}'''
    module = _parse(plain)
    before = to_text(module)
    assert lower_torchao_affine_quant(module) == 0
    assert to_text(module) == before


def test_the_opaque_externs_are_removed_with_their_calls():
    """A body-less func left behind is compiled as a kernel and fails far from its cause."""
    module = _parse(_module_text((1, 32)))
    assert lower_torchao_affine_quant(module) == 2
    names = {op.sym_name.data for op in module.walk() if op.name == "func.func"}
    assert names == {"forward"}
    assert not [op for op in module.walk() if op.name == "func.call"]


def test_the_scheme_entry_records_where_it_came_from():
    spec = ACTIVATION_QUANT[_SCHEME]
    assert (spec.mapping, spec.quant_min, spec.quant_max, spec.eps) == ("SYMMETRIC", -127, 127, 1e-5)
    assert "quant_api" in spec.source


# --- bit-exactness against torchao itself -------------------------------------------------------

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
    return np.asarray(res[0]), np.asarray(res[1])


@pytest.mark.parametrize("case", ["ties", "ulp", "tiny", "zeros", "r3"])
def test_bit_exact_against_torchao(case):
    if not _REF.is_file():
        pytest.skip(f"torchao reference fixture absent: {_REF}")
    ref = np.load(_REF)
    x, want_s, want_q = ref[f"{case}::x"], ref[f"{case}::scale"], ref[f"{case}::q"]
    try:
        got_s, got_q = _run_case(x)
    except Exception as exc:                                  # noqa: BLE001
        pytest.skip(f"host kernel toolchain unavailable: {type(exc).__name__}: {exc}")
    got_s = got_s.reshape(want_s.shape)
    got_q = got_q.reshape(want_q.shape)
    # f32 scales compared on their BITS: an equal-looking scale that differs in the last ulp
    # changes the integers, which is the whole failure mode this guards.
    assert (got_s.view(np.uint32) == want_s.view(np.uint32)).all(), (
        f"{case}: scale differs from torchao's own (max |delta| "
        f"{np.abs(got_s.astype(np.float64) - want_s.astype(np.float64)).max()})")
    assert (got_q.astype(np.int64) == want_q.astype(np.int64)).all(), (
        f"{case}: {int((got_q != want_q).sum())} of {want_q.size} quantized elements differ")
