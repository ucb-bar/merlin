"""WS-C C3: the int8 quant-pass registry (the quantization region's registrable edit-point)."""
from __future__ import annotations

from merlin.llvmlower import quant_passes as QP

_REAL_FNS = ("lower_contraction_int8", "lower_conv_int8", "lower_softmax_int",
             "lower_gelu_int", "lower_silu_int", "lower_rsqrt_int")


def test_canonical_order_is_the_historical_sequence():
    assert QP.known() == ("contraction_int8", "conv_int8", "softmax_int",
                          "gelu_int", "silu_int", "rsqrt_int")


def test_registry_wires_the_real_passes_quant_int_functions():
    # apply_quant runs the REAL lower_*_int passes -> the int8 datapath is byte-identical
    from merlin.llvmlower import passes_quant_int as Q
    reg = QP.registry()
    assert set(reg) == set(QP.known())
    for name, fn in zip(QP.known(), _REAL_FNS):
        assert reg[name].fn is getattr(Q, fn), name


def test_apply_quant_default_runs_all_six_in_order(monkeypatch):
    from merlin.llvmlower import passes_quant_int as Q
    calls: list[str] = []
    for fn in _REAL_FNS:
        monkeypatch.setattr(Q, fn, lambda _m, _f=fn: (calls.append(_f), 0)[1])
    counts = QP.apply_quant(object())
    assert calls == list(_REAL_FNS)                 # all six, canonical order (byte-identical)
    assert set(counts) == set(QP.known())


def test_apply_quant_selective_subset_in_canonical_order(monkeypatch):
    from merlin.llvmlower import passes_quant_int as Q
    calls: list[str] = []
    for fn in _REAL_FNS:
        monkeypatch.setattr(Q, fn, lambda _m, _f=fn: (calls.append(_f), 0)[1])
    QP.apply_quant(object(), passes=["softmax_int", "contraction_int8"])
    # selected subset, still emitted in the canonical order (contraction before softmax)
    assert calls == ["lower_contraction_int8", "lower_softmax_int"]
