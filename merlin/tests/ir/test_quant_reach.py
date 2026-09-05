"""The int8 (W8A8) quant passes' REACH is a variable, and the default reach is unchanged.

Why this matters. Grading our int8 datapath against an independent W8A8 reference (torchao
``int8_dyn_act_int8_weight``, which rewrites ``nn.Linear`` and nothing else) only measures
ARITHMETIC when both sides quantize the same operations. Ours quantizes every recognized
contraction — attention ``bmm``s, spectral DFT matmuls, im2col conv matmuls — plus
softmax/GELU/SiLU/rsqrt, so the residual is error PLUS a wider policy, summed. ``select``
makes the policy half measurable by replaying the same model with a restricted reach.

Two properties are gated here because a comment cannot detect either failing:
  * the DEFAULT is byte-identical — ``select=None`` must not even be passed down, so a pass
    that never learned the kwarg keeps working and the shipped datapath cannot drift;
  * the restriction actually REACHES the ops — a ``select`` that admits nothing must lower
    nothing, and a provenance-based one must lower exactly the classified subset. A filter
    silently ignored would report "policy costs 0" for every model, which is precisely the
    wrong answer and looks like a clean result.
"""
from __future__ import annotations

import pytest

from merlin.common.artifacts import recaptures_dir
from merlin.llvmlower import quant_passes as QP

_REAL_FNS = ("lower_contraction_int8", "lower_conv_int8", "lower_softmax_int",
             "lower_gelu_int", "lower_silu_int", "lower_rsqrt_int")

#: the bundle the reach is measured on. Cheapest int8 capture that has an independent W8A8
#: reference; the test is about the PASS, not this model, and skips when it is absent.
_BUNDLE = "small_llama_int8_consistent"

#: what an ``nn.Linear`` decomposes to under torch.export, per the capture's own ``prov.aten``.
_LINEAR_ATEN = {"aten.mm.default", "aten.addmm.default", "aten.linear.default"}


def test_default_reach_passes_no_select_at_all(monkeypatch):
    from merlin.llvmlower import passes_quant_int as Q
    seen: list[tuple[str, tuple[str, ...]]] = []
    for fn in _REAL_FNS:
        monkeypatch.setattr(Q, fn, lambda _m, _f=fn, **kw: (seen.append((_f, tuple(kw))), 0)[1])
    QP.apply_quant(object())
    # no kwargs at all on the default path (named_contraction False, select None)
    assert seen == [(f, ()) for f in _REAL_FNS]


def test_select_reaches_every_pass(monkeypatch):
    from merlin.llvmlower import passes_quant_int as Q
    seen: list[tuple[str, object]] = []
    for fn in _REAL_FNS:
        monkeypatch.setattr(
            Q, fn, lambda _m, _f=fn, **kw: (seen.append((_f, kw.get("select"))), 0)[1])
    pred = lambda _op: True                       # noqa: E731 - identity predicate under test
    QP.apply_quant(object(), select=pred)
    assert seen == [(f, pred) for f in _REAL_FNS]


def test_run_model_forwards_the_restricted_reach(monkeypatch):
    """``run_model`` must hand its ``quant_passes``/``quant_select`` to ``apply_quant``.

    Checked by making ``apply_quant`` raise with what it received: the alternative (running the
    model) costs minutes and would not prove the forwarding any better.
    """
    from merlin.runtime import dispatch_runtime as DR

    bundle = recaptures_dir() / _BUNDLE
    if not (bundle / "model.mlir").is_file():
        pytest.skip(f"{_BUNDLE} recapture not present")

    class _Reached(Exception):
        def __init__(self, args): self.args_seen = args

    def _spy(module, passes=None, *, named_contraction=False, select=None):
        raise _Reached((passes, select))

    monkeypatch.setattr("merlin.llvmlower.quant_passes.apply_quant", _spy)
    pred = lambda _op: False                      # noqa: E731
    with pytest.raises(_Reached) as ei:
        DR.run_model(str(bundle), tmp_path_unused := "/nonexistent-workdir-never-created",
                     int8_compute=True, quant_passes=["contraction_int8"], quant_select=pred)
    assert ei.value.args_seen == (["contraction_int8"], pred)
    del tmp_path_unused


def _prepared_module(bundle):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.passes_xdsl import collapse_overrank_matmul
    from merlin.runtime.dispatch_runtime import _propagate_quant_inner
    module = parse_mlir_file(bundle / "model.mlir")
    collapse_overrank_matmul(module)
    _propagate_quant_inner(module)
    return module


def test_contraction_reach_is_restrictable_by_provenance():
    """On a real capture: no-op select lowers nothing; a provenance select lowers exactly the
    Linear-descended contractions, strictly fewer than the unrestricted pass."""
    from merlin.llvmlower.passes_quant_int import lower_contraction_int8

    bundle = recaptures_dir() / _BUNDLE
    if not (bundle / "model.mlir").is_file():
        pytest.skip(f"{_BUNDLE} recapture not present")

    def aten(op):
        a = op.attributes.get("prov.aten")
        d = getattr(a, "data", None)
        return d if isinstance(d, str) else ""

    assert lower_contraction_int8(_prepared_module(bundle), select=lambda _op: False) == 0
    n_linear = lower_contraction_int8(_prepared_module(bundle),
                                      select=lambda op: aten(op) in _LINEAR_ATEN)
    n_all = lower_contraction_int8(_prepared_module(bundle))
    assert 0 < n_linear < n_all
    # the non-Linear remainder is the attention QK^T / PV pair per layer -- the ops torch eager
    # (and therefore the independent reference) keeps in fp32.
    assert n_all - n_linear == lower_contraction_int8(
        _prepared_module(bundle), select=lambda op: aten(op) not in _LINEAR_ATEN)
