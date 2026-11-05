"""The full-model gate never demands an accelerator epilogue that is not proved to round as the source."""

from __future__ import annotations

from merlin_experiments.phase1.feedback import private_full_models as PFM

from merlin.xdsl_dialects.lowering import group_exact_float as GEF


def test_the_gate_forms_its_groups_again_without_an_unproved_float_epilogue(monkeypatch, tmp_path):
    calls = []

    def form(module, target, **kwargs):
        calls.append(kwargs)
        return ["regrouped"] if kwargs else ["first"]

    monkeypatch.setattr(GEF, "refusals", lambda groups, capture: {1: "unproved"} if groups == ["first"] else {})
    assert PFM._exact_source_groups(None, "synthetic", tmp_path, form) == (["regrouped"], 1)
    assert calls == [{}, {"numeric_refusals": {1: "unproved"}}]
    monkeypatch.setattr(GEF, "refusals", lambda groups, capture: {})
    assert PFM._exact_source_groups(None, "synthetic", tmp_path, form) == (["first"], 0)
