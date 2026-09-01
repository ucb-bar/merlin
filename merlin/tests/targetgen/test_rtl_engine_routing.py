"""The cert adapter must ROUTE to the selected engine and say which one answered.

The tier records a fidelity; which simulator produced it is an availability choice, and one that has to
survive onto the result. A tier that silently resolved to a different engine than the capsule asked for
produces correct-looking numbers that cannot be audited afterwards.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import program_oracle as PO


def test_every_registered_engine_declares_a_wrapper_filename():
    for engine, (suffix, fname) in PO._RTL_ENGINES.items():
        assert suffix and fname.endswith(".py"), engine


def test_the_engine_dir_is_derived_from_the_target_never_hardcoded(monkeypatch):
    seen = []
    monkeypatch.setattr(PO, "ext_path", lambda key: seen.append(key) or "/nope")
    for engine in PO._RTL_ENGINES:
        PO._rtl_engine_dir("someTarget", engine)
    assert seen == [f"someTarget_{PO._RTL_ENGINES[e][0]}" for e in PO._RTL_ENGINES]


def test_an_unregistered_engine_probes_unavailable_with_a_reason(monkeypatch):
    monkeypatch.setattr(PO, "ext_path", lambda key: (_ for _ in ()).throw(KeyError(key)))
    ok, why = PO._rtl_engine_probe("t", "gsim")()
    assert ok is False and "GSIM" in why


def test_selection_prefers_gsim_over_verilator_when_both_are_built(monkeypatch, tmp_path):
    for engine, (_suffix, fname) in PO._RTL_ENGINES.items():
        if engine == "vcs":
            continue
        d = tmp_path / engine
        d.mkdir()
        (d / fname).write_text("def run_program(*a, **k): pass\n")

    def fake_ext(key):
        for engine, (suffix, _f) in PO._RTL_ENGINES.items():
            if key.endswith(f"_{suffix}"):
                if engine == "vcs":
                    raise KeyError(key)
                return tmp_path / engine
        raise KeyError(key)

    monkeypatch.setattr(PO, "ext_path", fake_ext)
    sel = PO.select_rtl_engine("t")
    assert sel["engine"] == "gsim"
    assert "vcs" in sel["passed_over"]


def test_no_engine_makes_the_tier_absent_not_a_lesser_oracle(monkeypatch):
    monkeypatch.setattr(PO, "ext_path", lambda key: (_ for _ in ()).throw(KeyError(key)))
    with pytest.raises(PO.OracleUnavailable):
        PO.select_rtl_engine("t")
    # and the adapter reports NO tier rather than falling through to a model oracle
    assert PO.program_verilator_adapter("t", model_ext="x") is None
