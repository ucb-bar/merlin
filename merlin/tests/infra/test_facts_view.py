"""merlin.common.facts_view.interface and rtl.facts.body_if_present: one lookup for RTL-facts blocks."""
from __future__ import annotations

from merlin.common.facts_view import interface
from merlin.targetgen.rtl import facts as F


def test_interface_finds_the_named_block():
    body = {"interfaces": [{"name": "a", "x": 1}, {"name": "funct_decode_table", "legal_funct": [0]}]}
    assert interface(body, "funct_decode_table") == {"name": "funct_decode_table", "legal_funct": [0]}


def test_every_absent_shape_answers_none():
    assert interface({}, "x") is None
    assert interface(None, "x") is None
    assert interface({"interfaces": None}, "x") is None
    assert interface({"interfaces": [{"name": "y"}]}, "x") is None


def test_malformed_entries_are_skipped_not_crashed_on():
    assert interface({"interfaces": ["junk", 3, {"name": "x", "v": 2}]}, "x") == {"name": "x", "v": 2}


def test_body_if_present_is_the_lax_reader(monkeypatch):
    monkeypatch.setattr(F, "load_facts", lambda target, **kw: {"facts": {"arrays": [1]}})
    assert F.body_if_present("t") == {"arrays": [1]}
    monkeypatch.setattr(F, "load_facts", lambda target, **kw: {"inputs": {}})
    assert F.body_if_present("t") == {}
    monkeypatch.setattr(F, "load_facts", lambda target, **kw: None)
    assert F.body_if_present("t") == {}
