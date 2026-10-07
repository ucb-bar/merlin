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


def test_body_if_present_reads_an_unconfigured_checkout_as_no_facts(monkeypatch):
    """A target whose facts could only be extracted from a checkout this host lacks has none here.

    The lookup's error used to escape from inside the extractor, crashing the readers documented to
    report "unavailable" -- only on a machine without the checkout. Direct and wrapped forms both count:
    a target that DECLARES its checkout wraps the lookup with ``raise ... from``.
    """
    from merlin.common.paths import ExternalPathUnset, ext_path
    from merlin.targetgen.rtl.introspect import RtlSourceInvalid

    monkeypatch.delenv("MERLIN_EXT_A_CHECKOUT_NO_HOST_HAS", raising=False)

    def unset(target, **kw):
        ext_path("a_checkout_no_host_has")

    monkeypatch.setattr(F, "load_facts", unset)
    assert F.body_if_present("t") == {}

    def declared_but_unset(target, **kw):
        try:
            ext_path("a_checkout_no_host_has")
        except KeyError as exc:
            raise RtlSourceInvalid("t: selected RTL declaration cannot resolve its checkout") from exc

    monkeypatch.setattr(F, "load_facts", declared_but_unset)
    assert F.body_if_present("t") == {}
    assert issubclass(ExternalPathUnset, KeyError), "existing `except KeyError` handlers must still apply"


def test_body_if_present_still_raises_what_is_not_absence(monkeypatch):
    """Only absence is absorbed: a defect's KeyError, or one raised merely while handling an absence,
    is not a missing checkout and must stay loud."""
    import pytest

    from merlin.common.paths import ext_path

    def defect(target, **kw):
        raise KeyError("a missing dict key inside the extractor")

    monkeypatch.setattr(F, "load_facts", defect)
    with pytest.raises(KeyError, match="missing dict key"):
        F.body_if_present("t")

    def defect_during_absence(target, **kw):
        try:
            ext_path("a_checkout_no_host_has")
        except KeyError:
            raise RuntimeError("a bug in the fallback path")  # noqa: B904 - implicit context on purpose

    monkeypatch.delenv("MERLIN_EXT_A_CHECKOUT_NO_HOST_HAS", raising=False)
    monkeypatch.setattr(F, "load_facts", defect_during_absence)
    with pytest.raises(RuntimeError, match="bug in the fallback"):
        F.body_if_present("t")
