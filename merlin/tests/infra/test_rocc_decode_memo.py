"""Decoding the same emitted program twice: it must be free, identical, and never stale.

The decode is a pure function of the text and the ISA constants derived from the target's RTL, and a
grade calls it with the same text repeatedly. Measured on one 24-capsule grade: 13.5 s in the decoder,
of which 13.1 s was the xDSL parse of IR that same grade had just parsed in order to compile it.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.rocc import decode as RD

MODULE = """
module {
  llvm.func @k() {
    llvm.return
  }
}
"""


@pytest.fixture(autouse=True)
def clean():
    RD._DECODE_MEMO.clear()
    yield
    RD._DECODE_MEMO.clear()


#: A real derived constants dict, so the decoder itself behaves normally; the tests vary it rather
#: than inventing a thin one the decode cannot use.
try:
    _REAL_ISA = dict(RD.isa_constants("gemmini"))
except Exception:                                      # noqa: BLE001
    _REAL_ISA = None


@pytest.fixture
def stub(monkeypatch):
    """Serve real derived constants (so the decode works) while counting how often it really runs."""
    if _REAL_ISA is None:
        pytest.skip("no target with derivable ISA constants on this host")
    calls = []
    monkeypatch.setattr(RD, "isa_constants", lambda target: dict(_REAL_ISA))
    real_scan = RD._decode_by_text_scan
    real_mod = RD.decode_module

    def scan(text, *, target, source=None):
        calls.append("scan")
        return real_scan(text, target=target, source=source)

    def mod(module, *, target, source=None):
        calls.append("module")
        return real_mod(module, target=target, source=source)
    monkeypatch.setattr(RD, "_decode_by_text_scan", scan)
    monkeypatch.setattr(RD, "decode_module", mod)
    return calls


def test_the_same_text_is_decoded_once(stub):
    a = RD.decode_text(MODULE, source="first", target="t")
    b = RD.decode_text(MODULE, source="second", target="t")
    assert len(stub) == 1, f"decoded {len(stub)} times"
    assert a.get("instructions") == b.get("instructions")


def test_the_source_label_is_per_caller_not_cached(stub):
    """`source` names the FILE a caller is describing, not part of the decode. Sharing the work must
    not make two callers disagree about which file they were talking about."""
    a = RD.decode_text(MODULE, source="alpha.mlir", target="t")
    b = RD.decode_text(MODULE, source="beta.mlir", target="t")
    assert a["source"] == "alpha.mlir" and b["source"] == "beta.mlir"


def test_different_text_is_decoded_again(stub):
    """The paired direction: a memo that always hits is the defect, not the feature."""
    RD.decode_text(MODULE, target="t")
    RD.decode_text(MODULE + "\n// edited\n", target="t")
    assert len(stub) == 2


def test_different_isa_constants_are_a_different_key(stub, monkeypatch):
    """The same target re-elaborated from different RTL derives different constants. Keying on the
    target's NAME alone would decode one revision's trace against another's."""
    RD.decode_text(MODULE, target="t")
    monkeypatch.setattr(RD, "isa_constants",
                        lambda target: {**_REAL_ISA, "CUSTOM_OPCODE": 0x2B})
    RD.decode_text(MODULE, target="t")
    assert len(stub) == 2, "a change in the derived ISA constants did not invalidate the memo"


def test_a_different_target_is_a_different_key(stub):
    RD.decode_text(MODULE, target="t")
    RD.decode_text(MODULE, target="u")
    assert len(stub) == 2


def test_unestablished_constants_decode_afresh_every_time(stub, monkeypatch):
    """A memo that cannot be keyed must not be guessed at. The KEY is what fails here -- the decode
    itself still works, which is exactly the situation: facts good enough to decode against but not
    establishable as a cache key."""
    monkeypatch.setattr(RD, "_decode_key", lambda text, target: None)
    RD.decode_text(MODULE, target="t")
    RD.decode_text(MODULE, target="t")
    assert len(stub) == 2
    assert RD._DECODE_MEMO == {}, "an unkeyable decode must not be stored"


def test_a_key_cannot_be_formed_without_the_constants():
    """The guard inside _decode_key itself: unresolvable facts yield no key, not a partial one."""
    import unittest.mock as _m
    with _m.patch.object(RD, "isa_constants",
                         side_effect=RuntimeError("no facts")):
        assert RD._decode_key(MODULE, "t") is None


def test_the_memo_never_aliases_a_callers_dict(stub):
    """BOTH directions of the same hazard, pinned on identity rather than on a symptom.

    A memo that STORES the object it returned lets the first caller's edit reach every later one; a
    memo that HANDS OUT its stored object lets a later caller's edit reach the one after that. Testing
    only by mutating the first return misses the second case entirely, because the shallow copy on the
    way out happens to absorb it.
    """
    first = RD.decode_text(MODULE, target="t")
    key, = RD._DECODE_MEMO
    stored = RD._DECODE_MEMO[key]
    assert first is not stored, "the memo stored the very dict it handed back"
    second = RD.decode_text(MODULE, target="t")
    assert second is not stored, "the memo handed out its own stored dict"
    assert second is not first


def test_a_caller_cannot_corrupt_the_memo(stub):
    """The behavioural consequence of the above, in both directions."""
    a = RD.decode_text(MODULE, target="t")
    a["instructions"] = "clobbered-by-first"
    b = RD.decode_text(MODULE, target="t")
    assert b["instructions"] != "clobbered-by-first"
    b["instructions"] = "clobbered-by-second"
    c = RD.decode_text(MODULE, target="t")
    assert c["instructions"] != "clobbered-by-second"


def test_the_memo_stays_bounded(stub, monkeypatch):
    monkeypatch.setattr(RD, "_DECODE_MEMO_MAX", 4)
    for i in range(12):
        RD.decode_text(MODULE + f"\n// {i}\n", target="t")
    assert len(RD._DECODE_MEMO) <= 4
