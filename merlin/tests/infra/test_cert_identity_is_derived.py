"""The certificate identity: which executables it covers, and why a refusal must be legible.

Two defects of the same shape. The identity took ONE executable, by the name the operator build path
happens to use — so a target whose grade links a differently-named one, or several, could never form an
identity. And every refusal was spelled ``None``, which a caller can only render as "nothing was
carried" — indistinguishable from a converged run with nothing worth carrying. Measured: two of three
targets had a certificate cache that never once fired, and nothing said so.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen import tier_cache as TC

SHAS = {"merlin": "d" * 40, "some_rtl": "a" * 40}


def _elf(d, name, body=b"body"):
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"\x7fELF" + body)
    return p


# --------------------------------------------------------------------------------------------
# which executables
# --------------------------------------------------------------------------------------------

def test_the_operator_name_is_the_fast_path(tmp_path):
    _elf(tmp_path, "package_kernel.elf")
    assert [p.name for p in CR.run_executables(tmp_path)] == ["package_kernel.elf"]


def test_differently_named_executables_are_found(tmp_path):
    """The paired direction: this is the case that could not form an identity at all."""
    _elf(tmp_path, "kernel.radiance.elf")
    _elf(tmp_path, "kernel.soc.elf")
    assert [p.name for p in CR.run_executables(tmp_path)] == ["kernel.radiance.elf", "kernel.soc.elf"]


def test_non_executables_are_not_mistaken_for_one(tmp_path):
    (tmp_path / "notes.txt").write_text("not an elf")
    (tmp_path / "kernel.ll").write_text("; ir")
    assert CR.run_executables(tmp_path) == ()


def test_an_empty_directory_yields_nothing(tmp_path):
    assert CR.run_executables(tmp_path) == ()
    assert CR.run_executables(tmp_path / "absent") == ()


def test_every_executable_is_in_the_identity(tmp_path):
    """Not one picked by name or mtime: keying a certificate on the wrong program is the direction
    this must never fail in, so a change to ANY of them must move the identity."""
    a, b = tmp_path / "one", tmp_path / "two"
    _elf(a, "kernel.x.elf", b"AAA"); _elf(a, "kernel.y.elf", b"BBB")
    _elf(b, "kernel.x.elf", b"AAA"); _elf(b, "kernel.y.elf", b"CHANGED")
    ia = TC.execution_identity(target="t", executables=CR.run_executables(a), toolchain_shas=SHAS)
    ib = TC.execution_identity(target="t", executables=CR.run_executables(b), toolchain_shas=SHAS)
    assert ia and ib and ia != ib, "a second executable's bytes did not reach the identity"


def test_the_identity_is_stable_for_the_same_bytes(tmp_path):
    a, b = tmp_path / "one", tmp_path / "two"
    for d in (a, b):
        _elf(d, "kernel.x.elf", b"AAA"); _elf(d, "kernel.y.elf", b"BBB")
    ia = TC.execution_identity(target="t", executables=CR.run_executables(a), toolchain_shas=SHAS)
    ib = TC.execution_identity(target="t", executables=CR.run_executables(b), toolchain_shas=SHAS)
    assert ia == ib


def test_a_renamed_executable_is_a_different_identity(tmp_path):
    """Names are part of what was produced, so a build that renames its output is not the same run."""
    a, b = tmp_path / "one", tmp_path / "two"
    _elf(a, "kernel.x.elf", b"AAA")
    _elf(b, "kernel.z.elf", b"AAA")
    assert (TC.execution_identity(target="t", executables=CR.run_executables(a), toolchain_shas=SHAS)
            != TC.execution_identity(target="t", executables=CR.run_executables(b), toolchain_shas=SHAS))


# --------------------------------------------------------------------------------------------
# why it refused
# --------------------------------------------------------------------------------------------

def _reason(**kw):
    base = {"target": "t", "executables": (), "toolchain_shas": SHAS}
    base.update(kw)
    return TC.execution_identity_reason(**base)


def test_a_hit_reports_no_reason(tmp_path):
    ex = (_elf(tmp_path, "package_kernel.elf"),)
    assert TC.execution_identity(target="t", executables=ex, toolchain_shas=SHAS)
    assert _reason(executables=ex) == "", "a formable identity must not also report a refusal"


@pytest.mark.parametrize("kw, must_mention", [
    ({"executables": ()}, "executable"),
    ({"toolchain_shas": {"merlin": "d" * 40}}, "hardware pin"),
    ({"toolchain_shas": {"some_rtl": "UNKNOWN"}}, "guess"),
    ({"target": ""}, "target"),
])
def test_every_refusal_says_which_one_it_was(tmp_path, kw, must_mention):
    ex = kw.pop("executables", (_elf(tmp_path, "package_kernel.elf"),))
    assert TC.execution_identity(target=kw.get("target", "t"), executables=ex,
                                 toolchain_shas=kw.get("toolchain_shas", SHAS)) is None
    why = _reason(executables=ex, **kw)
    assert why and must_mention in why, f"refusal did not explain itself: {why!r}"


def test_the_reason_cannot_drift_from_the_decision(tmp_path):
    """Both forms run the same computation, so a refusal is never explained as a success or vice versa."""
    cases = [((), SHAS, "t"), ((_elf(tmp_path, "package_kernel.elf"),), SHAS, "t"),
             ((_elf(tmp_path, "package_kernel.elf"),), {"merlin": "d" * 40}, "t")]
    for ex, shas, target in cases:
        got = TC.execution_identity(target=target, executables=ex, toolchain_shas=shas)
        why = TC.execution_identity_reason(target=target, executables=ex, toolchain_shas=shas)
        assert bool(got) == (why == ""), f"identity and reason disagree for {shas}/{len(ex)}"


def test_a_refusal_reaches_the_capsule_result():
    """The whole point: unrecorded, "could not ask" and "asked and missed" read identically."""
    unavailable = CR.TierResult("L3", "pass", True, cache_unavailable="no pin resolved")
    plain = CR.TierResult("L2", "pass", True)
    assert unavailable.to_dict()["cache_unavailable"] == "no pin resolved"
    assert "cache_unavailable" not in plain.to_dict(), "a normal record must stay byte-identical"
    block = TC.reuse_block({"L2": plain.to_dict(), "L3": unavailable.to_dict()})
    assert block["unavailable"] == {"L3": "no pin resolved"}
    assert TC.reuse_block({"L2": plain.to_dict()}).get("unavailable") is None


def test_a_miss_is_not_reported_as_a_refusal(tmp_path, monkeypatch):
    """Asking and missing is ordinary. Only a question that could not be ASKED is a refusal."""
    monkeypatch.setattr(TC, "lookup", lambda *a, **k: None)
    monkeypatch.setattr(TC, "instrument_digest", lambda *a, **k: "instr")
    _elf(tmp_path, "package_kernel.elf")
    got, why = CR.carried_tier_result("C0", "L3", True, target="t", generated=tmp_path,
                                      shas=SHAS, from_rtl=True)
    assert got is None and why == ""


def test_an_unaskable_question_is_reported(tmp_path):
    _elf(tmp_path, "package_kernel.elf")
    got, why = CR.carried_tier_result("C0", "L3", True, target="t", generated=tmp_path,
                                      shas={"merlin": "d" * 40}, from_rtl=True)
    assert got is None and "hardware pin" in why


# --------------------------------------------------------------------------------------------
# stamping the refusal onto the record
# --------------------------------------------------------------------------------------------

def test_a_refusal_is_stamped_onto_its_tier_record():
    tiers = {"L2": CR.TierResult("L2", "pass", True), "L3": CR.TierResult("L3", "pass", True)}
    CR.stamp_cache_refusals(tiers, {"L3": "no pin resolved"})
    assert tiers["L3"].cache_unavailable == "no pin resolved"
    assert tiers["L2"].cache_unavailable == "", "an untouched tier must stay untouched"


def test_stamping_does_not_overwrite_a_reason_already_there():
    tiers = {"L3": CR.TierResult("L3", "pass", True, cache_unavailable="the first reason")}
    CR.stamp_cache_refusals(tiers, {"L3": "a later reason"})
    assert tiers["L3"].cache_unavailable == "the first reason"


def test_stamping_tolerates_a_tier_that_produced_no_record():
    """A refusal for a tier the ladder never reached must not raise -- a cache accounting failure
    may not gate a grade."""
    CR.stamp_cache_refusals({}, {"L3": "why"})
    CR.stamp_cache_refusals({"L2": CR.TierResult("L2", "pass", True)}, {"L3": "why"})
    CR.stamp_cache_refusals(None, None)


def test_an_empty_reason_is_not_stamped():
    """A miss reports an empty reason, and an empty `cache_unavailable` must stay absent from the
    record -- otherwise every ordinary miss would render as an unavailable cache."""
    tiers = {"L3": CR.TierResult("L3", "pass", True)}
    CR.stamp_cache_refusals(tiers, {"L3": ""})
    assert tiers["L3"].cache_unavailable == ""
    assert "cache_unavailable" not in tiers["L3"].to_dict()
