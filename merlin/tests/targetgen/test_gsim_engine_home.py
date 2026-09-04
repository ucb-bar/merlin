"""The GSIM engine must be SELECTED when it is there, REFUSED when its lineage does not match, and
loudly PASSED OVER when it is absent.

All three are the same defect seen from three sides. GSIM is ranked above Verilator on cost at equal
fidelity (:mod:`merlin.targetgen.rtl_engine_policy`), and it kept losing anyway — because it was
reachable only through an environment variable nobody had exported, because falling back said nothing,
and because a binary sitting in the right place proved nothing about which RTL it was elaborated from.
The policy was never wrong; what was missing was a home, a reason, and a gate. These tests pin each.

Everything here uses a MADE-UP target name and a redirected output root, so nothing depends on which
targets this checkout happens to have built.
"""
from __future__ import annotations

import json
import stat

import pytest

from merlin.targetgen import gsim_emulator as GE
from merlin.targetgen import rtl_engine_policy as POL


@pytest.fixture()
def out_root(monkeypatch, tmp_path):
    """Redirect the generated-output root so the derived engine home lands under the test's tmp dir."""
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    # The resolver reads overrides through `paths.env`, which caches `.env` — and this repo's `.env`
    # legitimately registers a real emulator for a real target. A made-up target name cannot collide
    # with it, which is exactly why these tests use one.
    return tmp_path


def _install_binary(target: str, *, executable: bool = True) -> "object":
    home = GE.gsim_home(target)
    home.mkdir(parents=True, exist_ok=True)
    emu = home / GE.BINARY_NAME
    emu.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    mode = emu.stat().st_mode
    emu.chmod(mode | stat.S_IXUSR if executable else mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    return emu


def _write_receipt(target: str, binary_sha256: str, **over) -> None:
    doc = {"schema_version": "merlin.gsim-model-build.v2", "status": "complete",
           "binary_sha256": binary_sha256,
           "artifacts": {"binary": {"sha256": binary_sha256},
                         "firrtl": {"sha256": "f" * 64, "path": "/nowhere/design.fir"},
                         "model_manifest": {"sha256": "m" * 64}},
           "tools": {"gsim_emitter": {"sha256": "e" * 64}}}
    doc.update(over)
    (GE.gsim_home(target) / GE.RECEIPT_NAME).write_text(json.dumps(doc), encoding="utf-8")


# --- present -> selected ------------------------------------------------------------------------

def test_a_gsim_build_in_the_derived_home_is_selected_over_verilator(out_root):
    """The whole point of a derived home: installing the binary IS registering the engine.

    No env var is set here. Before the home existed, this exact state — a built, working GSIM model on
    the machine — resolved to unavailable and every capsule certified on Verilator.
    """
    target = "fixture_np"
    _install_binary(target)
    ok, why = GE.probe(target)
    assert ok is True, why

    sel = POL.select(target, {"gsim": lambda: GE.probe(target),
                              "verilator": lambda: (True, "verilator is present")})
    assert sel["engine"] == "gsim"
    assert sel["fidelity"] == POL.ELABORATED_RTL


def test_an_env_override_still_wins_over_the_derived_home(out_root, tmp_path, monkeypatch):
    """A caller pointing at a freshly built model must not have to install it first."""
    target = "fixture_np"
    _install_binary(target)
    elsewhere = tmp_path / "fresh_build"
    elsewhere.mkdir()
    emu = elsewhere / "emu"
    emu.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    emu.chmod(emu.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("MERLIN_MY_OWN_SPELLING", str(emu))

    assert GE.emulator_path(target, env_var="MERLIN_MY_OWN_SPELLING") == emu
    ok, why = GE.probe(target, env_var="MERLIN_MY_OWN_SPELLING")
    assert ok is True
    assert "env:MERLIN_MY_OWN_SPELLING" in why


# --- absent -> verilator, WITH the reason ---------------------------------------------------------

def test_an_absent_gsim_build_falls_back_to_verilator_and_records_why(out_root):
    """Falling back is fine. Falling back SILENTLY is the defect.

    The recorded reason has to be actionable — it must name the place a build would go, because the
    person reading it is the person who would build one.
    """
    target = "fixture_np"                              # nothing installed
    ok, why = GE.probe(target)
    assert ok is False
    assert str(GE.gsim_home(target)) in why, why

    sel = POL.select(target, {"gsim": lambda: GE.probe(target),
                              "verilator": lambda: (True, "verilator sim present")})
    assert sel["engine"] == "verilator"
    assert sel["passed_over"] == ["gsim"]
    passed_over = [c for c in sel["considered"] if c["engine"] == "gsim"][0]
    assert passed_over["available"] is False
    assert str(GE.gsim_home(target)) in passed_over["reason"]
    # And the one-line render a report prints says it was chosen over something, not merely chosen.
    assert "over gsim" in POL.describe(sel)


def test_an_unexecutable_emulator_is_unavailable_not_a_crash(out_root):
    """An artifact copied without its mode bit, or the emitted .cpp rather than the built model, both
    exist. A cert tier reported available and then failing to exec is worse than one reported absent."""
    target = "fixture_np"
    _install_binary(target, executable=False)
    ok, why = GE.probe(target)
    assert ok is False
    assert "not executable" in why


# --- present but unattributable / mis-attributed --------------------------------------------------

def test_an_emulator_whose_receipt_binds_other_bytes_is_refused_not_used(out_root):
    """THE hazard: a result attributed to the wrong device.

    A receipt beside a binary reads, to anyone who opens it, as that binary's lineage — which FIRRTL,
    which emitter, which compiler. If it binds a different digest it says nothing about these bytes, and
    using it would file this run's numbers under another build's RTL revision. Refused, and refused
    LOUDLY: `refused` is a different state from absent, because bytes that exist and may not be trusted
    need different work from bytes that were never built.
    """
    target = "fixture_np"
    _install_binary(target)
    _write_receipt(target, binary_sha256="0" * 64)     # deliberately not this binary

    res = GE.resolve(target)
    assert res.ok is False
    assert res.refused is True
    assert res.receipt_status == "invalid"
    assert "REFUSED" in res.reason and "DIFFERENT binary" in res.reason

    # And the policy must then pick the slower engine rather than the refused one.
    sel = POL.select(target, {"gsim": lambda: GE.probe(target),
                              "verilator": lambda: (True, "verilator sim present")})
    assert sel["engine"] == "verilator"


def test_a_matching_receipt_binds_the_lineage_and_the_citation_carries_it(out_root):
    target = "fixture_np"
    emu = _install_binary(target)
    from merlin.common import provenance
    _write_receipt(target, binary_sha256=provenance.file_digest(emu))

    res = GE.resolve(target)
    assert res.ok is True
    assert res.receipt_status == "bound"
    assert res.receipt["firrtl_sha256"] == "f" * 64

    cite = GE.citation(target)
    assert cite["available"] is True and cite["refused"] is False
    assert cite["binary_sha256"] == provenance.file_digest(emu)
    assert cite["receipt"]["tools"]["gsim_emitter"] == "e" * 64


def test_an_unreceipted_emulator_says_so_and_can_be_made_fatal(out_root, monkeypatch):
    """Absent lineage is USABLE but never silent, and a run that publishes a verdict can make it fatal.

    Defaulting to fatal would break the developer who just built a model; defaulting to quiet is how an
    unattributed binary ends up cited. So it resolves, and the sentence travels with it.
    """
    target = "fixture_np"
    _install_binary(target)

    ok, why = GE.probe(target)
    assert ok is True
    assert "UNRECORDED" in why

    monkeypatch.setenv(GE.REQUIRE_RECEIPT_ENV, "1")
    res = GE.resolve(target)
    assert res.ok is False and res.refused is True


def test_installing_refuses_a_receipt_that_does_not_bind_the_binary(out_root, tmp_path):
    """The gate is at INSTALL too, not only at resolve: an engine home whose receipt describes some
    other build is a trap laid for every later reader, and the moment to refuse it is before it lands."""
    target = "fixture_np"
    src = tmp_path / "emu_src"
    src.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    src.chmod(src.stat().st_mode | stat.S_IXUSR)
    bad = tmp_path / "receipt.json"
    bad.write_text(json.dumps({"schema_version": "merlin.gsim-model-build.v2", "status": "complete",
                               "binary_sha256": "0" * 64}), encoding="utf-8")

    with pytest.raises(ValueError):
        GE.install(target, src, receipt=bad)
    # Nothing half-installed: a home holding a binary with no receipt would read as merely unattributed.
    assert not (GE.gsim_home(target) / GE.BINARY_NAME).exists()


# --- the home is per (target, engine), and derived from the target --------------------------------

def test_the_engine_home_is_derived_per_target_and_per_engine(out_root):
    a, b = GE.gsim_home("fixture_np"), GE.gsim_home("fixture_simt")
    assert a != b
    assert a.name == "gsim" and a.parent.name == "fixture_np"
    assert GE.engine_home("fixture_np", "vcs") != a
    # The per-target env spelling is DERIVED, so a new target has an override without a code edit.
    assert GE.derived_env_var("fixture-np") == "MERLIN_GSIM_EMU_FIXTURE_NP"


# --- the WRAPPER flavour ---------------------------------------------------------------------------
#
# `engine_home` has always documented two legitimate shapes under one home, and `record_adoption` exists
# to install the directory-shaped one — but `resolve` looked only for the emulator BINARY, so a target
# whose engine ships an `<engine>_run.py` wrapper probed ABSENT however completely it was built. Measured
# on this host: a target had a GSIM engine in its derived home, cycle-exact against Verilator on 17/17
# programs at 32x the speed, and `gsim_emulator.probe` answered False for it. A probe that cannot see a
# working engine is the same defect as an env var nobody exported.

def _install_wrapper(target: str, *, engine: str = "gsim") -> "object":
    home = GE.engine_home(target, engine)
    home.mkdir(parents=True, exist_ok=True)
    wrapper = home / GE.wrapper_name(engine)
    wrapper.write_text("def run_program(words, **kw):\n    return {}\n", encoding="utf-8")
    return wrapper


def _adopt(target: str, *, engine: str = "gsim", cover: bool = True) -> None:
    """Write the adoption record `record_adoption` writes — optionally with a digest that does NOT
    cover the wrapper, which is what a stale record from an earlier install looks like."""
    home = GE.engine_home(target, engine)
    wrapper = home / GE.wrapper_name(engine)
    from merlin.common import provenance
    digest = provenance.file_digest(wrapper) if cover else "0" * 64
    (home / GE.ADOPTION_NAME).write_text(
        json.dumps({"schema_version": "merlin.gsim-emulator-adoption.v1", "target": target,
                    "files": {GE.wrapper_name(engine): {"sha256": digest}}}),
        encoding="utf-8")


def test_a_wrapper_only_home_is_a_built_engine_and_is_selected(out_root):
    """The atlas shape: no `emulator` binary, an `<engine>_run.py` beside its own sim. It is an engine."""
    target = "fixture_wrap"
    _install_wrapper(target)
    _adopt(target)

    res = GE.resolve(target)
    assert res.ok is True, res.reason
    assert res.flavour == "wrapper"
    assert res.path.name == GE.wrapper_name("gsim")

    sel = POL.select(target, {"gsim": lambda: GE.probe(target),
                              "verilator": lambda: (True, "verilator is present")})
    assert sel["engine"] == "gsim"


def test_a_wrapper_whose_adoption_record_misses_its_bytes_says_provenance_is_unrecorded(out_root):
    """A record listing a DIFFERENT digest describes an earlier install. Calling that "provenance
    recorded" would attribute these bytes to bytes that are gone."""
    target = "fixture_wrap_stale"
    _install_wrapper(target)
    _adopt(target, cover=False)

    res = GE.resolve(target)
    assert res.ok is True, res.reason          # loud, not fatal — same policy as an unreceipted binary
    assert res.receipt_status == "absent"
    assert "UNRECORDED" in res.reason


def test_an_adopted_wrapper_is_refused_when_a_bound_receipt_is_required(out_root, monkeypatch):
    """An ADOPTED lineage is weaker than a BUILT-and-bound one, and a run that publishes a verdict must
    be able to insist on the stronger claim rather than have the two quietly equated."""
    target = "fixture_wrap_strict"
    _install_wrapper(target)
    _adopt(target)
    assert GE.resolve(target).receipt_status == "adopted"

    monkeypatch.setenv(GE.REQUIRE_RECEIPT_ENV, "1")
    res = GE.resolve(target)
    assert res.ok is False
    assert res.refused is True
    assert GE.REQUIRE_RECEIPT_ENV in res.reason


def test_a_binary_still_wins_when_a_home_holds_both_shapes(out_root):
    """Ambiguity resolved in one direction, on purpose: the binary is the self-contained model."""
    target = "fixture_both"
    _install_binary(target)
    _install_wrapper(target)
    assert GE.resolve(target).flavour == "binary"


def test_an_empty_home_names_both_shapes_so_the_fix_is_actionable(out_root):
    """"Absent" must say what would make it present — both installable shapes, not only one."""
    target = "fixture_empty"
    GE.gsim_home(target).mkdir(parents=True, exist_ok=True)
    res = GE.resolve(target)
    assert res.ok is False
    assert GE.BINARY_NAME in res.reason and GE.wrapper_name("gsim") in res.reason
