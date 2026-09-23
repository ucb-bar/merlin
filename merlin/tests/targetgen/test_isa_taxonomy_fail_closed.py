"""``isa_taxonomy`` must never report "this target has no ISA" when it simply could not look.

Both states used to be the same value — a bare ``{}`` returned from behind two bare ``except``
clauses — and roughly ten consumers (coverage_report, conformance, operation_capabilities,
readout_facet, capability_manifests, rtl_check_compiler, rtl_check_runner, capability_derive,
corpus_spec, generate_corpus) skipped their taxonomy-powered check identically on it. Measured
2026-09-21 the empty set was gemmini, gemmini_universal, muon, saturn and toy_npu, and only two of
those five were empty for a reason anyone had established.
"""

from __future__ import annotations

import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import isa_taxonomy as IT


class _Stub:
    """The two descriptor attributes :func:`derive_isa_taxonomy` reads, and nothing else."""

    def __init__(self, target, isa_headers):
        self.target = target
        self.isa_headers = list(isa_headers)
        self.hwbringup_set = None


@pytest.fixture(autouse=True)
def _no_cache():
    IT.clear_cache()
    yield
    IT.clear_cache()


def test_undeterminable_taxonomy_is_not_reported_as_no_isa():
    """THE MUTATION TEST for fix 1: a target we could not examine must not read as examined.

    A name with no descriptor is the exact live state of muon/saturn/toy_npu. The record has to say
    UNKNOWN — if it said ``not_applicable`` (or, as before, said nothing at all), every consumer
    would record "this target declares no ISA" about a target nobody read a single byte of.
    """
    tax = IT.taxonomy_for_target("a-target-that-does-not-exist")
    assert IT.status_of(tax) == IT.STATUS_UNKNOWN
    assert IT.is_unknown(tax) is True
    # NOT a statement that the target has no ISA.
    assert IT.declares_isa(tax) is True
    assert IT.reasons_of(tax), "an empty taxonomy that records no reason is the original defect"
    # It is still EMPTY, so an untaught consumer keeps skipping rather than screening a target
    # against a taxonomy holding zero instruction classes (a vacuous pass).
    assert not tax
    assert (tax.get("by_class"), tax.get("by_mnemonic")) == ({}, {})


def test_a_target_declaring_no_isa_definition_is_not_unknown():
    """The other half: a RoCC/command-ISA target genuinely has nothing to derive here."""
    tax = IT.derive_isa_taxonomy(_Stub("rocc-ish", ["some/dir/isa_include/target.h"]))
    assert IT.status_of(tax) == IT.STATUS_NOT_APPLICABLE
    assert IT.is_unknown(tax) is False
    assert IT.declares_isa(tax) is False
    assert not tax


def test_declared_isa_file_that_is_absent_is_unknown_not_missing_isa():
    tax = IT.derive_isa_taxonomy(_Stub("t", ["nowhere/at/all/isa_definition.py"]))
    assert IT.status_of(tax) == IT.STATUS_UNKNOWN
    assert IT.declares_isa(tax) is True


@pytest.mark.parametrize(
    "boom",
    [
        pytest.param(
            lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "ImportError: boom"), id="helper-exits-nonzero"
        ),
        pytest.param(lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("cmd", 1)), id="helper-times-out"),
        pytest.param(lambda *a, **k: (_ for _ in ()).throw(OSError("no such interpreter")), id="helper-cannot-launch"),
    ],
)
def test_helper_failure_is_unknown_not_no_isa(tmp_path, monkeypatch, boom):
    """A crashed / timed-out / unlaunchable introspect subprocess is a MISSING INPUT.

    This is the ``~:94-95`` conflation: the target ships an ISA definition, so an empty result can
    only mean the derivation failed — never that the hardware has no instruction taxonomy.
    """
    isa = tmp_path / "isa_definition.py"
    isa.write_text("# a real file the descriptor points at\n", encoding="utf-8")
    monkeypatch.setattr(IT.subprocess, "run", boom)
    tax = IT.derive_isa_taxonomy(_Stub("t", [str(isa)]))
    assert IT.status_of(tax) == IT.STATUS_UNKNOWN
    assert IT.declares_isa(tax) is True
    assert IT.reasons_of(tax)


def test_helper_that_succeeds_but_grounds_nothing_is_unknown(tmp_path, monkeypatch):
    isa = tmp_path / "isa_definition.py"
    isa.write_text("#\n", encoding="utf-8")

    def _empty(cmd, **kw):
        out = cmd[cmd.index("--out") + 1]
        from pathlib import Path as _P

        _P(out).write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(IT.subprocess, "run", _empty)
    tax = IT.derive_isa_taxonomy(_Stub("t", [str(isa)]))
    assert IT.status_of(tax) == IT.STATUS_UNKNOWN


def test_a_bare_empty_dict_reads_as_unknown_never_as_no_isa():
    """Fail-closed for the legacy value every one of these call sites used to return."""
    assert IT.status_of({}) == IT.STATUS_UNKNOWN
    assert IT.status_of(None) == IT.STATUS_UNKNOWN
    assert IT.declares_isa({}) is True


def test_require_taxonomy_distinguishes_the_two_refusals():
    unknown = IT.taxonomy_for_target("a-target-that-does-not-exist")
    with pytest.raises(IT.TaxonomyUnknown):
        IT.require_taxonomy(unknown, "x", needs="the required instruction classes")
    na = IT.derive_isa_taxonomy(_Stub("rocc-ish", ["d/isa_include/t.h"]))
    with pytest.raises(NotImplementedError):
        IT.require_taxonomy(na, "x", needs="the required instruction classes")


def test_a_derived_taxonomy_is_still_truthy_and_derived():
    tax = IT.Taxonomy({"by_class": {"C": [{"role": "matmul"}]}, "by_mnemonic": {}, "asm_mnemonics": {}})
    assert bool(tax) is True
    assert IT.status_of(tax) == IT.STATUS_DERIVED
    assert IT.require_taxonomy(tax, "x", needs="y")["by_class"] == {"C": [{"role": "matmul"}]}


def test_live_targets_land_in_the_right_bucket():
    """The in-tree descriptors, read for real. Neither case runs a subprocess."""
    gemmini = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    if not gemmini.is_file():
        pytest.skip("no in-tree gemmini descriptor")
    # gemmini ships C ISA headers and no isa_definition.py: nothing to derive, and nothing wrong.
    assert IT.status_of(IT.taxonomy_for_target("gemmini")) == IT.STATUS_NOT_APPLICABLE
    # muon ships no capsule-bench descriptor at all: UNDETERMINED, and that is a finding.
    assert IT.status_of(IT.taxonomy_for_target("muon")) == IT.STATUS_UNKNOWN
