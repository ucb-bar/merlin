"""The expressiveness measurement: the triple, what is charged to its denominator, and what is not.

Every assertion here is about a way this number could read HIGH while meaning nothing. A ratio that
drops the fields it could not decide, a denominator that shrinks to whatever happened to be on disk,
an agreement reported on a facet the comparator never looked at, a corpus of our own compiler's output
standing in for experts -- each has a test, because each is a number that cannot fail.

Fast and CPU-only: no simulator, no external checkout, no build. The corpus tests use fakes for the
resolution so they assert the POLICY rather than this host's filesystem; one structural test reads the
real register and the real registry and holds them together.
"""

from __future__ import annotations

import pytest

from merlin.kernels import cca as cca_mod
from merlin.kernels.cca import CCA, CommunicationFacet, DispatchFacet, MemoryFacet, lift_schedule
from merlin.sched import expressiveness as ex
from merlin.sched.ir import Kernel, Stage, TensorArg, call, loop
from merlin.sched.mach import Machine, Sync

ARG = (TensorArg("a", (64,), "i8", "read"),)


def _cca(**facets) -> CCA:
    return CCA(op="matmul", backend=["t"], provenance={"level": "test", "source": "fixture"}, **facets)


# -- the triple ------------------------------------------------------------------------------------


def test_a_triple_that_does_not_close_is_refused():
    """The arithmetic invariant, enforced rather than commented.

    ``expressed + unexercised == denominator`` is the whole guarantee that no field went missing. A
    triple that does not close is one whose parts were computed against different field sets, which is
    exactly how a denominator shrinks without anyone deciding to shrink it.
    """
    ex.Triple(2, 9, 7)
    with pytest.raises(ex.ExpressivenessError):
        ex.Triple(2, 9, 6)


def test_thin_evidence_reads_as_a_wide_range_not_a_high_number():
    """Two comparable fields out of nine is 2/9/7 and the interval [0.22, 1.0] -- not '100%'."""
    thin = ex.Triple(2, 9, 7)
    lo, hi = thin.bounds()
    assert (round(lo, 2), hi) == (0.22, 1.0)
    # The temptation this exists to defeat: reporting only the fields that were decided.
    decided_only = ex.Triple(2, 2, 0)
    assert decided_only.bounds() == (1.0, 1.0)
    assert thin.bounds() != decided_only.bounds()


def test_nothing_measured_bounds_nothing():
    assert ex.Triple(0, 0, 0).bounds() == (0.0, 1.0)


# -- what is charged to the denominator ------------------------------------------------------------


def test_a_field_only_one_side_populates_is_undecidable_and_still_counted():
    """The comparator skips a one-sided field silently -- absent from the disagreements AND from
    ``compared_fields``. Left there it is in neither the numerator nor the denominator, which is how a
    coverage ratio reads high because the hard cases were quietly dropped."""
    expert = _cca(memory=MemoryFacet(banks_used=4))
    ours = _cca(memory=MemoryFacet(banks_used=None))
    (v,) = ex.field_verdicts(expert, ours, ["memory.banks_used"])
    assert v.state == ex.UNDECIDABLE
    assert "only the corpus" in v.reason
    assert ex.triple_of([v]) == ex.Triple(0, 1, 1)


def test_a_field_neither_side_determines_is_undecidable_not_agreement():
    expert, ours = _cca(memory=MemoryFacet()), _cca(memory=MemoryFacet())
    (v,) = ex.field_verdicts(expert, ours, ["memory.banks_used"])
    assert v.state == ex.UNDECIDABLE
    assert ex.triple_of([v]).expressed == 0


def test_a_field_on_a_facet_the_comparator_skips_is_detected_by_observation():
    """The blindness this measurement must not inherit, and the way it must not be detected.

    ``cca_agree`` iterates a hardcoded facet tuple that omits two facets, so a field on one of them is
    invisible to it even when BOTH sides populate it -- and the report still says the two agree. The
    reason is derived from the two CCAs plus the comparator's own ``compared_fields``, never from a
    copy of the blind-facet list kept here: a second copy of a known limitation is a second thing to
    forget, and it would keep reporting the old answer after the tuple was widened.
    """
    expert = _cca(communication=CommunicationFacet(fences=3))
    ours = _cca(communication=CommunicationFacet(fences=3))
    assert cca_mod.cca_agree(expert, ours).agree, "the premise: identical values, reported as agreeing"
    (v,) = ex.field_verdicts(expert, ours, ["communication.fences"])
    assert v.state == ex.UNDECIDABLE, (
        "a facet cca_agree does not iterate was read as agreement. If the tuple was widened, this "
        "measurement should now DECIDE the field -- which is an improvement, and this assertion is "
        "how you find out it happened."
    )
    assert "did not compare it" in v.reason
    assert v.expert == v.ours == 3, "both sides were populated; the comparator simply never looked"


def test_only_agreement_on_a_compared_field_is_expressed():
    same = ex.field_verdicts(
        _cca(dispatch=DispatchFacet(n_dispatches=12)),
        _cca(dispatch=DispatchFacet(n_dispatches=12)),
        ["dispatch.n_dispatches"],
    )
    assert [v.state for v in same] == [ex.AGREE]
    assert ex.triple_of(same) == ex.Triple(1, 1, 0)

    differ = ex.field_verdicts(
        _cca(dispatch=DispatchFacet(n_dispatches=12)),
        _cca(dispatch=DispatchFacet(n_dispatches=9)),
        ["dispatch.n_dispatches"],
    )
    assert [v.state for v in differ] == [ex.DISAGREE]
    assert ex.triple_of(differ) == ex.Triple(0, 1, 1), "a disagreement is charged, never dropped"


def test_the_whole_report_denominator_is_the_union_not_the_intersection():
    """``report_triple`` takes the comparator's compared fields UNION the ones it could not compare.
    Intersecting instead would let a thinner corpus produce a better-looking number."""
    expert = _cca(dispatch=DispatchFacet(n_dispatches=12, config_fraction=0.5))
    ours = _cca(dispatch=DispatchFacet(n_dispatches=12))
    triple, verdicts = ex.report_triple(expert, ours)
    axes = {v.axis for v in verdicts}
    assert "dispatch.n_dispatches" in axes and "dispatch.config_fraction" in axes
    assert triple.denominator == len(verdicts) and triple.expressed == 1


# -- what UNMEASURED means -------------------------------------------------------------------------


def _row(rid="ax", axes=("dispatch.n_dispatches", "memory.banks_used"), corpus="c"):
    return {"id": rid, "evidence": {"corpus": corpus, "cca_axes": list(axes)}}


def test_an_unresolvable_corpus_reports_unmeasured_naming_the_missing_input():
    """Never a zero denominator: a measurement of nothing reads exactly like agreement on everything."""
    unresolved = ex.CorpusStatus(name="c", resolved=False, missing_input="$SOME_ROOT (unset)")
    m = ex.measure_axis(_row(), corpus=unresolved)
    assert m.status == "UNMEASURED"
    assert "$SOME_ROOT" in m.missing_input
    assert m.triple == ex.Triple(0, 2, 2), "the fields are still counted; the denominator does not shrink"
    assert all(v.state == ex.UNDECIDABLE for v in m.verdicts)


def test_a_resolvable_corpus_with_no_lifter_names_the_lifter_not_the_path():
    resolved = ex.CorpusStatus(name="c", resolved=True, admitted=57, observed_commit="0" * 40, pin="p")
    m = ex.measure_axis(_row(), corpus=resolved)
    assert m.status == "UNMEASURED" and "lifts it" in m.missing_input
    assert m.measured_on, "a resolvable corpus records which bytes it is, even when nothing lifted them"


def test_a_row_naming_no_corpus_makes_no_agreement_claim():
    m = ex.measure_axis(_row(corpus=""))
    assert m.status == "EXPRESSED" and m.triple == ex.Triple(0, 2, 2) and not m.missing_input


def test_an_axis_is_exercised_only_when_every_field_it_names_agreed():
    resolved = ex.CorpusStatus(name="c", resolved=True, admitted=1, observed_commit="a" * 40, pin="p")
    both = _cca(dispatch=DispatchFacet(n_dispatches=12), memory=MemoryFacet(banks_used=4))
    assert ex.measure_axis(_row(), expert=both, ours=both, corpus=resolved).status == "EXERCISED"

    half = _cca(dispatch=DispatchFacet(n_dispatches=12), memory=MemoryFacet(banks_used=None))
    partial = ex.measure_axis(_row(), expert=both, ours=half, corpus=resolved)
    assert partial.status == "EXPRESSED", "one undecidable field is enough to stop EXERCISED"
    assert partial.triple == ex.Triple(1, 2, 1)


def test_an_off_pin_corpus_is_measured_and_says_so_rather_than_passing_silently():
    """Drift changes WHICH bytes the result is about; it does not make the result unmeasurable. The
    stamp has to carry it, or a number gets attributed to a revision it did not come from."""
    drifted = ex.CorpusStatus(
        name="c",
        resolved=True,
        admitted=1,
        pin="p",
        declared_commit="d" * 40,
        observed_commit="0" * 40,
        drift=("commit is 000000000000 but the pin declares dddddddddddd",),
    )
    both = _cca(dispatch=DispatchFacet(n_dispatches=1), memory=MemoryFacet(banks_used=1))
    m = ex.measure_axis(_row(), expert=both, ours=both, corpus=drifted)
    assert m.status == "EXERCISED"
    assert "OFF-PIN" in m.measured_on and "dddddddddddd" in m.measured_on
    assert m.notes == drifted.drift


def test_totals_count_the_rows_that_could_not_be_measured():
    rows = [ex.measure_axis(_row(rid=f"a{i}"), corpus=ex.CorpusStatus("c", False, "x")) for i in range(3)]
    assert ex.totals(rows) == ex.Triple(0, 6, 6), "0/N/N, never 0/0/0"


# -- eligibility: a corpus of our own output is not an expert corpus --------------------------------


def test_a_corpus_with_no_provenance_audit_admits_nothing(tmp_path):
    """'We did not check who wrote these' and 'experts wrote these' are different statements, and only
    the second is the claim an expressiveness measurement would be making by using them."""
    admitted, why = ex._eligibility({}, tmp_path, "directory", "MERLIN_X")
    assert admitted == 0 and "provenance_record" in why


def test_an_audit_admits_only_its_declared_verdicts_scoped_and_deduped(tmp_path, monkeypatch):
    """The measured bug this guards: counting the audit's rows gave more admitted kernels than the
    corpus has directories, because one record classifies several checkouts and names the same kernel
    in each -- a numerator that outran its own denominator."""
    for name in ("k_hand", "k_generated"):
        (tmp_path / name).mkdir()
    record = tmp_path / "audit.yaml"
    record.write_text(
        "kernels:\n"
        "- {name: k_hand, repo: '${MERLIN_X}', verdict: hand}\n"
        "- {name: k_hand, repo: '${MERLIN_OTHER}', verdict: hand}\n"
        "- {name: k_generated, repo: '${MERLIN_X}', verdict: compiler_generated}\n"
        "- {name: k_absent, repo: '${MERLIN_X}', verdict: hand}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("merlin.targetgen.corpora.sched_corpus_record", lambda spec: record)
    admitted, why = ex._eligibility({"admits": ["hand"]}, tmp_path, "directory", "MERLIN_X")
    assert admitted == 1, why  # the other checkout's duplicate and the absent kernel are both out


# -- our side of the comparison --------------------------------------------------------------------


def _staged(name, bank, writes):
    return Stage(memory="spad", row=0, rows=4, bank=bank, writes=writes)


def test_lift_schedule_counts_dynamic_instances_not_source_sites():
    """A loop nest denotes more commands than it spells; the CCA field means the former."""
    k = Kernel("k", ARG, (loop("i", 4, loop("j", 2, call("mac", a=0))),))
    got = lift_schedule(k, op="matmul", backend="t")
    assert got.dispatch.n_dispatches == 8


def test_lift_schedule_reports_the_configuration_share_and_the_bank_count():
    k = Kernel(
        "k",
        ARG,
        (
            call("config", sets={"stride": 1}),
            loop("i", 3, call("mac", a=0, stages=(_staged("a", 0, False), _staged("d", 1, True)))),
        ),
    )
    got = lift_schedule(k, op="matmul", backend="t")
    assert got.dispatch.n_dispatches == 4
    assert got.dispatch.config_fraction == pytest.approx(0.25)
    assert got.memory.banks_used == 2


def test_lift_schedule_reads_overlap_off_the_token_not_off_statement_order():
    """A dependence recorded only as statement order does not survive a reordering, which is the thing
    a schedule does. Overlap is the distance between a token's producer and its consumer."""
    serial = Kernel("k", ARG, (call("dma", produces="t"), call("mac", awaits="t")))
    assert lift_schedule(serial, op="m", backend="t").dispatch.dma_overlap is False
    assert lift_schedule(serial, op="m", backend="t").dispatch.dma_issue_to_wait == 0

    overlapped = Kernel("k", ARG, (call("dma", produces="t"), call("mac"), call("mac", awaits="t")))
    got = lift_schedule(overlapped, op="m", backend="t")
    assert got.dispatch.dma_overlap is True and got.dispatch.dma_issue_to_wait == 1

    none = Kernel("k", ARG, (call("mac"),))
    assert lift_schedule(none, op="m", backend="t").dispatch.dma_overlap is None, (
        "a schedule with nothing asynchronous makes no claim about overlap; False would be one"
    )


def test_fences_are_counted_only_against_a_machine_that_declares_them():
    """'This schedule has no fences' and 'nobody wrote down what a fence is here' are different
    statements, and only the first is a measurement."""
    k = Kernel("k", ARG, (call("fence"), call("mac")))
    assert lift_schedule(k, op="m", backend="t").communication is None
    mach = Machine(target="t", hazard_resolution="explicit", syncs=(Sync(instr="fence", orders="completion"),))
    got = lift_schedule(k, machine=mach, op="m", backend="t")
    assert got.communication.fences == 1


def test_an_empty_schedule_reports_none_rather_than_zero_dispatches():
    got = lift_schedule(Kernel("k", ARG, ()), op="m", backend="t")
    assert got.dispatch.n_dispatches is None and got.dispatch.config_fraction is None


# -- the register and the registry, held together ---------------------------------------------------


def test_every_corpus_the_register_names_is_one_the_registry_declares():
    """An evidence pointer to a corpus nobody declared is the one omission that makes the measurement
    impossible while leaving the register looking complete."""
    from merlin.common.paths import merlin_dir
    from merlin.common.yaml import load_yaml
    from merlin.targetgen.corpora import sched_corpora

    doc = load_yaml(merlin_dir() / "contract" / "schedule_ir_coverage.yaml") or {}
    declared = set(sched_corpora())
    named = {
        str((row.get("evidence") or {}).get("corpus"))
        for row in (doc.get("entries") or [])
        if (row.get("evidence") or {}).get("corpus")
    }
    assert named, "no row names a corpus; the measurement has nothing to resolve"
    assert named <= declared, f"the register names corpora the registry does not declare: {sorted(named - declared)}"


def test_every_declared_corpus_names_a_pin_that_exists():
    """A corpus resolved by a pin gets its root variable, its declared revision and drift detection.
    A pin name that resolves to nothing gets none of them and fails as if the checkout were absent."""
    from merlin.common import provenance
    from merlin.targetgen.corpora import sched_corpora

    pins = set(provenance.load_pins())
    for name, spec in sched_corpora().items():
        assert spec.get("pin") in pins, f"corpus {name!r} names pin {spec.get('pin')!r}, which is not declared"


def test_measure_register_resolves_each_corpus_once(monkeypatch):
    seen: list[str] = []

    def fake(name):
        seen.append(name)
        return ex.CorpusStatus(name=name, resolved=False, missing_input="$X")

    monkeypatch.setattr(ex, "corpus_status", fake)
    rows = [_row(rid="a", corpus="c"), _row(rid="b", corpus="c"), _row(rid="d", corpus="other")]
    out = ex.measure_register(rows)
    assert seen == ["c", "other"], "a report cannot show one row's corpus present and another's absent"
    assert [m.status for m in out] == ["UNMEASURED"] * 3
