"""An absent event is not a zero, and this is the whole point of the trace vocabulary.

A trace with no ``dma_read`` events can mean no DMA happened, or that this producer cannot see DMA.
Those readings differ by everything. Collapsing them is a recorded failure in this tree - a path that
turned every failure into "no result" - so a Trace declares what its producer CAN emit, and every
aggregate over a kind outside that set is UNKNOWN rather than 0.
"""
from __future__ import annotations

from merlin.kernels.dyntrace import EVENT_KINDS, OVERHEAD_KINDS, Trace, TraceEvent


def _t(records, events=(), **kw):
    return Trace(source="t", records=frozenset(records), events=tuple(events), **kw)


class TestAbsenceIsNotZero:
    def test_a_producer_that_cannot_see_a_kind_answers_unknown(self):
        t = _t({"compute"})
        assert t.bytes_moved(kinds=("dma_read",)) is None
        assert t.cycles_in("dma_read") is None

    def test_a_producer_that_can_see_it_and_saw_none_answers_zero(self):
        """The other half: declared-recordable and genuinely absent IS a real zero."""
        t = _t({"dma_read"})
        assert t.bytes_moved(kinds=("dma_read",)) == 0
        assert t.cycles_in("dma_read") == 0

    def test_an_undeclared_producer_answers_unknown_for_everything(self):
        t = _t(set(), [])
        assert t.bytes_moved(kinds=("dma_read",)) is None
        assert t.overhead_cycles() is None
        assert "UNKNOWN" in t.gaps()[0]

    def test_a_partially_sized_total_is_not_a_total(self):
        """NEGATIVE CASE: one unsized event makes the sum a lower bound, so it is refused."""
        t = _t({"dma_read"}, [TraceEvent(kind="dma_read", nbytes=64),
                              TraceEvent(kind="dma_read", nbytes=None)])
        assert t.bytes_moved(kinds=("dma_read",)) is None

    def test_a_partially_timed_total_is_not_a_total(self):
        t = _t({"compute"}, [TraceEvent(kind="compute", start=0, end=5),
                             TraceEvent(kind="compute", start=None, end=None)])
        assert t.cycles_in("compute") is None


class TestOverheadIsAllOrNothing:
    def test_overhead_is_unknown_unless_every_overhead_kind_is_visible(self):
        """Summing only the overhead kinds a producer happens to emit understates overhead by
        exactly the kinds it cannot see - the direction that flatters the result."""
        partial = _t({"sync", "queue_wait"})
        assert partial.overhead_cycles() is None

    def test_overhead_sums_when_every_kind_is_visible(self):
        t = _t(OVERHEAD_KINDS, [TraceEvent(kind="sync", start=0, end=3),
                                TraceEvent(kind="engine_idle", start=3, end=10)])
        assert t.overhead_cycles() == 10


class TestTheVocabularyIsClosedAndChecked:
    def test_an_unknown_event_kind_is_a_problem(self):
        assert _t({"compute"}, [TraceEvent(kind="teleport")]).problems()

    def test_declaring_an_unknown_recordable_kind_is_a_problem(self):
        assert _t({"teleport"}).problems()

    def test_emitting_a_kind_the_producer_did_not_declare_is_a_problem(self):
        """The capability declaration is what makes an absence readable, so it must be complete."""
        probs = _t({"dma_read"}, [TraceEvent(kind="compute")]).problems()
        assert any("without declaring" in p for p in probs), probs

    def test_an_event_cannot_end_before_it_starts(self):
        assert TraceEvent(kind="compute", start=10, end=3).problems()

    def test_overhead_kinds_are_a_subset_of_the_vocabulary(self):
        assert OVERHEAD_KINDS <= set(EVENT_KINDS)

    def test_a_well_formed_trace_has_no_problems(self):
        t = _t({"compute"}, [TraceEvent(kind="compute", engine="spatial", start=0, end=4)])
        assert t.problems() == () and t.engines_seen() == ("spatial",)
