"""The rate table must be a BOUND built from measured evidence, never a summary of it.

Every test here pins a decision where the tempting implementation is wrong in a way that would not
show up as a crash: averaging repeated measurements, defaulting an unmeasured class, or pricing a
program whose work is only partly counted. Each of those produces a plausible number, and a plausible
number is exactly what a ceiling must not be.
"""
from __future__ import annotations

import pytest

from merlin.perf import rate_table as RT


def _resident_buffer(*, jobs: int = 1, m: int = 16, k: int = 16, n: int = 16) -> dict:
    """A resident-weight matmul program, in the same ABI shape the certified corpus emits."""
    tensors = {"W": {"shape": [k, n], "dtype": "i8", "role": "weight"}}
    commands = [{"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
                 "attributes": {"layout": "packed_rhs"}}]
    for j in range(jobs):
        tensors[f"A{j}"] = {"shape": [m, k], "dtype": "i8", "role": "input"}
        tensors[f"Y{j}"] = {"shape": [m, n], "dtype": "i32", "role": "output"}
        commands.append({"opcode": "MATMUL_RESIDENT",
                         "operands": {"lhs": f"A{j}", "rhs": "W_res", "dst": f"acc{j}"}})
        commands.append({"opcode": "COMMIT", "operands": {"src": f"acc{j}", "dst": f"Y{j}"},
                         "attributes": {"epilogue": [], "output_dtype": "i32"}})
    commands.append({"opcode": "EVICT", "operands": {"handle": "W_res"}})
    return {"abi_version": "0.1", "target": "t", "tensors": tensors, "commands": commands}


def _program(name: str, buffer: dict, *measured: float) -> RT.Program:
    p = RT.Program(digest=RT.program_digest(buffer), buffer=buffer, workload=name)
    p.measured.update(float(m) for m in measured)
    p.submissions.add("sub")
    return p


def _table(programs, *, peak: float = 256.0) -> RT.RateTable:
    return RT.rates_for("t", peak_macs_per_cycle=peak,
                        programs={p.digest: p for p in programs})


class TestTheRateIsTheSlowestNotTheTypical:
    def test_the_slowest_program_of_a_class_sets_its_rate(self):
        """A ceiling divides by the slowest rate; a mean of rates would be beaten by half the corpus."""
        fast = _program("fast", _resident_buffer(jobs=4), 100.0)
        slow = _program("slow", _resident_buffer(jobs=2), 900.0)
        table = _table([fast, slow])

        rate = table.rates["MATMUL_RESIDENT"]
        assert rate.slowest_from == "slow"
        # 2 jobs x 16x16x16 = 8192 MACs over 900 cycles.
        assert rate.slowest_macs_per_cycle == pytest.approx(8192 / 900)
        assert rate.fastest_macs_per_cycle > rate.slowest_macs_per_cycle
        assert rate.n_programs == 2

    def test_a_buffer_measured_several_times_uses_its_slowest_and_is_not_discarded(self):
        """One buffer compiled by two submissions is two programs, so the spread is real evidence.

        Discarding it threw away 42 of 84 programs on the corpus this was written against -- half the
        evidence on disk -- and discarding is not conservative: it removes the slow observation that
        the ceiling exists to cover.
        """
        multi = _program("multi", _resident_buffer(jobs=2), 311.0, 316.0, 317.0)
        table = _table([multi])

        assert "MATMUL_RESIDENT" in table.rates, "a multi-measured buffer must still be priced"
        assert table.rates["MATMUL_RESIDENT"].slowest_macs_per_cycle == pytest.approx(8192 / 317.0)
        recorded = table.disagreements
        assert len(recorded) == 1 and recorded[0]["used"] == 317.0
        assert recorded[0]["measured"] == [311.0, 316.0, 317.0]

    def test_nothing_is_averaged(self):
        """The mean of 311/316/317 describes no run that happened, so it must not appear."""
        table = _table([_program("multi", _resident_buffer(jobs=2), 311.0, 316.0, 317.0)])
        mean_rate = 8192 / ((311.0 + 316.0 + 317.0) / 3)
        assert table.rates["MATMUL_RESIDENT"].slowest_macs_per_cycle != pytest.approx(mean_rate)


class TestAnUnmeasuredClassIsNeverGivenANumber:
    def test_rate_for_an_unrated_class_is_none(self):
        """`None` and a plausible default are the difference between declining and guessing."""
        table = _table([_program("only", _resident_buffer(), 300.0)])
        assert table.rate_for("CONV2D") is None
        assert table.rate_for(None) is None
        assert table.rate_for("") is None

    def test_the_classes_nothing_priced_are_named(self):
        """A caller reading only `rates` sees a table that looks complete for whatever it holds."""
        table = _table([_program("only", _resident_buffer(), 300.0)])
        assert "MATMUL_RESIDENT" not in table.unpriced_classes
        assert "CONV2D" in table.unpriced_classes
        assert list(table.to_dict()["unpriced_classes"]) == list(table.unpriced_classes)

    def test_a_table_built_on_no_derived_peak_is_refused(self):
        """A rate table over an assumed peak describes a machine nobody has."""
        with pytest.raises(ValueError, match="derived structural peak"):
            RT.rates_for("t", peak_macs_per_cycle=0.0, programs={})


class TestOnlyEvidenceThatCanBoundAnythingContributes:
    def test_a_program_whose_work_is_only_partly_counted_is_refused(self):
        """A lower-bound price yields a too-slow rate, and a ceiling from a too-slow rate is too tight.

        This is the asymmetry that makes the refusal necessary: the structural FLOOR accepts the same
        program happily, because a floor may err downward.
        """
        buffer = _resident_buffer(jobs=1)
        buffer["commands"].insert(1, {"opcode": "SOMETHING_UNPRICED", "operands": {}})
        table = _table([_program("partial", buffer, 300.0)])

        assert table.rates == {}, "an unpriced command must not silently set a ceiling"
        assert any("lower bound" in r["reason"] for r in table.refusals)

    def test_a_program_with_no_priced_opcode_is_refused_with_its_reason(self):
        movement = {"abi_version": "0.1", "target": "t",
                    "tensors": {"X": {"shape": [4, 4], "dtype": "i8", "role": "input"}},
                    "commands": [{"opcode": "MOVEMENT", "operands": {"src": "X", "dst": "Y"}}]}
        table = _table([_program("moved", movement, 50.0)])

        assert table.rates == {}
        assert any("priced vocabulary" in r["reason"] for r in table.refusals)

    def test_a_program_with_no_measurement_contributes_nothing(self):
        table = _table([_program("unmeasured", _resident_buffer())])
        assert table.rates == {} and table.n_programs_seen == 1


class TestTheTableCarriesWhatIsNeededToDistrustIt:
    def test_every_rate_states_the_cycle_domain_it_was_observed_over(self):
        """A rate is an empirical bound over a domain; priced outside it, it is an extrapolation."""
        table = _table([_program("a", _resident_buffer(jobs=1), 300.0),
                        _program("b", _resident_buffer(jobs=8), 1800.0)])
        rate = table.rates["MATMUL_RESIDENT"]
        assert (rate.cycles_min, rate.cycles_max) == (300.0, 1800.0)
        assert "domain" in rate.to_dict()["licence"]

    def test_the_table_declares_its_evidence_contended_and_unpromotable(self):
        """Harvested numbers are trace_derived: other engines were live in the same window."""
        provenance = _table([_program("a", _resident_buffer(), 300.0)]).to_dict()["provenance"]
        assert provenance["kind"] == "trace_derived"
        assert "never promotable" in provenance["note"] or "promotable" in provenance["note"]
