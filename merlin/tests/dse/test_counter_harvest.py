"""Harvesting counter-bracketed runs, and refusing the fabricated set that sits beside the real one.

Measured 2026-09-08: one graded capsule run writes a console per oracle tier into one directory. The
cycle-accurate console reported the seven ``MAIN_*`` counters as 0..83 against a 366-cycle window;
the functional ISS console, in the same directory, reported 2,609..4,744 against its own 65-cycle
window — **35x the window**, because that model increments every counter with ``rand()``. Another
capsule's fabricated set charged 751,149 cycles against 1,174 (640x). The numbers look equally
plausible in isolation, so the engine is read from the console's own name and admitted only through
the trust contract, and the partition sum is independently required not to exceed its window.
"""
from __future__ import annotations

import pytest

from merlin.perf import counter_harvest as CH
from merlin.perf import hw_counters as HC

HEADER = "\n".join((
    "#define MAIN_LD_CYCLES 1", "#define MAIN_ST_CYCLES 2", "#define MAIN_EX_CYCLES 3",
    "#define MAIN_LD_ST_CYCLES 4", "#define MAIN_LD_EX_CYCLES 5",
    "#define MAIN_ST_EX_CYCLES 6", "#define MAIN_LD_ST_EX_CYCLES 7",
))
KINDS = {"EX": "compute", "LD": "movement", "ST": "movement"}
#: The target's declared oracle vocabulary, as a backend's ORACLE keys supply it.
ENGINES = ("gsim", "spike", "verilator")

#: The real GSIM reading from A3_k_accumulation.
REAL = {"MAIN_EX_CYCLES": 83, "MAIN_LD_EX_CYCLES": 52, "MAIN_LD_ST_EX_CYCLES": 0,
        "MAIN_ST_EX_CYCLES": 0, "MAIN_LD_CYCLES": 39, "MAIN_LD_ST_CYCLES": 0,
        "MAIN_ST_CYCLES": 44}
#: The fabricated reading the functional ISS produced for the SAME capsule.
FAKE = {"MAIN_EX_CYCLES": 4744, "MAIN_LD_CYCLES": 4358, "MAIN_LD_EX_CYCLES": 3282,
        "MAIN_LD_ST_CYCLES": 4156, "MAIN_LD_ST_EX_CYCLES": 3793, "MAIN_ST_CYCLES": 3962,
        "MAIN_ST_EX_CYCLES": 2609}


def _console(readings, cycles):
    lines = [f"{HC.COUNTER_MARKER} {k} {v}" for k, v in readings.items()]
    lines.append(f"{CH.CYCLE_MARKER} {cycles}")
    lines.append("DONE")
    return "\n".join(lines) + "\n"


def _run(tmp_path, workload, filename, readings, cycles):
    d = tmp_path / workload / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(_console(readings, cycles), encoding="utf-8")
    return d / filename


class TestTheEngineIsReadNotAssumed:
    def test_the_cycle_accurate_console_is_trusted(self, tmp_path):
        _run(tmp_path, "A3", "rtl_gsim_console.log", REAL, 366)
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        assert len(got.runs) == 1
        run = got.runs[0]
        assert run.engine == "gsim" and run.trusted and run.workload == "A3"
        assert run.total_cycles == 366 and run.charged == 218

    def test_the_functional_iss_console_is_refused_by_the_trust_contract(self, tmp_path):
        _run(tmp_path, "A3", "spike_gemmini_functional_console.log", FAKE, 65)
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        assert got.trusted() == []
        assert "fabricated" in got.runs[0].refusal

    def test_both_consoles_in_ONE_directory_are_separated(self, tmp_path):
        """The situation as it actually occurs: one graded run, one console per tier, same dir."""
        _run(tmp_path, "A3", "rtl_gsim_console.log", REAL, 366)
        _run(tmp_path, "A3", "spike_gemmini_functional_console.log", FAKE, 65)
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        assert len(got.runs) == 2 and len(got.trusted()) == 1
        assert got.trusted()[0].engine == "gsim"

    def test_a_console_whose_engine_cannot_be_named_is_refused_not_defaulted(self, tmp_path):
        _run(tmp_path, "A3", "mystery_console.log", REAL, 366)
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        assert got.runs == []
        assert "names none of the declared engines" in got.refusals[0]["reason"]

    def test_the_longest_engine_name_wins_so_a_substring_cannot_capture_a_tier(self):
        """`spike_gemmini_functional_console` must not be claimed by a shorter key it contains."""
        from pathlib import Path
        assert CH.engine_of(Path("spike_gemmini_functional_console.log"),
                            ("spike", "spike_gemmini_functional")) == "spike_gemmini_functional"
        assert CH.engine_of(Path("rtl_gsim_console.log"), ENGINES) == "gsim"
        assert CH.engine_of(Path("mystery.log"), ENGINES) is None

    def test_the_engine_vocabulary_comes_from_the_caller_not_a_table_here(self):
        """A filename table in shared code would name one target's console conventions."""
        assert not hasattr(CH, "CONSOLE_ENGINE_MARKERS")
        from pathlib import Path
        assert CH.engine_of(Path("rtl_gsim_console.log"), ()) is None


class TestArithmeticImpossibilityIsCheckedIndependently:
    def test_a_partition_charging_more_than_its_window_is_refused(self, tmp_path):
        """Caught the fabricated set at 35x, and would catch a wrapped counter on a real engine."""
        _run(tmp_path, "A3", "rtl_gsim_console.log", FAKE, 65)
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        assert got.trusted() == []
        reason = got.runs[0].refusal
        assert "cannot exceed its own window" in reason and "wrapped" in reason

    def test_the_real_reading_passes_that_same_check(self, tmp_path):
        _run(tmp_path, "A3", "rtl_gsim_console.log", REAL, 366)
        assert CH.harvest_counter_runs(tmp_path, engines=ENGINES).trusted()[0].charged <= 366

    def test_a_console_with_no_cycle_line_is_refused(self, tmp_path):
        """Without a window there is no host residue, only a bag of busy counts."""
        d = tmp_path / "A3" / "artifacts"
        d.mkdir(parents=True)
        (d / "rtl_gsim_console.log").write_text(
            "\n".join(f"{HC.COUNTER_MARKER} {k} {v}" for k, v in REAL.items()) + "\n",
            encoding="utf-8")
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        assert got.runs == [] and "no window" in got.refusals[0]["reason"]


class TestTheCorpusMap:
    def test_it_closes_accelerator_busy_and_names_the_host_residue(self, tmp_path):
        _run(tmp_path, "A3", "rtl_gsim_console.log", REAL, 366)
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        corpus, refusals = CH.corpus_map(got.runs, header_text=HEADER, kind_of=KINDS)
        assert refusals == [] and corpus is not None
        att = corpus.workloads["A3"]
        host = next(c for c in att.components if c.bucket == "host")
        assert host.measured_cycles == 366 - 218, "rdcycle minus the partition sum, closed"

    def test_a_fabricated_run_contributes_nothing_and_says_why(self, tmp_path):
        _run(tmp_path, "A3", "spike_gemmini_functional_console.log", FAKE, 65)
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        corpus, refusals = CH.corpus_map(got.runs, header_text=HEADER, kind_of=KINDS)
        assert corpus is None
        assert refusals and "fabricated" in refusals[0]["reason"]

    def test_the_formatted_map_reports_NONE_where_there_is_nothing_to_win(self, tmp_path):
        _run(tmp_path, "A3", "rtl_gsim_console.log", REAL, 366)
        got = CH.harvest_counter_runs(tmp_path, engines=ENGINES)
        corpus, refusals = CH.corpus_map(got.runs, header_text=HEADER, kind_of=KINDS)
        text = CH.format_corpus_map(corpus, refusals=refusals)
        assert "A3" in text and "host" in text
        # stall/control carry no measured cycles here, and NONE is the finding for them.
        assert "none" in text

    def test_an_empty_harvest_formats_as_a_refusal_not_an_empty_table(self, tmp_path):
        text = CH.format_corpus_map(None, refusals=[{"workload": "A3", "reason": "fabricated"}])
        assert "no workload produced a closed partition" in text and "A3" in text
