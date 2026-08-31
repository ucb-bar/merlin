"""Joint occupancy from a target's own hardware counters.

Realised overlap normally needs a per-cycle trace, and that needs a waveform build or a co-simulation
model. A target whose RTL counts the cycles in which each COMBINATION of engines was busy needs
neither -- a combination counter is a joint-occupancy reading. These tests hold the derivation and the
three states, and specifically that the engine names are factored out of the header rather than typed.
"""
from __future__ import annotations

import textwrap

from merlin.perf import hw_counters as H

# A synthetic header in the shape a real one has: three singles, three pairs, the triple. Names chosen
# NOT to match any real target, so a passing test cannot be passing because a spelling was hardcoded.
_HDR = textwrap.dedent("""\
    #define WHOLE_ALPHA_CYCLES 1
    #define WHOLE_BETA_CYCLES 2
    #define WHOLE_GAMMA_CYCLES 3
    #define WHOLE_ALPHA_BETA_CYCLES 4
    #define WHOLE_ALPHA_GAMMA_CYCLES 5
    #define WHOLE_BETA_GAMMA_CYCLES 6
    #define WHOLE_ALPHA_BETA_GAMMA_CYCLES 7
    #define SOMETHING_EVENTS 8
    #define A_MACRO(x) ((x) + 1)
    """)


def _counters(text=_HDR):
    return H.derive_occupancy_counters(text)


class TestDerivation:
    def test_the_engines_are_factored_out_of_the_names(self):
        oc = _counters()
        assert oc.engines == ("ALPHA", "BETA", "GAMMA")
        assert oc.prefix == "WHOLE"

    def test_every_combination_is_found_and_the_set_is_complete(self):
        oc = _counters()
        assert len(oc.by_combination) == 7
        assert oc.complete()
        assert oc.by_combination[frozenset({"ALPHA", "BETA", "GAMMA"})] == "WHOLE_ALPHA_BETA_GAMMA_CYCLES"

    def test_singles_and_overlaps_are_separated(self):
        oc = _counters()
        assert set(oc.singles()) == {"ALPHA", "BETA", "GAMMA"}
        assert len(oc.overlaps()) == 4

    def test_an_event_counter_is_not_a_cycle_counter(self):
        # A joint-occupancy figure is a DURATION; an event count is not one.
        assert all("EVENTS" not in n for n in _counters().by_combination.values())

    def test_a_function_like_macro_is_ignored(self):
        assert "A_MACRO" not in str(_counters().by_combination)

    def test_one_engine_cannot_overlap_with_itself(self):
        one = "#define P_ONLY_CYCLES 1\n#define P_ONLY_ONLY_CYCLES 2\n"
        assert H.derive_occupancy_counters(one).by_combination == {}

    def test_an_incomplete_set_is_usable_but_not_complete(self):
        # With a pair missing, the realised total is a LOWER BOUND; reporting it as the total
        # understates eta, so the distinction is carried rather than smoothed over.
        partial = _HDR.replace("#define WHOLE_BETA_GAMMA_CYCLES 6\n", "")
        oc = H.derive_occupancy_counters(partial)
        assert oc.engines == ("ALPHA", "BETA", "GAMMA") and not oc.complete()


class TestEta:
    def _oc(self):
        return _counters()

    def _vals(self, **over):
        v = {"WHOLE_ALPHA_CYCLES": 100, "WHOLE_BETA_CYCLES": 100, "WHOLE_GAMMA_CYCLES": 100,
             "WHOLE_ALPHA_BETA_CYCLES": 0, "WHOLE_ALPHA_GAMMA_CYCLES": 0,
             "WHOLE_BETA_GAMMA_CYCLES": 0, "WHOLE_ALPHA_BETA_GAMMA_CYCLES": 0}
        v.update(over)
        return v

    def test_a_serialised_schedule_measures_zero(self):
        got = H.eta_from_counters(self._vals(), self._oc())
        assert got["state"] == "measured" and got["eta"] == 0.0

    def test_an_overlapped_schedule_measures_more(self):
        got = H.eta_from_counters(
            self._vals(WHOLE_ALPHA_BETA_CYCLES=60, WHOLE_ALPHA_CYCLES=40, WHOLE_BETA_CYCLES=40),
            self._oc())
        assert got["state"] == "measured" and got["eta"] == 0.6
        # The busy totals are unchanged -- the same work, differently scheduled, which is the whole
        # premise of the A/B this feeds.
        assert got["busy_cycles"] == {"ALPHA": 100, "BETA": 100, "GAMMA": 100}

    def test_a_per_engine_total_includes_its_overlapping_cycles(self):
        # Reading a single as the whole-engine total understates the busiest engine and inflates eta.
        got = H.eta_from_counters(
            self._vals(WHOLE_ALPHA_CYCLES=10, WHOLE_ALPHA_BETA_CYCLES=90), self._oc())
        assert got["busy_cycles"]["ALPHA"] == 100

    def test_a_missing_reading_is_unknown_not_zero(self):
        v = self._vals()
        v.pop("WHOLE_ALPHA_BETA_GAMMA_CYCLES")
        got = H.eta_from_counters(v, self._oc())
        assert got["state"] == "unknown" and got["eta"] is None
        assert "would report overlap that was never measured" in got["why"]

    def test_no_available_overlap_is_undefined_not_zero(self):
        idle = {n: 0 for n in self._oc().by_combination.values()}
        got = H.eta_from_counters(idle, self._oc())
        assert got["state"] == "unknown" and "0/0 is undefined" in got["why"]


class TestThreeStatesForATarget:
    def test_a_header_with_no_combination_block_is_absent_not_unavailable(self, tmp_path):
        h = tmp_path / "x.h"
        h.write_text("#define SOMETHING 1\n")
        got = H.counters_for_target("t", sources=[h])
        assert got["status"] == "absent" and "does not count overlap in hardware" in got["why"]

    def test_an_unlocatable_header_is_unavailable_and_says_it_is_not_absence(self, tmp_path):
        got = H.counters_for_target("t", sources=[tmp_path / "nope.h"])
        assert got["status"] == "unavailable"
        assert "UNKNOWN, not absent" in got["why"]

    def test_a_real_target_resolves_its_own_counters(self):
        # No target name and no counter spelling is asserted here; only that IF a block resolves, it
        # resolves completely and names its own header.
        import pytest

        from merlin.targetgen import target_registry as tr
        for name in sorted(tr.all_targets())[:12]:
            got = H.counters_for_target(name)
            if got.get("status") == "derived":
                c = got["counters"]
                assert len(c["engines"]) >= 2 and c["by_combination"]
                assert got["header"].endswith(".h")
                return
        pytest.skip("no target in this checkout ships a combination-counter header")


class TestBracket:
    def _oc(self):
        return _counters()

    def _codes(self):
        return H.event_codes(_HDR)

    def test_every_derived_counter_gets_a_slot_and_its_own_event_code(self):
        b = H.counter_bracket_c(self._oc(), self._codes(), slots=8)
        assert len(b["slot_of"]) == 7
        assert sorted(b["slot_of"].values()) == list(range(7))
        for name, code in ((n, self._codes()[n]) for n in b["slot_of"]):
            assert f"counter_configure({b['slot_of'][name]}, {code})" in b["prologue"]

    def test_a_missing_event_code_is_refused(self):
        # A read at an unconfigured slot returns whatever that slot last held, which would be
        # reported as this run's overlap.
        codes = dict(self._codes())
        codes.pop("WHOLE_ALPHA_BETA_GAMMA_CYCLES")
        try:
            H.counter_bracket_c(self._oc(), codes, slots=8)
        except ValueError as e:
            assert "unconfigured slot" in str(e)
        else:
            raise AssertionError("a missing event code must be refused, not skipped")

    def test_too_few_slots_is_refused_rather_than_partially_emitted(self):
        # A missing combination makes the realised total a LOWER BOUND; emitting a partial bracket
        # would have that reported as the total.
        try:
            H.counter_bracket_c(self._oc(), self._codes(), slots=3)
        except ValueError as e:
            assert "lower" in str(e)
        else:
            raise AssertionError("a counter set larger than the slot count must be refused")

    def test_the_reader_attributes_by_name_not_by_position(self):
        # A simulator console interleaves writers; a positional reader mis-attributes silently.
        console = ("garbage from another writer\n"
                   f"{H.COUNTER_MARKER} WHOLE_BETA_CYCLES 22\n"
                   "more noise\n"
                   f"{H.COUNTER_MARKER} WHOLE_ALPHA_CYCLES 11\n")
        assert H.parse_counter_output(console) == {"WHOLE_ALPHA_CYCLES": 11, "WHOLE_BETA_CYCLES": 22}

    def test_a_truncated_value_is_dropped_not_coerced(self):
        # A truncated console is a MISSING reading, which eta already refuses on -- never a zero.
        got = H.parse_counter_output(f"{H.COUNTER_MARKER} WHOLE_ALPHA_CYCLES \n")
        assert got == {}

    def test_an_empty_console_yields_nothing(self):
        assert H.parse_counter_output("") == {}

    def test_bracket_to_eta_round_trips(self):
        # The whole path: derive the counters, emit the bracket, read a console back, compute eta.
        oc = self._oc()
        b = H.counter_bracket_c(oc, self._codes(), slots=8)
        assert b["epilogue"].count("counter_read(") == 7
        console = "\n".join(
            f"{H.COUNTER_MARKER} {n} {60 if len(k) >= 2 else 40}"
            for k, n in oc.by_combination.items())
        got = H.eta_from_counters(H.parse_counter_output(console), oc)
        assert got["state"] == "measured" and got["eta"] > 0
