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


def _proof_args(header=_HDR):
    counters = _counters(header)
    codes = H.event_codes(header)
    leaves = {engine: f"%busy_{index}" for index, engine in enumerate(counters.engines)}
    lines = ["hw.module @Device() {"]
    negated = {}
    for index, engine in enumerate(counters.engines):
        negated[engine] = f"%not_{index}"
        lines.append(f"  %not_{index} = comb.xor bin {leaves[engine]}, %true : i1")
    event_refs = {}
    for index, (combo, _name) in enumerate(sorted(
            counters.by_combination.items(), key=lambda item: sorted(item[0]))):
        event_refs[combo] = f"%event_{index}"
        operands = [leaves[engine] if engine in combo else negated[engine]
                    for engine in counters.engines]
        lines.append(f"  %event_{index} = comb.and bin {', '.join(operands)} : i1")
    ports = []
    for combo, name in counters.by_combination.items():
        ports.append(f"io_event_io_event_signal_{codes[name]}: {event_refs[combo]}: i1")
    lines.append(f'  %unused = hw.instance "meter" @Meter({", ".join(ports)}) -> (x: i1)')
    lines.append("}")
    return {"hw_text": "\n".join(lines), "codes": codes,
            "module": "Device", "counter_module": "Meter", "measurement_cycles": 1000,
            "source": "synthetic.mlir"}


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
        got = H.eta_from_counters(self._vals(), self._oc(), **_proof_args())
        assert got["state"] == "measured" and got["eta"] == 0.0

    def test_an_overlapped_schedule_measures_more(self):
        # 60 overlapped cycles against three engines each busy 100. Available is min(total - busiest,
        # total // 2) = min(200, 150) = 150, so eta is 0.4.
        #
        # It was 0.6 here, under a denominator that took the second-largest per-engine total. That is
        # right for TWO engines and wrong for three: with three engines overlapping in disjoint pairs
        # the numerator counts every pair while the denominator admits only the top pair's ceiling, and
        # eta exceeds 1. The first real run of this instrument on hardware returned 1.1726 and 1.0253 --
        # not fractions, and not quotable as percentages. The bound is now what the per-engine totals
        # actually admit.
        got = H.eta_from_counters(
            self._vals(WHOLE_ALPHA_BETA_CYCLES=60, WHOLE_ALPHA_CYCLES=40, WHOLE_BETA_CYCLES=40),
            self._oc(), **_proof_args())
        assert got["state"] == "measured" and got["eta"] == 0.4
        assert got["available_cycles"] == 150
        # The busy totals are unchanged -- the same work, differently scheduled, which is the whole
        # premise of the A/B this feeds.
        assert got["busy_cycles"] == {"ALPHA": 100, "BETA": 100, "GAMMA": 100}

    def test_eta_never_exceeds_one(self):
        # The property the old denominator broke. Three engines overlapping in disjoint pairs is the
        # case that produced it, so it is the case tested.
        got = H.eta_from_counters(
            self._vals(WHOLE_ALPHA_BETA_CYCLES=100, WHOLE_ALPHA_GAMMA_CYCLES=100,
                       WHOLE_BETA_GAMMA_CYCLES=100, WHOLE_ALPHA_CYCLES=0,
                       WHOLE_BETA_CYCLES=0, WHOLE_GAMMA_CYCLES=0),
            self._oc(), **_proof_args())
        assert got["state"] == "measured" and got["eta"] <= 1.0

    def test_two_engines_still_use_the_second_largest_total(self):
        # The generalisation must REDUCE to the convention headroom and the falsifier were written for,
        # or this eta and theirs stop being the same quantity wherever both apply.
        hdr = ("#define P_A_CYCLES 1\n#define P_B_CYCLES 2\n#define P_A_B_CYCLES 3\n")
        oc = H.derive_occupancy_counters(hdr)
        got = H.eta_from_counters(
            {"P_A_CYCLES": 40, "P_B_CYCLES": 0, "P_A_B_CYCLES": 60}, oc,
            **_proof_args(hdr))
        # busy: A=100, B=60 -> second-largest 60; min(160-100, 160//2) = min(60, 80) = 60.
        assert got["available_cycles"] == 60 and got["eta"] == 1.0

    def test_a_per_engine_total_includes_its_overlapping_cycles(self):
        # Reading a single as the whole-engine total understates the busiest engine and inflates eta.
        got = H.eta_from_counters(
            self._vals(WHOLE_ALPHA_CYCLES=10, WHOLE_ALPHA_BETA_CYCLES=90), self._oc(),
            **_proof_args())
        assert got["busy_cycles"]["ALPHA"] == 100

    def test_a_missing_reading_is_unknown_not_zero(self):
        v = self._vals()
        v.pop("WHOLE_ALPHA_BETA_GAMMA_CYCLES")
        got = H.eta_from_counters(v, self._oc(), **_proof_args())
        assert got["state"] == "unknown" and got["eta"] is None
        assert "would report overlap that was never measured" in got["why"]

    def test_no_available_overlap_is_undefined_not_zero(self):
        idle = {n: 0 for n in self._oc().by_combination.values()}
        got = H.eta_from_counters(idle, self._oc(), **_proof_args())
        assert got["state"] == "unknown" and "0/0 is undefined" in got["why"]

    def test_negative_or_non_integer_readings_are_unknown(self):
        for invalid in (-1, 1.5, True, "7"):
            values = self._vals(WHOLE_ALPHA_CYCLES=invalid)
            got = H.eta_from_counters(values, self._oc(), **_proof_args())
            assert got["state"] == "unknown" and "non-negative integers" in got["why"]

    def test_readings_from_another_counter_pass_are_not_ignored(self):
        values = self._vals()
        values["ANOTHER_WINDOW_BYTES"] = 99
        got = H.eta_from_counters(values, self._oc(), **_proof_args())
        assert got["state"] == "unknown" and "exactly its proved partition" in got["why"]

    def test_partition_cannot_exceed_the_identical_cycle_window(self):
        args = _proof_args()
        args["measurement_cycles"] = 299
        got = H.eta_from_counters(self._vals(), self._oc(), **args)
        assert got["state"] == "unknown" and "mixed, corrupt, or wrapped" in got["why"]

    def test_missing_cycle_window_is_unknown(self):
        args = _proof_args()
        args.pop("measurement_cycles")
        got = H.eta_from_counters(self._vals(), self._oc(), **args)
        assert got["state"] == "unknown" and "identical counter window" in got["why"]


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


class TestCounterCapacityFromCirct:
    def _hw(self):
        return textwrap.dedent("""\
            module {
              hw.module private @Meter(in %clock : !seq.clock) {
                %bank_0 = seq.firreg %a clock %clock : i9
                %bank_1 = seq.firreg %b clock %clock : i9
                %shadow_0 = seq.firreg %c clock %clock : i64
                %shadow_1 = seq.firreg %d clock %clock : i64
                %value_0 = seq.firreg %e clock %clock : i64
                %value_1 = seq.firreg %f clock %clock : i64
                hw.output
              }
            }
            """)

    def test_capacity_is_derived_from_matching_elaborated_state_families(self):
        got = H.counter_slots_from_circt(
            self._hw(), module="Meter", state_families=("bank", "shadow", "value"),
            source="synthetic.mlir")
        assert got["status"] == "derived" and got["slots"] == 2
        assert got["provenance"]["method"] == "dense_seq_firreg_families"
        assert len(got["provenance"]["sha256"]) == 64

    def test_sparse_indices_are_unknown_not_an_inferred_capacity(self):
        hw = self._hw().replace("bank_1", "bank_2").replace("shadow_1", "shadow_2").replace(
            "value_1", "value_2")
        got = H.counter_slots_from_circt(hw, module="Meter",
                                         state_families=("bank", "shadow", "value"))
        assert got["status"] == "unknown" and got["slots"] is None
        assert "dense zero-based" in got["why"]

    def test_disagreeing_state_families_are_unknown(self):
        hw = self._hw().replace("    %shadow_1 = seq.firreg %d clock %clock : i64\n", "")
        got = H.counter_slots_from_circt(hw, module="Meter",
                                         state_families=("bank", "shadow", "value"))
        assert got["status"] == "unknown" and "disagree" in got["why"]

    def test_an_absent_module_is_unknown_without_a_numeric_fallback(self):
        got = H.counter_slots_from_circt(self._hw(), module="Elsewhere",
                                         state_families=("bank", "shadow", "value"))
        assert got["status"] == "unknown" and got["slots"] is None
        assert "absent" in got["why"]


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

    def test_two_names_cannot_share_one_event_code(self):
        codes = dict(self._codes())
        codes["WHOLE_BETA_CYCLES"] = codes["WHOLE_ALPHA_CYCLES"]
        try:
            H.counter_bracket_c(self._oc(), codes, slots=8)
        except ValueError as exc:
            assert "share an event code" in str(exc)
        else:
            raise AssertionError("duplicate event codes must not receive two labelled slots")

    def test_too_few_slots_is_refused_rather_than_partially_emitted(self):
        # A missing combination makes the realised total a LOWER BOUND; emitting a partial bracket
        # would have that reported as the total.
        try:
            H.counter_bracket_c(self._oc(), self._codes(), slots=3)
        except ValueError as e:
            assert "lower" in str(e)
        else:
            raise AssertionError("a counter set larger than the slot count must be refused")

    def test_all_physical_slots_can_be_padded_with_a_derived_disabled_code(self):
        bracket = H.counter_bracket_for_names(
            ("A", "B"), {"A": 4, "B": 7}, slots=5, padding_code=0)

        assert bracket["configured_slots"] == 5
        assert bracket["prologue"].count("counter_configure(") == 5
        assert bracket["epilogue"].count("counter_read(") == 2
        assert "counter_configure(4, 0);  // padding: disabled event" in bracket["prologue"]

    def test_padding_never_assumes_a_disabled_event_code(self):
        bracket = H.counter_bracket_for_names(("A",), {"A": 4}, slots=5)

        assert bracket["configured_slots"] == 1
        assert bracket["prologue"].count("counter_configure(") == 1

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

    def test_duplicate_counter_lines_are_ambiguous_not_last_writer_wins(self):
        console = (f"{H.COUNTER_MARKER} WHOLE_ALPHA_CYCLES 1\n"
                   f"{H.COUNTER_MARKER} WHOLE_ALPHA_CYCLES 2\n")
        assert H.parse_counter_output(console) == {}

    def test_counter_schema_requires_one_unique_content_digest(self):
        digest = "a" * 64
        assert H.parse_counter_schema(f"{H.COUNTER_SCHEMA_MARKER} {digest}\n") == digest
        assert H.parse_counter_schema(
            f"{H.COUNTER_SCHEMA_MARKER} {digest}\n{H.COUNTER_SCHEMA_MARKER} {'b' * 64}\n") is None

    def test_bracket_to_eta_round_trips(self):
        # The whole path: derive the counters, emit the bracket, read a console back, compute eta.
        oc = self._oc()
        b = H.counter_bracket_c(oc, self._codes(), slots=8)
        assert b["epilogue"].count("counter_read(") == 7
        console = "\n".join(
            f"{H.COUNTER_MARKER} {n} {60 if len(k) >= 2 else 40}"
            for k, n in oc.by_combination.items())
        got = H.eta_from_counters(H.parse_counter_output(console), oc, **_proof_args())
        assert got["state"] == "measured" and got["eta"] > 0


class TestObservationsFromCounters:
    """The hop from counter readings to a `merlin.perf.observations` block.

    The three properties that decide whether the numbers mean anything: a per-engine busy total
    includes its shared cycles, the block declares itself NON-partitioned (which is what licenses the
    overlap reading), and a missing combination costs its engines their entry rather than zeroing them.
    """

    # ALPHA alone 10, BETA alone 20, GAMMA alone 30, A+B 4, A+G 5, B+G 6, A+B+G 7.  Charged cycles 82.
    _VALUES = {"WHOLE_ALPHA_CYCLES": 10, "WHOLE_BETA_CYCLES": 20, "WHOLE_GAMMA_CYCLES": 30,
               "WHOLE_ALPHA_BETA_CYCLES": 4, "WHOLE_ALPHA_GAMMA_CYCLES": 5,
               "WHOLE_BETA_GAMMA_CYCLES": 6, "WHOLE_ALPHA_BETA_GAMMA_CYCLES": 7}

    def _block(self, values=None, total=100):
        return H.observations_from_counters(values if values is not None else self._VALUES,
                                            _counters(), total_cycles=total, source="unit test")

    def _q(self, block):
        from merlin.perf import observations as OBS
        return {e["quantity"]: e["value"] for e in block[OBS.TIMING_OBSERVATIONS_KEY]}

    def test_a_per_engine_busy_total_includes_its_shared_cycles(self):
        q = self._q(self._block())
        # ALPHA = 10 + 4 + 5 + 7.  Reading the single alone would say 10 and understate the engine.
        assert q["busy_cycles.ALPHA.in_program"] == 26
        assert q["busy_cycles.BETA.in_program"] == 20 + 4 + 6 + 7
        assert q["busy_cycles.GAMMA.in_program"] == 30 + 5 + 6 + 7

    def test_the_totals_deliberately_exceed_the_window(self):
        """Overlap is counted once per engine sharing it, so the busy totals do NOT partition."""
        q = self._q(self._block(total=100))
        busy = sum(v for k, v in q.items() if k.startswith("busy_cycles."))
        assert busy > 100

    def test_overlap_is_the_multi_engine_combinations(self):
        q = self._q(self._block())
        assert q["overlap_cycles.observed"] == 4 + 5 + 6 + 7

    def test_idle_is_the_window_less_the_charged_cycles(self):
        q = self._q(self._block(total=100))
        assert q["idle_cycles.no_unit_busy"] == 100 - 82

    def test_the_block_declares_itself_not_partitioned_so_overlap_survives(self):
        from merlin.perf import observations as OBS
        block = self._block()
        assert block[OBS.PARTITIONED_KEY] is False
        validated = OBS.validate_block(block)
        assert validated is not None and validated.partitioned is False
        kept = {e["quantity"] for e in validated.observations}
        assert OBS.OVERLAP_OBSERVED in kept, "a partitioned block would have had its overlap refused"

    def test_a_missing_combination_unmeasures_its_engines_rather_than_zeroing_them(self):
        from merlin.perf import observations as OBS
        values = dict(self._VALUES)
        values.pop("WHOLE_ALPHA_BETA_CYCLES")
        block = self._block(values)
        assert block[OBS.UNMEASURED_UNITS_KEY] == ["ALPHA", "BETA"]
        q = self._q(block)
        assert "busy_cycles.ALPHA.in_program" not in q
        assert "busy_cycles.BETA.in_program" not in q
        assert q["busy_cycles.GAMMA.in_program"] == 30 + 5 + 6 + 7, "GAMMA is still fully readable"

    def test_a_missing_combination_withholds_overlap_and_idle_entirely(self):
        values = dict(self._VALUES)
        values.pop("WHOLE_BETA_GAMMA_CYCLES")
        q = self._q(self._block(values))
        assert "overlap_cycles.observed" not in q, "a partial sum would be read as the total"
        assert "idle_cycles.no_unit_busy" not in q

    def test_no_total_cycles_means_no_idle_quantity_rather_than_a_guess(self):
        q = self._q(H.observations_from_counters(self._VALUES, _counters(), total_cycles=None))
        assert "idle_cycles.no_unit_busy" not in q
        assert "overlap_cycles.observed" in q

    def test_a_window_smaller_than_the_charged_cycles_drops_idle(self):
        """Counters and the cycle window disagreeing is a fact to notice, not a negative duration."""
        q = self._q(self._block(total=10))
        assert "idle_cycles.no_unit_busy" not in q

    def test_the_engine_names_are_not_hardcoded(self):
        other = textwrap.dedent("""\
            #define PART_RED_CYCLES 1
            #define PART_BLUE_CYCLES 2
            #define PART_RED_BLUE_CYCLES 3
            """)
        block = H.observations_from_counters(
            {"PART_RED_CYCLES": 5, "PART_BLUE_CYCLES": 6, "PART_RED_BLUE_CYCLES": 2},
            H.derive_occupancy_counters(other), total_cycles=20)
        q = self._q(block)
        assert q["busy_cycles.RED.in_program"] == 7
        assert q["busy_cycles.BLUE.in_program"] == 8
        assert q["overlap_cycles.observed"] == 2
        assert q["idle_cycles.no_unit_busy"] == 20 - 13

    def test_across_kind_overlap_excludes_two_engines_of_one_kind(self):
        """`eta_from_timing_block` resolves on the KIND axis, so it needs this quantity, not the other.

        ALPHA computes; BETA and GAMMA both move. Their shared cycles are real overlap but they are not
        movement/compute overlap, and charging them to the kind-axis reading would overstate what the
        pairing achieved.
        """
        block = H.observations_from_counters(
            self._VALUES, _counters(), total_cycles=100,
            kind_of={"ALPHA": "compute", "BETA": "movement", "GAMMA": "movement"})
        q = self._q(block)
        assert q["overlap_cycles.observed"] == 4 + 5 + 6 + 7
        assert q["overlap_cycles.across_kinds"] == 4 + 5 + 7, "the BETA+GAMMA 6 is within one kind"

    def test_all_distinct_kinds_make_the_two_overlap_readings_agree(self):
        block = H.observations_from_counters(
            self._VALUES, _counters(), total_cycles=100,
            kind_of={"ALPHA": "compute", "BETA": "movement", "GAMMA": "other"})
        q = self._q(block)
        assert q["overlap_cycles.across_kinds"] == q["overlap_cycles.observed"]

    def test_no_kind_map_means_no_kind_axis_reading(self):
        """A unit's kind is declared, never derived from a counter name."""
        q = self._q(self._block())
        assert "overlap_cycles.across_kinds" not in q
        assert "overlap_cycles.observed" in q

    def test_a_partial_kind_map_is_refused_rather_than_guessed(self):
        q = self._q(H.observations_from_counters(self._VALUES, _counters(), total_cycles=100,
                                                  kind_of={"ALPHA": "compute"}))
        assert "overlap_cycles.across_kinds" not in q, (
            "an unclassified engine makes every combination containing it unclassifiable")
    def test_incomplete_header_set_never_produces_a_measured_eta(self):
        header = _HDR.replace("#define WHOLE_BETA_GAMMA_CYCLES 6\n", "")
        counters = _counters(header)
        values = {name: 10 for name in counters.by_combination.values()}
        got = H.eta_from_counters(values, counters, **_proof_args(header))
        assert got["state"] == "unknown" and got["eta"] is None
        assert "exclusivity" in got["why"]

    def test_header_grammar_without_matching_rtl_partition_is_unknown(self):
        args = _proof_args()
        args["hw_text"] = args["hw_text"].replace(
            "%event_0 = comb.and", "%event_0 = comb.or")
        got = H.eta_from_counters(
            {name: 10 for name in _counters().by_combination.values()}, _counters(), **args)
        assert got["state"] == "unknown" and got["eta"] is None
        assert got["partition_proof"]["status"] == "unknown"


class TestDerivedCounterExpressions:
    def test_incremental_counter_codes_are_resolved_without_copying_the_values(self):
        header = textwrap.dedent("""\
            #define FIRST_BLOCK 41
            #define READ_BYTES (FIRST_BLOCK + 4)
            #define WRITE_BYTES (FIRST_BLOCK + 5)
            #define LATENCY_TOTAL (FIRST_BLOCK + 6)
            """)
        assert H.event_codes(header)["READ_BYTES"] == 45
        assert H.counters_with_unit(header, "bytes") == {
            "READ_BYTES": 45, "WRITE_BYTES": 46,
        }

    def test_unit_selection_uses_the_headers_tokens_not_target_spellings(self):
        header = "#define ODD_TRANSPORTED_WORDS 9\n#define ODD_BYTES_MOVED 10\n"
        assert H.counters_with_unit(header, "BYTES") == {"ODD_BYTES_MOVED": 10}

    def test_a_unit_selected_set_uses_the_same_fail_closed_bracket(self):
        header = "#define NORTH_BYTES 9\n#define SOUTH_BYTES 10\n"
        selected = H.counters_with_unit(header, "BYTES")
        bracket = H.counter_bracket_for_names(tuple(selected), H.event_codes(header), slots=2)
        assert set(bracket["slot_of"]) == set(selected)
        assert bracket["epilogue"].count("counter_read(") == 2
