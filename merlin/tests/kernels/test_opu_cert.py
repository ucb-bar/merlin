"""The certification harness is judged on what it does with a BAD run, not a good one.

A harness that only reports pass/fail correctly for a working kernel is worthless — every interesting
failure here is one where something looks like a pass. So most of these tests feed it a console that is
wrong in a specific way (a shape that does not match the case, a digest that does not match numpy, a
missing case, a geometry the corpus was not selected for, an image with none of the unit's instructions)
and require that it says so.

The one property that cannot be tested from Python is that the emitted C hashes bytes the same way
:func:`fnv1a64` does. That is established by running the image: the scalar pre-flight build computes a
digest per case on a real RISC-V target and every one matches the host's.
"""
from __future__ import annotations

import pytest

from merlin.kernels import opu_cert as C
from merlin.kernels import opu_corpus as K
from merlin.kernels.opu_kernel import KernelSpec

_SPEC = KernelSpec(accumulate="ACC", broadcast="BCAST", readout="READOUT")


def _cases(tile: int = 16):
    runnable, deferred = K.select(tile)
    return runnable, deferred


def _console(cases, expected, *, tile: int, mismatches: int = 0, digest_delta: int = 0,
             drop: set[str] | None = None, shape_shift: int = 0, done: bool = True) -> str:
    drop = drop or set()
    lines = [f"TILE {tile}"]
    for c in cases:
        if c.name in drop:
            continue
        e = expected[c.name]
        lines.append(f"CASE {c.name} {int(c.m) + shape_shift} {c.n} {c.k} {mismatches} "
                     f"{-1 if not mismatches else 0} {(e['digest'] + digest_delta) & ((1 << 64) - 1):#018x}")
    if done:
        lines.append("DONE")
    return "\n".join(lines) + "\n"


class TestTheHash:
    def test_the_empty_input_is_the_offset_basis(self):
        assert C.fnv1a64(b"") == 0xCBF29CE484222325

    def test_it_matches_the_published_fnv1a_vector(self):
        # "a" -> 0xaf63dc4c8601ec8c is the canonical 64-bit FNV-1a test vector; if this drifts, the
        # emitted C mirror and the host would still agree with each other while both being wrong.
        assert C.fnv1a64(b"a") == 0xAF63DC4C8601EC8C

    def test_transposing_two_bytes_changes_it(self):
        # A sum or an xor would not notice, and two compensating element errors are a realistic bug.
        assert C.fnv1a64(bytes([1, 2])) != C.fnv1a64(bytes([2, 1]))


class TestTheTileEdge:
    def test_it_is_vlmax_for_the_operand_width(self):
        assert C.logical_tile_edge(256) == 32
        assert C.logical_tile_edge(128) == 16

    def test_a_wider_element_narrows_the_tile(self):
        assert C.logical_tile_edge(256, sew_bits=32) == 8

    @pytest.mark.parametrize("bad", [0, -8])
    def test_a_nonsense_vector_length_raises_rather_than_defaulting(self, bad):
        with pytest.raises(ValueError):
            C.logical_tile_edge(bad)

    def test_an_ungroundable_config_raises_instead_of_guessing(self, tmp_path):
        cfg = tmp_path / "c.scala"
        cfg.write_text("class Foo extends Config(new Whatever(1) ++ new Base)", encoding="utf-8")
        with pytest.raises(ValueError, match="vector length"):
            C.tile_edge_for_config("Foo", config_scala=cfg, mixin_scala=[])

    def test_it_reads_the_config_and_its_mixin(self, tmp_path):
        cfg = tmp_path / "c.scala"
        cfg.write_text("class C256 extends Config(\n  new a.b.WithShuttleVectorUnit(256, 128, P) ++\n"
                       "  new Base)\n", encoding="utf-8")
        mixin = tmp_path / "m.scala"
        mixin.write_text("class WithShuttleVectorUnit(\n  vLen: Int = 128,\n  dLen: Int = 64) "
                         "extends Config((s, h, u) => { })\n", encoding="utf-8")
        assert C.tile_edge_for_config("C256", config_scala=cfg, mixin_scala=[mixin]) == 32


class TestTheImageCarriesNoAnswerKey:
    """An image holding its expected results can pass by copying them, and the console cannot tell."""

    def test_no_int32_array_other_than_a_declared_input_is_embedded(self):
        # Results are int32, so every int32 constant array in the image has to be an INPUT the case
        # declares -- a bias, or a requant multiplier. Widened when the epilogue arrived: a multiplier is an
        # input like a bias, not an expected output, so it does not weaken the property. What must stay
        # impossible is embedding a RESULT, which is what the digest tests below check.
        cases, _ = _cases(16)
        src = C.emit_image_c(cases)
        declared = sum(1 for c in cases if c.bias) + sum(1 for c in cases if c.requant)
        assert src.count("static const int32_t") == declared

    def test_no_expected_digest_appears_in_the_source(self):
        cases, _ = _cases(16)
        src = C.emit_image_c(cases)
        for name, exp in C.expected_digests(cases).items():
            assert f"{exp['digest']:#x}" not in src, f"{name}'s expected digest is in the image"

    def test_the_operands_are_embedded_so_both_sides_compute_on_the_same_bytes(self):
        cases, _ = _cases(16)
        src = C.emit_image_c(cases)
        for c in cases:
            assert f"at_{c.name}[" in src and f"b_{c.name}[" in src

    def test_an_empty_corpus_is_refused(self):
        # An image that runs nothing prints DONE and would read as a clean pass.
        with pytest.raises(ValueError, match="no cases"):
            C.emit_image_c([])

    def test_both_implementations_are_called_for_every_case(self):
        cases, _ = _cases(16)
        src = C.emit_image_c(cases, kernel_func="kern", ref_func="refr")
        assert "kern(c_dev," in src and "refr(c_ref," in src


class TestTheConsoleParser:
    def test_a_good_console_round_trips(self):
        cases, _ = _cases(16)
        exp = C.expected_digests(cases)
        got = C.parse_console(_console(cases, exp, tile=16))
        assert got["done"] and got["tile"] == 16
        assert len(got["cases"]) == len(cases) and not got["malformed"]

    def test_a_short_case_line_is_collected_not_dropped(self):
        # Dropping it would shrink the set of cases the report believes ran, which reads as coverage.
        got = C.parse_console("TILE 16\nCASE only_three_fields 1 2\nDONE\n")
        assert got["malformed"] and not got["cases"]

    def test_a_nonnumeric_field_is_collected_not_dropped(self):
        got = C.parse_console("TILE 16\nCASE x 1 2 3 zero -1 0x1\nDONE\n")
        assert got["malformed"] and not got["cases"]

    def test_a_nonnumeric_tile_is_collected(self):
        got = C.parse_console("TILE wide\nDONE\n")
        assert got["tile"] is None and got["malformed"]

    def test_a_truncated_run_has_no_done(self):
        assert not C.parse_console("TILE 16\n")["done"]


class TestTheVerdict:
    def _judge(self, **kw):
        cases, deferred = _cases(16)
        exp = C.expected_digests(cases)
        console = _console(cases, exp, tile=kw.pop("tile", 16), **kw)
        return cases, C.verdict(console, cases, config="T", tile_edge=16, deferred=deferred,
                                uses_unit=True)

    def test_a_clean_run_certifies(self):
        cases, rep = self._judge()
        assert rep.certified and len(rep.results) == len(cases)
        assert not rep.gaps

    def test_deferred_cases_are_reported_rather_than_omitted(self):
        cases, deferred = _cases(16)
        _, rep = self._judge()
        assert len(rep.deferred) == len(deferred)
        if deferred:
            assert all(why for _, why in rep.deferred), "a deferred case must carry its reason"

    def test_an_in_image_mismatch_fails_the_case(self):
        _, rep = self._judge(mismatches=3)
        assert not rep.certified
        assert all(not r.in_image_agrees and r.mismatches == 3 for r in rep.results)

    def test_a_wrong_digest_fails_even_when_the_in_image_check_passed(self):
        # This is the case where BOTH in-image implementations are wrong the same way. The in-image
        # comparison is happy; only numpy disagrees, and that has to be enough to fail.
        _, rep = self._judge(digest_delta=1)
        assert not rep.certified
        assert all(r.in_image_agrees and not r.digest_agrees for r in rep.results)

    def test_a_missing_case_is_not_a_pass(self):
        cases, _ = _cases(16)
        victim = cases[0].name
        _, rep = self._judge(drop={victim})
        got = next(r for r in rep.results if r.name == victim)
        assert not got.ran and not got.certified and "no CASE line" in got.note
        assert not rep.certified

    def test_a_shape_the_case_does_not_name_fails_with_a_note(self):
        # A run that computed a different shape but hashed it consistently would otherwise pass.
        _, rep = self._judge(shape_shift=1)
        assert not rep.certified
        assert all(not r.digest_agrees and "not the shape" in r.note for r in rep.results)

    def test_a_run_that_never_finished_is_a_gap(self):
        _, rep = self._judge(done=False)
        assert not rep.certified
        assert any("DONE" in g for g in rep.gaps)

    def test_a_geometry_the_corpus_was_not_selected_for_is_a_gap(self):
        # Certifying a 16-lane run against a corpus chosen for 32 lanes would claim shapes it never ran.
        cases, deferred = _cases(16)
        exp = C.expected_digests(cases)
        rep = C.verdict(_console(cases, exp, tile=32), cases, config="T", tile_edge=16,
                        uses_unit=True)
        assert not rep.certified
        assert any("tile edge" in g for g in rep.gaps)
        assert rep.tile_edge_reported == 32

    def test_an_image_without_the_units_instructions_is_a_gap(self):
        cases, _ = _cases(16)
        exp = C.expected_digests(cases)
        rep = C.verdict(_console(cases, exp, tile=16), cases, config="T", tile_edge=16,
                        uses_unit=False)
        assert not rep.certified
        assert any("did not use the unit" in g for g in rep.gaps)

    def test_a_malformed_line_is_a_gap_even_if_every_other_case_passed(self):
        cases, _ = _cases(16)
        exp = C.expected_digests(cases)
        rep = C.verdict(_console(cases, exp, tile=16) + "CASE broken 1\n", cases, config="T",
                        tile_edge=16, uses_unit=True)
        assert not rep.certified and any("unparseable" in g for g in rep.gaps)

    def test_an_empty_result_set_is_never_certified(self):
        rep = C.verdict("TILE 16\nDONE\n", [], config="T", tile_edge=16, uses_unit=True)
        assert not rep.certified

    def test_the_report_serialises_with_both_checks_visible(self):
        _, rep = self._judge()
        d = rep.to_dict()
        assert d["version"] == C.REPORT_VERSION and d["certified"]
        assert d["n_certified"] == d["n_cases"]
        assert all("in_image_agrees" in r and "digest_agrees" in r for r in d["results"])

    def test_the_report_writes_and_names_its_deferred_cases(self, tmp_path):
        import json
        _, rep = self._judge()
        p = rep.write(tmp_path / "r.json")
        got = json.loads(p.read_text())
        assert got["config"] == "T" and got["tile_edge"] == 16
        assert isinstance(got["deferred"], list)


class TestTheCorpusSelectionFeedsTheImage:
    def test_every_selected_case_is_fully_resolved_and_positive(self):
        # The old form of this test asserted every selected case fit ONE tile, which stopped being the
        # relevant property when the emitter gained M/N tiling -- the workload-scale cases are hundreds of
        # tiles wide by design. What must still hold is that nothing tile-relative reaches the image: an
        # unresolved extent would generate operands for a different shape than the case names.
        for tile in (4, 8, 16, 32):
            runnable, _ = _cases(tile)
            assert runnable, f"tile {tile} selected nothing"
            for c in runnable:
                assert all(isinstance(e, int) and e > 0 for e in (c.m, c.n, c.k)), (tile, c.name)

    def test_the_image_buffers_hold_the_largest_selected_case(self):
        # The image sizes its device and reference result buffers to the largest selected output. A case
        # selected but too large would overrun them -- and the corpus now carries workload-scale shapes
        # whose outputs are three orders of magnitude bigger than the probes'.
        for tile in (16, 32):
            runnable, _ = _cases(tile)
            biggest = max(int(c.m) * int(c.n) for c in runnable)
            src = C.emit_image_c(runnable)
            assert f"c_dev[{biggest}]" in src and f"c_ref[{biggest}]" in src

    def test_workload_scale_cases_are_selected_rather_than_deferred(self):
        # THE regression: the deferral criterion was the single-tile kernel's, so every real workload shape
        # was reported as "needs a kernel that tiles M and N" long after the emitter tiled.
        for tile in (16, 32):
            runnable, deferred = _cases(tile)
            names = {c.name for c in runnable}
            assert any(n.startswith("workload") for n in names), (tile, sorted(names))
            assert not [c.name for c, _why in deferred if c.name.startswith("workload")]

    def test_the_named_regressions_are_in_reach_at_every_tile(self):
        for tile in (4, 8, 16, 32):
            names = {c.name for c in _cases(tile)[0]}
            assert {"narrow_m_1", "narrow_n_1", "narrow_both_1"} <= names

    def test_the_buffers_are_sized_for_the_largest_selected_case(self):
        cases, _ = _cases(16)
        src = C.emit_image_c(cases)
        want = max(int(c.m) * int(c.n) for c in cases)
        assert f"c_dev[{want}]" in src and f"c_ref[{want}]" in src


class TestAScreeningRunIsNotACertification:
    """The in-image reference costs more on RTL than the kernel it checks, so it can be skipped -- and a
    run that skipped it must not be able to report a pass."""

    def _screened(self, *, digest_ok: bool = True):
        cases, _ = _cases(16)
        exp = C.expected_digests(cases)
        lines = ["TILE 16"]
        for c in cases:
            d = exp[c.name]["digest"] + (0 if digest_ok else 1)
            # -1 is the image's "not computed" sentinel for the in-image comparison.
            lines.append(f"CASE {c.name} {c.m} {c.n} {c.k} -1 -1 {d & ((1 << 64) - 1):#018x}")
        lines.append("DONE")
        return cases, C.verdict("\n".join(lines), cases, config="T", tile_edge=16, uses_unit=True)

    def test_a_skipped_reference_is_recorded_as_not_checked(self):
        _, rep = self._screened()
        assert all(not r.in_image_checked for r in rep.results)
        assert all(not r.in_image_agrees for r in rep.results), (
            "a check that did not run must never read as agreement")

    def test_it_is_not_certified_even_when_every_digest_matches(self):
        # The digest is the external oracle and it passed, but nothing in the image cross-checked which
        # buffer was hashed. Calling that certified is the vacuous pass this whole harness exists against.
        _, rep = self._screened(digest_ok=True)
        assert all(r.digest_agrees for r in rep.results)
        assert not rep.certified and not any(r.certified for r in rep.results)

    def test_the_reason_is_stated_per_case(self):
        _, rep = self._screened()
        assert all("screened, not certified" in r.note for r in rep.results)

    def test_a_screening_run_still_catches_a_wrong_result(self):
        # Weaker is not useless: the host digest is what makes it worth running at all.
        _, rep = self._screened(digest_ok=False)
        assert all(not r.digest_agrees for r in rep.results)

    def test_a_full_run_is_marked_checked(self):
        cases, _ = _cases(16)
        exp = C.expected_digests(cases)
        rep = C.verdict(_console(cases, exp, tile=16), cases, config="T", tile_edge=16, uses_unit=True)
        assert all(r.in_image_checked for r in rep.results) and rep.certified

    def test_the_flag_is_visible_in_the_serialised_report(self):
        _, rep = self._screened()
        assert all(r["in_image_checked"] is False for r in rep.to_dict()["results"])


class TestOperandPanelsMustBeAligned:
    """The constraint that made a correct kernel look non-deterministic."""

    def test_the_alignment_is_derived_from_the_datapath_width(self, tmp_path):
        cfg = tmp_path / "c.scala"
        cfg.write_text("class C extends Config(\n  new a.WithShuttleVectorUnit(256, 128, P) ++\n"
                       "  new Base)\n", encoding="utf-8")
        mixin = tmp_path / "m.scala"
        mixin.write_text("class WithShuttleVectorUnit(\n  vLen: Int = 128,\n  dLen: Int = 64) "
                         "extends Config((s, h, u) => { })\n", encoding="utf-8")
        # dLen bits / 8 = bytes per vector-load beat.
        assert C.operand_alignment_for_config("C", config_scala=cfg, mixin_scala=[mixin]) == 16

    def test_a_narrower_datapath_needs_less(self, tmp_path):
        cfg = tmp_path / "c.scala"
        cfg.write_text("class C extends Config(new a.WithShuttleVectorUnit(128, 64, P) ++ new Base)",
                       encoding="utf-8")
        mixin = tmp_path / "m.scala"
        mixin.write_text("class WithShuttleVectorUnit(\n  vLen: Int = 128,\n  dLen: Int = 64) "
                         "extends Config((s, h, u) => { })\n", encoding="utf-8")
        assert C.operand_alignment_for_config("C", config_scala=cfg, mixin_scala=[mixin]) == 8

    def test_an_ungroundable_datapath_width_raises(self, tmp_path):
        # An under-aligned panel returns WRONG DATA rather than failing, so guessing is not an option.
        cfg = tmp_path / "c.scala"
        cfg.write_text("class C extends Config(new Whatever(1) ++ new Base)", encoding="utf-8")
        with pytest.raises(ValueError, match="operand alignment"):
            C.operand_alignment_for_config("C", config_scala=cfg, mixin_scala=[])

    def test_every_operand_array_carries_the_alignment(self):
        cases, _ = _cases(16)
        src = C.emit_image_c(cases, operand_align=16)
        # One attribute per embedded operand array (at_, b_, any bias_, any mult_).
        n_arrays = (2 * len(cases) + sum(1 for c in cases if c.bias)
                    + sum(1 for c in cases if c.requant))
        assert src.count("__attribute__((aligned(16)))") == n_arrays

    def test_the_alignment_is_absent_unless_asked_for(self):
        # It has to be derived and passed in; defaulting to a number would be the guess this avoids.
        cases, _ = _cases(16)
        assert "aligned" not in C.emit_image_c(cases)

    def test_the_answer_key_property_survives_alignment(self):
        # The attribute must not accidentally introduce an int32 array that is not a declared input.
        cases, _ = _cases(16)
        src = C.emit_image_c(cases, operand_align=16)
        declared = sum(1 for c in cases if c.bias) + sum(1 for c in cases if c.requant)
        assert src.count("static const int32_t") == declared


class TestTheResultIsAttributableToAHardwareRevision:
    """A certification whose hardware revision nobody recorded is a number that cannot be attributed."""

    def _prov(self, **kw):
        base = {"merlin": {"commit": "m" * 40, "dirty_files": 0},
                "source_digest": "s" * 64,
                "hardware_pins": {"unit": {"pin": "unit", "ok": True, "drift": [],
                                           "missing_paths": [], "forbidden_present": [],
                                           "observed": {"commit": "e" * 40}}},
                "all_pins_ok": True}
        base.update(kw)
        return base

    def _judge(self, prov, *, stamp=None, console_stamp=None):
        cases, _ = _cases(16)
        exp = C.expected_digests(cases)
        body = _console(cases, exp, tile=16)
        if console_stamp is not None:
            body = f"PROV {console_stamp}\n" + body
        return C.verdict(body, cases, config="T", tile_edge=16, uses_unit=True,
                         provenance=prov, expected_stamp=stamp)

    def test_the_provenance_is_carried_into_the_report(self):
        rep = self._judge(self._prov())
        assert rep.certified
        assert rep.to_dict()["provenance"]["hardware_pins"]["unit"]["ok"] is True

    def test_a_missing_required_path_blocks_certification(self):
        # The checkout does not contain the hardware the result claims to be about.
        prov = self._prov(hardware_pins={"unit": {"ok": False, "drift": [],
                                                  "missing_paths": ["src/TheUnit.scala"],
                                                  "forbidden_present": [], "observed": {}}})
        rep = self._judge(prov)
        assert not rep.certified and any("does not" in g and "contain" in g for g in rep.gaps)

    def test_a_forbidden_path_blocks_certification(self):
        # This is the tapeout case: the revision claims not to have the unit, but it does.
        prov = self._prov(hardware_pins={"unit": {"ok": False, "drift": [],
                                                  "missing_paths": [],
                                                  "forbidden_present": ["src/TheUnit.scala"],
                                                  "observed": {}}})
        rep = self._judge(prov)
        assert not rep.certified and any("not the revision it claims" in g for g in rep.gaps)

    def test_a_wrong_commit_blocks_certification(self):
        prov = self._prov(hardware_pins={"unit": {"ok": False, "drift": ["commit is abc but the pin "
                                                                        "declares def"],
                                                  "missing_paths": [], "forbidden_present": [],
                                                  "observed": {}}})
        rep = self._judge(prov)
        assert not rep.certified and any("commit is" in g for g in rep.gaps)

    def test_a_dirty_tree_is_recorded_but_does_not_block(self):
        # source_digest already pins the bytes that were read, so an unrelated edit elsewhere in the
        # checkout is recorded rather than treated as disqualifying.
        prov = self._prov(hardware_pins={"unit": {"ok": False,
                                                  "drift": ["8 uncommitted change(s): ..."],
                                                  "missing_paths": [], "forbidden_present": [],
                                                  "observed": {}}})
        rep = self._judge(prov)
        assert rep.certified
        assert rep.to_dict()["provenance"]["hardware_pins"]["unit"]["drift"]

    def test_a_stale_binary_is_caught_by_its_own_stamp(self):
        # The image prints what it was generated from; if that disagrees with this build, an old ELF ran.
        rep = self._judge(self._prov(), stamp="unit=aaaa src=bbbb", console_stamp="unit=zzzz src=yyyy")
        assert not rep.certified and any("stale binary ran" in g for g in rep.gaps)

    def test_a_matching_stamp_certifies(self):
        rep = self._judge(self._prov(), stamp="unit=aaaa src=bbbb", console_stamp="unit=aaaa src=bbbb")
        assert rep.certified

    def test_an_image_with_no_stamp_cannot_be_attributed(self):
        rep = self._judge(self._prov(), stamp="unit=aaaa src=bbbb", console_stamp=None)
        assert not rep.certified and any("no provenance stamp" in g for g in rep.gaps)

    def test_the_stamp_is_compact_and_names_the_revision(self):
        got = C.provenance_stamp(self._prov())
        assert "unit=" + "e" * 12 in got and "src=" + "s" * 12 in got
        assert len(got) < 200, "it travels through a bare-metal console one character at a time"


class TestTheRunAlsoMeasures:
    """A certification run reports what the device kernel COST, not only whether it was right.

    This exists because ``routing.MeasuredCost`` declines a unit absent from its throughput table, so
    without a measured figure the router correctly -- but permanently -- refuses to move work onto the unit.
    The certification already executes the real shapes on the real RTL, so it is the cheapest honest place
    to produce that figure.
    """

    def test_the_image_times_the_device_kernel_and_nothing_else(self):
        src = C.emit_image_c(_cases(16)[0])
        before = src.index("csrr %0, mcycle")
        call = src.index("opu_gemm_i8(c_dev", before)
        after = src.index("csrr %0, mcycle", call)
        # The reference must be OUTSIDE the bracket: it is a scalar triple loop costing far more than the
        # kernel it checks, so including it would report a number about the harness.
        assert "opu_gemm_i8_ref" not in src[before:after]
        # And so must the buffer zeroing.
        assert "c_dev[i] = 0" not in src[before:after]

    def test_the_cycles_line_is_its_own_line(self):
        # Widening the CASE line would make every existing reader mis-split it; an unknown line is ignored.
        src = C.emit_image_c(_cases(16)[0])
        assert 'htif_puts("CYCLES ")' in src

    def test_the_parser_reads_the_measurement(self):
        got = C.parse_console("TILE 16\nCASE a 4 4 4 0 -1 0x1\nCYCLES a 4242\nDONE\n")
        assert got["cycles"] == {"a": 4242}
        assert got["malformed"] == ()

    def test_a_malformed_cycles_line_is_collected_not_dropped(self):
        got = C.parse_console("TILE 16\nCYCLES only-a-name\nDONE\n")
        assert got["cycles"] == {} and got["malformed"]

    def test_throughput_is_derived_per_shape(self):
        # Per shape, not per unit: a tiled unit's throughput depends on how well the shape fills the tile,
        # so one headline figure would flatter the narrow shapes and understate the wide ones -- the exact
        # error that made a crude cost model route nearly every contraction onto the matrix unit.
        wide = C.CaseResult(name="w", m=32, n=32, k=32, ran=True, in_image_agrees=True,
                            digest_agrees=True, cycles=1024)
        narrow = C.CaseResult(name="n", m=32, n=1, k=32, ran=True, in_image_agrees=True,
                              digest_agrees=True, cycles=1024)
        assert wide.macs_per_cycle > narrow.macs_per_cycle
        assert wide.macs == 32 * 32 * 32

    def test_an_unreported_measurement_is_absent_rather_than_zero(self):
        # A zero cycle count would compute an infinite throughput and a unit that looks infinitely fast
        # would win every routing decision.
        r = C.CaseResult(name="x", m=4, n=4, k=4, ran=True, in_image_agrees=True, digest_agrees=True)
        assert r.cycles == -1 and r.macs_per_cycle is None

    def test_the_measurement_never_changes_the_verdict(self):
        # A slow kernel is still a correct kernel. Conflating the two is how a performance regression gets
        # recorded as a numerical failure.
        slow = C.CaseResult(name="x", m=4, n=4, k=4, ran=True, in_image_agrees=True,
                            digest_agrees=True, cycles=10**9)
        fast = C.CaseResult(name="x", m=4, n=4, k=4, ran=True, in_image_agrees=True,
                            digest_agrees=True, cycles=1)
        assert slow.certified and fast.certified

    def test_the_report_carries_the_measurement(self):
        cases = _cases(16)[0][:1]
        console = (f"TILE 16\nCASE {cases[0].name} {cases[0].m} {cases[0].n} {cases[0].k} 0 -1 "
                   f"{C.expected_digests(cases)[cases[0].name]['digest']:#x}\n"
                   f"CYCLES {cases[0].name} 777\nDONE\n")
        rep = C.verdict(console, cases, config="T", tile_edge=16, uses_unit=True)
        row = rep.to_dict()["results"][0]
        assert row["cycles"] == 777 and row["macs_per_cycle"] is not None


class TestTheImageIsVectorLengthAgnostic:
    """The certification image must not contain compiler-generated vector code.

    A `zvl<N>b` declaration makes the compiler address scalable-vector spill slots from `vlenb` READ AT RUN
    TIME, so a build and a simulator that disagree on the vector length place every such slot differently and
    it writes over neighbouring stack objects. That defect is real on the whole-model path. This image is
    immune only because its vector code is hand-written asm with explicit `vsetvli` and the surrounding C is
    scalar — so it is built with a plain `rv64gcv`. That immunity is a property to CHECK, not to assume:
    without this test, the day the image gains vectorised C its numbers would quietly become wrong on any
    configuration whose VLEN differs from the compiler's assumption.
    """

    @pytest.fixture
    def built(self, tmp_path):
        from merlin.common.paths import env as _env
        from merlin.llvmlower import opu_shim as S
        if not _env("MERLIN_CHIPYARD"):
            pytest.skip("needs the hardware checkout ($MERLIN_CHIPYARD)")
        contract = S.load_contract("saturn_opu")
        derived = S.derive_encodings(contract)
        if not derived.ok:
            pytest.skip("encoding derivation unavailable")
        cases, _ = _cases(16)
        return C.build_image(cases[:4], derived.encodings, contract.spec(), tmp_path)

    @staticmethod
    def _instructions(disassembly: str) -> list[str]:
        """Only the disassembled INSTRUCTION lines.

        objdump prints the object's PATH in its header, so searching the whole output matches anything the
        path happens to contain — this test failed against its own tmp directory, named after the test.
        An instruction line is `<hex addr>:\t<encoding>\t<mnemonic> ...`, so require the colon-tab shape.
        """
        out = []
        for line in disassembly.splitlines():
            head, sep, rest = line.partition(":\t")
            if sep and head.strip() and all(c in "0123456789abcdef " for c in head.strip()):
                out.append(rest)
        return out

    def test_it_reads_no_vlenb(self, built):
        import subprocess
        from merlin.runtime.backends import spike as _spike
        od = _spike.gcc_path().with_name("riscv64-unknown-elf-objdump")
        got = subprocess.run([str(od), "-d", str(built)], capture_output=True, text=True)
        if got.returncode != 0:
            pytest.skip("objdump unavailable")
        offenders = [i for i in self._instructions(got.stdout) if "vlenb" in i]
        assert not offenders, (
            f"the image reads vlenb ({offenders[:2]}), so its stack addressing now depends on the vector "
            "length and the build must declare zvl<N>b matching the target")

    def test_it_spills_no_vector_register(self, built):
        import subprocess
        from merlin.runtime.backends import spike as _spike
        od = _spike.gcc_path().with_name("riscv64-unknown-elf-objdump")
        got = subprocess.run([str(od), "-d", str(built)], capture_output=True, text=True)
        if got.returncode != 0:
            pytest.skip("objdump unavailable")
        instrs = self._instructions(got.stdout)
        for mnem in ("vs1r.v", "vs2r.v", "vs4r.v", "vs8r.v", "vl1r.v"):
            hits = [i for i in instrs if mnem in i]
            assert not hits, f"the image spills vector registers ({mnem}): {hits[:2]}"
