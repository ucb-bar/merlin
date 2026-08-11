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

    def test_no_int32_array_other_than_a_bias_is_embedded(self):
        cases, _ = _cases(16)
        src = C.emit_image_c(cases)
        n_bias = sum(1 for c in cases if c.bias)
        # Results are int32. Every int32 constant array in the image must be a declared bias.
        assert src.count("static const int32_t") == n_bias

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
    def test_every_selected_case_fits_the_tile_it_was_selected_for(self):
        for tile in (4, 8, 16, 32):
            runnable, _ = _cases(tile)
            assert runnable, f"tile {tile} selected nothing"
            assert all(int(c.m) <= tile and int(c.n) <= tile for c in runnable)

    def test_the_named_regressions_are_in_reach_at_every_tile(self):
        for tile in (4, 8, 16, 32):
            names = {c.name for c in _cases(tile)[0]}
            assert {"narrow_m_1", "narrow_n_1", "narrow_both_1"} <= names

    def test_the_buffers_are_sized_for_the_largest_selected_case(self):
        cases, _ = _cases(16)
        src = C.emit_image_c(cases)
        want = max(int(c.m) * int(c.n) for c in cases)
        assert f"c_dev[{want}]" in src and f"c_ref[{want}]" in src
