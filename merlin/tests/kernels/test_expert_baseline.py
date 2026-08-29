"""attainment_vs_expert is the number every headline claim rests on. These tests are about the ways a
baseline can be the wrong comparand while the arithmetic stays perfectly consistent."""
import pytest

from merlin.mining.baseline import UNRECORDED, ExpertBaseline, attainment, _norm_dtype


def _b(**kw):
    kw.setdefault("wall_ns", 100.0)
    return ExpertBaseline(**kw)


class TestABaselineCarriesItsOwnIdentity:

    def test_a_bare_number_is_usable_but_stamped_unrecorded(self):
        """Existing callers keep working; nothing may mistake the result for a verified comparison."""
        value, problems, recorded = attainment(100.0, 50.0, workload="w", dtype="int8")
        assert value == 2.0 and problems == () and recorded is False

    def test_none_baseline_yields_no_attainment(self):
        assert attainment(None, 50.0) == (None, (), False)

    def test_no_measurement_yields_no_attainment(self):
        assert attainment(_b(), None) == (None, (), False)

    def test_a_bool_is_not_a_wall_time(self):
        assert ExpertBaseline.of(True) is None


class TestTheMismatchThatActuallyHappened:
    """Two recorded int8 runs were scored against their fp32 sibling's wall time -- to the digit --
    while correctly using the int8 expert's disassembly. Both reported beating the expert. int8 is
    faster than fp32 on the same silicon, so what they measured was the baseline's dtype."""

    def test_scoring_int8_against_an_fp32_baseline_is_refused(self):
        b = _b(workload="bitvla_fp32_consistent", dtype="fp32", substrate="k1")
        value, problems, _ = attainment(b, 50.0, workload="bitvla_int8_consistent", dtype="int8")
        assert value is None
        assert any("measured in 'fp32' but this run is 'int8'" in p for p in problems)

    def test_a_different_workload_is_refused(self):
        b = _b(workload="bitvla_fp32_consistent", dtype="fp32")
        value, problems, _ = attainment(b, 50.0, workload="openvla_fp32_consistent", dtype="fp32")
        assert value is None
        assert any("openvla_fp32_consistent" in p for p in problems)

    def test_a_matching_baseline_still_scores(self):
        b = _b(workload="bitvla_fp32_consistent", dtype="fp32")
        value, problems, recorded = attainment(b, 50.0, workload="bitvla_fp32_consistent",
                                               dtype="fp32")
        assert value == 2.0 and problems == () and recorded is True


class TestSpellingIsNotAMismatch:
    """`f32` and `fp32` both appear in the recorded runs. A spelling difference must neither read as a
    dtype mismatch nor silently split one benchmark cell into two."""

    @pytest.mark.parametrize("a,b", [("f32", "fp32"), ("fp16", "f16"), ("F32", "fp32")])
    def test_equivalent_spellings_normalize_together(self, a, b):
        assert _norm_dtype(a) == _norm_dtype(b)

    def test_genuinely_different_formats_stay_different(self):
        assert _norm_dtype("fp32") != _norm_dtype("fp16")
        assert _norm_dtype("int8") != _norm_dtype("fp32")

    def test_a_spelling_difference_does_not_refuse_the_comparison(self):
        b = _b(workload="w", dtype="fp32")
        value, problems, _ = attainment(b, 50.0, workload="w", dtype="f32")
        assert value == 2.0 and problems == ()


class TestAnUnrecordedBaselineCannotBeShownToMismatch:
    """Which is exactly why it cannot be cited: the check that would catch the error has nothing to
    compare against, so the honest output is 'provenance not recorded', not 'verified'."""

    def test_it_reports_no_problems_and_no_provenance(self):
        value, problems, recorded = attainment(100.0, 50.0, workload="anything", dtype="anything")
        assert value == 2.0 and problems == () and recorded is False

    def test_partial_declaration_only_checks_what_was_declared(self):
        b = _b(dtype="fp32")                       # workload deliberately unstated
        value, problems, _ = attainment(b, 50.0, workload="whatever", dtype="fp32")
        assert value == 2.0 and problems == ()

    def test_a_bare_number_is_marked_unrecorded(self):
        assert ExpertBaseline.of(100.0).note == UNRECORDED
        assert ExpertBaseline.of(100.0).provenance_recorded is False
