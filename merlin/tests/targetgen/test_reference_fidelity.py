"""A golden composed around a gap in OUR reference must not be scored as the submission's failure.

Two sub-byte MX flash capsules failed identically in all four arms of a ladder comparison with ~1000
mismatched elements and max relative error in the thousands. The cause was not the compilers: the fused
flash reference exists only for e4m3, so the generator composed those goldens from a DIFFERENT
arithmetic than the datapath executes (a normalized softmax + palette requant, versus the kernel's
unnormalized-P / e4m3-block-requant / 1-over-l finalize). The generator says so in its own comment. No
submission could match them, which also made `all_pass` -- and therefore the loop's only early exit --
unreachable from round 0.

These tests pin the three links in the chain: the reference DECLARES what it can reproduce, the golden
RECORDS which kind it is, and grading ROUTES on the record rather than on a capsule name.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_golden as CG, capsule_runner as CR, mx_flash_ref as MXF


def test_the_reference_declares_the_formats_it_reproduces_faithfully():
    assert MXF.reference_fidelity("fp8_e4m3") == MXF.FIDELITY_FAITHFUL
    for sub_byte in ("fp6_e3m2", "fp4_e2m1"):
        assert MXF.reference_fidelity(sub_byte) == MXF.FIDELITY_APPROXIMATE, (
            f"{sub_byte} has no fused flash reference — bf16_to_e4m3_scaled is written to the e4m3 "
            f"exponent bounds — so a golden for it is a composition, not the datapath's arithmetic")


def test_the_declaration_drives_the_answer_not_a_hardcoded_format_list():
    """Adding a sub-byte reference must flip the verdict by editing ONE declaration."""
    original = MXF.FAITHFUL_FORMATS
    try:
        MXF.FAITHFUL_FORMATS = original + ("fp4_e2m1",)
        assert MXF.reference_fidelity("fp4_e2m1") == MXF.FIDELITY_FAITHFUL
    finally:
        MXF.FAITHFUL_FORMATS = original
    assert MXF.reference_fidelity("fp4_e2m1") == MXF.FIDELITY_APPROXIMATE


def _capsule_dir(tmp_path, fidelity: str | None):
    """A capsule dir whose golden.yaml records (or omits) the generator's fidelity."""
    d = tmp_path / "cap"
    d.mkdir(parents=True, exist_ok=True)
    prov = ["golden_source: mlc_mx_ref_hardware_semantics",
            "oracle_provenance:",
            "  engine: test"]
    if fidelity:
        prov.append(f"  reference_fidelity: {fidelity}")
    prov += ["  grade_policy:", "    compare: tolerance_float", "outputs:", "  Y0: [1.0]", ""]
    (d / "golden.yaml").write_text("\n".join(prov))
    return d


def test_reader_routes_on_what_the_generator_recorded(tmp_path):
    assert CG.golden_is_datapath_faithful({}, _capsule_dir(tmp_path / "a", MXF.FIDELITY_FAITHFUL))
    assert not CG.golden_is_datapath_faithful({}, _capsule_dir(tmp_path / "b",
                                                               MXF.FIDELITY_APPROXIMATE))


def test_a_golden_written_before_this_field_existed_is_graded_exactly_as_before(tmp_path):
    """FAIL-OPEN on absence: the change is additive, so an old corpus grades unchanged."""
    assert CG.golden_is_datapath_faithful({}, _capsule_dir(tmp_path / "c", None))


@pytest.mark.parametrize("fidelity,expect_withheld", [(MXF.FIDELITY_APPROXIMATE, True),
                                                      (MXF.FIDELITY_FAITHFUL, False),
                                                      (None, False)])
def test_the_runner_withholds_only_the_unfaithful_ones(tmp_path, fidelity, expect_withheld):
    d = _capsule_dir(tmp_path / "r", fidelity)
    caps = [{"name": "CAP0", "kind": "model_slice", "label": "public", "__dir__": d}]
    keep, withheld = CR._split_no_reference(caps)
    assert bool(withheld) is expect_withheld
    assert bool(keep) is not expect_withheld
    if expect_withheld:
        assert withheld[0]["status"] == "not_gradeable_no_reference"
        # The reason has to name OUR gap, or a reader charges it to the compiler.
        assert "our own reference" in withheld[0]["failure"]["detail"]


def test_a_withheld_capsule_is_in_neither_numerator_nor_denominator():
    """The status must not land in the graded set — that is the whole point of the routing."""
    results = [{"capsule": "A", "status": "pass"},
               {"capsule": "B", "status": "fail"},
               {"capsule": "C", "status": "not_gradeable_no_reference"}]
    graded = [r for r in results
              if r.get("status") not in ("not_graded", "gated", "screened_only",
                                         "not_gradeable_no_reference")]
    assert [r["capsule"] for r in graded] == ["A", "B"]
    assert sum(1 for r in graded if r["status"] == "pass") == 1
