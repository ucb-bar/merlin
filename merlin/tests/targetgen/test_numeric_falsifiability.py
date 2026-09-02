"""A capsule that cannot fail is not evidence.

``AF2_softmax`` declared ``atol: 0.25`` against a golden whose whole range is ``[0.0139, 0.1523]``, so a
constant answer passed at every element. While a corpus-wide defect was leaving operands out of DRAM
entirely, that capsule returned the constant ``softmax(0)`` and PASSED -- reading as the one working
elementwise path and masking the defect. A whole-model capstone in the same corpus accepted all-zeros.

The gate is corpus-wide and target-agnostic: every graded capsule must REJECT the answers a kernel can
produce without computing.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_common, numeric_falsifiability as NF

CAPSULES = repo_root() / "merlin/contract/capsules"


def _all_capsules():
    for p in sorted(CAPSULES.rglob("capsule.yaml")):
        try:
            yield p.parent, capsule_common.load_capsule(p.parent)
        except Exception:                                # a corpus-loading bug is another gate's business
            continue


def test_no_capsule_accepts_an_answer_that_computes_nothing():
    offenders = []
    for cd, cap in _all_capsules():
        for r in NF.audit_capsule(cap, cd):
            offenders.append(f"{cd.name}/{r['output']} accepts '{r['answer']}' "
                             f"(atol={r['atol']} vs golden range {r['range'][0]:.4g}..{r['range'][1]:.4g})")
    assert not offenders, (
        "these capsules pass without the kernel computing anything; tighten atol below the "
        "falsifiability ceiling (numeric_falsifiability.max_falsifiable_atol):\n  " + "\n  ".join(offenders))


def test_the_gate_is_not_vacuous_it_catches_a_loose_tolerance():
    """A tolerance wider than the golden's spread must be REPORTED, or the corpus check above proves
    nothing about capsules it happens to visit."""
    cap = {"numeric_policy": {"compare": "tolerance_float", "atol": 10.0, "rtol": 0.0}}
    golden = {"Y0": [[0.1, 0.2, 0.3]]}
    audit = NF.audit_capsule(cap, _FakeDir(golden))
    assert {r["answer"] for r in audit} == {"zeros", "mean", "midrange"}


def test_a_tight_tolerance_is_reported_clean():
    cap = {"numeric_policy": {"compare": "tolerance_float", "atol": 0.001, "rtol": 0.0}}
    assert NF.audit_capsule(cap, _FakeDir({"Y0": [[0.1, 0.2, 0.3]]})) == []


def test_the_derived_ceiling_is_the_actual_boundary():
    """At the ceiling the best constant still fails; just above it, it passes. Both directions, or the
    number is a guess rather than a boundary."""
    exp = np.array([0.01385, 0.05, 0.1523])
    ceil = NF.max_falsifiable_atol(exp, rtol=0.0)
    below = {"numeric_policy": {"compare": "tolerance_float", "atol": ceil * 0.99, "rtol": 0.0}}
    above = {"numeric_policy": {"compare": "tolerance_float", "atol": ceil * 1.01, "rtol": 0.0}}
    g = {"Y0": exp.tolist()}
    assert NF.audit_capsule(below, _FakeDir(g)) == [], "below the ceiling a constant must FAIL"
    assert NF.audit_capsule(above, _FakeDir(g)), "above the ceiling a constant must PASS"


class _FakeDir:
    """Stands in for a capsule dir: `audit_capsule` reads the golden through `capsule_golden.golden`,
    which we short-circuit by monkeypatching in the fixture below."""

    def __init__(self, golden):
        self.golden = golden


@pytest.fixture(autouse=True)
def _golden_from_fake(monkeypatch):
    from merlin.targetgen import capsule_golden
    real = capsule_golden.golden

    def stub(capsule, capsule_dir=None):
        if isinstance(capsule_dir, _FakeDir):
            return capsule_dir.golden
        return real(capsule, capsule_dir)

    monkeypatch.setattr(capsule_golden, "golden", stub)


# --- the derived tolerance -------------------------------------------------------------------------
# A profile declares ONE absolute tolerance for a whole target. That is the right shape for a datapath
# error budget and the wrong shape for a small-magnitude output: measured, a softmax capsule whose golden
# spans 0.0139..0.1523 was graded at `atol: 0.25`, so zeros, the mean and the midrange all passed it.

def test_a_tolerance_below_the_ceiling_is_left_exactly_alone():
    """The rule only fires on a vacuous policy. Tightening a sound one would reject correct submissions."""
    pol = {"compare": "tolerance_float", "atol": 0.001, "rtol": 0.02}
    out, prov = NF.falsifiable_policy(pol, {"Y0": [0.0, 1.0, 2.0]})
    assert out == pol
    assert prov["falsifiable"] is True and "applied_atol" not in prov


def test_a_vacuous_tolerance_is_replaced_by_the_profiles_own_relative_one():
    """Nothing is invented: `rtol` and the golden are both already declared. The substitution only stops
    an absolute budget written for large outputs from swallowing a small one."""
    golden = [0.01385, 0.05, 0.1523]
    pol = {"compare": "tolerance_float", "atol": 0.25, "rtol": 0.02}
    out, prov = NF.falsifiable_policy(pol, {"Y0": golden}, name="AF2")
    assert out["atol"] == pytest.approx(0.02 * 0.1523)
    assert out["atol"] < prov["ceiling_atol"], "the replacement must clear the ceiling it was called for"
    assert prov["declared_atol"] == 0.25
    # and the constants that used to pass no longer do
    import numpy as np
    exp = np.asarray(golden, dtype=np.float64)
    for name, ans in NF.degenerate_answers(exp).items():
        assert not NF._accepts(ans, exp, out["atol"], out["rtol"]), f"{name} still passes"


def test_a_golden_with_no_spread_raises_rather_than_grading_on_a_loosened_tolerance():
    """The defect is then the capsule's stimulus, not its tolerance. A silently loosened grade is how a
    corpus ends up certifying constants."""
    with pytest.raises(NF.UnfalsifiablePolicy):
        NF.falsifiable_policy({"atol": 0.25, "rtol": 0.02}, {"Y0": [1.0, 1.0, 1.0]}, name="flat")


def test_an_exact_integer_policy_is_not_applicable():
    out, prov = NF.falsifiable_policy({"compare": "exact_int"}, {"Y0": [1, 2, 3]})
    assert prov["status"] == "not_applicable" and "atol" not in out


def test_a_ragged_golden_is_skipped_rather_than_crashing_the_pass():
    """A gate that dies finds nothing. One unsizeable output must not take the whole corpus check down."""
    out, prov = NF.falsifiable_policy({"atol": 0.25, "rtol": 0.02},
                                      {"Y0": [[1.0, 2.0], [3.0]], "Y1": [0.0, 1.0]})
    assert prov["status"] == "ok"
