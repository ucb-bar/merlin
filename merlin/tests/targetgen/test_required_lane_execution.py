"""A REQUIRED lane contract, proved on the operator path from what actually executed.

The four conv-epilogue capsules (``GQ0``..``GQ3``) declare ``lanes.require: [<accelerator lane>]``:
they are the only capsules in the corpus asserting that a conv's requant/bias/relu rides the mesh
store path, which is the gap a ResNet-50 capture measured (all 53 convolutions route on-mesh, while
50 quantize / 49 minmax / 16 elementwise regions stay on the host -- 91.68% of all host IR work, and
116 of 119 host refusals contradicted by the target's own derived capability). ``lanes.require`` is
the mechanism that makes such a host fallback a hard failure rather than a tolerated route.

They could not pass at ANY compiler quality. ``lane_report`` is fed by the whole-model path, which
owns a routing plan and a dispatch ledger; the op tier ladder owns neither, so it proved only the
NEGATIVE half (``elf_lanes.lane_report_from_elf``, rung ``LINKED_ELF_EVIDENCE``, deliberately not a
member of ``EXECUTED_LANE_EVIDENCE``) and every grade recorded ``LANE_CONTRACT_NOT_EVALUATED`` beside
clean numerics and four passing tiers.

What the op ladder DOES own is the two halves of the positive claim, and this pins that it credits the
lane on both together and on neither alone:

* an oracle tier executed this program on an instrument IN THIS RUN and passed, and the decoded stream
  carries instructions of the target's own accelerator classes -> the lane is credited, at rung
  ``"execution"``, and the capsule can reach ``pass``;
* no executed tier (skipped, deferred, or a carried verdict whose certificate is not attributed to the
  bytes under grade), or a trace with no accelerator instruction, or no trace at all -> nothing is
  credited, and the capsule stays ``incomplete``. A hollow pass is worse than the refusal it replaces.

A CARRIED verdict is the interesting case. Per-capsule promotion is the designed behaviour in
continuous mode, so most graded capsules carry their expensive tier rather than re-executing it (78 of
103 on one live run), and refusing every carried verdict would make these capsules flap on cache state
instead of on compiler quality. A certificate is keyed on ``tier_cache.execution_identity`` -- a
content digest of the executables, the target and the toolchain pins -- and is never re-hashed, so it
is a statement about exactly those bytes. So it is credited when that identity matches the one
recomputed here from the bytes on disk, and refused when it does not, when either digest is malformed,
and when no identity can be formed at all.

The negative half is untouched throughout: a ``forbid`` is still judged by the artifact scan alone.
"""

from __future__ import annotations

import pytest

from merlin.targetgen import capsule_runner as R
from merlin.targetgen import elf_lanes as EL
from merlin.targetgen.capsule_common import make_run_paths
from merlin.targetgen.target_registry import all_targets

LANE = R._ACCELERATOR_LANE


@pytest.fixture(scope="module")
def ladder_target() -> str:
    """A registered target whose runner config declares tiers that EXECUTE a program.

    Derived over the registry rather than named, so this test pins no target; it asserts (never skips)
    that one exists, because a suite where none did would report success while measuring nothing."""
    have = []
    for target in all_targets():
        try:
            cfg = R._config_for_target(target, "t", "fp32")
        except Exception:  # noqa: BLE001 -- a target whose manifest will not load declares no ladder
            continue
        if cfg.oracle_tiers:
            have.append(target)
    assert have, "no registered target declares an execution tier ladder"
    return have[0]


@pytest.fixture()
def paths(tmp_path, ladder_target):
    return make_run_paths(tmp_path / "runs", "cap", suite="t", target=ladder_target, dtype="fp32", benchmark="cap")


def _capsule(**over) -> dict:
    cap = {"name": "cap", "kind": "layer", "label": "public", "lanes": {"require": [LANE]}}
    cap.update(over)
    return cap


def _trace(n_accelerator: int, n_other: int = 1) -> dict:
    """A decoded trace. An instruction carries a ``funct`` exactly when the target's own decode table
    claimed its opcode, which is what makes it the accelerator's -- so accelerator work is spelled by
    the presence of that field and never by a class name written here."""
    ins = [{"index": i, "class": "COMPUTE", "funct": 3} for i in range(n_accelerator)]
    ins += [{"index": n_accelerator + i, "class": "OTHER", "funct": None} for i in range(n_other)]
    return {"instructions": ins}


def _finalize(paths, capsule, target, *, tiers, decoded_trace, execution_tiers=None, required=("L2",)):
    return R._finalize_capsule_result(
        name="cap",
        capsule=capsule,
        status="pass",
        failure=None,
        tiers=tiers,
        trace_check_res={"status": "skipped", "violations": []},
        numeric={"status": "pass"},
        required=set(required),
        no_oracle=False,
        eff_target=target,
        paths=paths,
        run_id="cap",
        cfg=R._config_for_target(target, "t", "fp32"),
        contract=None,
        decoded_trace=decoded_trace,
        execution_tiers=execution_tiers,
    )


def _passing_l2() -> dict:
    return {"L2": R.TierResult("L2", "pass", True)}


# --- the two halves, in isolation ---------------------------------------------------------------
def test_executed_program_tiers_counts_only_what_ran_here():
    """An execution tier is credited only by a record that says it ran, in this run, and passed. The
    L0/L1 command-buffer interpretation is not in the set at all -- interpreting a buffer we built is
    not running the program."""
    tiers = {
        "L0": R.TierResult("L0", "pass", True),
        "L2": R.TierResult("L2", "pass", True),
        "L3": R.TierResult("L3", "fail", True),
        "L4": R.TierResult("L4", "skipped", True, budget_deferred=True),
        "L5": R.TierResult("L5", "skipped", True, not_applicable=True),
    }
    assert R.executed_program_tiers(tiers, {"L2", "L3", "L4", "L5"}) == ["L2"]
    assert R.executed_program_tiers(tiers, {"L0"}) == ["L0"]  # only if the caller calls L0 an execution tier
    assert R.executed_program_tiers(tiers, ()) == []


#: Two well-formed execution identities. The shape is the one `oracle_schedule.valid_execution_digest`
#: accepts (64 lowercase hex); the values are arbitrary because what is under test is the COMPARISON.
_THESE_BYTES = "a" * 64
_OTHER_BYTES = "b" * 64


def _carried(identity, tier: str = "L2") -> dict:
    return {tier: R.TierResult(tier, "pass", True, carried={"carried": True, "execution_identity": identity})}


def test_a_carried_certificate_counts_when_it_is_about_these_bytes():
    """The certificate is keyed on the content identity of what runs, so a hit IS a statement that these
    exact bytes passed this tier. Refusing it would make the lane verdict a function of cache state:
    per-capsule promotion means most capsules carry their expensive tier rather than re-executing it."""
    assert R.executed_program_tiers(_carried(_THESE_BYTES), {"L2"}, execution_identity=_THESE_BYTES) == ["L2"]
    assert R.carried_verdict_is_attributed(_carried(_THESE_BYTES)["L2"], _THESE_BYTES) is True


@pytest.mark.parametrize(
    "why,earned,graded",
    [
        ("the certificate belongs to different bytes", _OTHER_BYTES, _THESE_BYTES),
        ("no identity could be formed for the bytes under grade", _THESE_BYTES, None),
        ("the certificate records no identity", None, _THESE_BYTES),
        ("the recorded identity is not a digest", "not-a-digest", _THESE_BYTES),
        ("the recomputed identity is not a digest", _THESE_BYTES, "not-a-digest"),
    ],
)
def test_an_unattributable_carried_certificate_counts_for_nothing(why, earned, graded):
    """FAIL CLOSED. Everything the relaxation does NOT cover stays refused: an unattributable certificate
    is the stale-verdict case, and it is the only thing the blanket refusal was actually buying."""
    assert R.executed_program_tiers(_carried(earned), {"L2"}, execution_identity=graded) == [], why
    assert R.carried_verdict_is_attributed(_carried(earned)["L2"], graded) is False, why


def test_a_carried_record_with_no_identity_block_at_all_is_refused():
    bare = {"L2": R.TierResult("L2", "pass", True, carried={"run_id": "older", "instrument": "d"})}
    assert R.executed_program_tiers(bare, {"L2"}, execution_identity=_THESE_BYTES) == []
    assert R.executed_program_tiers(bare, {"L2"}) == []


def test_the_identity_compared_against_is_the_caches_own(monkeypatch, tmp_path):
    """``graded_execution_identity`` must be the SAME function the certificate is keyed on, called on the
    bytes present now -- otherwise the comparison restates the cache's claim instead of checking it."""
    from merlin.targetgen import tier_cache as TC

    seen = {}

    def _fake(*, target, executables, toolchain_shas):
        seen.update(target=target, executables=tuple(executables), shas=toolchain_shas)
        return _THESE_BYTES

    monkeypatch.setattr(TC, "execution_identity", _fake)
    got = R.graded_execution_identity(target="t", generated=tmp_path, shas={"rtl": "x"})
    assert got == _THESE_BYTES and seen["target"] == "t" and seen["shas"] == {"rtl": "x"}


def test_the_accelerator_instruction_count_is_the_decoders_own_claim():
    from merlin.targetgen import trace_check as TCK

    assert TCK.accelerator_instruction_count(_trace(3)) == 3
    assert TCK.accelerator_instruction_count(_trace(0, 4)) == 0
    assert TCK.drives_accelerator(_trace(1)) is True
    assert TCK.drives_accelerator(_trace(0, 2)) is False


def test_neither_half_alone_supplies_lane_execution():
    cap = _capsule()
    both = dict(tiers=_passing_l2(), execution_tiers={"L2"}, decoded_trace=_trace(2))
    # A carried tier reaches the same place a freshly executed one does, but only when attributed.
    attributed = dict(both, tiers=_carried(_THESE_BYTES), execution_identity=_THESE_BYTES)
    assert R.operator_path_lane_execution(cap, **attributed) == {LANE: 2}
    assert R.operator_path_lane_execution(cap, **{**attributed, "tiers": _carried(_OTHER_BYTES)}) is None
    assert R.operator_path_lane_execution(cap, **both) == {LANE: 2}
    assert R.operator_path_lane_execution(cap, **{**both, "tiers": {}}) is None
    assert R.operator_path_lane_execution(cap, **{**both, "decoded_trace": _trace(0, 3)}) is None
    assert R.operator_path_lane_execution(cap, **{**both, "decoded_trace": None}) is None
    # A capsule that does not REQUIRE the lane is never handed a count for it, whatever ran.
    assert R.operator_path_lane_execution({"lanes": {"forbid": [LANE]}}, **both) is None
    assert R.operator_path_lane_execution({}, **both) is None


# --- the verdict ---------------------------------------------------------------------------------
def test_an_executed_tier_and_accelerator_instructions_credit_the_required_lane(paths, ladder_target):
    """THE FIX. Without it this row is ``incomplete`` / ``LANE_CONTRACT_NOT_EVALUATED``, which is what
    every grade of GQ0..GQ3 recorded beside passing numerics and four passing tiers."""
    row = _finalize(
        paths, _capsule(), ladder_target, tiers=_passing_l2(), decoded_trace=_trace(5), execution_tiers={"L2"}
    )
    assert row["status"] == "pass", row.get("failure")
    report = row["lane_report"]
    assert report["evidence"][LANE] in R.EXECUTED_LANE_EVIDENCE
    assert report["observed"] == [LANE] and report["unexercised"] == []
    assert report["execution_evidence"]["lane_counts"] == {LANE: 5}
    # Held to the same bar the refusal used, not waved past it.
    assert EL.unjudged_lanes(report, _capsule()["lanes"]) == []


def test_the_credit_is_derived_from_the_configs_own_ladder_when_none_is_passed(paths, ladder_target):
    """``execution_tiers`` is the runner's live set (the ``tier_sim`` ladder plus injected adapters);
    when a caller supplies none the target's declared ladder answers, so no tier name is written here."""
    cfg = R._config_for_target(ladder_target, "t", "fp32")
    tier = cfg.oracle_tiers[0]
    row = _finalize(
        paths,
        _capsule(),
        ladder_target,
        tiers={tier: R.TierResult(tier, "pass", True)},
        decoded_trace=_trace(2),
        required=(tier,),
    )
    assert row["status"] == "pass", row.get("failure")
    assert row["lane_report"]["evidence"][LANE] in R.EXECUTED_LANE_EVIDENCE


@pytest.mark.parametrize(
    "why,tiers,trace",
    [
        ("no tier executed", {"L2": R.TierResult("L2", "skipped", True)}, _trace(5)),
        ("tier failed", {"L2": R.TierResult("L2", "fail", True)}, _trace(5)),
        (
            "verdict carried, and no identity can be formed from the bytes to attribute it",
            {"L2": R.TierResult("L2", "pass", True, carried={"carried": True, "execution_identity": _THESE_BYTES})},
            _trace(5),
        ),
        ("nothing drove the accelerator", _passing_l2(), _trace(0, 6)),
        ("no trace was decoded", _passing_l2(), None),
    ],
)
def test_a_missing_half_credits_nothing_and_the_capsule_stays_incomplete(paths, ladder_target, why, tiers, trace):
    """SOUNDNESS. Each row removes exactly one half of the claim; the capsule must go back to the honest
    refusal rather than acquire a hollow pass."""
    row = _finalize(paths, _capsule(), ladder_target, tiers=tiers, decoded_trace=trace, execution_tiers={"L2"})
    assert row["status"] == "incomplete", why
    assert row["failure"]["category"] == "LANE_CONTRACT_NOT_EVALUATED", why
    assert row["lane_report"]["evidence"][LANE] not in R.EXECUTED_LANE_EVIDENCE, why
    assert "execution_evidence" not in row["lane_report"], why


def test_the_finalizer_credits_a_carried_tier_attributed_to_these_bytes(paths, ladder_target, monkeypatch):
    """The whole point of the relaxation, at the verdict. In continuous mode a capsule's expensive tier
    is promoted and CARRIED rather than re-executed on most grades, so a rule that refused every carried
    verdict would make this capsule read `incomplete` on the common path and `pass` only on the rare
    grade that happens to re-execute -- a verdict that tracks cache state instead of compiler quality."""
    monkeypatch.setattr(R, "graded_execution_identity", lambda **_kw: _THESE_BYTES)
    row = _finalize(
        paths,
        _capsule(),
        ladder_target,
        tiers=_carried(_THESE_BYTES),
        decoded_trace=_trace(7),
        execution_tiers={"L2"},
    )
    assert row["status"] == "pass", row.get("failure")
    assert row["lane_report"]["evidence"][LANE] in R.EXECUTED_LANE_EVIDENCE


def test_the_finalizer_refuses_a_carried_tier_earned_by_other_bytes(paths, ladder_target, monkeypatch):
    """THE SOUNDNESS ROW for the relaxation. A certificate whose identity is not the identity of the
    bytes under grade proves nothing about them, and the capsule goes back to the honest refusal."""
    monkeypatch.setattr(R, "graded_execution_identity", lambda **_kw: _THESE_BYTES)
    row = _finalize(
        paths,
        _capsule(),
        ladder_target,
        tiers=_carried(_OTHER_BYTES),
        decoded_trace=_trace(7),
        execution_tiers={"L2"},
    )
    assert row["status"] == "incomplete"
    assert row["failure"]["category"] == "LANE_CONTRACT_NOT_EVALUATED"
    assert "execution_evidence" not in row["lane_report"]


def test_the_identity_is_not_computed_when_nothing_was_carried(paths, ladder_target, monkeypatch):
    """Forming the identity hashes every executable. A grade in which every tier ran fresh has nothing
    to attribute, so it must not pay for the hash -- and must still credit the lane."""
    called = []
    monkeypatch.setattr(R, "graded_execution_identity", lambda **kw: called.append(kw) or _THESE_BYTES)
    row = _finalize(
        paths, _capsule(), ladder_target, tiers=_passing_l2(), decoded_trace=_trace(3), execution_tiers={"L2"}
    )
    assert row["status"] == "pass" and called == []


def test_a_routing_plan_still_credits_nothing():
    """The rule the whole ladder is built around, restated where the new evidence enters: a plan is
    INTENT. Measured on one submission, 15 matmuls were assigned to the mesh and 15 fell back to the
    host at run time, so a plan-evidenced lane is never carried and never reaches an executed rung."""
    plan_only = R.lane_report(_capsule(), {LANE: {"matmul": 15}})
    assert plan_only["observed"] == [] and plan_only["evidence"][LANE] == "routing_plan"
    assert R.credit_executed_required_lanes(None, _capsule(), None) is None
    assert EL.unjudged_lanes(plan_only, _capsule()["lanes"]) == [LANE]


def test_the_negative_half_is_untouched(paths, ladder_target):
    """A forbidding capsule is judged exactly as before: the artifact scan owns that verdict, execution
    evidence is never solicited for it, and ``ACCELERATED_A_FORBIDDEN_LANE`` keeps its reach."""
    forbidding = {"name": "cap", "kind": "isa", "label": "public", "lanes": {"forbid": [LANE]}}
    row = _finalize(
        paths, forbidding, ladder_target, tiers=_passing_l2(), decoded_trace=_trace(9), execution_tiers={"L2"}
    )
    report = row["lane_report"]
    assert "execution_evidence" not in report
    # Whatever the scan concluded, a decoded accelerator instruction never turned the FORBIDDEN lane
    # into a satisfied one by way of the new positive rung.
    assert LANE not in (report.get("observed") or [])


def test_a_capsule_with_no_lane_contract_is_untouched(paths, ladder_target):
    row = _finalize(
        paths,
        {"name": "cap", "kind": "isa", "label": "public"},
        ladder_target,
        tiers=_passing_l2(),
        decoded_trace=_trace(4),
        execution_tiers={"L2"},
    )
    assert row["status"] == "pass" and "lane_report" not in row
