"""The cycle-capture taps: what a graded run may say about WHERE its cycles went, and what it may not.

The pipe from an oracle's per-unit timing to a performance term
(``TierResult.timing_observations -> capsule_result.json -> harvest._fine_grained -> Observation``)
was built, tested and left completely empty: across every result file on disk the key appeared zero
times, and the retro-miner recovered exactly one quantity, ``total_cycles``. These tests cover the
taps that fill it, and -- more importantly -- the refusals that keep a filled pipe honest.

The example block is a REAL capture: ``MatmulProgram`` on the atlas program-driven Verilator oracle
(elaborated RTL, 463 cycles), whose per-cycle dump reconciles with these aggregates exactly.
"""
from __future__ import annotations

import json

import pytest

from merlin.perf import observations as OBS

# A faithful subset of one real run's block. mxu0Comp/mxu0Data are the compute engine, lsu/dma the
# movement engines; 65 cycles had >= 2 signals high, 0 had a compute AND a movement signal high, and
# 36 had nothing high at all. The per-cycle CSV of the same run sums to these numbers column by
# column, which is why they are usable as a fixture rather than as a guess.
REAL_BLOCK = {
    "cycles": 463,
    "timing_observations": [
        {"quantity": "busy_cycles.mxu0Comp.in_program", "value": 94, "unit": "cycles",
         "kind": "compute"},
        {"quantity": "busy_cycles.mxu0Data.in_program", "value": 64, "unit": "cycles",
         "kind": "compute"},
        {"quantity": "busy_cycles.xlu.in_program", "value": 0, "unit": "cycles", "kind": "compute"},
        {"quantity": "busy_cycles.lsu.in_program", "value": 136, "unit": "cycles",
         "kind": "movement"},
        {"quantity": "busy_cycles.dma0.in_program", "value": 68, "unit": "cycles",
         "kind": "movement"},
        {"quantity": "busy_cycles.dma1.in_program", "value": 131, "unit": "cycles",
         "kind": "movement"},
        {"quantity": OBS.IDLE_QUANTITY, "value": 36, "unit": "cycles", "kind": "fixed"},
        {"quantity": OBS.OVERLAP_OBSERVED, "value": 65, "unit": "cycles"},
        {"quantity": OBS.OVERLAP_ACROSS_KINDS, "value": 0, "unit": "cycles"},
        {"quantity": OBS.SAMPLED_QUANTITY, "value": 464, "unit": "cycles"},
        {"quantity": "dma_beats.channel0", "value": 1, "unit": "beats", "kind": "movement"},
        {"quantity": "dma_beats.channel1", "value": 63, "unit": "beats", "kind": "movement"},
        {"quantity": "dma_beats.ambiguous", "value": 64, "unit": "beats", "kind": "movement"},
        {"quantity": "alias_collisions.window", "value": 0, "unit": "accesses"},
    ],
    "unmeasured_units": ["scaleRegs", "dbg0", "dbg1"],
    "partitioned": False,
    "alias_collisions": 0,
}


# --------------------------------------------------------------------------------------------
# Absent is not zero
# --------------------------------------------------------------------------------------------

def test_an_oracle_with_no_timing_capability_emits_nothing_not_zeros():
    """``None`` and an empty block are different answers, and collapsing them turns a missing
    instrument into a measurement of zero."""
    assert OBS.validate_block({"cycles": 400, "outputs": {}}) is None
    empty = OBS.validate_block({"timing_observations": [], "unmeasured_units": [],
                                "partitioned": False, "alias_collisions": 0})
    assert empty is not None and not empty.usable      # reports timing, none of it survived


def test_a_null_value_is_dropped_with_a_reason_never_recorded_as_zero():
    raw = dict(REAL_BLOCK)
    raw["timing_observations"] = [{"quantity": "busy_cycles.mxu1Comp.in_program", "value": None,
                                   "unit": "cycles", "kind": "compute"}]
    block = OBS.validate_block(raw)
    assert block.busy_by_unit() == {}
    assert any("not reported" in r for r in block.refusals)


def test_unmeasured_units_is_required_and_may_not_default_to_empty():
    """A block that does not say which units it failed to read claims a completeness it has not
    earned. The whole block is refused rather than believed."""
    raw = {k: v for k, v in REAL_BLOCK.items() if k != OBS.UNMEASURED_UNITS_KEY}
    block = OBS.validate_block(raw)
    assert block is not None and not block.usable
    assert any(OBS.UNMEASURED_UNITS_KEY in r for r in block.refusals)
    # ... and an explicitly-written empty list IS believable, because a producer wrote it.
    ok = OBS.validate_block({**REAL_BLOCK, OBS.UNMEASURED_UNITS_KEY: []})
    assert ok.usable and ok.unmeasured_units == ()


def test_a_missing_alias_count_is_unknown_not_zero():
    raw = {k: v for k, v in REAL_BLOCK.items() if k != OBS.ALIAS_COLLISIONS_KEY}
    block = OBS.validate_block(raw)
    assert block.alias_collisions is None
    assert any(OBS.ALIAS_COLLISIONS_KEY in r for r in block.refusals)


# --------------------------------------------------------------------------------------------
# A contended number stays contended
# --------------------------------------------------------------------------------------------

def test_an_unnamespaced_per_unit_busy_count_is_refused():
    """``busy_cycles.<unit>`` without ``.in_program`` could be paired with an isolation-probe
    constant of the same name. Those are different measurements, so the spelling is enforced."""
    raw = dict(REAL_BLOCK)
    raw["timing_observations"] = [{"quantity": "busy_cycles.mxu0Comp", "value": 94,
                                   "unit": "cycles", "kind": "compute"}]
    block = OBS.validate_block(raw)
    assert block.busy_by_unit() == {}
    assert any("in_program" in r for r in block.refusals)


def test_promote_still_raises_for_a_per_unit_count():
    """A per-unit busy count from a running program is exactly as contended as a total: the other
    engines were running inside the same window. There is no door from here to a constant."""
    from merlin.perf.harvest import ContendedTermError, promote
    from merlin.perf.term import PerformanceTerm, Provenance, Validity
    term = PerformanceTerm(
        name="busy_cycles.mxu0Comp.in_program", value=94.0, unit="cycles",
        provenance=Provenance(kind="trace_derived", evidence=("a graded capsule run",)),
        validity=Validity(validated_regime="one program, with every other engine of the same core "
                                           "free to run inside the same window"))
    with pytest.raises(ContendedTermError):
        promote(term, experiment="the cleanest per-unit waterfall we have")


# --------------------------------------------------------------------------------------------
# Overlap is joint, and a partition may never speak about it
# --------------------------------------------------------------------------------------------

def test_the_joint_vector_asserts_partitioned_false_explicitly():
    block = OBS.validate_block(REAL_BLOCK)
    assert block.partitioned is False
    assert block.overlap_cycles(across_kinds=False) == 65


def test_an_overlap_entry_from_a_partitioned_source_is_refused():
    """Buckets that partition a timeline report zero overlap by construction. If this vector is ever
    reduced to marginal counts the failure must be loud, not a quiet zero."""
    block = OBS.validate_block({**REAL_BLOCK, OBS.PARTITIONED_KEY: True})
    assert block.overlap_cycles() is None
    assert any(OBS.OVERLAP_OBSERVED in r for r in block.refusals)
    # and a producer that says nothing at all fails closed the same way
    silent = OBS.validate_block({k: v for k, v in REAL_BLOCK.items()
                                 if k != OBS.PARTITIONED_KEY})
    assert silent.partitioned is True and silent.overlap_cycles() is None


def test_composition_operator_resolves_once_it_has_the_joint_vector():
    """``perf.headroom.composition_operator`` has been returning Unavailable for want of exactly this
    argument. With the vector it returns an operator; without it, it still refuses."""
    from merlin.perf.decompose import ResourceKind, Unavailable, activity_from_busy
    from merlin.perf.headroom import Composition, composition_operator

    block = OBS.validate_block(REAL_BLOCK)
    busy = block.busy_by_unit()
    kinds = {u: ResourceKind(k) for u, k in block.kinds().items()}
    src = activity_from_busy("matmul@L4", block.quantity(OBS.SAMPLED_QUANTITY), busy, kinds,
                             partitioned=block.partitioned, provenance="elaborated_rtl")

    assert isinstance(composition_operator([src]), Unavailable)      # no overlap argument: refuses
    got = composition_operator([src], observed_overlap_cycles={"matmul@L4": block.overlap_cycles()})
    assert not isinstance(got, Unavailable), got
    op, eta = got
    # This program never had a compute and a movement engine busy in the same cycle -- MEASURED, not
    # assumed, and the same instrument reports 65 cycles of within-kind overlap, so it is not blind.
    assert op is Composition.SUM and eta == 0.0


# --------------------------------------------------------------------------------------------
# Concurrency and submission
# --------------------------------------------------------------------------------------------

def test_a_record_without_a_concurrency_stamp_is_marked_not_invented():
    """22,845 timing blocks on disk predate the stamp. They are read as unrecorded; they are never
    retro-labelled, because the concurrency they ran at is not recoverable."""
    legacy = {"status": "pass", "timing": {"build_s": 1.0, "sim_active_s": 2.0}}
    assert OBS.concurrency_of(legacy) == OBS.CONCURRENCY_UNRECORDED
    assert OBS.concurrency_of(legacy).startswith("concurrency unrecorded")
    assert OBS.concurrency_of({"concurrency": {"workers": 16}}) == {"workers": 16}


def test_concurrency_stamp_records_the_fan_out_and_refuses_to_assume_one():
    from merlin.targetgen.capsule_runner import concurrency_stamp
    at16 = concurrency_stamp(16)
    assert at16["workers"] == 16 and at16["serial"] is False and at16["nproc"]
    serial = concurrency_stamp(1)
    assert serial["serial"] is True
    unstated = concurrency_stamp(None)
    assert unstated["workers"] is None and unstated["serial"] is None   # not assumed serial


def test_submission_identity_states_the_package_and_names_what_it_lacks(monkeypatch, tmp_path):
    from merlin.targetgen.capsule_runner import submission_identity
    for var in ("MERLIN_SUBMISSION_RUN_ID", "MERLIN_SUBMISSION_ARM", "MERLIN_SUBMISSION_ROUND"):
        monkeypatch.delenv(var, raising=False)
    bare = submission_identity(tmp_path)
    assert bare["package"] == str(tmp_path.resolve())
    assert bare["run_id"] is None and bare["arm"] is None and bare["round"] is None
    monkeypatch.setenv("MERLIN_SUBMISSION_ARM", "arm-4")
    monkeypatch.setenv("MERLIN_SUBMISSION_ROUND", "7")
    stamped = submission_identity(tmp_path, run_id="20260830T000000Z_x")
    assert stamped["arm"] == "arm-4" and stamped["round"] == "7"
    assert set(stamped["stated"]) == {"run_id", "arm", "round"}


def test_tier_result_stays_byte_identical_when_the_new_fields_are_unset():
    from merlin.targetgen.capsule_runner import TierResult
    plain = TierResult("L3", "pass", True, cycles=463).to_dict()
    for key in ("concurrency", "submission", "timing_capability"):
        assert key not in plain
    rich = TierResult("L4", "pass", True, cycles=463,
                      concurrency={"workers": 8}, submission={"package": "/p"},
                      timing_capability={"partitioned": False}).to_dict()
    assert rich["concurrency"] == {"workers": 8}
    assert rich["submission"] == {"package": "/p"}
    assert rich["timing_capability"] == {"partitioned": False}


# --------------------------------------------------------------------------------------------
# End to end: the block survives the whole pipe
# --------------------------------------------------------------------------------------------

def _capsule_result(tmp_path, *, target: str, block: dict) -> "object":
    d = tmp_path / "runs" / "suite" / "AF_matmul"
    d.mkdir(parents=True)
    p = d / "capsule_result.json"
    p.write_text(json.dumps({
        "capsule": "AF_matmul", "status": "pass", "contract_version": "0.1",
        "trace_check": {}, "numeric": {},
        "tiers": {"L4": {"status": "pass", "not_run_is_not_pass": True, "mandatory": True,
                         "cycles": block["cycles"], "derived_from_rtl": True,
                         "cycle_accurate": True, "fidelity": "elaborated_rtl",
                         "evidence": "verilator-rtl_console.log",
                         "concurrency": {"workers": 8, "nproc": 48, "serial": False},
                         "submission": {"package": "/some/submission"},
                         "timing_observations": block["timing_observations"]}},
    }, indent=2))
    return p


def test_harvest_recovers_more_than_total_cycles_once_the_tap_is_open(tmp_path):
    """The retro-miner recovered exactly one quantity over the whole corpus. Given a block it
    recovers the decomposition too -- through the existing ``_fine_grained`` path, unchanged."""
    from merlin.kernels.measurement import MeasurementAuthority
    from merlin.perf.harvest import harvest_capsule_result

    p = _capsule_result(tmp_path, target="t", block=REAL_BLOCK)
    auth = MeasurementAuthority(target="t", citable_tier="rtl", declared=True)
    obs, _refusals = harvest_capsule_result(p, authority=auth, submission="/some/submission")
    quantities = {o.quantity for o in obs}
    assert "total_cycles" in quantities
    assert len(quantities) > 1, "the tap is open but only the total came through"
    assert "busy_cycles.mxu0Comp.in_program" in quantities
    assert OBS.IDLE_QUANTITY in quantities and OBS.OVERLAP_OBSERVED in quantities
    # the submission travels EXPLICITLY -- it is not read off the path
    assert {o.submission for o in obs} == {"/some/submission"}
    # every recovered value carries what else was running while it was taken
    assert all(o.concurrent for o in obs)


# --------------------------------------------------------------------------------------------
# The perf-ledger product
# --------------------------------------------------------------------------------------------

def test_perf_ledger_is_its_own_product_and_names_what_it_refused(tmp_path, monkeypatch):
    """``perf-ledger`` is what ONE graded run measured. It is deliberately a third topic beside
    ``perf-records`` (a designed suite) and ``perf-harvest`` (retro-mined) -- three concerns, three
    products, so a reader never has to guess which kind of evidence a file holds."""
    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen.capsule_runner import suite_for

    target = "atlas"                       # a test may name the target it is about; library code may not
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    runs_root = tmp_path / "grading"
    d = runs_root / "runs" / suite_for(target) / "AF_matmul"
    d.mkdir(parents=True)
    tier = {"status": "pass", "not_run_is_not_pass": True, "mandatory": True,
            "cycles": REAL_BLOCK["cycles"], "derived_from_rtl": True, "cycle_accurate": True,
            "fidelity": "elaborated_rtl", "evidence": "verilator-rtl_console.log",
            "concurrency": {"workers": 8, "nproc": 48, "load_avg": 3.1, "serial": False},
            "submission": {"package": "/some/submission"},
            "timing_observations": REAL_BLOCK["timing_observations"],
            "timing_capability": {"unmeasured_units": ["scaleRegs"], "partitioned": False,
                                  "alias_collisions": 0, "refusals": []}}
    # A second capsule whose oracle reports NO timing at all: it must be named as having no
    # capability, never filled in with zeros.
    blind = {"status": "pass", "not_run_is_not_pass": True, "mandatory": True, "cycles": 1090,
             "derived_from_rtl": False, "fidelity": "functional_model"}
    (d / "capsule_result.json").write_text(json.dumps({
        "capsule": "AF_matmul", "status": "pass", "contract_version": "0.1",
        "trace_check": {}, "numeric": {}, "tiers": {"L4": tier}}))
    d2 = runs_root / "runs" / suite_for(target) / "AF_blind"
    d2.mkdir(parents=True)
    (d2 / "capsule_result.json").write_text(json.dumps({
        "capsule": "AF_blind", "status": "pass", "contract_version": "0.1",
        "trace_check": {}, "numeric": {}, "tiers": {"L2": blind}}))

    results = [{"capsule": "AF_matmul", "status": "pass", "tiers": {"L4": tier}},
               {"capsule": "AF_blind", "status": "pass", "tiers": {"L2": blind}}]
    summary = CG.emit_perf_ledger(results, target=target, runs_root=runs_root)

    from pathlib import Path as _P
    pdir = _P(summary["product"])
    assert pdir.parent.parent.name == target and "perf-ledger" in str(pdir)
    ledger = json.loads((pdir / "ledger.json").read_text())
    refusals = json.loads((pdir / "refusals.json").read_text())
    lines = [json.loads(x) for x in (pdir / "observations.jsonl").read_text().splitlines()]

    # the decomposition survived the whole path, not just the total
    assert {o["quantity"] for o in lines} > {"total_cycles"}
    # the blind oracle is NAMED, not zero-filled
    assert refusals["capsules_with_no_timing_capability"] == ["AF_blind"]
    assert not any(o["workload"] == "AF_blind" and o["quantity"] != "total_cycles" for o in lines)
    # the operator resolves, from the joint vector
    assert ledger["composition_operator"]["resolved"] is True
    assert ledger["composition_operator"]["operator"] == "sum"
    # concurrency travels with the row that was measured, and the blind row is MARKED unrecorded
    rows = {r["capsule"]: r for r in ledger["rows"]}
    assert rows["AF_matmul"]["concurrency"]["workers"] == 8
    assert rows["AF_blind"]["concurrency"] == OBS.CONCURRENCY_UNRECORDED
    # per-cycle detail is a purgeable cache, never a product file
    assert ledger["per_cycle_occupancy"]["purgeable"] is True
    assert not list(pdir.glob("*.csv"))
