"""The DERIVED achievable bound: what the machine's own facts allow, and how it can be wrong.

The defect these pin is not an imprecision. ``perf_model.achievable_ceiling`` derives the optimisation
loop's notion of success from the loop's own past output -- ``max(demand/busy)`` over the points that
ran -- so a loop that has only ever emitted poor schedules certifies itself finished at a fraction of
the machine. Measured on one device: **80.01 mac/cycle, 31.3% of the 256 mac/cycle structural peak**,
while a hand-written schedule on that same device and the same cycle-accurate engine reached **96.2%
of structural (~246 mac/cycle)** across six GEMM shapes, digest-checked against an exact integer
oracle.

Those six measurements appear here as a TEST SET and never as an input. They check the instrument:
a bound that sits below what a real schedule already reached is wrong, and a bound that a measurement
EXCEEDS is refuted rather than merely loose. :func:`test_the_answer_key_is_never_an_input` holds that
line explicitly -- the bound must be bit-identical whether or not the measurements are shown to it.

Every test carries a mutation that must make it fail, because a test that cannot fail is a comment.
"""

from __future__ import annotations

import json

from merlin.common.paths import repo_root
from merlin.perf import derived_bound as DB
from merlin.perf.decompose import UNKNOWN, Unavailable
from merlin.perf.envelope import FixedTerm, Peak

DATA = repo_root() / "merlin" / "tests" / "data" / "derived_bound"

#: (M, N, K, measured cycles). The hand-written schedule, GSIM, digest-checked against an exact
#: integer oracle; artifact ``out/artifacts/probes/our_gemm_vs_vendor_fsm_20260920T000000Z/probe.json``.
#: These are the ANSWER KEY. They are compared against, never fed in.
GROUND_TRUTH = (
    (512, 512, 512, 531318),
    (12544, 256, 64, 963270),
    (12544, 64, 256, 809084),
    (3136, 512, 128, 806883),
    (3136, 128, 512, 810586),
    (784, 1024, 256, 812583),
)

#: The aggregate rate the hand-written schedule reached over those six shapes. A derived bound below
#: this is refuted by construction: a real schedule already beat it.
GROUND_TRUTH_MAC_PER_CYCLE = sum(m * n * k for m, n, k, _ in GROUND_TRUTH) / sum(c for *_s, c in GROUND_TRUTH)

#: What the history-calibrated ceiling evaluated to on the same device.
OBSERVED_CEILING_MAC_PER_CYCLE = 80.01


def _facts(name: str) -> dict:
    return json.loads((DATA / f"{name}_facts.json").read_text(encoding="utf-8"))


def _machine(name: str, facts: dict | None = None) -> DB.Machine:
    """A machine from a fixture's facts, with the circuit deliberately not consulted.

    ``measure_fill=False`` keeps these hermetic: the fill/drain depth is read from an elaborated
    circuit that is a host artifact, not a repo one. Its absence only removes an intercept, which
    RAISES the bound, so every assertion below holds a fortiori when the circuit is available.
    """
    return DB.machine_from_facts(name, facts=facts if facts is not None else _facts(name), measure_fill=False)


# --------------------------------------------------------------------------------------------
# the answer key checks the instrument
# --------------------------------------------------------------------------------------------


def test_the_derived_bound_is_not_below_what_a_real_schedule_already_reached():
    """A bound under ~246 mac/cycle on these shapes is WRONG: a hand-written schedule beat it."""
    machine = _machine("device_a")
    for m, n, k, cycles in GROUND_TRUTH:
        gemm = DB.Gemm(m, n, k)
        bound = DB.achievable_bound(gemm, machine)
        assert bound.partial_rate >= GROUND_TRUTH_MAC_PER_CYCLE, (
            f"{gemm.label}: derived bound {bound.partial_rate} sits below the {GROUND_TRUTH_MAC_PER_CYCLE} "
            "mac/cycle a measured schedule reached, so it would cap the compiler below the hardware"
        )
        assert bound.partial_rate >= gemm.macs / cycles

    # MUTATION: a machine whose array is read half as wide cannot bound these shapes -- and the
    # test must notice, rather than the bound quietly shrinking with the facts it was read from.
    halved = _facts("device_a")
    halved["facts"]["arrays"][0]["cols"] = 8
    narrow = _machine("device_a", halved)
    worst = min(DB.achievable_bound(DB.Gemm(m, n, k), narrow).partial_rate for m, n, k, _ in GROUND_TRUTH)
    assert worst < GROUND_TRUTH_MAC_PER_CYCLE, "the halved-array mutation must break the bound this test holds"


def test_the_derived_bound_is_multiples_above_the_history_calibrated_ceiling():
    """The whole point: the loop's own history sat ~3x below what its machine's facts allow."""
    machine = _machine("device_a")
    bound = DB.achievable_bound(DB.Gemm(512, 512, 512), machine)
    assert bound.partial_rate / OBSERVED_CEILING_MAC_PER_CYCLE > 3.0
    # ...and the derived bound is not a fantasy either: the measured schedule is within a few
    # percent of it, which is what distinguishes a bound from a nameplate nobody can approach.
    assert GROUND_TRUTH_MAC_PER_CYCLE / bound.partial_rate > 0.9


def test_the_answer_key_is_never_an_input():
    """The six measurements may falsify the bound; they may not move it."""
    machine = _machine("device_a")
    for m, n, k, cycles in GROUND_TRUTH:
        gemm = DB.Gemm(m, n, k)
        clean = DB.achievable_bound(gemm, machine)
        shown = clean.confront([(gemm.label, cycles)])
        assert shown.partial_rate == clean.partial_rate
        assert shown.terms == clean.terms and shown.cycles == clean.cycles
        # Pretending the schedule was twice as slow must not lower the bound either: the bound is a
        # property of the machine, and a model that tracked its inputs would be the old defect again.
        slower = clean.confront([(gemm.label, cycles * 2)])
        assert slower.partial_rate == clean.partial_rate


# --------------------------------------------------------------------------------------------
# falsifiability
# --------------------------------------------------------------------------------------------


def test_a_measurement_that_beats_the_bound_makes_it_unknown_rather_than_capping():
    """The mirror of ``observed_ceiling``'s rule, and the reason this bound may be believed."""
    # device_c grounds every term, so ``known`` here turns on falsification alone and not on a
    # missing input -- which is what this test is about.
    machine = _machine("device_c")
    gemm = DB.Gemm(1024, 1024, 1024)
    bound = DB.achievable_bound(gemm, machine)
    assert bound.known and not bound.refuted

    impossible = int(gemm.macs / bound.partial_rate) - 1  # one cycle faster than the bound allows
    refuted = bound.confront([("impossible", impossible)])
    assert refuted.refuted and refuted.rate is UNKNOWN
    assert refuted.refuted_by[0][0] == "impossible"
    assert "REFUTED" in refuted.reasons["refuted"]
    # The weaker figure is still published under its own name, so a reader can see WHAT was refuted.
    assert refuted.partial_rate == bound.partial_rate

    # MUTATION: one cycle slower than the bound allows is inside it, and must NOT refute.
    inside = bound.confront([("inside", impossible + 2)])
    assert not inside.refuted and inside.known


def test_the_derived_peak_goes_unknown_when_a_measurement_exceeds_it():
    """The same rule at the :class:`Peak` boundary, which is what the roofline consumes."""
    machine = _machine("device_a")
    peak = DB.falsifiable_peak(
        machine,
        [(f"w{i}", 1 << 20, (1 << 20) // 200) for i in range(4)],
        resource="compute",
        provenance="unit test",
    )
    assert peak.known and peak.is_ceiling and peak.value == 256 and peak.n_samples == 4

    # MUTATION: one sample that retires faster than the geometry allows refutes the geometry.
    beaten = DB.falsifiable_peak(
        machine,
        [("ok", 1 << 20, (1 << 20) // 200), ("beats_it", 1 << 20, (1 << 20) // 300)],
        resource="compute",
        provenance="unit test",
    )
    assert not beaten.known and "REFUTED" in beaten.reason


# --------------------------------------------------------------------------------------------
# admission in the roofline -- the nameplate guard must survive
# --------------------------------------------------------------------------------------------


def _roofline(peak: Peak, receipt_kind: str | None):
    """One roofline point over one COMPUTE demand, with every other input already impeccable.

    Everything but the peak's own evidence is supplied at full strength, so a refusal below can only
    be about the peak -- which is what these assertions are trying to isolate.
    """
    from merlin.perf.decompose import ResourceKind
    from merlin.perf.envelope import Basis, ResourceDemand
    from merlin.perf.headroom import Composition
    from merlin.perf.roofline import EmpiricalObservation, EvidenceReceipt, empirical_roofline

    receipts = {
        "composition": EvidenceReceipt("a" * 64, "rtl_counter_partition", ("overlap-0",)),
        "observation:job": EvidenceReceipt("b" * 64, "rtl_cycle_measurement", ("cycles-job",)),
        "work:job:compute": EvidenceReceipt("c" * 64, "compiler_ir", ("work-job",)),
        "traffic:job:dram": EvidenceReceipt("e" * 64, "physical_counter", ("traffic-job-dram",)),
        "peak:dram": EvidenceReceipt("f" * 64, "calibration_fit", tuple(f"peak-dram-{i}" for i in range(4))),
        "fixed:job": EvidenceReceipt("0" * 64, "calibration_fit", tuple(f"fixed-job-{i}" for i in range(4))),
    }
    if receipt_kind is not None:
        receipts["peak:compute"] = EvidenceReceipt(
            "d" * 64, receipt_kind, tuple(f"peak-compute-{index}" for index in range(6))
        )
    dram = Peak("dram", 16.0, "bytes", evidence_kind="measured", provenance="bus sweep", n_samples=4, is_ceiling=True)
    observation = EmpiricalObservation(
        workload="job",
        cycles=4096,
        work=ResourceDemand("compute", ResourceKind.COMPUTE, 524288.0, "mac", Basis.MOVED, provenance="counter"),
        moved_bytes=(ResourceDemand("dram", ResourceKind.MOVEMENT, 4096.0, "bytes", Basis.MOVED, provenance="bus"),),
        provenance="cycle-accurate measurement receipt",
    )
    return empirical_roofline(
        [observation],
        peaks={"compute": peak, "dram": dram},
        fixed_terms={"job": ()},  # explicitly measured as zero, which is not the same as absent
        evidence_receipts=receipts,
        composition=Composition.SUM,
        composition_eta=0.0,
        composition_provenance="measured zero-overlap trace",
        expected_workloads=("job",),
    )


def _admitted(report) -> bool:
    point = report.points["job"]
    return point.resolved and not point.refusals


def _refusals(report) -> str:
    return " ".join(str(refusal) for refusal in report.points["job"].refusals)


def test_a_nameplate_peak_is_still_refused_and_a_derived_bound_is_admitted():
    """The guard that excludes a nameplate is untouched; what is added is a second, stricter path."""
    machine = _machine("device_a")
    samples = [(f"w{i}", 1 << 20, (1 << 20) // 200) for i in range(6)]

    nameplate = Peak(
        resource="compute",
        value=256.0,
        unit="mac",
        evidence_kind="structural_bound",
        provenance="the datasheet",
        n_samples=6,
        is_ceiling=True,
    )
    refused = _roofline(nameplate, "calibration_fit")
    assert not _admitted(refused), "a nameplate peak must never enter the roofline"
    assert "structural_bound" in _refusals(refused)

    derived = DB.falsifiable_peak(machine, samples, resource="compute", provenance="derived from this target's facts")
    receipt = DB.falsification_receipt(samples, artifact_sha256="e" * 64)
    assert receipt.source_kind == DB.FALSIFICATION_SOURCE_KIND
    assert _admitted(_roofline(derived, DB.FALSIFICATION_SOURCE_KIND)), "a confronted derived bound is admissible"

    # MUTATION 1: the same derived peak with no falsification set is an unrefuted claim only because
    # nobody tried to refute it, so it is refused.
    assert not _admitted(_roofline(derived, None))
    # MUTATION 2: a receipt of the EMPIRICAL kind does not stand in for a falsification set, and a
    # nameplate relabelled with the derived kind still fails the ceiling/sample requirements.
    assert not _admitted(_roofline(derived, "calibration_fit"))
    relabelled = Peak(
        resource="compute",
        value=256.0,
        unit="mac",
        evidence_kind=DB.DERIVED_BOUND_EVIDENCE_KIND,
        provenance="the datasheet",
        n_samples=1,
        is_ceiling=False,
    )
    assert not _admitted(_roofline(relabelled, DB.FALSIFICATION_SOURCE_KIND))


# --------------------------------------------------------------------------------------------
# target-agnosticism: data only
# --------------------------------------------------------------------------------------------


def test_a_second_machine_with_different_facts_gets_a_different_bound_with_no_core_edit():
    """Adding a target is adding a facts document. This test's only input is one.

    ``device_c`` is a synthetic machine nothing in ``merlin/python/merlin`` knows about: wider array,
    two multipliers per element, stores under different names, a narrow transfer path. It must not
    only produce a different number but a different SHAPE of answer -- its DRAM traffic binds where
    device_a's array does -- and it must do so with no code change at all.
    """
    a, c = _machine("device_a"), _machine("device_c")
    assert a.peak_macs_per_cycle == 256 and c.peak_macs_per_cycle == 2048
    assert c.provenance["muls_per_element"].endswith("muls = 2")

    gemm = DB.Gemm(1024, 1024, 1024)
    bound_c = DB.achievable_bound(gemm, c)
    assert bound_c.known, bound_c.reasons
    assert bound_c.limiter == DB.TRAFFIC_TERM, "this machine's narrow bus binds before its array does"
    assert bound_c.partial_rate < c.peak_macs_per_cycle / 4
    assert DB.achievable_bound(gemm, a).limiter == DB.COMPUTE_TERM

    # MUTATION: widen only this machine's transfer path, in the DATA, and the bound moves with it.
    facts = _facts("device_c")
    for datapath in facts["facts"]["datapaths"]:
        if datapath.get("bits"):
            datapath["bits"] = 1024
    widened = DB.achievable_bound(gemm, _machine("device_c", facts))
    assert widened.partial_rate > bound_c.partial_rate
    assert widened.limiter == DB.COMPUTE_TERM


def test_the_per_loop_capacity_is_the_ping_pong_share_not_the_whole_store():
    """The FSM runs several loop contexts at once, so one loop sees ``capacity / ways``.

    Derived from the stores' own bank decomposition -- explicitly where the facts state ``banks``,
    and from ``bytes / (depth x row)`` where they do not -- never by halving out of habit.
    """
    for name, operand_total, result_total in (("device_a", 262144, 65536), ("device_b", 262144, 65536)):
        machine = _machine(name)
        assert machine.ping_pong_ways == 2, name
        assert machine.operand_store_bytes == operand_total // 2
        assert machine.accumulate_store_bytes == result_total // 2
        assert "concurrent" in machine.provenance["ping_pong_ways"]

    # MUTATION: a design whose result store is one bank runs one loop context, so the whole store is
    # one loop's -- the split must follow the facts, not a constant.
    facts = _facts("device_b")
    for memory in facts["facts"]["memories"]:
        memory["banks"] = 1
    single = _machine("device_b", facts)
    assert single.ping_pong_ways == 1
    assert single.accumulate_store_bytes == 65536

    # ...and a larger per-loop capacity admits a tiling that moves less, which is a real consequence
    # and not a bookkeeping change.
    gemm = DB.Gemm(3136, 512, 128)
    assert DB._best_tiling(gemm, single).traffic_bytes < DB._best_tiling(gemm, _machine("device_b")).traffic_bytes


# --------------------------------------------------------------------------------------------
# UNKNOWN is named, never defaulted
# --------------------------------------------------------------------------------------------


def test_an_underivable_input_is_named_and_the_term_it_feeds_drops_out():
    """No default, no nameplate: the refusal says what would settle it, and the bound loosens."""
    machine = _machine("device_a")
    # This device's facts ground its array but not its transfer width, so the traffic term is absent
    # BY NAME and the published rate is the weaker, compute-only one under its own name.
    bound = DB.achievable_bound(DB.Gemm(512, 512, 512), machine)
    assert bound.rate is UNKNOWN and bound.partial_rate > 0
    assert DB.TRAFFIC_TERM in bound.unresolved
    assert "measurement at >=2 transfer sizes" in machine.refusals["dram_bytes_per_cycle"]

    # MUTATION: strip the datapaths and the element widths go with them -- capacities, the ping-pong
    # split and the traffic term all refuse together, and nothing substitutes a width.
    blind = _facts("device_a")
    blind["facts"]["datapaths"] = []
    stripped = _machine("device_a", blind)
    assert stripped.operand_bytes is UNKNOWN and stripped.ping_pong_ways is UNKNOWN
    assert stripped.refusals["operand_bytes"] and stripped.refusals["ping_pong_ways"]
    # The compute term survives, because the geometry did: a partial refusal is not a total one.
    assert stripped.peak_macs_per_cycle == 256

    # MUTATION: remove the array and there is no bound at all, with the reason, rather than a zero.
    empty = _facts("device_a")
    empty["facts"]["arrays"] = []
    nothing = DB.achievable_bound(DB.Gemm(64, 64, 64), _machine("device_a", empty))
    assert nothing.rate is UNKNOWN and DB.COMPUTE_TERM in nothing.unresolved
    assert "fabricated hardware claim" in nothing.reasons[DB.COMPUTE_TERM]


def test_the_fill_drain_delay_line_is_charged_once_and_is_never_a_law():
    """The intercept the emitter did not name: measured from the circuit, or UNKNOWN and looser."""
    machine = _machine("device_a")
    assert machine.fill_drain_cycles is UNKNOWN
    assert isinstance(DB.fill_term(machine, resource="compute"), Unavailable)
    open_bound = DB.bound_for_macs(1 << 20, machine)

    # A machine whose circuit WAS read charges the delay line once, which lowers the bound: the
    # intercept dominates exactly where the tiles are small, which is most of a corpus.
    with_fill = DB.Machine(
        **{
            **{f.name: getattr(machine, f.name) for f in machine.__dataclass_fields__.values()},
            "fill_drain_cycles": 17,
            "provenance": {**machine.provenance, "fill_drain_cycles": "measured valid path"},
        }
    )
    term = DB.fill_term(with_fill, resource="compute")
    assert isinstance(term, FixedTerm) and term.cycles == 17 and term.resource == "compute"
    small = DB.bound_for_macs(1 << 10, with_fill)
    large = DB.bound_for_macs(1 << 24, with_fill)
    assert small.partial_rate < large.partial_rate < 256
    assert DB.bound_for_macs(1 << 20, with_fill).partial_rate < open_bound.partial_rate

    # MUTATION: consult the circuit for a device that has none, and the refusal must name the
    # circuit and REFUSE the closed-form law, which is refuted on at least one design in this repo.
    unreadable = DB.machine_from_facts("device_c", facts=_facts("device_c"), measure_fill=True)
    assert unreadable.fill_drain_cycles is UNKNOWN
    assert "law" in unreadable.refusals["fill_drain_cycles"]
    assert unreadable.provenance.get("fill_drain_cycles") is None
