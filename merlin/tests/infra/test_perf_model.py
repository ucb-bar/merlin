"""The derived performance model: both ceilings, honest skips, and a measured error.

These pin the properties that make the model citable: it never invents a ceiling, it never builds a
rate on understated work, it ranks against the rate something actually reached, and it reports its
own error instead of assuming it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"))
import perf_model as PM  # noqa: E402


def _result(root: Path, name: str, *, cycles, commands=None, buffer: bool = True) -> None:
    """One capsule run as the functional harness leaves it on disk."""
    d = root / name
    (d / "generated").mkdir(parents=True)
    (d / "capsule_result.json").write_text(
        json.dumps({"tiers": {"L3": {"cycles": cycles, "status": "pass"}}}), encoding="utf-8")
    if buffer:
        (d / "generated" / "command_buffer.json").write_text(
            json.dumps({"tensors": {"A": {"shape": [16, 16], "dtype": "i8"},
                                    "W": {"shape": [16, 16], "dtype": "i8"}},
                        "commands": commands if commands is not None else
                        [{"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "W", "dst": "acc"}}]}),
            encoding="utf-8")


def test_a_measured_point_needs_both_exact_work_and_a_cycle_verdict(tmp_path):
    """A rate built on understated work would flatter every candidate that follows."""
    _result(tmp_path, "priced", cycles=303)
    _result(tmp_path, "no_cycles", cycles=None)
    _result(tmp_path, "no_buffer", cycles=500, buffer=False)
    _result(tmp_path, "no_work", cycles=400, commands=[])

    points, skipped = PM.harvest_measured_points(tmp_path)
    assert [p.capsule for p in points] == ["priced"]
    # A missing cycle verdict is not an anomaly and is not reported; the other two are, with reasons.
    joined = " ".join(skipped)
    assert "no_buffer" in joined and "command buffer" in joined
    assert "no_work" in joined and "lower bound or zero" in joined
    assert "no_cycles" not in joined


def test_the_structural_ceiling_refuses_rather_than_inventing_one(tmp_path):
    """A peak is a hardware fact. With no array to ground it, the honest answer is None."""
    facts_path = repo_root() / "merlin/targets/gemmini/contracts/rtl_facts/facts.json"
    if not facts_path.is_file():
        pytest.skip("RTL facts are absent in this checkout")
    value, basis = PM.structural_ceiling(facts_path, "gemmini")
    declared = json.loads(facts_path.read_text(encoding="utf-8"))["facts"]["arrays"][0]
    assert value == declared["rows"] * declared["cols"] * declared["mac_idiom"]["muls"]
    assert "facts-derived" in basis

    stripped = json.loads(facts_path.read_text(encoding="utf-8"))
    stripped["facts"]["arrays"] = []
    blinded = tmp_path / "facts.json"
    blinded.write_text(json.dumps(stripped), encoding="utf-8")
    value, basis = PM.structural_ceiling(blinded, "gemmini")
    assert value is None and basis


def test_the_achievable_ceiling_is_a_falsified_ceiling_not_a_nameplate():
    """`merlin.perf.roofline` admits only a measured ceiling, so this must be one."""
    points = [PM.MeasuredPoint("slow", 4096, 400, "x"), PM.MeasuredPoint("fast", 8192, 400, "x")]
    peak = PM.achievable_ceiling(points, provenance="unit test")
    assert peak.known and peak.is_ceiling and peak.n_samples == 2
    assert peak.value == pytest.approx(8192 / 400)      # the best rate anything reached
    # Every point must satisfy demand/rate <= its own cycles, which is what makes it a bound.
    for p in points:
        assert p.macs / peak.value <= p.cycles + 1e-9

    empty = PM.achievable_ceiling([], provenance="unit test")
    assert not empty.known and empty.reason


def test_headroom_ranks_the_least_utilised_shape_first():
    """This ordering is the optimisation order, so the worst offender must lead it."""
    points = [PM.MeasuredPoint("at_ceiling", 8192, 400, "x"),
              PM.MeasuredPoint("far_below", 4096, 4000, "x")]
    peak = PM.achievable_ceiling(points, provenance="unit test")
    ranked = PM.rank_headroom(points, achievable=peak, structural=256)
    assert [h.point.capsule for h in ranked] == ["far_below", "at_ceiling"]
    assert ranked[-1].share_of_achievable == pytest.approx(1.0)
    assert ranked[0].factor_to_achievable > 10
    # Both shares are reported, and the structural one is the smaller: it is the unreachable ceiling.
    assert ranked[0].share_of_structural < ranked[0].share_of_achievable


def test_prediction_error_is_reported_not_assumed():
    """A model whose error nobody measured is a model nobody may cite."""
    points = [PM.MeasuredPoint("a", 4096, 4000, "x"), PM.MeasuredPoint("b", 8192, 400, "x")]
    report = PM.prediction_error(points, PM.achievable_ceiling(points, provenance="unit test"))
    assert report["status"] == "measured" and report["n"] == 2
    # The point the ceiling came from is exact; the other is under-predicted, and both are recorded.
    by_capsule = {row["capsule"]: row for row in report["rows"]}
    assert by_capsule["b"]["relative_error"] == pytest.approx(0.0, abs=1e-9)
    assert by_capsule["a"]["relative_error"] < -0.5
    assert report["worst_relative_error"] <= report["mean_relative_error"] <= report["best_relative_error"]

    unavailable = PM.prediction_error(points, PM.achievable_ceiling([], provenance="unit test"))
    assert unavailable["status"] == "unavailable"


# --- overlap falsifier and schedule ordering ------------------------------------------------------

#: The measured gemmini probe, from out/artifacts/perf-bench/gemmini/composition_smoke.json.
_MEASURED_OVERLAP = {"realised_cycles": 15, "available_cycles": 90,
                     "busy_cycles": {"EX": 84, "LD": 53, "ST": 43},
                     "engines": ["EX", "LD", "ST"], "measurement_cycles": 286}


def test_the_overlap_denominator_convention_is_the_counter_module_s_and_is_stated():
    """falsifier and hw_counters disagree at >=3 engines; the choice must be explicit, not silent."""
    observation = PM.overlap_observation("base", _MEASURED_OVERLAP, work="probe")
    assert observation.measured
    # >=2-busy bound (90), not the busiest-pair bound (the second-largest busy, 53).
    assert observation.available_cycles == 90
    assert observation.realised_cycles == 15
    assert observation.eta == pytest.approx(15 / 90)
    assert "hw_counters" in observation.detail
    assert observation.engines == ("EX", "LD", "ST")

    unusable = PM.overlap_observation("base", {"realised_cycles": 15, "available_cycles": 0})
    assert not unusable.measured and unusable.detail


def test_bit_exactness_alone_is_not_evidence_that_a_reordering_bought_anything():
    """On an interlocked machine a reordering is correct by construction, so ACCEPT needs a risen eta."""
    base = PM.overlap_observation("base", _MEASURED_OVERLAP)
    flat = PM.overlap_observation("cand", _MEASURED_OVERLAP)
    verdict = PM.overlap_verdict(base, flat, bit_exact=True, invariants_held=True)
    assert verdict["state"] != "accept"          # eta did not rise
    assert verdict["baseline_eta"] == pytest.approx(verdict["candidate_eta"])

    risen = PM.overlap_observation("cand", {**_MEASURED_OVERLAP, "realised_cycles": 45})
    accepted = PM.overlap_verdict(base, risen, bit_exact=True, invariants_held=True)
    assert accepted["candidate_eta"] > accepted["baseline_eta"]

    # A wrong answer is rejected however much overlap it bought.
    assert PM.overlap_verdict(base, risen, bit_exact=False)["state"] != "accept"
    # An unestablished invariant is undeterminable, never a quiet pass.
    assert PM.overlap_verdict(base, risen, bit_exact=True, invariants_held=None)["state"] != "accept"


def test_schedule_ordering_refuses_on_a_partial_overlap_machine_and_says_why():
    """Measured on gemmini: eta 0.1667 -> PARTIAL -> differential refuses. Pin it so it stays visible."""
    from merlin.perf.decompose import ResourceKind
    from merlin.perf.envelope import Basis, Peak, ResourceDemand, compose, resource_time
    from merlin.perf.headroom import Composition

    peak = Peak.observed_ceiling("compute", [(4096, 303), (32768, 610)], unit="mac",
                                 provenance="test")
    unknown = Peak.unknown("movement", "bytes", "no measured byte ceiling", provenance="test")

    def composed(macs, operator, eta):
        times = [
            resource_time(ResourceDemand("compute", ResourceKind.COMPUTE, macs, "mac",
                                         Basis.MOVED), peak),
            resource_time(ResourceDemand("movement", ResourceKind.MOVEMENT, 1024, "bytes",
                                         Basis.MOVED), unknown),
        ]
        return compose(times, operator=operator, eta=eta)

    demands = {"movement": ResourceDemand("movement", ResourceKind.MOVEMENT, 1024, "bytes",
                                          Basis.MOVED)}
    partial = PM.schedule_ordering(composed(4096, Composition.PARTIAL, 0.1667),
                                   composed(3000, Composition.PARTIAL, 0.1667),
                                   demands_a=demands, demands_b=demands)
    assert partial["usable"] is False
    assert "partial" in partial["reason"].lower()

    # The same call on an additive machine yields an exact ordering -- the tool is sound, the
    # machine is what makes it inapplicable here.
    additive = PM.schedule_ordering(composed(4096, Composition.SUM, 0.0),
                                    composed(3000, Composition.SUM, 0.0),
                                    demands_a=demands, demands_b=demands, label_a="a", label_b="b")
    assert additive["usable"] is True and additive["faster"] == "b"
    assert additive["delta_cycles"] is not None
    assert "movement" in additive["cancelled"]


# --- activity sources: the type the corpus-level analyses were all waiting on ---------------------

_KINDS = {"EX": "compute", "LD": "movement", "ST": "movement"}


def test_counter_activity_is_not_partitioned_so_it_can_witness_overlap():
    """A partitioned source reports zero overlap by construction and is refused as evidence."""
    source = PM.activity_source_from_counters("probe", _MEASURED_OVERLAP, _KINDS)
    assert source.partitioned is False
    assert source.total_cycles == 286
    kinds = {r.name: r.kind.name for r in source.resources}
    assert kinds == {"EX": "COMPUTE", "LD": "MOVEMENT", "ST": "MOVEMENT"}

    # An engine with no derived role becomes OTHER; it is never guessed into compute or movement.
    unknown = PM.activity_source_from_counters("probe", _MEASURED_OVERLAP, {"EX": "compute"})
    assert {r.name: r.kind.name for r in unknown.resources}["LD"] == "OTHER"


def test_every_measured_cycle_lands_in_a_bucket_including_the_residual():
    """A residual is always emitted: unattributed time must be visible, not absorbed."""
    source = PM.activity_source_from_counters("probe", _MEASURED_OVERLAP, _KINDS)
    components = {c["bucket"]: c for c in PM.attribute_gap(source)["components"]}
    assert set(components) >= {"compute", "dma", "stall", "control", "host", "residual"}
    assert components["compute"]["measured_cycles"] == 84          # EX
    assert components["dma"]["measured_cycles"] == 53 + 43         # LD + ST
    assert components["residual"]["measured_cycles"] > 0
    # With no structural envelope supplied the gap is UNKNOWN, never zero.
    assert str(components["compute"]["structural_cycles"]) == "UNKNOWN"


def test_roles_separate_what_can_be_optimised_from_what_only_calibrates():
    """A workload isolating one engine calibrates that term; it is not an optimisation target."""
    compute = PM.activity_source_from_counters("compute", _MEASURED_OVERLAP, _KINDS)
    copy = PM.activity_source_from_counters(
        "copy", {"busy_cycles": {"EX": 0, "LD": 35, "ST": 7}, "measurement_cycles": 129}, _KINDS)
    report = PM.classify_roles([compute, copy])
    assert report["status"] == "classified"
    assert report["by_role"]["OPTIMIZE"] == ["compute"]
    assert report["by_role"]["CALIBRATION"] == ["copy"]
    assert all(row["rule"] for row in report["rows"])              # every verdict carries its rule

    assert PM.classify_roles([])["status"] == "unavailable"


def test_an_oracle_query_too_expensive_for_the_budget_is_refused_before_it_is_spent():
    """The point is to keep a large shape out of the expensive tier without paying to find out."""
    from merlin.perf.oracle_cost import CostSample, ProbeKind, fit_cost_law
    law = fit_cost_law(
        [CostSample(seconds=10.0 + 0.005 * c, cycles=c, words=0, concurrency=1,
                    kind=ProbeKind.CORPUS) for c in (200, 1000, 5000, 20000)],
        substrate="test")
    cheap = PM.oracle_affordability(law, predicted_cycles=600, budget_seconds=300)
    dear = PM.oracle_affordability(law, predicted_cycles=451_584, budget_seconds=300)
    assert cheap["affordable"] is True and dear["affordable"] is False
    # A term nobody isolated makes the estimate a lower bound, and that is reported, not hidden.
    assert dear["is_lower_bound"] is True and "per_word" in dear["excluded_terms"]

    assert PM.oracle_affordability(None, predicted_cycles=1, budget_seconds=1)["status"] == "undeterminable"


# --- schedulability -------------------------------------------------------------------------------

def _facts_path():
    p = repo_root() / "merlin/targets/gemmini/contracts/rtl_facts/facts.json"
    if not p.is_file():
        pytest.skip("RTL facts are absent in this checkout")
    return p


def test_the_machine_budget_is_derived_and_unpublished_limits_stay_none():
    """A limit nobody published must become an UNCHECKED refusal, never a silent pass."""
    budget = PM.machine_budget(_facts_path(), "gemmini")
    declared = json.loads(_facts_path().read_text(encoding="utf-8"))["facts"]["arrays"][0]
    assert (budget.tile_rows, budget.tile_cols) == (declared["rows"], declared["cols"])
    assert budget.operand_bytes == 1 and budget.accum_bytes == 4     # i8 operands, i32 accumulator
    assert budget.dram_window is None and budget.imem_words is None
    assert "rtl facts" in budget.provenance

    report = PM.preflight_shape("probe", m=16, k=128, n=16, budget=budget)
    codes = {r["code"] for r in report["refusals"]}
    assert {"dram_window_unchecked", "imem_unchecked"} <= codes
    assert report["ok"] is False        # could-not-check is not a pass


def test_a_rate_needs_two_distinct_pass_counts_before_it_is_a_rate():
    """One point cannot separate a slope from an intercept, and the basis must say so."""
    budget = PM.machine_budget(_facts_path(), "gemmini")
    one = PM.tile_pass_rate([PM.MeasuredPoint("a", 4096, 303, "x")], budget=budget)
    assert one.n_points == 1 and "EXTRAPOLATION" in one.basis.name

    many = PM.tile_pass_rate([PM.MeasuredPoint("a", 4096, 303, "x"),
                              PM.MeasuredPoint("b", 32768, 610, "x"),
                              PM.MeasuredPoint("c", 65536, 1112, "x")], budget=budget)
    assert many.basis.name == "FITTED" and many.per_tile_pass > 0

    none = PM.tile_pass_rate([], budget=budget)
    assert none.basis.name == "UNKNOWN"


def test_projected_cycles_feed_the_affordability_gate():
    """The point of projecting a shape is to refuse the expensive tier before paying for it."""
    from merlin.perf.oracle_cost import CostSample, ProbeKind, fit_cost_law
    budget = PM.machine_budget(_facts_path(), "gemmini")
    rate = PM.tile_pass_rate([PM.MeasuredPoint("a", 4096, 303, "x"),
                              PM.MeasuredPoint("b", 2_101_248, 28_118, "x")], budget=budget)
    law = fit_cost_law([CostSample(seconds=12.9 + 0.0046 * c, cycles=c, words=0, concurrency=1,
                                   kind=ProbeKind.CORPUS) for c in (200, 1000, 5000, 20000)],
                       substrate="gsim")
    small = PM.preflight_shape("small", m=16, k=128, n=16, budget=budget, rate=rate)
    large = PM.preflight_shape("layer", m=3136, k=576, n=64, budget=budget, rate=rate)
    assert large["tile_passes"] > small["tile_passes"] * 100
    assert large["projected_cycles"] > small["projected_cycles"]

    verdict = PM.oracle_affordability(law, predicted_cycles=large["projected_cycles"],
                                      budget_seconds=300)
    assert verdict["affordable"] is False, verdict["reason"]


def test_counter_availability_never_infers_absence_from_a_failed_lookup():
    """"absent" is a claim about the machine and may only follow reading a real header."""
    report = PM.counter_availability("gemmini")
    assert report["status"] in {"derived", "absent", "unavailable"}
    if report["status"] == "derived":
        assert report["header"] and report["header_sha256"]
    missing = PM.counter_availability("a_target_that_does_not_exist")
    assert missing["status"] == "unavailable" and missing["reason"]


def test_the_capability_report_invokes_every_analysis_and_names_each_refusal():
    """"Is the tooling used?" must be a measurement, not a reading of the import list.

    Every analysis is CALLED. A module built for another archetype refuses in its own words -- a
    command-buffer target ships no ISA definition, and a target with no vector unit has no vector
    term -- and those refusals are results, not gaps.
    """
    report = PM.capability_report("gemmini", rtl_facts_path=_facts_path())
    analyses = report["analyses"]
    assert report["summary"]["analyses"] == len(analyses) >= 15
    assert report["summary"]["derived"] + report["summary"]["unavailable"] == len(analyses)
    # Every entry is one or the other, and an unavailable always says why.
    for name, row in analyses.items():
        assert row["status"] in {"derived", "unavailable"}, name
        assert ("value" in row) if row["status"] == "derived" else bool(row["reason"]), name
    # The archetype boundaries are stated, not silently missing.
    # The ISA now derives from the RoCC decode table, so this no longer refuses on the ISA. It
    # refuses further along, on a probe that needs the device -- which is a real boundary.
    assert analyses["schedule_dependence"]["status"] == "unavailable"
    assert analyses["schedule_dependence"]["reason"]
    assert analyses["vector_term"]["status"] == "unavailable"
    # And the things this target genuinely establishes are established.
    assert analyses["structural_ceiling"]["status"] == "derived"
    assert analyses["tile_geometry"]["status"] == "derived"
    assert "established" in PM.render_capabilities(report)


# --- the ISA a RoCC target derives from its own decoder -------------------------------------------

def test_a_rocc_target_derives_its_isa_from_its_own_decode_table():
    """"ships no ISA definition" meant "we looked in two places", not "the ISA is unknown".

    The RTL facts carry `interfaces.funct_decode_table` -- custom_opcode, legal_funct and names --
    which is the same table `merlin.kernels.decode.rocc` already disassembles against.
    """
    from merlin.kernels.decode.rocc import fields_of, funct_table_for

    model = PM.isa_model_from_rocc_facts("gemmini", _facts_path())
    assert not model.is_empty()
    assert model.inst_width == 32

    shipped = funct_table_for("gemmini")
    assert len(model.by_mnemonic) == len(shipped["names"])

    # Every encoding round-trips through the INDEPENDENT shipped decoder back to the same mnemonic.
    for mnemonic, entry in model.by_mnemonic.items():
        decoded = fields_of(entry["fixed_value"])
        assert decoded["opcode"] == shipped["custom_opcode"] == entry["opcode"]
        assert shipped["names"][str(decoded["funct"])] == mnemonic

    # xd/xs1/xs2 are operand fields, never identity bits: in RoCC they vary between instructions of
    # the same command, and pinning them dropped conformant instructions.
    entry = next(iter(model.by_mnemonic.values()))
    assert {"xd", "xs1", "xs2", "rs1", "rs2", "rd"} <= set(entry["fields"])
    for bit in (12, 13, 14):
        assert not (entry["fixed_mask"] >> bit) & 1


def test_a_bundle_with_no_decode_table_yields_an_empty_model(tmp_path):
    """No table means no encoding may be assumed; the caller must no-op, not guess."""
    blank = tmp_path / "facts.json"
    blank.write_text(json.dumps({"facts": {"interfaces": []}}), encoding="utf-8")
    assert PM.isa_model_from_rocc_facts("gemmini", blank).is_empty()


def _composed(macs, moved, operator, eta):
    from merlin.perf.decompose import ResourceKind
    from merlin.perf.envelope import Basis, Peak, ResourceDemand, compose, resource_time
    peak = Peak.observed_ceiling("compute", [(4096, 303), (32768, 610)], unit="mac", provenance="t")
    unknown = Peak.unknown("movement", "bytes", "no measured byte ceiling", provenance="t")
    return compose([
        resource_time(ResourceDemand("compute", ResourceKind.COMPUTE, macs, "mac", Basis.MOVED), peak),
        resource_time(ResourceDemand("movement", ResourceKind.MOVEMENT, moved, "bytes", Basis.MOVED),
                      unknown)], operator=operator, eta=eta)


def _demand(moved):
    from merlin.perf.decompose import ResourceKind
    from merlin.perf.envelope import Basis, ResourceDemand
    return {"movement": ResourceDemand("movement", ResourceKind.MOVEMENT, moved, "bytes", Basis.MOVED)}


def test_a_refusal_names_the_resource_that_broke_it():
    """A verdict the reader must take on faith is not evidence. The per-resource proof is the basis."""
    from merlin.perf.headroom import Composition

    # Same unresolved resource, DIFFERENT work asked of it -> the unknown cannot cancel.
    report = PM.schedule_ordering(_composed(4096, 1024, Composition.SUM, 0.0),
                                  _composed(3000, 4096, Composition.SUM, 0.0),
                                  demands_a=_demand(1024), demands_b=_demand(4096))
    assert report["usable"] is False
    assert "movement" in report["comparable_reason"]
    proof = {row["resource"]: row for row in report["cancellation_proof"]}
    assert proof["movement"]["cancels"] is False
    assert proof["movement"]["demand_a"] != proof["movement"]["demand_b"]


def test_on_a_partial_machine_the_evidence_cancels_and_only_the_operator_refuses():
    """The useful diagnostic: nothing is missing here -- the machine's composition is the blocker."""
    from merlin.perf.headroom import Composition

    report = PM.schedule_ordering(_composed(4096, 1024, Composition.PARTIAL, 0.1667),
                                  _composed(3000, 1024, Composition.PARTIAL, 0.1667),
                                  demands_a=_demand(1024), demands_b=_demand(1024),
                                  label_a="cand", label_b="base")
    assert report["usable"] is False
    assert "partial" in report["comparable_reason"].lower()
    # The evidence WOULD have cancelled; the operator is the sole obstacle.
    assert all(row["cancels"] for row in report["cancellation_proof"])
    assert report["cand"]["operator"] == "PARTIAL"
    assert report["cand"]["eta"] == pytest.approx(0.1667)
    assert report["cand"]["unresolved"] == ["movement"]


def test_an_additive_machine_gives_an_exact_delta_and_shows_what_dropped_out():
    from merlin.perf.headroom import Composition

    report = PM.schedule_ordering(_composed(4096, 1024, Composition.SUM, 0.0),
                                  _composed(3000, 1024, Composition.SUM, 0.0),
                                  demands_a=_demand(1024), demands_b=_demand(1024),
                                  label_a="cand", label_b="base")
    assert report["usable"] is True and report["basis"] == "exact"
    assert report["faster"] == "base" and report["delta_cycles"] < 0
    assert report["cancelled"] == ["movement"]


def test_ranking_keeps_a_candidate_it_could_not_compare():
    """A candidate excluded for want of evidence is a hole in the search, not an answer about it."""
    from merlin.perf.headroom import Composition

    cands = {"c1": _composed(4096, 1024, Composition.SUM, 0.0),
             "c2": _composed(3000, 1024, Composition.SUM, 0.0),
             "c3": _composed(5000, 1024, Composition.SUM, 0.0)}
    ranked = PM.rank_candidates(cands, demands={k: _demand(1024) for k in cands})
    assert ranked["order"] == ["c2", "c1", "c3"]
    assert ranked["fully_comparable"] is True

    # One candidate asking different work of the unresolved resource cannot be compared -- and stays.
    cands["c4"] = _composed(3500, 9999, Composition.SUM, 0.0)
    demands = {k: _demand(1024) for k in cands}
    demands["c4"] = _demand(9999)
    mixed = PM.rank_candidates(cands, demands=demands)
    assert "c4" in mixed["order"] and mixed["fully_comparable"] is False
    assert mixed["refusals"]

    assert PM.rank_candidates({})["status"] == "unavailable"


def test_the_roofline_names_what_limits_each_shape_not_just_how_far_it_is():
    """A gap is a number; a limiter is a lever. The corpus is overwhelmingly movement-bound."""
    from merlin.perf.headroom import Composition
    from merlin.common.paths import repo_root
    run = (repo_root() / "out/runs/gemmini/capsule-bench/merlin_assisted"
           / "merlincirct_arm4_func_20260902_codex5_evidence_gsim")
    if not run.is_dir():
        pytest.skip("the frozen functional run is absent in this checkout")
    report = PM.empirical_roofline_report(
        run, operand_bytes=1, composition=Composition.PARTIAL,
        composition_eta=0.16666666666666666, composition_provenance="measured counters")
    assert report["status"] == "derived"
    assert report["resolved"] > 0 and report["expected"] >= report["resolved"]
    # Only an achievable ceiling is admissible here -- a nameplate peak cannot enter.
    assert 0 < report["compute_ceiling"] < 256
    assert report["movement_ceiling"] > 0
    limiters = {row["limiter"] for row in report["rows"]}
    assert limiters <= {"compute", "movement"} and limiters
    # Efficiency is a ratio in (0, 1]; a point above its own bound would be a broken derivation.
    for row in report["rows"]:
        assert 0 < row["efficiency"] <= 1.0, row


def test_traffic_is_counted_from_declared_extents_not_from_mnemonics():
    """Nothing here matches an opcode or class name, so another target counts the same way."""
    trace = {"instructions": [
        {"index": 0, "class": "ANYTHING", "decoded": {"rows": 4, "cols": 8}},
        {"index": 1, "class": "OTHER", "decoded": {"rows": 2, "cols": 2}},
        {"index": 2, "class": "NO_EXTENT", "decoded": {}},
        {"index": 3, "class": "MALFORMED", "decoded": {"rows": 0, "cols": 5}},
    ]}
    elements, counted = PM.moved_elements_from_trace(trace)
    assert (elements, counted) == (4 * 8 + 2 * 2, 2)   # zero-extent and extent-less contribute nothing
    demand = PM.traffic_demand(trace, operand_bytes=4)
    assert demand.amount == (4 * 8 + 2 * 2) * 4 and demand.unit == "bytes"
    assert "decoded transfers" in demand.provenance


def test_fill_depth_asks_each_circuit_for_its_own_modules():
    """Atlas and gemmini are different machines; neither may be asked for the other's modules.

    The pass names the outer array WRAPPER and the inner MESH separately, and the discovered array
    fact names the mesh. Mapping it onto the wrapper broke atlas, which reads correctly under the
    defaults; leaving it defaulted asked gemmini for '@SystolicArray', a module its circuit does not
    have, so the refusal read as 'unreadable circuit' instead of 'wrong module'.
    """
    from merlin.perf.handshake import HandshakeUnavailable, measure_fill_depth

    try:
        atlas = measure_fill_depth("atlas")
    except HandshakeUnavailable:
        pytest.skip("the atlas circuit is not reachable in this checkout")
    # Atlas carries an output-valid delay line, so its depth IS statically measurable.
    assert atlas.dim > 0 and atlas.measured_cycles > 0
    assert atlas.law_cycles == atlas.measured_cycles     # the systolic_2d law holds for this design

    # The other design carries no NAMED delay line -- its emitter left "valid" only on the signals
    # each stage samples -- so the depth is recovered by walking the valid path instead. It must
    # still be answered from ITS OWN modules: the evidence names the container its facts declare,
    # never the other target's wrapper. (This asserted a REFUSAL until the walk was added; the
    # property under test is unchanged, it is now satisfied by a measurement rather than an error.)
    other = measure_fill_depth("gemmini", law="systolic_2d")
    assert other.dim > 0 and other.measured_cycles > 0
    assert "SystolicArray" not in other.source, other.source
    assert "Mesh" in other.source, other.source
    # And the two designs must not be forced to agree: the law that holds for atlas is refuted here.
    assert other.law_agrees is False, (other.law_cycles, other.measured_cycles)
    assert other.measured_cycles != atlas.measured_cycles


def test_a_per_unit_analysis_is_asked_only_where_the_rtl_has_that_unit():
    """Reusable by construction: no unit name is written down, so the hardware decides.

    `vector_cycles` prices a NAMED unit and delegates lane discovery to an external pass, which
    refuses with a manifest complaint when the unit is absent -- reading as a broken analysis. The
    applicability question belongs to the target's own derived compute units, and those follow the
    RTL: a design that gains a vector lane gains the unit without anything being re-declared.
    """
    units = dict(PM.compute_units_of("gemmini"))
    assert units and "vector" not in units.values()
    report = PM.vector_term_for("gemmini")
    assert report["status"] == "not_applicable"
    assert "no vector compute unit" in report["reason"]
    assert "systolic_mesh" in report["reason"]          # says what it DOES have
    assert "manifest" not in report["reason"]           # not the foreign error

    atlas_units = dict(PM.compute_units_of("atlas"))
    if "vector" not in atlas_units.values():
        pytest.skip("no target with a derived vector unit in this checkout")
    atlas = PM.vector_term_for("atlas")
    assert atlas["status"] == "derived" and atlas["units"]
    term = next(iter(atlas["terms"].values()))
    assert term.get("complete") is True and "provenance" in term
