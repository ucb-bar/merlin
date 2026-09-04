"""The claim path dispatches on a family's OWN declaration, never on a family name.

The runner used to refuse ``--mode claim`` for anything that was not ``PK``/``PREDICTS``, and both
its preflight and its decision imported ``perf_pk_claim`` by name.  PK is refuted, so that literal
was the single thing standing between a measured result and a verdict for any other family.

These tests pin the replacement property, which is stronger than "PR also works": the module,
function and version that decide a family come out of that family's own
``performance.acceptance.analyzer``, so a family this repo has never heard of is dispatched with no
edit to the runner, and a family that declares no analyzer is REFUSED by name rather than decided by
somebody else's arithmetic.

Every negative case here is paired with a positive one that must actually fire.  A refusal that
happens for the wrong reason -- a mistyped import inside a broad ``except`` making every call answer
"no capability" -- passes a negative test and proves nothing, and this repo has been bitten by
exactly that.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.perf import claim_reach   # the declaration parsing itself is covered in
                                      # test_perf_family_satisfiability.py

SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
CAPSULES = repo_root() / "merlin/contract/capsules/_perf"
PROFILE = repo_root() / "merlin/contract/capsules/profiles/_perf.yaml"
sys.path.insert(0, str(SCRIPTS))

import perf_campaign as PC  # noqa: E402  (experiment-local module under test)
import perf_pk_claim as PK  # noqa: E402
import perf_pr_claim as PRC  # noqa: E402
import perf_reporting as REP  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_dispatch_{name}", SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RR = _load("run_perf_bench")


def _profile() -> dict:
    return yaml.safe_load(PROFILE.read_text())


def _frozen_descriptors(prefix: str) -> list[dict]:
    """The real frozen ``_perf`` capsule descriptors of one family, read-only."""
    return [yaml.safe_load((path / "capsule.yaml").read_text())
            for path in sorted(CAPSULES.iterdir()) if path.name.startswith(prefix)]


def _capsules(descriptors) -> tuple:
    return tuple(SimpleNamespace(family=descriptor["performance"]["family"],
                                 capsule=descriptor["name"], descriptor=descriptor)
                 for descriptor in descriptors)


def _counter_dict() -> dict:
    """A derived-counter set shaped exactly as ``counters_for_target`` hands one over."""
    from merlin.perf.hw_counters import OccupancyCounters

    engines = ("ex", "ld", "st")
    combinations = {frozenset({"ex"}): "G_EX_CYCLES", frozenset({"ld"}): "G_LD_CYCLES",
                    frozenset({"st"}): "G_ST_CYCLES",
                    frozenset({"ex", "ld"}): "G_EX_LD_CYCLES"}
    return OccupancyCounters(prefix="G", engines=engines,
                             by_combination=combinations).to_dict()


# --------------------------------------------------------------------------------------------
# PK: the family the literal used to name must behave exactly as it did.
# --------------------------------------------------------------------------------------------

def test_pk_is_dispatched_through_its_declaration_and_is_unchanged() -> None:
    descriptors = _frozen_descriptors("PK")
    capsules = _capsules(descriptors)
    expected = PC.expected_perf_cells(capsules, 3)

    identity, module = RR._declared_analyzer(capsules)
    assert (identity.module, identity.function, identity.version) == (
        "perf_pk_claim", "analyze_pk_claim", "v1")
    assert module is PK

    preflight = RR._pk_preflight(capsules, 3, expected)
    assert preflight == PK.preflight_pk_claim(descriptors)
    assert RR.measurement_mode("claim", preflight)["claim_launch_status"] == "GO"

    # The boundary PK seals is byte-identical to the literal the runner used to carry.
    assert RR._boundary_record(module, identity, "PK", None) == {
        "module": "perf_pk_claim",
        "identity_bridge": "analyze_pk_claim(frozen_descriptors,sealed_result_rows)",
        "promotion_integration": "integrated",
        "promotion_status": "PENDING",
        "reason": "the predeclared PK analyzer has not yet consumed the sealed result rows",
    }
    assert RR._boundary_record(module, identity, "PK", {"status": "REFUTED"}) == {
        "module": "perf_pk_claim",
        "identity_bridge": "analyze_pk_claim(frozen_descriptors,sealed_result_rows)",
        "promotion_integration": "integrated",
        "promotion_status": "BLOCKED",
        "reason": "the predeclared PK quantitative decision was refuted",
    }


def test_pk_still_refuses_a_replica_count_its_own_declaration_does_not_predeclare() -> None:
    capsules = _capsules(_frozen_descriptors("PK"))
    expected = PC.expected_perf_cells(capsules, 3)
    with pytest.raises(PC.CampaignGateError, match="predeclared 3 replicates"):
        RR._pk_preflight(capsules, 1, expected)


# --------------------------------------------------------------------------------------------
# PR: DIFFERENTIAL, which the old family literal refused outright.
# --------------------------------------------------------------------------------------------

def _pr_descriptors() -> list[dict]:
    """Frozen PR descriptors carrying the acceptance block the PR sweep declares.

    ⚠️ The materialised ``_perf/PR*`` capsules on disk predate the acceptance block that was added
    to the PR sweep in ``_perf.yaml``, so they carry no ``analyzer`` yet and this dispatch refuses
    them fail-closed (see the companion test below).  Injecting the sweep's own block here is what
    those capsules will carry once regenerated -- it is copied from the profile, never written out.
    """
    sweep = {row["id"]: row for row in _profile()["sweeps"]}["PR"]
    acceptance = sweep["base"]["performance"]["acceptance"]
    descriptors = [copy.deepcopy(descriptor) for descriptor in _frozen_descriptors("PR")]
    for descriptor in descriptors:
        descriptor["performance"]["acceptance"] = copy.deepcopy(acceptance)
    return descriptors


def test_the_frozen_pr_capsules_are_refused_by_name_while_they_carry_no_analyzer() -> None:
    """Fail-closed, and the reason names the omission rather than blaming the rows."""
    on_disk = _frozen_descriptors("PR")
    assert on_disk, "the _perf corpus has no PR capsules"
    declared = [claim_reach.has_decision_procedure(d["performance"]) for d in on_disk]
    if all(declared):                      # regenerated since; the injection below is then moot
        pytest.skip("the frozen PR capsules now carry their declared analyzer")
    with pytest.raises(PC.CampaignGateError, match="declares no acceptance.analyzer"):
        RR._declared_analyzer(_capsules(on_disk))


def test_pr_launches_and_reaches_a_verdict_through_its_own_analyzer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    descriptors = _pr_descriptors()
    capsules = _capsules(descriptors)
    expected = PC.expected_perf_cells(capsules, 3)

    identity, module = RR._declared_analyzer(capsules)
    assert (identity.module, identity.function) == ("perf_pr_claim", "analyze_pr_claim")
    assert module is PRC

    # The old line-70 check refused this outright: PR's claim is DIFFERENTIAL, not PREDICTS.
    preflight = RR._pk_preflight(capsules, 3, expected, target="any-target")
    assert (preflight["status"], preflight["family"], preflight["claim"]) == (
        "READY", "PR", "DIFFERENTIAL")
    assert preflight["contract_frozen"] is True
    assert RR.measurement_mode("claim", preflight) == {
        "experiment_mode": "formal_claim", "claim_launch_status": "GO",
        "claim_launch_blocker": None, "claim_preflight": dict(preflight),
    }
    # PR declares a replicate FLOOR, not an exact cohort, and PR's own preflight is what refuses a
    # run below it -- the refusal text is PR's, quoted verbatim by the launch gate.
    thin = RR._pk_preflight(capsules, 1, PC.expected_perf_cells(capsules, 1), target="any-target")
    assert thin["status"] == "REFUSED" and "UNDETERMINABLE" in thin["refusal_reasons"][0]
    with pytest.raises(PC.CampaignGateError, match="PR preflight refused: .*UNDETERMINABLE"):
        RR.measurement_mode("claim", thin)

    monkeypatch.setattr("merlin.perf.hw_counters.counters_for_target",
                        lambda target, **_kw: {"status": "derived", "counters": _counter_dict()})
    decision = RR._claim_decision(capsules, [], replicates=3, target="any-target")

    # The refusal is PR's OWN: PK's decision record carries no `contract_frozen` and never says
    # DIFFERENTIAL, so this cannot be PK's analyzer answering for PR.
    assert (decision["family"], decision["claim"], decision["status"]) == (
        "PR", "DIFFERENTIAL", "REFUSED")
    assert decision["contract_frozen"] is True
    assert "result rows" in decision["refusal_reasons"][0]
    assert decision["method"] == PRC.supported_acceptance()["fit"]

    boundary = RR._boundary_record(module, identity, "PR", decision)
    assert boundary["module"] == "perf_pr_claim"
    assert boundary["identity_bridge"].startswith("analyze_pr_claim(")
    assert boundary["promotion_status"] == "BLOCKED"


def test_pr_refuses_rather_than_deciding_without_the_counters_it_declares() -> None:
    """A fact the analyzer's signature asks for and the run cannot derive fails closed."""
    capsules = _capsules(_pr_descriptors())
    with pytest.raises(PC.CampaignGateError, match="'counters' parameter"):
        RR._claim_decision(capsules, [], replicates=3, target=None)


# --------------------------------------------------------------------------------------------
# A family that declares no analyzer, and a family this repo has never heard of.
# --------------------------------------------------------------------------------------------

def _blocked_family(name: str) -> dict | None:
    for row in _profile().get("blocked_unimplemented") or []:
        if row.get("family") == name:
            return row
    return None


def test_a_declared_family_without_an_analyzer_is_refused_by_name() -> None:
    blocked = _blocked_family("PT")
    assert blocked is not None, "the profile no longer declares the analyzer-less PT family"
    performance = blocked["performance"]
    assert claim_reach.has_decision_procedure(performance) is False
    assert claim_reach.analyzer_identity(performance) is None

    descriptor = {"name": "PT00", "performance": copy.deepcopy(performance)}
    with pytest.raises(PC.CampaignGateError) as refusal:
        RR._declared_analyzer(_capsules([descriptor]))
    message = str(refusal.value)
    assert "'PT'" in message and "declares no acceptance.analyzer" in message

    # Vacuity guard: the SAME descriptor resolves the moment it declares one, so the refusal above
    # is about the missing analyzer and not about something else in this descriptor.
    decidable = copy.deepcopy(descriptor)
    decidable["performance"]["acceptance"]["analyzer"] = "perf_pk_claim.analyze_pk_claim/v1"
    identity, module = RR._declared_analyzer(_capsules([decidable]))
    assert identity.function == "analyze_pk_claim" and module is PK


_NEW_ANALYZER = '''
"""A claim analyzer this repository has never heard of, reached only via its declaration."""
NONCE = "zz-analyzer-was-actually-called"


def preflight_zz_claim(descriptors, *, replicates):
    return {
        "schema_version": 1, "family": "ZZ", "claim": "COUNTERFACTUAL", "status": "READY",
        "declaration": {"replicates": {"minimum_count": 2}}, "cohort": {},
        "replicates": list(replicates),
        "expected_identities": [
            {"family": "ZZ", "capsule": descriptor["name"], "simulator": simulator,
             "replicate": replicate, "tier": tier}
            for descriptor in descriptors
            for replicate in replicates
            for simulator, tier in (("spike", "L2"), ("verilator", "L3"))],
        "refusal_reasons": [],
    }


def analyze_zz_claim(descriptors, results, *, replicates):
    return {"family": "ZZ", "status": "REFUSED", "nonce": NONCE,
            "replicates": list(replicates), "rows": len(results),
            "refusal_reasons": ["ZZ was handed no rows"]}


def decision_boundary(decision):
    return {"module": "unused-overwritten-by-the-resolved-module",
            "identity_bridge": "analyze_zz_claim(frozen_descriptors,sealed_result_rows)",
            "promotion_integration": "integrated", "promotion_status": "BLOCKED",
            "reason": "ZZ published its own boundary"}
'''


@pytest.fixture()
def new_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A family declaring a brand-new analyzer, dispatched with no edit to the runner."""
    (tmp_path / "perf_zz_claim.py").write_text(_NEW_ANALYZER, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "perf_zz_claim", raising=False)
    descriptors = [
        {"name": f"ZZ{index:02d}",
         "performance": {"family": "ZZ", "claim": "COUNTERFACTUAL",
                         "acceptance": {"schema_version": 1,
                                        "analyzer": "perf_zz_claim.analyze_zz_claim/v1"}}}
        for index in range(2)]
    return descriptors, _capsules(descriptors)


def test_a_newly_declared_family_dispatches_with_no_edit_to_the_runner(new_family) -> None:
    descriptors, capsules = new_family
    expected = PC.expected_perf_cells(capsules, 2)

    preflight = RR._pk_preflight(capsules, 2, expected)
    assert preflight["status"] == "READY" and preflight["family"] == "ZZ"
    # The run supplied `replicates` because ZZ's own signature asks for it -- nothing in the runner
    # decides which facts a family needs.
    assert preflight["replicates"] == ["r000", "r001"]
    assert RR.measurement_mode("claim", preflight)["claim_launch_status"] == "GO"

    decision = RR._claim_decision(capsules, [{"row": 1}], replicates=2)
    import perf_zz_claim  # noqa: PLC0415 - imported only after the dispatch created it
    assert decision["nonce"] == perf_zz_claim.NONCE     # the declared function really ran
    assert decision["rows"] == 1 and decision["replicates"] == ["r000", "r001"]

    identity, module = RR._declared_analyzer(capsules)
    boundary = RR._boundary_record(module, identity, "ZZ", decision)
    assert boundary["module"] == "perf_zz_claim"        # resolved module wins over the module's text
    assert boundary["reason"] == "ZZ published its own boundary"


def test_a_declaration_naming_an_absent_or_malformed_analyzer_fails_closed(new_family) -> None:
    descriptors, _capsules_unused = new_family
    absent = copy.deepcopy(descriptors)
    for descriptor in absent:
        descriptor["performance"]["acceptance"]["analyzer"] = "perf_nope_claim.analyze_nope/v1"
    with pytest.raises(PC.CampaignGateError, match="is unavailable"):
        RR._declared_analyzer(_capsules(absent))

    malformed = copy.deepcopy(descriptors)
    for descriptor in malformed:
        descriptor["performance"]["acceptance"]["analyzer"] = "analyze_zz_claim"
    with pytest.raises(PC.CampaignGateError, match="unusable acceptance.analyzer"):
        RR._declared_analyzer(_capsules(malformed))

    mixed = copy.deepcopy(descriptors)
    mixed[0]["performance"]["acceptance"]["analyzer"] = "perf_pk_claim.analyze_pk_claim/v1"
    with pytest.raises(PC.CampaignGateError, match="claim analyzers"):
        RR._declared_analyzer(_capsules(mixed))


def test_every_analyzer_the_profile_declares_actually_resolves() -> None:
    """A declaration that names a module or function nobody wrote is a broken contract."""
    profile = _profile()
    families = [row["base"]["performance"] for row in profile["sweeps"]]
    families += [row["performance"] for row in profile.get("blocked_unimplemented") or []]
    declared = 0
    for performance in families:
        identity = claim_reach.analyzer_identity(performance)
        if identity is None:
            continue
        declared += 1
        module = __import__(identity.module)
        assert callable(getattr(module, identity.function, None)), identity.declared
        entries = [name for name in dir(module) if name.startswith("preflight_")
                   and callable(getattr(module, name))]
        assert len(entries) == 1, f"{identity.module} publishes {entries}"
    assert declared >= 2, "the profile declares fewer analyzers than this test claims to cover"


# --------------------------------------------------------------------------------------------
# The reporting gate reads the boundary module off the sealed declaration.
# --------------------------------------------------------------------------------------------

def _formal_campaign(boundary: dict) -> dict:
    return {
        "status": "GO", "measurement_status": "GO", "refusal": None, "approach": "arm4",
        "experiment_mode": "formal_claim", "claim_launch_status": "GO",
        "claim_launch_blocker": None, "claim_status": "REFUTED",
        "claim_preflight": PK.preflight_pk_claim(_frozen_descriptors("PK")),
        "claim_decision": {"status": "REFUTED"},
        "decision_boundary": boundary,
    }


_BOUNDARY_REFUSAL = "does not name the analyzer the frozen declaration declares"


def test_reporting_resolves_the_formal_boundary_module_from_the_declaration() -> None:
    correct = {"module": "perf_pk_claim",
               "identity_bridge": "analyze_pk_claim(frozen_descriptors,sealed_result_rows)",
               "promotion_integration": "integrated", "promotion_status": "BLOCKED",
               "reason": "the predeclared PK quantitative decision was refuted"}
    # Positive: the boundary check is PASSED, so the campaign fails later and for another reason.
    with pytest.raises(REP.ReportingGateError) as passed:
        REP.validate_campaign(_formal_campaign(correct))
    assert _BOUNDARY_REFUSAL not in str(passed.value)

    for wrong in ({**correct, "module": "perf_pr_claim"},
                  {**correct, "identity_bridge": "analyze_pr_claim(a,b)"}):
        with pytest.raises(REP.ReportingGateError, match=_BOUNDARY_REFUSAL):
            REP.validate_campaign(_formal_campaign(wrong))


def test_reporting_reads_the_analyzer_identity_off_a_sealed_preflight() -> None:
    preflight = PK.preflight_pk_claim(_frozen_descriptors("PK"))
    identity = REP._declared_analyzer_identity(preflight)
    assert (identity.module, identity.function) == ("perf_pk_claim", "analyze_pk_claim")

    with pytest.raises(REP.ReportingGateError, match="declares no acceptance.analyzer"):
        REP._declared_analyzer_identity({"declaration": {"schema_version": 1}})
    with pytest.raises(REP.ReportingGateError, match="unusable analyzer"):
        REP._declared_analyzer_identity({"declaration": {"analyzer": "nope"}})


def _smoke_campaign(module: object) -> dict:
    return {
        "status": "GO", "measurement_status": "GO", "refusal": None, "approach": "arm4",
        "experiment_mode": "measurement_smoke_only", "claim_status": "NOT_ESTABLISHED",
        "claim_launch_status": "NOT_REQUESTED",
        "claim_launch_blocker": PC.SMOKE_CLAIM_NONCLAIM,
        "claim_preflight": None, "claim_decision": None,
        "decision_boundary": {
            "module": module, "identity_bridge": "not_invoked_in_measurement_smoke",
            "promotion_integration": "blocked", "promotion_status": "BLOCKED",
            "reason": PC.SMOKE_CLAIM_NONCLAIM},
    }


@pytest.mark.parametrize("module", ["perf_pk_claim", "perf_pr_claim"])
def test_reporting_accepts_a_smoke_boundary_naming_any_declared_analyzer(module: str) -> None:
    with pytest.raises(REP.ReportingGateError) as passed:
        REP.validate_campaign(_smoke_campaign(module))
    assert "non-claim boundary" not in str(passed.value)


@pytest.mark.parametrize("module", [None, "", 7])
def test_reporting_still_refuses_a_smoke_boundary_naming_no_analyzer(module: object) -> None:
    with pytest.raises(REP.ReportingGateError, match="non-claim boundary"):
        REP.validate_campaign(_smoke_campaign(module))


def test_the_replicate_contract_enforced_is_the_one_the_declaration_states(
        new_family, monkeypatch: pytest.MonkeyPatch) -> None:
    """A floor is enforced as a floor, and a declaration stating neither is refused."""
    _descriptors, capsules = new_family
    # ZZ declares `minimum_count: 2` and its own preflight admits anything, so the floor the runner
    # enforces is read off the declaration rather than assumed.
    assert RR._pk_preflight(capsules, 2, PC.expected_perf_cells(capsules, 2))["status"] == "READY"
    with pytest.raises(PC.CampaignGateError, match="at least the predeclared 2 replicates"):
        RR._pk_preflight(capsules, 1, PC.expected_perf_cells(capsules, 1))

    import perf_zz_claim
    original = perf_zz_claim.preflight_zz_claim

    def silent(descriptors, *, replicates):
        preflight = original(descriptors, replicates=replicates)
        preflight["declaration"] = {"schema_version": 1}
        return preflight

    monkeypatch.setattr(perf_zz_claim, "preflight_zz_claim", silent)
    with pytest.raises(PC.CampaignGateError, match="states no replicate contract"):
        RR._pk_preflight(capsules, 2, PC.expected_perf_cells(capsules, 2))
