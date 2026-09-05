"""The authoring stage dispatches on a family's OWN declaration, never on a family name.

The stage that seals a candidate handoff used to filter capsules by ``family == "PK"`` and call one
analyzer by name.  A family declaring a different analyzer -- ``PR`` does, and so do the affine and
paired families -- could not produce a handoff at all, and one declaring none -- ``PT``'s deliberate
state -- would have been silently skipped instead of refused.  These tests pin what must now be true
together:

* a family that declares an analyzer produces a handoff, and one nobody has written code for
  (``_SYNTHETIC``) does too, which is the only real test of "no edit needed for a new family";
* PK's handoff is unchanged -- asserted against a DIRECT call to PK's own preflight, so the check
  cannot pass by both sides drifting;
* an analyzer-less family is refused with a reason that names the omission;
* every analyzer any shipped family names publishes exactly ONE ``preflight_*`` entry point, which
  is what the reporting gate resolves it by.  Two of the four published zero, and the consequence
  was silent: the differential and law families measured and never reached a verdict.

Every negative here is paired with a mutation showing the positive stops passing when the declared
fact is removed.  This repo has had negative tests pass for the wrong reason -- an unimported name
inside a broad ``except`` once made every call return "no capability" while both negatives were
green -- so a refusal is only credited when its own distinct reason text appears AND the unmutated
input still succeeds in the same test.

RECONCILED against this tree, deliberately, where an earlier draft of this file asserted an API that
does not exist here:

* ``exact_count=2``, not 3.  ``contract/capsules/profiles/_perf.yaml`` freezes PK's cohort at two
  replicates with identities ``r000``/``r001``, and these tests read the REAL frozen descriptors, so
  three was simply a different corpus's number.
* ``PAS.PerformanceCapsule``, not ``perf_campaign``'s: the authoring stage owns that view here.
* the runner-side preflight assertion is dropped: ``run_perf_bench`` publishes no ``_pk_preflight``
  in this tree (nor in any worktree of it), so an assertion about it could only ever fail or be
  deleted; the handoff-vs-preflight equality it wanted is covered against the stage's own sealing
  validator instead, which is the code that exists.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from merlin.common.paths import merlin_dir


_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import perf_agent_stage as PAS  # noqa: E402
import perf_pk_claim as PK  # noqa: E402


_PERF_ROOT = merlin_dir() / "contract/capsules/_perf"
_PROFILE = merlin_dir() / "contract/capsules/profiles/_perf.yaml"
_SHA = "a" * 64


def _capsules_named(prefix: str) -> tuple[PAS.PerformanceCapsule, ...]:
    """The REAL frozen descriptors on disk, not a hand-built cohort."""
    rows = []
    for source in sorted(_PERF_ROOT.iterdir()):
        if not source.is_dir() or not source.name.startswith(prefix):
            continue
        descriptor = yaml.safe_load((source / "capsule.yaml").read_text(encoding="utf-8"))
        rows.append(PAS.PerformanceCapsule(
            str(descriptor["performance"]["family"]), str(descriptor["name"]), source,
            f"_perf/{source.name}", descriptor, _SHA, 3, 300))
    assert rows, f"no frozen capsules named {prefix!r}"
    return tuple(rows)


def _rewritten(capsules, edit) -> tuple[PAS.PerformanceCapsule, ...]:
    rows = []
    for capsule in capsules:
        descriptor = copy.deepcopy(capsule.descriptor)
        edit(descriptor)
        rows.append(PAS.PerformanceCapsule(
            capsule.family, capsule.capsule, capsule.source_dir,
            capsule.source_relative_path, descriptor, capsule.source_sha256,
            capsule.n_files, capsule.n_bytes))
    return tuple(rows)


def _pk_capsules() -> tuple[PAS.PerformanceCapsule, ...]:
    return _capsules_named("PK")


def _pr_capsules() -> tuple[PAS.PerformanceCapsule, ...]:
    return _capsules_named("PR")


# --------------------------------------------------------------------------------------------
# A family nobody wrote code for.  If dispatch really reads the declaration, this works.
# --------------------------------------------------------------------------------------------

_SYNTHETIC_MODULE = "perf_zz_claim_for_tests"
_SYNTHETIC_ACCEPTANCE = {
    "schema_version": 1,
    "analyzer": f"{_SYNTHETIC_MODULE}.analyze_zz_claim/v1",
    "fit": {"form": "affine", "independent_variable": "K"},
    "replicates": {"minimum_count": 2, "identities_authored_by": "run"},
    "evidence": {"timing_simulator": "gsim", "timing_tier": "L3",
                 "correctness_simulator": "spike", "correctness_tier": "L2"},
}


def _install_synthetic_analyzer() -> ModuleType:
    module = ModuleType(_SYNTHETIC_MODULE)

    def supported_acceptance():
        return copy.deepcopy(_SYNTHETIC_ACCEPTANCE)

    def preflight_zz_claim(descriptors, *, replicates):
        names = [str(row["name"]) for row in descriptors]
        return {
            "schema_version": 1, "family": "ZZ", "claim": "DIFFERENTIAL", "status": "READY",
            "declaration": supported_acceptance(),
            "cohort": {"capsules": names},
            "replicates": list(replicates),
            "expected_identities": [
                {"family": "ZZ", "capsule": name, "simulator": simulator,
                 "replicate": replicate, "tier": tier}
                for name in names for replicate in replicates
                for simulator, tier in (("spike", "L2"), ("gsim", "L3"))],
            "refusal_reasons": [],
        }

    def analyze_zz_claim(descriptors, results, *, replicates):
        return {"status": "REFUSED", "refusal_reasons": ["no rows"]}

    module.supported_acceptance = supported_acceptance
    module.preflight_zz_claim = preflight_zz_claim
    module.analyze_zz_claim = analyze_zz_claim
    sys.modules[_SYNTHETIC_MODULE] = module
    return module


def _synthetic_capsules(n: int = 3) -> tuple[PAS.PerformanceCapsule, ...]:
    rows = []
    for index in range(n):
        name = f"ZZ{index:02d}_k{16 * (index + 1)}"
        descriptor = {
            "name": name, "kind": "model_slice", "label": "dev",
            "performance": {
                "level": "L2_intra_layer", "family": "ZZ", "lever": "made_up_lever",
                "claim": "DIFFERENTIAL",
                "acceptance": copy.deepcopy(_SYNTHETIC_ACCEPTANCE),
                "comparand": {"kind": "paired_run", "against": "itself", "cancels": ["shape"],
                              "demand_equal": ["operation"]},
                "falsifier": {"observation": "cycle_delta", "negative_control": "same_member"},
                "emitter": {"knobs": {"varied_axis": "K"}},
            },
        }
        rows.append(PAS.PerformanceCapsule(
            "ZZ", name, Path(f"/frozen/_perf/{name}"), f"_perf/{name}", descriptor,
            _SHA, 2, 100))
    return tuple(rows)


@pytest.fixture()
def synthetic_analyzer():
    module = _install_synthetic_analyzer()
    try:
        yield module
    finally:
        sys.modules.pop(_SYNTHETIC_MODULE, None)


# --------------------------------------------------------------------------------------------
# POSITIVE: a family other than PK reaches a candidate handoff.
# --------------------------------------------------------------------------------------------

def test_a_replicate_floor_family_produces_a_candidate_handoff() -> None:
    """PR declares ``minimum_count`` and no identities; that is a contract, not a refusal."""
    capsules = _pr_capsules()
    claim = PAS.prepare_formal_pk_claim(capsules)

    assert claim["status"] == "READY"
    assert claim["family"] == "PR" and claim["claim"] == "DIFFERENTIAL"
    # The declaration carries neither exact_count nor identities; the run authored them.
    contract = claim["declaration"]["replicates"]
    assert "exact_count" not in contract and "identities" not in contract
    assert contract["identities_authored_by"] == "run"
    assert PAS._preflight_cohort(claim) == ("r000", "r001")
    assert len(claim["expected_identities"]) == len(capsules) * 2 * 2

    families = PAS._family_declarations(capsules, claim)
    assert [row.family for row in families] == ["PR"]
    assert families[0].acceptance == claim["declaration"]


def _sealing_inputs(capsules, claim):
    identities = list(PAS._preflight_cohort(claim))
    simulator = str(claim["declaration"]["evidence"]["timing_simulator"])
    cells = [{"family": row.family, "capsule": row.capsule, "simulator": row.simulator,
              "replicate": row.replicate}
             for row in PAS.expected_perf_cells(capsules, len(identities), simulator)]
    families = json.loads(json.dumps([
        {"family": row.family, "claim": row.claim, "negative_control": row.negative_control,
         "falsifier_observation": row.falsifier_observation,
         "differential_basis": row.differential_basis,
         "fitted_parameters": list(row.fitted_parameters), "acceptance": row.acceptance}
        for row in PAS._family_declarations(capsules, claim)]))
    return identities, cells, families


def test_the_handoff_survives_its_own_sealed_record_validation() -> None:
    capsules = _pr_capsules()
    claim = PAS.prepare_formal_pk_claim(capsules)
    identities, cells, families = _sealing_inputs(capsules, claim)

    PAS._validate_formal_claim_facts(claim, len(identities), identities, 1, cells, families)

    # Vacuity guard: the same call must FAIL when the record no longer agrees with itself.
    with pytest.raises(PAS.StageGateError, match="formal replicates drift"):
        PAS._validate_formal_claim_facts(claim, len(identities), ["r000", "r009"], 1, cells,
                                         families)
    with pytest.raises(PAS.StageGateError, match="drift from expected cells"):
        PAS._validate_formal_claim_facts(claim, len(identities), identities, 1, cells[:-2],
                                         families)


def test_every_shipped_claim_family_now_reaches_a_handoff() -> None:
    """The campaign families the differential and law analyzers were written for.

    Before dispatch, only PK could produce one: every other family measured and no code path ever
    turned its rows into a verdict.  A family that still cannot reach one must say which declaration
    it is missing -- refusing by name is the whole point, and a silent skip is the defect.
    """
    prefixes = sorted({source.name[:2] for source in _PERF_ROOT.iterdir() if source.is_dir()})
    ready, refused = {}, {}
    for prefix in prefixes:
        capsules = _capsules_named(prefix)
        try:
            claim = PAS.prepare_formal_pk_claim(capsules)
        except PAS.StageGateError as exc:
            refused[prefix] = str(exc)
            continue
        ready[prefix] = PAS._preflight_cohort(claim)
    assert {"PK", "PQ", "PM", "PV"} <= set(ready), (ready, refused)
    for prefix, reason in refused.items():
        assert ("declares no acceptance.analyzer" in reason
                or "replicate count" in reason), (prefix, reason)


def test_a_family_with_no_code_written_for_it_is_dispatched_from_its_declaration(
        synthetic_analyzer) -> None:
    """The real test of "derive, never hardcode": nothing in the stage has heard of ``ZZ``."""
    assert "ZZ" not in _SCRIPTS.joinpath("perf_agent_stage.py").read_text(encoding="utf-8")

    capsules = _synthetic_capsules()
    claim = PAS.prepare_formal_pk_claim(capsules)

    assert claim["family"] == "ZZ" and claim["status"] == "READY"
    assert PAS._preflight_cohort(claim) == ("r000", "r001")
    assert PAS._family_declarations(capsules, claim)[0].family == "ZZ"

    # Its floor is honoured as a floor, and a request below it is refused.
    raised = PAS.prepare_formal_pk_claim(capsules, 4)
    assert PAS._preflight_cohort(raised) == ("r000", "r001", "r002", "r003")
    with pytest.raises(PAS.StageGateError, match="minimum_count=2"):
        PAS.prepare_formal_pk_claim(capsules, 1)


# --------------------------------------------------------------------------------------------
# REGRESSION: PK's handoff is what PK's own analyzer produces, byte for byte.
# --------------------------------------------------------------------------------------------

def test_pk_handoff_is_exactly_its_own_analyzers_preflight() -> None:
    capsules = _pk_capsules()
    direct = PK.preflight_pk_claim([capsule.descriptor for capsule in capsules])

    dispatched = PAS.prepare_formal_pk_claim(capsules)

    assert PAS._canonical_json(dispatched) == PAS._canonical_json(direct)
    assert dispatched["declaration"] == PK.supported_acceptance()
    assert PAS._preflight_cohort(dispatched) == ("r000", "r001")
    assert PAS._family_declarations(capsules, dispatched)[0].acceptance == \
        PK.supported_acceptance()
    # An exact_count stays exact: a request that differs is still refused by its own message.
    with pytest.raises(PAS.StageGateError, match="exact_count=2"):
        PAS.prepare_formal_pk_claim(capsules, 3)


# --------------------------------------------------------------------------------------------
# REFUSALS: each names the declared fact that is missing, and each has a live positive beside it.
# --------------------------------------------------------------------------------------------

def test_a_family_declaring_no_analyzer_is_refused_by_name() -> None:
    """PT is declared, blocked and deliberately analyzer-less; it must not fall through to PK."""
    profile = yaml.safe_load(_PROFILE.read_text(encoding="utf-8"))
    blocked = {str(row["family"]): row for row in profile["blocked_unimplemented"]}
    performance = blocked["PT"]["performance"]
    assert "analyzer" not in performance["acceptance"], "PT gained an analyzer; retarget this test"

    capsules = tuple(
        PAS.PerformanceCapsule(
            "PT", f"PT{index:02d}", Path(f"/frozen/_perf/PT{index:02d}"), f"_perf/PT{index:02d}",
            {"name": f"PT{index:02d}", "kind": "model_slice", "label": "dev",
             "performance": copy.deepcopy(performance)}, _SHA, 2, 100)
        for index in range(4))

    with pytest.raises(PAS.StageGateError, match="declares no acceptance.analyzer"):
        PAS.prepare_formal_pk_claim(capsules)

    # ...and mixing it with a family that DOES declare one still refuses, rather than quietly
    # authoring PK's claim over a corpus PT is part of.
    mixed = capsules[:2] + _pk_capsules()[:2]
    with pytest.raises(PAS.StageGateError, match="declares no acceptance.analyzer"):
        PAS.prepare_formal_pk_claim(mixed)


def test_removing_the_declared_analyzer_is_what_makes_the_positive_stop_passing() -> None:
    """Mutation check: the PR handoff succeeds, and fails for THIS reason once the fact is gone."""
    capsules = _pr_capsules()
    assert PAS.prepare_formal_pk_claim(capsules)["status"] == "READY"

    stripped = _rewritten(capsules,
                          lambda d: d["performance"]["acceptance"].pop("analyzer"))
    with pytest.raises(PAS.StageGateError, match="declares no acceptance.analyzer"):
        PAS.prepare_formal_pk_claim(stripped)

    malformed = _rewritten(
        capsules, lambda d: d["performance"]["acceptance"].__setitem__("analyzer", "not-a-path"))
    with pytest.raises(PAS.StageGateError, match="unusable"):
        PAS.prepare_formal_pk_claim(malformed)

    absent = _rewritten(
        capsules,
        lambda d: d["performance"]["acceptance"].__setitem__(
            "analyzer", "perf_no_such_analyzer.analyze/v1"))
    with pytest.raises(PAS.StageGateError, match="is unavailable"):
        PAS.prepare_formal_pk_claim(absent)


def test_the_refusal_comes_from_the_analyzer_the_declaration_NAMES(synthetic_analyzer) -> None:
    """A refusal must be the declared analyzer's own, quoted verbatim -- not some other family's.

    The distinctive reason text is what makes this non-vacuous: a stage that still called PK would
    refuse too, but with PK's words, and the assertion below would not see them.
    """
    def refuse(descriptors, *, replicates):
        return {"schema_version": 1, "family": "ZZ", "claim": "DIFFERENTIAL", "status": "REFUSED",
                "declaration": None, "cohort": None, "expected_identities": [],
                "refusal_reasons": ["ZZ has not measured its own settling depth"]}

    synthetic_analyzer.preflight_zz_claim = refuse
    with pytest.raises(PAS.StageGateError,
                       match="frozen ZZ formal claim preflight refused: ZZ has not measured"):
        PAS.prepare_formal_pk_claim(_synthetic_capsules())

    # Naming a real module is not enough either: the contract it freezes must be this family's.
    borrowed = _rewritten(
        _synthetic_capsules(),
        lambda d: d["performance"]["acceptance"].__setitem__(
            "analyzer", "perf_pk_claim.analyze_pk_claim/v3"))
    with pytest.raises(PAS.StageGateError, match="frozen ZZ formal claim preflight refused"):
        PAS.prepare_formal_pk_claim(borrowed)


def test_a_corpus_declaring_two_analyzers_is_refused_rather_than_split() -> None:
    mixed = _pk_capsules()[:2] + _pr_capsules()[:3]
    with pytest.raises(PAS.StageGateError, match="declares 2 claim analyzers"):
        PAS.prepare_formal_pk_claim(mixed)


def test_a_declaration_stating_no_replicate_count_is_refused() -> None:
    def drop_counts(descriptor):
        contract = descriptor["performance"]["acceptance"]["replicates"]
        contract.pop("minimum_count", None)
        contract.pop("exact_count", None)

    with pytest.raises(PAS.StageGateError, match="neither an exact nor a minimum"):
        PAS.prepare_formal_pk_claim(_rewritten(_pr_capsules(), drop_counts))


def test_run_facts_a_stage_cannot_supply_fail_closed() -> None:
    """An analyzer asking for a fact nobody derived must refuse, never be called without it."""
    def needs_counters(descriptors, *, replicates, counters):  # pragma: no cover - never called
        raise AssertionError("the entry point must not be invoked")

    with pytest.raises(PAS.StageGateError, match="requires run facts this stage cannot supply"):
        PAS._analyzer_kwargs(needs_counters, {"replicates": lambda: ["r000", "r001"]},
                             label="probe")

    # ...and the same helper does supply what it CAN, so the refusal above is not vacuous.
    def needs_replicates(descriptors, *, replicates):  # pragma: no cover - never called
        raise AssertionError("the entry point must not be invoked")

    assert PAS._analyzer_kwargs(
        needs_replicates, {"replicates": lambda: ["r000", "r001"]},
        label="probe") == {"replicates": ["r000", "r001"]}


# --------------------------------------------------------------------------------------------
# The gate that made this a blocker: one entry point per analyzer, resolved by prefix.
# --------------------------------------------------------------------------------------------

def test_every_declared_analyzer_publishes_exactly_one_preflight_entry_point() -> None:
    """``perf_reporting.validate_claim_decision`` resolves the preflight by prefix and requires one.

    Two of the four analyzers published ZERO, so every family declaring them could be measured and
    never decided -- and nothing said so, because the gate only speaks at report time.  Asserting it
    over the SHIPPED profile means a newly declared family cannot reintroduce the hole.
    """
    import importlib

    from merlin.perf import claim_reach

    profile = yaml.safe_load(_PROFILE.read_text(encoding="utf-8"))
    declared = {}
    for sweep in profile["sweeps"]:
        performance = sweep["base"]["performance"]
        identity = claim_reach.analyzer_identity(performance)
        if identity is not None:
            declared.setdefault(identity.module, set()).add(str(sweep["id"]))
    assert len(declared) >= 3, f"the profile names too few analyzers to be this test: {declared}"

    for module_name, families in sorted(declared.items()):
        module = importlib.import_module(module_name)
        entries = sorted(name for name in dir(module)
                         if name.startswith("preflight_") and callable(getattr(module, name)))
        assert len(entries) == 1, (
            f"{module_name} publishes {entries}; the reporting gate requires exactly one, and "
            f"{sorted(families)} would be measured and never decided")
