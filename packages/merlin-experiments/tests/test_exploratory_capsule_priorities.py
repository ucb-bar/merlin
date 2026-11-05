"""Real source-work parsing plus synthetic queue arithmetic, never execution authority."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import exploratory_capsule_priorities as P
from merlin_experiments.phase2 import search_coordinator as SC
from merlin_experiments.phase2.contracts import StageGateError

BASE, CANDIDATE, CORPUS = "b" * 64, "c" * 64, "f" * 64
KEYS = (("public", "left"), ("public", "right"))


def pin(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def source(m=2, extra=""):
    return f'''module {{func.func @entry(%a: tensor<{m}x3xi8>, %b: tensor<3x4xi8>,
      %c: tensor<{m}x4xi32>) -> tensor<{m}x4xi32> {{
      %r = linalg.matmul ins(%a, %b: tensor<{m}x3xi8>, tensor<3x4xi8>)
        outs(%c: tensor<{m}x4xi32>) -> tensor<{m}x4xi32>
      {extra}
      func.return %r : tensor<{m}x4xi32>
    }}}}'''


def arm(cycles, identity):
    estimate = dict(
        schema=P.ESTIMATE.PROTOCOL, status="estimated", estimated_cycles=cycles,
        model_sha256="a" * 64, comparison_domain_sha256="d" * 64,
        scope_sha256="e" * 64, report_sha256=identity * 64,
        qualified_for_promotion=False, event_accounting_complete=True,
        unresolved_events=[], domain_deviations=[dict(feature="load", value=999, training_range=[0, 20])],
        calibration_residuals={"applicability": "no out-of-domain error guarantee"},
        uncertainty_interval_cycles=None, scope="fixed-model exploration only",
    )
    return dict(functional_status="PASS", compiled=dict(status="observed", elf_sha256=identity * 64),
                native_exploration=dict(exploratory_estimate=estimate, observed_counts=dict(
                    provenance=dict(report_sha256=identity * 64, scope_binding=dict(scope_sha256="e" * 64)))))


@pytest.fixture
def setup(tmp_path):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(dict(facts=dict(arrays=[dict(rows=2, cols=2, mac_idiom=dict(muls=1))]))))
    members = []
    for index, key in enumerate(KEYS):
        root = tmp_path / key[1]
        root.mkdir()
        mlir = root / "source.mlir"
        mlir.write_text(source(2 + 2 * index))
        members.append(SimpleNamespace(family=key[0], capsule=key[1], source_dir=root,
                                       source_sha256=str(index) * 64,
                                       descriptor=dict(frontend_trace=dict(source_mlir=mlir.name,
                                                                          source_mlir_sha256=pin(mlir)["sha256"]))))
    corpus = SimpleNamespace(capsules=tuple(members), capsules_sha256=CORPUS)
    calls = []

    def validate(raw, revision, kind):
        calls.append((revision.source_sha256, kind))
        if raw.get("corrupted"):
            raise StageGateError("original result refused")
        return copy.deepcopy(raw)

    observations = []
    issued = {}

    def status(identity):
        if identity not in issued:
            raise StageGateError("job not issued by original pool")
        return copy.deepcopy(issued[identity])

    pool = SimpleNamespace(
        inputs=SimpleNamespace(validate=lambda: set(KEYS), corpus=corpus, validate_result=validate),
        policy=SimpleNamespace(_native_exploration=SimpleNamespace(exploratory_estimate_protocol=P.ESTIMATE.PROTOCOL),
                               binding=lambda: dict(rtl_facts=pin(facts))),
        baseline_revision=SimpleNamespace(public_record=lambda: dict(source_sha256=BASE)),
        revisions=SimpleNamespace(get=lambda identity: SimpleNamespace(source_sha256=identity)),
        observations=lambda: copy.deepcopy(observations),
        status=status,
    )

    def observation(cycles=(60, 120), *, selected=KEYS):
        cells = []
        for key, member, cycle in zip(KEYS, members, cycles, strict=True):
            cells.append(dict(family=key[0], capsule=key[1], source_sha256=member.source_sha256,
                              baseline=arm(240, "b") if key in selected else None,
                              candidate=arm(cycle, "c") if key in selected else None,
                              diagnostic_pair_check=dict(status="PASS") if key in selected else None))
        row = dict(state="completed", kind=P.ORDINARY_SELECTED_FAST_KIND, revision_id=CANDIDATE,
                   source_sha256=CANDIDATE, job_id=str(len(issued)),
                   result=dict(candidate_sha256=CANDIDATE, tuning_corpus_sha256=CORPUS, cells=cells))
        # Synthetic original custody: mutation tests deliberately alter the
        # issued feedback too, then verify its normal validator/arithmetic gates.
        issued[row["job_id"]] = row
        return row

    return SimpleNamespace(pool=pool, facts=facts, members=members, calls=calls,
                           observations=observations, observation=observation, issued=issued)


def test_exact_source_work_and_worst_repeat_order_are_separate_from_convergence(setup):
    s = setup
    s.observations.extend([s.observation(), s.observation((120, 120))])
    before = copy.deepcopy(s.observations)
    result = P.project(s.pool)
    assert len(s.calls) == 2
    assert [r["useful_macs"] for r in result["source_compute_floors"]] == [24, 48]
    assert [r["floor_cycles"] for r in result["ranked"]] == [6, 12]
    assert result["ranked"][0]["attainment"] == dict(worst=.05, best=.1)
    assert result["ranked"][0]["repeat_count"] == 2
    assert result["next"] == [list(k) for k in KEYS]
    assert result["unknown"] == result["under_floor"] == []
    assert result["status"] == "EXPLORATORY" and result["convergence"] is False
    assert result["measured_cycles"] is None and result["qualified_for_promotion"] is False
    assert result["completion_blocked"] is True and result["plateau_evidence"] is False
    assert result["ranked"][0]["provenance"][0]["estimate"]["domain_deviations"]
    assert s.observations == before


def test_partial_selection_keeps_all_roster_unknowns_and_baseline_source_group(setup):
    s = setup
    s.observations.append(s.observation(selected=KEYS[:1]))
    result = P.project(s.pool)
    assert result["unknown"][0]["member"] == list(KEYS[1])
    assert len(result["ranked"]) == 1
    assert {row["source_sha256"] for row in result["sources"]} == {BASE, CANDIDATE}


@pytest.mark.parametrize("fault", ["source_bytes", "source_missing", "source_escape", "source_link", "source_unknown",
                                  "trace_missing", "arithmetic_missing"])
def test_corrupt_missing_or_unsupported_source_work_is_unknown(setup, fault):
    s = setup
    member = s.members[0]
    trace = member.descriptor["frontend_trace"]
    path = member.source_dir / "source.mlir"
    if fault == "source_bytes":
        path.write_text(source(8))
    elif fault == "source_missing":
        path.unlink()
    elif fault == "source_escape":
        trace["source_mlir"] = "../right/source.mlir"
    elif fault == "source_link":
        path.unlink()
        path.symlink_to(s.members[1].source_dir / "source.mlir")
    elif fault == "source_unknown":
        path.write_text(source().replace("tensor<2x3xi8>", "tensor<?x3xi8>"))
        trace["source_mlir_sha256"] = pin(path)["sha256"]
    elif fault == "trace_missing":
        member.descriptor.clear()
    elif fault == "arithmetic_missing":
        s.facts.write_text(json.dumps(dict(facts=dict(arrays=[dict(rows=2, cols=2)]))))
    s.observations.append(s.observation())
    result = P.project(s.pool)
    assert any(row["member"] == list(KEYS[0]) for row in result["unknown"])
    assert all(row["member"] != list(KEYS[0]) for row in result["ranked"])


@pytest.mark.parametrize("fault", ["functional", "elf", "counts", "report", "scope", "accounting", "unresolved",
                                  "pair", "promotion", "underfloor", "repeat_domain", "validation"])
def test_incomplete_corrupt_or_underfloor_estimates_cannot_order_a_capsule(setup, fault):
    s = setup
    good = s.observation()
    bad = s.observation()
    arm_ = bad["result"]["cells"][0]["candidate"]
    estimate = arm_["native_exploration"]["exploratory_estimate"]
    if fault == "functional":
        arm_["functional_status"] = "FAIL"
    elif fault == "elf":
        arm_["compiled"]["elf_sha256"] = "wrong"
    elif fault == "counts":
        arm_["native_exploration"]["observed_counts"] = None
    elif fault in {"report", "scope"}:
        estimate[{"report": "report_sha256", "scope": "scope_sha256"}[fault]] = "9" * 64
    elif fault == "accounting":
        estimate["event_accounting_complete"] = False
    elif fault == "unresolved":
        estimate["unresolved_events"] = ["unpriced store"]
    elif fault == "pair":
        bad["result"]["cells"][0]["diagnostic_pair_check"]["status"] = "UNKNOWN"
    elif fault == "promotion":
        estimate["qualified_for_promotion"] = True
    elif fault == "underfloor":
        estimate["estimated_cycles"] = 5
    elif fault == "repeat_domain":
        for a in ("baseline", "candidate"):
            projected = bad["result"]["cells"][0][a]["native_exploration"]["exploratory_estimate"]
            projected["comparison_domain_sha256"] = "9" * 64
    elif fault == "validation":
        bad["result"]["corrupted"] = True
    s.observations.extend([good, bad])
    result = P.project(s.pool)
    assert all(row["member"] != list(KEYS[0]) for row in result["ranked"])
    assert any(row["member"] == list(KEYS[0]) for row in result["unknown"] + result["under_floor"])
    if fault == "validation":
        assert result["rejected_observations"][0]["reason"] == "original result refused"


def test_running_and_other_evaluator_rows_do_not_supply_estimates(setup):
    s = setup
    for state, kind in (("running", P.ORDINARY_SELECTED_FAST_KIND), ("completed", "component-rtl-feedback")):
        row = s.observation()
        row.update(state=state, kind=kind)
        s.observations.append(row)
    result = P.project(s.pool)
    assert s.calls == [] and result["ranked"] == []
    assert [row["member"] for row in result["unknown"]] == [list(k) for k in KEYS]


def test_replayed_or_unissued_serialized_observation_has_no_queue_authority(setup):
    s = setup
    row = s.observation()
    s.observations.append(row)
    s.issued.clear()
    result = P.project(s.pool)
    assert result["ranked"] == [] and s.calls == []
    assert result["rejected_observations"][0]["reason"] == "job not issued by original pool"


def test_known_mac_plus_unproved_called_work_remains_unknown(setup):
    s = setup
    member = s.members[0]
    path = member.source_dir / "source.mlir"
    text = source(extra='func.call @helper() : () -> ()')
    text = text[:-1] + ' func.func private @helper() }'
    path.write_text(text)
    member.descriptor["frontend_trace"]["source_mlir_sha256"] = pin(path)["sha256"]
    floors = P.derive_compute_floors(s.pool.inputs.corpus, facts_pin=pin(s.facts))
    floor = floors[KEYS[0]]
    assert {row["status"] for row in floor["work"]} == {"derived", "UNKNOWN"}
    assert floor["status"] == "UNKNOWN" and floor["floor_cycles"] is None


def test_changed_selected_facts_pin_fails_closed(setup):
    s = setup
    selected = pin(s.facts)
    s.facts.write_text("{}")
    with pytest.raises(StageGateError, match="source pin changed"):
        P.derive_compute_floors(s.pool.inputs.corpus, facts_pin=selected)


def test_coordinator_helper_does_not_touch_incumbent_or_target_state(setup):
    owner = object.__new__(SC.ComponentSearchCoordinator)
    owner.pool = setup.pool
    owner._incumbent = {"untouched": True}
    assert owner.exploratory_capsule_priorities()["status"] == "EXPLORATORY"
    assert owner._incumbent == {"untouched": True}
