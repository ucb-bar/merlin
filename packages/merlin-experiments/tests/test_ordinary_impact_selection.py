"""Original product comparison controls with harmless process copies; no engines.

These fixtures establish byte/provenance selection only, not numerical or
hardware qualification. Cheap gate verdicts are synthetic input observations.
"""

import json
import sys
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import ordinary_impact_selection as IMPACT
from merlin_experiments.phase2 import queued_search_actions as QSA
from merlin_experiments.phase2.contracts import sha256_file
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs

from merlin.common import invocation_record as I

COPY = "import pathlib,sys; pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())"
ROSTER = (("family", "a"), ("family", "b"))


def _arm(root, payload):
    root.mkdir(parents=True)
    llvm = root / "lowered.llvm.mlir"
    llvm.write_bytes(payload)
    ll, obj, elf = root / "kernel.ll", root / "kernel.o", root / IMPACT.FAST.PACKAGE_ELF_NAME
    for stage, source, output in (("llvm_translation", llvm, ll), ("object", ll, obj), ("elf", obj, elf)):
        I.run(
            [sys.executable, "-c", COPY, str(source), str(output)],
            directory=root,
            stage=stage,
            inputs=(source,),
            outputs=(output,),
            capture_output=True,
            text=True,
        )
    value = {
        "functional_status": "PASS",
        "compiled": {
            "status": "observed",
            "artifact_join": {"kind": "original_operator_products"},
            "llvm_mlir_sha256": sha256_file(llvm),
            "elf_sha256": sha256_file(elf),
        },
    }
    path = root / "ordinary_observation.json"
    path.write_text(json.dumps({**value, "original_grade": {}}))
    return {**value, "evidence_sha256": sha256_file(path)}


def _document(root):
    cells = []
    for index, (family, capsule) in enumerate(ROSTER):
        cells.append(
            {
                "family": family,
                "capsule": capsule,
                "baseline": _arm(root / f"m{index:03d}_baseline", b"observed fixture"),
                "candidate": _arm(root / f"m{index:03d}_candidate", b"changed" if index else b"observed fixture"),
            }
        )
    return {"cells": cells}


def test_only_reopened_same_products_can_omit_development_timing(tmp_path):
    document = _document(tmp_path)
    impact = IMPACT.plan(document, roster=ROSTER, cheap_root=tmp_path, deadline=time.monotonic() + 30)
    assert [row["impact"] for row in impact["members"]] == ["unchanged_products", "changed_products"]
    assert impact["selected"] == [["family", "b"]]
    assert impact["promotion"] == "DIAGNOSTIC_ONLY"
    assert "no new hardware" in impact["unchanged_meaning"]


@pytest.mark.parametrize("damage", ["elf", "llvm", "receipt", "functional", "missing_join", "private_evidence"])
def test_unknown_and_mutated_original_evidence_includes_member(tmp_path, damage):
    document = _document(tmp_path)
    root = tmp_path / "m000_candidate"
    if damage == "elf":
        (root / IMPACT.FAST.PACKAGE_ELF_NAME).write_bytes(b"after observation")
    elif damage == "llvm":
        (root / "lowered.llvm.mlir").write_bytes(b"after observation")
    elif damage == "receipt":
        next(root.rglob("invocation.json")).write_text("{}")
    elif damage == "private_evidence":
        (root / "ordinary_observation.json").write_text("{}")
    else:
        path = root / "ordinary_observation.json"
        private = json.loads(path.read_bytes())
        if damage == "functional":
            private["functional_status"] = "UNKNOWN_OR_FAILURE"
        else:
            private["compiled"]["artifact_join"] = {}
        path.write_text(json.dumps(private))
        document["cells"][0]["candidate"] = {key: value for key, value in private.items() if key != "original_grade"}
        document["cells"][0]["candidate"]["evidence_sha256"] = sha256_file(path)
    impact = IMPACT.plan(document, roster=ROSTER, cheap_root=tmp_path, deadline=time.monotonic() + 30)
    assert impact["members"][0]["impact"] == "UNKNOWN"
    assert impact["selected"] == [list(row) for row in ROSTER]


def test_missing_or_late_completed_job_broadens_to_full_roster(tmp_path):
    for document, deadline in ((None, time.monotonic() + 1), ({"cells": []}, time.monotonic() - 1)):
        result = IMPACT.plan(document, roster=ROSTER, cheap_root=tmp_path, deadline=deadline)
        assert result["selected"] == [list(row) for row in ROSTER]
        assert all(row["impact"] == "UNKNOWN" for row in result["members"])


def test_existing_original_action_dispatch_selects_no_candidate_supplied_scope(monkeypatch):
    action = next(row for row in QSA.actions() if row.name == IMPACT.ACTION)
    assert action.placeholders == ("revision_id",)
    calls = []
    coordinator = object()
    policy = SimpleNamespace(
        validate_search_inputs=lambda: None, search_coordinator=coordinator, evaluation_pool=object()
    )
    monkeypatch.setattr(
        IMPACT, "submit", lambda owner, revision, **kwargs: calls.append((owner, revision)) or {"ok": True}
    )
    response, _ = QSA.execute(policy, IMPACT.ACTION, {"revision_id": "issued"}, 10, time.monotonic())
    assert calls == [(coordinator, "issued")]
    assert json.loads(response["stdout"]) == {"ok": True}


def test_live_original_queue_without_completed_evidence_dispatches_all_members(ordinary_inputs, tmp_path, monkeypatch):
    """Exercise live ownership/closure joining; intercept dispatch before any jobs."""
    import os

    from merlin_experiments.phase2 import ordinary_measurement_selection as SELECT
    from merlin_experiments.phase2 import ordinary_search_policy as OS
    from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
    from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore
    from merlin_experiments.phase2.search_coordinator import ComponentSearchCoordinator
    from test_ordinary_fast_feedback import extend_corpus

    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    if len(cpus) < 2:
        pytest.skip("two CPUs required to construct the original resource owner")
    extend_corpus(ordinary_inputs, tmp_path)
    policy = OS.prepare_ordinary_search_policy(**ordinary_inputs.arguments)
    store = RetainedRevisionStore(
        tmp_path / "sources",
        edit_authority=policy.edit_authority,
        max_bytes=16_384,
        max_files=16,
        max_total_bytes=65_536,
    )
    pool = policy.enable_evaluation_pool(
        revisions=store,
        resources=AssignedResources(cpus, 8 << 30, 32 << 20, 1, author_cores=1),
        job_resources=JobResources(1, 2 << 30, 4 << 20),
        root=tmp_path / "queue",
    )
    try:
        coordinator = policy.search_coordinator = ComponentSearchCoordinator(pool)
        revision = store.freeze(policy.candidate, parent_revision=pool.baseline_revision.revision_id)
        calls = []
        monkeypatch.setattr(
            pool,
            "submit",
            lambda revision_id, **kwargs: calls.append((revision_id, kwargs)) or {"job_id": "dispatch_intercepted"},
        )
        result = IMPACT.submit(coordinator, revision.revision_id, timeout_s=20)
        assert result["status"] == "queued_diagnostic"
        assert result["impact"]["source_sha256"] == revision.source_sha256
        assert result["impact"]["configuration_sha256"] == policy.search_closure_sha256
        assert result["impact"]["cheap_evidence_sha256"] is None
        assert calls[0][1]["kind"] == SELECT.KIND
        assert calls[0][1]["selected_members"] == tuple(sorted(policy.validate_search_inputs()))
        assert pool.observations() == ()
        assert not ordinary_inputs.executions
    finally:
        pool.close()
