"""Real retained/edit-authority controls; neutral queue transport, no target jobs."""

import json
import os
import shutil
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import campaign
from merlin_experiments.phase2 import public_diagnostic_launch as L
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore

from merlin.perf.phase2_edit_contract import seal


@pytest.fixture
def retained(tmp_path):
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "direct_codegen.py").write_text("def emit_direct_matmul():\n    return ('m', 'n')\n")
    (initial / "emitter.py").write_text("def make_c():\n    return 1\n")
    (initial / "manifest.yaml").write_text("components:\n  emit: [direct_codegen.py, emitter.py]\n")
    contract = seal({
        "schema": "compiler_edit_contract_v1",
        "existing_symbols": [{"surface_id": "loop_order", "path": "direct_codegen.py",
                              "symbol": "emit_direct_matmul"}],
        "helper_extensions": [],
    })
    baseline = tmp_path / "baseline"
    campaign.materialize_readonly_tree(initial, baseline)
    output = tmp_path / "authority"
    output.mkdir()
    authority = FrozenEditAuthority(output)
    authority.freeze(initial, contract, has_iterations=False)
    store = RetainedRevisionStore(tmp_path / "revisions", edit_authority=authority,
                                  max_bytes=1 << 20, max_files=32, max_total_bytes=4 << 20)
    original = store.freeze(initial, options={"hypothesis": "control", "edit_family": "unchanged-source-control"})
    source = tmp_path / "selected"
    shutil.copytree(initial, source)
    (source / "direct_codegen.py").write_text("def emit_direct_matmul():\n    return ('n', 'm')\n")
    return store, original, source, baseline, contract


def test_original_authority_retains_distinct_bytes_and_preserves_baseline(retained):
    store, initial, source, baseline, _ = retained
    before = L._inventory(baseline)
    selected = L._inventory(source)
    revision, receipt = L._retain_candidate_source(store, initial, source, selected)
    assert revision.parent_revision == initial.revision_id
    assert revision.source_sha256 != initial.source_sha256
    assert receipt["distinct_source"] is True
    assert receipt["selected_inventory"] == selected
    assert receipt["submitted_revision"] == revision.public_record()
    assert receipt["initial_revision"] == initial.public_record()
    assert L._inventory(baseline) == before
    assert all(not path.stat().st_mode & 0o222 for path in (baseline, *baseline.rglob("*")))
    (source / "direct_codegen.py").write_text("def emit_direct_matmul():\n    return ('later',)\n")
    revision.verify(authority=store.authority)
    assert "('n', 'm')" in (revision.source / "direct_codegen.py").read_text()


def test_outside_emitter_edit_cannot_borrow_direct_codegen_authority(retained):
    store, initial, source, baseline, _ = retained
    before = L._inventory(baseline)
    (source / "emitter.py").write_text("def make_c():\n    return 2\n")
    with pytest.raises(ValueError, match="exceeds host-frozen authority"):
        L._retain_candidate_source(store, initial, source, L._inventory(source))
    assert L._inventory(baseline) == before


def test_source_drift_between_host_preflight_and_retention_refuses(retained):
    store, initial, source, _, _ = retained
    before = L._inventory(source)
    (source / "direct_codegen.py").write_text("def emit_direct_matmul():\n    return ('changed',)\n")
    with pytest.raises(StageGateError, match="changed after preflight"):
        L._retain_candidate_source(store, initial, source, before)


def test_source_drift_after_original_retention_cannot_be_submitted(retained, monkeypatch):
    store, initial, source, _, _ = retained
    before = L._inventory(source)
    freeze = store.freeze

    def changed(*args, **kwargs):
        revision = freeze(*args, **kwargs)
        (source / "direct_codegen.py").write_text("def emit_direct_matmul():\n    return ('later',)\n")
        return revision

    monkeypatch.setattr(store, "freeze", changed)
    with pytest.raises(StageGateError, match="changed during retention"):
        L._retain_candidate_source(store, initial, source, before)


def test_equal_source_is_explicitly_reported_as_unchanged(retained):
    store, initial, source, _, _ = retained
    (source / "direct_codegen.py").write_bytes((initial.source / "direct_codegen.py").read_bytes())
    revision, receipt = L._retain_candidate_source(store, initial, source, L._inventory(source))
    assert receipt["distinct_source"] is False
    assert revision.source_sha256 == initial.source_sha256
    assert revision.parent_revision == initial.revision_id


def test_linked_candidate_source_is_refused(retained, tmp_path):
    store, initial, source, _, _ = retained
    linked = tmp_path / "linked"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(StageGateError, match="canonical"):
        L._retain_candidate_source(store, initial, linked, L._inventory(source))


@pytest.mark.parametrize("selected", [False, True])
def test_launch_submits_the_original_issued_selected_revision(retained, tmp_path, monkeypatch, selected):
    """Exercise launcher orchestration with real retention, neutral nonexecuting queue."""
    import time

    from merlin_experiments.phase0 import operator_exclusion
    from merlin_experiments.phase2 import compiler_library_state, public_diagnostic_corpus, public_source_diagnostic

    from merlin.targetgen import checked_source_model_diagnostic, common_transforms, target_experiment

    _, _, source, baseline, contract = retained
    compiler = tmp_path / "compiler"
    shutil.copytree(baseline, compiler)
    guidance = tmp_path / "contract"
    (guidance / "schemas").mkdir(parents=True)
    (guidance / "schemas/manifest.schema.json").write_text("{}")
    leaf = tmp_path / "leaf.json"
    leaf.write_text("{}")
    from merlin_experiments.phase2.contracts import sha256_file

    pin = {"path": str(leaf), "sha256": sha256_file(leaf)}
    edit = tmp_path / "edit.json"
    edit.write_text(json.dumps(contract))
    plan = dict(compiler=str(compiler), compiler_inventory=L._inventory(compiler),
                cpu_ids=sorted(os.sched_getaffinity(0)),
                target_experiment=pin, source_root=str(tmp_path), operator_exclusion=pin,
                sources=[str(tmp_path)], planning_inventory=pin, contract_root=str(guidance),
                profile=dict(archive=pin, workspace=str(tmp_path), repo=str(tmp_path),
                             profile_sha256="neutral-profile", snapshot_sha256="neutral-snapshot"),
                edit_contract={"path": str(edit), "sha256": sha256_file(edit)}, library_selection=pin,
                factory={}, board_config="neutral")
    leaf.write_text('{"selection": {}}')
    pin["sha256"] = sha256_file(leaf)
    target = SimpleNamespace(target="neutral", path=leaf)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda *a, **kw: target)
    monkeypatch.setattr(operator_exclusion, "select", lambda *a: object())
    monkeypatch.setattr(public_diagnostic_corpus, "freeze_public_diagnostic_corpus", lambda **kw: object())

    class NeutralService:
        profile = {"sha256": "neutral-profile"}
        snapshot_digest = "neutral-snapshot"

        def __init__(self, **kwargs):
            pass

        def _verify(self):
            pass

    monkeypatch.setattr(common_transforms, "CommonTransformService", NeutralService)
    monkeypatch.setattr(compiler_library_state, "restore_selection", lambda *a: object())
    monkeypatch.setattr(L, "restore_original_factory", lambda *a: SimpleNamespace(
        inputs={"facts": str(leaf)},
        create=lambda **kw: SimpleNamespace(_capabilities=[SimpleNamespace(
            mechanism="source_execution_diagnostic", binding=None)])))
    monkeypatch.setattr(L, "CompilerPassSelection", lambda *a, **kw: object())
    monkeypatch.setattr(checked_source_model_diagnostic, "select_checked_source_diagnostic", lambda *a, **kw: object())
    selected_policy = []

    def prepare(**kwargs):
        output = kwargs["work_root"]
        policy = SimpleNamespace(receipt_path=output / "policy.json", binding=lambda: {},
                                 validate_search_inputs=lambda: {("neutral", "member")}, **kwargs)
        selected_policy.append(policy)
        return policy

    monkeypatch.setattr(public_source_diagnostic, "prepare_public_source_diagnostic", prepare)
    monkeypatch.setattr(L, "current_thread_owns_cpu_lease", lambda: False)
    submissions = []

    def create_pool(*, revisions, **kwargs):
        def submit(revision_id, **options):
            revision = revisions.get(revision_id)
            revision.verify(authority=revisions.authority)
            submissions.append((revision, revisions, options))
            return {"job_id": "neutral-job"}

        return SimpleNamespace(root=tmp_path / "neutral-queue", submit=submit, close=lambda: None,
                               status=lambda job: {"state": "completed", "terminal": True,
                                                   "result": {"promotion": "DIAGNOSTIC_ONLY",
                                                              "screening": {"status": "UNKNOWN"}}},
                               observations=lambda: ())

    monkeypatch.setattr(L, "_create_pool", create_pool)
    out = tmp_path / "launch"
    result = L.run_plan(
        plan,
        {"deadline_unix": time.time()+600, "cleanup_reserve_seconds": 480,
         "peak_rss_bytes": 1 << 30, "total_artifact_bytes": 768 << 20},
        output=out,
        candidate_source=source if selected else None,
    )
    assert len(submissions) == 1 and result["queue_submitted"] == 1
    revision, store, options = submissions[0]
    assert store.authority is selected_policy[0].edit_authority
    assert options["selected_members"] == (("neutral", "member"),)
    if selected:
        receipt = json.loads((out / "candidate-source-selection.json").read_text())
        assert revision.revision_id == receipt["submitted_revision"]["revision_id"]
        assert revision.source_sha256 != receipt["initial_revision"]["source_sha256"]
        assert revision.parent_revision == receipt["initial_revision"]["revision_id"]
        assert result["stage"] == "completed_unscored_retained_candidate_queue"
    else:
        assert revision.parent_revision is None
        assert not (out / "candidate-source-selection.json").exists()
        assert result["stage"] == "completed_unscored_same_revision_queue"
    assert (out / "baseline/direct_codegen.py").read_bytes() == (compiler / "direct_codegen.py").read_bytes()
