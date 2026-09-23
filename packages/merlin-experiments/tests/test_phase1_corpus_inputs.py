"""Run-owned corpus inputs survive live-source changes without changing the grading policy."""

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.target_experiment import load_target_experiment


def _capsule(root, name, label="public"):
    directory = root / name
    directory.mkdir(parents=True)
    document = {
        "name": name,
        "kind": "isa",
        "source_role": "handauthored_compiler_test",
        "label": label,
        "operation": {"op": "matmul", "attributes": {}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"instruction_classes": [], "modes": {}},
        "required_oracle_tiers": ["L0", "L2"],
        "interface_mlir": "capsule.interface.mlir",
    }
    (directory / "capsule.yaml").write_text(yaml.safe_dump(document))
    (directory / "capsule.interface.mlir").write_text("module {}\n")
    (directory / "golden.yaml").write_text("outputs: {value: PRIVATE_SENTINEL}\n")
    return directory


def test_fullsuite_view_uses_frozen_bytes_and_preserves_nested_resources(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import corpus_inputs as CI

    contract = repo_root() / "merlin/contract"
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(contract))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    corpus = tmp_path / "corpus"
    public = _capsule(corpus / "isa", "member")
    _capsule(corpus / "hidden", "hidden_member", "hidden")
    (public / "nested").mkdir()
    (public / "nested/resource.bin").write_bytes(b"dependent resource")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "fixture", "capsule_corpus": str(corpus / "isa")}))
    te = load_target_experiment(descriptor)
    run = tmp_path / "run"
    run.mkdir()
    bundle = {"bundle_id": "fixture", "allowed": [], "denied": []}
    effective, record = CI.stage(run, te, bundle, contract=contract, capsules_root=corpus)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    view = CI.resolve(ws, effective, record, repo=tmp_path)
    assert (view.public / "isa/member/nested/resource.bin").read_bytes() == b"dependent resource"
    assert not (view.public / "hidden").exists()
    assert (view.public / "isa/member/capsule.yaml").read_bytes() == (public / "capsule.yaml").read_bytes()
    (public / "golden.yaml").write_text("changed live source")
    resumed = CI.resolve(ws, effective, record, repo=tmp_path)
    assert (resumed.public / "isa/member/golden.yaml").read_text() == "outputs: {value: PRIVATE_SENTINEL}\n"
    assert effective["allowed"] == bundle["allowed"]
    assert "PRIVATE_SENTINEL" not in str(BW.snapshot_record(ws))


@pytest.fixture
def corpus_fixture(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_runner

    contract = repo_root() / "merlin/contract"
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(contract))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "public-cas"))
    monkeypatch.setattr(capsule_runner, "oracle_adapters", lambda *args: {"L2": lambda: None, "L3": lambda: None})
    primary = tmp_path / "corpus/isa"
    member = _capsule(primary, "member")
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "fixture", "capsule_corpus": str(primary)}))
    run = tmp_path / "run"
    run.mkdir()
    bundle = {"bundle_id": "fixture", "allowed": [], "denied": []}
    return load_target_experiment(descriptor), member, run, bundle, contract


def test_default_view_does_not_follow_republication_or_changed_source(corpus_fixture, tmp_path):
    from merlin_experiments.corpus import admission as corpus_workflow
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, source, run, bundle, contract = corpus_fixture
    effective, record = CI.stage(run, te, bundle, contract=contract)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    view = CI.resolve(ws, effective, record, repo=tmp_path)
    expected = (view.public / "member/capsule.yaml").read_bytes()
    document = yaml.safe_load((source / "capsule.yaml").read_text())
    document["required_oracle_tiers"] = ["L0", "L3"]
    (source / "capsule.yaml").write_text(yaml.safe_dump(document))
    published = corpus_workflow.public_capsules_for(te)
    assert (published / "member/capsule.yaml").read_bytes() != expected
    resumed = CI.resolve(ws, effective, record, repo=tmp_path)
    assert (resumed.public / "member/capsule.yaml").read_bytes() == expected
    assert yaml.safe_load((resumed.policy / "0/member/capsule.yaml").read_text())["required_oracle_tiers"] == [
        "L0",
        "L2",
    ]
    assert not (tmp_path / "public-cas").exists()


def test_frozen_corpus_mutation_cannot_resume(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, _, run, bundle, contract = corpus_fixture
    effective, record = CI.stage(run, te, bundle, contract=contract)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    view = CI.resolve(ws, effective, record, repo=tmp_path)
    member = view.public / "member/golden.yaml"
    member.chmod(0o600)
    member.write_text("tampered frozen bytes")
    with pytest.raises(RuntimeError, match="snapshot"):
        CI.resolve(ws, effective, record, repo=tmp_path)


def test_declared_external_resource_refuses_before_snapshot(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, source, run, bundle, contract = corpus_fixture
    document = yaml.safe_load((source / "capsule.yaml").read_text())
    document["interface_mlir"] = "../external.mlir"
    (source / "capsule.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="non-capsule-local resource: interface_mlir"):
        CI.stage(run, te, bundle, contract=contract)
    assert not BW.bundle_snapshot_root(tmp_path / "workspace").exists()


def test_real_effective_bundle_archive_resumes_without_live_corpus(corpus_fixture, tmp_path):
    import shutil

    from merlin_experiments.phase1 import corpus_inputs as CI

    te, source, run, bundle, contract = corpus_fixture
    authored = tmp_path / "authored.yaml"
    authored.write_text(yaml.safe_dump(bundle))
    prepared = CI.prepare_bundle(run, te, authored, bundle, contract=contract)
    assert (run / "authored_input_bundle_manifest.yaml").read_bytes() == authored.read_bytes()
    assert prepared.authored_sha256 != prepared.effective_sha256
    environment = {
        "bundle_manifest_sha256": prepared.effective_sha256,
        "authored_bundle_manifest_sha256": prepared.authored_sha256,
        "public_corpus_input": prepared.corpus_record,
    }
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, prepared.bundle, repo=tmp_path)
    shutil.rmtree(source)
    resumed = CI.prepare_bundle(run, te, authored, bundle, contract=contract, environment=environment)
    view = CI.resolve(ws, resumed.bundle, resumed.corpus_record, repo=tmp_path)
    assert (view.public / "member/capsule.yaml").is_file()
    assert (view.contract / "schemas/capsule.schema.json").is_file()
    assert not BW.is_exposed(BW.base_argv(ws, resumed.bundle, repo=tmp_path), view.public / "member/golden.yaml")
    assert not BW.is_exposed(
        BW.base_argv(ws, resumed.bundle, repo=tmp_path), run / "private_corpus_input/public/member/golden.yaml"
    )
    with pytest.raises(FileExistsError):
        CI.prepare_bundle(run, te, authored, bundle, contract=contract)


def test_effective_bundle_keeps_native_treatment_binding_to_authored_bytes(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI
    from merlin_experiments.phase1 import run_inputs as RI

    te, _, run, bundle, contract = corpus_fixture
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    authored = bundle_dir / "input_bundle_manifest.yaml"
    authored.write_text(yaml.safe_dump(bundle))
    prepared = CI.prepare_bundle(run, te, authored, bundle, contract=contract)
    ws = tmp_path / "workspace"
    ws.mkdir()
    for directory in (ws, run):
        (directory / "TASK.md").write_text("synthetic task")
    receipt = RI.treatment_snapshot_record(ws, run, bundle_dir, [])
    rows = {row["name"]: row for row in receipt["files"]}
    assert rows["archived_bundle/input_bundle_manifest.yaml"]["sha256"] == prepared.effective_sha256
    assert rows["archived_bundle/authored_input_bundle_manifest.yaml"]["sha256"] == prepared.authored_sha256
    authored.write_text(authored.read_text() + "# drift\n")
    with pytest.raises(RuntimeError, match="source input bundle manifest changed"):
        RI.treatment_snapshot_record(ws, run, bundle_dir, [])


def test_native_resume_cannot_remove_or_change_recorded_operator_review():
    import ast

    from merlin.common.paths import module_source_path

    controller = module_source_path("merlin_experiments.phase1.session")
    tree = ast.parse(controller.read_text())
    check = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "_resuming and _environment_record.get('corpus_review') != _corpus_review"
    )
    code = compile(ast.Module(body=[check], type_ignores=[]), str(controller), "exec")
    original = {"review_digest": "operator-reviewed-identity"}
    scope = {"_resuming": True, "_environment_record": {"corpus_review": original}, "_corpus_review": original}
    exec(code, scope)
    for replacement in (None, {"review_digest": "different-review"}):
        scope["_corpus_review"] = replacement
        with pytest.raises(RuntimeError, match="corpus-review identity changed or was removed"):
            exec(code, scope)


def test_default_materialization_keeps_declared_loader_assets(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, member, run, bundle, contract = corpus_fixture
    document = yaml.safe_load((member / "capsule.yaml").read_text())
    document["operation"]["attributes"].update(
        {"weights": "capsule.weights.safetensors", "loader_dependencies": "capsule.loader_deps"}
    )
    (member / "capsule.yaml").write_text(yaml.safe_dump(document))
    (member / "capsule.weights.safetensors").write_bytes(b"synthetic opaque weights")
    (member / "capsule.loader_deps/nested").mkdir(parents=True)
    (member / "capsule.loader_deps/nested/loader.py").write_text("VALUE = 3\n")
    effective, record = CI.stage(run, te, bundle, contract=contract)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    view = CI.resolve(ws, effective, record, repo=tmp_path)
    assert (view.public / "member/capsule.weights.safetensors").read_bytes() == b"synthetic opaque weights"
    assert (view.public / "member/capsule.loader_deps/nested/loader.py").read_text() == "VALUE = 3\n"


@pytest.mark.parametrize("change", ["effective_grant", "effective_bytes", "authored_bytes", "record_owner"])
def test_resume_refuses_declaration_drift(corpus_fixture, tmp_path, change):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, _, run, bundle, contract = corpus_fixture
    authored = tmp_path / "authored.yaml"
    authored.write_text(yaml.safe_dump(bundle))
    prepared = CI.prepare_bundle(run, te, authored, bundle, contract=contract)
    environment = {
        "bundle_manifest_sha256": prepared.effective_sha256,
        "authored_bundle_manifest_sha256": prepared.authored_sha256,
        "public_corpus_input": dict(prepared.corpus_record),
    }
    effective = run / "input_bundle_manifest.yaml"
    if change == "effective_grant":
        document = yaml.safe_load(effective.read_text())
        document["allowed"].append({"path": str(tmp_path)})
        effective.write_text(yaml.safe_dump(document))
    elif change == "effective_bytes":
        effective.write_text(effective.read_text() + "# changed bytes\n")
    elif change == "authored_bytes":
        archive = run / "authored_input_bundle_manifest.yaml"
        archive.write_text(archive.read_text() + "# changed bytes\n")
    else:
        environment["public_corpus_input"]["staging_path"] = str(tmp_path / "other-run")
    before = effective.read_bytes()
    with pytest.raises(RuntimeError, match="resume refused"):
        CI.prepare_bundle(run, te, authored, bundle, contract=contract, environment=environment)
    assert effective.read_bytes() == before


def test_descriptor_exclusion_does_not_remove_policy_inventory(corpus_fixture, tmp_path):
    import json

    from merlin_experiments.phase1 import corpus_inputs as CI

    te, _, run, bundle, contract = corpus_fixture
    _capsule(te.capsule_corpus, "excluded_member")
    document = yaml.safe_load(te.path.read_text())
    document["grading"] = {
        "capability_exclude_capsules": ["excluded_member"],
        "expected_cohort": {"source_capsules": 2, "admitted_capsules": 1},
    }
    te.path.write_text(yaml.safe_dump(document))
    te = load_target_experiment(te.path)
    effective, record = CI.stage(run, te, bundle, contract=contract)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    view = CI.resolve(ws, effective, record, repo=tmp_path)
    assert not (view.public / "excluded_member").exists()
    assert (view.policy / "0/excluded_member/capsule.yaml").is_file()
    admission = json.loads((view.public / ".cohort_admission.json").read_text())
    assert admission["n_source_capsules"] == 2
    assert admission["n_admitted_capsules"] == 1
    assert admission["policy"] == "descriptor_capability_and_resource_v1"


def test_staged_unreviewed_bytes_cannot_borrow_restored_original_snapshot(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, member, run, bundle, contract = corpus_fixture
    bundle["host_inputs"] = [{"path": str(te.capsule_corpus)}]
    golden = member / "golden.yaml"
    reviewed = golden.read_bytes()
    golden.write_text("unreviewed substitution during staging")
    effective, record = CI.stage(run, te, bundle, contract=contract)
    golden.write_bytes(reviewed)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    with pytest.raises(RuntimeError, match="differs from original frozen source"):
        CI.resolve(ws, effective, record, repo=tmp_path, reviewed_roots=tuple(te.graded_roots()))


def test_reviewed_binding_requires_original_snapshot_coverage(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, _, run, bundle, contract = corpus_fixture
    effective, record = CI.stage(run, te, bundle, contract=contract)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    with pytest.raises(RuntimeError, match="outside its original declared snapshot"):
        CI.resolve(ws, effective, record, repo=tmp_path, reviewed_roots=tuple(te.graded_roots()))


def test_reviewed_binding_does_not_approve_override_by_broad_snapshot_grant(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, _, run, bundle, contract = corpus_fixture
    override = tmp_path / "unreviewed"
    _capsule(override, "extra")
    bundle["host_inputs"] = [{"path": str(te.capsule_corpus)}, {"path": str(override)}]
    effective, record = CI.stage(run, te, bundle, contract=contract, capsules_root=override)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    assert CI.resolve(ws, effective, record, repo=tmp_path).public.is_dir()
    with pytest.raises(RuntimeError, match="outside its reviewed original roots"):
        CI.resolve(ws, effective, record, repo=tmp_path, reviewed_roots=tuple(te.graded_roots()))


def test_reviewed_originals_match_and_resume_after_live_sources_change(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, member, run, bundle, contract = corpus_fixture
    reviewed = tuple(te.graded_roots())
    bundle["host_inputs"] = [{"path": str(root)} for root in reviewed]
    effective, record = CI.stage(run, te, bundle, contract=contract, capsules_root=te.capsule_corpus)
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    (member / "golden.yaml").write_text("later live source")
    view = CI.resolve(ws, effective, record, repo=tmp_path, reviewed_roots=reviewed)
    assert (view.public / "member/golden.yaml").read_text() != "later live source"


def test_admission_uses_copied_schema_and_declared_schema_snapshot_is_compared(corpus_fixture, tmp_path, monkeypatch):
    import shutil

    from merlin_experiments.phase1 import corpus_inputs as CI

    te, _, run, bundle, original_contract = corpus_fixture
    contract = tmp_path / "contract"
    shutil.copytree(original_contract / "schemas", contract / "schemas")
    schema = contract / "schemas/capsule.schema.json"
    original = schema.read_bytes()
    real_copy = CI.copy_input

    def copy_then_change_schema(source, destination, **kwargs):
        result = real_copy(source, destination, **kwargs)
        if source == contract / "schemas":
            schema.write_text("invalid live schema after initial copy")
        return result

    monkeypatch.setattr(CI, "copy_input", copy_then_change_schema)
    # The existing cohort transform/coverage validation reads YAML, not live schemas.
    # Discovery and asset validation must use the copied schema even during preparation.
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(contract))
    bundle["host_inputs"] = [{"path": str(contract / "schemas")}]
    effective, record = CI.stage(run, te, bundle, contract=contract)
    assert schema.read_bytes() != original
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, effective, repo=tmp_path)
    with pytest.raises(RuntimeError, match="differs from original frozen source"):
        CI.resolve(ws, effective, record, repo=tmp_path)


def test_source_category_name_collision_is_not_silently_overwritten(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, _, run, bundle, contract = corpus_fixture
    _capsule(te.capsule_corpus.parent / "layers", "member")
    with pytest.raises(ValueError, match="duplicate capsule names"):
        CI.stage(run, te, bundle, contract=contract)
    assert not BW.bundle_snapshot_root(tmp_path / "workspace").exists()


def test_prepared_resources_are_private_before_and_after_freeze(corpus_fixture, tmp_path):
    from merlin_experiments.phase1 import corpus_inputs as CI

    te, _, run, bundle, contract = corpus_fixture
    authored = tmp_path / "authored.yaml"
    authored.write_text(yaml.safe_dump(bundle))
    prepared = CI.prepare_bundle(run, te, authored, bundle, contract=contract)
    private = run / "private_corpus_input"
    assert not private.stat().st_mode & 0o077
    assert not (run / "input_bundle_manifest.yaml").stat().st_mode & 0o077
    assert not (run / "authored_input_bundle_manifest.yaml").stat().st_mode & 0o077
    assert all(not member.stat().st_mode & 0o222 for member in [private, *private.rglob("*")])
    ws = tmp_path / "workspace"
    BW.materialize_bundle_inputs(ws, prepared.bundle, repo=tmp_path)
    assert BW._snapshot_grants(ws, prepared.bundle, tmp_path)[1] == []
    assert not BW.bundle_snapshot_root(ws).stat().st_mode & 0o077
