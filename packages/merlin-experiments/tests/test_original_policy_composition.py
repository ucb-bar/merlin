"""Only the issued original owner may compose before the final closed policy."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as C

from merlin.common.paths import repo_root
from merlin.targetgen import package_runtime as P
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "0")
    public = tmp_path / "public"
    public.mkdir()
    (public / "allowed.txt").write_text("public input")
    hidden = public / "hidden"
    hidden.mkdir()
    secret = hidden / "secret"
    secret.write_text("private input")
    workspace = tmp_path / "original"
    workspace.mkdir()
    repo = repo_root()
    bundle = {
        "arm": "raw_baseline",
        "tools": [],
        "allowed": [{"path": str(public)}],
        "denied": [],
        "host_inputs": [{"path": str(hidden)}],
    }
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V3, python_source_root=repo / "src", host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(repo / "src" / row["path"].removeprefix("merlin/python/"))
    archive = tmp_path / "archive.yaml"
    archive.write_text(yaml.safe_dump(bundle))
    BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
    owner = R.CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo)
    [frozen_hidden] = BW.snapshot_input_paths(workspace, bundle, [hidden], repo=repo)
    [frozen_public] = BW.snapshot_input_paths(workspace, bundle, [public], repo=repo)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("# unchanged candidate\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    paths = SimpleNamespace(repo=repo)
    sim = SimpleNamespace()
    monkeypatch.setattr(TC.ToolchainPaths, "from_checkout", lambda: paths)
    monkeypatch.setattr(TC, "_sim", lambda _: sim)
    monkeypatch.setattr(TC, "curated_harness_dir", lambda _: "")
    monkeypatch.setattr(TC, "toolchain_binds", lambda *_a, **_k: ["--ro-bind", str(runtime), "/runtime-view"])
    monkeypatch.setattr(TC, "required_tool_probes", lambda *_a, **_k: (SimpleNamespace(label="tool", cmd="true"),))
    monkeypatch.setattr(TC, "sandbox_env", lambda *_a, **_k: "export FIXTURE=1; ")
    monkeypatch.setattr(BW, "base_argv", lambda ws, *_a, **_k: ["box", "--bind", str(ws), str(ws)])
    discoveries, answers = [], []

    def discover(_target, **_kwargs):
        discoveries.append(tuple(answers))
        return list(answers)

    monkeypatch.setattr(C, "answer_surfaces", discover)
    monkeypatch.setattr(BW, "answer_surfaces", discover)
    value = SimpleNamespace(
        owner=owner,
        candidate=candidate,
        staging=tmp_path / "staging",
        target=SimpleNamespace(target="fixture", capsule_corpus=None, corpus_siblings=lambda: ()),
        paths=paths,
        sim=sim,
        runtime=runtime,
        frozen_hidden=frozen_hidden,
        frozen_public=frozen_public,
        discoveries=discoveries,
        answers=answers,
        root=tmp_path,
    )
    try:
        yield value
    finally:
        BW.remove_bundle_snapshot(workspace)


def policy(value):
    call = value.root / f"call-{len(value.discoveries)}"
    call.mkdir()
    return P.active_package_executor()._policy(call, value.candidate)


def test_one_fresh_final_pass_and_no_unmasked_policy_is_constructed(selected, monkeypatch):
    value = selected
    answer = value.runtime / "answer"
    answer.write_text("answer")
    value.answers.append(AnswerSurface("answer", answer, "file", "golden"))
    assert BW.is_exposed(["--ro-bind", str(value.runtime), "/runtime-view"], Path("/runtime-view/answer"))
    constructor = C.PackageSandboxPolicy
    constructed = []

    def closed(*args, **kwargs):
        assert len(value.discoveries) == 1
        result = constructor(*args, **kwargs)
        assert not BW.is_exposed(result.argv, Path("/runtime-view/answer"))
        assert not BW.is_exposed(result.argv, value.frozen_hidden / "secret")
        constructed.append(result)
        return result

    monkeypatch.setattr(C, "PackageSandboxPolicy", closed)
    with R.original_compiler_execution(
        value.owner, target_experiment=value.target, candidate=value.candidate, work_root=value.staging
    ):
        result = policy(value)
    assert constructed == [result] and len(value.discoveries) == 1
    assert result.coverage_gap == () and result.selected_inputs.surfaces == tuple(value.answers)
    assert "export FIXTURE=1; export PYTHONPATH=" in result.env_prefix


def test_new_answer_and_new_private_hardlink_are_masked_on_next_call(selected):
    value = selected
    with R.original_compiler_execution(
        value.owner, target_experiment=value.target, candidate=value.candidate, work_root=value.staging
    ):
        policy(value)
        answer = value.runtime / "new-answer"
        answer.write_text("new answer")
        value.answers.append(AnswerSurface("new", answer, "file", "golden"))
        os.link(value.frozen_hidden / "secret", value.runtime / "private-alias")
        result = policy(value)
    assert len(value.discoveries) == 2
    assert not BW.is_exposed(result.argv, Path("/runtime-view/new-answer"))
    assert not BW.is_exposed(result.argv, Path("/runtime-view/private-alias"))


def test_explicit_selected_surfaces_are_not_lost(selected):
    value = selected
    explicit = value.runtime / "explicit"
    explicit.write_text("explicitly withheld")
    inputs = C.PackageSandboxInputs(
        value.paths, value.sim, "", (AnswerSurface("explicit", explicit, "file", "oracle"),)
    )
    with R.original_compiler_execution(
        value.owner,
        target_experiment=value.target,
        candidate=value.candidate,
        work_root=value.staging,
        sandbox_inputs=inputs,
    ):
        result = policy(value)
    assert len(value.discoveries) == 1 and result.selected_inputs is inputs
    assert not BW.is_exposed(result.argv, Path("/runtime-view/explicit"))


def test_public_source_mutation_before_final_masks_refuses(selected, monkeypatch):
    value = selected
    compose = R._public_package_parts

    def changed(*args):
        result = compose(*args)
        source = value.frozen_public / "allowed.txt"
        source.chmod(0o644)
        source.write_text("changed selected public bytes")
        return result

    monkeypatch.setattr(R, "_public_package_parts", changed)
    with pytest.raises(RuntimeError, match="snapshot|changed|content"):
        with R.original_compiler_execution(
            value.owner, target_experiment=value.target, candidate=value.candidate, work_root=value.staging
        ):
            policy(value)


def test_changed_or_expired_original_owner_cannot_use_composer(selected):
    value = selected
    assert R._original_package_composition(value.target, None, None) is None
    with R.original_compiler_execution(
        value.owner, target_experiment=value.target, candidate=value.candidate, work_root=value.staging
    ):
        executor = P.active_package_executor()
        assert R._original_package_composition(value.target, object(), executor.policy_scope) is None
        assert R._original_package_composition(object(), None, executor.policy_scope) is None
        assert R._original_package_composition(value.target, None, None) is None
        compose = R._original_package_composition(value.target, None, executor.policy_scope)
        for target, scope, surfaces in [
            (object(), executor.policy_scope, ()),
            (value.target, object(), ()),
            (value.target, executor.policy_scope, (object(),)),
        ]:
            with pytest.raises(ValueError, match="changed its owner"):
                compose([], "", target=target, scope=scope, selected_surfaces=surfaces)
    with pytest.raises(ValueError, match="changed its owner|expired"):
        compose([], "", target=value.target, scope=executor.policy_scope, selected_surfaces=())


def test_standalone_public_policy_still_applies_final_masks(selected):
    value = selected
    answer = value.root / "answer"
    answer.write_text("private")
    value.answers.append(AnswerSurface("answer", answer, "file", "golden"))
    base = C.PackageSandboxPolicy(
        ("box", "--ro-bind", str(value.root), str(value.root)), (), (), value.root, value.candidate, value.target, ""
    )
    result = R.public_package_policy(base, target=value.target, owner=value.owner)
    assert len(value.discoveries) == 1
    assert not BW.is_exposed(result.argv, answer)


def test_composed_and_legacy_policies_have_the_same_visible_files(selected, monkeypatch):
    value = selected
    answer = value.runtime / "answer"
    answer.write_text("answer")
    value.answers.append(AnswerSurface("answer", answer, "file", "golden"))
    os.link(value.frozen_hidden / "secret", value.runtime / "alias")
    with R.original_compiler_execution(
        value.owner, target_experiment=value.target, candidate=value.candidate, work_root=value.staging
    ):
        composed = policy(value)
        with monkeypatch.context() as patch:
            patch.setattr(R, "_original_package_composition", lambda *_: None)
            legacy = policy(value)
    assert len(value.discoveries) == 3  # One final discovery versus the old two.
    assert composed.env_prefix == legacy.env_prefix
    assert composed.selected_inputs == legacy.selected_inputs
    assert BW.is_exposed(composed.argv, value.root / "public/allowed.txt")
    for path in [
        value.candidate,
        value.root / "public/allowed.txt",
        value.frozen_hidden / "secret",
        Path("/runtime-view/answer"),
        Path("/runtime-view/alias"),
    ]:
        assert BW.is_exposed(composed.argv, path) == BW.is_exposed(legacy.argv, path)


def test_answer_created_during_grant_composition_is_masked(selected, monkeypatch):
    value = selected
    original = R._public_package_parts

    def compose(*args):
        result = original(*args)
        answer = value.runtime / "late-answer"
        answer.write_text("answer created after tool selection")
        value.answers.append(AnswerSurface("late", answer, "file", "golden"))
        return result

    monkeypatch.setattr(R, "_public_package_parts", compose)
    with R.original_compiler_execution(
        value.owner, target_experiment=value.target, candidate=value.candidate, work_root=value.staging
    ):
        result = policy(value)
    assert len(value.discoveries) == 1
    assert not BW.is_exposed(result.argv, Path("/runtime-view/late-answer"))


def test_failed_final_coverage_never_constructs_a_policy(selected, monkeypatch):
    value = selected
    answer = value.runtime / "answer"
    answer.write_text("must be masked")
    value.answers.append(AnswerSurface("answer", answer, "file", "golden"))
    monkeypatch.setattr(BW, "apply_answer_masks", lambda argv, _surfaces: list(argv))
    monkeypatch.setattr(C, "PackageSandboxPolicy", lambda *_a, **_k: pytest.fail("unmasked policy constructed"))
    with pytest.raises(C.CampaignGateError, match="exposes derived"):
        with R.original_compiler_execution(
            value.owner, target_experiment=value.target, candidate=value.candidate, work_root=value.staging
        ):
            policy(value)
