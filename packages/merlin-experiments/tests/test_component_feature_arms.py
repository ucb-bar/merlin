"""Real owned processes exercise reuse; no test provider qualifies measurement."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import component_feature_arms as A
from merlin_experiments.phase2 import component_feature_inputs as F
from merlin_experiments.phase2 import corpus as C
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, exact_tree_record, sha256_file
from merlin_experiments.phase2.feedback_protocol import FeedbackValueLimits

from merlin.common import invocation_record as I
from merlin.perf.component_cost import ComponentCostScope
from merlin.targetgen import component_program


def source_corpus(root, *, extent=2, members=1):
    root.mkdir()
    capsules = root / "capsules"
    capsules.mkdir()
    rows, selected = [], []
    for index in range(members):
        program = {
            "inputs": [{"name": "A", "role": "input", "shape": [1, extent], "dtype": "operand"}],
            "nodes": [{"name": "P", "op": "copy", "inputs": ["A"]}],
            "outputs": [{"name": "Y", "value": "P"}],
        }
        typed, source = component_program.render(program, operand_dtype="i8", accumulator_dtype="i32")
        capsule = {
            "interface_mlir": "source.mlir",
            "inputs": typed["inputs"],
            "component_program": typed,
            "operation": {"op": "component_program", "attributes": {"program": program}},
        }
        directory = capsules / ("member" + str(index))
        directory.mkdir()
        (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
        (directory / "source.mlir").write_text(source)
        tree = exact_tree_record(directory)
        selected.append(
            C.PerformanceCapsule(
                "copy",
                directory.name,
                directory,
                directory.name,
                capsule,
                tree["sha256"],
                tree["n_files"],
                tree["n_bytes"],
            )
        )
        rows.append(
            {
                "family": "copy",
                "capsule": directory.name,
                "snapshot_sha256": tree["sha256"],
                "n_files": tree["n_files"],
                "n_bytes": tree["n_bytes"],
            }
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"capsules": rows}))
    return C.FrozenPerformanceCorpus(
        root, capsules, manifest, sha256_file(manifest), exact_tree_record(capsules)["sha256"], tuple(selected)
    )


def budget(**changes):
    return {
        "schema": F.B.SCHEMA,
        "max_scalar_bits": 64,
        **{"max_" + key: 10000 for key in F.B._METRICS},
        **{"max_total_" + key: 100000 for key in F.B._METRICS},
        **changes,
    }


@pytest.fixture
def selected(tmp_path):
    corpus = source_corpus(tmp_path / "corpus")
    scope = ComponentCostScope(*(document_sha256(name) for name in ("timer", "accuracy", "input")))
    inputs = F.prepare_feature_inputs(
        corpus=corpus,
        scope=scope,
        execution_budget=budget(),
        destination=tmp_path / "inputs",
        limits=FeedbackValueLimits(),
    )
    executable = Path(sys.executable).resolve()
    fixture = Path(__file__).with_name("feature_arm_control.py").resolve()
    dependency = tmp_path / "selected-dependency"
    dependency.write_text("original independently selected dependency")
    command = A.FeatureArmCommand(
        executable,
        (str(fixture), *A._TOKENS),
        (("LC_ALL", "C"),),
        tuple(sorted(F.pin(path) for path in (executable, fixture, dependency, Path(A.__file__), Path(I.__file__)))),
        "diagnostic_process.v1",
    )
    provider = A.IndependentArmFeatures(inputs, command, tmp_path / "cache", FeedbackValueLimits())
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    for directory, content in ((baseline, b"baseline"), (candidate, b"first candidate")):
        directory.mkdir()
        (directory / "compiler").write_bytes(content)
    target = tmp_path / "target.yaml"
    target.write_text("independent descriptor bytes")
    return SimpleNamespace(
        corpus=corpus,
        scope=scope,
        inputs=inputs,
        provider=provider,
        baseline=baseline,
        candidate=candidate,
        target=target,
        dependency=dependency,
        root=tmp_path,
    )


def run(selected, *, context=None, provider=None, baseline=None, candidate=None):
    root = selected.root / ("call" + str(len(tuple(selected.root.glob("call*")))))
    root.mkdir(mode=0o700)
    return (provider or selected.provider).pair(
        baseline=baseline or selected.baseline,
        candidate=candidate or selected.candidate,
        member=selected.corpus.capsules[0],
        corpus=selected.corpus,
        target_descriptor=selected.target,
        scope=selected.scope,
        workspace=root,
        timeout_s=15,
        cache_context={"fixture": "owned process diagnostic"} if context is None else context,
    )


def calls(selected):
    records = tuple(selected.provider.cache_root.glob("executions/*/invocations/*/invocation.json"))
    result = []
    for path in records:
        document = I.verify(path)
        I.require_environment(path, environment=dict(selected.provider.command.environment))
        assert document["stage"] == "independent_feature_arm"
        result.append(document)
    return result


def new_compiler(selected, name, content):
    root = selected.root / name
    root.mkdir()
    (root / "compiler").write_bytes(content)
    return root


def preserve(selected, path):
    """Keep original bytes before a test deliberately invalidates an I pin."""
    root = selected.root / "retained-originals"
    root.mkdir(exist_ok=True)
    backup = root / (str(len(tuple(root.glob("*.original")))) + ".original")
    backup.write_bytes(path.read_bytes())
    (backup.with_suffix(".json")).write_text(
        json.dumps({"original": [str(path), sha256_file(path)], "backup": F.pin(backup)})
    )


def test_changed_candidate_reuses_actual_unchanged_baseline(selected):
    first = run(selected, context={"runtime": document_sha256("runtime"), "domain": document_sha256("domain")})
    changed = new_compiler(selected, "candidate2", b"new candidate")
    second = run(
        selected,
        candidate=changed,
        context={"runtime": document_sha256("runtime"), "domain": document_sha256("domain")},
    )
    assert len(calls(selected)) == 3
    assert first[0] == second[0]
    assert first[1].compiler_sha256 != second[1].compiler_sha256
    assert first[0].inputs_sha256 == second[1].inputs_sha256
    assert first[0].functional_status == first[0].legality_status == "UNKNOWN"
    assert len({path.read_bytes() for path in selected.provider.cache_root.glob("executions/*/request.json")}) == 3
    for record in calls(selected):
        argv = record["argv"]
        assert argv.count(str(selected.baseline)) + argv.count(str(selected.candidate)) + argv.count(str(changed)) == 1


def test_changed_baseline_does_not_reuse_its_previous_observation(selected):
    run(selected)
    changed = new_compiler(selected, "baseline2", b"changed baseline")
    run(selected, baseline=changed)
    assert len(calls(selected)) == 3


@pytest.mark.parametrize("coordinate", ["runtime", "calibration", "domain", "timer", "accuracy", "input_policy"])
def test_changed_qualified_context_invalidates_both_arms(selected, coordinate):
    run(selected, context={coordinate: document_sha256("original")})
    run(selected, context={coordinate: document_sha256("changed")})
    assert len(calls(selected)) == 4


def test_changed_dependency_requires_new_selection_and_native_calls(selected):
    run(selected)
    changed = selected.root / "new-dependency"
    changed.write_text("new selected dependency")
    command = replace(
        selected.provider.command,
        source_pins=tuple(
            sorted(
                (path, digest)
                for path, digest in selected.provider.command.source_pins
                if path != str(selected.dependency)
            )
        )
        + (F.pin(changed),),
    )
    provider = replace(selected.provider, command=command)
    run(selected, provider=provider)
    assert len(calls(selected)) == 4


@pytest.mark.parametrize("mutation", ["artifact", "stdout", "record"])
def test_stale_actual_product_or_producer_is_a_miss(selected, mutation):
    run(selected)
    cached = json.loads(next(selected.provider.cache_root.glob("*.json")).read_bytes())
    if mutation == "artifact":
        product = Path(cached["product"][0])
        changed = product.parent / json.loads(product.read_bytes())["artifact_files"][0][0]
    elif mutation == "stdout":
        changed = Path(cached["record"][0]).with_name("stdout.bin")
    else:
        changed = Path(cached["record"][0])
    preserve(selected, changed)
    changed.write_bytes(b"changed actual product")
    run(selected)
    # One arm is re-extracted; its unchanged partner remains cached. The
    # original changed observation is retained, never relabelled successful.
    assert len(tuple(selected.provider.cache_root.glob("executions/*/invocations/*/invocation.json"))) == 3


def test_changed_prepared_input_bytes_refuse_before_new_process(selected):
    run(selected)
    preserve(selected, selected.inputs.members[0][1])
    selected.inputs.members[0][1].chmod(0o600)
    selected.inputs.members[0][1].write_text("changed inputs")
    with pytest.raises(StageGateError, match="inputs|changed"):
        run(selected)
    assert len(tuple(selected.provider.cache_root.glob("executions/*/invocations/*/invocation.json"))) == 2


def test_changed_selected_source_refuses_before_process(selected):
    preserve(selected, selected.dependency)
    selected.dependency.write_text("changed selected source")
    with pytest.raises(StageGateError, match="source|changed"):
        run(selected)
    assert not tuple(selected.provider.cache_root.glob("executions/*/invocations/*/invocation.json"))


def test_new_source_and_new_inputs_use_a_new_cache_identity(selected):
    run(selected)
    corpus = source_corpus(selected.root / "new-corpus", extent=3)
    inputs = F.prepare_feature_inputs(
        corpus=corpus,
        scope=selected.scope,
        execution_budget=budget(),
        destination=selected.root / "new-inputs",
        limits=FeedbackValueLimits(),
    )
    provider = replace(selected.provider, inputs=inputs)
    current = SimpleNamespace(**{**selected.__dict__, "corpus": corpus, "inputs": inputs, "provider": provider})
    run(current)
    assert len(calls(selected)) == 4


def test_whole_roster_budget_refuses_before_any_value_allocation(tmp_path, monkeypatch):
    corpus = source_corpus(tmp_path / "corpus", members=2)
    scope = ComponentCostScope(*(document_sha256(name) for name in ("timer", "accuracy", "input")))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no value realization before aggregate admission")

    monkeypatch.setattr(F.capsule_inputs, "materialize_capsule_leaves", forbidden)
    destination = tmp_path / "inputs"
    with pytest.raises(StageGateError, match="whole-roster"):
        F.prepare_feature_inputs(
            corpus=corpus,
            scope=scope,
            execution_budget=budget(max_total_materialized_elements=6),
            destination=destination,
            limits=FeedbackValueLimits(),
        )
    assert not destination.exists()


def test_large_extents_refuse_before_tensor_realization(tmp_path, monkeypatch):
    corpus = source_corpus(tmp_path / "corpus", extent=10**12)
    scope = ComponentCostScope(*(document_sha256(name) for name in ("timer", "accuracy", "input")))
    monkeypatch.setattr(F.capsule_inputs, "materialize_capsule_leaves", lambda *_args: pytest.fail("allocated values"))
    with pytest.raises(StageGateError, match="budgets"):
        F.prepare_feature_inputs(
            corpus=corpus,
            scope=scope,
            execution_budget=budget(),
            destination=tmp_path / "inputs",
            limits=FeedbackValueLimits(),
        )


def test_diagnostic_command_cannot_become_independent_feedback(selected):
    with pytest.raises(StageGateError, match="strict process owner"):
        selected.provider.require_feedback_owner(SimpleNamespace(qualification=object()), selected.provider.pair)


def test_saved_input_owner_cannot_mint_preparation(selected):
    with pytest.raises(StageGateError, match="actual ordinary preparation"):
        replace(selected.inputs).verify()


def test_dependency_grant_cannot_contain_another_compiler(selected):
    command = replace(
        selected.provider.command,
        source_pins=selected.provider.command.source_pins + (F.pin(selected.candidate / "compiler"),),
    )
    with pytest.raises(StageGateError, match="another compiler"):
        run(selected, provider=replace(selected.provider, command=command))


def test_source_interface_mismatch_refuses_without_prepared_values(tmp_path):
    corpus = source_corpus(tmp_path / "corpus")
    (corpus.capsules[0].source_dir / "source.mlir").write_text("wrong source")
    scope = ComponentCostScope(*(document_sha256(name) for name in ("timer", "accuracy", "input")))
    with pytest.raises(StageGateError, match="changed"):
        F.prepare_feature_inputs(
            corpus=corpus,
            scope=scope,
            execution_budget=budget(),
            destination=tmp_path / "inputs",
            limits=FeedbackValueLimits(),
        )


def test_symlink_cache_cannot_read_excluded_file(selected):
    run(selected)
    cache = next(selected.provider.cache_root.glob("*.json"))
    excluded = selected.root / "excluded"
    excluded.write_text("original excluded bytes")
    original = excluded.read_bytes()
    cache.unlink()
    cache.symlink_to(excluded)
    run(selected)
    assert excluded.read_bytes() == original
    assert not cache.is_symlink()


def test_changed_scope_requires_new_actual_input_preparation(selected):
    changed_scope = replace(selected.scope, timer_sha256=document_sha256("changed timer"))
    selected.scope = changed_scope
    with pytest.raises(StageGateError, match="scope"):
        run(selected)
    assert not tuple(selected.provider.cache_root.glob("executions/*/invocations/*/invocation.json"))


def test_mutable_descriptor_cannot_change_live_prepared_inputs(selected):
    selected.corpus.capsules[0].descriptor["stimulus_range"] = [-8, 8]
    with pytest.raises(StageGateError, match="actual ordinary preparation"):
        run(selected)


def test_native_observation_cannot_read_an_excluded_artifact(selected):
    run(selected)
    cached = json.loads(next(selected.provider.cache_root.glob("*.json")).read_bytes())
    product = Path(cached["product"][0])
    excluded = selected.root / "excluded-artifact"
    excluded.write_bytes(b"owned excluded bytes")
    original = excluded.read_bytes()
    value = json.loads(product.read_bytes())
    value["artifact_files"] = [[str(excluded), sha256_file(excluded)]]
    preserve(selected, product)
    product.write_text(json.dumps(value))
    with pytest.raises(StageGateError, match="escapes"):
        A._observation(product, selected.provider.limits)
    assert excluded.read_bytes() == original


def test_cache_cannot_join_a_foreign_native_record(selected):
    run(selected)
    cache = next(selected.provider.cache_root.glob("*.json"))
    saved = json.loads(cache.read_bytes())
    foreign = selected.root / "excluded-invocation.json"
    foreign.write_text("owned excluded record")
    saved["record"] = [str(foreign), sha256_file(foreign)]
    with pytest.raises(StageGateError, match="escapes"):
        selected.provider._reopen(saved, saved["selection"], selected.provider.cache_root / "executions")
    assert foreign.read_text() == "owned excluded record"


def test_default_qualification_calls_keep_complete_native_denominator(selected):
    # No explicit feedback cache context: each independently prepared control
    # receives its own two actual invocations in the original private owner.
    for index in range(2):
        workspace = selected.root / ("qualification-control" + str(index))
        workspace.mkdir(mode=0o700)
        selected.provider.pair(
            baseline=selected.baseline,
            candidate=selected.candidate,
            member=selected.corpus.capsules[0],
            corpus=selected.corpus,
            target_descriptor=selected.target,
            scope=selected.scope,
            workspace=workspace,
            timeout_s=15,
        )
        records = tuple(workspace.glob("*/invocations/*/invocation.json"))
        assert len(records) == 2
        for record in records:
            I.verify(record)
    assert not tuple(selected.provider.cache_root.glob("*.json"))


@pytest.mark.parametrize("defect", ["missing-inputs", "duplicate-compiler", "unknown-placeholder"])
def test_closed_command_refuses_ambiguous_native_consumption(selected, defect):
    argv = selected.provider.command.argv_template
    if defect == "missing-inputs":
        argv = tuple(token for token in argv if token != "{inputs}")
    elif defect == "duplicate-compiler":
        argv += ("{compiler}",)
    else:
        argv += ("{other_arm}",)
    with pytest.raises(StageGateError, match="closed explicit"):
        replace(selected.provider.command, argv_template=argv).verify()


def test_generated_compiler_members_invalidate_unchanged_source_identity(selected):
    generated = selected.baseline / "build"
    generated.mkdir()
    (generated / "consumed-dependency").write_text("first generated dependency")
    first = run(selected)
    changed = new_compiler(selected, "baseline2", b"baseline")
    (changed / "build").mkdir()
    (changed / "build" / "consumed-dependency").write_text("different generated dependency")
    second = run(selected, baseline=changed)
    assert first[0].compiler_sha256 == second[0].compiler_sha256
    assert len(calls(selected)) == 3


def test_compiler_directory_alias_refuses_before_native_consumption(selected):
    excluded = selected.root / "excluded-directory"
    excluded.mkdir()
    (excluded / "data").write_text("owned excluded bytes")
    (selected.baseline / "linked").symlink_to(excluded, target_is_directory=True)
    with pytest.raises(StageGateError, match="canonical unlinked"):
        run(selected)
    assert not tuple(selected.provider.cache_root.glob("executions/*/invocations/*/invocation.json"))


def test_compiler_whole_byte_budget_precedes_hash_allocation(selected, monkeypatch):
    (selected.baseline / "compiler").write_bytes(b"x" * 8192)
    provider = replace(selected.provider, limits=replace(selected.provider.limits, max_bytes=4096))
    monkeypatch.setattr(A, "hash_tree", lambda *_args: pytest.fail("hashed oversized compiler before admission"))
    with pytest.raises(StageGateError, match="whole byte/member budget"):
        run(selected, provider=provider)


def test_strict_command_reapplies_only_selected_environment(selected):
    executable = selected.provider.command.executable
    sandbox = Path("/usr/bin/true").resolve()
    command = replace(
        selected.provider.command,
        mode="strict_namespace.v1",
        runtime=(A.RuntimeGrant(executable, "/usr/bin/selected-tool", sha256_file(executable)),),
        sandbox=sandbox,
        source_pins=selected.provider.command.source_pins + (F.pin(sandbox),),
    )
    command.verify()
    provider = replace(selected.provider, command=command)
    argv = provider._argv(
        selected.baseline,
        selected.corpus.capsules[0].source_dir / "source.mlir",
        selected.inputs.members[0][1],
        selected.root / "request.json",
        selected.root / "output" / "observation.json",
    )
    assert argv[argv.index("--setenv") : argv.index("--setenv") + 3] == ["--setenv", "LC_ALL", "C"]
    assert "--clearenv" in argv and "--unshare-all" in argv
    assert "--proc" not in argv
