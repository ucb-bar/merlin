"""Suite isolation, structural claim splitting and snapshot import ownership."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import merlin_dir

SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))
import perf_suite as SUITE
import perf_snapshot as SNAP


def _restore_snapshot_permissions(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o700)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o700 if path.is_dir() else 0o600)


@pytest.fixture
def snapshot(tmp_path):
    src = tmp_path / "src"
    (src / "code").mkdir(parents=True)
    (src / "code/a.py").write_text("value = 1\n")
    output = tmp_path / "products"
    output.mkdir()
    dst = tmp_path / "snapshot"
    SNAP.create(src, dst, output_root=output, source_roots=("code",))
    yield src, dst
    _restore_snapshot_permissions(dst)


def test_snapshot_is_a_copy_and_live_edits_do_not_change_it(snapshot):
    src, dst = snapshot
    first = SNAP.verify(dst)
    (src / "code/a.py").write_text("value = 2\n")
    assert SNAP.verify(dst) == first
    assert (src / "code/a.py").stat().st_ino != (dst / "code/a.py").stat().st_ino


def test_default_snapshot_pins_canonical_numeric_schema_data(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "snapshot"
    for relative in SNAP.SOURCE_ROOTS:
        (source / relative).mkdir(parents=True, exist_ok=True)
    schema_root = source / "merlin/schemas"
    schema_root.mkdir(parents=True, exist_ok=True)
    leaves = ("quant_formats.registry.yaml", "quant_format.schema.yaml")
    for name in leaves:
        (schema_root / name).write_text("canonical: " + name + "\n")
    try:
        SNAP.create(source, destination, output_root=tmp_path / "out")
        receipt = SNAP.verify(destination)
        for name in leaves:
            relative = "merlin/schemas/" + name
            assert relative in receipt["files"]
            assert receipt["files"][relative] == SNAP.sha_file(schema_root / name)
            assert (destination / relative).read_bytes() == (schema_root / name).read_bytes()
        (schema_root / leaves[0]).write_text("changed live schema\n")
        assert SNAP.verify(destination) == receipt
    finally:
        _restore_snapshot_permissions(destination)


def test_snapshot_keeps_runtime_package_data_but_omits_redundant_capsule_mirror(tmp_path):
    source = tmp_path / "source"
    data = source / "merlin/python/merlin/_data"
    (data / "schemas").mkdir(parents=True)
    (data / "schemas/quant_formats.registry.yaml").write_text("formats: {}\n")
    mirror = data / "contract/capsules/model/large"
    mirror.mkdir(parents=True)
    (mirror / "capsule.weights.safetensors").write_bytes(b"redundant")
    output = tmp_path / "out"
    output.mkdir()

    destination = tmp_path / "snapshot"
    try:
        SNAP.create(source, destination, output_root=output, source_roots=("merlin/python",))

        assert (destination / "merlin/python/merlin/_data/schemas/"
                "quant_formats.registry.yaml").is_file()
        assert not (destination / "merlin/python/merlin/_data/contract/capsules").exists()
    finally:
        _restore_snapshot_permissions(destination)


def test_snapshot_excludes_unrelated_target_sources(tmp_path):
    source = tmp_path / "source"
    for relative in (
            "merlin/experiments/capsule_bench/targets/gemmini/target.yaml",
            "merlin/experiments/capsule_bench/targets/atlas/target.yaml",
            "merlin/targets/gemmini/backend.py", "merlin/targets/atlas/backend.py",
            "merlin/contract/capsules/atlas/isa/a/capsule.yaml",
            "merlin/contract/capsules/isa/g/capsule.yaml"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n")
    output = tmp_path / "out"
    output.mkdir()
    destination = tmp_path / "snapshot"

    try:
        SNAP.create(source, destination, output_root=output, target_name="gemmini",
                    source_roots=("merlin/experiments/capsule_bench", "merlin/targets",
                                  "merlin/contract"))

        assert (destination / "merlin/experiments/capsule_bench/targets/gemmini").is_dir()
        assert not (destination / "merlin/experiments/capsule_bench/targets/atlas").exists()
        assert (destination / "merlin/targets/gemmini").is_dir()
        assert not (destination / "merlin/targets/atlas").exists()
        assert not (destination / "merlin/contract/capsules/atlas").exists()
        assert (destination / "merlin/contract/capsules/isa").is_dir()
    finally:
        _restore_snapshot_permissions(destination)


@pytest.mark.parametrize("mutation", ["bytes", "writable", "extra", "missing", "link"])
def test_snapshot_refuses_drift(snapshot, mutation):
    _src, dst = snapshot
    p = dst / "code/a.py"
    (dst / "code").chmod(0o700)
    if mutation in ("bytes", "writable"):
        p.chmod(0o600)
        if mutation == "bytes":
            p.write_text("value = 9\n")
            p.chmod(0o444)
    elif mutation == "extra":
        (dst / "code/extra.py").write_text("pass\n")
    elif mutation == "missing":
        p.unlink()
    else:
        p.unlink()
        p.symlink_to(_src / "code/a.py")
    with pytest.raises(SNAP.SnapshotError):
        SNAP.verify(dst)


def test_claims_split_by_declared_family_not_name_prefix_or_analyzer(monkeypatch, tmp_path):
    members = [SimpleNamespace(capsule=f"same_prefix_{i}", family=family,
                               source_sha256=str(i), descriptor={})
               for i, family in enumerate(("gamma", "alpha", "alpha", "beta"))]
    monkeypatch.setattr(SUITE, "load_target_experiment", lambda _: object())
    monkeypatch.setattr(SUITE.STAGE, "discover_performance_corpus",
                        lambda _: SimpleNamespace(capsules=members))
    monkeypatch.setattr(SUITE.GATE, "load_certificate", lambda *_a, **_k: object())
    monkeypatch.setattr(SUITE.PAIRED, "_gsim_workload", lambda m: m.capsule)
    monkeypatch.setattr(SUITE.GATE, "plan_evaluation", lambda *_a, **_k: SimpleNamespace(
        admitted=True, eligible=True, use_gsim=True, selected_engine="gsim"))
    seen = []

    def formal(group):
        seen.append({m.family for m in group})
        return {"status": "READY", "analyzer": "shared.analyzer"}

    monkeypatch.setattr(SUITE.STAGE, "prepare_formal_claim", formal)
    names = [m.capsule for m in members]
    groups = SUITE.claim_groups(tmp_path, tmp_path, "0" * 64, names)
    assert [g["family"] for g in groups] == ["gamma", "alpha", "beta"]
    assert [len(g["members"]) for g in groups] == [1, 2, 1]
    assert all(len(s) == 1 for s in seen)
    with pytest.raises(SUITE.SuiteError, match="undiscovered"):
        SUITE.claim_groups(tmp_path, tmp_path, "0" * 64, ["not_discovered"])
    with pytest.raises(SUITE.SuiteError, match="unique"):
        SUITE.claim_groups(tmp_path, tmp_path, "0" * 64, names * 2)


@pytest.mark.parametrize("script", [SCRIPTS / "_pbcommon.py",
    merlin_dir() / "experiments/capsule_bench/harness/_common.py"])
def test_bootstrap_imports_explicit_snapshot_not_enclosing_git(script, tmp_path):
    root = tmp_path / "snapshot"
    package = root / "merlin/python/merlin"
    for part in (package, package / "common", package / "perf"):
        part.mkdir(parents=True, exist_ok=True)
        (part / "__init__.py").write_text("")
    (package / "benchharness.py").write_text(
        "from pathlib import Path\n"
        "def helper(*a, **k): return Path('.')\n"
        "runs_root=reports_root=sh=hash_tree=repo_sha=helper\n")
    (package / "common/paths.py").write_text("def env(key, default=None): return default\n")
    (package / "perf/workload_gen.py").write_text(
        "from types import SimpleNamespace\n"
        "def tile_geometry(_): return SimpleNamespace(rows=2, cols=2)\n")
    env = {**os.environ, "MERLIN_REPO_ROOT": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("MERLIN_TARGET_EXPERIMENT", None)
    result = subprocess.run([sys.executable, "-c",
        "import runpy,sys,json; d=runpy.run_path(sys.argv[1]); "
        "import merlin.benchharness as b; print(json.dumps([str(d['REPO']),b.__file__]))",
        str(script)], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [str(root), str(package / "benchharness.py")]


def test_launcher_cli_keeps_repeated_waivers_and_explicit_negative_flag():
    assert SUITE.coordinator_args({"hardware_counters": False,
                                  "waive_functional_gate": ["a", "b"]}) == [
        "--no-hardware-counters", "--waive-functional-gate", "a", "--waive-functional-gate", "b"]


def test_chia_python_input_preserves_virtualenv_entrypoint_symlink(tmp_path):
    base = tmp_path / "base-python"
    base.write_text("binary\n")
    venv = tmp_path / "venv/bin"
    venv.mkdir(parents=True)
    entrypoint = venv / "python"
    entrypoint.symlink_to(base)

    assert SUITE._input_path(entrypoint, preserve_entrypoint=True) == entrypoint.absolute()
    assert SUITE._input_path(entrypoint) == base.resolve()


def test_suite_environment_pins_warm_measured_only_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_CACHE_STATE", "cold")
    env = SUITE.environment(tmp_path, {
        "output_root": str(tmp_path / "out"), "trial_workers": 1, "sim_workers": 2})
    assert env["MERLIN_CACHE_STATE"] == "warm"
    assert env["MERLIN_REQUIRED_RTL_ENGINE"] == "gsim"
    assert env["MERLIN_CLANG_INSTALL"] == str(tmp_path / "source/third_party/llvm-install")
    assert env["MERLIN_CLANG"] == str(
        tmp_path / "source/third_party/llvm-install/bin/clang-23")


def test_suite_seals_the_evaluator_contract_runner_and_harness_before_authoring():
    boundary = SUITE.evaluation_boundary(SUITE.repo_root())
    assert boundary["frozen_before_authoring"] is True
    assert set(boundary["files"]) == {
        "backend_contract", "command_buffer_schema", "gemmini_harness", "paired_runner"}
    assert all(len(record["sha256"]) == 64 and record["path"]
               for record in boundary["files"].values())


@pytest.mark.parametrize("failure", [None, "missing_evidence", "wrong_output", "baseline_mutation"])
def test_runtime_preflight_executes_every_member_and_refuses_bad_evidence(
        tmp_path, monkeypatch, failure):
    config = SimpleNamespace(descriptor=tmp_path / "descriptor", perf_families="all",
        perf_capsules="one,two", functional_run_id="seed", functional_submission_sha256="a" * 64,
        waive_functional_gate=(), gsim_certificate=tmp_path / "certificate",
        gsim_certificate_sha256="b" * 64, rtl_facts=tmp_path / "facts", measurement_timeout=120)
    members = [SimpleNamespace(family="f", capsule=name, source_sha256="c" * 64,
                               descriptor={"required_oracle_tiers": ["L2", "L3"]})
               for name in ("one", "two")]
    corpus = SimpleNamespace(capsules=members, capsules_sha256="d" * 64)
    functional = SimpleNamespace(digest="a" * 64, run_dir=tmp_path / "functional")
    baseline = tmp_path / "baseline"
    monkeypatch.setattr(SUITE, "load_target_experiment", lambda _: SimpleNamespace(target="test"))
    monkeypatch.setattr(SUITE.STAGE, "inspect_stage_functional_run", lambda *_a, **_k: functional)
    monkeypatch.setattr(SUITE.STAGE, "discover_performance_corpus", lambda *_a, **_k: corpus)
    monkeypatch.setattr(SUITE.STAGE, "freeze_performance_corpus", lambda *_a: corpus)
    monkeypatch.setattr(SUITE.STAGE, "prepare_formal_claim", lambda _members: {
        "declaration": {}, "expected_identities": []})
    monkeypatch.setattr(SUITE.STAGE.PC, "materialize_perf_workspace", lambda *_a: baseline)
    monkeypatch.setattr(SUITE.STAGE, "hash_tree", lambda _: {
        "sha256": "e" * 64 if failure == "baseline_mutation" else functional.digest})
    calls = []

    class Evaluator:
        decisions = {(m.family, m.capsule): object() for m in members}

        def _execute(self, **kwargs):
            calls.append(kwargs)
            return {"private_evidence": kwargs["member"].capsule}

        def _redact_execution(self, raw, decision, **kwargs):
            if failure == "missing_evidence" and kwargs["capsule"] == "two":
                raise SUITE.STAGE.StageGateError("no GSIM execution evidence")
            return {"correct": not (failure == "wrong_output" and kwargs["capsule"] == "two"),
                    "gsim_cycles": 123}

    monkeypatch.setattr(SUITE.STAGE, "prepare_development_feedback", lambda **_k: Evaluator())
    destination = tmp_path / "runtime"
    if failure:
        with pytest.raises((SUITE.SuiteError, SUITE.STAGE.StageGateError)):
            SUITE.runtime_preflight(config, destination)
        _, receipt = SNAP.load_seal(destination, "runtime_preflight")
        assert receipt["status"] == "NO_GO"
        assert receipt["failure"]
    else:
        SUITE.runtime_preflight(config, destination)
        _, receipt = SNAP.load_seal(destination, "runtime_preflight")
        assert receipt["status"] == "GO"
        assert len(receipt["cells"]) == 2
    assert [call["member"].capsule for call in calls] == ["one", "two"]
    assert all(call["arm"] == "baseline" and call["package"] == baseline for call in calls)
    assert all(call["package_sha256"] == functional.digest for call in calls)
    assert len(list(destination.glob("execution.*.json"))) == 2


def test_runtime_preflight_does_not_run_an_unmeasured_functional_ancestor_for_candidate_only_claim(
        tmp_path, monkeypatch):
    config = SimpleNamespace(descriptor=tmp_path / "descriptor", perf_families="all",
        perf_capsules="one,two", functional_run_id="seed", functional_submission_sha256="a" * 64,
        waive_functional_gate=(), gsim_certificate=tmp_path / "certificate",
        gsim_certificate_sha256="b" * 64, rtl_facts=tmp_path / "facts", measurement_timeout=120)
    members = [SimpleNamespace(family="f", capsule=name, source_sha256="c" * 64,
                               descriptor={"required_oracle_tiers": ["L2", "L3"]})
               for name in ("one", "two")]
    corpus = SimpleNamespace(capsules=members, capsules_sha256="d" * 64)
    functional = SimpleNamespace(digest="a" * 64, run_dir=tmp_path / "functional")
    baseline = tmp_path / "baseline"
    monkeypatch.setattr(SUITE, "load_target_experiment", lambda _: SimpleNamespace(target="test"))
    monkeypatch.setattr(SUITE.STAGE, "inspect_stage_functional_run", lambda *_a, **_k: functional)
    monkeypatch.setattr(SUITE.STAGE, "discover_performance_corpus", lambda *_a, **_k: corpus)
    monkeypatch.setattr(SUITE.STAGE, "freeze_performance_corpus", lambda *_a: corpus)
    monkeypatch.setattr(SUITE.STAGE.PC, "materialize_perf_workspace", lambda *_a: baseline)
    monkeypatch.setattr(SUITE.STAGE, "hash_tree", lambda _: {"sha256": functional.digest})
    identities = [{"family": "f", "capsule": member.capsule,
                   "program_arm": "candidate", "simulator": "gsim", "tier": "L3"}
                  for member in members]
    monkeypatch.setattr(SUITE.STAGE, "prepare_formal_claim", lambda _members: {
        "declaration": {"program_arm": "candidate"}, "expected_identities": identities})

    class Decision:
        def to_dict(self):
            return {"admitted": True, "selected_engine": "gsim"}

    class Evaluator:
        decisions = {(member.family, member.capsule): Decision() for member in members}

        def _execute(self, **_kwargs):
            pytest.fail("the functional ancestor is not a member of this candidate-only claim")

    monkeypatch.setattr(SUITE.STAGE, "prepare_development_feedback", lambda **_k: Evaluator())
    destination = tmp_path / "runtime"
    SUITE.runtime_preflight(config, destination)
    _, receipt = SNAP.load_seal(destination, "runtime_preflight")
    assert receipt["status"] == "GO"
    assert receipt["program_arm"] == "candidate"
    assert receipt["live_execution"].startswith("required_on_first_candidate_feedback")
    assert {row["execution_status"] for row in receipt["cells"]} == {
        "gated_until_candidate_exists"}
    assert not list(destination.glob("execution.*.json"))


@pytest.mark.parametrize("refuse_second", [False, True])
def test_all_static_gates_precede_any_live_baseline(tmp_path, monkeypatch, refuse_second):
    root = tmp_path / "suite"
    (root / "source").mkdir(parents=True)
    campaigns = [{"family": name, "members": [name], "config": {
        "descriptor": "d", "gsim_certificate": "c", "gsim_certificate_sha256": "a" * 64,
        "experiment_id": name}} for name in ("one", "two")]
    monkeypatch.setattr(SUITE, "load", lambda _: {"campaigns": campaigns})
    monkeypatch.setattr(SUITE, "repo_root", lambda: root / "source")
    monkeypatch.setattr(SUITE, "sys", SimpleNamespace(modules={}))
    monkeypatch.setattr(SUITE, "config_from", lambda c: SimpleNamespace(**c))
    monkeypatch.setattr(SUITE, "claim_groups", lambda _d, _c, _s, members: [
        {"family": members[0], "members": members}])
    monkeypatch.setattr(SUITE, "completion", lambda _: None)
    calls = []

    def static(config, **kwargs):
        calls.append(("static", config.experiment_id))
        return {"status": "NO_GO" if refuse_second and config.experiment_id == "two" else "GO",
                "blockers": []}

    def runtime(config, destination):
        calls.append(("runtime", config.experiment_id))
        return destination / "receipt.json"

    monkeypatch.setattr(SUITE.ORCH, "preflight", static)
    monkeypatch.setattr(SUITE, "runtime_preflight", runtime)
    if refuse_second:
        with pytest.raises(SUITE.SuiteError, match="preflight refused"):
            SUITE.preflight(root)
        assert calls == [("static", "one"), ("static", "two")]
    else:
        SUITE.preflight(root)
        assert calls == [("static", "one"), ("static", "two"),
                         ("runtime", "one"), ("runtime", "two")]
