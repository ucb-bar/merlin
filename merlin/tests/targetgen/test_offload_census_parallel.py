"""Diagnostic fan-out retains original execution custody and deterministic accounting."""

from __future__ import annotations

import contextlib
import contextvars
import threading
from types import SimpleNamespace

import pytest

from merlin.targetgen import offload_census as OC
from merlin.targetgen import package_runtime as P


@pytest.fixture
def harness(monkeypatch, tmp_path):
    leased = contextvars.ContextVar("census_test_cpu_lease", default=False)
    built = []
    dirs = []
    pkg = SimpleNamespace(directory=tmp_path, manifest={}, tool=tmp_path / "tool")

    @contextlib.contextmanager
    def lease():
        token = leased.set(True)
        try:
            yield
        finally:
            leased.reset(token)

    def build(selected):
        assert selected is pkg and not leased.get()
        built.append(selected)

    def paths(root, name, **kwargs):
        assert leased.get()
        generated = root / name / "generated"
        generated.mkdir(parents=True)
        dirs.append(generated)
        return SimpleNamespace(generated=generated)

    def audit(name, buffer, artifact, outcome):
        assert leased.get()
        return {"program": name, "verdict": "declared_host" if outcome == "host_only" else "clean"}

    monkeypatch.delenv("MERLIN_GRADE_WORKERS", raising=False)
    monkeypatch.setattr(OC.cpu_pool, "grading_cpus", lambda: ())
    monkeypatch.setattr(OC.cpu_pool, "shared_compute_enabled", lambda: False)
    monkeypatch.setattr(OC.cpu_pool, "shared_compute_pool", lease)
    monkeypatch.setattr(OC.cpu_pool, "selected_pool", lambda cpus: contextlib.nullcontext())
    monkeypatch.setattr(P, "load_package", lambda *args, **kwargs: pkg)
    monkeypatch.setattr(P, "integrity_scan", lambda selected: None)
    monkeypatch.setattr(P, "build_package", build)
    monkeypatch.setattr(OC, "make_run_paths", paths)
    monkeypatch.setattr(OC, "host_compute_row", audit)
    monkeypatch.setattr(OC, "instruction_use", lambda artifacts, **kwargs: {"artifacts": list(artifacts)})
    return SimpleNamespace(pkg=pkg, built=built, dirs=dirs, leased=leased)


def capsules(*names):
    return [{"name": name, "interface_mlir": "input.mlir", "index": index} for index, name in enumerate(names)]


def host_buffer():
    return {"commands": [], "params": {"lane_placement": [{"lane": "host", "family": "elementwise_map"}]}}


def test_parallel_cases_copy_original_owner_build_once_and_keep_input_order(monkeypatch, harness):
    owner = SimpleNamespace(run_entrypoint=lambda *args, **kwargs: None, build_package=lambda *args, **kwargs: None)
    original = contextvars.ContextVar("census_original_common_service")
    original.set(object())
    expected = original.get()
    barrier = threading.Barrier(2)
    fast_done = threading.Event()
    completion = []

    def emit(pkg, package, capsule, paths, **kwargs):
        assert pkg is harness.pkg and P._scoped_executor() is owner
        assert original.get() is expected and not harness.leased.get()
        assert kwargs["timeout"] == 23
        barrier.wait(timeout=3)
        if capsule["name"] == "slow":
            assert fast_done.wait(timeout=3)
        completion.append(capsule["name"])
        if capsule["name"] == "fast":
            fast_done.set()
        return pkg, host_buffer(), capsule["name"]

    monkeypatch.setattr(OC, "run_entrypoints", emit)
    with P.scoped_package_executor(owner):
        report = OC.census("package", target="synthetic", capsules=capsules("slow", "fast"), workers=2, timeout=23)
    assert completion == ["fast", "slow"]
    assert len(harness.built) == 1
    assert [row["program"] for row in report["programs"]] == ["slow", "fast"]
    assert report["instruction_use"]["artifacts"] == ["slow", "fast"]
    assert report["all_accounted"] and report["outcomes"] == {"host_only": 2}
    assert report["execution"] == {"workers": 2, "programs_requested": 2}


def test_duplicate_names_have_independent_artifact_directories(monkeypatch, harness):
    def emit(pkg, package, capsule, paths, **kwargs):
        assert not harness.leased.get()
        (paths.generated / "marker").write_text(str(capsule["index"]))
        return pkg, host_buffer(), str(capsule["index"])

    monkeypatch.setattr(OC, "run_entrypoints", emit)
    report = OC.census("package", target="synthetic", capsules=capsules("same", "same"), workers=2)
    assert len(set(harness.dirs)) == 2
    assert report["instruction_use"]["artifacts"] == ["0", "1"]


def test_parallel_refusals_and_silent_programs_do_not_become_passes(monkeypatch, harness):
    def emit(pkg, package, capsule, paths, **kwargs):
        if capsule["name"] == "declined":
            raise P.BackendDeclined("unsupported shape")
        if capsule["name"] == "error":
            raise P.CertFailure("parse", "tool_crash", "bad input")
        return pkg, {"commands": []}, "silent artifact"

    monkeypatch.setattr(OC, "run_entrypoints", emit)
    report = OC.census("package", target="synthetic", capsules=capsules("declined", "error", "silent"), workers=3)
    assert [row["outcome"] for row in report["programs"]] == ["declined", "error", "silent"]
    assert report["programs"][0]["reason"] == "unsupported shape"
    assert report["programs"][1]["reason"] == "parse: bad input"
    assert report["silent"] == ["silent"] and not report["all_accounted"]
    assert report["instruction_use"]["artifacts"] == ["silent artifact"]
    assert [row["program"] for row in report["host_compute"]] == ["silent"]


def test_zero_workers_uses_environment_and_shared_pool_caps_explicit_workers(monkeypatch, harness):
    monkeypatch.setenv("MERLIN_GRADE_WORKERS", "2")
    monkeypatch.setattr(OC, "run_entrypoints", lambda pkg, *args, **kwargs: (pkg, host_buffer(), "artifact"))
    cases = capsules("a", "b", "c")
    assert OC.census("package", target="synthetic", capsules=cases)["execution"]["workers"] == 2
    monkeypatch.setattr(OC.cpu_pool, "grading_cpus", lambda: (31,))
    monkeypatch.setattr(OC.cpu_pool, "shared_compute_enabled", lambda: True)
    assert OC.census("package", target="synthetic", capsules=cases, workers=15)["execution"]["workers"] == 1


@pytest.mark.parametrize("workers", [-1, True, 1.5])
def test_invalid_workers_refuse_before_build(workers, harness):
    with pytest.raises(ValueError, match="nonnegative integer"):
        OC.census("package", target="synthetic", capsules=capsules("a"), workers=workers)
    assert harness.built == []


def test_bad_environment_workers_refuse_and_build_failure_remains_error(monkeypatch, harness):
    monkeypatch.setenv("MERLIN_GRADE_WORKERS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        OC.census("package", target="synthetic", capsules=capsules("a"))
    monkeypatch.setenv("MERLIN_GRADE_WORKERS", "2")

    def bad_build(pkg):
        raise P.CertFailure("build", "tool_crash", "cannot build")

    monkeypatch.setattr(P, "build_package", bad_build)
    monkeypatch.setattr(OC, "run_entrypoints", lambda *args, **kwargs: pytest.fail("failed build was invoked"))
    report = OC.census("package", target="synthetic", capsules=capsules("a", "b"))
    assert report["outcomes"] == {"error": 2} and not report["all_accounted"]
    assert all(row["reason"] == "build: cannot build" for row in report["programs"])


def test_capsules_without_interface_are_skipped_without_build(monkeypatch, harness):
    monkeypatch.setattr(OC, "run_entrypoints", lambda *args, **kwargs: pytest.fail("no interface was invoked"))
    report = OC.census("package", target="synthetic", capsules=[{"name": "metadata_only"}])
    assert report["programs"] == [] and harness.built == []


def test_selfcheck_dispatch_forwards_requested_census_workers_and_timeout(monkeypatch, tmp_path, capsys):
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.feedback import selfcheck

    context = InvocationContext(
        tmp_path, tmp_path / "target.yaml", tmp_path, "fixture",
        tmp_path / "runs", tmp_path / "reports", tmp_path / "bundles", (),
    )
    observed = []

    def census_of_corpus(package, **kwargs):
        observed.append((package, kwargs))
        return {"all_accounted": True, "host_compute_vetoed": [], "programs": []}

    monkeypatch.setattr(OC, "census_of_corpus", census_of_corpus)
    result = selfcheck.main(
        ["--submission", str(tmp_path), "--sim", "spike", "--offload-census", "--workers", "15", "--timeout", "29"],
        context=context, capsules_root=tmp_path / "public", contract=tmp_path / "contract",
    )
    assert result == 0 and len(observed) == 1
    assert observed[0][1]["workers"] == 15 and observed[0][1]["timeout"] == 29
    assert '"all_accounted": true' in capsys.readouterr().out
