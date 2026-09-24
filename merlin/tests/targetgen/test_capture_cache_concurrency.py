"""Shared model capture slots publish only after the parent finishes validation.

Synthetic workers only: no framework imports, model downloads or hardware.
"""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen import capture_cache


def _upstream_source_fixture(root: Path) -> Path:
    """A cacheable fake checkout must provide the direct model2MLIR owners."""
    for relative in (
        "m2m/api.py",
        "m2m/ir/import_fx.py",
        "m2m/capture/torchao_pipeline.py",
        "m2m/capture/torchao_schemes.py",
        "m2m/capture/pt2e_integerize.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original source\n")
    return root


def test_cache_owner_import_does_not_load_capture_or_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys\n"
                "from merlin.targetgen import capture_cache\n"
                "assert not {'torch', 'xdsl', 'merlin.targetgen.capsule_source', "
                "'merlin.frontends.capture_normalization'} & sys.modules.keys()\n"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_cache_request_preserves_environment_boundaries_and_order(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source as source

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    capture = source.PytorchRefSource(
        m2m_dir=_upstream_source_fixture(tmp_path / "upstream"), python=tmp_path / "python"
    )
    # These two requests collided in the previous delimiter-joined encoding.
    single = capture._cache_slot("model", "f32", "source", None, {"A": "one\x1fB=two"})
    split = capture._cache_slot("model", "f32", "source", None, {"A": "one", "B": "two"})
    reordered = capture._cache_slot("model", "f32", "source", None, {"B": "two", "A": "one"})
    assert single is not None and split is not None
    assert single != split
    assert split == reordered
    assert len(split.name.rsplit("_", 1)[1]) == 64


def test_local_implementation_bytes_change_cache_identity(tmp_path, monkeypatch):
    from merlin.common import paths
    from merlin.targetgen import capsule_source as source

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    owners = {}

    def owner_path(name):
        if name not in owners:
            owners[name] = tmp_path / (name + ".py")
            owners[name].write_text("original source\n")
        return owners[name]

    monkeypatch.setattr(paths, "module_source_path", owner_path)
    capture = source.PytorchRefSource(
        m2m_dir=_upstream_source_fixture(tmp_path / "upstream"), python=tmp_path / "python"
    )
    original = capture._cache_slot("model", "f32", "source", None)
    assert original is not None
    assert "merlin.targetgen._m2m_capture_worker" in owners
    assert "merlin.frontends.capture_normalization" in owners
    for path in owners.values():
        path.write_text("changed source\n")
        assert capture._cache_slot("model", "f32", "source", None) != original
        path.write_text("original source\n")
        assert capture._cache_slot("model", "f32", "source", None) == original
    # A missing declared owner cannot reuse an old slot or invent a source hash.
    owners["merlin.targetgen._m2m_capture_worker"].unlink()
    assert capture._cache_slot("model", "f32", "source", None) is None


def test_static_pt2e_cache_binds_upstream_integerizer_bytes(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source as source

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    upstream = _upstream_source_fixture(tmp_path / "upstream")
    capture = source.PytorchRefSource(m2m_dir=upstream, python=tmp_path / "python")
    request = ("model", "int8", "loader", None)
    before = capture._cache_slot(*request, recipe_sha256="recipe", static_pt2e=True)
    assert before is not None
    integerizer = upstream / "m2m/capture/pt2e_integerize.py"
    integerizer.write_text("changed integerization\n")
    assert capture._cache_slot(*request, recipe_sha256="recipe", static_pt2e=True) != before
    integerizer.unlink()
    assert capture._cache_slot(*request, recipe_sha256="recipe", static_pt2e=True) is None


def test_implementation_drift_refuses_publication(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source as source

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    identity = {"revision": "before"}
    monkeypatch.setattr(capture_cache, "implementation_identity", lambda: dict(identity))
    capture = source.PytorchRefSource(m2m_dir=_upstream_source_fixture(tmp_path), python=tmp_path / "python")
    slot = capture._cache_slot("model", "f32", "source", None)

    def changed_capture(*args, **kwargs):
        identity["revision"] = "after"
        return object(), tmp_path / "attempt"

    monkeypatch.setattr(capture, "_run_capture", changed_capture)
    monkeypatch.setattr(capture_cache, "commit", lambda *args: pytest.fail("published under stale identity"))
    with pytest.raises(source.M2MUnavailable, match="implementation changed"):
        capture._run(tmp_path / "loader.py", "model", "f32", workdir=tmp_path / "work", src="source")
    assert capture_cache.committed_attempt(slot) is None


def _capture(slot, entered, written, release, finished, results, failure=False):
    from merlin.targetgen import capsule_source as source

    def worker(argv, **kwargs):
        output = Path(argv[argv.index("--out") + 1])
        output.mkdir(parents=True, exist_ok=True)
        (output / "linalg.mlir").write_text("synthetic module")
        (output / "inputs.json").write_text("[1]")
        (output / "golden.json").write_text("[2]")
        weights = output / "weights.safetensors"
        weights.write_bytes(b"synthetic weights")
        manifest = output / "weights.safetensors.manifest.json"
        manifest.write_text("{}")
        (output / "meta.json").write_text(
            json.dumps(
                {
                    "ok": True,
                    "opaque": 0,
                    "loader_provenance_status": "synthetic",
                    "loader_dependency_sources": [],
                    "capture_abi_version": source._MODEL_CAPTURE_ABI_VERSION,
                    "output_abi": [],
                    "weights_manifest": str(manifest),
                    "weights": str(weights),
                }
            )
        )
        written.set()
        assert release.wait(5), "test did not release synthetic worker"
        if failure:
            raise source.subprocess.TimeoutExpired(argv, 1)
        return SimpleNamespace(returncode=0, stderr="")

    source.subprocess.run = worker
    capture = source.PytorchRefSource(m2m_dir=slot.parent, python=Path("unused"))
    capture._cache_slot = lambda *args: slot
    entered.set()
    try:
        value = capture._run(slot / "loader.py", "model", "f32", workdir=slot, src="synthetic")
        results.put((value.linalg_mlir, value.inputs, value.golden))
    except BaseException as error:
        results.put((type(error).__name__, str(error)))
    finally:
        finished.set()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires POSIX process locking")
def test_reader_cannot_accept_worker_metadata_before_parent_completion(tmp_path):
    context = multiprocessing.get_context("fork")
    release = context.Event()
    entered = [context.Event(), context.Event()]
    written = [context.Event(), context.Event()]
    finished = [context.Event(), context.Event()]
    results = [context.Queue(), context.Queue()]
    processes = [
        context.Process(
            target=_capture,
            args=(tmp_path / "slot", entered[i], written[i], release, finished[i], results[i]),
        )
        for i in range(2)
    ]
    try:
        processes[0].start()
        assert written[0].wait(5)
        processes[1].start()
        assert entered[1].wait(5)
        assert not finished[1].wait(0.3), "reader accepted uncommitted worker metadata"
        assert not written[1].is_set(), "second writer entered the same slot"
        release.set()
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
        assert [queue.get(timeout=1) for queue in results] == [("synthetic module", [1], [2])] * 2
        assert not written[1].is_set(), "reader recaptured a committed cache hit"
    finally:
        release.set()
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires POSIX process locking")
@pytest.mark.parametrize("interruption", ["kill", "timeout", "stale_commit"])
def test_incomplete_or_changed_attempt_is_recaptured(tmp_path, interruption):
    context = multiprocessing.get_context("fork")
    slot = tmp_path / "slot"
    release = context.Event()
    entered, written, finished = context.Event(), context.Event(), context.Event()
    results = context.Queue()
    first = context.Process(
        target=_capture,
        args=(
            slot,
            entered,
            written,
            release,
            finished,
            results,
            interruption == "timeout",
        ),
    )
    second = None
    try:
        first.start()
        assert written.wait(5)
        if interruption == "kill":
            first.terminate()
            # A process killed inside Condition.wait may leave that test Event
            # unusable. It is not a cache lock and must not be reused by peers.
            release = context.Event()
        else:
            release.set()
        first.join(5)
        assert not first.is_alive()
        if interruption == "stale_commit":
            assert first.exitcode == 0
            assert (slot / "capture_complete.json").is_file()
            attempt = slot / json.loads((slot / "capture_complete.json").read_text())["attempt"]
            metadata = json.loads((attempt / "meta.json").read_text())
            metadata["changed"] = True
            (attempt / "meta.json").write_text(json.dumps(metadata))
        else:
            assert not (slot / "capture_complete.json").exists()
        release.set()
        written = context.Event()
        second_results = context.Queue()
        second = context.Process(
            target=_capture,
            args=(
                slot,
                context.Event(),
                written,
                release,
                context.Event(),
                second_results,
            ),
        )
        second.start()
        second.join(5)
        assert second.exitcode == 0
        assert written.is_set(), "uncommitted/changed result was incorrectly reused"
        assert second_results.get(timeout=1) == ("synthetic module", [1], [2])
    finally:
        release.set()
        for process in (first, second):
            if process is not None and process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires POSIX process locking")
def test_different_slots_capture_in_parallel(tmp_path):
    context = multiprocessing.get_context("fork")
    release = context.Event()
    written = [context.Event(), context.Event()]
    processes = [
        context.Process(
            target=_capture,
            args=(
                tmp_path / f"slot-{i}",
                context.Event(),
                written[i],
                release,
                context.Event(),
                context.Queue(),
            ),
        )
        for i in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert all(event.wait(5) for event in written), "different keys were serialized"
        release.set()
        for process in processes:
            process.join(5)
            assert process.exitcode == 0
    finally:
        release.set()
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)


def test_unavailable_lock_bypasses_shared_slot_to_distinct_retained_directories(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source as source

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setitem(sys.modules, "fcntl", None)
    shared = tmp_path / "shared"
    shared.mkdir()
    marker = shared / "do-not-change"
    marker.write_text("prior capture")
    capture = source.PytorchRefSource(m2m_dir=tmp_path, python=Path("unused"))
    monkeypatch.setattr(capture, "_cache_slot", lambda *args: shared)
    directories = []

    def private_capture(*args, slot, workdir, **kwargs):
        assert slot is None
        assert workdir != shared and workdir.is_dir()
        directories.append(workdir)
        return object(), workdir

    monkeypatch.setattr(capture, "_run_capture", private_capture)
    for _ in range(2):
        capture._run(tmp_path / "loader.py", "model", "f32", workdir=shared, src="source")
    assert directories[0] != directories[1]
    assert all(directory.is_dir() for directory in directories)
    assert list(shared.iterdir()) == [marker]
    assert marker.read_text() == "prior capture"


def _orphan_capture(slot, release):
    from merlin.targetgen import capsule_source as source

    real_run = subprocess.run
    source.subprocess.run = lambda argv, **kwargs: real_run(
        [sys.executable, __file__, "--synthetic-worker", argv[argv.index("--out") + 1], str(release)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    capture = source.PytorchRefSource(m2m_dir=slot.parent, python=Path("unused"))
    capture._cache_slot = lambda *args: slot
    capture._run(slot / "loader.py", "model", "f32", workdir=slot, src="synthetic")


def _surviving_worker_case(root, results):
    # Isolated harness process adopts and actually reaps its orphan grandchild;
    # never change the pytest process's child ownership or rely on host PID1.
    from merlin_experiments.execution.native_guardian import _subreaper

    _subreaper()
    context = multiprocessing.get_context("fork")
    slot, release = root / "slot", root / "release"
    parent = context.Process(target=_orphan_capture, args=(slot, release))
    worker_pid = None
    try:
        parent.start()
        deadline = time.monotonic() + 5
        ready = []
        while time.monotonic() < deadline:
            ready = list(slot.glob("**/worker.pid"))
            if ready:
                break
            time.sleep(0.01)
        assert ready, "synthetic subprocess did not start"
        worker_pid = int(ready[0].read_text())
        old_attempt = ready[0].parent
        parent.terminate()
        parent.join(5)
        assert not parent.is_alive()
        os.kill(worker_pid, 0)  # The real child survives its original parent.
        resumed = context.Event()
        resumed.set()
        values = context.Queue()
        _capture(slot, context.Event(), context.Event(), resumed, context.Event(), values)
        assert values.get(timeout=1) == ("synthetic module", [1], [2])
        publication = json.loads((slot / "capture_complete.json").read_text())
        current_attempt = slot / publication["attempt"]
        assert current_attempt != old_attempt
        release.touch()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if os.waitpid(worker_pid, os.WNOHANG)[0] == worker_pid:
                worker_pid = None
                break
            time.sleep(0.01)
        assert worker_pid is None, "synthetic orphan not reaped"
        assert (old_attempt / "linalg.mlir").read_text() == "late orphan"
        assert (current_attempt / "linalg.mlir").read_text() == "synthetic module"
        assert json.loads((slot / "capture_complete.json").read_text()) == publication
        results.put("passed")
    except BaseException as error:
        results.put(f"{type(error).__name__}: {error}")
    finally:
        if parent.is_alive():
            parent.terminate()
        parent.join(5)
        if worker_pid is not None:
            os.kill(worker_pid, signal.SIGKILL)
            os.waitpid(worker_pid, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="isolated orphan harness requires Linux subreaper")
def test_surviving_subprocess_cannot_overwrite_next_parents_publication(tmp_path):
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    harness = context.Process(target=_surviving_worker_case, args=(tmp_path, results))
    harness.start()
    try:
        harness.join(20)
        assert harness.exitcode == 0
        assert results.get(timeout=1) == "passed"
    finally:
        if harness.is_alive():
            harness.terminate()
        harness.join(5)


if __name__ == "__main__" and sys.argv[1] == "--synthetic-worker":
    output, release = Path(sys.argv[2]), Path(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    (output / "linalg.mlir").write_text("unfinished orphan")
    (output / "worker.pid").write_text(str(os.getpid()))
    deadline = time.monotonic() + 10
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    (output / "linalg.mlir").write_text("late orphan")
