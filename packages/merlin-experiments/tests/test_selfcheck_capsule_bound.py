"""The ``auto`` self-check capsule bound always admits the public corpus it grades."""

import pytest
from merlin_experiments.phase1.brokers import selfcheck as B


def _corpus(root, count):
    for index in range(count):
        capsule = root / ("isa" if index % 2 else "layers") / f"C{index:03d}"
        capsule.mkdir(parents=True)
        (capsule / "capsule.yaml").write_text(f"name: C{index:03d}\n")
    return root


@pytest.fixture
def pool(tmp_path, monkeypatch):
    # The v2 run's shape: 18 pool CPUs (auto = 144 from the pool alone) and a 146-capsule suite.
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(str(cpu) for cpu in range(6, 24)))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    return _corpus(tmp_path / "capsules", 146)


def test_auto_bound_covers_the_public_corpus_and_admits_the_full_suite(pool, monkeypatch):
    monkeypatch.setenv("MERLIN_SELFCHECK_MAX_CAPSULES", "auto")
    size = B._public_corpus_size(pool)
    assert size == 146
    assert B._ordinary_capsule_limit() == 144  # the pool alone, as before
    assert B._ordinary_capsule_limit(size) == 146
    assert B._feedback_scope_error({"capsules": "all"}, corpus_size=size) is None
    names = [f"C{index:03d}" for index in range(147)]
    assert B._feedback_scope_error({"capsules": ",".join(names[:146])}, corpus_size=size) is None
    assert "at most 146 named capsules" in B._feedback_scope_error({"capsules": ",".join(names)}, corpus_size=size)


def test_auto_bound_keeps_its_pool_size_above_a_small_corpus(pool, tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_SELFCHECK_MAX_CAPSULES", "auto")
    small = B._public_corpus_size(_corpus(tmp_path / "small", 12))
    assert B._ordinary_capsule_limit(small) == 144
    assert B._feedback_scope_error({"capsules": "all"}, corpus_size=small) is None


def test_uncounted_corpus_and_explicit_bounds_keep_their_refusals(pool, monkeypatch):
    monkeypatch.setenv("MERLIN_SELFCHECK_MAX_CAPSULES", "auto")
    assert B._public_corpus_size(None) is None
    assert B._feedback_scope_error({"capsules": "all"}) is not None
    monkeypatch.setenv("MERLIN_SELFCHECK_MAX_CAPSULES", "100")
    assert B._ordinary_capsule_limit(146) == 100
    assert "at most 100 named capsules" in B._feedback_scope_error({"capsules": "all"}, corpus_size=146)
    names = ",".join(f"C{index}" for index in range(101))
    assert B._feedback_scope_error({"capsules": names}, corpus_size=146) is not None


@pytest.mark.parametrize(("bound", "scheduled"), [("auto", True), ("100", False)])
def test_broker_counts_its_own_corpus_before_refusing_a_full_suite(pool, tmp_path, monkeypatch, bound, scheduled):
    import json
    import time
    from types import SimpleNamespace

    from merlin_experiments.phase1 import frozen_facts

    class Launched(Exception):
        pass

    def launch(*args, **kwargs):
        raise Launched("worker selected")

    monkeypatch.setenv("MERLIN_SELFCHECK_MAX_CAPSULES", bound)
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "1")
    monkeypatch.setattr(frozen_facts, "select", lambda *args: None)
    monkeypatch.setattr(B, "resolve_context", lambda *args: SimpleNamespace(repo=tmp_path))
    monkeypatch.setattr(B, "worker_command", launch)
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    channel = ws / ".qa_channel"
    channel.mkdir()
    deadline = time.time_ns() + 30_000_000_000
    body = {"capsules": "all", "workers": 1, "sim": "spike", "deadline_unix_ns": deadline}
    (channel / "req_full.json").write_text(json.dumps(body))
    try:
        B.main(["--ws", str(ws), "--capsules-root", str(pool), "--one-request", "req_full.json"])
        launched = False
    except Launched:
        launched = True
    assert launched is scheduled
    if not scheduled:
        assert (channel / "done_full").read_text() == "not_scheduled"
        assert "at most 100 named capsules" in json.loads((channel / "resp_full.json").read_text())["error"]
