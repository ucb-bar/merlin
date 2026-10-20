"""Concurrent broker completions must not overwrite another source epoch's ledger."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from merlin_experiments.phase1.feedback import promotion as P


def test_completed_results_serialize_whole_read_modify_write(tmp_path, monkeypatch):
    P._save_tier_state(tmp_path, {
        name: {"L3": {"status": "pending", "digest": epoch}}
        for name, epoch in (("first", "old"), ("second", "current"))
    })
    original = P._tier_state
    first_read, release, second_read = Event(), Event(), Event()
    calls = []

    def observed_read(ws):
        calls.append(1)
        if len(calls) == 1:
            state = original(ws)
            first_read.set()
            assert release.wait(3)
            return state
        second_read.set()
        return original(ws)

    monkeypatch.setattr(P, "_tier_state", observed_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(P.record_cert, tmp_path, {"per_capsule": [{"capsule": "first", "pass": True}]},
                            "L3", identity="submission:old", source_identity_verified=True)
        assert first_read.wait(3)
        second = pool.submit(P.record_cert, tmp_path, {"per_capsule": [{"capsule": "second", "pass": True}]},
                             "L3", identity="submission:current", source_identity_verified=True)
        try:
            assert not second_read.wait(0.05)
        finally:
            release.set()
        assert first.result(timeout=3) == ["first=pass"]
        assert second.result(timeout=3) == ["second=pass"]
    state = original(tmp_path)
    assert state["first"]["L3"]["digest"] == "old"
    assert state["second"]["L3"]["digest"] == "current"
    assert state["first"]["L3"]["status"] == state["second"]["L3"]["status"] == "pass"
    assert P.record_cert(tmp_path, {"per_capsule": [{"capsule": "first", "pass": True}]},
                         "L3", identity="submission:old") == []
    assert P.record_cert(tmp_path, {"per_capsule": [{"capsule": "unknown", "pass": True}]},
                         "L3", identity="submission:unknown") == []
    assert "unknown" not in original(tmp_path)


def test_queue_prefers_current_source_without_cancelling_unknown_or_old(tmp_path, monkeypatch):
    import json

    from merlin_experiments.phase1.brokers import simjob as B
    paths = []
    for name, row in (("a_old", {"promoted": True, "submission_digest": "old"}),
                      ("b_unknown", {"promoted": True}),
                      ("c_current", {"promoted": True, "submission_digest": "current"}),
                      ("d_direct", {})):
        path = tmp_path / f"simreq_{name}.json"
        path.write_text(json.dumps(row))
        paths.append(path)
    monkeypatch.setattr(B._TP, "_submission_digest", lambda ws: "current")
    assert B._current_source_first(paths, tmp_path) == [paths[2], paths[3], paths[0], paths[1]]
    assert all(p.exists() for p in paths)
    digests = iter(("current", "changed"))
    monkeypatch.setattr(B._TP, "_submission_digest", lambda ws: next(digests))
    assert B._current_source_first(paths, tmp_path) == paths
