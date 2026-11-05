"""A deployed standalone client submits parallel ordinary checks and ranks only matching evidence."""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest

from merlin_experiments.phase1.feedback import result_details
from merlin_experiments.phase1.tools import selfcheck as source


def workspace(tmp_path):
    ws = tmp_path / "ws"
    for label in ("submission", "variants/a", "variants/b"):
        root = ws / label
        root.mkdir(parents=True)
        (root / "backend.py").write_text(f"candidate = {label!r}\n")
    (ws / "submission" / "tuning.json").write_text('{"block": 1}')
    (ws / "submission" / "TUNING_KNOBS.json").write_text(json.dumps({
        "schema": "merlin.tuning_knobs.v1", "settings_file": "tuning.json",
        "knobs": {"block": {"values": [1, 2, 4]}}}))
    path = ws / "agent_selfcheck.py"
    path.write_bytes(Path(source.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("deployed_selfcheck", path)
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    return ws, client


def finish(ws, request, *, factor=1.0, passed=True, model="m1", compact=False):
    rid = request["request_id"]
    digest = request["submission_sha256"]
    doc = {"selfcheck_protocol": source.PROTOCOL_VERSION, "selfcheck_request_id": rid,
           "selfcheck_submission": request["submission"], "submission_sha256": digest,
           "graded_at_unix_ns": time.time_ns(), "all_pass": passed,
           "cases": [{"capsule": name, "pass": passed, "tiers": {"L2": "pass" if passed else "fail"},
                      "performance": {"estimated_whole_soc_cycles": (100 if name == "X" else 200) * factor,
                                      "selection_sha256": "s1", "model_sha256": model,
                                      "timing_mode": "untuned", "timing_epoch": 0,
                                      "scope_sha256": name, "comparison_domain_sha256": name,
                                      "elf_sha256": digest + name},
                      "large_diagnostic": "x" * 4000} for name in request["capsules"].split(",")]}
    if request["submission"].startswith("variants/"):
        doc["variant"] = {"name": request["submission"].partition("/")[2], "sha256": digest, "official": False}
    elif request.get("tuning_overrides"):
        doc["variant"] = {"kind": "knob_overrides", "sha256": "copy-" + digest, "official": False,
                          "tuning_overrides": request["tuning_overrides"], "settings_file": "tuning.json",
                          "applied": True}
    if request.get("tuning_experiment"):
        doc["tuning_experiment"] = request["tuning_experiment"]
    if compact:
        doc = result_details.compact_document(doc, workspace=ws, namespace="selfcheck")
    channel = ws / ".qa_channel"
    (channel / f"resp_{rid}.json").write_text(json.dumps(doc))
    (channel / f"done_{rid}").write_text("ok")


def test_sweep_submits_all_candidates_before_waiting_and_reads_compact_details(tmp_path, capsys):
    ws, client = workspace(tmp_path)
    problems, seen = [], []

    def broker():
        try:
            until = time.monotonic() + 5
            channel = ws / ".qa_channel"
            paths = []
            while time.monotonic() < until:
                paths = list(channel.glob("req_*.json"))
                if len(paths) == 5:
                    break
                time.sleep(0.01)
            assert len(paths) == 5, "baseline, two variants and two knob alternatives must all be queued"
            for path in paths:
                request = json.loads(path.read_text())
                seen.append(request)
                assert (channel / f"detached_{request['request_id']}").exists()
                key = (request.get("tuning_experiment") or {}).get("candidate", {}).get("key")
                finish(ws, request, factor={"variant:a": 0.8, "variant:b": 0.1,
                    "knob:block=2": 0.75, "knob:block=4": 1.2}.get(key, 1),
                    passed=key != "variant:b", compact=True)
        except BaseException as exc:
            problems.append(exc)

    thread = threading.Thread(target=broker)
    thread.start()
    code = client.main(["--sweep", "variants/a,variants/b", "--knob", "block=2,4",
                        "--capsules", "X,Y", "--wait-budget", "10"])
    thread.join(timeout=10)
    assert not thread.is_alive() and not problems
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and report["status"] == "complete" and report["all_pass"] is False
    assert report["ranked_candidates"][0]["candidate"] == "knob:block=2"
    assert report["ranked_candidates"][0]["geomean_ratio"] == 0.75
    assert report["ranked_candidates"][-1]["numerics_changed_or_unknown"] == 2
    assert all(r["suite"] == "scalability" and r["sim"] == "spike" for r in seen)
    assert (ws / "submission" / "tuning.json").read_text() == '{"block": 1}'
    assert (ws / report["all"]).is_file()
    assert all(c["identity"]["baseline_sha256"] and c["identity"]["candidate_sha256"]
               and c["identity"]["model"]["scope_sha256"] for c in report["comparisons"])


def test_sweep_attach_keeps_its_original_requests_and_source_identities(tmp_path, capsys):
    ws, client = workspace(tmp_path)
    assert client.main(["--sweep", "variants/a", "--capsules", "X", "--wait-budget", "1"]) == 3
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "in_progress" and first["all_pass"] is None
    channel = ws / ".qa_channel"
    requests = [json.loads(p.read_text()) for p in channel.glob("req_*.json")]
    (ws / "submission" / "backend.py").write_text("edited = True\n")
    (ws / "variants" / "a" / "backend.py").write_text("also_edited = True\n")
    for request in requests:
        finish(ws, request, factor=0.8 if request["submission"].startswith("variants/") else 1)
    assert client.main(["--attach", first["sweep_id"]]) == 0
    second = json.loads(capsys.readouterr().out)
    assert len(list(channel.glob("req_*.json"))) == 2
    assert second["comparisons"][0]["delta_pct"] == -20
    baseline = next(r for r in requests if r["submission"] == "submission")
    assert second["comparisons"][0]["identity"]["baseline_sha256"] == baseline["submission_sha256"]


def test_sweep_different_model_epochs_are_unranked(tmp_path):
    ws, client = workspace(tmp_path)
    bp = {"capsule": "X", "pass": True, "performance": {"estimated_whole_soc_cycles": 100,
                                                                       "model_sha256": "m1"}}
    cp = {"capsule": "X", "pass": True, "performance": {"estimated_whole_soc_cycles": 50,
                                                                       "model_sha256": "m2"}}
    manifest = {"baseline_request_id": "base", "requests": [{"request_id": "base"},
                  {"request_id": "candidate", "key": "variant:a"}]}
    ranked, compared = client._sweep_rank(ws, manifest, {"base": {"cases": [bp]},
                                                       "candidate": {"cases": [cp]}})
    assert ranked[0]["geomean_ratio"] is None and ranked[0]["cases_compared"] == 0
    assert compared[0]["delta_pct"] is None and "model_sha256" in compared[0]["not_compared"]


def test_sweep_bundle_and_invalid_values_do_not_write_requests(tmp_path, capsys):
    ws, client = workspace(tmp_path)
    (ws / "qa").mkdir()
    (ws / "qa" / "idle_tuning.json").write_text(json.dumps({"bundles": [
        {"representative": "X", "siblings": ["Y"], "label": "one shape class"}]}))
    assert client._sweep_cases(ws, "all", "X") == "X,Y"
    assert client.main(["--sweep", "--knob", "block=3", "--bundle", "X"]) == 2
    assert "not declared" in json.loads(capsys.readouterr().out)["error"]
    assert not list((ws / ".qa_channel").glob("req_*.json"))
    with pytest.raises(ValueError, match="named"):
        client._sweep_cases(ws, "all", "")


def test_sweep_details_refuse_changed_or_external_payloads(tmp_path):
    ws, client = workspace(tmp_path)
    detail = ws / "detail.json"
    detail.write_text('{"capsule":"X"}')
    row = {"capsule": "X", "detail": {"path": "detail.json", "sha256": "wrong"}}
    assert client._sweep_rows(ws, {"cases": [row]}) == {}
    row["detail"]["path"] = "../outside.json"
    assert client._sweep_rows(ws, {"cases": [row]}) == {}
