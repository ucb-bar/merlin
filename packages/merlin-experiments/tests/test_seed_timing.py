"""A seeded run's carried performance: model cycles by executable bytes, the per-case baseline, staging."""

import hashlib
import json

from merlin.targetgen import seeded_grade
from merlin_experiments.phase1 import joint_seed
from merlin_experiments.phase1.feedback import joint_performance, seed_timing

KEY = {"capsule_sha256": "c" * 64, "selection_sha256": "s" * 64, "model_sha256": "m" * 64,
       "facts_sha256": "f" * 64, "timing_epoch": 0}


def _run(tmp_path, monkeypatch, elf_bytes=b"elf"):
    run = tmp_path / "run"
    (run / seed_timing.TIMING_DIR).mkdir(parents=True)
    (run / "joint_campaign.json").write_text("{}")
    monkeypatch.setenv("MERLIN_JOINT_CAMPAIGN_STATE", str(run / "joint_campaign.json"))
    digest = hashlib.sha256(elf_bytes).hexdigest()
    record = {"capsule": "conv", "key": dict(KEY, elf_sha256=digest), "seed_run": "seedrun",
              "performance": {"status": "PROVISIONAL", "estimated_whole_soc_cycles": 100, "reason": "observed"},
              "compiler_diagnostics": {"feedback_brief": {"x": 1}}}
    (run / seed_timing.TIMING_DIR / f"{digest}.json").write_text(json.dumps({"seed_run": "seedrun",
                                                                             "records": [record]}))
    elf = tmp_path / "package_kernel.elf"
    elf.write_bytes(elf_bytes)
    return run, elf


def test_carried_cycles_need_the_same_bytes_and_the_same_model(tmp_path, monkeypatch):
    run, elf = _run(tmp_path, monkeypatch)
    hit = seed_timing.carried(elf, capsule="conv", key=KEY)
    assert hit["performance"]["estimated_whole_soc_cycles"] == 100 and hit["seed_run"] == "seedrun"
    assert hit["performance"]["carried_from"]["run"] == "seedrun"
    assert hit["performance"]["reason"].startswith("carried from seedrun")
    assert hit["compiler_diagnostics"] == {"feedback_brief": {"x": 1}}
    assert seed_timing.carried(elf, capsule="other", key=KEY) is None
    assert seed_timing.carried(elf, capsule="conv", key=dict(KEY, model_sha256="n" * 64)) is None
    elf.write_bytes(b"changed")
    assert seed_timing.carried(elf, capsule="conv", key=KEY) is None
    elf.write_bytes(b"elf")
    monkeypatch.setenv(seed_timing.ENV, "0")
    assert seed_timing.carried(elf, capsule="conv", key=KEY) is None


def test_no_seed_state_carries_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_JOINT_CAMPAIGN_STATE", raising=False)
    elf = tmp_path / "elf"
    elf.write_bytes(b"elf")
    assert seed_timing.carried(elf, capsule="conv", key=KEY) is None
    assert seed_timing.annotate([{"test_case": "conv", "cycles": 5}]) is None


def test_attach_uses_the_seed_cycles_for_an_unchanged_executable(tmp_path, monkeypatch):
    run, elf = _run(tmp_path, monkeypatch)

    class Runtime:
        model_path = tmp_path / "model.json"

    monkeypatch.setattr(joint_performance, "_runtime", lambda runtime, facts_pin: Runtime)
    monkeypatch.setattr(joint_performance.F, "digest", lambda path: "m" * 64)
    monkeypatch.setattr(joint_performance, "original_executable", lambda root: elf)
    monkeypatch.setattr(joint_performance, "capsule_identity", lambda capsule: "c" * 64)
    hit = joint_performance._seed_timing(tmp_path, {"runtime": {"x": 1}, "epoch": 0},
                                         {"facts": {"sha256": "f" * 64}}, {"sha256": "s" * 64},
                                         {"name": "conv"}, {})
    assert hit["performance"]["estimated_whole_soc_cycles"] == 100
    owned = joint_performance._seed_timing(tmp_path, {"runtime": {"x": 1}, "epoch": 0},
                                           {"facts": {"sha256": "f" * 64}}, {"sha256": "s" * 64},
                                           {"name": "conv"}, {"owned_host_native_execution": {}})
    assert owned is None


def test_brief_entries_show_cycles_against_the_seed(tmp_path, monkeypatch):
    run, _ = _run(tmp_path, monkeypatch)
    (run / seed_timing.BASELINE).write_text(json.dumps({"cases": {
        "conv": {"cycles": 200, "evidence": "modeled: Spike timing model", "model_sha256": "m" * 64},
        "dense": {"cycles": 100, "evidence": "modeled: Spike timing model", "model_sha256": "m" * 64}}}))
    entries = [{"test_case": "conv", "cycles": 100, "evidence": "modeled: Spike timing model (self-check)"},
               {"test_case": "dense", "cycles": 90, "evidence": "estimated: fast census estimate"},
               {"test_case": "new", "cycles": 7, "evidence": "modeled: x"}]
    rows = {"conv": {"performance": {"model_sha256": "m" * 64}}}
    summary = seed_timing.annotate(entries, rows)
    assert entries[0]["seed_cycles"] == 200 and entries[0]["vs_seed"] == 0.5
    assert entries[1]["seed_cycles"] == 100 and "vs_seed" not in entries[1]  # estimate vs model: not compared
    assert "seed_cycles" not in entries[2]
    assert summary["compared"] == 1 and summary["faster_than_seed"] == 1 and summary["geomean_vs_seed"] == 0.5
    exact = [{"test_case": "dense", "cycles": 95, "evidence": "modeled: Spike timing model (provisional)"}]
    seed_timing.annotate(exact, {"dense": {"performance": {"carried_from": {"run": "seedrun"}}}})
    assert exact[0]["seed_cycles"] == 95 and exact[0]["vs_seed"] == 1.0
    assert exact[0]["seed_basis"] == "the same executable the seed timed"
    other = [{"test_case": "conv", "cycles": 100, "evidence": "modeled: x"}]
    seed_timing.annotate(other, {"conv": {"performance": {"model_sha256": "n" * 64}}})
    assert "vs_seed" not in other[0]


def test_run_private_evidence_goes_to_the_run_directory_only(tmp_path):
    evidence = tmp_path / "evidence"
    (evidence / joint_seed.RUN_PRIVATE / "seed_timing").mkdir(parents=True)
    (evidence / joint_seed.RUN_PRIVATE / "seed_timing" / "a.json").write_text("{}")
    (evidence / joint_seed.RUN_PRIVATE / "seed_baseline.json").write_text("{}")
    (evidence / "SEED_INDEX.md").write_text("index\n")
    workspace, run = tmp_path / "ws", tmp_path / "run"
    workspace.mkdir()
    run.mkdir()
    (run / "seed_baseline.json").write_text("kept")
    pin = {"path": str(evidence), "sha256": "unused"}
    staged = joint_seed.stage_evidence(pin, workspace)
    assert staged["staged"] == ["SEED_INDEX.md"] and joint_seed.RUN_PRIVATE in staged["skipped"]
    assert not (workspace / joint_seed.RUN_PRIVATE).exists()
    private = joint_seed.stage_run_evidence(pin, run)
    assert private == {"staged": ["seed_timing"], "skipped": ["seed_baseline.json"]}
    assert (run / "seed_timing/a.json").is_file() and (run / "seed_baseline.json").read_text() == "kept"


def test_a_development_grade_pass_carried_in_full_is_labelled():
    carried = {"status": "pass", "carried": {"carried": True, "source": "carried from seedrun"}}
    document = {"status": "incomplete", "failure": {"plane": "oracle_unavailable"},
                "tiers": {"L2": carried, "L3": {"status": "unavailable"}}}
    assert seeded_grade.label(document, ("L2", "L3")) == "carried from seedrun"
    failed = dict(document, tiers={"L2": carried, "L3": {"status": "fail"}})
    assert seeded_grade.label(failed, ("L2", "L3")) is None
    other = dict(document, failure={"plane": "numeric"})
    assert seeded_grade.label(other, ("L2", "L3")) is None
    executed = dict(document, tiers={"L2": {"status": "pass"}, "L3": {"status": "unavailable"}})
    assert seeded_grade.label(executed, ("L2", "L3")) is None
