"""Real registered schedule branches, one encoder, external completion custody."""

import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from author_skill_fixtures import load_script


def recipe():
    return load_script("heavy-pass-recipes", "parallel_schedule_recipe")


def test_registered_heavy_actions_change_emission_and_alias_is_identical():
    R = recipe()
    counted = load_script("heavy-pass-recipes", "counted_emission_recipe")
    binding = R.neutral_binding((7, 12, 19), counted._support().capacities(None))
    original = binding["source"].kernel.digest()
    rows = list(R.candidates(binding))
    assert all(row.evaluation.status == "compiled" for row in rows)
    assert len({row.emission.data for row in rows}) == 3
    assert rows[-1].duplicate_of == rows[0].evaluation.variant
    assert binding["source"].kernel.digest() == original
    assert all(row.evaluation.result.receipt["source"]["source_sha256"] == original for row in rows)
    assert rows[1].evaluation.result.artifact.realization["dynamic_epoch_drains"] < (
        rows[0].evaluation.result.artifact.realization["dynamic_epoch_drains"])
    assert len({evidence["same_effect_encoder"] for _, evidence in binding["emitted"].values()}) == 1
    for row in rows:
        assert b"llvm.func @effect_compute" in row.emission.data
        assert b"llvm.func @original_full_completion" in row.emission.data


def test_neutral_external_callback_records_unknown_without_count_ranking(tmp_path):
    R = recipe()
    counted = load_script("heavy-pass-recipes", "counted_emission_recipe")
    with ThreadPoolExecutor(max_workers=2) as executor:
        report = R.run([(7, 12, 19), (8, 12, 20)], counted._support().capacities(None),
            executor=executor, max_in_flight=2, directory=tmp_path/"run")
    assert report["schedules_constructed"] == 2
    assert len(report["candidates"]) == 8 and len(report["completions"]) == 6
    assert [r["shape"] for r in report["candidates"][:4]] == [(7, 12, 19), (8, 12, 20)]*2
    assert all(r["status"] == "unranked" and r["cycles"] is None for r in report["completions"])
    assert all(r["ranking"] is None for r in report["rankings"])
    assert report["missing_target_binding"] is True
    persisted = [json.loads(row) for row in (tmp_path/"run/completions.jsonl").read_text().splitlines()]
    assert persisted == json.loads(json.dumps(report["completions"]))
    assert all("source_check" in json.loads(p.read_text()) for p in (tmp_path/"run").glob("*-*.json"))


def test_external_owner_refills_while_maintenance_is_pending(tmp_path):
    from merlin.transforms.evaluated_exploration import ExplorationIdentity, exploration_context_identity

    R = recipe()
    counted = load_script("heavy-pass-recipes", "counted_emission_recipe")
    release_slow, maintenance_started, release_maintenance = (threading.Event() for _ in range(3))
    calls, maintenance_jobs, constructed = [], [], []
    lock = threading.Lock()

    def binding(shape, caps):
        result = R.neutral_binding(shape, caps)
        constructed.append(shape)
        return result

    def identity(shape, binding):
        return ExplorationIdentity(*(R.digest([name, shape]) for name in
            ("source", "abi", "inputs", "reference", "facts", "library", "provider", "scope", "roi")),
            exploration_context_identity(binding["context"]), "llvm-mlir", "modeled_roi")

    def evaluate(program, receipt, domain):
        with lock:
            calls.append(domain.digest())
            index = len(calls)
        if index == 1:
            assert release_slow.wait(10), "first shape blocked admission of the second shape"
        elif index == 2:
            release_slow.set()
        elif index == 3:
            assert maintenance_started.wait(10)
            assert not release_maintenance.is_set()  # slow cleanup did not hold admission
            assert (tmp_path/"run/completions.jsonl").read_text()
        return None  # actual owner has no completed cycles: retain UNKNOWN, no invented timing

    def maintenance():
        maintenance_started.set()
        assert release_maintenance.wait(10)

    with ThreadPoolExecutor(max_workers=1) as maintenance_executor, ThreadPoolExecutor(max_workers=2) as executor:
        owner = SimpleNamespace(binding=binding, identity=identity, evaluate=evaluate,
            on_completion=lambda request, completion: maintenance_jobs.append(
                maintenance_executor.submit(maintenance)))
        try:
            report = R.run([(7, 12, 19), (8, 12, 20)], counted._support().capacities(None),
                executor=executor, max_in_flight=2, directory=tmp_path/"run", owner=owner)
        finally:
            release_maintenance.set()
        for future in maintenance_jobs:
            future.result()
    assert constructed == [(7, 12, 19), (8, 12, 20)]
    assert calls[0] != calls[1] and len(calls) == 6  # each unique variant x geometry, alias skipped
    assert len(report["completions"]) == 6
    assert all(row["status"] == "pending" for row in report["completions"])


def test_cli_rejects_nonindependent_shapes_before_evaluation(tmp_path):
    with pytest.raises(SystemExit):
        recipe().main(["--shape", "7,12,19", "--shape", "7,12,19", "--output", str(tmp_path/"bad")])
    assert not (tmp_path/"bad").exists()


def test_missing_adapter_is_an_explicit_construction_task(tmp_path):
    with ThreadPoolExecutor(max_workers=1) as executor, pytest.raises(ValueError, match="construct adapter hooks"):
        recipe().run([(7, 12, 19)], {}, executor=executor, max_in_flight=1,
                     directory=tmp_path/"bad", owner=SimpleNamespace())
    assert not (tmp_path/"bad").exists()


def test_cli_help_needs_only_standard_library():
    result = subprocess.run([sys.executable, "-I", recipe().__file__, "--help"],
                            capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0 and "--adapter" in result.stdout
