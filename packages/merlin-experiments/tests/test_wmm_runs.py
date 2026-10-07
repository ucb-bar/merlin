"""A measured run on disk: prepared with its policy stamped in, large inputs frozen by content, resumed
with the SAME method and roles, its OOT history continued, and relaunched by a watchdog that never
drops what the run is."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import config as C
from merlin_experiments.phase2.whole_model_measured import runs as RUNS
from merlin_experiments.phase2.whole_model_measured import watchdog as WD

from merlin.common import oot_repo


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))


def _factory(root: Path):
    counter = iter(range(1, 1000))

    def make(*, target, method):
        run_dir = root / "runs" / target / "phase2" / f"2026093{next(counter)}T000000Z_{method}_abcdef0"
        run_dir.mkdir(parents=True)
        return run_dir

    return make


def _config(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec, pin = FX.write_builder(tmp_path)
    return {
        "schema": C.CONFIG_SCHEMA,
        "builder": {"spec": spec, "sha256": pin},
        "store": str(tmp_path / "store"),
        "screen": {"machine": FX.spike_machine(tmp_path), "build_options": {"model_capsule": "{input:model_capsule}"}},
    }


def _capsule(tmp_path: Path) -> Path:
    capsule = tmp_path / "capsules" / "SY_model"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("name: SY_model\n")
    (capsule / "weights.bin").write_bytes(os.urandom(4096))
    return capsule


def _prepare(tmp_path: Path, **kw) -> RUNS.PreparedRun:
    kw.setdefault("phase0_manifest", FX.write_phase0_manifest(tmp_path))
    return RUNS.prepare(
        target="toy",
        method="whole_model_measured_nofsm",
        objective_config=_config(tmp_path),
        seed=FX.package(tmp_path, "seed"),
        prohibited_roles=["loop_descriptor"],
        inputs={"model_capsule": _capsule(tmp_path)},
        why="first run",
        run_factory=_factory(tmp_path),
        **kw,
    )


def test_a_prepared_run_carries_its_policy_its_frozen_inputs_and_its_seed(tmp_path):
    run = _prepare(tmp_path)
    config = json.loads(run.config_path.read_text())
    options = config["screen"]["build_options"]
    assert options["prohibited_roles"] == ["loop_descriptor"] and config["prohibited_instruction_roles"] == [
        "loop_descriptor"
    ]
    # The sealed Phase 0 policy is stamped in by value, and the run records where it came from.
    sealed = config[C.SEALED_POLICY]
    assert sealed["prohibited_instructions"] == FX.SEALED_POLICY["prohibited_instructions"]
    record = json.loads((run.run_dir / "run.json").read_text())
    assert record["instruction_policy_source"] == sealed["sealed_source"]
    assert options["model_capsule"].startswith(str(run.run_dir / "inputs" / "model_capsule"))
    assert not os.access(run.config_path, os.W_OK)
    seed = json.loads((run.run_dir / "resumed_seed.json").read_text())
    assert seed["schema"] == RUNS.RESUMED_SEED_SCHEMA and seed["method"] == "whole_model_measured_nofsm"
    assert (
        seed["prohibited_instruction_roles"] == ["loop_descriptor"]
        and seed["seed_package_sha256"] == run.seed_package_sha256
    )
    assert (run.run_dir / "workspace" / "program.json").is_file()


def test_preparation_derives_mechanisms_from_each_frozen_model_and_records_the_choice(tmp_path, monkeypatch):
    from merlin.perf import whole_model_open

    selected = []

    def closure(capsule, target):
        selected.append((Path(capsule), target))
        return False

    monkeypatch.setattr(whole_model_open, "is_open_model", closure)
    config = _config(tmp_path)
    config.pop("store")  # generated output paths are not operator-authored scientific inputs
    config["mechanism_policy"] = C.DERIVED_MECHANISMS
    run = RUNS.prepare(
        target="toy",
        method="whole_model_measured",
        objective_config=config,
        seed=FX.package(tmp_path, "seed"),
        prohibited_roles=[],
        inputs={"model_capsule": _capsule(tmp_path)},
        why="derived mechanisms",
        run_factory=_factory(tmp_path),
    )
    prepared = json.loads(run.config_path.read_text())
    assert prepared["store"].endswith("/perf-studies/whole-model/toy")
    assert prepared["screen"]["build_options"]["allow_passes"] is True
    assert prepared["screen"]["build_options"]["allow_regions"] is True
    assert prepared["mechanism_derivation"]["sections"]["screen"]["model_closure"] == "closed"
    assert selected == [(Path(prepared["screen"]["build_options"]["model_capsule"]), "toy")]
    continued = RUNS.resume(run.run_dir, why="continue the same experiment", run_factory=_factory(tmp_path / "next"))
    assert continued.store_roots == run.store_roots
    assert json.loads(continued.config_path.read_text())["mechanism_derivation"] == prepared["mechanism_derivation"]


def test_open_model_derivation_disables_mechanisms_and_refuses_manual_override(tmp_path, monkeypatch):
    from merlin.perf import whole_model_open

    monkeypatch.setattr(whole_model_open, "is_open_model", lambda capsule, target: True)
    capsule = _capsule(tmp_path)
    config = _config(tmp_path)
    config["mechanism_policy"] = C.DERIVED_MECHANISMS
    run = RUNS.prepare(
        target="toy",
        method="open",
        objective_config=config,
        seed=FX.package(tmp_path, "seed"),
        prohibited_roles=[],
        inputs={"model_capsule": capsule},
        why="open model",
        run_factory=_factory(tmp_path),
    )
    prepared = json.loads(run.config_path.read_text())
    assert prepared["screen"]["build_options"]["allow_passes"] is False
    assert prepared["screen"]["build_options"]["allow_regions"] is False
    config["screen"]["build_options"]["allow_regions"] = True
    with pytest.raises(C.ConfigError, match="contradicts"):
        RUNS.prepare(
            target="toy",
            method="open",
            objective_config=config,
            seed=FX.package(tmp_path, "seed2"),
            prohibited_roles=[],
            inputs={"model_capsule": capsule},
            why="no manual override",
            run_factory=_factory(tmp_path / "second"),
        )


def test_derived_mechanisms_refuse_an_unfrozen_capsule(tmp_path, monkeypatch):
    from merlin.perf import whole_model_open

    monkeypatch.setattr(whole_model_open, "is_open_model", lambda capsule, target: pytest.fail("not frozen"))
    config = _config(tmp_path)
    config["mechanism_policy"] = C.DERIVED_MECHANISMS
    config["screen"]["build_options"]["model_capsule"] = str(_capsule(tmp_path))
    with pytest.raises(RUNS.RunError, match="declared frozen input"):
        RUNS.prepare(
            target="toy",
            method="unfrozen",
            objective_config=config,
            seed=FX.package(tmp_path, "seed"),
            prohibited_roles=[],
            why="must freeze",
            run_factory=_factory(tmp_path),
        )


def test_roles_without_an_enforceable_sealed_policy_are_refused_before_anything_is_frozen(tmp_path):
    """A no-FSM run whose Phase 0 policy matched no instruction measured programs nobody checked."""
    vacuous = FX.sealed_policy()
    vacuous["prohibited_instructions"] = {"loop_descriptor": []}
    vacuous["vacuous_roles"] = ["loop_descriptor"]
    with pytest.raises(RUNS.RunError, match="prohibits no instruction"):
        _prepare(tmp_path / "a", phase0_manifest=FX.write_phase0_manifest(tmp_path / "vacuous", vacuous))
    with pytest.raises(RUNS.RunError, match="does not exist"):
        _prepare(tmp_path / "b", phase0_manifest=tmp_path / "missing" / "MANIFEST.yaml")
    assert not (tmp_path / "a" / "runs").exists() and not (tmp_path / "b" / "runs").exists()


def test_large_inputs_are_hard_linked_from_the_content_store_never_copied(tmp_path):
    """The old line copied 1.2-1.8 GB of model capsule per run: 35.6 GB of duplicates."""
    first = _prepare(tmp_path / "a")
    source = tmp_path / "a" / "capsules" / "SY_model" / "weights.bin"
    frozen = first.run_dir / "inputs" / "model_capsule" / "SY_model" / "weights.bin"
    assert frozen.read_bytes() == source.read_bytes() and frozen.stat().st_nlink >= 2
    assert not os.access(frozen, os.W_OK)
    second = RUNS.freeze_inputs(tmp_path / "other_run", {"model_capsule": source.parent})
    again = Path(second["model_capsule"]["frozen"]) / "weights.bin"
    assert os.path.samestat(again.stat(), frozen.stat())  # one inode for every run of the same bytes
    assert second["model_capsule"]["linked_files"] == second["model_capsule"]["files"]
    source.write_bytes(b"edited in place")
    assert frozen.read_bytes() != b"edited in place"  # the freeze holds


def test_a_resume_keeps_the_method_and_roles_and_refuses_to_change_them(tmp_path):
    """A hand-spelled method once dropped `_nofsm` on an auto-relaunch."""
    first = _prepare(tmp_path)
    (first.run_dir / "workspace" / "program.json").write_text(
        json.dumps({"groups": {"1": {"cycles": 5, "sum": 7, "fnv": 9, "want_sum": 7, "want_fnv": 9}}, "argmax": 3})
    )
    nxt = RUNS.resume(first.run_dir, why="launcher exited", run_factory=_factory(tmp_path / "next"))
    assert nxt.method == first.method and nxt.roles == ("loop_descriptor",)
    assert json.loads((nxt.run_dir / "seed" / "submission" / "program.json").read_text())["groups"]["1"]["cycles"] == 5
    assert nxt.store_roots == first.store_roots
    record = json.loads((nxt.run_dir / "resumed_seed.json").read_text())
    assert record["resumed_from_run"] == str(first.run_dir) and record["lineage"]["why"] == "first run"
    with pytest.raises(RUNS.RunError, match="rename"):
        RUNS.resume(first.run_dir, why="x", method="whole_model_measured", run_factory=_factory(tmp_path / "bad"))
    with pytest.raises(RUNS.RunError, match="prohibits"):
        RUNS.resume(first.run_dir, why="x", prohibited_roles=[], run_factory=_factory(tmp_path / "bad2"))


def test_a_resume_that_would_move_the_store_needs_a_reason(tmp_path):
    first = _prepare(tmp_path)
    config = json.loads(first.config_path.read_text())
    config["screen"]["build_options"]["allow_regions"] = True
    with pytest.raises(RUNS.RunError, match="moves the measurement store"):
        RUNS.resume(first.run_dir, why="x", objective_config=config, run_factory=_factory(tmp_path / "n1"))
    moved = RUNS.resume(
        first.run_dir,
        why="x",
        objective_config=config,
        allow_new_store="regions on",
        run_factory=_factory(tmp_path / "n2"),
    )
    record = json.loads((moved.run_dir / "resumed_seed.json").read_text())
    assert (
        record["store_moved_because"] == "regions on"
        and record["store_moved"]["screen"]["to"] != record["store_moved"]["screen"]["from"]
    )


def test_the_oot_history_starts_from_phase_ones_frozen_and_continues_across_a_resume(tmp_path):
    phase1 = tmp_path / "phase1_oot"
    oot_repo.init(phase1)
    frozen = oot_repo.commit_candidate(phase1, FX.package(tmp_path, "p1"), label="round 1", when="20260929T000000Z")
    oot_repo.tag(phase1, oot_repo.FROZEN_TAG, frozen.commit)
    first = _prepare(tmp_path, phase1_oot=phase1, oot=oot_repo)
    repo = first.run_dir / "oot"
    assert oot_repo.tags(repo)[oot_repo.FROZEN_TAG] == frozen.commit
    assert oot_repo.tree_digest(repo, "HEAD") == first.seed_package_sha256
    nxt = RUNS.resume(first.run_dir, why="relaunch", run_factory=_factory(tmp_path / "next"), oot=oot_repo)
    history = oot_repo.history(nxt.run_dir / "oot")
    assert [row.label for row in history][-3:] == ["round 1", "seed", "seed"]


# --------------------------------------------------------------- the watchdog
def _prepared(run_dir: Path, method: str = "m_nofsm") -> RUNS.PreparedRun:
    return RUNS.PreparedRun(run_dir, run_dir / "c.json", "0" * 64, method, ("loop_descriptor",), "1" * 64, {})


def test_the_watchdog_relaunches_an_exited_run_with_its_own_method(tmp_path):
    first = tmp_path / "run1"
    first.mkdir()
    resumed, launched = [], []
    clock = iter(float(t) for t in range(0, 100_000, 1000))

    def resume(run_dir, *, why):
        resumed.append(run_dir)
        nxt = tmp_path / f"run{len(resumed) + 1}"
        nxt.mkdir()
        return _prepared(nxt)

    document = WD.watch(
        first,
        101,
        launch=lambda prepared: launched.append(prepared.method) or 200 + len(launched),
        why="launcher exited",
        policy=WD.WatchPolicy(max_relaunches=2, min_seconds=10),
        resume=resume,
        is_alive=lambda pid: False,
        sleep=lambda s: None,
        clock=lambda: next(clock),
        log=lambda line: None,
    )
    assert launched == ["m_nofsm", "m_nofsm"] and document["stopped"]["kind"] == "budget"


def test_the_watchdog_never_relaunches_a_run_that_stopped_on_evidence_or_crashed_at_once(tmp_path):
    run = tmp_path / "run"
    (run / "stage").mkdir(parents=True)
    (run / "stage" / "sessions.json").write_text(json.dumps({"stopped": {"kind": "plateau", "reason": "6 h"}}))
    common = dict(
        launch=lambda p: pytest.fail("relaunched"),
        why="x",
        is_alive=lambda pid: False,
        sleep=lambda s: None,
        log=lambda line: None,
    )
    assert WD.watch(run, 1, **common)["stopped"]["kind"] == "evidence"
    crashed = tmp_path / "crashed"
    crashed.mkdir()
    assert WD.watch(crashed, 1, clock=lambda: 0.0, **common)["stopped"]["kind"] == "crash_loop"


def test_a_run_whose_launcher_recorded_its_own_stop_is_never_relaunched(tmp_path):
    """Only an exit WITHOUT the launcher's own stop record is a crash to recover from."""
    run = tmp_path / "run"
    (run / "stage").mkdir(parents=True)
    (run / "stage" / "sessions.json").write_text(json.dumps({"stopped": {"kind": "budget", "reason": "spent"}}))
    clock = iter(float(t) for t in range(0, 100_000, 1000))
    document = WD.watch(
        run,
        1,
        launch=lambda p: pytest.fail("relaunched"),
        why="x",
        is_alive=lambda pid: False,
        sleep=lambda s: None,
        clock=lambda: next(clock),
        log=lambda line: None,
    )
    assert document["stopped"]["kind"] == "recorded" and "budget" in document["stopped"]["reason"]


def test_an_imported_seed_is_recorded_as_imported_never_as_a_freeze(tmp_path):
    """A champion measured on another line is a seed with a measurement, not a Phase 1 freeze: the run
    says which, carries the measurement by content, and refuses evidence of other bytes."""
    from merlin_experiments.phase2.whole_model_measured.identity import package_digest

    seed = FX.package(tmp_path, "seed")
    evidence = tmp_path / "result.json"
    evidence.write_text(
        json.dumps(
            {
                "package_sha256": package_digest(seed),
                "timing_status": "MEASURED",
                "objective_cycles": 38_290_856,
                "device": {"artifact": "a_board"},
            }
        )
    )
    run = _prepare(tmp_path / "r", import_evidence=evidence, oot=oot_repo)
    record = json.loads((run.run_dir / "resumed_seed.json").read_text())
    assert record["lineage_kind"] == "imported" and record["frozen"] is False and record["oot_source"] is None
    assert record["imported"]["evidence"]["objective_cycles"] == 38_290_856
    assert [row.label for row in oot_repo.history(run.run_dir / "oot")] == ["imported seed"]
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"package_sha256": "f" * 64}))
    # A RESUME of it is a resume, not a Phase 1 freeze, and the chain still names where it began.
    nxt = RUNS.resume(run.run_dir, why="relaunch", run_factory=_factory(tmp_path / "next"), oot=oot_repo)
    again = json.loads((nxt.run_dir / "resumed_seed.json").read_text())
    assert again["lineage_kind"] == "resumed" and again["origin_kind"] == "imported" and again["frozen"] is False
    third = RUNS.resume(nxt.run_dir, why="relaunch", run_factory=_factory(tmp_path / "third"), oot=oot_repo)
    assert json.loads((third.run_dir / "resumed_seed.json").read_text())["origin_kind"] == "imported"
    assert record["origin_kind"] == "imported"
    misnamed = {"lineage_kind": "phase1_freeze", "lineage": {"lineage_kind": "imported", "lineage": None}}
    assert RUNS._origin_kind(misnamed) == "imported"
    with pytest.raises(RUNS.RunError, match="not this seed's bytes"):
        _prepare(tmp_path / "s", import_evidence=other)
    with pytest.raises(RUNS.RunError, match="never both"):
        _prepare(tmp_path / "t", import_evidence=evidence, phase1_oot=tmp_path / "p1")


def test_the_cli_prepares_a_run_and_its_start_requests_the_seed_first(tmp_path, monkeypatch):
    """`prepare` from the command line, then `start`: the seed is on the screen before any round."""
    from types import SimpleNamespace

    from merlin_experiments.phase2.whole_model_measured import cli
    from merlin_experiments.phase2.whole_model_measured import profiles as P
    from merlin_experiments.phase2.whole_model_measured import service as S
    from merlin_experiments.phase2.whole_model_measured import sessions as SES

    monkeypatch.setattr(S, "spawn", lambda argv, **kw: SimpleNamespace(pid=999_999_999))
    monkeypatch.setattr(S, "alive", lambda pid, owner: pid == 999_999_999)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))  # the run lands in this test's own out root
    config = tmp_path / "objective.json"
    config.write_text(json.dumps(_config(tmp_path)))
    seed = FX.package(tmp_path, "seed")
    cli.main(
        [
            "prepare",
            "--input",
            f"model_capsule={_capsule(tmp_path)}",
            "--target",
            "toy",
            "--method",
            "m_nofsm",
            "--why",
            "e2e",
            "--objective-config",
            str(config),
            "--seed",
            str(seed),
        ]
    )
    (run_dir,) = sorted((tmp_path / "out" / "runs" / "toy" / "phase2").iterdir())
    assert (run_dir / "oot").is_dir() and (run_dir / "run.json").is_file()
    monkeypatch.setattr(P, "check", lambda profile, price_table: {"resolved_model": profile["model"]})
    seen = {}

    def sessions(objective, **kw):
        seen["jobs"] = [j.get("label") for j in objective.screen.jobs()]
        return {"stopped": {"kind": "budget"}}

    monkeypatch.setattr(SES, "run_sessions", sessions)
    from merlin_experiments.phase2.whole_model_measured import identity

    monkeypatch.setattr(identity, "load_builder", lambda spec, **kw: lambda **driver_kw: None)
    cli.start(
        run_dir, profile_name=P.names()[0], round_driver=cli.DEFAULT_ROUND_DRIVER, price_table=tmp_path / "p.yaml"
    )
    assert seen["jobs"] == ["seed"]
    # A run opened by code that keys a DIFFERENT store (its builder's source changed) is refused, never
    # silently started on an empty store.
    record = json.loads((run_dir / "resumed_seed.json").read_text())
    record["store_roots"]["screen"] = str(tmp_path / "another_store")
    (run_dir / "resumed_seed.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="was prepared on the store"):
        cli.start(
            run_dir, profile_name=P.names()[0], round_driver=cli.DEFAULT_ROUND_DRIVER, price_table=tmp_path / "p.yaml"
        )
