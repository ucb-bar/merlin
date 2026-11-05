"""Fleet limits hold across both waves, peer changes and pending startups."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module(name, path=None):
    path = path or Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("test_xs_" + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


S = module("study")


@pytest.mark.parametrize("own,external,pending,expected", [
    (0, 0, 0, True), (0, 2, 2, True), (0, 2, 3, False),
    (2, 2, 0, True), (3, 0, 0, False), (3, 2, 0, False),
    (1, 3, 0, True), (2, 3, 0, False), (0, 5, 0, False),
])
def test_global_and_local_caps(own, external, pending, expected):
    ours = {f"own{i}" for i in range(6)}
    active = {f"own{i}" for i in range(own)} | {f"external{i}" for i in range(external)}
    assert S.admissible(active, ours, pending=pending) is expected


def test_cpu_pools_are_three_disjoint_fifteen_core_budgets():
    assert len(set(sum(S._cpu_groups(S.DEFAULT_CPUS), []))) == 45
    assert not set(sum(S._cpu_groups(S.DEFAULT_CPUS), [])) & set(S.RESERVED_CPUS)
    with pytest.raises(ValueError, match="disjoint"):
        S._cpu_groups([range(15)] * 3)
    with pytest.raises(ValueError, match="fifteen"):
        S._cpu_groups([range(14), range(15, 30), range(30, 45)])
    with pytest.raises(ValueError, match="reserved"):
        S._cpu_groups([range(38, 53), range(53, 68), range(68, 83)])


def test_scratch_wave_supplies_no_seed_or_other_arm(tmp_path, monkeypatch):
    import yaml

    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps(dict(schema="merlin.study.transfer_panel.v1", cases=[{}] * 34)))
    base = dict(run={}, inputs={"environment": {}}, plan={"policy": {"base": S.pin(policy)}, "argv": []})
    campaign = SimpleNamespace(load_spec=lambda path: base,
        Context=lambda *args, **kwargs: SimpleNamespace(lookup=lambda key: str(tmp_path), x=lambda value: value),
        git=lambda *args: "a" * 40)
    monkeypatch.setattr(S, "_module", lambda name: campaign if name == "fresh_campaign" else pytest.fail(name))
    record = S.specs(tmp_path / "base.yaml", tmp_path / "study", panel=panel, arms=("el1", "el2", "el3"))
    assert [row["arm"] for row in record["arms"]] == ["el1", "el2", "el3"]
    assert record["seed"] is record["seed_tools"] is record["seed_library_selection"] is None
    for row in record["arms"]:
        spec = yaml.safe_load(Path(row["spec"]["path"]).read_text())
        assert "seed" not in spec
        assert spec["plan"]["reference_guidance"] == "off"
        assert spec["plan"]["author_profile"] == "xs-el-v1/" + row["arm"]
        assert spec["variables"]["INPUTS"].startswith("${FRAMEWORK}/out/artifacts/")


def test_run_never_launches_without_explicit_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(S.subprocess, "Popen", lambda *_a, **_k: pytest.fail("author launched"))
    with pytest.raises(ValueError, match="confirm"):
        S.run(tmp_path / "study.json")


@pytest.mark.parametrize("status,active,available", [
    ("claimed", False, False), ("started", False, False),
    ("finished", True, False), ("finished", False, True),
    ("interrupted", False, True),
])
def test_reused_wave_pool_waits_for_owner(status, active, available):
    first = dict(arm="first", unit="firstservice", cpus=list(range(15)))
    second = dict(arm="second", unit="secondservice", cpus=list(range(15)))
    assert S.pool_available(second, [first, second], {"first": {"status": status}},
                            {"firstservice"} if active else set()) is available


def test_dispatch_persists_claim_before_spawn_and_never_retries(tmp_path, monkeypatch):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return S.pin(path)

    row = dict(arm="first", unit="firstservice", cpus=list(range(15)),
               plan=save("plan.json", {"cwd": str(tmp_path)}), policy=save("policy.json", {}))
    record = dict(schema=S.SCHEMA, state="prepared", operator_task=str(tmp_path),
                  reserved_external_cpus=[],
                  launch_review=save("review.json", {}), panel=save("panel.json", {}),
                  prepared_arms=["first"], arms=[row, dict(arm="later", unit="later", cpus=list(range(15)))])
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(record))
    monkeypatch.setattr(S, "active_units", lambda: set())
    monkeypatch.setattr(S, "_reservation_command", lambda plan: ["never-executed"])

    def fail_spawn(*args, **kwargs):
        dispatch = json.loads((tmp_path / "dispatch.json").read_text())
        assert dispatch["runs"]["first"]["status"] == "claimed"
        assert "later" not in dispatch["runs"]
        raise OSError("simulated spawn failure")

    monkeypatch.setattr(S.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="simulated"):
        S.run(study_path, confirm=True)
    dispatch_path = tmp_path / "dispatch.json"
    dispatch = json.loads(dispatch_path.read_text())
    assert dispatch["runs"]["first"]["status"] == "failed"
    # Simulate interruption after durable intent but before recording a PID.
    dispatch["runs"]["first"]["status"] = "claimed"
    dispatch_path.write_text(json.dumps(dispatch))
    monkeypatch.setattr(S.subprocess, "Popen", lambda *_a, **_k: pytest.fail("interrupted author retried"))
    assert S.run(study_path, confirm=True) == 1
    assert json.loads(dispatch_path.read_text())["runs"]["first"]["status"] == "interrupted"


def test_preserved_host_masks_survive_out_of_order_release(tmp_path, monkeypatch):
    reservation = module("reservation", S.HERE.parents[1] / "launch" / "resource_reservation.py")
    states = {}
    for index in range(2):
        path = tmp_path / f"{index}.json"
        states[path] = dict(host_isolation="preserve", assigned_cpus=list(range(index * 15, index * 15 + 15)),
                            online_cpus=list(range(96)))
    monkeypatch.setattr(reservation, "root_state", lambda path: states[path])
    monkeypatch.setattr(reservation, "set_allowed", lambda *_: pytest.fail("peer mask changed"))
    monkeypatch.setattr(reservation, "allowed", lambda *_: pytest.fail("peer mask read for restoration"))
    monkeypatch.setattr(reservation, "save", lambda p, value: states.update({p: dict(value)}))
    paths = list(states)
    for path in paths:
        reservation.acquire(path)
        assert states[path]["status"] == "reserved"
    for path in reversed(paths):
        reservation.release(path)
        assert states[path]["release_complete"] is True
        assert states[path]["host_masks_modified"] is False
