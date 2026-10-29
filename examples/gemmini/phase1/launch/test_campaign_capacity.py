"""Accounting distinguishes active native service identity from author campaigns."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name):
    path = Path(__file__).with_name(name + ".py")
    if name == "study" and not path.exists():
        path = Path(__file__).parents[1] / "studies/kernel_abstractions/study.py"
    spec = importlib.util.spec_from_file_location("test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load("campaign_capacity")


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "CGROUP", tmp_path / "cgroups")
    monkeypatch.setattr(C, "PROC", tmp_path / "proc")
    units = {}

    def add(unit, command, processes=(), *, state=False, cpus="0-14"):
        group = C.CGROUP / "merlinphase1.slice" / (unit + ".service")
        group.mkdir(parents=True)
        if cpus is not None:
            (group / "cpuset.cpus.effective").write_text(cpus)
        (group / "cgroup.procs").write_text("".join(str(pid) + "\n" for pid, _ in processes))
        for pid, argv in processes:
            root = C.PROC / str(pid)
            root.mkdir(parents=True)
            (root / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
        if state:
            path = tmp_path / (unit + ".json")
            path.write_text(json.dumps(dict(schema="merlin.operator.cpu_reservation.v1", unit=unit, argv=command)))
            command = ["/usr/bin/python3", "/framework/launch/resource_reservation.py", "_child", "--state", str(path)]
        units[unit] = "ControlGroup=/merlinphase1.slice/" + unit + ".service\nActiveState=active\nExecStart={ argv[]=" + " ".join(command) + " ; ignore_errors=no ; }\n"
        return group

    def run(argv, **kwargs):
        if argv[1] == "list-units":
            assert "--plain" in argv
            return SimpleNamespace(stdout="\n".join(unit + ".service loaded active running description" for unit in units))
        return SimpleNamespace(stdout=units[argv[2].removesuffix(".service")])

    monkeypatch.setattr(C.subprocess, "run", run)
    return add, units


def native(run):
    return ["/venv/bin/python3.12", "-B", "-m", "merlin_experiments.phase1", "--run-id", run]


def family(config):
    return ["python", "/framework/studies/kernel_abstractions/run.py", "launch", "--config", config]


def test_realistic_two_authors_three_operator_services_leave_three_seats(host):
    add, _ = host
    add("merlinphase1fresh10cca", native("fresh10cca"), [(11, native("fresh10cca"))], state=True)
    add("merlinphase1fresh10seed", native("fresh10seed"), [(12, native("fresh10seed"))], state=True)
    add("merlinphase1familieseval", ["python", "/task/coordinate-evaluation.py"], state=True)
    add("merlinphase1familiesevalw2", ["python", "/task/evaluate-worker.py", "plan.json"])
    add("merlinphase1physical", ["python", "/task/run-physical-remaining-v1.py"])
    active = C.active_campaigns()
    assert len(active) == 5
    assert sum(active.author_counts.values()) == 2
    assert active.intersection({"merlinphase1physical"})
    S = load("study")
    assert S.admissible(active, {"our1", "our2", "our3"}, pending=2)
    assert not S.admissible(active, {"our1", "our2", "our3"}, pending=3)


def test_multiple_native_and_family_controllers_count_individually(host):
    add, _ = host
    add("merlinphase1family", family("r0.json"), [
        (21, family("r0.json")), (22, family("r1.json")), (23, native("other")),
        (24, ["codex", "exec", "-C", "/helper"]),
    ], state=True)
    active = C.active_campaigns()
    assert active.author_counts["merlinphase1family"] == 3
    assert not load("study").admissible(active, set(), pending=2)


def test_reservation_intent_counts_before_author_controller_appears(host):
    add, _ = host
    add("merlinphase1starting", native("starting"), state=True)
    assert C.active_campaigns().author_counts == {"merlinphase1starting": 1}


def test_live_author_overrides_operator_role(host):
    add, _ = host
    add("merlinphase1eval", ["python", "/task/coordinate-evaluation.py"], [(31, native("child"))])
    assert C.active_campaigns().author_counts == {"merlinphase1eval": 1}


def test_orphan_codex_retains_seat_and_helpers_are_not_replica_authors(host):
    add, _ = host
    add("merlinphase1orphan", ["python", "/unknown.py"], [
        (41, ["codex", "exec", "-C", "/author"]), (42, ["codex", "exec", "-C", "/helper"]),
    ])
    assert C.active_campaigns().author_counts == {"merlinphase1orphan": 1}


@pytest.mark.parametrize("failure", ["unknown", "missing_cgroup", "unreadable_proc", "broken_state"])
def test_unknown_or_racy_unit_blocks_conservatively(host, failure):
    add, units = host
    group = add("merlinphase1unknown", ["python", "/unknown.py"])
    if failure == "missing_cgroup":
        (group / "cgroup.procs").unlink()
        (group / "cpuset.cpus.effective").unlink()
        group.rmdir()
    elif failure == "unreadable_proc":
        (group / "cgroup.procs").write_text("51\n")
        (C.PROC / "51").mkdir(parents=True)
    elif failure == "broken_state":
        units["merlinphase1unknown"] = units["merlinphase1unknown"].replace("python /unknown.py", "python /framework/resource_reservation.py _child --state /absent-state.json")
    active = C.active_campaigns()
    assert "merlinphase1unknown" in active
    assert active.author_counts["merlinphase1unknown"] == C.UNKNOWN_CAMPAIGNS
    assert not load("study").admissible(active, set())


def test_native_preflight_does_not_take_author_seat(host):
    add, _ = host
    add("merlinphase1preflight", [*native("preflight"), "--preflight-only"], state=True)
    assert C.active_campaigns().author_counts == {"merlinphase1preflight": 0}


def test_reading_fails_closed_before_any_start(host, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("systemd unavailable")
    monkeypatch.setattr(C.subprocess, "run", fail)
    with pytest.raises(OSError, match="systemd unavailable"):
        C.active_campaigns()



@pytest.mark.parametrize("argv", [
    ["python", "/unknown.py", "/task/evaluate-worker.py"],
    ["python", "/unknown.py", "-m", "merlin_experiments.phase1", "--preflight-only"],
    ["python", "/unknown.py", "/framework/kernel_abstractions/run.py", "evaluate"],
    ["/bin/echo", "python", "/task/coordinate-evaluation.py"],
    ["python", "-c", "pass", "/task/run-physical-remaining-v1.py"],
    ["/usr/bin/env", "ROLE=/task/evaluate-worker.py", "python", "/unknown.py"],
])
def test_argument_names_are_not_executed_operator_or_preflight_ownership(host, argv):
    add, _ = host
    add("merlinphase1unknown", argv)
    active = C.active_campaigns()
    assert active.author_counts == {"merlinphase1unknown": C.UNKNOWN_CAMPAIGNS}


@pytest.mark.parametrize("prefix", [
    ["python"], ["/venv/bin/python3.12", "-B"],
    ["/usr/bin/env", "-i", "ROLE=operator", "/venv/bin/python3.12", "-B"],
    ["/usr/bin/env", "--ignore-environment", "--", "python3", "-W", "default"],
])
def test_true_python_operator_entrypoint_survives_interpreter_and_env_flags(host, prefix):
    add, _ = host
    add("merlinphase1eval", [*prefix, "/task/evaluate-worker.py", "/task/plan.json"])
    assert C.active_campaigns().author_counts == {"merlinphase1eval": 0}


def test_env_wrapped_native_startup_and_preflight_have_distinct_namespace(host):
    add, _ = host
    prefix = ["/usr/bin/env", "-i", "ROLE=author"]
    add("merlinphase1starting", [*prefix, *native("starting")], state=True)
    add("merlinphase1preflight", [*prefix, *native("preflight"), "--preflight-only"], state=True)
    assert C.active_campaigns().author_counts == {"merlinphase1starting": 1, "merlinphase1preflight": 0}


def test_reservation_name_in_argument_data_cannot_select_another_command(host, tmp_path):
    add, _ = host
    state = tmp_path / "operator.json"
    state.write_text(json.dumps(dict(schema="merlin.operator.cpu_reservation.v1",
                                    unit="merlinphase1unknown", argv=["python", "/task/evaluate-worker.py"])))
    add("merlinphase1unknown", ["python", "/unknown.py", "/framework/resource_reservation.py",
                               "_child", "--state", str(state)])
    assert C.active_campaigns().author_counts == {"merlinphase1unknown": C.UNKNOWN_CAMPAIGNS}


@pytest.mark.parametrize("prefix", [[], ["/usr/bin/env", "-i", "ROLE=operator"]])
def test_exact_installed_physical_wrapper_consumes_no_author_seat(host, prefix):
    add, _ = host
    add("merlinphase1physical", [*prefix, "/home/ubuntu/bin/fpga-run-elf", "/task/panel.elf", "frozen-study"])
    active = C.active_campaigns()
    assert active.author_counts == {"merlinphase1physical": 0}
    assert "merlinphase1physical" in active


@pytest.mark.parametrize("argv", [
    ["python", "/unknown.py", "/home/ubuntu/bin/fpga-run-elf", "/task/panel.elf"],
    ["/bin/echo", "/home/ubuntu/bin/fpga-run-elf", "/task/panel.elf"],
    ["/unreviewed/fpga-run-elf", "/task/panel.elf"],
    ["fpga-run-elf", "/task/panel.elf"],
    ["/bin/bash", "-c", "/home/ubuntu/bin/fpga-run-elf /task/panel.elf"],
])
def test_physical_wrapper_name_in_data_or_another_path_does_not_grant_operator_role(host, argv):
    add, _ = host
    add("merlinphase1unknown", argv)
    assert C.active_campaigns().author_counts == {"merlinphase1unknown": C.UNKNOWN_CAMPAIGNS}


def test_author_descendants_override_physical_wrapper_role(host):
    add, _ = host
    add("merlinphase1physical", ["/home/ubuntu/bin/fpga-run-elf", "/task/panel.elf"],
        [(91, native("unexpected-author-one")), (92, native("unexpected-author-two"))])
    assert C.active_campaigns().author_counts == {"merlinphase1physical": 2}


@pytest.mark.parametrize("cpus,width", [("0-1", 2), ("0-3", 4), ("4-7,48-55", 12), ("0-13", 14)])
def test_small_enforced_cpu_groups_keep_identity_and_raw_author_evidence_without_full_size_seat(host, cpus, width):
    add, _ = host
    add("merlinphase1small", native("small"), [(101, native("small"))], cpus=cpus)
    active = C.active_campaigns()
    row = active.observations["merlinphase1small"]
    assert "merlinphase1small" in active
    assert active.author_counts == {"merlinphase1small": 0}
    assert active.raw_author_counts == {"merlinphase1small": 1}
    assert row["enforced_cpu_width"] == width and row["effective_cpus"] == cpus
    assert row["raw_classification"] == "author_campaign"
    assert row["classification"] == "small_cpu_scope_excluded"


@pytest.mark.parametrize("cpus", ["0-14", "0-15", "0-7,32-39"])
def test_fifteen_or_more_enforced_cpus_retain_every_author_seat(host, cpus):
    add, _ = host
    add("merlinphase1full", family("r0"), [(102, family("r1"))], cpus=cpus)
    active = C.active_campaigns()
    assert active.author_counts == {"merlinphase1full": 2}
    assert active.raw_author_counts == active.author_counts
    assert active.observations["merlinphase1full"]["campaign_scope"] == "full_size_cpu_scope"


@pytest.mark.parametrize("cpus", [None, "", "0--14", "5-4", "0-foo", "-1", "0-1000000"])
@pytest.mark.parametrize("command", [native("author"), ["python", "/task/evaluate-worker.py"]])
def test_missing_or_malformed_enforced_width_stays_conservative_even_for_known_ownership(host, cpus, command):
    add, _ = host
    add("merlinphase1unknownwidth", command, cpus=cpus)
    active = C.active_campaigns()
    assert active.author_counts == {"merlinphase1unknownwidth": C.UNKNOWN_CAMPAIGNS}
    assert active.observations["merlinphase1unknownwidth"]["enforced_cpu_width"] is None
    assert not load("study").admissible(active, set())


@pytest.mark.parametrize("cpus", ["0", "4-7,48-55"])
def test_proven_small_width_excludes_unknown_ownership_but_retains_conservative_raw_record(host, cpus):
    add, _ = host
    add("merlinphase1unknownsmall", ["python", "/unknown.py"], cpus=cpus)
    active = C.active_campaigns()
    row = active.observations["merlinphase1unknownsmall"]
    assert active.author_counts == {"merlinphase1unknownsmall": 0}
    assert active.raw_author_counts == {"merlinphase1unknownsmall": C.UNKNOWN_CAMPAIGNS}
    assert row["raw_classification"] == "unknown_conservative" and "reason" in row


def test_enforced_cpuset_is_authority_over_declared_cpu_metadata(host):
    add, units = host
    add("merlinphase1small", native("small"), cpus="0-1")
    units["merlinphase1small"] += "AllowedCPUs=0-31\n"
    assert C.active_campaigns().author_counts == {"merlinphase1small": 0}


def test_two_full_size_authors_and_six_small_studies_leave_exactly_three_full_size_seats(host):
    add, _ = host
    for index in range(2):
        add("merlinphase1large" + str(index), native("large" + str(index)), cpus="0-14")
    for index in range(6):
        add("merlinphase1small" + str(index), family("small" + str(index)), cpus="0-3")
    add("merlinphase1parent", ["python", "/unknown-parent.py"], cpus="4-7,48-55")
    add("merlinphase1monitor", ["python", "/unknown-monitor.py"], cpus="0")
    active = C.active_campaigns()
    assert len(active) == 10 and sum(active.author_counts.values()) == 2
    assert sum(row["raw_author_campaigns"] for row in active.observations.values()
               if row["raw_classification"] == "author_campaign") == 8
    study = load("study")
    assert all(study.admissible(active, {"our1", "our2", "our3"}, pending=n) for n in (0, 1, 2))
    assert not study.admissible(active, {"our1", "our2", "our3"}, pending=3)
