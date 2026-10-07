"""One whole-model job end to end on harmless local fixtures, and the refusals that precede any run."""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import gates as G
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import machines as M
from merlin_experiments.phase2.whole_model_measured import registry as R
from merlin_experiments.phase2.whole_model_measured import worker as W

from merlin.perf import whole_model_verdict as V


# --------------------------------------------------------------- the verdict
def _tolerance_log(device: int) -> str:
    return "\n".join(
        [
            "GM_GROUP 1 add 10 sum=UNKNOWN fnv1a=UNKNOWN",
            "GM_BOUND 1 max_abs=200 over=1 bound=1",
            f"GM_WITNESS 1 i=0 lhs=100 rhs=100 device={device} reference=5",
            "FM full model cycles: 10",
            "GM_ARGMAX got=0 want=0 agrees=1",
            "MERLIN_WINDOW end label=m",
        ]
    )


def _tolerance_expectations(width):
    group = {"compare": "bounded_int", "bound_lsb": 1, "operands": {"lhs_load": 1.0, "rhs_load": 1.0, "readout": 1.0}}
    if width is not None:
        group["output_element_bytes"] = width
    return V.Expectations.from_record({"groups": {"1": group}, "argmax": 0})


def test_an_operand_alone_signature_saturates_to_the_stated_width_not_an_assumed_one():
    """A 2-byte group's lone operand (100) is inside its range; a 1-byte range would also hold it, but a
    300 would not -- the range comes from the stated width, never a literal byte range."""
    row = V.judge(_tolerance_log(100), _tolerance_expectations(2))["groups"][0]
    assert row["worst_element"]["lhs_alone"] == 100 and row["worst_element"]["equals"] == "both"
    wide = _tolerance_log(300).replace("lhs=100", "lhs=300").replace("rhs=100", "rhs=1")
    row = V.judge(wide, _tolerance_expectations(2))["groups"][0]
    assert row["worst_element"]["lhs_alone"] == 300 and row["worst_element"]["equals"] == "lhs"
    row = V.judge(wide, _tolerance_expectations(1))["groups"][0]
    assert row["worst_element"]["lhs_alone"] == 127


def test_no_stated_width_means_no_signature_rather_than_a_guessed_one():
    row = V.judge(_tolerance_log(100), _tolerance_expectations(None))["groups"][0]
    assert "worst_element" not in row


# --------------------------------------------------------------- machines and the registry
def test_a_queue_command_that_forwards_the_submitter_identity_is_refused(tmp_path):
    kwargs = dict(
        hw_config="h",
        chipyard=tmp_path,
        workload="w",
        bootbinary="b",
        jobs_root=tmp_path,
        uart_relative="u",
    )
    with pytest.raises(M.MachineRefusal, match="identity"):
        M.FiresimMachine("toy", queue_command=("firesim-queue",), **kwargs)
    with pytest.raises(M.MachineRefusal, match="LOGNAME"):
        M.FiresimMachine("toy", queue_command=("env", "-u", "HOME", "-u", "USER", "firesim-queue"), **kwargs)
    M.FiresimMachine("toy", queue_command=("env", "-u", "HOME", "-u", "USER", "-u", "LOGNAME", "q"), **kwargs)


def _registry(tmp_path: Path, **machines) -> Path:
    import yaml

    path = tmp_path / "machines.yaml"
    path.write_text(yaml.safe_dump({"schema": R.SCHEMA, "target": "toy", "machines": machines}))
    return path


def test_the_registry_expands_a_pairs_halves_and_resolves_host_locations(tmp_path):
    path = _registry(
        tmp_path,
        model={"kind": "spike", "command": [{"env": "TOY_TOOLS", "join": "bin/sim"}], "adjudicates": ["correctness"]},
        pair={"kind": "paired", "timing": "model", "local": "model", "timing_verify": "words"},
    )
    with pytest.raises(R.RegistryError, match="TOY_TOOLS"):
        R.resolve(path, "pair", environment={})
    spec = R.resolve(path, "pair", environment={"TOY_TOOLS": "/opt/toy"})
    assert spec["timing"]["command"] == ["/opt/toy/bin/sim"] and spec["local"]["kind"] == "spike"
    assert spec["target"] == "toy" and spec["registry_name"] == "pair"
    assert R.adjudicates(spec)["status"] == "UNADJUDICATED"
    assert (
        R.adjudicates(R.resolve(path, "model", environment={"TOY_TOOLS": "/x"}), "correctness")["status"]
        == "ADJUDICATED"
    )
    with pytest.raises(R.RegistryError, match="hw_config"):
        R.resolve(path, "model", environment={"TOY_TOOLS": "/x"}, overrides={"hw_config": "other"})


def test_a_machine_limit_must_state_its_reason(tmp_path):
    path = _registry(
        tmp_path,
        board={"kind": "spike", "command": ["x"], "cannot_express": [{"output_element_bytes": 4, "compare": "exact"}]},
    )
    with pytest.raises(R.RegistryError, match="reason"):
        R.load(path)


# --------------------------------------------------------------- the instruction rule
def _gate_job(*, sealed=True, **options):
    roles = options.get("prohibited_roles") or ()
    return {
        "package_sha256": "d" * 64,
        "target": "toy",
        "build_options": options,
        "role": J.ROLE_CANDIDATE,
        "instruction_policy": FX.sealed_policy(roles) if roles and sealed else None,
    }


def test_the_instruction_rule_fails_closed_when_the_scan_cannot_run():
    def broken(build, *, target, roles):
        raise RuntimeError("no disassembler")

    refused = G.isa_gate(_gate_job(prohibited_roles=["loop_descriptor"]), {"elf": "/x"}, None, "timing", checker=broken)
    assert refused["timing_status"] == V.TIMING_REFUSED and "could not be checked" in refused["refusal"]
    assert refused["isa_prohibited"]["scope"] == "whole_elf"


def test_the_instruction_rule_refuses_a_hit_anywhere_in_the_program_and_keeps_a_clean_census():
    seen = {}

    def scan(build, *, target, roles):
        seen["roles"] = roles
        return {"clean": False, "summary": {"LOOP_WS in program": 2}}

    refused = G.isa_gate(_gate_job(prohibited_roles=["loop_descriptor"]), {"elf": "/x"}, None, "timing", checker=scan)
    assert seen["roles"] == ["loop_descriptor"] and refused["refusal"].startswith("isa_prohibited: LOOP_WS in program")
    build: dict = {"elf": "/x"}
    census = {"per_group": {"1": {"total": 3}}}
    prohibited = {"8": "LOOP_0"}
    assert (
        G.isa_gate(
            _gate_job(prohibited_roles=["loop_descriptor"]),
            build,
            None,
            "t",
            checker=lambda b, **k: {"clean": True, "census": census, "prohibited": prohibited, "status": "measured"},
        )
        is None
    )
    assert build["isa_census"] == census
    # A clean build keeps what it was held to: the champion export requires that set, non-empty.
    assert build["isa_prohibition"]["verdict"] == "clean" and build["isa_prohibition"]["prohibited"] == prohibited


def test_the_instruction_rule_refuses_a_clean_verdict_that_checked_nothing():
    """`clean` over an empty prohibited set, or one that misses a sealed instruction, is no verdict."""
    for prohibited in ({}, {"9": "OTHER"}):
        refused = G.isa_gate(
            _gate_job(prohibited_roles=["loop_descriptor"]),
            {"elf": "/x"},
            None,
            "t",
            checker=lambda b, p=prohibited, **k: {"clean": True, "summary": {}, "prohibited": p},
        )
        assert refused["timing_status"] == V.TIMING_REFUSED and "could not be checked" in refused["refusal"]


def test_the_instruction_rule_refuses_a_clean_verdict_the_scan_did_not_mark_measured():
    """A clean scan is recorded with the status the scanner reported; one it did not report is no verdict."""
    for status in (None, "unmeasured"):
        report = {"clean": True, "summary": {}, "prohibited": {"8": "LOOP_0"}}
        if status is not None:
            report["status"] = status
        build: dict = {"elf": "/x"}
        refused = G.isa_gate(
            _gate_job(prohibited_roles=["loop_descriptor"]),
            build,
            None,
            "t",
            checker=lambda b, r=report, **k: dict(r),
        )
        assert refused["timing_status"] == V.TIMING_REFUSED and "not 'measured'" in refused["refusal"]
        assert "isa_prohibition" not in build


def test_the_instruction_rule_refuses_a_job_with_no_enforceable_sealed_policy():
    def scan(build, **kw):
        raise AssertionError("scanned without a sealed policy")

    refused = G.isa_gate(_gate_job(prohibited_roles=["loop_descriptor"], sealed=False), {}, None, "t", checker=scan)
    assert refused["timing_status"] == V.TIMING_REFUSED and "sealed instruction policy" in refused["refusal"]
    vacuous = _gate_job(prohibited_roles=["loop_descriptor"])
    vacuous["instruction_policy"]["prohibited_instructions"] = {"loop_descriptor": []}
    refused = G.isa_gate(vacuous, {}, None, "t", checker=scan)
    assert "prohibits no instruction" in refused["refusal"]


def test_the_reference_arm_and_an_undeclared_rule_are_not_scanned():
    def scan(build, **kw):
        raise AssertionError("scanned")

    reference = {**_gate_job(prohibited_roles=["r"]), "role": J.ROLE_REFERENCE}
    assert G.isa_gate(reference, {}, None, "t", checker=scan) is None
    assert G.isa_gate(_gate_job(), {}, None, "t", checker=scan) is None


def test_the_circuit_breaker_counts_only_an_unbroken_infra_streak_at_the_end():
    infra = {"refusal": "build: ImportError: cannot import name X"}
    ok = {"refusal": None}
    assert G.infra_circuit_breaker([infra, infra, ok], limit=3) is None
    assert G.infra_circuit_breaker([infra, ok, infra, infra], limit=3) is None
    assert "3 consecutive" in G.infra_circuit_breaker([ok, infra, infra, infra], limit=3)
    assert G.infra_circuit_breaker([{"refusal": "isa_prohibited: LOOP in g3"}] * 5, limit=3) is None


def test_a_subset_capsule_report_passes_only_when_every_graded_row_passed():
    subset = {
        "scope": "subset",
        "all_pass": False,
        "n_capsules": 2,
        "n_passed": 2,
        "per_capsule": [{"pass": True}, {"pass": True}],
    }
    assert G.capsule_report_passed(subset, 0)
    assert not G.capsule_report_passed({**subset, "n_passed": 1, "per_capsule": [{"pass": True}, {"pass": False}]}, 0)
    assert not G.capsule_report_passed({"error": "unknown capsule X"}, 0)


# --------------------------------------------------------------- one job, end to end
def test_a_correct_candidate_is_measured_end_to_end_on_a_local_machine(tmp_path):
    builder = FX.write_builder(tmp_path)
    job_dir = FX.job_dir_for(
        tmp_path / "store", FX.package(tmp_path, "p"), machine=FX.spike_machine(tmp_path), builder=builder
    )
    result = W.work(job_dir)
    assert result["timing_status"] == V.TIMING_MEASURED, result.get("refusal")
    assert result["objective_cycles"] == 300 and result["verdict"]["correctness"]["status"] == "pass"
    assert result["cycle_adjudication"]["status"] == "UNKNOWN"
    assert (job_dir / "build_record.json").is_file()


def test_a_wrong_group_has_its_cycles_recorded_and_no_objective(tmp_path):
    groups = {"1": {"cycles": 50, "sum": 1, "fnv": 1, "want_sum": 7, "want_fnv": 9}}
    builder = FX.write_builder(tmp_path)
    job_dir = FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, "p", groups=groups),
        machine=FX.spike_machine(tmp_path),
        builder=builder,
    )
    result = W.work(job_dir)
    assert result["timing_status"] == V.TIMING_MEASURED_INVALID and result["objective_cycles"] is None
    assert result["verdict"]["whole_window_cycles"] == 50


def test_a_prohibited_program_is_refused_before_the_machine_runs(tmp_path, monkeypatch):
    builder = FX.write_builder(tmp_path)
    job_dir = FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, "p"),
        machine=FX.spike_machine(tmp_path),
        builder=builder,
        build_options={"prohibited_roles": ["loop_descriptor"]},
        instruction_policy=FX.sealed_policy(),
    )
    ran = []
    monkeypatch.setattr(M.SpikeMachine, "run", lambda self, *a, **k: ran.append(1))
    monkeypatch.setattr(
        G, "_default_checker", lambda: lambda build, *, target, roles: {"clean": False, "summary": {"X in g1": 1}}
    )
    result = W.work(job_dir)
    assert result["timing_status"] == V.TIMING_REFUSED and "isa_prohibited" in result["refusal"] and not ran


def test_a_snapshot_that_does_not_hash_to_its_request_is_refused(tmp_path):
    builder = FX.write_builder(tmp_path)
    job_dir = FX.job_dir_for(
        tmp_path / "store", FX.package(tmp_path, "p"), machine=FX.spike_machine(tmp_path), builder=builder
    )
    (job_dir / "package" / "manifest.yaml").write_text("name: edited\n")
    result = W.work(job_dir)
    assert result["refusal"].startswith("snapshot:")


def test_a_builder_crash_is_a_refusal_with_its_traceback(tmp_path):
    path = tmp_path / "boom.py"
    path.write_text("def build(*a, **k):\n    raise KeyError('group 7')\n")
    from merlin_experiments.phase2.whole_model_measured.identity import sha256_file

    job_dir = FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, "p"),
        machine=FX.spike_machine(tmp_path),
        builder=(f"{path}:build", sha256_file(path)),
    )
    result = W.work(job_dir)
    assert result["refusal"].startswith("run: KeyError") and "Traceback" in result["traceback"]
    assert G.is_infra_refusal(result["refusal"])
    assert json.loads((job_dir / "job.json").read_text())["state"] == J.PENDING


def _identity(abi: str) -> M.DeviceIdentity:
    return M.DeviceIdentity(
        machine="gsim",
        target="toy",
        binary="/emu",
        binary_sha256="e" * 64,
        artifact="emu",
        config="c",
        built_from=(),
        abi_header_sha256=abi,
        rung="elaborated_rtl",
    )


def test_a_program_built_for_another_header_is_refused_before_it_runs():
    machine = M.GsimMachine("toy", max_cycles=10)
    machine.admit(_identity("a" * 64), program_header_sha256="a" * 64)
    with pytest.raises(M.MachineRefusal, match="header"):
        machine.admit(_identity("a" * 64), program_header_sha256="b" * 64)
    with pytest.raises(M.MachineRefusal, match="no ABI header"):
        machine.admit(_identity(M.UNKNOWN), program_header_sha256="a" * 64)


def test_an_emulator_assertion_ends_the_run_and_is_its_incomplete_reason(tmp_path, monkeypatch):
    """A design that asserts once per cycle is stopped at its FIRST assertion (not run to its cycle
    budget, writing one line per cycle to disk), and the run says why it is incomplete."""
    emulator = tmp_path / "emu"
    emulator.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "while True:\n"
        "    sys.stderr.write('Assertion failed: pipeline stall\\n    at Station.scala:568 assert(x)\\n')\n"
        "    sys.stderr.flush()\n"
        "    time.sleep(0.001)\n"
    )
    emulator.chmod(0o755)
    machine = M.GsimMachine("toy", max_cycles=10)
    identity = _identity("a" * 64)
    monkeypatch.setattr(machine, "identity", lambda: dataclasses.replace(identity, binary=str(emulator)))
    run = machine.run(tmp_path / "program.elf", tmp_path / "run", timeout_s=60)
    assert run["completed"] is False and run["timed_out"] is False and run["wall_seconds"] < 30
    assert run["incomplete_reason"] == "hardware_assertion: pipeline stall at Station.scala:568 assert(x)"
    assert run["diagnostics_capture"]["stopped_on_assertion"] is True
    assert (tmp_path / "run" / "sim.pid").read_text().isdigit()


# --------------------------------------------------------------- the coverage gate, on the paired path
_GATE = {
    "floor_package_sha256": "s" * 64,
    "floor_share": 1.0,
    "floor_groups": ["1", "2"],
    "price": {"1": 100, "2": 200},
}


def _paired_job(tmp_path: Path, *, declined: str | None) -> Path:
    groups = {
        "1": {
            "cycles": 100,
            "sum": 7,
            "fnv": 9,
            "want_sum": 7,
            "want_fnv": 9,
            "on": "host" if declined == "1" else "package",
        },
        "2": {"cycles": 200, "sum": 5, "fnv": 6, "want_sum": 5, "want_fnv": 6},
    }
    machine = {
        "kind": "paired",
        "target": "toy",
        "timing": FX.spike_machine(tmp_path),
        "local": FX.spike_machine(tmp_path),
        "timing_verify": "words",
    }
    return FX.job_dir_for(
        tmp_path / "store",
        FX.package(tmp_path, f"p_{declined}", groups=groups),
        machine=machine,
        builder=FX.write_builder(tmp_path),
        coverage_gate=_GATE,
    )


def test_a_candidate_declining_a_group_the_seed_answered_is_refused_before_any_run_on_the_paired_path(tmp_path):
    """The production screen's machine KIND, end to end through work(): a group routed to the host's
    fallback (its real ``on: host`` label) is not package work, and a candidate that declines a group
    the floor answered is refused right after its build -- no functional run, no board time."""
    result = W.work(_paired_job(tmp_path, declined="1"))
    assert result["timing_status"] == V.TIMING_REFUSED and "no board time" in result["refusal"]
    assert result["coverage_regression"]["declined_groups"] == ["1"]
    assert result["coverage_regression"]["priced_share"] == round(200 / 300, 4)
    assert "device" not in result


def test_a_candidate_answering_every_floor_group_passes_the_gate_on_the_paired_path(tmp_path, monkeypatch):
    """The pass is shown by getting PAST the gate (the next step runs), never by the absence of a
    regression in a result that was refused for some other reason -- which is all the toy target's
    missing backend ever let this assert."""
    seen = {}

    def next_step(job, raws):
        seen["local"] = raws["local"]
        raise RuntimeError("past the coverage gate")

    monkeypatch.setattr(W, "same_program", next_step)
    job_dir = _paired_job(tmp_path, declined=None)
    result = W.work(job_dir)
    assert "past the coverage gate" in str(result.get("refusal")) and "coverage_regression" not in result
    # The PASS is on the record too, with the shares it compared: a silent pass reads like no gate at all.
    job = json.loads((job_dir / "job.json").read_text())
    build = {"groups": [{"group": "1", "on": "package"}, {"group": "2", "on": "package"}]}
    assert G.coverage_assessment(job, build) == {
        "passed": True,
        "priced_share": 1.0,
        "floor_share": 1.0,
        "declined_groups": [],
        "floor_package_sha256": job["coverage_gate"].get("floor_package_sha256"),
    }
    declined = G.coverage_assessment(job, {"groups": [{"group": "2", "on": "package"}]})
    assert declined["passed"] is False and declined["declined_groups"] == ["1"]


def test_a_screened_subset_passes_and_its_record_states_certification_apart(tmp_path):
    """A subset screen on a functional model passes every capsule it grades while certifying none: its
    report's ``all_pass`` is the certification flag and is false.  Copied as ``all_pass`` beside
    ``passed: true`` it read as a contradiction; it is carried as what it means."""
    report = {
        "scope": "subset",
        "n_passed": 2,
        "n_capsules": 2,
        "n_certified": 0,
        "n_screened_only": 2,
        "all_pass": False,
        "per_capsule": [{"capsule": "A", "pass": True}, {"capsule": "B", "pass": True}],
    }
    writer = f"import json,sys; json.dump(json.loads({json.dumps(json.dumps(report))}), open(sys.argv[1], 'w'))"
    spec = {"argv": [sys.executable, "-c", writer, "{out}"], "capsules": "A,B", "label": "screen", "required": True}
    record = G.run_capsule_check(spec, tmp_path, tmp_path / "report.json", cwd=tmp_path)
    assert record["passed"] is True and "all_pass" not in record["summary"]
    summary = record["summary"]
    assert summary["certified_all"] is False and summary["n_screened_only"] == 2 and summary["n_certified"] == 0
    assert "certified_all" in record["passed_basis"]
