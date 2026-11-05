"""Package-owned tool broker contracts, including real isolated-process channel IPC.

These tests also run from an external directory with installed distributions.
Only ISA hardware discovery is substituted; the broker, encoder and public client
execute their actual package implementations. This is not kernel sandbox qualification.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers import cca, isa_tools, simjob
from merlin_experiments.phase1.brokers import selfcheck as selfcheck_broker
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.tools import simjob as simjob_client

from merlin.common.paths import module_source_path, python_import_roots


def _invocation(root, target="synthetic"):
    descriptor = root / f"{target}.yaml"
    descriptor.write_text(f"target: {target}\n")
    return InvocationContext(root, descriptor, root, target, root / "runs", root / "reports", root / "bundles", ())


def test_selfcheck_snapshot_keeps_graded_bytes_while_author_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    workspace = tmp_path / "workspace"
    submission = workspace / "submission"
    submission.mkdir(parents=True)
    (submission / "manifest.yaml").write_text("commands: {}\n")
    (submission / "compiler.py").write_text("VALUE = 1\n")
    digest = selfcheck_broker._submission_digest(submission)
    snapshot = selfcheck_broker._working_snapshot(workspace, digest)
    try:
        (submission / "compiler.py").write_text("VALUE = 2\n")
        assert (snapshot / "submission/compiler.py").read_text() == "VALUE = 1\n"
        assert selfcheck_broker._submission_digest(snapshot / "submission") == digest
        assert selfcheck_broker._submission_digest(submission) != digest
    finally:
        shutil.rmtree(snapshot)


def test_simjob_parallel_capsules_respect_global_verilator_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(simjob, "GLOBAL_VERIL_SLOTS", tmp_path)
    assert simjob._capsule_workers(0, sim="gsim", n_capsules=12) == 5
    assert simjob._capsule_workers(0, sim="verilator", n_capsules=12) == 4
    assert simjob._capsule_workers(0, sim="verilator", n_capsules=1) == 1
    first = simjob._reserve_veril_workers(2, 2)
    assert len(first) == 2
    assert simjob._reserve_veril_workers(2, 2) == ()
    simjob._release_veril_workers(first[:1])
    second = simjob._reserve_veril_workers(2, 2)
    assert len(second) == 1
    simjob._release_veril_workers(first[1:])
    simjob._release_veril_workers(second)
    assert not list(tmp_path.glob("slot_*"))


def test_simjob_repeated_capsule_names_cannot_reserve_extra_workers(tmp_path):
    (tmp_path / "A").mkdir()
    names = simjob._valid_capsules("A,A,A,A", tmp_path)
    assert names == ["A"]
    assert simjob._capsule_workers(4, sim="gsim", n_capsules=len(names)) == 1
    assert simjob._valid_capsules("A,../private,A", tmp_path) is None


def test_simjob_dedupes_only_exact_source_capsule_engine_and_tier(tmp_path):
    key = ("a" * 64, "SY_conv_operand_capacity_boundary", "gsim", "L3")
    active = {"original": {"dedupe_key": key, "aliases": []}}
    assert simjob._dedupe_sim_job(tmp_path, "duplicate", key, active, {})
    assert active["original"]["aliases"] == ["duplicate"]
    assert (tmp_path / "simrun_duplicate").is_file()
    assert not simjob._dedupe_sim_job(tmp_path, "new_source", ("b" * 64, *key[1:]), active, {})
    assert not simjob._dedupe_sim_job(tmp_path, "new_capsule", (key[0], "other", *key[2:]), active, {})
    result = {"all_pass": True, "per_capsule": [{"capsule": key[1], "pass": True}]}
    assert simjob._dedupe_sim_job(tmp_path, "after", key, {}, {key: ("original", result)})
    assert json.loads((tmp_path / "simresp_after.json").read_text()) == result
    assert (tmp_path / "simdone_after").is_file()


def test_simjob_cancellation_stops_process_group_and_releases_slots(tmp_path):
    import subprocess
    import sys
    import time

    ch = tmp_path / "channel"
    ch.mkdir()
    slot = tmp_path / "slot_0"
    slot.write_text("held")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               start_new_session=True, stdout=subprocess.DEVNULL)
    log = (ch / "simlog_job.txt").open("wb")
    job = {"proc": process, "timing": {"job_id": "job"}, "started_monotonic": time.monotonic(),
           "sim": "gsim", "aliases": ["duplicate"], "log": log, "slots": (slot,),
           "snapshot_root": snapshot}
    try:
        simjob._cancel_running_job(ch, "job", job, "submission source superseded")
        assert process.poll() is not None
        assert not slot.exists() and not snapshot.exists() and log.closed
        for jid in ("job", "duplicate"):
            response = json.loads((ch / f"simresp_{jid}.json").read_text())
            assert response["state"] == "cancelled" and response["all_pass"] is False
            assert (ch / f"simerr_{jid}").exists()
        assert json.loads((ch / "simtiming_job.json").read_text())["state"] == "cancelled"
    finally:
        simjob._stop_sim_process(process)


def test_simjob_poll_reports_completed_capsules_while_long_tail_is_pending(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(simjob_client, "CH", tmp_path)
    (tmp_path / "simreq_job.json").write_text("{}")
    (tmp_path / "simrun_job").write_text("running")
    progress = {"status": "running", "n_expected": 2, "n_finished": 1, "n_remaining": 1,
                "per_capsule": [{"capsule": "A", "status": "pass"}]}
    (tmp_path / "progress_job.json").write_text(json.dumps(progress))
    assert simjob_client.main(["poll", "--job-id", "job"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["state"] == "pending" and observed["result"] == progress
    (tmp_path / "simresp_job.json").write_text(json.dumps({"all_pass": True}))
    (tmp_path / "simdone_job").write_text("ok")
    assert simjob_client.main(["poll", "--job-id", "job"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "done"


def test_context_caches_models_per_invocation_but_reloads_schedule(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_runner as runner
    from merlin.targetgen import isa_model, target_experiment

    invocation = _invocation(tmp_path)
    hardware = tmp_path / "hardware"
    hardware.mkdir()
    schedule = hardware / "schedule_contract.yaml"
    schedule.write_text("version: 1\n")
    seen = []
    monkeypatch.setattr(
        target_experiment,
        "load_target_experiment",
        lambda path: seen.append(path) or SimpleNamespace(target=Path(path).stem, hwbringup_set=str(hardware)),
    )
    monkeypatch.setattr(runner, "_endpoint_of", lambda target: ("external_backend", "fixture"))
    models = []

    def derive(target):
        model = isa_model.IsaModel(target=target, field_layout={"opcode": (0, 8)}, opcode_table={"Halt": 3})
        models.append(model)
        return model

    monkeypatch.setattr(isa_model, "isa_model_for_target", derive)
    first = isa_tools.broker_ctx(invocation)
    second = isa_tools.broker_ctx(_invocation(tmp_path, "another"))
    assert models == [], "model discovery must remain lazy"
    assert first.model() is first.model()
    assert first.model() is not second.model()
    assert [model.target for model in models] == ["synthetic", "another"]
    assert first.schedule_contract() == {"version": 1}
    schedule.write_text("version: 2\n")
    assert first.schedule_contract() == {"version": 2}
    assert invocation.descriptor in seen


@pytest.mark.parametrize("endpoint", [isa_tools.ROCC_ENDPOINT, "external_backend"])
def test_debug_callback_has_explicit_invocation_and_preserves_response(tmp_path, monkeypatch, endpoint):
    invocation = _invocation(tmp_path)
    monkeypatch.setattr(isa_tools, "_endpoint_and_target", lambda context: (endpoint, context.target))
    monkeypatch.setattr(isa_tools, "_model", lambda context: SimpleNamespace(is_empty=lambda: False))
    seen = []
    response = {"state": {"pc": 4}, "regions": [], "capsule": "public"}
    monkeypatch.setattr(isa_tools, "_rocc_debug", lambda req, target, context: seen.append((req, context)) or response)
    monkeypatch.setattr(isa_tools, "_handle_debug", lambda req, context: seen.append((req, context)) or response)
    request = {"cmd": "debug", "capsule": "public", "regions": [[0, 4]]}
    assert isa_tools._handle(request, isa_tools.broker_ctx(invocation)) is response
    assert seen == [(request, invocation)]


def test_debug_uses_declared_descriptor_and_existing_redaction_authority(tmp_path, monkeypatch):
    from merlin_experiments.corpus import admission as corpus_workflow

    from merlin.targetgen import capsule_runner as runner
    from merlin.targetgen import program_oracle, target_experiment

    invocation = _invocation(tmp_path)
    public = tmp_path / "public"
    capsule = public / "fixture"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("id: fixture\n")
    (capsule / "golden.yaml").write_text("expected: PRIVATE_SENTINEL\n")
    seen = []
    monkeypatch.setattr(
        target_experiment,
        "load_target_experiment",
        lambda path: seen.append(path) or SimpleNamespace(target="synthetic"),
    )
    monkeypatch.setattr(runner, "_endpoint_of", lambda target: ("external_backend", "fixture-model"))
    monkeypatch.setattr(corpus_workflow, "public_capsules_for", lambda target: public)
    monkeypatch.setattr(program_oracle, "build_debug_cb", lambda target, path: {"input": "opaque"})

    def run(target, **kwargs):
        assert target == "synthetic"
        assert kwargs["model_ext"] == "fixture-model"
        assert kwargs["dump_regions"] == [[1, 2]]
        assert kwargs["kernel_s"].read_text() == "submitted kernel"
        return {"regions": [], "state": {"pc": 4}}  # Existing oracle owns output filtering.

    monkeypatch.setattr(program_oracle, "run_program_debug", run)
    result = isa_tools._handle_debug(
        {"capsule": "fixture", "kernel_s": "submitted kernel", "regions": [[1, 2]]},
        invocation,
    )
    assert result == {"regions": [], "state": {"pc": 4}, "capsule": "fixture"}
    assert seen == [invocation.descriptor]
    assert "PRIVATE_SENTINEL" not in json.dumps(result)


_BOOTSTRAP = """
import importlib.abc,json,sys
sys.path[:0]=json.loads(sys.argv.pop(1))
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname in {'_common','qa_check','tier_promote','run_baseline_qa_loop',
                        'isa_tools_broker','cca_broker'}:
            raise AssertionError('package broker imported native owner: '+fullname)
sys.meta_path.insert(0,NoNative())
"""


def _command(script, *args):
    return [
        sys.executable,
        "-I",
        "-c",
        _BOOTSTRAP + script,
        json.dumps([str(path) for path in python_import_roots()]),
        *map(str, args),
    ]


@pytest.mark.parametrize("module", ["isa_tools", "cca"])
def test_package_import_and_help_are_inert(tmp_path, module):
    script = f"""
import os,pathlib,subprocess
before=dict(os.environ)
def forbidden(*args,**kwargs): raise AssertionError('help initialized invocation')
subprocess.run=forbidden
original_read=pathlib.Path.read_text
def guarded_read(path,*args,**kwargs):
    assert path.name not in {{'experiment.env','target_experiment.yaml'}}, 'help read experiment inputs'
    return original_read(path,*args,**kwargs)
pathlib.Path.read_text=guarded_read
from merlin_experiments.phase1.brokers import {module} as broker
try: broker.main(['--help'])
except SystemExit as exc: assert exc.code==0
else: raise AssertionError('help did not exit')
assert dict(os.environ)==before
"""
    result = subprocess.run(_command(script), cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_installed_isa_cli_requires_explicit_context(tmp_path):
    result = subprocess.run(
        _command(
            "from merlin_experiments.phase1.brokers.isa_tools import main\nmain(sys.argv[1:])\n", "--ws", tmp_path
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 2
    assert "requires --descriptor and --repo" in result.stderr
    assert not (tmp_path / ".isa_channel").exists()


def _wait_for(path, process):
    deadline = time.monotonic() + 15
    while not path.exists():
        assert process.poll() is None, process.communicate()[1]
        assert time.monotonic() < deadline, f"broker did not publish {path.name}"
        time.sleep(0.02)


@pytest.mark.parametrize(
    "module,channel,client,client_args",
    [
        ("isa_tools", ".isa_channel", "isa", ["asm", "Halt"]),
        ("cca", ".cca_channel", "cca", ["escalation-ladder", "synthetic.axis", "synthetic"]),
    ],
)
def test_real_package_broker_and_standalone_client_ipc(tmp_path, module, channel, client, client_args):
    descriptor = tmp_path / "explicit.yaml"
    descriptor.write_text("target: synthetic\n")
    (tmp_path / "private.txt").write_text("PRIVATE_SENTINEL")
    broker_script = ""
    args = ["--ws", tmp_path, "--poll", "0.01"]
    if module == "isa_tools":
        # Substitute external target discovery, not broker routing/encoding/protocol.
        broker_script = """
from merlin.targetgen import capsule_runner,isa_model
capsule_runner._endpoint_of=lambda target: ('external_backend','synthetic')
isa_model.isa_model_for_target=lambda target: isa_model.IsaModel(
    target=target,by_mnemonic={'Halt':{'class':'Halt','role':'scalar',
    'fixed_mask':0xffffffff,'fixed_value':0x73,'fields':{}}})
"""
        args += ["--descriptor", descriptor, "--repo", tmp_path]
    broker_script += f"from merlin_experiments.phase1.brokers.{module} import main\nmain(sys.argv[1:])\n"
    env = {key: value for key, value in os.environ.items() if not key.startswith("MERLIN_")}
    channel_path = tmp_path / channel
    if module == "isa_tools":
        channel_path.mkdir()
        (channel_path / "req_previous.json").write_text('{"cmd": "asm", "text": "NOPE"}')
        (channel_path / "resp_previous.json").write_text('{"previous": "unchanged"}')
        (channel_path / "done_previous").write_text("ok")
    process = subprocess.Popen(
        _command(broker_script, *args), cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        _wait_for(channel_path, process)
        staged = tmp_path / "public_client.py"
        source = module_source_path(f"merlin_experiments.phase1.tools.{client}")
        shutil.copyfile(source, staged)
        assert staged.read_bytes() == source.read_bytes()
        result = subprocess.run(
            [sys.executable, "-I", str(staged), *client_args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        response = json.loads(result.stdout)
        if module == "isa_tools":
            assert response == {"words": ["0x00000073"], "word_lines": ".word 0x00000073\n", "n": 1}
            assert json.loads((channel_path / "resp_previous.json").read_text()) == {"previous": "unchanged"}
        else:
            assert response == cca._handle(
                {"cmd": "escalation_ladder", "axis": "synthetic.axis", "target": "synthetic"}
            )
        assert "PRIVATE_SENTINEL" not in result.stdout
        (channel_path / "req_malformed.json").write_text("[")
        _wait_for(channel_path / "done_malformed", process)
        malformed = json.loads((channel_path / "resp_malformed.json").read_text())
        assert "JSONDecodeError" in malformed["error"]
        assert process.poll() is None, "malformed request must not kill broker"
    finally:
        channel_path.mkdir(exist_ok=True)
        (channel_path / "STOP").write_text("stop")
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    assert process.returncode == 0, stdout + stderr


def test_new_broker_sources_are_in_the_existing_phase1_inventory():
    from merlin_experiments.phase1 import source_inputs

    package = module_source_path("merlin_experiments.phase1").parent
    members = source_inputs._python_members(package, "phase1:source:")
    for name in ("__init__", "isa_tools", "cca"):
        assert f"phase1:source:brokers/{name}.py" in members
