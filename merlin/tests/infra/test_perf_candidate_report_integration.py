"""Real Phase-2 evidence admission with explicitly synthetic execution inputs.

This fixture never qualifies hardware or OS sandbox isolation. Provider and simulator
transports will be synthetic; source, snapshot, corpus and evidence verifiers are real.
The functional scores describe a synthetic input fixture, not a historical run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import candidate_record as RECORD
from merlin_experiments.phase2 import candidate_verification as VERIFY
from merlin_experiments.phase2 import corpus as CORPUS
from merlin_experiments.phase2 import development_feedback as DF
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import gsim_workload as WORKLOAD
from merlin_experiments.phase2 import stage_inputs as INPUTS

# Reuse the existing synthetic contracts, not their mocked producer/verifier paths.
from test_perf_agent_stage import PAS, _pk_capsules
from test_perf_gate_incomplete_ratio import _clean_run
from test_produce_gsim_certificate import PRODUCER, _artifacts

from merlin.benchharness import hash_tree
from merlin.common.paths import build_dir, repo_root
from merlin.targetgen.rtl.mlc_bridge import core_hw_mlir
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.target_experiment import load_target_experiment


def _yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def admitted_inputs(monkeypatch):
    """Real sealed bundle and PK admission; all input bytes belong to this test."""
    parent = build_dir() / "tests" / "candidate-report"
    parent.mkdir(parents=True, exist_ok=True)
    # Sentinel mapping intentionally requires a repository-relative source grant.
    # TemporaryDirectory restores readonly fixture permissions during cleanup.
    with tempfile.TemporaryDirectory(prefix="fixture-", dir=parent) as temporary:
        root = Path(temporary)
        monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
        monkeypatch.setenv("MERLIN_OUT_ROOT", str(root / "generated"))
        corpus = root / "inputs" / "capsules"
        public = corpus / "public"
        public.mkdir(parents=True)
        for prototype in _pk_capsules():
            descriptor = dict(prototype.descriptor, source_role="derived_sweep")
            capsule = corpus / prototype.source_relative_path
            _yaml(capsule / "capsule.yaml", descriptor)
            (capsule / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
        _yaml(
            corpus / "MANIFEST.yaml",
            {
                "generated": [member.source_relative_path for member in _pk_capsules()],
                "hand_authored": [],
                "performance_generation": {
                    "fixture": {
                        "errors": [],
                        "phase": {"category": "_perf", "label": "dev", "included_in_functional_grade": False},
                    }
                },
            },
        )
        sentinel = public / "p0"
        _yaml(
            sentinel / "capsule.yaml",
            {
                "name": "p0",
                "kind": "model",
                "label": "public",
                "lanes": {"require": ["on_mesh", "scalar_rvv_lane"]},
                "required_oracle_tiers": ["L2", "L3"],
                "performance": {"global_objective": True},
            },
        )
        (sentinel / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
        host = root / "inputs" / "host"
        _yaml(host / "manifest.yaml", {"name": "synthetic-host"})
        descriptor = root / "target.yaml"
        _yaml(descriptor, {"target": "fixture", "capsule_corpus": str(public)})
        target = load_target_experiment(descriptor)

        runs = root / "functional-runs"
        run, digest = _clean_run(runs, "synthetic-functional")
        _install_compiler(run / "submission")
        digest = hash_tree(run / "submission")["sha256"]
        freeze_path = run / "freeze.json"
        freeze = json.loads(freeze_path.read_text())
        freeze.update(submission_sha256=digest, submission_sha256_recheck=digest)
        freeze_path.write_text(json.dumps(freeze))
        run_manifest = yaml.safe_load((run / "run_manifest.yaml").read_text())
        run_manifest["submission_sha256"] = digest
        _yaml(run / "run_manifest.yaml", run_manifest)
        workspace = run / "workspace"
        workspace.mkdir()
        bundle = {"allowed": [{"path": (root / "inputs").relative_to(repo_root()).as_posix()}]}
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo_root())
        snapshot = BW.snapshot_record(workspace)
        frozen_host = BW.snapshot_input_paths(workspace, bundle, [host], repo=repo_root())[0]
        environment = yaml.safe_load((run / "environment.yaml").read_text())
        environment["bundle_input_snapshot"] = snapshot
        environment["model_host_lane_snapshot"] = {
            "target": "fixture-host",
            "run_id": "synthetic-host",
            "run_snapshot": snapshot,
            "package": host.relative_to(repo_root()).as_posix(),
            "resolved_package": str(frozen_host),
            "package_sha256": hash_tree(frozen_host)["sha256"],
            "n_files": hash_tree(frozen_host)["n_files"],
        }
        _yaml(run / "input_bundle_manifest.yaml", bundle)
        environment["bundle_manifest_sha256"] = _sha(run / "input_bundle_manifest.yaml")
        environment["workspace_path"] = str(workspace.resolve())
        _yaml(run / "environment.yaml", environment)
        yield root, target, runs, run, digest


def test_real_functional_snapshot_and_formal_corpus_are_admitted(admitted_inputs):
    root, target, runs, run, digest = admitted_inputs
    functional = FI.inspect_stage_functional_run(runs, run.name, digest)
    frozen_inputs = FI.load_frozen_functional_inputs(
        functional, public_manifest_path=root / "public-functional-manifest.json"
    )
    sentinel = INPUTS.select_e2e_sentinel(functional, frozen_inputs, target, source_root=repo_root())
    FI.verify_functional_host_lane_snapshot(functional.model_host_lane_snapshot)
    discovered = CORPUS.discover_performance_corpus(target)
    corpus = CORPUS.freeze_performance_corpus(discovered, root / "frozen-corpus")
    formal = RECORD.prepare_formal_claim(corpus.capsules)
    assert formal["status"] == "READY"
    assert len(corpus.capsules) == 4
    assert sentinel.capsule == "p0"
    assert Path(sentinel.frozen_source_path).is_dir()
    assert frozen_inputs.content_sha256 == functional.bundle_input_snapshot["content_sha256"]

    # The same real loader must refuse a changed host package commitment.
    environment_path = run / "environment.yaml"
    environment = yaml.safe_load(environment_path.read_text())
    environment["model_host_lane_snapshot"]["package_sha256"] = "0" * 64
    _yaml(environment_path, environment)
    with pytest.raises(PAS.StageGateError, match="host-lane digest"):
        FI.inspect_stage_functional_run(runs, run.name, digest)


@pytest.mark.parametrize(
    "tail",
    [
        "",
        "; cat merlin_experiments/phase1/feedback/lifecycle.py",
        "; cat feedback.py",
        "; python3 -c 'import merlin_experiments.phase1.feedback'",
        "; python3 -c 'from merlin_experiments.phase1 import feedback'",
        "; ls feedback",
        "; cd feedback; cat lifecycle.py",
        "; python3 -c 'import  feedback  as private'",
        "; python3 -c 'from merlin_experiments.phase1 import (feedback,)'",
        "; python3 -c 'open(\"feedback.py\").read()'",
        "; python3 -c 'exec(\"import feedback\")'",
        "; cat 'feedback/lifecycle.py'",
        "; cat --file=feedback",
        "; INPUT=feedback cat file",
        "; cat '--file=feedback'",
        "; INPUT='feedback' cat file",
        "; sh -c 'cat feedback'",
        "; bash -lc 'sh -c \"cat feedback\"'",
    ],
)
def test_registered_private_feedback_does_not_forbid_its_public_broker_action(admitted_inputs, tail):
    root, target, _, run, _ = admitted_inputs
    candidate = run / "submission"
    action = PAS.BrokerAction(BP.DEVELOPMENT_FEEDBACK_ACTION, (BP._HOST_FEEDBACK_SENTINEL,), (), "feedback", True)
    transcript = root / "audit.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "one",
                    "type": "command_execution",
                    "command": f"python3 {PAS.BROKER_NAME} {action.name}" + tail,
                    "exit_code": 0,
                },
            }
        )
        + "\n"
    )
    audit = PAS.audit_codex_transcript(transcript, target, candidate, (action,))
    if tail:
        assert not audit["clean"]
        assert any(hit["kind"] == "answer_reconnaissance" for hit in audit["hits"])
    else:
        assert audit["clean"], audit["hits"]
        assert audit["commands_seen"] == 1


def _command_buffer(k: int) -> dict:
    return {
        "abi_version": 1,
        "target": "fixture",
        "tensors": {
            "A": {"role": "input", "shape": [16, k], "dtype": "i8", "data": [1] * (16 * k)},
            "B": {"role": "weight", "shape": [k, 16], "dtype": "i8", "data": [1] * (16 * k)},
            "Y": {"role": "output", "shape": [16, 16], "dtype": "i32"},
        },
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "B", "dst": "acc"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc", "dst": "Y"},
                "attrs": {"epilogue": [], "output_dtype": "i32"},
            },
        ],
    }


def _install_compiler(submission: Path) -> None:
    commands = {
        name: {"argv": ["{tool}", name, "{input_mlir}"]}
        for name in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
    }
    commands["emit_command_buffer"]["argv"].append("{output_json}")
    _yaml(
        submission / "manifest.yaml",
        {
            "artifact_type": "mlir_oot_target_backend",
            "target": "fixture",
            "language": "python",
            "authoring": {"mode": "deterministic_generated_from_spec"},
            "integrity_exempt": False,
            "entrypoints": {"tool": "tool.py"},
            "commands": commands,
        },
    )
    tool = submission / "tool.py"
    tool.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\n"
        + "assert sys.argv[1] in "
        + repr(tuple(commands))
        + "\n"
        + "assert Path(sys.argv[2]).is_file()\n"
        + "if sys.argv[1] == 'emit_command_buffer':\n"
        + "    Path(sys.argv[3]).write_text(json.dumps("
        + repr(_command_buffer(16))
        + "))\n"
        + "print('module {}')\n"
    )
    tool.chmod(0o755)


class _SyntheticSimulator:
    """External execution fake only; independent integer reference stays real."""

    def __init__(self, k: int):
        self.k = k
        self.engines = []

    def available(self, engine: str) -> bool:
        return engine in {"gsim", "verilator"}

    def run_elf(self, elf: Path, *, simulator: str, timeout: int) -> str:
        assert elf.read_bytes() == f"synthetic-elf-k{self.k}".encode()
        assert timeout > 0
        self.engines.append(simulator)
        return json.dumps({"Y": [[self.k] * 16 for _ in range(16)]})

    def parse_output(self, console: str):
        return json.loads(console), console


def _certificate(root: Path, target, *, members=None) -> Path:
    artifact_root = root / "synthetic-hardware"
    artifact_root.mkdir()
    artifacts, build_receipt = _artifacts(artifact_root)
    captures = []
    for member in CORPUS.discover_performance_corpus(target).capsules if members is None else members:
        k = member.descriptor["inputs"][0]["shape"][1]
        lowered = artifact_root / member.capsule
        lowered.mkdir()
        (lowered / "command_buffer.json").write_text(json.dumps(_command_buffer(k)))
        (lowered / "lowered.llvm.mlir").write_text("module {}\n")
        simulator = _SyntheticSimulator(k)

        def build_elf(buffer, llvm, destination, *, k=k):
            assert buffer == _command_buffer(k)
            assert llvm == "module {}\n"
            elf = destination / "case.elf"
            elf.write_bytes(f"synthetic-elf-k{k}".encode())
            return elf

        capture = PRODUCER.capture_case(
            target=target.target,
            capsule_manifest=member.source_dir / "capsule.yaml",
            artifact_dir=lowered,
            workdir=lowered / "execution",
            artifacts=artifacts,
            backend=simulator,
            build_elf=build_elf,
        )
        assert simulator.engines == ["verilator", "gsim"]
        assert capture["reference"]["elf_sha256"] == capture["candidate"]["elf_sha256"]
        path = lowered / "capture.json"
        path.write_text(json.dumps(capture))
        captures.append(path)
    document = PRODUCER.produce_certificate(
        target=target.target, captures=captures, artifacts=artifacts, build_receipt=build_receipt
    )
    certificate = artifact_root / "certificate.json"
    certificate.write_text(json.dumps(document))
    return certificate


def test_real_certificate_covers_each_admitted_pk_workload(admitted_inputs):
    root, target, *_ = admitted_inputs
    certificate = _certificate(root, target)
    loaded = GATE.load_certificate(certificate, expected_sha256=_sha(certificate))
    assert loaded.target == target.target
    assert len(loaded.members) == 4
    for member in CORPUS.discover_performance_corpus(target).capsules:
        workload = WORKLOAD.derive_workload(member.source_dir / "capsule.yaml")
        decision = GATE.plan_evaluation(loaded, workload, phase="development_correctness", gsim_available=True)
        assert decision.admitted and decision.selected_engine == "gsim"


def _install_transport(root: Path, target, submission: Path, monkeypatch) -> tuple[Path, Path]:
    """Do not interpret a shell or execute submitted bytes outside a real sandbox.

    This fake accepts the actual policy argv, emulates three known version probes,
    and emulates only this fixture's exact compiler bytes and four ABI commands.
    Everything else is refused. It establishes no kernel-isolation property.
    """
    bins = root / "bin"
    bins.mkdir()
    original = (submission / "tool.py").read_bytes()
    script = """import hashlib,json,sys
from pathlib import Path
cfg=json.loads(Path(__file__).with_suffix('.json').read_text())
root=Path(cfg['root']).resolve()
args=sys.argv[1:]
mounts=[]
cwd=None
arity={
 '--ro-bind':2,'--bind':2,'--ro-bind-try':2,'--bind-try':2,'--dev-bind':2,
 '--setenv':2,'--unsetenv':1,'--tmpfs':1,'--proc':1,'--dev':1,'--dir':1,
 '--chdir':1,'--die-with-parent':0,'--new-session':0,'--clearenv':0,
 '--unshare-user':0,'--unshare-pid':0,'--unshare-uts':0,'--unshare-ipc':0,
 '--unshare-cgroup':0,'--unshare-cgroup-try':0,'--unshare-user-try':0,
 '--uid':1,'--gid':1,'--cap-drop':1,'--hostname':1,
}
while args and args[0].startswith('--'):
    option=args.pop(0)
    assert option in arity, ('unexpected bwrap option',option)
    count=arity[option]
    assert len(args)>=count
    values,args=args[:count],args[count:]
    if option=='--chdir': cwd=Path(values[0]).resolve()
    if option in ('--ro-bind','--bind','--ro-bind-try','--bind-try'):
        mounts.append(tuple(values))
assert cwd and cwd.is_relative_to(root), ('foreign cwd',cwd)
assert args[:2]==['bash','-c'], ('unexpected transport',args)
if len(args)==3:
    assert args[2] in cfg['probes'], ('unknown probe',args[2])
    print('synthetic fixture version probe (OS isolation unqualified)')
    raise SystemExit(0)
assert args[2]==cfg['execution_prefix'] and args[3]=='perf-tool'
tool=Path(args[4]).resolve()
assert tool.is_relative_to(root) and tool.name=='tool.py'
assert hashlib.sha256(tool.read_bytes()).hexdigest() in cfg['tool_sha256']
payload=args[5:]
assert payload[0] in cfg['commands']
assert len(payload)==(3 if payload[0]=='emit_command_buffer' else 2)
input_path=Path(payload[1])
for source,destination in reversed(mounts):
    dest=Path(destination)
    if input_path.is_relative_to(dest):
        input_path=Path(source)/input_path.relative_to(dest)
        break
assert input_path.resolve().is_relative_to(root) and input_path.is_file()
if payload[0]=='emit_command_buffer':
    output=Path(payload[2])
    if not output.is_absolute(): output=cwd/output
    assert output.resolve().is_relative_to(cwd) and not output.is_symlink()
    output.write_text(json.dumps(cfg['command_buffer']))
print('module {}')
"""
    bwrap = bins / "bwrap"
    bwrap.write_text(f"#!{sys.executable}\n" + script)
    bwrap.chmod(0o755)
    # sandbox_env is candidate-root-dependent only for declared curated harnesses,
    # which this fixture intentionally does not have.
    prefix = TC.sandbox_env(target, submission)
    (bins / "bwrap.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "probes": [prefix + probe.cmd for probe in TC.required_tool_probes(target)],
                "execution_prefix": prefix + 'exec "$@"',
                "commands": list(yaml.safe_load((submission / "manifest.yaml").read_text())["commands"]),
                "tool_sha256": [
                    hashlib.sha256(value).hexdigest() for value in (original, original + b"# synthetic optimization\n")
                ],
                "command_buffer": _command_buffer(16),
            }
        )
    )
    codex = bins / "codex"
    codex.write_text("#!/bin/sh\nexit 91\n")
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(bins) + os.pathsep + os.environ["PATH"])
    prices = root / "prices.yaml"
    prices.write_text("gpt-fixture: [5, 30, 0.5, 5]\n")
    return codex, prices


def _simulator_execution(**kwargs) -> dict:
    certificate, decision = kwargs["certificate"], kwargs["decision"]
    k = kwargs["member"].descriptor["inputs"][0]["shape"][1]
    cycles = 100 + k if kwargs["arm"] == "baseline" else 90 + k
    pins = certificate.pins
    observed = {
        "engine": "gsim",
        "status": "pass",
        "derived_from_rtl": True,
        "cycle_accurate": True,
        "binary_sha256": pins["gsim_binary"]["sha256"],
        "firrtl_sha256": pins["gsim_firrtl"]["sha256"],
        "model_sha256": pins["gsim_model"]["sha256"],
        "elf_sha256": hashlib.sha256(f"synthetic-elf-k{k}".encode()).hexdigest(),
        "cycles": cycles,
        "observed_engine_binaries": {"status": "observed", "digests": [_sha(Path(pins["gsim_binary"]["path"]))]},
    }
    return {
        "measurement": {
            "numeric": "pass",
            "status": "pass",
            "failure": None,
            "per_sim": {"spike": {"correct": True}, "gsim": {"correct": True, "cycles": cycles}},
            "gsim_qualification": GATE.validate_execution(certificate, decision, observed),
        }
    }


def _provider_round(
    workspace, stage_root, prompt, target, agent_inputs, frozen, baseline, corpus_manifest, control, **kwargs
):
    candidate = workspace / "submission"
    with (candidate / "tool.py").open("ab") as stream:
        stream.write(b"# synthetic optimization\n")
    actions = BP.action_registry(BP.CORPUS_FEEDBACK_V1, candidate, target)
    events = [{"type": "thread.started", "thread_id": stage_root.name}, {"type": "turn.started"}]
    interface = next(path for path in sorted(agent_inputs.root.rglob("capsule.interface.mlir")))
    public_input = str(PAS.AGENT_CORPUS_MOUNT / interface.relative_to(agent_inputs.root))
    for index, action in enumerate(action for action in actions if action.required):
        bindings = []
        for name in action.placeholders:
            assert name in {"input_mlir", "output_json"}
            value = public_input if name == "input_mlir" else str(candidate / "output.json")
            bindings.append(f"{name}={value}")
        argv = [action.name, *bindings]
        command = shlex.join(["python3", PAS.BROKER_NAME, *argv])
        item = {"id": str(index), "type": "command_execution", "command": command}
        events.append({"type": "item.started", "item": item})
        result = subprocess.run(
            [sys.executable, str(control / Path(PAS.BROKER_NAME).name), *argv],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, (action.name, result.stdout, result.stderr)
        events.append(
            {
                "type": "item.completed",
                "item": {
                    **item,
                    "exit_code": result.returncode,
                    "aggregated_output": result.stdout,
                },
            }
        )
    events.append(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 2000,
                "cached_input_tokens": 500,
                "output_tokens": 400,
            },
        }
    )
    rounds = stage_root / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    raw = rounds / "round_00.codex_events.raw.jsonl"
    raw.write_text("".join(json.dumps(event) + "\n" for event in events))
    (rounds / "round_00.codex_events.timestamped.jsonl").write_text(
        "".join(
            json.dumps({"seq": i, "arrived_at": f"2026-09-20T00:00:{i:02d}+00:00", "event": event}) + "\n"
            for i, event in enumerate(events, 1)
        )
    )
    for suffix, content in (("codex_stderr.log", ""), ("prompt.txt", prompt.text), ("final.txt", "synthetic run")):
        (rounds / f"round_00.{suffix}").write_text(content)
    (rounds / "round_00.codex_summary.json").write_text(
        json.dumps(
            {
                "billing_mode": "subscription_notional",
                "exit_code": 0,
                "usage_complete": True,
                "timed_out": False,
                "wall_s": len(events),
            }
        )
    )
    outer = AW.outer_codex_policy(workspace, agent_inputs, (), target, frozen, baseline, control, corpus_manifest)
    return 0, raw, outer


@pytest.fixture
def qualified_stage(admitted_inputs, monkeypatch):
    root, target, runs, run, digest = admitted_inputs
    certificate = _certificate(root, target)
    codex, prices = _install_transport(root, target, run / "submission", monkeypatch)
    mlc = root / "synthetic-modelir"
    (mlc / "mlc").mkdir(parents=True)
    # This is the upstream ModeLIR-owned layout, not a Merlin run artifact.
    circt = mlc / "runs" / "circt-arc" / target.target / "outputs" / f"{target.target}_core_hw.mlir"
    circt.parent.mkdir(parents=True)
    circt.write_text("module {}\n")
    monkeypatch.setenv("MERLIN_MLC_DIR", str(mlc))
    assert core_hw_mlir(target.target) == circt
    rtl_facts = root / "rtl_facts.json"
    rtl_facts.write_text(
        json.dumps(
            {
                "target": target.target,
                "inputs": {"core_hw_sha256": _sha(circt)},
                "facts": {"arrays": [{"name": "synthetic", "rows": 16, "cols": 16, "source": "synthetic fixture"}]},
            }
        )
    )
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(rtl_facts))
    prepare = DF.prepare_development_feedback

    def synthetic_execution_transport(**kwargs):
        feedback = prepare(**kwargs)
        feedback.executor = _simulator_execution
        return feedback

    monkeypatch.setattr(DF, "prepare_development_feedback", synthetic_execution_transport)
    monkeypatch.setattr(PAS, "_codex_round", _provider_round)
    stage_kwargs = dict(
        sandbox_inputs=PAS.PC.select_package_sandbox_inputs(target),
        suite="fixture-candidate-report",
        contract_root=repo_root() / "merlin/contract",
        source_root=repo_root(),
        functional_runs_root=runs,
        functional_run_id=run.name,
        functional_submission_sha256=digest,
        target_experiment=target,
        stage_root=root / "stage",
        model="gpt-fixture",
        effort="low",
        wall_budget_seconds=120,
        rounds=1,
        round_timeout_seconds=60,
        max_tool_calls=10,
        tool_timeout_seconds=20,
        codex_binary=str(codex),
        telemetry_price_table=prices,
        gsim_certificate=certificate,
        gsim_certificate_sha256=_sha(certificate),
        rtl_facts=rtl_facts,
    )
    record = PAS.run_stage(**stage_kwargs)
    document = json.loads(record.read_text())
    assert document["admission"]["consumable"], json.dumps(
        {
            "admission": document["admission"],
            "audit": document["agent"]["audit"],
            "commands": [
                row["item"]["command"]
                for row in map(json.loads, Path(document["agent"]["transcript"]).read_text().splitlines())
                if row.get("type") == "item.started"
            ],
        },
        indent=2,
    )
    verified = VERIFY.verify_candidate_handoff(record, verify_authoring_tools=True, target_experiment=target)
    assert verified.functional_submission_sha256 == digest
    assert verified.candidate_sha256 != digest
    return dict(
        root=root,
        target=target,
        runs=runs,
        run=run,
        digest=digest,
        certificate=certificate,
        stage_kwargs=stage_kwargs,
        record=record,
    )


def test_real_one_round_producer_verifies_its_candidate(qualified_stage, monkeypatch):
    _assert_requalification_pins_shared_matcher(
        qualified_stage["root"], qualified_stage["record"], qualified_stage["target"], monkeypatch
    )


@pytest.mark.parametrize("changed_role", ["audit_implementation", "phase2:candidate_verification.py"])
def test_requalification_v4_pins_actual_owners_without_native_execution(tmp_path, monkeypatch, changed_role):
    import copy

    from test_perf_agent_stage import _audit_only_refusal_record

    source = _audit_only_refusal_record()
    source_path = tmp_path / "source.json"
    source_path.write_bytes(PAS._canonical_json(source))
    source_path.chmod(0o444)
    clean = copy.deepcopy(source["agent"]["audit"])
    clean.update(clean=True, hits=[])
    monkeypatch.setattr(VERIFY, "verify_candidate_record", lambda path, **kwargs: json.loads(path.read_bytes()))
    monkeypatch.setattr(
        VERIFY,
        "audit_tokens",
        lambda target: {
            "answer": ("private-answer",),
            "grader": ("private-grader",),
            "oracle_subpath": ("private-oracle",),
        },
    )
    monkeypatch.setattr(VERIFY, "_recomputed_candidate_audits", lambda *args, **kwargs: (clean, [clean]))
    output = VERIFY.requalify_audit_only_candidate(source_path, tmp_path / "qualified/candidate.json", object())
    document = json.loads(output.read_bytes())
    RECORD._validate_audit_requalification(document)
    VERIFY._verify_audit_requalification(output, document, object())
    qualification = document["audit_requalification"]
    assert qualification["schema_version"] == 4
    assert {row["role"] for row in qualification["policy_snapshots"]} == {
        "audit_implementation",
        "answer_surface_policy",
        "shared_access_policy",
        "python_source_membership",
        *("phase2:" + name for name in qualification["phase2_source_identity"]["members"]),
    }
    for row in qualification["policy_snapshots"]:
        parent = Path(row["frozen_path"]).parent
        while parent != output.parent.parent:
            assert not parent.stat().st_mode & 0o222
            parent = parent.parent
    for version in (1, 2, 3):
        historical = copy.deepcopy(document)
        evidence = historical["audit_requalification"]
        evidence["schema_version"] = version
        roles = {"audit_implementation", "answer_surface_policy"}
        if version >= 2:
            roles.add("shared_access_policy")
        evidence["policy_snapshots"] = [row for row in evidence["policy_snapshots"] if row["role"] in roles]
        if version == 3:
            controller = copy.deepcopy(qualification["policy_snapshots"][0])
            controller["role"] = "native_controller"
            evidence["policy_snapshots"].append(controller)
        evidence.pop("phase2_source_identity")
        evidence["policy_set_sha256"] = PAS._sha256(PAS._canonical_json(evidence["policy_snapshots"]))
        RECORD._validate_audit_requalification(historical)
        VERIFY._verify_audit_requalification(output, historical, None)
        with pytest.raises(PAS.StageGateError, match="original native audit owner"):
            VERIFY._verify_audit_requalification(output, historical, object())
    row = next(row for row in qualification["policy_snapshots"] if row["role"] == changed_role)
    isolated = tmp_path / "isolated.py"
    isolated.write_bytes(Path(row["source_path"]).read_bytes())
    original_source = VERIFY.inspect.getsourcefile
    if changed_role.startswith("phase2:"):
        identity = copy.deepcopy(qualification["phase2_source_identity"])
        monkeypatch.setattr(VERIFY.TEL, "_package_source_record", lambda: identity)
    else:
        monkeypatch.setattr(
            VERIFY.inspect,
            "getsourcefile",
            lambda obj: str(isolated) if obj is VERIFY.audit_codex_transcript else original_source(obj),
        )
    VERIFY._verify_audit_requalification(output, document, object())
    isolated.write_bytes(isolated.read_bytes() + b"\n# drift\n")
    if changed_role.startswith("phase2:"):
        identity["members"]["candidate_verification.py"] = hashlib.sha256(isolated.read_bytes()).hexdigest()
    with pytest.raises(PAS.StageGateError, match="live .* differs"):
        VERIFY._verify_audit_requalification(output, document, object())


def _assert_requalification_pins_shared_matcher(root, record, target, monkeypatch):
    """A synthetic old audit policy, never an edited historical run or receipt."""
    original_bytes = record.read_bytes()
    old = json.loads(original_bytes)
    tokens = VERIFY.audit_tokens(target)
    # Explicit synthetic former policy declares the entire public action a
    # withheld token. Its false-positive audit is produced, not hand-certified.
    tokens["grader"] = (*tokens["grader"], BP.DEVELOPMENT_FEEDBACK_ACTION)
    combined, rounds = VERIFY._recomputed_candidate_audits(old, target, audit_token_set=tokens)
    assert combined["clean"] is False
    old["agent"]["audit"] = combined
    for row, audit in zip(old["agent"]["rounds"], rounds, strict=True):
        row["audit"] = audit
    old["state"] = "refused"
    old["admission"].update(consumable=False, refusal="combined Codex transcript failed the answer/tool-access audit")
    source = root / "synthetic-old-policy.json"
    source.write_bytes(PAS._canonical_json(old))
    source.chmod(0o444)
    source_bytes = source.read_bytes()
    updated = VERIFY.requalify_audit_only_candidate(source, root / "requalified/candidate.json", target)
    qualified = VERIFY.verify_candidate_record(updated, target_experiment=target)
    attribution = qualified["audit_requalification"]
    assert attribution["schema_version"] == 4
    assert "phase2:candidate_verification.py" in {row["role"] for row in attribution["policy_snapshots"]}
    assert record.read_bytes() == original_bytes and source.read_bytes() == source_bytes

    # Historical two-role records remain readable offline without inventing a
    # current helper pin; new two-role records are explicitly incomplete.
    historical = json.loads(updated.read_bytes())
    old_policy = historical["audit_requalification"]
    old_policy["schema_version"] = 1
    old_policy["policy_snapshots"] = [
        row
        for row in old_policy["policy_snapshots"]
        if row["role"] in {"audit_implementation", "answer_surface_policy"}
    ]
    old_policy["policy_set_sha256"] = hashlib.sha256(PAS._canonical_json(old_policy["policy_snapshots"])).hexdigest()
    RECORD._validate_audit_requalification(historical)
    old_policy["schema_version"] = 2
    with pytest.raises(PAS.StageGateError, match="provenance is incomplete"):
        RECORD._validate_audit_requalification(historical)

    old_policy["policy_snapshots"] = [
        row
        for row in attribution["policy_snapshots"]
        if row["role"] in {"audit_implementation", "answer_surface_policy", "shared_access_policy"}
    ]
    old_policy["policy_set_sha256"] = hashlib.sha256(PAS._canonical_json(old_policy["policy_snapshots"])).hexdigest()
    RECORD._validate_audit_requalification(historical)
    with pytest.raises(PAS.StageGateError, match="original native audit owner"):
        VERIFY._verify_audit_requalification(updated, historical, target)

    # Isolated byte-identical owners first verify, then independent drift refuses.
    for role, owner, refusal in (
        (
            "audit_implementation",
            Path(VERIFY.inspect.getsourcefile(VERIFY.audit_codex_transcript)),
            "live audit implementation",
        ),
    ):
        isolated = root / f"isolated_{role}.py"
        isolated.write_bytes(owner.read_bytes())
        original_source = VERIFY.inspect.getsourcefile
        with monkeypatch.context() as patch:
            patch.setattr(
                VERIFY.inspect,
                "getsourcefile",
                lambda obj: str(isolated) if obj is VERIFY.audit_codex_transcript else original_source(obj),
            )
            VERIFY.verify_candidate_record(updated, target_experiment=target)
            isolated.write_bytes(isolated.read_bytes() + b"\n# changed isolated owner\n")
            with pytest.raises(PAS.StageGateError, match=refusal):
                VERIFY.verify_candidate_record(updated, target_experiment=target)

    inspect_source = VERIFY.inspect.getsourcefile
    helper = root / "isolated_access.py"
    helper.write_bytes(Path(inspect_source(VERIFY.audit_token_in)).read_bytes())
    monkeypatch.setattr(
        VERIFY.inspect,
        "getsourcefile",
        lambda obj: str(helper) if obj is VERIFY.audit_token_in else inspect_source(obj),
    )
    VERIFY.verify_candidate_record(updated, target_experiment=target)
    helper.write_bytes(helper.read_bytes() + b"\n# changed matcher\n")
    with pytest.raises(PAS.StageGateError, match="live shared access policy differs"):
        VERIFY.verify_candidate_record(updated, target_experiment=target)


@pytest.mark.parametrize(
    "module",
    ["merlin.common.access", "merlin.targetgen.sandbox.answer_surfaces", "merlin.targetgen.sandbox.read_audit"],
)
def test_native_source_gate_refuses_changed_audit_owner(tmp_path, monkeypatch, module):
    from merlin_experiments.phase1 import source_inputs as SI
    from merlin_experiments.spec import SpecError

    harness = tmp_path / "merlin/experiments/capsule_bench/harness"
    harness.mkdir(parents=True)
    entry = harness / "run_baseline_qa_loop.py"
    entry.write_text("# synthetic native entry\n")
    for name in ("_common.py", "sandbox_toolchain.py"):
        (harness / name).write_text("# synthetic native startup\n")
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: fixture\n")
    source = tmp_path / "audit_owner.py"
    source.write_bytes(SI._source(module).read_bytes())
    original = SI._source
    monkeypatch.setattr(SI, "_source", lambda name: source if name == module else original(name))
    context = {"repo": tmp_path, "entrypoint": entry, "require_native": True, "descriptor": descriptor}
    record = SI.record(**context)
    SI.verify(record, **context)
    source.write_bytes(source.read_bytes() + b"\n# changed audit owner\n")
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **context)
