"""Shared synthetic installed-owner fixture; no compiler or simulator execution."""

import json
from pathlib import Path
from types import SimpleNamespace

from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import portfolio_analysis as PA
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.global_inputs import GlobalExperimentInputs
from merlin_experiments.phase2.mechanism_program import MechanismProgram
from merlin_experiments.phase2.mechanism_rounds import MechanismRounds
from merlin_experiments.phase2.portfolio_evaluation import FastPortfolioEvaluation
from merlin_experiments.phase2.revision_journal import RevisionJournal
from merlin_experiments.phase2.revision_session import RevisionSession
from merlin_experiments.phase2.stage_inputs import StageE2ESentinel

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes


def digest(text):
    return sha256_bytes(text.encode())


class SyntheticAnalyzer:
    """Implement the default analyzer's artifact-sink protocol using only synthetic bytes."""

    def __init__(self):
        self.calls = []
        self.hook = None
        self.substitute = None
        self.fail_member = None

    def __call__(self, baseline, candidate, sentinel, **kwargs):
        self.calls.append((candidate, sentinel.capsule_sha256, kwargs["timeout_s"]))
        if self.hook is not None:
            self.hook(candidate, sentinel, kwargs)
        if sentinel.capsule == self.fail_member:
            raise RuntimeError("synthetic lowering failure")
        revision = int((candidate / "version.txt").read_text())
        candidate_sha = hash_tree(candidate)["sha256"]
        interface = Path(sentinel.frozen_source_path) / "capsule.interface.mlir"
        source_sha = C.sha256_file(interface)
        lowered = f"synthetic lowered revision={revision} member={sentinel.capsule}"
        commands = json.dumps({"revision": revision, "member": sentinel.capsule})
        baseline_lowered, baseline_commands = "synthetic baseline IR", '{"baseline":true}'
        emission = {
            "candidate_lowered_sha256": digest(lowered),
            "candidate_command_buffer_sha256": digest(commands),
            "baseline_lowered_sha256": digest(baseline_lowered),
            "baseline_command_buffer_sha256": digest(baseline_commands),
        }
        graph = digest(f"graph:{sentinel.capsule}")
        kwargs["artifact_sink"](
            {
                "candidate_sha256": candidate_sha,
                "candidate_lowered_sha256": emission["candidate_lowered_sha256"],
                "candidate_command_buffer_sha256": emission["candidate_command_buffer_sha256"],
                "decoded_trace": {},
                "task_instruction_evidence": {},
                "lowered_text": lowered,
                "command_buffer_text": commands,
                "command_buffer": json.loads(commands),
                "interface": str(interface),
                "baseline_artifacts": {
                    "identity": {
                        "baseline_sha256": hash_tree(baseline)["sha256"],
                        "capsule_sha256": sentinel.capsule_sha256,
                        "target": kwargs["target"],
                    },
                    "lowered_text": baseline_lowered,
                    "lowered_sha256": digest(baseline_lowered),
                    "command_buffer_text": baseline_commands,
                    "command_buffer_sha256": digest(baseline_commands),
                },
            }
        )
        result = {
            "candidate_sha256": candidate_sha,
            "workload": {"capsule_sha256": sentinel.capsule_sha256},
            "emission": emission,
            "diagnostics": {
                "captured_logical_graph": {
                    "status": "verified",
                    "logical_dispatch_digest": graph,
                    "source_sha256": source_sha,
                },
                "arms": {"candidate": {"status": "emitted", "macs": 1, "exact": True, "movement": {"known_bytes": 1}}},
                "verified_global_plan_emission": {
                    "status": "verified",
                    "plan_digest": digest(lowered + "plan"),
                    "candidate_sha256": candidate_sha,
                    "logical_dispatch_digest": graph,
                    "source_sha256": source_sha,
                    **emission,
                    "host_activity": {
                        "load_payload_bytes": 100 - revision * 10,
                        "store_payload_bytes": 20,
                        "static_operations": {"allocation": 1},
                    },
                },
            },
        }
        if self.substitute == "candidate":
            result["candidate_sha256"] = digest("wrong-candidate")
        elif self.substitute == "member":
            result["workload"]["capsule_sha256"] = digest("wrong-member")
        return result


def build_case(tmp_path, monkeypatch, *, source_snapshot_root=None, source_snapshot_files_sha256=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline, candidate, shared = [tmp_path / name for name in ("baseline", "candidate", "shared")]
    for path in (baseline, candidate):
        path.mkdir()
        (path / "compiler.py").write_text("from merlin.helper import VALUE\n")
        (path / "version.txt").write_text("0")
        (path / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    (shared / "helper.py").write_text("VALUE = 1\n")
    sentinels = []
    for index in range(2):
        source = tmp_path / f"capsule-{index}"
        source.mkdir()
        (source / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
        (source / "capsule.interface.mlir").write_text(f"module {{ // synthetic member {index}\n}}\n")
        sentinels.append(
            StageE2ESentinel(
                source.name, str(source), str(source), C.exact_tree_record(source)["sha256"], ("lane",), ("L2",)
            )
        )
    controller = tmp_path / "controller.py"
    controller.write_text("SYNTHETIC_CONTROLLER = True\n")
    resources = tmp_path / "contract"
    for relative in HP.RESOURCE_FILES:
        path = resources / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"synthetic_resource": relative}))
    output = tmp_path / "run"
    prepared = GlobalExperimentInputs.prepare(
        baseline=baseline,
        baseline_sha256=hash_tree(baseline)["sha256"],
        sentinel=sentinels[0],
        portfolio_sentinels=sentinels[1:],
        target="synthetic-unregistered-target",
        target_sha256=digest("target"),
        output=output,
        compiler_shared_source_root=shared,
        contract_root=resources,
        controller_source=controller,
        source_snapshot_root=source_snapshot_root,
        source_snapshot_files_sha256=source_snapshot_files_sha256,
    )
    output.mkdir()
    inputs = prepared.materialize(output)
    edit = FrozenEditAuthority(output)
    program = MechanismProgram(
        output,
        edit,
        portfolio_identity=inputs.portfolio_identity,
        portfolio_identity_sha256=inputs.portfolio_identity_sha256,
    )
    journal = RevisionJournal(output)
    session = RevisionSession(
        inputs=inputs,
        edit_authority=edit,
        mechanism_program=program,
        mechanism_rounds=MechanismRounds(program),
        fast_evaluation=FastPortfolioEvaluation(inputs.portfolio_sentinels),
        journal=journal,
    )
    analyzer = SyntheticAnalyzer()
    # Preserve the historical default-analyzer artifact protocol, without running the real compiler.
    monkeypatch.setattr(EA, "analyze_whole_model_emission", analyzer)
    owner = PA.PortfolioAnalysis(session, analyzer=analyzer)
    experiment = {
        "baseline_sha256": inputs.baseline_sha256,
        "target_sha256": inputs.target_sha256,
        "optimization_baseline_sha256": inputs.optimization_baseline_sha256,
        "optimization_baseline": inputs.optimization_baseline_binding,
        "phase1_qualification": inputs.phase1_binding,
        "portfolio": inputs.portfolio_identity,
        "portfolio_sha256": inputs.portfolio_identity_sha256,
    }
    (output / "experiment.json").write_bytes(C.canonical_json(experiment))
    return SimpleNamespace(
        root=tmp_path,
        output=output,
        inputs=inputs,
        candidate=candidate,
        analyzer=analyzer,
        owner=owner,
        session=session,
        journal=journal,
    )
