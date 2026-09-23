"""File-only synthetic portfolio evidence; concrete admission remains unpatched."""

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import portfolio_checkpoint as PC
from merlin_experiments.phase2 import static_identity as SI

from merlin.benchharness import hash_tree


def write(path, document):
    path.write_text(json.dumps(document, sort_keys=True))


def digest(value):
    return C.document_sha256({"synthetic": value})


@dataclass
class Checkpoint:
    path: Path
    context: PC.CheckpointVerificationContext
    document: dict
    iteration: dict
    experiment: dict

    @property
    def candidate(self):
        return Path(self.document["candidate_path"])

    def save(self, *, repin_iteration=False):
        if repin_iteration:
            self.document["analysis_sha256"] = C.document_sha256(self.iteration["analysis"])
            self.document["portfolio_iteration_sha256"] = C.document_sha256(self.iteration["portfolio"])
            iteration_path = Path(self.document["iteration_record"])
            write(iteration_path, self.iteration)
            self.document["iteration_record_sha256"] = C.sha256_file(iteration_path)
        write(self.path.parent / "experiment.json", self.experiment)
        write(self.path, self.document)

    def consume(self):
        return PC.consume_round_checkpoint(self.path, context=self.context)


def build_checkpoint(tmp_path, *, status="ready", shared_source_root=None, host_policy=None):
    root = tmp_path / "run"
    root.mkdir()
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("from merlin.helper import VALUE\n\ndef schedule():\n    return VALUE\n")
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    shared = shared_source_root
    if shared is None:
        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "__init__.py").write_text("")
        (shared / "helper.py").write_text("VALUE = 1\n")
    candidate_digest = hash_tree(candidate)["sha256"]
    dependencies = SI.compiler_dependency_record(candidate, shared_source_root=shared)
    identity = {"members": [{"capsule": name, "capsule_sha256": digest(name)} for name in ("first", "second")]}
    identity_digest = C.document_sha256(identity)
    analyses = []
    for index, member in enumerate(identity["members"]):
        graph = digest(f"graph-{index}")
        emission = {
            "candidate_lowered_sha256": digest(f"lowered-{index}"),
            "candidate_command_buffer_sha256": digest(f"buffer-{index}"),
        }
        plan = {
            "status": "verified",
            "plan_digest": digest(f"plan-{index}"),
            "candidate_sha256": candidate_digest,
            "logical_dispatch_digest": graph,
            **emission,
        }
        if status == "blocked" and index == 1:
            plan = {}
        analyses.append(
            {
                "candidate_sha256": candidate_digest,
                "workload": {"capsule_sha256": member["capsule_sha256"]},
                "emission": emission,
                "diagnostics": {
                    "captured_logical_graph": {"status": "verified", "logical_dispatch_digest": graph},
                    "arms": {"candidate": {"status": "emitted"}},
                    "verified_global_plan_emission": plan,
                },
            }
        )
    members = [
        {"identity": member, "readiness": EA.global_iteration_readiness(analysis)}
        for member, analysis in zip(identity["members"], analyses, strict=True)
    ]
    members[0].update(analysis_ref="/analysis", static_comparison_ref="/static_comparison")
    members[1]["analysis"] = analyses[1]
    ready_count = 2 if status == "ready" else 1
    portfolio = {
        "portfolio_sha256": identity_digest,
        "candidate_sha256": candidate_digest,
        "members": members,
        "members_total": 2,
        "members_ready": ready_count,
        "full_model_simulation_allowed": False,
    }
    readiness = EA.global_iteration_readiness(analyses[1])
    iteration = {
        "schema": "global_perf_iteration_v1",
        "candidate_sha256": candidate_digest,
        "compiler_dependencies": dependencies,
        "analysis": analyses[0],
        "readiness": readiness,
        "portfolio": portfolio,
    }
    experiment = {
        "baseline_sha256": digest("baseline"),
        "target_sha256": digest("target"),
        "portfolio": identity,
        "portfolio_sha256": identity_digest,
    }
    policy = (
        host_policy
        if host_policy is not None
        else {"schema": "synthetic_current_host_policy", "sha256": digest("policy")}
    )
    document = {
        **experiment,
        "schema": "global_perf_candidate_v1" if status == "ready" else "global_authoring_checkpoint_v1",
        "promotion_status": "blocked_authoring_checkpoint",
        "consumer": "next_bounded_global_authoring_round_only",
        "full_model_timing_status": "UNMEASURED",
        "full_model_cycles": None,
        "global_speedup_proven": False,
        "host_verification_policy": policy,
        "candidate_path": str(candidate),
        "candidate_sha256": candidate_digest,
        "candidate_read_only": True,
        "compiler_dependencies": dependencies,
        "iteration_record": str(root / "iteration.json"),
        "readiness": readiness,
        "portfolio_members_total": 2,
        "portfolio_members_ready": ready_count,
    }
    fixture = Checkpoint(
        root / "checkpoint.json",
        PC.CheckpointVerificationContext(policy, shared),
        document,
        iteration,
        copy.deepcopy(experiment),
    )
    fixture.save(repin_iteration=True)
    return fixture
