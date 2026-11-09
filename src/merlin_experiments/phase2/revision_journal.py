"""Static revision publication and retained artifacts for a model portfolio.

Admission, compilation and probe accounting remain with their callers. In particular,
``iterations`` is the live evidence view: later probes may extend those dictionaries.
The sealed static files and their original hashes never follow those mutations.
Callers serialize publication with their existing analysis lock.

Artifact accessors take already-admitted rows and explicit objective/target identities.
They establish retained correspondence, not live candidate admission. Baseline byte
checking and context formation are separate so the controller can refresh its current
artifacts at the original admission checkpoint between those operations.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes
from merlin.perf.mechanism_probe import ProbeBinding
from merlin_experiments.phase2 import broker_evidence as BE
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2.stage_inputs import StageE2ESentinel, sentinel_identity


class RevisionJournal:
    """Own chronological static records and the artifacts retained for their probes."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.iterations: list[dict[str, Any]] = []
        self.record_sha256: dict[int, str] = {}
        self.artifacts: dict[str, Any] = {}
        self.previous_artifacts: Mapping[str, Any] | None = None
        self.iteration_artifacts: dict[int, Mapping[str, Any]] = {}
        self.portfolio_artifacts: dict[str, Mapping[str, Any]] = {}
        self.previous_portfolio_artifacts: Mapping[str, Mapping[str, Any]] | None = None
        self.iteration_portfolio_artifacts: dict[int, Mapping[str, Mapping[str, Any]]] = {}
        self.baseline_artifacts: Mapping[str, Any] | None = None
        self.portfolio_baseline_artifacts: dict[str, Mapping[str, Any]] = {}
        self.optimization_baseline_sandbox: Mapping[str, Any] | None = None
        self.optimization_baseline_sandbox_sha256: str | None = None
        self.compiler_sandboxes: dict[int, Mapping[str, Any]] = {}
        self.compiler_sandbox_sha256: dict[int, str] = {}

    def _seal(self, record: Mapping[str, Any]) -> Path:
        iteration = record.get("iteration")
        if (
            not isinstance(iteration, int)
            or isinstance(iteration, bool)
            or iteration < 0
            or iteration in self.record_sha256
            or iteration != len(self.iterations)
        ):
            raise ValueError("global static iteration identity is invalid or already sealed")
        path = self.output / f"iteration_{iteration:04d}.json"
        payload = contracts.canonical_json(record)
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o444)
        self.record_sha256[iteration] = contracts.sha256_file(path)
        return path

    def _publish(
        self,
        record: dict[str, Any],
        primary: dict[str, Any],
        portfolio: dict[str, Mapping[str, Any]],
        *,
        rotate_previous: bool,
    ) -> None:
        self._seal(record)
        self.iterations.append(record)
        if rotate_previous:
            self.previous_artifacts = self.artifacts
            self.previous_portfolio_artifacts = self.portfolio_artifacts or None
        self.artifacts = primary
        self.portfolio_artifacts = portfolio
        self.iteration_artifacts[record["iteration"]] = primary
        self.iteration_portfolio_artifacts[record["iteration"]] = portfolio

    def publish_analysis(
        self,
        record: dict[str, Any],
        primary: dict[str, Any],
        portfolio: dict[str, Mapping[str, Any]],
    ) -> None:
        """Publish a fresh analysis, then detach its primary baseline artifacts."""
        self._publish(record, primary, portfolio, rotate_previous=True)
        if "baseline_artifacts" in primary:
            self.baseline_artifacts = primary.pop("baseline_artifacts")

    def publish_reuse(self, record: dict[str, Any], primary: dict[str, Any], *, source_iteration: int) -> None:
        """Publish an admitted reuse; retain primary identity and copy portfolio state."""
        portfolio = copy.deepcopy(self.iteration_portfolio_artifacts.get(source_iteration) or {})
        self._publish(record, primary, portfolio, rotate_previous=True)
        source_sandboxes = self.compiler_sandboxes.get(source_iteration)
        if source_sandboxes is not None:
            self.compiler_sandboxes[record["iteration"]] = copy.deepcopy(source_sandboxes)
            digest = self.compiler_sandbox_sha256.get(source_iteration)
            if digest is not None:
                self.compiler_sandbox_sha256[record["iteration"]] = digest

    def publish_imported_seed(
        self,
        record: dict[str, Any],
        primary: dict[str, Any],
        portfolio: dict[str, Mapping[str, Any]],
        *,
        reconstructed_sandboxes: Mapping[str, Any] | None,
        sandbox_digest: str | None,
    ) -> None:
        """Publish an admitted cross-run seed without creating previous/probe evidence."""
        if record.get("iteration") != 0 or self.iterations:
            raise ValueError("imported static seed must be the first iteration")
        if reconstructed_sandboxes is not None and (
            not isinstance(sandbox_digest, str)
            or not sandbox_digest
            or not isinstance(reconstructed_sandboxes.get("baseline"), Mapping)
        ):
            raise ValueError("reconstructed static seed requires its sandbox digest and baseline policy")
        self._publish(record, primary, portfolio, rotate_previous=False)
        baseline = primary.pop("baseline_artifacts", None)
        if baseline is not None:
            self.baseline_artifacts = baseline
        if reconstructed_sandboxes is not None:
            assert sandbox_digest is not None
            self.compiler_sandboxes[0] = reconstructed_sandboxes
            self.compiler_sandbox_sha256[0] = sandbox_digest
            self.register_optimization_baseline_sandbox(reconstructed_sandboxes["baseline"])

    def register_compiler_sandboxes(self, iteration: int, sandboxes: Mapping[str, Any]) -> None:
        """Retain completed compiler policies at the caller's pre-publication checkpoint."""
        self.compiler_sandboxes[iteration] = copy.deepcopy(sandboxes)
        self.compiler_sandbox_sha256[iteration] = contracts.document_sha256(sandboxes)

    def register_optimization_baseline_sandbox(self, sandbox: Mapping[str, Any]) -> None:
        self.optimization_baseline_sandbox = copy.deepcopy(sandbox)
        self.optimization_baseline_sandbox_sha256 = contracts.document_sha256(sandbox)

    def retain_portfolio_baseline(self, capsule_sha256: str, artifacts: dict[str, Any]) -> None:
        """Detach a member baseline at the original pre-publication observation point."""
        baseline = artifacts.pop("baseline_artifacts", None)
        if baseline is not None:
            self.portfolio_baseline_artifacts[capsule_sha256] = baseline

    def current_artifacts(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        """Host-only artifacts retained from exactly the current full-model invocation."""
        if (
            not self.artifacts
            or self.artifacts["candidate_sha256"] != row["candidate_sha256"]
            or self.artifacts["candidate_lowered_sha256"] != row["analysis"]["emission"]["candidate_lowered_sha256"]
        ):
            raise ValueError("current analysis has no retained emitted artifacts")
        return self.artifacts

    def current_portfolio_artifacts(
        self, row: Mapping[str, Any], *, sentinels: Sequence[StageE2ESentinel]
    ) -> Mapping[str, Mapping[str, Any]]:
        """Exact retained artifacts for every member of the current analyzed portfolio."""
        expected = [member.capsule_sha256 for member in sentinels]
        if list(self.portfolio_artifacts) != expected:
            raise ValueError("current analysis has no complete retained portfolio artifacts")
        for index, capsule_sha256 in enumerate(expected):
            analysis = row["analysis"] if index == 0 else row["portfolio"]["members"][index]["analysis"]
            artifacts = self.portfolio_artifacts[capsule_sha256]
            if (
                artifacts.get("candidate_sha256") != row["candidate_sha256"]
                or artifacts.get("candidate_lowered_sha256") != analysis["emission"]["candidate_lowered_sha256"]
            ):
                raise ValueError("retained portfolio artifacts changed identity")
        return self.portfolio_artifacts

    @staticmethod
    def portfolio_member_analysis(row: Mapping[str, Any], index: int) -> Mapping[str, Any]:
        if index == 0:
            return row["analysis"]
        members = (row.get("portfolio") or {}).get("members")
        if not isinstance(members, list) or index >= len(members):
            raise ValueError("portfolio iteration omits a declared member analysis")
        analysis = members[index].get("analysis")
        if not isinstance(analysis, Mapping):
            raise ValueError("portfolio secondary member analysis is malformed")
        return analysis

    def validated_portfolio_member_context(
        self,
        row: Mapping[str, Any],
        artifacts_by_capsule: Mapping[str, Mapping[str, Any]],
        *,
        index: int,
        arm: str,
        sentinels: Sequence[StageE2ESentinel],
        target_sha256: str,
    ) -> dict[str, Any]:
        """Bind one member's source, plan, emitted artifacts, compiler and target exactly."""
        if arm not in ("previous", "current") or not 0 <= index < len(sentinels):
            raise ValueError("portfolio member context arm or index is invalid")
        sentinel = sentinels[index]
        expected = [member.capsule_sha256 for member in sentinels]
        if list(artifacts_by_capsule) != expected:
            raise ValueError(f"{arm} portfolio artifact set is incomplete or reordered")
        analysis = self.portfolio_member_analysis(row, index)
        artifacts = artifacts_by_capsule.get(sentinel.capsule_sha256)
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"{arm} portfolio member has no retained artifacts")
        portfolio_member = (row.get("portfolio") or {}).get("members", [])[index]
        expected_identity = sentinel_identity(sentinel, role="primary" if index == 0 else "training")
        if (
            portfolio_member.get("identity") != expected_identity
            or analysis.get("candidate_sha256") != row.get("candidate_sha256")
            or analysis.get("workload", {}).get("capsule_sha256") != sentinel.capsule_sha256
            or EA.global_iteration_readiness(analysis).get("status") != "ready_for_probe_admission"
        ):
            raise ValueError(f"{arm} portfolio analysis identity or readiness changed")
        diagnostics = analysis.get("diagnostics") or {}
        graph = diagnostics.get("captured_logical_graph") or {}
        plan = diagnostics.get("verified_global_plan_emission") or {}
        emission = analysis.get("emission") or {}
        lowered_text = artifacts.get("lowered_text")
        command_text = artifacts.get("command_buffer_text")
        if not isinstance(lowered_text, str) or not isinstance(command_text, str):
            raise ValueError(f"{arm} portfolio emitted artifact bytes are unavailable")
        lowered_sha256 = sha256_bytes(lowered_text.encode("utf-8"))
        command_sha256 = sha256_bytes(command_text.encode("utf-8"))
        try:
            parsed_command_buffer = json.loads(command_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{arm} portfolio command buffer is malformed") from exc
        if (
            artifacts.get("candidate_sha256") != row.get("candidate_sha256")
            or artifacts.get("candidate_lowered_sha256") != lowered_sha256
            or artifacts.get("candidate_command_buffer_sha256") != command_sha256
            or artifacts.get("command_buffer") != parsed_command_buffer
            or emission.get("candidate_lowered_sha256") != lowered_sha256
            or emission.get("candidate_command_buffer_sha256") != command_sha256
            or plan.get("status") != "verified"
            or plan.get("candidate_sha256") != row.get("candidate_sha256")
            or plan.get("logical_dispatch_digest") != graph.get("logical_dispatch_digest")
            or plan.get("candidate_lowered_sha256") != lowered_sha256
            or plan.get("candidate_command_buffer_sha256") != command_sha256
        ):
            raise ValueError(f"{arm} portfolio source/plan/emitted artifact binding changed")
        interface_value = artifacts.get("interface")
        if not isinstance(interface_value, (str, Path)):
            raise ValueError(f"{arm} portfolio interface identity is unavailable")
        interface = Path(interface_value)
        source_root = Path(sentinel.frozen_source_path).resolve()
        if (
            interface.is_symlink()
            or not interface.is_file()
            or not interface.resolve().is_relative_to(source_root)
            or contracts.sha256_file(interface) != plan.get("source_sha256")
        ):
            raise ValueError(f"{arm} portfolio source does not match its verified plan")
        dependencies = row.get("compiler_dependencies") or {}
        compiler_sha256 = dependencies.get("compiler_implementation_sha256")
        if not BE._is_sha256(compiler_sha256):
            raise ValueError(f"{arm} portfolio compiler dependency identity is unavailable")
        binding = ProbeBinding(
            graph_digest=graph["logical_dispatch_digest"],
            plan_digest=plan["plan_digest"],
            compiler_digest=compiler_sha256,
            target_digest=target_sha256,
        )
        member_binding = {
            "schema": "global_portfolio_member_artifact_binding_v1",
            "arm": arm,
            "portfolio_index": index,
            "capsule": sentinel.capsule,
            "capsule_sha256": sentinel.capsule_sha256,
            "source_sha256": plan["source_sha256"],
            "candidate_sha256": row["candidate_sha256"],
            "compiler_implementation_sha256": compiler_sha256,
            "target_sha256": target_sha256,
            "logical_dispatch_digest": graph["logical_dispatch_digest"],
            "plan_digest": plan["plan_digest"],
            "lowered_sha256": lowered_sha256,
            "command_buffer_sha256": command_sha256,
        }
        return {
            "identity": expected_identity,
            "analysis": analysis,
            "artifacts": artifacts,
            "interface": interface,
            "probe_binding": binding,
            "member_binding": member_binding,
        }

    def current_portfolio_member_context(
        self,
        row: Mapping[str, Any],
        *,
        index: int,
        sentinels: Sequence[StageE2ESentinel],
        target_sha256: str,
    ) -> dict[str, Any]:
        """Strict current analysis/artifact context for one ordered portfolio member."""
        return self.validated_portfolio_member_context(
            row,
            self.portfolio_artifacts,
            index=index,
            arm="current",
            sentinels=sentinels,
            target_sha256=target_sha256,
        )

    def previous_portfolio_member_context(
        self,
        *,
        index: int,
        sentinels: Sequence[StageE2ESentinel],
        target_sha256: str,
    ) -> dict[str, Any]:
        """Strict immediately preceding analysis/artifact context for one portfolio member."""
        if len(self.iterations) < 2 or self.previous_portfolio_artifacts is None:
            raise ValueError("changed-region qualification requires prior portfolio artifacts")
        previous = self.iterations[-2]
        if previous.get("readiness", {}).get("status") != "ready_for_probe_admission":
            raise ValueError("changed-region qualification requires a ready prior portfolio")
        return self.validated_portfolio_member_context(
            previous,
            self.previous_portfolio_artifacts,
            index=index,
            arm="previous",
            sentinels=sentinels,
            target_sha256=target_sha256,
        )

    @staticmethod
    def probe_binding(row: Mapping[str, Any], *, target_sha256: str) -> ProbeBinding:
        diag = row["analysis"]["diagnostics"]
        return ProbeBinding(
            graph_digest=diag["captured_logical_graph"]["logical_dispatch_digest"],
            plan_digest=diag["verified_global_plan_emission"]["plan_digest"],
            compiler_digest=row["compiler_dependencies"]["compiler_implementation_sha256"],
            target_digest=target_sha256,
        )

    def retained_previous_artifacts(self) -> Mapping[str, Any]:
        """Host-only preceding submitted artifact; never infer a previous revision from input IR."""
        if len(self.iterations) < 2 or not self.previous_artifacts:
            raise ValueError("changed-region qualification requires two submitted full-model revisions")
        previous = self.iterations[-2]
        if (
            self.previous_artifacts.get("candidate_sha256") != previous["candidate_sha256"]
            or self.previous_artifacts.get("candidate_lowered_sha256")
            != previous["analysis"]["emission"]["candidate_lowered_sha256"]
        ):
            raise ValueError("previous full-model artifact identity does not match its analysis")
        return self.previous_artifacts

    def previous_probe_binding(self, *, target_sha256: str) -> ProbeBinding:
        """Bind the actual preceding verified submitted revision after artifact admission."""
        previous = self.iterations[-2]
        if previous["readiness"]["status"] != "ready_for_probe_admission":
            raise ValueError("paired diagnostic requires a verified preceding global plan")
        return self.probe_binding(previous, target_sha256=target_sha256)

    def baseline_artifact_bytes(
        self,
        row: Mapping[str, Any],
        *,
        baseline_sha256: str,
        sentinel: StageE2ESentinel,
        target: str,
    ) -> Mapping[str, Any]:
        """Exact optimization-comparison bytes, not a preceding candidate or Phase-1 verdict.

        The baseline cache does not carry a verified global-plan receipt. This accessor
        verifies artifact identity only; a mechanism qualifier must independently check
        source/plan/implementation correspondence before accepting semantic evidence.
        """
        artifacts = self.baseline_artifacts
        expected = {
            "baseline_sha256": baseline_sha256,
            "capsule_sha256": sentinel.capsule_sha256,
            "target": target,
        }
        if not isinstance(artifacts, Mapping) or artifacts.get("identity") != expected:
            raise ValueError("no exact retained optimization-baseline artifacts")
        emission = row["analysis"]["emission"]
        for field, recorded, analyzed in (
            ("lowered_text", "lowered_sha256", "baseline_lowered_sha256"),
            ("command_buffer_text", "command_buffer_sha256", "baseline_command_buffer_sha256"),
        ):
            text = artifacts.get(field)
            if (
                not isinstance(text, str)
                or not BE._is_sha256(artifacts.get(recorded))
                or sha256_bytes(text.encode()) != artifacts[recorded]
                or artifacts[recorded] != emission.get(analyzed)
            ):
                raise ValueError("retained optimization-baseline artifact bytes changed")
        return artifacts

    def optimization_baseline_artifacts(
        self,
        row: Mapping[str, Any],
        artifacts: Mapping[str, Any],
        current_artifacts: Mapping[str, Any],
        *,
        baseline_sha256: str,
        sentinel: StageE2ESentinel,
    ) -> Mapping[str, Any]:
        """Form identity-only baseline context from checked bytes and admitted current artifacts."""
        interface = Path(current_artifacts["interface"])
        source_root = Path(sentinel.frozen_source_path).resolve(strict=True)
        if (
            interface.is_symlink()
            or not interface.is_file()
            or not interface.resolve(strict=True).is_relative_to(source_root)
        ):
            raise ValueError("optimization-baseline interface is outside the frozen objective")
        source_sha = contracts.sha256_file(interface)
        graph = row["analysis"]["diagnostics"]["captured_logical_graph"]
        if graph.get("source_sha256") != source_sha:
            raise ValueError("optimization-baseline source is not the analyzed full objective")
        buffer = json.loads(artifacts["command_buffer_text"])
        if not isinstance(buffer, dict) or buffer.get("declined") is not None:
            raise ValueError("optimization baseline did not emit a usable command buffer")
        # Decode from checked bytes, rather than exporting mutable/unbound cached trace objects.
        return {
            "arm": "optimization_baseline",
            "interface": interface,
            "source_sha256": source_sha,
            "lowered_text": artifacts["lowered_text"],
            "command_buffer_text": artifacts["command_buffer_text"],
            "command_buffer": buffer,
            "compiler_sha256": baseline_sha256,
            "lowered_sha256": artifacts["lowered_sha256"],
            "command_buffer_sha256": artifacts["command_buffer_sha256"],
            "structural_plan_status": "UNVERIFIED",
            "numerical_qualification": "UNPROVEN",
        }

    @staticmethod
    def optimization_baseline_artifact_binding(
        row: Mapping[str, Any],
        artifact: Mapping[str, Any],
        *,
        baseline_sha256: str,
        baseline_dependencies: Mapping[str, Any],
        baseline_binding_sha256: str,
        sentinel: StageE2ESentinel,
        target_sha256: str,
    ) -> Mapping[str, Any]:
        """Identity-only comparison binding; deliberately not a verified ``ProbeBinding``."""
        return {
            "schema": "optimization_baseline_artifact_binding_v1",
            "arm": "optimization_baseline",
            "compiler_sha256": baseline_sha256,
            "compiler_dependencies": copy.deepcopy(baseline_dependencies),
            "optimization_baseline_binding_sha256": baseline_binding_sha256,
            "capsule_sha256": sentinel.capsule_sha256,
            "source_sha256": artifact["source_sha256"],
            "logical_dispatch_digest": row["analysis"]["diagnostics"]["captured_logical_graph"][
                "logical_dispatch_digest"
            ],
            "target_sha256": target_sha256,
            "lowered_sha256": artifact["lowered_sha256"],
            "command_buffer_sha256": artifact["command_buffer_sha256"],
            "structural_plan_status": "UNVERIFIED",
            "numerical_qualification": "UNPROVEN",
            "phase1_qualification_extended": False,
        }
