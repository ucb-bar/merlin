"""Admitted short-probe execution and scoped portfolio evidence publication.

Compose concrete analysis and revision authority without a native controller.
Provider callbacks receive this probe owner, not a bag of controller callbacks.
Probe evidence remains scoped and never establishes full-model timing.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin.perf.agent_guidance import inspect_compiler_package
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS
from merlin.perf.mechanism_probe import ProbeBinding, ProbeObservation, require_probe_admission
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2.portfolio_analysis import PortfolioAnalysis
from merlin_experiments.phase2.portfolio_checkpoint import (
    paired_context_decision_feedback,
    verify_changed_region_semantic_receipt,
)


def compiler_dependency_mounts(argv: Sequence[str], dependencies: Mapping[str, Any], overlay_root: Path) -> list[str]:
    """Merge new exact helpers into already granted directories without widening those grants.

    bwrap cannot create a new file mountpoint beneath a read-only directory mount. Only that
    existing granted directory is copied, extended with verified helper bytes, and re-bound RO;
    the caller reapplies every original answer mask afterward.
    """
    result = list(argv)
    root = Path(dependencies["shared_source_root"])
    directory_mounts = [
        (index, Path(result[index + 1]), Path(result[index + 2]))
        for index, option in enumerate(result[:-2])
        if option == "--ro-bind" and Path(result[index + 1]).is_dir()
    ]
    additions: dict[int, list[tuple[Path, Path]]] = {}
    direct: list[Path] = []
    for relative, digest in dependencies["shared_sources"].items():
        source = root / relative
        if source.is_symlink() or P2_CONTRACTS.sha256_file(source) != digest:
            raise ValueError("shared compiler dependency changed before sandbox binding")
        containing = [
            (index, original, destination)
            for index, original, destination in directory_mounts
            if destination in source.parents
        ]
        if containing:
            index, original, destination = max(containing, key=lambda row: (len(row[2].parts), row[0]))
            inside = source.relative_to(destination)
            if not (original / inside).exists():
                additions.setdefault(index, []).append((source, inside))
                continue
        direct.append(source)
    for index, sources in additions.items():
        overlay = overlay_root / str(index)
        shutil.copytree(result[index + 1], overlay, symlinks=True)
        AW._make_writable(overlay)
        for source, inside in sources:
            destination = overlay / inside
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if P2_CONTRACTS.sha256_file(source) != P2_CONTRACTS.sha256_file(destination):
                raise ValueError("compiler helper changed while creating its read-only overlay")
        for path in overlay.rglob("*"):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        overlay.chmod(0o555)
        result[index + 1] = str(overlay)
    for source in direct:
        result.extend(("--ro-bind", str(source), str(source)))
    return result


class PortfolioProbes:
    """Own short-probe policy reuse, execution boundaries and scoped receipts."""

    def __init__(self, analysis: PortfolioAnalysis) -> None:
        self.analysis = analysis

    def compile_probe_candidate(
        self, candidate: Path, interface: Path, scratch: Path, *, timeout_s: float, emit_command_buffer: bool = False
    ):
        """Compile a host-selected separate probe under the same answer-masked compiler policy."""
        return self._compile_probe_revision(
            candidate, interface, scratch, timeout_s=timeout_s, emit_command_buffer=emit_command_buffer, previous=False
        )

    def compile_previous_probe_candidate(
        self, candidate: Path, interface: Path, scratch: Path, *, timeout_s: float, emit_command_buffer: bool = False
    ):
        """Compile the same witness using the exact preceding analyzed compiler and masks."""
        return self._compile_probe_revision(
            candidate, interface, scratch, timeout_s=timeout_s, emit_command_buffer=emit_command_buffer, previous=True
        )

    def compile_optimization_baseline_probe_candidate(
        self, candidate: Path, interface: Path, scratch: Path, *, timeout_s: float, emit_command_buffer: bool = False
    ):
        """Compile a separate host-selected probe with the actual optimization-comparison arm."""
        return self._compile_probe_revision(
            candidate,
            interface,
            scratch,
            timeout_s=timeout_s,
            emit_command_buffer=emit_command_buffer,
            previous=False,
            optimization_baseline=True,
        )

    def _compile_probe_revision(
        self,
        candidate: Path,
        interface: Path,
        scratch: Path,
        *,
        timeout_s: float,
        emit_command_buffer: bool,
        previous: bool,
        optimization_baseline: bool = False,
    ):
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker, _read_output, run_sandboxed_entrypoint
        from merlin.targetgen import oot_runner

        started = time.monotonic()
        with self.analysis.session.action(candidate) as (_row, owns_action):
            if not isinstance(self.analysis.analyzer, IsolatedAnalysisWorker):
                raise ValueError("short candidate compilation requires the production isolated compiler policy")
            if optimization_baseline:
                if not math.isfinite(timeout_s) or not 0 < timeout_s <= ITERATION_MAX_SECONDS:
                    raise ValueError("optimization-baseline probe needs a finite bounded timeout")
                timeout_s = min(
                    timeout_s, self.analysis.timeout_s - self.analysis.session.current(candidate)["elapsed_seconds"]
                )
                if timeout_s <= 0:
                    raise TimeoutError("optimization-baseline probe exhausted its iteration budget")
            if previous and optimization_baseline:
                raise ValueError("preceding candidate and optimization baseline are distinct arms")
            binding = (
                self.analysis.session.optimization_baseline_artifact_binding(candidate)
                if optimization_baseline
                else self.analysis.session.previous_probe_binding(candidate)
                if previous
                else self.analysis.session.current_probe_binding(candidate)
            )
            selected = (
                self.analysis.session.journal.iterations[-2]
                if previous
                else self.analysis.session.journal.iterations[-1]
            )

            def check_revision(*, revalidate: bool, full: bool = False) -> None:
                if full:
                    self.analysis.session.refresh(candidate)
                elif revalidate:
                    self.analysis.session.revalidate(candidate)
                else:
                    self.analysis.session.current(candidate)
                if optimization_baseline:
                    if self.analysis.session.optimization_baseline_artifact_binding(candidate) != binding:
                        raise ValueError("optimization-baseline witness binding changed")
                    return
                actual = (
                    self.analysis.session.previous_probe_binding(candidate)
                    if previous
                    else self.analysis.session.current_probe_binding(candidate)
                )
                submitted = Path(selected["submitted_snapshot"])
                if (
                    actual != binding
                    or hash_tree(submitted)["sha256"] != selected["candidate_sha256"]
                    or self.analysis.session.inputs.compiler_dependencies(submitted)
                    != selected["compiler_dependencies"]
                ):
                    raise ValueError("selected short-witness compiler identity changed")

            # The outermost action was freshly verified on entry. A provider-owned parent action
            # may have done arbitrary host work since its last checkpoint, so refresh before use.
            check_revision(revalidate=not owns_action)
            scratch.mkdir()
            source = scratch / "interface.mlir"
            source.write_bytes(interface.read_bytes())
            sandbox = (
                self._probe_sandbox(candidate, scratch, optimization_baseline=True)
                if optimization_baseline
                else self._probe_sandbox(candidate, scratch, previous=previous)
            )
            package = oot_runner.load_package(Path(sandbox["package_path"]))
            compiler_invoked = False
            try:
                compiler_invoked = True
                lowered = run_sandboxed_entrypoint(
                    package,
                    "lower_target_to_llvm",
                    source,
                    sandbox=sandbox,
                    timeout_s=timeout_s - (time.monotonic() - started),
                )
                if not emit_command_buffer:
                    return lowered
                buffer_result = None
                command_buffer = None
                if lowered.returncode == 0:
                    buffer_path = scratch / "command_buffer.json"
                    buffer_result = run_sandboxed_entrypoint(
                        package,
                        "emit_command_buffer",
                        source,
                        buffer_path,
                        sandbox=sandbox,
                        timeout_s=timeout_s - (time.monotonic() - started),
                    )
                    if buffer_result.returncode == 0:
                        command_buffer = json.loads(_read_output(buffer_path))
                        if not isinstance(command_buffer, dict):
                            raise ValueError("probe command buffer must be a JSON object")
                return {"lowered": lowered, "command_buffer_emission": buffer_result, "command_buffer": command_buffer}
            finally:
                if compiler_invoked:
                    # A containing semantic action owns the expensive immutable-input audit and
                    # repeats it once before accepting evidence. Always rehash the candidate,
                    # selected submitted snapshot and compiler dependencies at this boundary.
                    check_revision(revalidate=True, full=owns_action)

    def _probe_sandbox(
        self, candidate: Path, scratch: Path, *, previous: bool = False, optimization_baseline: bool = False
    ) -> Mapping[str, Any]:
        """Reuse the successful analysis policy for its immutable compiler, plus public scratch."""
        row = self.analysis.session.current(candidate)
        if previous and optimization_baseline:
            raise ValueError("preceding candidate and optimization baseline are distinct arms")
        if optimization_baseline:
            self.analysis.session.optimization_baseline_artifact_binding(candidate)
            sandbox = self.analysis.session.journal.optimization_baseline_sandbox
            if (
                not isinstance(sandbox, Mapping)
                or self.analysis.session.journal.optimization_baseline_sandbox_sha256 is None
                or P2_CONTRACTS.document_sha256(sandbox)
                != self.analysis.session.journal.optimization_baseline_sandbox_sha256
            ):
                raise ValueError("optimization baseline has no intact recorded answer-masked policy")
            expected_package = self.analysis.session.inputs.optimization_baseline
            expected_dependencies = self.analysis.session.inputs.optimization_baseline_binding["compiler_dependencies"]
        elif previous:
            self.analysis.session.previous_probe_binding(candidate)
            row = self.analysis.session.journal.iterations[-2]
            completed = self.analysis.session.journal.compiler_sandboxes.get(row["iteration"])
            if completed is None:
                raise ValueError("preceding compiler has no retained successful answer-masked policy")
        else:
            # A reverted candidate may be backed by an older immutable analysis snapshot,
            # while the worker's mutable ``completed_sandboxes`` points at the rejected edit.
            completed = self.analysis.session.journal.compiler_sandboxes.get(row["iteration"]) or getattr(
                self.analysis.analyzer, "completed_sandboxes", None
            )
        if not optimization_baseline:
            if completed is None:
                return self.analysis.analyzer.sandbox_factory(
                    self.analysis.session.inputs.optimization_baseline, candidate, scratch
                )["candidate"]
            retained_sha256 = self.analysis.session.journal.compiler_sandbox_sha256.get(row["iteration"])
            if retained_sha256 is not None and P2_CONTRACTS.document_sha256(completed) != retained_sha256:
                raise ValueError("retained compiler sandbox policy changed identity")
            sandbox = completed["candidate"]
            expected_package = Path(row["submitted_snapshot"])
            expected_dependencies = row["compiler_dependencies"]
        if (
            Path(sandbox["package_path"]).resolve() != expected_package.resolve()
            or sandbox.get("compiler_dependencies") != expected_dependencies
        ):
            raise ValueError("prepared sandbox is not bound to the current immutable compiler")
        for overlay_path, overlay_sha256 in (sandbox.get("overlay_trees") or {}).items():
            overlay = Path(overlay_path)
            if (
                overlay.is_symlink()
                or not overlay.is_dir()
                or P2_CONTRACTS.exact_tree_record(overlay)["sha256"] != overlay_sha256
            ):
                raise ValueError("prepared sandbox dependency overlay changed identity")
        directory = scratch.resolve(strict=True)
        if scratch.is_symlink() or not directory.is_dir():
            raise ValueError("probe scratch must be a real dedicated directory")
        for surface in sandbox["answer_surfaces"]:
            answer = Path(surface["path"]).resolve()
            if (
                directory == answer
                or directory in answer.parents
                or surface["kind"] == "dir"
                and answer in directory.parents
            ):
                raise ValueError("new probe scratch would expose a masked answer surface")
        prefix = list(sandbox["command_prefix"])
        boundary = sandbox["bwrap_argv_length"]
        if not isinstance(boundary, int) or not 0 < boundary < len(prefix):
            raise ValueError("prepared policy has no exact bwrap/payload boundary")
        return {
            **sandbox,
            "command_prefix": [*prefix[:boundary], "--bind", str(directory), str(directory), *prefix[boundary:]],
            "reuse_scope": "exact analyzed compiler and existing answer masks; public scratch only",
        }

    def native_probe_policy(self, candidate: Path, scratch: Path) -> dict[str, Any]:
        """Host-provider-only copy of the exact existing policy; this grants nothing new."""
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker

        row = self.analysis.session.current(candidate)
        if not isinstance(self.analysis.analyzer, IsolatedAnalysisWorker):
            raise ValueError("native probes require the production isolated compiler policy")
        scratch = scratch.resolve(strict=True)
        if not scratch.is_dir():
            raise ValueError("native probe scratch must be an existing dedicated directory")
        cache = getattr(self, "_native_probe_policies", None)
        if cache is None:
            self._native_probe_policies = cache = {}
        key = (row["compiler_dependencies"]["compiler_implementation_sha256"], str(candidate.resolve()), str(scratch))
        if key not in cache:
            cache[key] = self._probe_sandbox(candidate, scratch)
        return copy.deepcopy(cache[key])

    def run_native_probe(
        self,
        candidate: Path,
        scratch: Path,
        argv: Sequence[str],
        *,
        timeout_s: float,
        _build_dependencies=None,
        _execution_dependencies=None,
    ):
        """Run a host-built changed-region witness inside the existing answer-free policy.

        The host provider selects the runner and owns its reference comparison. Native candidate
        instructions never execute in the host interpreter or inherit access to host answers.
        """
        with self.analysis.session.action(candidate) as (_row, owns_action):
            if not owns_action:
                self.analysis.session.revalidate(candidate)
            try:
                return self._run_native_probe_after_action_check(
                    candidate,
                    scratch,
                    argv,
                    timeout_s=timeout_s,
                    _build_dependencies=_build_dependencies,
                    _execution_dependencies=_execution_dependencies,
                )
            finally:
                if owns_action:
                    self.analysis.session.refresh(candidate)
                else:
                    self.analysis.session.revalidate(candidate)

    def _run_native_probe_after_action_check(
        self,
        candidate: Path,
        scratch: Path,
        argv: Sequence[str],
        *,
        timeout_s: float,
        _build_dependencies=None,
        _execution_dependencies=None,
    ):
        import math
        import subprocess

        from merlin.perf.analysis_worker import IsolatedAnalysisWorker, _kill_group

        started = time.monotonic()
        row = self.analysis.session.current(candidate)
        if not isinstance(self.analysis.analyzer, IsolatedAnalysisWorker):
            raise ValueError("native probes require the production isolated compiler policy")
        if not math.isfinite(timeout_s) or timeout_s <= 0 or not argv:
            raise ValueError("native probes require a positive finite deadline and an argv")
        if _execution_dependencies is not None and (_build_dependencies is not None or timeout_s > 60):
            raise ValueError("raw-engine runtime requires a separate action bounded to 60 seconds")
        scratch = scratch.resolve(strict=True)
        if not scratch.is_dir():
            raise ValueError("native probe scratch must be an existing dedicated directory")
        sandbox = self.native_probe_policy(candidate, scratch)
        import contextlib
        import tempfile

        with contextlib.ExitStack() as scope:
            prefix = list(sandbox["command_prefix"])
            if _build_dependencies is not None:
                from merlin.targetgen.sandbox.build_dependencies import HostBuildDependencies

                if type(_build_dependencies) is not HostBuildDependencies:
                    raise ValueError("build grants require an exact host-owned dependency capability")
                # This fresh source overlay is private to the host and never
                # cached or served writable to a candidate/native invocation.
                overlay = Path(scope.enter_context(tempfile.TemporaryDirectory(prefix="merlin_build_sources_")))
                prefix = _build_dependencies.extend(
                    sandbox, argv, overlay_root=overlay, overlay_builder=compiler_dependency_mounts
                )
            if _execution_dependencies is not None:
                from merlin.targetgen.sandbox.executable_dependencies import HostExecutableDependencies

                if type(_execution_dependencies) is not HostExecutableDependencies:
                    raise ValueError("runtime grants require an exact trusted executable capability")
                prefix = _execution_dependencies.extend(sandbox, argv)
            if not prefix or Path(prefix[0]).name != "bwrap" or "--clearenv" not in prefix:
                raise ValueError("native probe requires the existing clear-environment bwrap policy")
            remaining = min(timeout_s, self.analysis.timeout_s - row["elapsed_seconds"]) - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("native probe iteration wall-clock budget exhausted")
            command = [*prefix, *map(str, argv)]
            process = subprocess.Popen(
                command,
                cwd=str(scratch),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=remaining)
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            finally:
                _kill_group(process.pid)
                process.wait()
                if _build_dependencies is not None:
                    _build_dependencies.revalidate(argv)
                if _execution_dependencies is not None:
                    _execution_dependencies.revalidate(argv)
        return result

    def charge_probe_preparation(self, candidate: Path, elapsed_seconds: float) -> None:
        import math

        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("probe preparation time must be finite and nonnegative")
        row = self.analysis.session.journal.iterations[-1]
        row["elapsed_seconds"] += elapsed_seconds
        receipt = {
            "schema": "global_probe_preparation_v1",
            "iteration": row["iteration"],
            "candidate_sha256": row["candidate_sha256"],
            "elapsed_seconds": elapsed_seconds,
        }
        self._write(f"probe_preparation_{time.time_ns()}.json", receipt)
        self.analysis.session.current(candidate)

    def prepare_source_convolution(self, candidate: Path, *, comparison_arm: str, timeout_s: float) -> dict[str, Any]:
        """Broker preparation feedback only: never a semantic pass or simulator admission."""
        import math

        from merlin.perf.source_convolution_preparation import prepare_source_convolution

        if comparison_arm not in {"optimization_baseline", "previous"}:
            raise ValueError("comparison_arm must explicitly select optimization_baseline or previous")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("source-convolution preparation needs a finite positive budget")
        started = time.monotonic()
        row = self.analysis.session.current(candidate)
        binding = self.analysis.session.current_probe_binding(candidate)
        budget = min(60.0, float(timeout_s), self.analysis.timeout_s - row["elapsed_seconds"])
        try:
            descriptor = P2_CONTRACTS.mapping_file(
                Path(self.analysis.session.inputs.sentinel.frozen_source_path) / "capsule.yaml", yaml_file=True
            )
            entry = descriptor.get("entry")
            if not isinstance(entry, str) or not entry:
                raise ValueError("frozen objective descriptor has no explicit source entry")
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("source-convolution preparation exhausted remaining iteration budget")
            evidence = prepare_source_convolution(
                candidate=candidate,
                probes=self,
                comparison_arm=comparison_arm,
                entry=entry,
                output=self.analysis.session.journal.output / "source_convolution_preparations",
                timeout_s=remaining,
            )
            if self.analysis.session.current_probe_binding(candidate) != binding:
                raise ValueError("compiler/source/target changed during source-convolution preparation")
            allowed = []
            if self.analysis.session.edit_authority.contract is not None:
                inventory = (
                    self.analysis.session.edit_authority.guidance_inventory.to_dict()
                    if self.analysis.session.edit_authority.guidance_inventory is not None
                    else inspect_compiler_package(
                        self.analysis.session.edit_authority.seed,
                        contract=self.analysis.session.inputs.contract_root,
                    ).to_dict()
                )
                surfaces = {row["id"]: row for row in inventory.get("surfaces", [])}
                effects = {"placement", "movement", "layout", "encoding", "issue", "tiling", "dtype", "quantization"}
                for owner in self.analysis.session.edit_authority.contract["existing_symbols"]:
                    surface = surfaces.get(owner["surface_id"], {})
                    if (
                        surface.get("path") == owner["path"]
                        and surface.get("symbol") == owner["symbol"]
                        and effects.intersection(surface.get("effects", []))
                    ):
                        allowed.append(
                            {
                                **owner,
                                "effects": surface["effects"],
                                "matching_effects": sorted(effects.intersection(surface["effects"])),
                            }
                        )
            elapsed = time.monotonic() - started
            if elapsed > budget:
                raise TimeoutError("source-convolution preparation exceeded shared action budget")
            receipt = {
                "schema": "global_source_convolution_preparation_receipt_v1",
                "iteration": row["iteration"],
                "binding": binding.to_dict(),
                "comparison_arm": comparison_arm,
                "host_verifier_policy_sha256": self.analysis.session.inputs.host_policy["sha256"],
                "status": "runtime_pending" if evidence.get("status") == "prepared" else "UNKNOWN",
                "evidence": evidence,
                "elapsed_seconds": elapsed,
                "numerical_pass": False,
                "runtime_admitted": False,
                "runtime_recipe_binding": "NOT_PREPARED",
                "full_model_numerics_qualified": False,
                "global_speedup_proven": False,
                "allowed_edit_surfaces": allowed,
                "edit_authority_sha256": (
                    self.analysis.session.edit_authority.binding["contract_document_sha256"]
                    if self.analysis.session.edit_authority.contract is not None
                    else None
                ),
                "surface_scope": (
                    "effect-matched subset of host-frozen AST authority; not proof of compiler reachability"
                ),
                "remaining_obligations": [
                    "verify emitted target instruction mechanism",
                    "bind complete-program runtime recipe/tools/ELF",
                    "admit both complete reduced programs within remaining 60-second runtime action",
                    "warm1/measured1 independent output checks",
                ],
                "next_action": (
                    "Inspect prepared source/route or missing proof; preparation is not qualify-changed-region success"
                ),
            }
            path = self._write(f"source_convolution_preparation_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            row.setdefault("preparation_receipts", []).append(
                {"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)}
            )
            return receipt
        finally:
            self.charge_probe_preparation(candidate, time.monotonic() - started)

    def prepare_source_contraction(
        self,
        candidate: Path,
        *,
        comparison_arm: str,
        source_op_index: int,
        max_m: int,
        max_n: int,
        max_k: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Prepare a selected current-source pair, without granting runtime or semantic authority."""
        import math

        from merlin.perf.source_contraction_preparation import prepare_source_contraction

        if (
            comparison_arm not in {"optimization_baseline", "previous"}
            or type(source_op_index) is not int
            or source_op_index < 0
            or any(type(value) is not int or not 1 <= value <= 4096 for value in (max_m, max_n, max_k))
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("invalid explicit source-contraction selection or budget")
        started = time.monotonic()
        row = self.analysis.session.current(candidate)
        binding = self.analysis.session.current_probe_binding(candidate)
        budget = min(60.0, float(timeout_s), self.analysis.timeout_s - row["elapsed_seconds"])
        try:
            descriptor = P2_CONTRACTS.mapping_file(
                Path(self.analysis.session.inputs.sentinel.frozen_source_path) / "capsule.yaml", yaml_file=True
            )
            entry = descriptor.get("entry")
            if entry is None:
                # Older frozen descriptors omit entry. Select only a UNIQUE
                # function from their already hash-bound source, never a guessed name.
                from merlin.frontends.linalg_mlir import parse_mlir_text

                artifact = self.analysis.session.current_artifacts(candidate)
                source = Path(artifact["interface"]).read_text()
                source_sha = row["analysis"]["diagnostics"]["captured_logical_graph"].get("source_sha256")
                if len(source.encode()) > 2_000_000 or sha256_bytes(source.encode()) != source_sha:
                    raise ValueError("entry inference requires exact bounded frozen source bytes")
                module = parse_mlir_text(source)
                functions = [op for op in module.body.block.ops if op.name == "func.func"]
                if len(functions) != 1:
                    raise ValueError("ambiguous frozen source requires an explicit host entry")
                entry = functions[0].sym_name.data
            if not isinstance(entry, str) or not entry:
                raise ValueError("frozen objective has no valid source entry")
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("source contraction preparation exhausted its action budget")
            evidence = prepare_source_contraction(
                candidate=candidate,
                probes=self,
                comparison_arm=comparison_arm,
                source_op_index=source_op_index,
                entry=entry,
                max_m=max_m,
                max_n=max_n,
                max_k=max_k,
                output=self.analysis.session.journal.output / "source_contraction_preparations",
                timeout_s=remaining,
            )
            if self.analysis.session.current_probe_binding(candidate) != binding or time.monotonic() - started > budget:
                raise ValueError("preparation binding changed or shared deadline expired")
            receipt = {
                "schema": "global_source_contraction_preparation_receipt_v1",
                "iteration": row["iteration"],
                "binding": binding.to_dict(),
                "comparison_arm": comparison_arm,
                "host_verifier_policy_sha256": self.analysis.session.inputs.host_policy["sha256"],
                "evidence": evidence,
                "status": "runtime_pending" if evidence.get("status") == "prepared" else "UNKNOWN",
                "elapsed_seconds": time.monotonic() - started,
                "numerical_pass": False,
                "runtime_admitted": False,
                "full_model_numerics_qualified": False,
                "global_speedup_proven": False,
                "emitted_route_correspondence": evidence.get("emitted_route_correspondence", "UNKNOWN"),
                "task_route_feedback": copy.deepcopy(
                    evidence.get("task_route_feedback")
                    or {
                        "status": "UNKNOWN",
                        "scope": "no bound full/short task route comparison available",
                        "timing_calibration_admissible": False,
                    }
                ),
                "next_action": (
                    "qualify-source-contraction with this exact preparation_sha256, if host provider available"
                ),
            }
            path = self._write(f"source_contraction_preparation_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            reference = {"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)}
            row.setdefault("source_contraction_preparation_receipts", []).append(reference)
            return {**receipt, "preparation_sha256": reference["sha256"]}
        finally:
            self.charge_probe_preparation(candidate, time.monotonic() - started)

    def qualify_source_contraction(
        self, candidate: Path, *, preparation_sha256: str, provider: Callable[..., Mapping[str, Any]], timeout_s: float
    ) -> dict[str, Any]:
        """Run only a current hash-bound host preparation; never accept an agent-selected path."""
        import math

        if (
            not isinstance(preparation_sha256, str)
            or len(preparation_sha256) != 64
            or any(c not in "0123456789abcdef" for c in preparation_sha256)
            or isinstance(timeout_s, bool)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("invalid source-pair preparation identity or budget")
        started = time.monotonic()
        row = self.analysis.session.current(candidate)
        binding = self.analysis.session.current_probe_binding(candidate)
        budget = min(60.0, float(timeout_s), self.analysis.timeout_s - row["elapsed_seconds"])
        try:
            references = [
                ref
                for ref in row.get("source_contraction_preparation_receipts", [])
                if ref["sha256"] == preparation_sha256
            ]
            if len(references) != 1:
                raise ValueError("preparation is not a unique current-iteration host receipt")
            path = Path(references[0]["path"])
            if path.is_symlink() or P2_CONTRACTS.sha256_file(path) != preparation_sha256:
                raise ValueError("prepared source-pair receipt changed")
            preparation = json.loads(path.read_text())
            if (
                preparation.get("schema") != "global_source_contraction_preparation_receipt_v1"
                or preparation.get("binding") != binding.to_dict()
                or preparation.get("host_verifier_policy_sha256") != self.analysis.session.inputs.host_policy["sha256"]
                or preparation.get("status") != "runtime_pending"
            ):
                raise ValueError("prepared source pair is stale, unavailable, or from another host policy")
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("source-pair action exhausted its iteration budget")
            evidence = dict(
                provider(candidate=candidate, probes=self, prepared=preparation["evidence"], timeout_s=remaining)
            )
            if (
                self.analysis.session.current_probe_binding(candidate) != binding
                or P2_CONTRACTS.sha256_file(path) != preparation_sha256
                or time.monotonic() - started > budget
            ):
                raise ValueError("source-pair binding changed or shared deadline expired")
            receipt = {
                "schema": "global_source_contraction_execution_receipt_v1",
                "iteration": row["iteration"],
                "binding": binding.to_dict(),
                "preparation_sha256": preparation_sha256,
                "host_verifier_policy_sha256": self.analysis.session.inputs.host_policy["sha256"],
                "evidence": evidence,
                "elapsed_seconds": time.monotonic() - started,
                "full_model_numerics_qualified": False,
                "global_speedup_proven": False,
                "task_route_feedback": copy.deepcopy(
                    evidence.get("task_route_feedback")
                    or {
                        "status": "UNKNOWN",
                        "scope": "no bound full/short task route comparison available",
                        "timing_calibration_admissible": False,
                    }
                ),
                "scope": "complete reduced source pair only; changed full-model task relevance requires separate proof",
            }
            result_path = self._write(
                f"source_contraction_execution_{row['iteration']:04d}_{time.time_ns()}.json", receipt
            )
            row.setdefault("source_pair_receipts", []).append(
                {"path": str(result_path), "sha256": P2_CONTRACTS.sha256_file(result_path)}
            )
            return receipt
        finally:
            self.charge_probe_preparation(candidate, time.monotonic() - started)

    def qualify_changed_region(
        self, candidate: Path, *, provider: Callable[..., Mapping[str, Any]], timeout_s: float
    ) -> dict[str, Any]:
        """Record a host-selected semantic mechanism witness without upgrading model timing."""
        with self.analysis.session.action(candidate):
            row = self.analysis.session.current(candidate)
            selected = self.analysis.session.selected_changed_portfolio_context(candidate)
            selection = selected["selection"]
            previous = selected["previous"]
            current = selected["current"]
            binding = current["probe_binding"]
            member_binding = {
                "selection": selection,
                "previous": previous["member_binding"],
                "current": current["member_binding"],
            }
            previous_record = (
                self.analysis.session.journal.output
                / f"iteration_{self.analysis.session.journal.iterations[-2]['iteration']:04d}.json"
            )
            previous_pointer = {
                "previous_iteration_record": str(previous_record.absolute()),
                "previous_iteration_record_sha256": P2_CONTRACTS.sha256_file(previous_record),
                "previous_portfolio_iteration_sha256": P2_CONTRACTS.document_sha256(
                    self.analysis.session.journal.iterations[-2]["portfolio"]
                ),
            }
            started = time.monotonic()
            remaining = min(float(timeout_s), self.analysis.timeout_s - row["elapsed_seconds"])
            if remaining <= 0:
                raise TimeoutError("changed-region qualification exhausted its iteration budget")
            final_integrity_checked = False
            try:
                evidence = dict(
                    provider(candidate=candidate, probes=self, timeout_s=remaining, portfolio_member=selection)
                )
                # The provider may execute compilers/runtimes or otherwise yield control. Recheck
                # the complete immutable experiment and live candidate before consuming evidence.
                self.analysis.session.refresh(candidate)
                final_integrity_checked = True
                selected_after = self.analysis.session.selected_changed_portfolio_context(candidate, selection)
                actual_member_binding = {
                    "selection": selected_after["selection"],
                    "previous": selected_after["previous"]["member_binding"],
                    "current": selected_after["current"]["member_binding"],
                }
                if (
                    actual_member_binding != member_binding
                    or selected_after["current"]["probe_binding"] != binding
                    or evidence.get("portfolio_member_binding") != member_binding
                ):
                    raise ValueError(
                        "compiler, portfolio member, source, plan, or artifacts changed during qualification"
                    )
                elapsed = time.monotonic() - started
                if elapsed > remaining:
                    raise TimeoutError("changed-region semantic qualification exceeded its wall budget")
                receipt = {
                    "schema": "global_changed_region_semantic_receipt_v2",
                    "iteration": row["iteration"],
                    "binding": binding.to_dict(),
                    **previous_pointer,
                    "portfolio_member_binding": member_binding,
                    "previous_artifact_sha256": previous["member_binding"]["lowered_sha256"],
                    "current_artifact_sha256": current["member_binding"]["lowered_sha256"],
                    "evidence": evidence,
                    "elapsed_seconds": elapsed,
                    "scope": "selected changed mechanism and tested reduced domain only",
                    "full_model_numerics_qualified": False,
                    "global_speedup_proven": False,
                    "full_model_cycles": None,
                }
                verify_changed_region_semantic_receipt(
                    receipt,
                    iteration=row,
                    portfolio_identity=self.analysis.session.inputs.portfolio_identity,
                    target_sha256=self.analysis.session.inputs.target_sha256,
                    experiment_root=self.analysis.session.journal.output,
                )
                path = self._write(f"semantic_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
                row.setdefault("semantic_receipts", []).append(
                    {"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)}
                )
                return receipt
            finally:
                # Charge failures too; the provider never charges these same seconds a second
                # time. A failed callback/compile still gets a fresh post-boundary integrity check.
                if not final_integrity_checked:
                    self.analysis.session.refresh(candidate)
                self.charge_probe_preparation(candidate, time.monotonic() - started)

    def profile_controlled_context(
        self, candidate: Path, *, provider: Callable[..., Mapping[str, Any]], timeout_s: float
    ) -> dict[str, Any]:
        """Time one bounded actual source prefix, without claiming containing-model equivalence."""
        started = time.monotonic()
        row = self.analysis.session.current(candidate)
        binding = self.analysis.session.current_probe_binding(candidate)
        captured = self.analysis.session.current_artifacts(candidate)
        budget = min(60.0, float(timeout_s), self.analysis.timeout_s - row["elapsed_seconds"])

        def remaining() -> float:
            value = budget - (time.monotonic() - started)
            if value <= 0:
                raise TimeoutError("controlled prefix preparation plus warm/measured execution exceeded 60s budget")
            return value

        try:
            action = provider(candidate=candidate, probes=self, timeout_s=remaining())
            inputs = action["controlled_context_inputs"]
            if (
                inputs["binding"] != binding
                or inputs["scope"] != "controlled_source_prefix"
                or inputs["model_artifact_sha256"] != captured["candidate_lowered_sha256"]
            ):
                raise ValueError("controlled source prefix is not bound to the current model and compiler")
            source, prepared = inputs["source_slice"], inputs["prepared"]
            indices = source["timed_source_instruction_indices"]
            if (
                source["source_artifact_sha256"] != captured["candidate_lowered_sha256"]
                or not 0 < len(indices) <= 32
                or len(indices) != len(set(indices))
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
                or len(source["source_instruction_indices"]) >= len(captured["decoded_trace"]["instructions"])
                or source.get("full_model_executed") is not False
                or source.get("full_layer_executed") is not False
            ):
                raise ValueError("controlled prefix must be a separate bounded subset of emitted model instructions")
            if (
                prepared["source_artifact_sha256"] != source["slice_source_sha256"]
                or prepared["measurement_scope"] != "controlled_source_prefix"
                or prepared.get("simulator_executed") is not False
                or not 0 < prepared["timed_instruction_count"] <= 32
                or not 0 < prepared["host_input_bytes"] <= 65536
                or not 0 < prepared["output_storage_bytes"] <= 65536
                or source["source_task_compute_pair_count"] <= 1
                or source["executed_compute_pair_count"] != 1
            ):
                raise ValueError("controlled executable does not match the extracted source prefix")
            elf = Path(prepared["workdir"]) / "primitive.elf"
            if P2_CONTRACTS.sha256_file(elf) != prepared["elf_sha256"]:
                raise ValueError("controlled prepared executable changed")
            observed = dict(action["execute"](timeout_s=remaining()))
            for key in (
                "elf_sha256",
                "wrapper_sha256",
                "primitive_mlir_sha256",
                "domain_digest",
                "source_artifact_sha256",
                "measurement_scope",
            ):
                if observed.get(key) != prepared[key]:
                    raise ValueError(f"controlled execution changed its admitted {key}")
            if (
                observed.get("correct") is not True
                or observed.get("warmup_runs") != 1
                or observed.get("measured_runs") != 1
                or observed.get("full_model_executed") is not False
                or observed.get("full_source_probe_executed") is not False
                or not isinstance(observed.get("total_compute_cycles"), int)
                or observed["total_compute_cycles"] <= 0
            ):
                raise ValueError("controlled execution lacks a correct warm1/measured1 prefix receipt")
            if (
                self.analysis.session.current_probe_binding(candidate) != binding
                or P2_CONTRACTS.sha256_file(elf) != prepared["elf_sha256"]
            ):
                raise ValueError("compiler or controlled executable changed during measurement")
            remaining()
            receipt = {
                "schema": "global_controlled_context_receipt_v1",
                "binding": binding.to_dict(),
                "iteration": row["iteration"],
                "source_slice": source,
                "execution": observed,
                "model_artifact_sha256": captured["candidate_lowered_sha256"],
                "scope": "controlled_source_prefix",
                "elapsed_seconds": time.monotonic() - started,
                "full_model_cycles": None,
                "global_cost_validated": False,
                "calibration_admissible": False,
                "global_speedup_proven": False,
            }
            path = self._write(f"context_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            row.setdefault("context_receipts", []).append({"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)})
            return receipt
        finally:
            self.charge_probe_preparation(candidate, time.monotonic() - started)

    def compare_controlled_context(
        self, candidate: Path, *, provider: Callable[..., Mapping[str, Any]], timeout_s: float
    ) -> dict[str, Any]:
        """Compare identical bounded work in two schedules, never projected model latency."""
        started = time.monotonic()
        row = self.analysis.session.current(candidate)
        bindings = {
            "before": self.analysis.session.previous_probe_binding(candidate),
            "after": self.analysis.session.current_probe_binding(candidate),
        }
        artifacts = {
            "before": self.analysis.session.previous_artifacts(candidate),
            "after": self.analysis.session.current_artifacts(candidate),
        }
        if bindings["before"].graph_digest != bindings["after"].graph_digest:
            raise ValueError("paired context requires the same complete logical graph")
        budget = min(60.0, float(timeout_s), self.analysis.timeout_s - row["elapsed_seconds"])

        def remaining() -> float:
            value = budget - (time.monotonic() - started)
            if value <= 0:
                raise TimeoutError("paired context preparation and both executions exceeded their total budget")
            return value

        def positive_int(value: Any, limit: int) -> bool:
            return type(value) is int and 0 < value <= limit

        try:
            action = provider(candidate=candidate, probes=self, timeout_s=remaining())
            inputs = action["paired_context_inputs"]
            proof = inputs["projection_proof"]
            contract = inputs["work_contract_sha256"]
            input_contract = inputs["deterministic_input_contract_sha256"]
            if (
                proof.get("schema") != "controlled_fixed_work_projection_v1"
                or proof.get("status") != "same_work_projection_verified"
                or proof.get("scope") != "controlled_fixed_work_slice"
                or proof.get("future_computes_omitted_symmetrically") is not True
                or proof.get("global_cost_validated") is not False
                or proof.get("work_contract_sha256") != contract
                or P2_CONTRACTS.document_sha256(proof["work_contract"]) != contract
                or not isinstance(input_contract, str)
                or len(input_contract) != 64
                or P2_CONTRACTS.document_sha256(inputs["deterministic_input_contract"]) != input_contract
            ):
                raise ValueError("paired context lacks a hash-bound same-work projection")
            executable_paths: dict[str, Path] = {}
            for arm in ("before", "after"):
                raw_sha = artifacts[arm]["candidate_lowered_sha256"]
                entry = inputs["arms"][arm]
                source, prepared = entry["source_slice"], entry["prepared"]
                indices = source["timed_source_instruction_indices"]
                if (
                    inputs[arm + "_binding"] != bindings[arm]
                    or inputs[arm + "_model_artifact_sha256"] != raw_sha
                    or proof[arm + "_artifact_sha256"] != raw_sha
                    or proof[arm + "_timed_indices"] != indices
                    or source["source_artifact_sha256"] != raw_sha
                    or source["work_contract_sha256"] != contract
                    or entry["work_contract_sha256"] != contract
                    or entry["deterministic_input_contract_sha256"] != input_contract
                    or not positive_int(len(indices), 32)
                    or len(set(indices)) != len(indices)
                    or any(type(i) is not int or i < 0 for i in indices)
                    or max(indices) >= len(artifacts[arm]["decoded_trace"]["instructions"])
                    or len(source["source_instruction_indices"]) >= len(artifacts[arm]["decoded_trace"]["instructions"])
                    or source.get("full_model_executed") is not False
                    or source.get("full_layer_executed") is not False
                    or source["source_task_compute_pair_count"] <= 1
                    or source["executed_compute_pair_count"] != 1
                ):
                    raise ValueError("paired context changed source identity, work, or extraction domain")
                if (
                    prepared["source_artifact_sha256"] != source["slice_source_sha256"]
                    or prepared["measurement_scope"] != "controlled_fixed_work_slice"
                    or prepared.get("simulator_executed") is not False
                    or not positive_int(prepared["timed_instruction_count"], 32)
                    or not positive_int(prepared["host_input_bytes"], 65536)
                    or not positive_int(prepared["output_storage_bytes"], 65536)
                ):
                    raise ValueError("paired context executable is not a bounded fixed-work slice")
                elf = Path(prepared["workdir"]) / "primitive.elf"
                if P2_CONTRACTS.sha256_file(elf) != prepared["elf_sha256"]:
                    raise ValueError("paired context prepared executable changed")
                executable_paths[arm] = elf
            executions = {}
            for arm in ("before", "after"):
                prepared = inputs["arms"][arm]["prepared"]
                observed = dict(action["execute"](arm=arm, timeout_s=remaining()))
                for key in (
                    "elf_sha256",
                    "wrapper_sha256",
                    "primitive_mlir_sha256",
                    "domain_digest",
                    "source_artifact_sha256",
                    "measurement_scope",
                ):
                    if observed.get(key) != prepared[key]:
                        raise ValueError(f"paired execution changed its admitted {arm} {key}")
                if (
                    observed.get("correct") is not True
                    or observed.get("warmup_runs") != 1
                    or observed.get("measured_runs") != 1
                    or observed.get("full_model_executed") is not False
                    or observed.get("full_source_probe_executed") is not False
                    or not positive_int(observed.get("total_compute_cycles"), 2**63 - 1)
                ):
                    raise ValueError("paired context requires correct warm1/measured1 evidence per arm")
                executions[arm] = observed
            if (
                self.analysis.session.current_probe_binding(candidate) != bindings["after"]
                or self.analysis.session.previous_probe_binding(candidate) != bindings["before"]
            ):
                raise ValueError("paired context compiler or graph binding changed")
            for arm, elf in executable_paths.items():
                if P2_CONTRACTS.sha256_file(elf) != inputs["arms"][arm]["prepared"]["elf_sha256"]:
                    raise ValueError("paired context executable changed during measurement")
            remaining()
            cycles = {arm: item["total_compute_cycles"] for arm, item in executions.items()}
            receipt = {
                "schema": "global_paired_controlled_context_receipt_v1",
                "iteration": row["iteration"],
                "binding": bindings["after"].to_dict(),
                "previous_binding": bindings["before"].to_dict(),
                "model_artifact_sha256": artifacts["after"]["candidate_lowered_sha256"],
                "previous_model_artifact_sha256": artifacts["before"]["candidate_lowered_sha256"],
                "scope": "controlled_fixed_work_slice",
                "projection_proof": proof,
                "previous_iteration_record": str(
                    (self.analysis.session.journal.output / f"iteration_{row['iteration'] - 1:04d}.json").resolve()
                ),
                "previous_iteration_record_sha256": P2_CONTRACTS.sha256_file(
                    self.analysis.session.journal.output / f"iteration_{row['iteration'] - 1:04d}.json"
                ),
                "deterministic_input_contract": inputs["deterministic_input_contract"],
                "deterministic_input_contract_sha256": input_contract,
                "executions": executions,
                "cycles": cycles,
                "after_minus_before_cycles": cycles["after"] - cycles["before"],
                "elapsed_seconds": time.monotonic() - started,
                "full_model_cycles": None,
                "global_cost_validated": False,
                "global_speedup_proven": False,
                "calibration_admissible": False,
                "statistical_confirmation": "not_performed",
            }
            feedback = paired_context_decision_feedback(
                row, receipt, target_sha256=self.analysis.session.inputs.target_sha256
            )
            receipt["decision_feedback"] = feedback
            path = self._write(f"paired_context_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            row.setdefault("paired_context_receipts", []).append(
                {"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)}
            )
            row["decision_feedback"] = {
                **feedback,
                "receipt": {"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)},
            }
            return receipt
        finally:
            self.charge_probe_preparation(candidate, time.monotonic() - started)

    def measure_probe(
        self,
        candidate: Path,
        *,
        admission_inputs: Mapping[str, Any],
        execute: Callable[..., ProbeObservation],
        timeout_s: int | None = None,
    ) -> dict[str, Any]:
        """Execute only an independently extracted, equivalent short mechanism after admission."""
        row = self.analysis.session.current(candidate)
        diag = row["analysis"]["diagnostics"]
        plan = diag["verified_global_plan_emission"]
        binding = ProbeBinding(
            graph_digest=diag["captured_logical_graph"]["logical_dispatch_digest"],
            plan_digest=plan["plan_digest"],
            compiler_digest=row["compiler_dependencies"]["compiler_implementation_sha256"],
            target_digest=self.analysis.session.inputs.target_sha256,
        )
        inputs = dict(admission_inputs)
        if "current_binding" in inputs:
            raise ValueError("the host derives current probe bindings; callers cannot replace them")
        if inputs["model"].artifact_digest != row["analysis"]["emission"]["candidate_lowered_sha256"]:
            raise ValueError("model mechanism evidence names a different compiled artifact")
        admitted = require_probe_admission(current_binding=binding, **inputs)
        if not admitted.admitted:
            raise ValueError("probe exceeds fast iteration domain: " + admitted.reason)
        if row["elapsed_seconds"] + float(admitted.estimated_seconds or 0) > self.analysis.timeout_s:
            raise ValueError("compile plus warm/measured probe pair exceeds iteration wall budget")
        remaining = min(
            inputs["budget"].timeout_seconds,
            self.analysis.timeout_s - row["elapsed_seconds"],
            self.analysis.timeout_s if timeout_s is None else timeout_s,
        )
        if float(admitted.estimated_seconds or 0) > remaining:
            raise ValueError("probe pair exceeds the remaining broker action budget")
        started = time.monotonic()
        try:
            observed = execute(timeout_s=remaining)
        finally:
            # Failed, timed-out, or subsequently rejected observations consume the same
            # iteration budget as valid measurements. Charge the originating row even
            # if the callback changed the candidate; never turn retries into free time.
            elapsed = time.monotonic() - started
            row["elapsed_seconds"] += elapsed
        if not isinstance(observed, ProbeObservation) or observed.evidence != inputs["probe"]:
            raise ValueError("executed probe does not match the admitted artifact and mechanism")
        self.analysis.session.current(candidate)
        if elapsed > remaining or row["elapsed_seconds"] > self.analysis.timeout_s:
            raise ValueError("probe execution exceeded its bounded iteration budget")
        receipt = {
            "schema": "global_mechanism_probe_receipt_v1",
            "iteration": row["iteration"],
            "binding": binding.to_dict(),
            "artifact_digest": observed.evidence.artifact_digest,
            "mechanism": observed.evidence.signature.to_dict(),
            "evidence": observed.evidence.to_dict(),
            "compute_receipt": observed.receipt.to_dict(),
            "counter_uncertainty_cycles": observed.counter_uncertainty_cycles,
            "target_timing_authority": (
                observed.timing_authority.to_evidence() if observed.timing_authority is not None else None
            ),
            "observed_timing_identity": (
                asdict(observed.observed_timing_identity) if observed.observed_timing_identity is not None else None
            ),
            "timing_scope": "raw probe observation; target-cycle inference requires separately validated calibration",
            "resource_profile": observed.resource_profile,
            "total_compute_cycles": observed.receipt.total_compute_cycles,
            "warmup_runs": 1,
            "measured_runs": 1,
            "elapsed_seconds": elapsed,
            "scope": "mechanism_probe_only",
            "full_model_cycles": None,
        }
        from merlin.perf.probe_relevance import classify_probe_relevance

        try:
            receipt["relevance"] = classify_probe_relevance(
                previous_analysis=self.analysis.session.journal.iterations[-2]["analysis"]
                if len(self.analysis.session.journal.iterations) > 1
                else None,
                current_analysis=row["analysis"],
                previous_artifacts=self.analysis.session.journal.previous_artifacts,
                current_artifacts=self.analysis.session.journal.artifacts,
                signature=observed.evidence.signature,
            )
        except ValueError as exc:
            receipt["relevance"] = {
                "status": "UNKNOWN",
                "reason": str(exc),
                "global_cost_validated": False,
                "scope": "isolated admitted calibration only",
            }
        path = self._write(f"probe_{row['iteration']:04d}_{len(row['probe_receipts']):04d}.json", receipt)
        row["probe_receipts"].append({"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)})
        return receipt

    def _write(self, name: str, record: Mapping[str, Any]) -> Path:
        path = self.analysis.session.journal.output / name
        payload = P2_CONTRACTS.canonical_json(record)
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o444)
        return path
