"""Explicit tool selections and cached compiler boundaries for portfolio analysis."""

from __future__ import annotations

import copy
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.analysis_worker import IsolatedAnalysisWorker
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.target_experiment import TargetExperiment
from merlin_experiments.frozen_python import inherited_python_command

from . import agent_workspace as AW
from . import broker as PB
from . import campaign as PC
from . import contracts as P2_CONTRACTS
from . import emission_analysis as EA
from . import emission_diagnostics as ED
from . import host_policy as HP
from . import static_identity as SI
from .functional_inputs import FrozenFunctionalInputs
from .portfolio_analysis import PortfolioAnalysis
from .portfolio_probes import compiler_dependency_mounts


def rebind_compiler_sandbox(
    cached: Mapping[str, Any], *, package: Path, scratch: Path, dependencies: Mapping[str, Any]
) -> dict[str, Any]:
    """Reuse exact deny masks; move only host-owned compiler/scratch sources, retaining aliases.

    The existing virtual cwd/import alias remains bound to the new immutable compiler bytes.
    Extra exact-path mounts make the worker's fresh absolute entrypoint/input paths available.
    No parent directory, environment, toolchain, or answer grant is enlarged.
    """
    cached = copy.deepcopy(cached)
    previous = cached["compiler_dependencies"]
    if any(
        previous.get(key) != dependencies.get(key)
        for key in ("shared_source_root", "shared_sources", "selected_lazy_exports")
    ):
        raise ValueError("cached compiler policy shared dependency closure changed")
    package, scratch = package.absolute(), scratch.absolute()
    if any(path.is_symlink() or not path.is_dir() or path.resolve() != path for path in (package, scratch)):
        raise ValueError("cached compiler policy requires real exact candidate and scratch directories")
    if package == scratch or package in scratch.parents or scratch in package.parents:
        raise ValueError("compiler and scratch grants must be independent")
    for surface in cached["answer_surfaces"]:
        answer = Path(surface["path"]).resolve()
        for path in (package, scratch):
            if path == answer or path in answer.parents or (surface["kind"] == "dir" and answer in path.parents):
                raise ValueError("cached compiler rebind overlaps a masked answer surface")
    if (
        SI.compiler_dependency_record(package, shared_source_root=Path(dependencies["shared_source_root"]))
        != dependencies
    ):
        raise ValueError("cached compiler rebind source identity changed")
    for directory, digest in cached.get("overlay_trees", {}).items():
        if P2_CONTRACTS.exact_tree_record(Path(directory))["sha256"] != digest:
            raise ValueError("cached read-only compiler overlay changed")
    prefix = list(cached["command_prefix"])
    boundary = cached["bwrap_argv_length"]
    old_package, old_scratch = str(cached["package_path"]), str(cached["scratch_path"])
    masks = []
    surface_paths = {row["path"] for row in cached["answer_surfaces"]}
    for index, token in enumerate(prefix[:boundary]):
        if token == "--tmpfs" and prefix[index + 1] in surface_paths:
            masks.append((index, tuple(prefix[index : index + 2])))
        elif token == "--ro-bind" and prefix[index + 1] == "/dev/null" and prefix[index + 2] in surface_paths:
            masks.append((index, tuple(prefix[index : index + 3])))
        elif token in ("--ro-bind", "--bind"):
            for before, after in ((old_package, str(package)), (old_scratch, str(scratch))):
                source = prefix[index + 1]
                if source == before or source.startswith(before + "/"):
                    prefix[index + 1] = after + source[len(before) :]
    if not masks:
        raise ValueError("cached compiler policy has no retained explicit answer masks")
    if any(tuple(prefix[index : index + len(mask)]) != mask for index, mask in masks):
        raise ValueError("cached compiler rebind changed an answer mask")
    insertion = min(index for index, _ in masks)
    additions = ["--ro-bind", str(package), str(package), "--bind", str(scratch), str(scratch)]
    prefix[insertion:insertion] = additions
    return {
        **cached,
        "package_path": str(package),
        "scratch_path": str(scratch),
        "compiler_dependencies": copy.deepcopy(dict(dependencies)),
        "command_prefix": prefix,
        "bwrap_argv_length": boundary + len(additions),
        "policy_reuse": {
            "status": "exact_masks_and_shared_closure_rebound",
            "cached_prefix_sha256": P2_CONTRACTS.document_sha256(cached["command_prefix"]),
            "retained_virtual_candidate_alias": old_package,
            "rebound_candidate_sha256": dependencies["candidate_sha256"],
            "source_grants_changed": [old_package, old_scratch],
            "answer_masks_changed": False,
        },
    }


class PortfolioSandboxFactory:
    """Build and rebind compiler policies from one explicit experiment selection."""

    def __init__(
        self,
        analysis: PortfolioAnalysis,
        *,
        target_experiment: TargetExperiment,
        agent_inputs: AW.AgentInputSnapshot,
        frozen_functional: FrozenFunctionalInputs,
        frozen_corpus_manifest: Path,
        sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs,
    ) -> None:
        if not isinstance(sandbox_inputs, (PC.PackageSandboxInputs, PC.FrozenPackageSandboxInputs)):
            raise ValueError("portfolio sandbox requires explicit selected tool inputs")
        self.analysis = analysis
        self.target_experiment = target_experiment
        self.agent_inputs = agent_inputs
        self.frozen_functional = frozen_functional
        self.frozen_corpus_manifest = Path(frozen_corpus_manifest).resolve()
        self._sandbox_inputs = copy.deepcopy(sandbox_inputs)
        self._cache: dict[str, tuple[str, dict[str, Any], AW.AgentSandboxPolicy]] = {}

    def __call__(self, baseline: Path, candidate: Path, scratch: Path) -> dict[str, Any]:
        inputs = self.analysis.session.inputs
        target_experiment = self.target_experiment
        agent_inputs = self.agent_inputs
        frozen_functional = self.frozen_functional
        frozen_corpus_manifest = self.frozen_corpus_manifest
        sandbox_inputs = self._sandbox_inputs
        started = time.monotonic()
        AW.verify_answer_free_agent_inputs(agent_inputs)
        selected_inputs = sandbox_inputs
        if isinstance(selected_inputs, PC.FrozenPackageSandboxInputs):
            from merlin_experiments.phase2.qualification_policy import restore

            selected_inputs = restore(selected_inputs.root, selected_inputs.record)
            # Rebinding may insert grants before an early answer mask, including
            # before the frozen tool block. Require disjoint roots regardless of
            # argv order so a tool grant cannot substitute compiler/scratch bytes.
            for root in (baseline, candidate, scratch):
                root = Path(root).absolute()
                if root.is_symlink() or not root.is_dir() or root.resolve() != root or ".." in root.parts:
                    raise ValueError("frozen compiler policy requires real exact compiler and scratch directories")
                for mount in selected_inputs.record["mounts"]:
                    tool = Path(mount["destination"])
                    if (
                        not tool.is_absolute()
                        or ".." in tool.parts
                        or mount["destination"].startswith("//")
                        or "\0" in mount["destination"]
                    ):
                        raise ValueError("frozen compiler policy has an unsafe tool destination")
                    if root == tool or root in tool.parents or tool in root.parents:
                        raise ValueError("compiler or scratch root overlaps a frozen tool destination")
        surfaces = selected_inputs.surfaces
        surfaces_record = [{"path": str(surface.path), "kind": surface.kind} for surface in surfaces]
        fixed = {
            "host_policy": HP.build_record(
                controller_source=inputs.controller_source, contract_root=inputs.contract_root
            ),
            "surfaces": surfaces_record,
            "descriptor": P2_CONTRACTS.sha256_file(target_experiment.path),
            "agent_inputs": P2_CONTRACTS.sha256_file(agent_inputs.manifest_path),
            "frozen_inputs": P2_CONTRACTS.sha256_file(frozen_functional.marker),
            "frozen_corpus": P2_CONTRACTS.sha256_file(frozen_corpus_manifest),
            "baseline_sha256": hash_tree(baseline)["sha256"],
            "qualification_baseline_sha256": hash_tree(inputs.baseline)["sha256"],
        }
        result = {}
        for arm, package in (("baseline", baseline), ("candidate", candidate)):
            dependencies = SI.compiler_dependency_record(package, shared_source_root=inputs.compiler_shared_source_root)
            key = P2_CONTRACTS.document_sha256(
                {
                    **fixed,
                    "shared_source_root": dependencies["shared_source_root"],
                    "shared_sources": dependencies["shared_sources"],
                    "selected_lazy_exports": dependencies.get("selected_lazy_exports", {}),
                }
            )
            if arm in self._cache and self._cache[arm][0] == key:
                result[arm] = rebind_compiler_sandbox(
                    self._cache[arm][1], package=package, scratch=scratch, dependencies=dependencies
                )
                if isinstance(selected_inputs, PC.FrozenPackageSandboxInputs):
                    replace(
                        self._cache[arm][2],
                        argv=tuple(result[arm]["command_prefix"][: result[arm]["bwrap_argv_length"]]),
                    ).verify_execution()
                result[arm]["policy_setup_elapsed_seconds"] = time.monotonic() - started
                continue
            policy = AW.inner_execution_policy(
                target_experiment,
                package,
                agent_inputs,
                frozen_functional,
                inputs.baseline,
                frozen_corpus_manifest,
                inputs=selected_inputs,
            )
            argv = [*policy.argv, "--ro-bind", str(package), str(package), "--bind", str(scratch), str(scratch)]
            # Frozen Phase-1 contract mounts remain untouched. The current
            # whole-program compiler API has a distinct, explicit namespace.
            schema = ED.whole_program_schema_record(inputs.contract_root / "schemas/command_buffer.schema.json")
            argv.extend(
                (
                    "--ro-bind",
                    schema["path"],
                    "/compiler-api/command_buffer.schema.json",
                    "--setenv",
                    "MERLIN_COMMAND_BUFFER_SCHEMA",
                    "/compiler-api/command_buffer.schema.json",
                )
            )
            overlay_root = scratch.parent / f"{scratch.name}.{arm}_dependency_overlays"
            argv = compiler_dependency_mounts(argv, dependencies, overlay_root)
            # Compiler imports never unmask an evaluator or answer surface, even when a static
            # import closure conservatively includes branches that this compiler does not execute.
            argv = BW.apply_answer_masks(argv, surfaces)
            gaps = BW.coverage_gap(argv, surfaces)
            if gaps:
                raise ValueError("compiler dependency grants expose answer surfaces")
            policy = replace(policy, argv=tuple(argv), candidate_writable=False)
            prefix = PB.inner_command(policy, target_experiment, package, ["PAYLOAD_MARKER"], 600)[:-1]
            result[arm] = {
                "package_path": str(package),
                "command_prefix": prefix,
                "scratch_path": str(scratch),
                "compiler_dependencies": dependencies,
                "bwrap_argv_length": len(policy.argv),
                "answer_surfaces": surfaces_record,
                "overlay_trees": {str(overlay_root): P2_CONTRACTS.exact_tree_record(overlay_root)["sha256"]}
                if overlay_root.exists()
                else {},
                "policy_setup_elapsed_seconds": time.monotonic() - started,
            }
            self._cache[arm] = (key, copy.deepcopy(result[arm]), policy)
        return result

    def install_worker(self, *, output: Path) -> None:
        """Install the canonical isolated worker without discovering a target backend."""
        if self.analysis.analyzer is EA.analyze_whole_model_emission:
            self.analysis.analyzer = IsolatedAnalysisWorker(
                analysis_source=Path(EA.__file__),
                contract_root=self.analysis.session.inputs.contract_root,
                sandbox_factory=self,
                output=Path(output),
                python_command=inherited_python_command,
            )
