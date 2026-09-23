#!/usr/bin/env python3
"""Compile real model candidates and consume global-plan evidence without full-model simulation.

The macro search has its own receipt schema. It cannot produce a microbenchmark promotion record
or stop because a microbenchmark plateaued. Static accounting is the objective; optional probe
observations calibrate specific mechanisms and never become measured model latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import portfolio_checkpoint as CHECKPOINT
from merlin_experiments.phase2 import stage_inputs as INPUTS
from merlin_experiments.phase2 import static_identity as SI
from merlin_experiments.phase2.emission_analysis import (
    seed_baseline_emission_cache_from_run as seed_baseline_emission_cache_from_run,
)
from merlin_experiments.phase2.global_experiment import GlobalPerfExperiment as InstalledGlobalPerfExperiment
from merlin_experiments.phase2.global_inputs import FrozenPhase1 as FrozenPhase1
from merlin_experiments.phase2.global_inputs import (
    current_machine_build_policy as _current_machine_build_policy,  # noqa: F401
)
from merlin_experiments.phase2.global_inputs import full_model_portfolio_identity as full_model_portfolio_identity
from merlin_experiments.phase2.portfolio_analysis import PortfolioAnalysis
from merlin_experiments.phase2.portfolio_checkpoint import (
    verify_changed_region_semantic_receipt as _verify_changed_region_semantic_receipt,
)
from merlin_experiments.phase2.portfolio_worker import configure_global_analysis as _configure_global_analysis
from merlin_experiments.phase2.portfolio_worker import declared_instruction_set_brief as declared_instruction_set_brief
from merlin_experiments.phase2.portfolio_worker import (
    validate_optimization_baseline_resume as validate_optimization_baseline_resume,
)

from merlin.common.paths import merlin_dir, repo_root
from merlin.perf.functional_gate import (
    load_functional_gate_config,
)

_GIB = 1024**3


def host_verification_policy_record(*, contract_root: Path | None = None) -> dict[str, Any]:
    """Bind the interpretation of structural evidence to current host implementation bytes."""
    return HP.build_record(controller_source=Path(__file__), contract_root=_contract_root_path(contract_root))


def _contract_root_path(contract_root: Path | None) -> Path:
    """Resolve the explicit selection or this native controller's historical layout."""
    return Path(contract_root).resolve() if contract_root is not None else repo_root() / "merlin/contract"


def _selected_host_policy(contract_root: Path | None) -> dict[str, Any]:
    """Keep the legacy no-argument observation when no resource selection was supplied."""
    if contract_root is None:
        return host_verification_policy_record()
    return host_verification_policy_record(contract_root=contract_root)


def compiler_dependency_record(candidate: Path, *, shared_source_root: Path | None = None) -> dict[str, Any]:
    """Bind the compiler's static import closure using this controller's legacy source layout."""
    return SI.compiler_dependency_record(
        candidate, shared_source_root=shared_source_root or merlin_dir() / "python" / "merlin"
    )


class GlobalPerfExperiment(InstalledGlobalPerfExperiment):
    """Native compatibility edge supplying the historical source and resource defaults."""

    def __init__(
        self,
        *,
        compiler_shared_source_root: Path | None = None,
        contract_root: Path | None = None,
        **kwargs: Any,
    ) -> None:
        # Preserve timeout refusal before resolving compatibility resources.
        PortfolioAnalysis.validate_timeout(kwargs.get("timeout_s", 300))
        super().__init__(
            **kwargs,
            compiler_shared_source_root=(
                compiler_shared_source_root
                if compiler_shared_source_root is not None
                else merlin_dir() / "python/merlin"
            ),
            contract_root=_contract_root_path(contract_root),
            controller_source=Path(__file__),
            prior_shared_source_relative=Path("merlin/python/merlin"),
            prior_shared_source_fallback=merlin_dir() / "python/merlin",
            guidance_contract=_contract_root_path(contract_root) if contract_root is not None else None,
        )


def consume_authoring_checkpoint(
    path: Path, *, contract_root: Path | None = None, compiler_shared_source_root: Path | None = None
) -> dict[str, Any]:
    return CHECKPOINT.consume_authoring_checkpoint(
        path,
        context=CHECKPOINT.CheckpointVerificationContext(
            host_policy=_selected_host_policy(contract_root),
            compiler_shared_source_root=(
                compiler_shared_source_root
                if compiler_shared_source_root is not None
                else merlin_dir() / "python/merlin"
            ),
        ),
    )


def consume_global_candidate(
    path: Path, *, contract_root: Path | None = None, compiler_shared_source_root: Path | None = None
) -> dict[str, Any]:
    return CHECKPOINT.consume_global_candidate(
        path,
        context=CHECKPOINT.CheckpointVerificationContext(
            host_policy=_selected_host_policy(contract_root),
            compiler_shared_source_root=(
                compiler_shared_source_root
                if compiler_shared_source_root is not None
                else merlin_dir() / "python/merlin"
            ),
        ),
    )


def write_semantic_supplement(
    *,
    original_candidate_receipt: Path,
    previous_iteration_receipt: Path,
    semantic_receipt: Path,
    output: Path,
    contract_root: Path | None = None,
) -> dict[str, Any]:
    """Add scoped evidence to an immutable checkpoint without relabeling its original verdict."""
    if output.exists() or output.is_symlink():
        raise ValueError("semantic supplement output must be fresh")
    pointers = {
        name: {"path": str(path.resolve()), "sha256": P2_CONTRACTS.sha256_file(path)}
        for name, path in (
            ("original_candidate_receipt", original_candidate_receipt),
            ("previous_iteration_receipt", previous_iteration_receipt),
            ("semantic_receipt", semantic_receipt),
        )
    }
    document = {
        "schema": "global_semantic_supplement_v2",
        **pointers,
        "legacy_receipt_policy": "v1_refused_requires_requalification",
        "host_verification_policy": _selected_host_policy(contract_root),
        "analysis_action": "reuse_existing_bound_immutable_graph_receipts",
        "original_verdict_modified": False,
        "full_model_numerics_qualified": False,
        "global_speedup_proven": False,
        "full_model_cycles": None,
    }
    P2_CONTRACTS.write_json(output, document)
    return consume_semantic_supplement(output, contract_root=contract_root)


def verify_retained_global_checkpoint(original_path: Path) -> dict[str, Any]:
    """Select the recorded installed verifier or this edge's historical native layout."""
    from merlin_experiments.phase2.portfolio_resume import NativeVerifierLayout, verify

    return verify(
        original_path,
        native_layout=NativeVerifierLayout(
            controller_relative="merlin/experiments/gemmini_perf_bench/scripts/run_global_perf_experiment.py",
            python_roots=("merlin/python", "merlin/experiments/gemmini_perf_bench/scripts"),
            module="run_global_perf_experiment",
        ),
    )


def consume_semantic_supplement(path: Path, *, contract_root: Path | None = None) -> dict[str, Any]:
    """Verify old checkpoint under its own policy, and new mechanism evidence under this policy."""
    document = P2_CONTRACTS.mapping_file(path)
    if (
        document.get("schema") != "global_semantic_supplement_v2"
        or document.get("legacy_receipt_policy") != "v1_refused_requires_requalification"
        or document.get("host_verification_policy") != _selected_host_policy(contract_root)
        or document.get("analysis_action") != "reuse_existing_bound_immutable_graph_receipts"
        or document.get("full_model_cycles") is not None
        or any(
            document.get(key) is not False
            for key in ("original_verdict_modified", "full_model_numerics_qualified", "global_speedup_proven")
        )
    ):
        raise ValueError("semantic supplement policy or scope changed")
    loaded = {}
    for key in ("original_candidate_receipt", "previous_iteration_receipt", "semantic_receipt"):
        pointer = document[key]
        source = Path(pointer["path"])
        if source.is_symlink() or P2_CONTRACTS.sha256_file(source) != pointer["sha256"]:
            raise ValueError("semantic supplement source receipt changed")
        loaded[key] = P2_CONTRACTS.mapping_file(source)
    original_path = Path(document["original_candidate_receipt"]["path"])
    original = verify_retained_global_checkpoint(original_path)
    final_iteration = P2_CONTRACTS.mapping_file(Path(original["iteration_record"]))
    semantic = loaded["semantic_receipt"]
    if document["previous_iteration_receipt"] != {
        "path": semantic.get("previous_iteration_record"),
        "sha256": semantic.get("previous_iteration_record_sha256"),
    }:
        raise ValueError("supplement preceding iteration differs from semantic evidence")
    binding = _verify_changed_region_semantic_receipt(
        semantic,
        iteration=final_iteration,
        portfolio_identity=original["portfolio"],
        target_sha256=original["target_sha256"],
        experiment_root=original_path.parent,
    )
    return {
        **document,
        "candidate_sha256": original["candidate_sha256"],
        "semantic_status": semantic["evidence"].get("status"),
        "portfolio_member_binding": binding,
        "original_checkpoint_verified": True,
    }


def configure_global_analysis(
    experiment: GlobalPerfExperiment,
    *,
    target_experiment: Any,
    agent_inputs: Any,
    frozen_functional: Any,
    frozen_corpus_manifest: Path,
    stage_root: Path,
    sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs | None = None,
) -> None:
    """Native edge supplies historical tool selection to the shared worker owner."""
    _configure_global_analysis(
        experiment,
        target_experiment=target_experiment,
        agent_inputs=agent_inputs,
        frozen_functional=frozen_functional,
        frozen_corpus_manifest=frozen_corpus_manifest,
        stage_root=stage_root,
        sandbox_inputs=sandbox_inputs
        if sandbox_inputs is not None
        else PC.select_package_sandbox_inputs(target_experiment),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-capsule", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument(
        "--functional-gate",
        type=Path,
        default=None,
        help="JSON (merlin_functional_gate_config_v1): model payload dir, toolchain "
        "paths/ISA strings, MERLIN_RESULT expectations and a timeout; when "
        "present the emitted program is built and executed each iteration",
    )
    args = parser.parse_args(argv)
    source = args.model_capsule.resolve()
    descriptor = P2_CONTRACTS.mapping_file(source / "capsule.yaml", yaml_file=True)
    sentinel = INPUTS.StageE2ESentinel(
        capsule=str(descriptor.get("id") or source.name),
        capsule_path=str(source),
        frozen_source_path=str(source),
        capsule_sha256=P2_CONTRACTS.exact_tree_record(source)["sha256"],
        required_lanes=tuple((descriptor.get("lanes") or {}).get("require") or ()),
        required_tiers=tuple(descriptor.get("required_oracle_tiers") or ()),
    )
    experiment = GlobalPerfExperiment(
        baseline=args.baseline,
        baseline_sha256=args.baseline_sha256,
        sentinel=sentinel,
        target=args.target,
        target_sha256=hashlib.sha256(args.target_descriptor.read_bytes()).hexdigest(),
        target_descriptor=args.target_descriptor,
        output=args.output,
        timeout_s=args.timeout_seconds,
        functional_gate=(load_functional_gate_config(args.functional_gate) if args.functional_gate else None),
    )
    record = experiment.analysis.analyze(args.candidate, hypothesis=args.hypothesis)
    print(
        json.dumps(
            {
                "readiness": record["readiness"],
                "elapsed_seconds": record["elapsed_seconds"],
                "functional_gate": {key: record["functional_gate"].get(key) for key in ("status", "reason", "stage")},
                "record": str(args.output / "iteration_0000.json"),
            },
            indent=2,
        )
    )
    return 0 if record["readiness"]["status"] == "ready_for_probe_admission" else 2


if __name__ == "__main__":
    raise SystemExit(main())
