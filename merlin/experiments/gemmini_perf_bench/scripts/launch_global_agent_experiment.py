#!/usr/bin/env python3
"""Launch full-graph authoring with optional bounded isolated probes, never model simulation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2.fast_evaluation_installation import (
    _HOST_QUALITY_OBSERVER_CONTRACT as _HOST_QUALITY_OBSERVER_CONTRACT,
)
from merlin_experiments.phase2.fast_evaluation_installation import (
    _exact_file as _exact_file,
)
from merlin_experiments.phase2.fast_evaluation_installation import (
    _load_host_quality_observer as _load_host_quality_observer,
)
from merlin_experiments.phase2.fast_evaluation_installation import (
    prepare as _prepare_fast_evaluator_installation,  # noqa: F401 - native inspection compatibility
)
from merlin_experiments.phase2.fast_evaluation_installation import (
    validate_cli as _validate_fast_evaluation_cli,  # noqa: F401 - native inspection compatibility
)
from merlin_experiments.phase2.fast_evaluation_installation import (
    worker_arguments as _fast_evaluation_worker_arguments,  # noqa: F401 - native inspection compatibility
)
from merlin_experiments.phase2.global_inputs import full_model_portfolio_identity as full_model_portfolio_identity
from merlin_experiments.phase2.portfolio_worker import (
    configure_global_analysis as configure_global_analysis,
)
from merlin_experiments.phase2.portfolio_worker import (
    declared_instruction_set_brief as declared_instruction_set_brief,
)
from merlin_experiments.phase2.portfolio_worker import (
    load_host_guidance_declarations as load_host_guidance_declarations,
)
from merlin_experiments.phase2.portfolio_worker import (
    run_analysis_only as run_analysis_only,
)
from merlin_experiments.phase2.portfolio_worker import (
    run_authoring_with_terminal_receipt as run_authoring_with_terminal_receipt,
)
from merlin_experiments.phase2.portfolio_worker import (
    validate_optimization_baseline_resume as validate_optimization_baseline_resume,
)

from merlin.benchharness import runs_root
from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.target_experiment import load_target_experiment


def _mechanism_catalog_worker_arguments(path: Path | None, digest: str | None) -> tuple[str, ...]:
    """Forward the already-validated raw pin without resolving or rediscovering it."""
    if path is None:
        return ()
    return ("--mechanism-catalog", str(path), "--mechanism-catalog-sha256", str(digest))


def _mechanism_work_order_worker_arguments(path: Path | None, digest: str | None) -> tuple[str, ...]:
    """Forward one already-validated immutable host work-order pin exactly."""
    if path is None:
        return ()
    return ("--mechanism-work-order", str(path), "--mechanism-work-order-sha256", str(digest))


def main(argv: list[str] | None = None) -> int:
    sys.dont_write_bytecode = True
    from merlin_experiments.phase2.portfolio_options import parse_invocation

    invocation = parse_invocation(argv, description=__doc__)
    args = invocation.args
    config_document = P2_CONTRACTS.mapping_file(args.campaign_config)
    config = (
        config_document["campaigns"][0]["config"]
        if "campaigns" in config_document
        else config_document.get("config", config_document)
    )
    if not args.source_worker:
        from merlin_experiments import source_snapshot as perf_snapshot
        from merlin_experiments.phase2.portfolio_launch import PortfolioDeployment, launch
        from source_snapshot_layout import snapshot_layout

        # Compatibility layout/provider selection is read-only and precedes acquiring
        # the installed launch owner's lease. Failed selection holds no resource lock.
        source = repo_root().resolve()
        snapshot = args.output.resolve().with_name(args.output.name + ".source")
        descriptor = P2_CONTRACTS.mapping_file(Path(config["descriptor"]), yaml_file=True)
        target_name = str(descriptor["target"])
        provider = perf_snapshot.selected_provider(target_name)
        return launch(
            invocation,
            deployment=PortfolioDeployment(
                source_root=source,
                output_root=source / "out",
                lease_path=source / "out/artifacts/cache/host_resources/full_model_perf.lock",
                worker_entrypoint=(
                    sys.executable,
                    str(snapshot / Path(__file__).resolve().relative_to(source)),
                ),
                snapshot_options=snapshot_layout(source, target_name=target_name, provider=provider),
                target_name=target_name,
                selected_provider=provider,
                declared_inputs={
                    key: Path(config[key])
                    for key in ("descriptor", "telemetry_price_table")
                    if config.get(key) is not None
                },
                inherited_environment=dict(os.environ),
                worker_python_roots=(
                    "merlin/python",
                    "merlin/experiments/gemmini_perf_bench/scripts",
                    "merlin/experiments/capsule_bench/harness",
                ),
            ),
        )
    from merlin_experiments import source_snapshot as perf_snapshot
    from merlin_experiments.phase2.portfolio_resume import NativeVerifierLayout
    from merlin_experiments.phase2.portfolio_worker import PortfolioWorkerContext, run
    from run_global_perf_experiment import __file__ as controller_source

    snapshot_root = repo_root().resolve()
    snapshot_receipt = perf_snapshot.verify(snapshot_root)
    descriptor = Path(config["descriptor"])
    if snapshot_receipt["schema"] == perf_snapshot.SCHEMA:
        descriptor = perf_snapshot.remap_input(snapshot_root, snapshot_receipt, descriptor, name="descriptor")
    target = load_target_experiment(descriptor, source_root=snapshot_root)
    return run(
        invocation,
        config,
        context=PortfolioWorkerContext(
            snapshot_root=snapshot_root,
            functional_runs_root=runs_root(target.target, "capsule-bench"),
            contract_root=snapshot_root / "merlin/contract",
            compiler_shared_source_root=merlin_dir() / "python/merlin",
            controller_source=Path(controller_source),
            prior_shared_source_relative=Path("merlin/python/merlin"),
            prior_shared_source_fallback=merlin_dir() / "python/merlin",
            guidance_contract=None,
            sandbox_inputs=PC.select_package_sandbox_inputs(target),
            target_experiment=target,
            native_verifier_layout=NativeVerifierLayout(
                controller_relative="merlin/experiments/gemmini_perf_bench/scripts/run_global_perf_experiment.py",
                python_roots=("merlin/python", "merlin/experiments/gemmini_perf_bench/scripts"),
                module="run_global_perf_experiment",
            ),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
