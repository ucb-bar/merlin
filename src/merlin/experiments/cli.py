"""One discoverable command surface for phase definitions and existing engines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from merlin.experiments.definition import ENGINE_STAGES, STAGE_ENGINES, SpecError, engine_definitions, load_spec
from merlin.experiments.definition import catalog as catalog
from merlin.experiments.options import parse_arguments
from merlin.experiments.runs import plan_execution as _merlin_plan_execution_1
from merlin.experiments.runs import plan_sources as runner
from merlin.experiments.runs import plans as _merlin_plans_1

# ---- cli ---------------------------------------------------------------------------------------

_LEGACY_OPTIONS = {
    "--phase0": "--testgen",
    "--phase0-bwrap": "--testgen-bwrap",
    "--phase0-capability-contract": "--testgen-capability-contract",
    "--phase0-capture-runtime-reuse": "--testgen-capture-runtime-reuse",
    "--phase0-capture-timeout-seconds": "--testgen-capture-timeout-seconds",
    "--phase0-component-coverage": "--testgen-component-coverage",
    "--phase0-conformance-spec": "--testgen-conformance-spec",
    "--phase0-evidence-mode": "--testgen-evidence-mode",
    "--phase0-generation-scope": "--testgen-generation-scope",
    "--phase0-hidden-profile": "--testgen-hidden-profile",
    "--phase0-m2m-python": "--testgen-m2m-python",
    "--phase0-m2m-root": "--testgen-m2m-root",
    "--phase0-rtl-facts": "--testgen-rtl-facts",
    "--phase0-rtl-source-relocations": "--testgen-rtl-source-relocations",
    "--phase0-synth-profile": "--testgen-synth-profile",
    "--phase1-driver": "--backend-driver",
    "--phase1-effort": "--backend-effort",
    "--phase1-model": "--backend-model",
    "--phase1-policy-descriptor": "--backend-policy-descriptor",
    "--phase1-provider": "--backend-provider",
}


def _present(value):
    """Show named stages without rewriting frozen engine records on disk."""
    if not isinstance(value, dict):
        return value
    result = dict(value)
    phases = result.get("phases")
    if isinstance(phases, dict):
        result.pop("phases")
        result["stages"] = {
            ENGINE_STAGES.get(str(number), str(number)): {"legacy_engine_id": str(number), **command}
            for number, command in phases.items()
        }
    if isinstance(result.get("spec"), dict) and result["spec"].get("schema_version") == 1:
        result["legacy_definition"] = result.pop("spec")
    if "attempts" in result:
        result["legacy_attempts"] = result.pop("attempts")
    for old, new in (("phase1_level", "backend_level"), ("phase0_synthesis", "test_generation")):
        if old in result:
            result[new] = result.pop(old)
    return result


def _source(value: str, catalog_path: Path | None) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path
    entries = catalog(catalog_path)
    if value not in entries:
        raise SpecError(f"unknown experiment {value!r}; use list or pass a definition path")
    return entries[value]


def _measured(call):
    """Run ``call`` with the whole-model measured mode's command module, its refusals as SpecErrors."""
    from merlin.development.backend.optimization.whole_model_measured import cli as measured

    try:
        return call(measured)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            raise SpecError(exc.code) from exc
        raise


def _measured_run(path: Path) -> Path | None:
    """``path`` as a whole-model measured run (itself, or the run an orchestration points at), or None."""
    try:
        return _measured(lambda measured: measured.resolve_run(path))
    except SpecError:
        return None


def _measured_status(run_dir: Path, *, stall_hours: float | None = None) -> dict:
    from merlin.development.backend.optimization.whole_model_measured import cli as measured
    from merlin.development.backend.optimization.whole_model_measured.runs import progress

    objective, error = measured.objective_or_error(run_dir)
    document = progress.run_status(run_dir, objective=objective, stall_hours=stall_hours)
    if error:
        document["objective_error"] = error
    return document


def _member_paths(values: list[str], flag: str) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for value in values:
        member, separator, location = value.partition("=")
        if not separator or not member or not location or member in selected:
            raise SpecError(f"invalid/duplicate {flag} {value!r}; use GUEST_MEMBER=PATH")
        selected[member] = Path(location).expanduser().absolute()
    return selected


def _full_capture_inputs(args: argparse.Namespace) -> dict:
    """Map the v2 (checkpoint) selection flags onto ``capture_selection.select`` keywords.

    Absent flags pass nothing, so a checkpoint-free v1 selection is unchanged. Each loader
    environment name is declared exactly once, present (``NAME=VALUE``) or absent (``NAME``).
    """
    selected: dict = {}
    if args.checkpoint is not None:
        ((member, path),) = _member_paths([args.checkpoint], "--checkpoint").items()
        selected.update(checkpoint=path, checkpoint_guest_member=member)
    if args.extra_input:
        selected["extra_inputs"] = _member_paths(args.extra_input, "--extra-input")
    environment: dict[str, str | None] = {}
    for value in args.loader_env:
        name, separator, setting = value.partition("=")
        if not separator or not name or name in environment:
            raise SpecError(f"invalid/duplicate --loader-env {value!r}; use NAME=VALUE")
        environment[name] = setting
    for name in args.loader_env_unset:
        if not name or "=" in name or name in environment:
            raise SpecError(f"invalid/duplicate --loader-env-unset {name!r}; use NAME")
        environment[name] = None
    if environment:
        selected["loader_env"] = environment
    if args.execution_timeout_seconds is not None:
        selected["execution_timeout_seconds"] = args.execution_timeout_seconds
    options = {}
    if args.stage_fp32:
        # A worker option: absent keeps the historical plan bytes.
        options["stage_fp32"] = True
    if args.cpu_threads is not None:
        options["cpu_threads"] = args.cpu_threads
    if options:
        selected["worker_options"] = options
    return selected


def _capture_sources() -> tuple[Path, Path]:
    """Keep a sealed capture's worker and schemas in the same Merlin installation.

    ``schemas_dir`` may prefer an ambient checkout even when ``merlin`` was
    imported from a wheel. That checkout is not part of the selected package,
    so the seal correctly refuses it. An explicit schema override is still
    passed through for the issuer to validate against the selected worker.
    """
    from merlin.common.paths import module_source_path, schemas_dir

    package = module_source_path("merlin").parent
    bundled = package / "_data/schemas"
    schemas = bundled if "MERLIN_SCHEMAS_DIR" not in os.environ and bundled.is_dir() else schemas_dir()
    return package / "capture/preparation/worker.py", schemas


def _dashboard_arguments(dashboard: argparse.ArgumentParser) -> None:
    """The richer dashboard views' options (test generation, explorer, comparison, operator records, live)."""
    dashboard.add_argument(
        "--testgen", dest="phase0", type=Path, help="a test generation derivation, generation run or corpus directory"
    )
    dashboard.add_argument("--explorer", action="store_true", help="with --target: every run, lineage and links")
    dashboard.add_argument(
        "--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"), help="two runs of the same phase side by side"
    )
    dashboard.add_argument("--monitor", type=Path, help="a monitor's notes (## <utc> headings, STATUS: lines)")
    dashboard.add_argument("--load", type=Path, help="host load samples (TSV: utc, load1, cpu_busy_pct, ...)")
    dashboard.add_argument("--corpus", type=Path, action="append", default=[], help="capsule corpus root(s)")
    dashboard.add_argument(
        "--measurement-root", type=Path, help="paired backend optimization: where cells are measured"
    )
    dashboard.add_argument("--stage-root", type=Path, help="paired backend optimization: where trials are authored")
    dashboard.add_argument(
        "--operator-private", action="store_true", help="name hidden capsules, held-out members, reference ratios"
    )
    dashboard.add_argument("--live", action="store_true", help="rewrite the page every --interval s and serve it")
    dashboard.add_argument("--interval", type=float, default=60.0, help="--live refresh period in seconds")
    dashboard.add_argument("--port", type=int, default=8765, help="--live port on 127.0.0.1")
    dashboard.add_argument("--cpus", help="--live CPU set, e.g. 24-31 (default: unpinned)")


def _dashboard_options(args: argparse.Namespace) -> dict:
    return {
        "run_dir": args.run_dir,
        "target": args.target,
        "store": args.store,
        "stall_hours": args.stall_hours,
        "phase0": args.phase0,
        "explorer": args.explorer or None,
        "compare": tuple(args.compare) if args.compare else None,
        "monitor": args.monitor,
        "load": args.load,
        "corpus": tuple(args.corpus) or None,
        "measurement_root": args.measurement_root,
        "stage_root": args.stage_root,
        "operator_private": args.operator_private or None,
    }


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    legacy_options = [value for value in raw if value.partition("=")[0] in _LEGACY_OPTIONS]
    raw = [
        _LEGACY_OPTIONS.get(value.partition("=")[0], value.partition("=")[0])
        + ("=" + value.partition("=")[2] if "=" in value else "")
        for value in raw
    ]
    if raw[:1] == ["export"]:
        from merlin.experiments.runs.catalogs import main as export_backend

        return export_backend(raw[1:])
    if raw[:1] == ["measured"]:
        # The whole-model measured mode's own command line, reachable from this one.
        from merlin.development.backend.optimization.whole_model_measured import cli as measured

        return measured.main(raw[1:])
    if raw[:1] == ["cell"]:
        from merlin.development.backend.optimization.whole_model_measured.scoring import cell_runs

        return cell_runs.main(raw[1:])
    if raw[:1] == ["census"]:
        # Exact, unpriced censuses of one observed execution, from records `inspect --trace` (locality)
        # or a target provider (call and stack boundaries) wrote.
        from merlin.analysis.optimization.candidates import census_cli

        return census_cli.main(raw[1:])
    if raw[:1] == ["instruction-admission"]:
        from merlin.development.backend.qualification.provenance import instruction_admission_preparation

        return instruction_admission_preparation.main(raw[1:])
    if raw[:1] == ["private-preflight"]:
        # The private full-model gate's compiler-independent input checks, before a launch.
        from merlin.development.backend.qualification.full_models import private_source_preflight

        return private_source_preflight.main(raw[1:])
    if raw[:1] == ["study"]:
        # A comparison study's register beside its run matrices, and the baseline its arms are scored on.
        from merlin.evaluation.harness import study_board as study_status

        return study_status.main(raw[1:])
    if raw[:1] == ["model-observe"]:
        from merlin.experiments.inspection.model_observation import main as observe_model

        return observe_model(raw[1:])
    if raw[:1] == ["chain-observe"]:
        from merlin.experiments.inspection.chain_observation import main as observe_chain

        return observe_chain(raw[1:])
    args, derive, evidence_custody, group_capsules = parse_arguments(
        raw=raw,
    )
    if args.verb == "inspect" and args.group is not None:
        # One group of a candidate (a measured job or a package), not an experiment definition.
        from merlin.experiments.inspection import group_inspect

        return group_inspect.run_from_args(args)
    try:
        if args.verb == "corpus":
            if args.operation == "evidence":
                result, code = evidence_custody.run(args)
                print(json.dumps(result, indent=2, sort_keys=True))
                return code
            if args.operation == "heldout-layers":
                from merlin.common.paths import repo_root
                from merlin.development.testgen.performance import heldout_layers as HL

                if args.output.absolute().is_relative_to(repo_root()):
                    raise SpecError("the held-out layer file is operator-private; write it outside the repository")
                document = HL.from_inventories(args.inventory)
                path = HL.write_private(document, args.output)
                print(json.dumps({"path": str(path), **HL.load(path).summary()}, indent=2))
                return 0
            if args.operation == "performance-demand":
                from merlin.development.testgen.performance.performance_corpus import derive_demand

                try:
                    result = derive_demand(
                        args.program,
                        policy=args.policy,
                        facts=args.rtl_facts,
                        template=args.template,
                        output=args.output,
                        existing_suite=args.existing_suite,
                    )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(_present(result), indent=2))
                return 0
            if args.operation == "chain-demand":
                from merlin.development.testgen.performance.chain_corpus import derive_chains

                try:
                    result = derive_chains(
                        args.program,
                        policy=args.policy,
                        facts=args.rtl_facts,
                        descriptor=args.descriptor,
                        recipe=args.recipe,
                        performance_template=args.performance_template,
                        software_spec=args.software_spec,
                        output=args.output,
                    )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(_present(result), indent=2))
                return 0
            if args.operation == "performance-epoch":
                from merlin.development.testgen.performance.performance_corpus import assemble_epoch

                try:
                    result = assemble_epoch(
                        args.previous,
                        plan=args.plan,
                        sources=args.sources,
                        output=args.output,
                        evidence_root=args.evidence,
                        target=args.target,
                        chain_plan=args.chain_plan,
                        chain_sources=args.chain_sources,
                    )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(_present(result), indent=2))
                return 0
            if args.operation == "stage-workloads":
                from merlin.development.testgen.performance.workload_roster import stage

                try:
                    result = stage(args.roster, args.output, descriptor=args.descriptor)
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(_present(result), indent=2))
                return 0
            if args.operation == "variants":
                from merlin.development.testgen.performance.workload_variants import materialize

                try:
                    result = materialize(
                        facts_path=args.facts,
                        template_path=args.template,
                        loader_path=args.loader,
                        output_root=args.output,
                    )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(_present(result), indent=2))
                return 0
            if args.operation == "capture":
                from merlin.development.testgen.sealed_capture import selection as capture_selection

                try:
                    if args.capture_operation == "select":
                        worker, schemas = _capture_sources()
                        result = capture_selection.select(
                            m2m_root=args.m2m_root,
                            workload_root=args.workload_root,
                            worker=worker,
                            venv=args.venv,
                            schemas_root=schemas,
                            run_dir=args.run_dir,
                            output_dir=args.output,
                            dtype=args.dtype,
                            recipe=args.recipe,
                            bwrap_binary=args.bwrap,
                            **_full_capture_inputs(args),
                        )
                    elif args.capture_operation == "attest":
                        result = capture_selection.attest(
                            args.selection, expected_sha256=args.expected_sha256, output=args.output
                        )
                    else:
                        if args.attestation_output is not None and (
                            args.attestation_output.exists() or not args.attestation_output.parent.is_dir()
                        ):
                            raise ValueError("capture attestation output must be fresh, under an existing parent")
                        if args.concurrent_replay and args.attestation_output is None:
                            raise ValueError("--concurrent-replay needs --attestation-output")
                        if args.concurrent_replay:
                            from merlin.experiments.capture_execution import m2m_sandbox as sealed_m2m

                            result = {
                                "phase0_admission": "not_granted",
                                **capture_selection.issue_and_attest(
                                    args.selection,
                                    expected_sha256=args.expected_sha256,
                                    output=args.attestation_output,
                                    cpus=sealed_m2m.replay_cpu_sets(os.sched_getaffinity(0)),
                                ),
                            }
                        else:
                            receipt = capture_selection.issue(args.selection, expected_sha256=args.expected_sha256)
                            result = {"sealed_receipt": str(receipt), "phase0_admission": "not_granted"}
                            if args.attestation_output is not None:
                                result.update(
                                    capture_selection.attest(
                                        args.selection,
                                        expected_sha256=args.expected_sha256,
                                        output=args.attestation_output,
                                    )
                                )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(_present(result), indent=2))
                return 0
            elif args.operation == "derive":
                from merlin.development.testgen.requirement_inputs import (
                    capture_selection_specs,
                    capture_selections,
                    quantization_policy_specs,
                )
                from merlin.development.testgen.requirements import derive

                try:
                    result = derive(
                        _source(args.definition, args.catalog),
                        capture_selections(args.application_capture),
                        rtl_facts=args.rtl_facts,
                        output_root=args.output,
                        native_qualifications=capture_selections(args.native_qualification),
                        capture_preselections=capture_selection_specs(args.application_capture_selection),
                        quantization_policies=quantization_policy_specs(args.application_quant_policy),
                        performance_captures=capture_selections(args.performance_capture),
                        performance_preselections=capture_selection_specs(args.performance_capture_selection),
                        heldout_layers=args.heldout_layer_shapes,
                    )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(_present(result), indent=2))
                return 0
            if args.operation == "groups":
                return group_capsules.run_from_args(args)
            if args.operation == "compare":
                from merlin.development.testgen.profiles.screens import build

                print(json.dumps(build([_source(value, args.catalog) for value in args.definitions]), indent=2))
                return 0
            if args.operation == "coverage":
                from merlin.experiments.corpus.coverage import inspect_run

                result = inspect_run(args.run_dir, args.spec)
                print(json.dumps(_present(result), indent=2))
                return 0
            from merlin.experiments.corpus import release_binding as corpus_release

            if args.operation == "prepare":
                result = corpus_release.prepare(
                    args.run_dir,
                    args.output,
                    private_baseline=args.private_baseline,
                    retirements=args.retirements,
                    generated_only=args.generated_only,
                    phase1_policy_descriptor=args.phase1_policy_descriptor,
                    optional_levels=tuple(args.with_level),
                    lean_toolchain_manifest=args.lean_toolchain_manifest,
                    **({"common_substrate": args.common_substrate} if args.common_substrate is not None else {}),
                    **({"python_source_root": args.python_source_root} if args.python_source_root is not None else {}),
                    host_index_bits=args.host_index_bits,
                    **({"public_scalability": args.public_scalability} if args.public_scalability is not None else {}),
                    **({"public_supplement": args.public_supplement} if args.public_supplement is not None else {}),
                )
            elif args.operation == "inspect":
                result = corpus_release.inspect_release(args.release)
            else:
                result = corpus_release.seal(
                    args.release,
                    expected_digest=args.expected_digest,
                    reviewed_by=args.reviewed_by,
                    review_note=args.review_note,
                )
        elif args.verb == "list":
            from merlin.development.backend.session.treatments import level_for_phase1

            result = []
            for name, path in catalog(args.catalog).items():
                spec = load_spec(path)
                if spec.id != name:
                    raise SpecError(f"catalog id {name!r} differs from definition id {spec.id!r}")
                result.append(
                    {
                        "id": name,
                        "target": spec.target,
                        "stages": [ENGINE_STAGES[number] for number in sorted(engine_definitions(spec.document))],
                        "legacy_definition": spec.document["schema_version"] == 1,
                        "definition": str(path),
                        "description": spec.document.get("description", ""),
                        "kind": spec.document.get("kind", "experiment"),
                        "backend_level": (
                            level_for_phase1(engine_definitions(spec.document)["1"]["config"])
                            if "1" in engine_definitions(spec.document)
                            else None
                        ),
                    }
                )
        elif args.verb == "levels":
            from merlin.development.backend.session.treatments import LEVELS

            result = list(LEVELS)
        elif args.verb == "runs":
            from merlin.experiments.runs.catalogs import runs

            result = runs(root=args.root, target=args.target, experiment=args.experiment)
        elif args.verb == "status":
            measured_run = None if (args.run_dir / "orchestration.json").is_file() else _measured_run(args.run_dir)
            result = (
                _measured_status(measured_run, stall_hours=args.stall_hours)
                if measured_run is not None
                else _merlin_plans_1.status(args.run_dir)
            )
        elif args.verb == "stop":
            result = _measured(lambda measured: measured.stop(args.run_dir, why=args.why))
        elif args.verb == "lineage":
            from merlin.development.targetgen.packaging import target_index
            from merlin.experiments.runs.catalogs import lineage

            if (args.run_dir is None) == (args.target is None):
                raise SpecError("lineage needs exactly one of a run directory or --target")
            if args.target is not None:
                result = target_index.build_index(args.target)
            else:
                result = lineage(args.run_dir)
                result["target_index"] = target_index.rows_citing(result["target"], result["run_dir"])
        elif args.verb == "index":
            from merlin.development.targetgen.packaging import target_index

            if args.check:
                current = target_index.is_current(args.target)
                print(json.dumps({"index": str(target_index.index_path(args.target)), "current": current}))
                return 0 if current else 1
            result = {"index": str(target_index.write_index(args.target))}
        elif args.verb == "dashboard":
            from merlin.experiments.tracking import serve_dashboard, write_dashboard

            options = _dashboard_options(args)
            if args.live:
                from merlin.experiments.tracking import destination_for

                out = args.out or destination_for(**options)
                return serve_dashboard(out=out, interval=args.interval, port=args.port, cpus=args.cpus, **options)
            result = write_dashboard(out=args.out, **options)
            if args.open:
                import webbrowser

                webbrowser.open(Path(result["dashboard"]).resolve().as_uri())
        elif args.verb == "watch":
            if (args.task is None) == (args.run_dir is None):
                raise SpecError("watch requires one run directory or --task TASK_DIR")
            if args.task is not None:
                from merlin.experiments.tracking.terminal.watch import main as terminal_main

                options = ["--task", str(args.task), "--window", str(args.window)]
                for flag, value in (("--interval", args.interval), ("--width", args.width), ("--height", args.height)):
                    if value is not None:
                        options += [flag, str(value)]
                for flag, enabled in (
                    ("--operator-private", args.operator_private),
                    ("--once", args.once),
                    ("--no-color", args.no_color),
                    ("--legacy", args.legacy),
                ):
                    if enabled:
                        options.append(flag)
                return terminal_main(options)
            from merlin.experiments.tracking import watch as watch_run

            return watch_run(
                args.run_dir,
                interval=30.0 if args.interval is None else args.interval,
                once=args.once,
                store=args.store,
                stall_hours=args.stall_hours,
                colour=False if args.no_color else None,
            )
        elif args.verb == "console":
            from merlin.experiments.tracking.console.cli import run as run_console

            return run_console(args)
        elif args.verb == "resume":
            code = _merlin_plan_execution_1.resume(args.run_dir)
            print(json.dumps(_present(_merlin_plans_1.status(args.run_dir)), indent=2))
            return code
        else:
            spec = load_spec(_source(args.spec, args.catalog))
            if args.phase is not None and args.workflow is not None:
                raise SpecError("select --workflow; do not combine it with a legacy --phase selector")
            if spec.document["schema_version"] == 2 and (args.phase is not None or legacy_options):
                raise SpecError(
                    "named definitions use --workflow, --testgen-* and --backend-*; numeric phase flags are legacy-only"
                )
            plan = _merlin_plans_1.resolve_plan(
                spec,
                phase=args.phase,
                workflow=args.workflow if args.phase is None else None,
                run_dir=args.run_dir,
                corpus_seal=args.corpus_seal,
                bundle_manifest=args.bundle_manifest,
                backend_joint_campaign=args.backend_joint_campaign,
                backend_oracle_timing=args.backend_oracle_timing,
                phase1_driver=args.phase1_driver,
                phase1_model=args.phase1_model,
                phase1_effort=args.phase1_effort,
                phase1_provider=args.phase1_provider,
                phase0_conformance_spec=args.phase0_conformance_spec,
                phase0_capability_contract=args.phase0_capability_contract,
                phase0_synth_profile=args.phase0_synth_profile,
                phase0_rtl_facts=args.phase0_rtl_facts,
                phase0_rtl_source_relocations=args.phase0_rtl_source_relocations,
                phase0_evidence_mode=args.phase0_evidence_mode,
                phase0_hidden_profile=args.phase0_hidden_profile,
                phase0_generation_scope=args.phase0_generation_scope,
                phase0_component_coverage=args.phase0_component_coverage,
                phase0_m2m_root=args.phase0_m2m_root,
                phase0_m2m_python=args.phase0_m2m_python,
                phase0_capture_timeout_seconds=args.phase0_capture_timeout_seconds,
                phase0_capture_runtime_reuse=args.phase0_capture_runtime_reuse,
                phase0_bwrap=args.phase0_bwrap,
            )
            if args.verb == "inspect":
                result = plan
            elif args.verb == "preflight":
                result = _merlin_plans_1.preflight(plan)
                print(json.dumps(_present(result), indent=2))
                return 0 if result["configuration_ready"] else 2
            else:
                code = _merlin_plans_1.run(plan)
                print(json.dumps(_present(_merlin_plans_1.status(Path(plan["run_dir"]))), indent=2))
                return code
        print(json.dumps(_present(result), indent=2))
        return 0
    except (SpecError, OSError) as exc:
        print(f"merlin experiment: {exc}", file=sys.stderr)
        return 2
