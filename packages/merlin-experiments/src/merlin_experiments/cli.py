"""One discoverable command surface for phase definitions and existing engines."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import runner
from .spec import SpecError, load_spec
from .spec import catalog as catalog


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
    from .phase2.whole_model_measured import cli as measured

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
    from .phase2.whole_model_measured import cli as measured
    from .phase2.whole_model_measured import progress

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
    if args.stage_fp32:
        # A worker option: absent keeps the historical plan bytes.
        selected["worker_options"] = {"stage_fp32": True}
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
    return package / "targetgen/_m2m_capture_worker.py", schemas


def _dashboard_arguments(dashboard: argparse.ArgumentParser) -> None:
    """The richer dashboard views' options (Phase 0, explorer, comparison, operator records, live)."""
    dashboard.add_argument("--phase0", type=Path, help="a Phase 0 derivation, generation run or corpus directory")
    dashboard.add_argument("--explorer", action="store_true", help="with --target: every run, lineage and links")
    dashboard.add_argument(
        "--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"), help="two runs of the same phase side by side"
    )
    dashboard.add_argument("--monitor", type=Path, help="a monitor's notes (## <utc> headings, STATUS: lines)")
    dashboard.add_argument("--load", type=Path, help="host load samples (TSV: utc, load1, cpu_busy_pct, ...)")
    dashboard.add_argument("--corpus", type=Path, action="append", default=[], help="capsule corpus root(s)")
    dashboard.add_argument("--measurement-root", type=Path, help="paired Phase 2: where cells are measured")
    dashboard.add_argument("--stage-root", type=Path, help="paired Phase 2: where trials are authored")
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
    if raw[:1] == ["measured"]:
        # The whole-model measured mode's own command line, reachable from this one.
        from .phase2.whole_model_measured import cli as measured

        return measured.main(raw[1:])
    if raw[:1] == ["cell"]:
        from .phase2.whole_model_measured import cell_runs

        return cell_runs.main(raw[1:])
    if raw[:1] == ["census"]:
        # Exact, unpriced censuses of one observed execution, from records `inspect --trace` (locality)
        # or a target provider (call and stack boundaries) wrote.
        from merlin.perf import census_cli

        return census_cli.main(raw[1:])
    if raw[:1] == ["study"]:
        # A comparison study's register beside its run matrices, and the baseline its arms are scored on.
        from merlin.benchharness import study_status

        return study_status.main(raw[1:])
    parser = argparse.ArgumentParser(prog="merlin experiment", description=__doc__)
    parser.add_argument("--catalog", type=Path, help="catalog YAML; paths inside it are relative to that file")
    commands = parser.add_subparsers(dest="verb", required=True)
    commands.add_parser("list", help="list the versioned experiment catalog")
    commands.add_parser("levels", help="show Phase 1 experiment levels and their stable bundle-arm ids")
    stored = commands.add_parser("runs", help="discover stored phase orchestrations (read-only)")
    stored.add_argument("--root", type=Path, help="run root; defaults to the configured out/runs")
    stored.add_argument("--target", help="filter by exact target identity")
    stored.add_argument("--experiment", help="filter by exact experiment identity")
    for verb in ("inspect", "preflight", "run"):
        child = commands.add_parser(verb)
        child.add_argument(
            "spec",
            help="definition path or catalog id"
            + ("; with --group, a measured job directory or a package directory" if verb == "inspect" else ""),
        )
        child.add_argument("--phase", choices=("0", "1", "2", "all"), default="all")
        child.add_argument("--run-dir", type=Path, help="explicit output; otherwise use the configured run root")
        child.add_argument("--corpus-seal", type=Path, help="reviewed Phase 0 release seal for Phase 1")
        child.add_argument("--bundle-manifest", type=Path, help="reviewed replacement Phase 1 input bundle")
        child.add_argument("--phase1-driver", help="Phase 1 agent driver; frozen with the selected run")
        child.add_argument("--phase1-model", help="Phase 1 model; frozen with the selected run")
        child.add_argument("--phase1-effort", help="Phase 1 reasoning effort; frozen with the selected run")
        child.add_argument("--phase1-provider", help="Phase 1 provider; frozen with the selected run")
        child.add_argument(
            "--phase0-conformance-spec", type=Path, help="new reviewed Phase 0 requirement (select with synth profile)"
        )
        child.add_argument(
            "--phase0-synth-profile", type=Path, help="new synthesized Phase 0 profile (select with requirement)"
        )
        child.add_argument(
            "--phase0-hidden-profile",
            type=Path,
            help="operator-owned private Phase 0 profile; never put it in examples",
        )
        child.add_argument(
            "--phase0-component-coverage",
            type=Path,
            help="explicit reviewed private independent component coverage plan",
        )
        child.add_argument("--phase0-rtl-facts", type=Path, help="select exact extracted facts for a new Phase 0 run")
        child.add_argument(
            "--phase0-capability-contract",
            type=Path,
            help="select exact same-target capability contract for a new Phase 0 run",
        )
        child.add_argument(
            "--phase0-evidence-mode",
            choices=("diagnostic", "verified"),
            help="diagnostic preserves unknowns; verified refuses unresolved required evidence",
        )
        child.add_argument(
            "--phase0-m2m-root", type=Path, help="explicit Model2MLIR source root for diagnostic capture"
        )
        child.add_argument(
            "--phase0-capture-timeout-seconds",
            type=int,
            help="sandbox timeout (120..14400 s) frozen into the plan for every sealed generation-time "
            "capture; default keeps the historical fixed 120 s",
        )
        child.add_argument(
            "--phase0-m2m-python", type=Path, help="explicit Model2MLIR venv Python for diagnostic capture"
        )
        child.add_argument(
            "--phase0-bwrap",
            type=Path,
            help="absolute bubblewrap binary frozen into the plan for every sealed generation-time capture; "
            "default resolves the system bwrap",
        )
        if verb == "inspect":
            from . import group_inspect

            group_inspect.configure_parser(child)
    status = commands.add_parser(
        "status", help="an orchestration's phases, or a whole-model measured run's status from its records"
    )
    status.add_argument("run_dir", type=Path)
    status.add_argument(
        "--stall-hours", type=float, help="a measured run with no candidate measured for this long is STALLED"
    )
    commands.add_parser(
        "measured",
        help="the whole-model measured mode's own commands: `merlin experiment measured --help`",
        add_help=False,
    )
    commands.add_parser(
        "cell",
        help="a cell run: `cell prepare <loop run> --cell ID`, `cell launch <run> --profile P`, `cell status`",
        add_help=False,
    )
    commands.add_parser(
        "census",
        help="exact censuses of one observed execution: `census locality|boundaries|boundary-domain`",
        add_help=False,
    )
    commands.add_parser(
        "study",
        help="a comparison study, read-only: `study board` (register vs run matrices), `study baseline`",
        add_help=False,
    )
    stop = commands.add_parser(
        "stop", help="ask a whole-model measured run to stop at its next session boundary (signals nothing)"
    )
    stop.add_argument("run_dir", type=Path, help="the measured run, or an orchestration run that points at one")
    stop.add_argument("--why", required=True, help="recorded with the request and as the run's stop reason")
    lineage_parser = commands.add_parser(
        "lineage", help="read frozen phase inputs and handoffs without executing engines"
    )
    lineage_parser.add_argument("run_dir", type=Path, nargs="?")
    lineage_parser.add_argument(
        "--target", help="print the target's index (phase-0 releases, frozen compilers, champions) instead"
    )
    index = commands.add_parser(
        "index", help="regenerate out/artifacts/targets/<target>/INDEX.yaml from existing records"
    )
    index.add_argument("target")
    index.add_argument("--check", action="store_true", help="exit 1 when the written index is stale; write nothing")
    from .tracking.records import DEFAULT_STALL_HOURS

    dashboard = commands.add_parser(
        "dashboard",
        help="write one self-contained HTML view of a run or a target, read from existing records only",
    )
    dashboard.add_argument("run_dir", type=Path, nargs="?", help="a run directory (orchestration, phase 1 or 2)")
    dashboard.add_argument("--target", help="every run of this target across phases, with champion lineage")
    dashboard.add_argument(
        "--out", type=Path, help="HTML file; defaults to out/artifacts/experiments/<target>/dashboard/<run>.html"
    )
    dashboard.add_argument("--open", action="store_true", help="also open the written page in a browser")
    _dashboard_arguments(dashboard)
    watch = commands.add_parser("watch", help="live terminal view of a run's records; refreshes until Ctrl-C")
    watch.add_argument("run_dir", type=Path)
    watch.add_argument("--interval", type=float, default=30.0, help="seconds between refreshes")
    watch.add_argument("--once", action="store_true", help="print once and exit")
    watch.add_argument("--no-color", action="store_true", help="plain text even on a terminal")
    for view in (dashboard, watch):
        view.add_argument(
            "--store", type=Path, help="phase-2 measurement store when the run records none (store_roots.screen)"
        )
        view.add_argument(
            "--stall-hours",
            type=float,
            default=DEFAULT_STALL_HOURS,
            help="hours without a measured candidate (or grade) before a run is STALLED",
        )
    child = commands.add_parser("resume")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--checkpoint", type=Path, help="sealed native checkpoint for a new model_portfolio segment")
    corpus = commands.add_parser(
        "corpus", help="derive capsule groups, inspect run coverage, or prepare and review a corpus release"
    )
    operations = corpus.add_subparsers(dest="operation", required=True)
    derive = operations.add_parser("derive", help="deterministic requirements and complete census; no agent execution")
    derive.add_argument("definition", help="explicit experiment definition or catalog id")
    derive.add_argument("--application-capture", action="append", required=True, metavar="LABEL=PATH")
    derive.add_argument(
        "--application-capture-selection",
        action="append",
        default=[],
        metavar="LABEL=PATH@SHA256",
        help="pre-execution selection for each selected capture; omitted legacy captures remain diagnostic",
    )
    derive.add_argument(
        "--application-quant-policy",
        action="append",
        default=[],
        metavar="LABEL=PATH@SHA256",
        help="independently selected policy bytes for each externally quantized capture",
    )
    derive.add_argument(
        "--performance-capture",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="capture of a declared workload_spec.performance_applications member; feeds only the Phase 2 form scope",
    )
    derive.add_argument(
        "--performance-capture-selection",
        action="append",
        default=[],
        metavar="LABEL=PATH@SHA256",
        help="pre-execution selection for each performance-scale capture (all or none)",
    )
    derive.add_argument(
        "--heldout-layer-shapes",
        type=Path,
        help="OPERATOR-PRIVATE merlin.heldout_layer_shapes.v1 file (owner-only, outside the repository): "
        "refuse any derived form member whose contraction equals a held-out network layer",
    )
    derive.add_argument(
        "--native-qualification",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="optional exact generated native-host qualification receipt; never grants RVV support",
    )
    derive.add_argument(
        "--rtl-facts", type=Path, required=True, help="exact extraction artifact; never re-extract implicitly"
    )
    derive.add_argument("--output", type=Path, required=True, help="new immutable artifact root")
    capture = operations.add_parser("capture", help="preselect and issue one fresh sealed CPU capture")
    capture_ops = capture.add_subparsers(dest="capture_operation", required=True)
    select_capture = capture_ops.add_parser("select", help="freeze source/runtime/tool bytes before capture")
    select_capture.add_argument("--m2m-root", type=Path, required=True)
    select_capture.add_argument("--workload-root", type=Path, required=True)
    select_capture.add_argument("--venv", type=Path, required=True)
    select_capture.add_argument("--dtype", choices=("fp32", "int8"), default="fp32")
    select_capture.add_argument("--recipe", type=Path)
    select_capture.add_argument("--run-dir", type=Path, required=True)
    select_capture.add_argument("--output", type=Path, required=True, help="fresh owner-only selection directory")
    select_capture.add_argument("--bwrap", type=Path)
    select_capture.add_argument(
        "--checkpoint",
        metavar="GUEST_MEMBER=PATH",
        help="explicit checkpoint file or tree (v2 full-model selection); the loader reads it at GUEST_MEMBER "
        "under its read-only input root",
    )
    select_capture.add_argument(
        "--extra-input",
        action="append",
        default=[],
        metavar="GUEST_MEMBER=PATH",
        help="additional selected input file or tree (v2 only; repeatable)",
    )
    select_capture.add_argument(
        "--loader-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="declared loader environment value (v2 only; repeatable); every literal loader read must be declared",
    )
    select_capture.add_argument(
        "--loader-env-unset",
        action="append",
        default=[],
        metavar="NAME",
        help="loader environment read selected as deliberately absent (v2 only; repeatable)",
    )
    select_capture.add_argument(
        "--stage-fp32",
        action="store_true",
        help="stage the exported program to FP32 before capture (and before an int8 recipe quantizes it), "
        "with the worker's exact-source precision audit; needed for models that compute in bf16/fp16",
    )
    select_capture.add_argument(
        "--execution-timeout-seconds",
        type=int,
        help="bounded sandbox execution time, used by issue and by the attestation replay: a checkpoint "
        "selection requires 120..43200; a checkpoint-free selection may select 120..14400 (default: the "
        "historical fixed 120 s, recorded nowhere so old selections keep their bytes)",
    )
    issue_capture = capture_ops.add_parser("issue", help="capture only from an exact preselected identity")
    issue_capture.add_argument("--selection", type=Path, required=True)
    issue_capture.add_argument("--expected-sha256", required=True)
    issue_capture.add_argument(
        "--attestation-output",
        type=Path,
        help="after issuing, replay the capture and write its sealed execution attestation here "
        "(v3 for a v2 checkpoint selection) as a fresh owner-only file",
    )
    attest_capture = capture_ops.add_parser(
        "attest", help="replay an issued preselected capture and write its sealed execution attestation"
    )
    attest_capture.add_argument("--selection", type=Path, required=True)
    attest_capture.add_argument("--expected-sha256", required=True)
    attest_capture.add_argument("--output", type=Path, required=True, help="fresh attestation JSON path")
    heldout = operations.add_parser(
        "heldout-layers",
        help="OPERATOR: write the private held-out layer-shape file from per-program inventories",
    )
    heldout.add_argument("--inventory", type=Path, action="append", required=True)
    heldout.add_argument("--output", type=Path, required=True, help="fresh owner-only file outside the repository")
    staging = operations.add_parser(
        "stage-workloads",
        help="write a fresh workload root (loader and optional profile) per declared roster label",
    )
    staging.add_argument("--roster", type=Path, required=True, help="merlin.phase0_workload_roster.v1 file")
    staging.add_argument("--output", type=Path, required=True, help="fresh capture-input directory")
    staging.add_argument(
        "--descriptor", type=Path, help="require the roster to stage exactly the descriptor's declared labels"
    )
    variants = operations.add_parser(
        "variants", help="derive bounded independent workload roots from selected RTL facts"
    )
    variants.add_argument("--facts", type=Path, required=True)
    variants.add_argument("--template", type=Path, required=True)
    variants.add_argument("--loader", type=Path, required=True)
    variants.add_argument("--output", type=Path, required=True, help="fresh generated capture-input directory")
    from merlin.targetgen import group_capsules

    groups = operations.add_parser(
        "groups",
        help="derive capsules from captured compute groups (does not seal or approve)",
        description=group_capsules.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group_capsules.configure_parser(groups)
    compare = operations.add_parser("compare", help="compare public recipes from explicit definitions; prints JSON")
    compare.add_argument("definitions", nargs="+", help="definition paths or catalog ids")
    coverage = operations.add_parser("coverage", help="inspect public coverage of one completed Phase 0 run")
    coverage.add_argument("run_dir", type=Path)
    coverage.add_argument("--spec", type=Path, required=True, help="explicit conformance requirement YAML")
    prepare = operations.add_parser("prepare")
    prepare.add_argument("run_dir", type=Path)
    prepare.add_argument(
        "--output",
        type=Path,
        help="fresh release root; defaults to out/artifacts/protocols/<target>/phase0-<TS>-<sha7>",
    )
    prepare.add_argument(
        "--generated-only",
        action="store_true",
        help="use only the selected Phase-0 output for public capsules; never read the historical corpus",
    )
    prepare.add_argument(
        "--phase1-policy-descriptor",
        type=Path,
        help="explicit Phase-1 gate policy; all other descriptor fields must match the frozen Phase-0 source",
    )
    prepare.add_argument(
        "--private-baseline",
        type=Path,
        help="operator-owned hidden capsule category when it is absent from the public source checkout",
    )
    prepare.add_argument(
        "--retirements",
        type=Path,
        help="reviewed public baseline-generated members intentionally retired by this derivation",
    )
    operations.add_parser("inspect").add_argument("release", type=Path)
    seal = operations.add_parser("seal")
    seal.add_argument("release", type=Path)
    seal.add_argument("--expected-digest", required=True)
    seal.add_argument("--reviewed-by", required=True)
    seal.add_argument("--review-note", required=True)
    args = parser.parse_args(raw)
    if args.verb == "inspect" and args.group is not None:
        # One group of a candidate (a measured job or a package), not an experiment definition.
        from . import group_inspect

        return group_inspect.run_from_args(args)
    try:
        if args.verb == "corpus":
            if args.operation == "heldout-layers":
                from merlin.common.paths import repo_root

                from .phase0 import heldout_layers as HL

                if args.output.absolute().is_relative_to(repo_root()):
                    raise SpecError("the held-out layer file is operator-private; write it outside the repository")
                document = HL.from_inventories(args.inventory)
                path = HL.write_private(document, args.output)
                print(json.dumps({"path": str(path), **HL.load(path).summary()}, indent=2))
                return 0
            if args.operation == "stage-workloads":
                from .phase0.workload_roster import stage

                try:
                    result = stage(args.roster, args.output, descriptor=args.descriptor)
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(result, indent=2))
                return 0
            if args.operation == "variants":
                from .phase0.workload_variants import materialize

                try:
                    result = materialize(
                        facts_path=args.facts,
                        template_path=args.template,
                        loader_path=args.loader,
                        output_root=args.output,
                    )
                except ValueError as exc:
                    raise SpecError(str(exc)) from exc
                print(json.dumps(result, indent=2))
                return 0
            if args.operation == "capture":
                from .phase0 import capture_selection

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
                print(json.dumps(result, indent=2))
                return 0
            elif args.operation == "derive":
                from .phase0.requirements import (
                    capture_selection_specs,
                    capture_selections,
                    derive,
                    quantization_policy_specs,
                )

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
                print(json.dumps(result, indent=2))
                return 0
            if args.operation == "groups":
                return group_capsules.run_from_args(args)
            if args.operation == "compare":
                from .phase0.comparison import build

                print(json.dumps(build([_source(value, args.catalog) for value in args.definitions]), indent=2))
                return 0
            if args.operation == "coverage":
                from .corpus.coverage import inspect_run

                result = inspect_run(args.run_dir, args.spec)
                print(json.dumps(result, indent=2))
                return 0
            from .corpus import release as corpus_release

            if args.operation == "prepare":
                result = corpus_release.prepare(
                    args.run_dir,
                    args.output,
                    private_baseline=args.private_baseline,
                    retirements=args.retirements,
                    generated_only=args.generated_only,
                    phase1_policy_descriptor=args.phase1_policy_descriptor,
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
            from .phase1.levels import level_for_phase1

            result = []
            for name, path in catalog(args.catalog).items():
                spec = load_spec(path)
                if spec.id != name:
                    raise SpecError(f"catalog id {name!r} differs from definition id {spec.id!r}")
                result.append(
                    {
                        "id": name,
                        "target": spec.target,
                        "phases": sorted(spec.document["phases"]),
                        "definition": str(path),
                        "description": spec.document.get("description", ""),
                        "kind": spec.document.get("kind", "experiment"),
                        "phase1_level": (
                            level_for_phase1(spec.document["phases"]["1"]["config"])
                            if "1" in spec.document["phases"]
                            else None
                        ),
                    }
                )
        elif args.verb == "levels":
            from .phase1.levels import LEVELS

            result = list(LEVELS)
        elif args.verb == "runs":
            from .history import runs

            result = runs(root=args.root, target=args.target, experiment=args.experiment)
        elif args.verb == "status":
            measured_run = None if (args.run_dir / "orchestration.json").is_file() else _measured_run(args.run_dir)
            result = (
                _measured_status(measured_run, stall_hours=args.stall_hours)
                if measured_run is not None
                else runner.status(args.run_dir)
            )
        elif args.verb == "stop":
            result = _measured(lambda measured: measured.stop(args.run_dir, why=args.why))
        elif args.verb == "lineage":
            from merlin.targetgen import target_index

            from .history import lineage

            if (args.run_dir is None) == (args.target is None):
                raise SpecError("lineage needs exactly one of a run directory or --target")
            if args.target is not None:
                result = target_index.build_index(args.target)
            else:
                result = lineage(args.run_dir)
                result["target_index"] = target_index.rows_citing(result["target"], result["run_dir"])
        elif args.verb == "index":
            from merlin.targetgen import target_index

            if args.check:
                current = target_index.is_current(args.target)
                print(json.dumps({"index": str(target_index.index_path(args.target)), "current": current}))
                return 0 if current else 1
            result = {"index": str(target_index.write_index(args.target))}
        elif args.verb == "dashboard":
            from .tracking import serve_dashboard, write_dashboard

            options = _dashboard_options(args)
            if args.live:
                from .tracking import destination_for

                out = args.out or destination_for(**options)
                return serve_dashboard(out=out, interval=args.interval, port=args.port, cpus=args.cpus, **options)
            result = write_dashboard(out=args.out, **options)
            if args.open:
                import webbrowser

                webbrowser.open(Path(result["dashboard"]).resolve().as_uri())
        elif args.verb == "watch":
            from .tracking import watch as watch_run

            return watch_run(
                args.run_dir,
                interval=args.interval,
                once=args.once,
                store=args.store,
                stall_hours=args.stall_hours,
                colour=False if args.no_color else None,
            )
        elif args.verb == "resume":
            code = runner.resume(args.run_dir, checkpoint=args.checkpoint)
            print(json.dumps(runner.status(args.run_dir), indent=2))
            return code
        else:
            spec = load_spec(_source(args.spec, args.catalog))
            plan = runner.resolve_plan(
                spec,
                phase=args.phase,
                run_dir=args.run_dir,
                corpus_seal=args.corpus_seal,
                bundle_manifest=args.bundle_manifest,
                phase1_driver=args.phase1_driver,
                phase1_model=args.phase1_model,
                phase1_effort=args.phase1_effort,
                phase1_provider=args.phase1_provider,
                phase0_conformance_spec=args.phase0_conformance_spec,
                phase0_capability_contract=args.phase0_capability_contract,
                phase0_synth_profile=args.phase0_synth_profile,
                phase0_rtl_facts=args.phase0_rtl_facts,
                phase0_evidence_mode=args.phase0_evidence_mode,
                phase0_hidden_profile=args.phase0_hidden_profile,
                phase0_component_coverage=args.phase0_component_coverage,
                phase0_m2m_root=args.phase0_m2m_root,
                phase0_m2m_python=args.phase0_m2m_python,
                phase0_capture_timeout_seconds=args.phase0_capture_timeout_seconds,
                phase0_bwrap=args.phase0_bwrap,
            )
            if args.verb == "inspect":
                result = plan
            elif args.verb == "preflight":
                result = runner.preflight(plan)
                print(json.dumps(result, indent=2))
                return 0 if result["configuration_ready"] else 2
            else:
                code = runner.run(plan)
                print(json.dumps(runner.status(Path(plan["run_dir"])), indent=2))
                return code
        print(json.dumps(result, indent=2))
        return 0
    except (SpecError, OSError) as exc:
        print(f"merlin experiment: {exc}", file=sys.stderr)
        return 2
