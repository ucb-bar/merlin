"""One discoverable command surface for phase definitions and existing engines."""

from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:1] == ["measured"]:
        # The whole-model measured mode's own command line, reachable from this one.
        from .phase2.whole_model_measured import cli as measured

        return measured.main(raw[1:])
    if raw[:1] == ["cell"]:
        from .phase2.whole_model_measured import cell_runs

        return cell_runs.main(raw[1:])
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
            "--phase0-m2m-python", type=Path, help="explicit Model2MLIR venv Python for diagnostic capture"
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
    issue_capture = capture_ops.add_parser("issue", help="capture only from an exact preselected identity")
    issue_capture.add_argument("--selection", type=Path, required=True)
    issue_capture.add_argument("--expected-sha256", required=True)
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
            if args.operation == "capture":
                from merlin.common.paths import module_source_path, schemas_dir

                from .phase0 import capture_selection

                try:
                    if args.capture_operation == "select":
                        result = capture_selection.select(
                            m2m_root=args.m2m_root,
                            workload_root=args.workload_root,
                            worker=module_source_path("merlin").parent / "targetgen/_m2m_capture_worker.py",
                            venv=args.venv,
                            schemas_root=schemas_dir(),
                            run_dir=args.run_dir,
                            output_dir=args.output,
                            dtype=args.dtype,
                            recipe=args.recipe,
                            bwrap_binary=args.bwrap,
                        )
                    else:
                        receipt = capture_selection.issue(args.selection, expected_sha256=args.expected_sha256)
                        result = {"sealed_receipt": str(receipt), "phase0_admission": "not_granted"}
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
            from .tracking import write_dashboard

            result = write_dashboard(
                run_dir=args.run_dir,
                target=args.target,
                out=args.out,
                store=args.store,
                stall_hours=args.stall_hours,
            )
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
                phase0_m2m_root=args.phase0_m2m_root,
                phase0_m2m_python=args.phase0_m2m_python,
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
