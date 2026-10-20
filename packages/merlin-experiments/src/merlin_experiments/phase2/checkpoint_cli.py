"""Explicit-resource CLI for the installed checkpoint experiment controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import checkpoint_controller as CTRL
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT


def main(argv: list[str] | None = None, *, resolve_context=None, invocation: tuple[str, ...] | None = None) -> int:
    original_args = list(sys.argv[1:] if argv is None else argv)
    invocation = invocation or (sys.executable, "-m", "merlin_experiments.phase2.checkpoint_cli", *original_args)
    parser = argparse.ArgumentParser()
    resource_names = (
        "source_root",
        "contract_root",
        "functional_runs_root",
        "stage_root",
        "measurement_root",
        "holdout_catalog",
        "core_package_root",
        "experiments_package_root",
        "experiments_namespace_root",
        "chia_wrapper",
    )
    for name in resource_names:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=resolve_context is None)
    parser.add_argument("--suite", required=resolve_context is None)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--functional-run-id", required=True)
    parser.add_argument(
        "--published-compiler-root",
        type=Path,
        help="relocated Merlin publication whose source payload exactly matches the graded Phase 1 submission",
    )
    parser.add_argument(
        "--perf-capsules", default="all", help="comma-separated performance capsules to measure, or 'all'"
    )
    parser.add_argument(
        "--perf-families", default="all", help="comma-separated performance families to measure, or 'all'"
    )
    parser.add_argument("--functional-submission-sha256", required=True)
    parser.add_argument(
        "--waive-functional-gate",
        action="append",
        default=[],
        metavar="PREDICATE",
        help="accept a NAMED completeness gap in the functional baseline instead of "
        "refusing (repeatable). Integrity predicates cannot be waived; asking is "
        "an error. Recorded in the manifest and marks the run not gate-clean.",
    )
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--rtl-facts", type=Path, required=True)
    parser.add_argument("--perf-profile", type=Path, required=True)
    parser.add_argument("--gsim-certificate", type=Path, required=True)
    parser.add_argument("--gsim-certificate-sha256", required=True)
    parser.add_argument(
        "--authoring-gsim-floor-budget",
        type=int,
        default=50_000,
        metavar="CYCLES",
        help="tuning members whose roofline floor exceeds CYCLES are not swept on gSIM during authoring; "
        "they are measured on gSIM only in the final cells (default 50000; 0 sweeps every member)",
    )
    parser.add_argument(
        "--single-observation-above-roofline-cycles",
        type=int,
        default=None,
        metavar="CYCLES",
        help="members whose predeclared roofline floor exceeds CYCLES get one gSIM observation cited by both "
        "replicate identities (gSIM is deterministic); recorded in the statistics predeclaration and every "
        "measurement plan (default: every replicate observed)",
    )
    parser.add_argument(
        "--certification",
        choices=("per_workload", "engine_qualified"),
        default="per_workload",
        help="gSIM timing admission policy, recorded in the declaration: per_workload (every measured "
        "workload captured on Verilator and gSIM; --gsim-certificate is a certificate) or "
        "engine_qualified (one gSIM build qualified on a stratified suite; a workload is admitted when "
        "its stratum and form are covered; --gsim-certificate is an engine qualification).",
    )
    parser.add_argument("--functional-gsim-certificate", type=Path)
    parser.add_argument("--functional-gsim-certificate-sha256")
    parser.add_argument(
        "--waive-functional-gsim-certificate",
        action="store_true",
        help="launch WITHOUT the public+hidden functional cross-validation "
        "certificate, accepting GSIM's functional verdicts on the same terms "
        "phase 1 accepted them. Recorded in the manifest and marks the run not "
        "gate-clean: the functional regrade becomes GSIM-only, uncorroborated "
        "by a second engine. Timing authority is unaffected -- it is pinned by "
        "the tuning certificate, which is never waivable.",
    )
    parser.add_argument("--heldout-qualification-timeout", type=int, default=600)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--wall-budget-seconds", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--round-timeout-seconds", type=int, required=True)
    parser.add_argument("--max-tool-calls", type=int, required=True)
    parser.add_argument("--tool-timeout-seconds", type=int, required=True)
    parser.add_argument("--smoke-replicates", type=int, default=1)
    parser.add_argument("--holdout-count", type=int, default=4)
    parser.add_argument(
        "--sim-workers",
        type=int,
        default=1,
        metavar="N",
        help="executions in flight per measurement cell (default 1 = serial)",
    )
    parser.add_argument("--generalization-count", type=int, default=4)
    form = parser.add_argument_group(
        "form-scale holdout",
        "an optional second held-out cohort, committed and revealed beside the PK holdout; "
        "absent, the campaign is unchanged",
    )
    form.add_argument(
        "--form-holdout-spec",
        type=Path,
        help="private YAML/JSON mapping with generated_root, applications and optional family; "
        "keeps the roster out of the recorded invocation",
    )
    form.add_argument(
        "--form-holdout-generated-root",
        type=Path,
        help="host-private Phase 0 run that generated the form-scale members",
    )
    form.add_argument(
        "--form-holdout-applications",
        help="comma-separated private performance-scale roster (recorded in the invocation; prefer the spec)",
    )
    form.add_argument("--form-holdout-family", help="form family to select (default PW)")
    parser.add_argument("--measurement-timeout", type=int, default=600)
    parser.add_argument("--gsim-max-cycles", type=int)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--hardware-counters", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--telemetry-price-table", type=Path)
    parser.add_argument("--chia-python", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(original_args)
    if resolve_context is None:
        context = AD.ExecutionContext(
            source_root=args.source_root,
            contract_root=args.contract_root,
            functional_runs_root=args.functional_runs_root,
            stage_root=args.stage_root,
            measurement_root=args.measurement_root,
            holdout_sources=HOLDOUT.HoldoutSourceContext(
                source_root=args.source_root,
                catalog_path=args.holdout_catalog,
                core_package_root=args.core_package_root,
                experiments_package_root=args.experiments_package_root,
                experiments_namespace_root=args.experiments_namespace_root,
            ),
            chia_wrapper=args.chia_wrapper,
            invocation=invocation,
            suite=args.suite,
        )
    else:
        context = resolve_context(args, invocation)
    _raw = {key: value for key, value in vars(args).items() if key not in (*resource_names, "suite", "dry_run")}
    _raw["context"] = context
    _raw["waive_functional_gate"] = tuple(_raw.pop("waive_functional_gate", ()) or ())
    _raw["authoring_gsim_floor_budget"] = _raw.get("authoring_gsim_floor_budget") or None  # 0: sweep every member
    _raw.update(_form_holdout_fields(_raw.pop("form_holdout_spec"), _raw))
    config = AD.Config(**_raw)
    try:
        outcome = CTRL.run(config, dry_run=args.dry_run)
    except Exception as exc:
        print(f"NO-GO: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outcome if isinstance(outcome, dict) else {"manifest": str(outcome)}, indent=2))
    return 0


def _form_holdout_fields(spec: Path | None, raw: dict) -> dict:
    """Config fields of the optional form-scale holdout, from a private spec or explicit flags."""
    root = raw.pop("form_holdout_generated_root")
    applications = raw.pop("form_holdout_applications")
    family = raw.pop("form_holdout_family")
    if spec is not None:
        if root is not None or applications is not None:
            raise SystemExit("--form-holdout-spec excludes --form-holdout-generated-root/--form-holdout-applications")
        try:
            fields = AD.load_form_holdout_spec(spec)
        except AD.ExperimentError as exc:
            raise SystemExit(f"NO-GO: {exc}") from exc
    elif root is None and applications is None:
        fields = {}
    else:
        if root is None or applications is None:
            raise SystemExit("--form-holdout-generated-root and --form-holdout-applications go together")
        labels = tuple(label.strip() for label in applications.split(",") if label.strip())
        fields = {"form_holdout_generated_root": root, "form_holdout_applications": labels}
    if family is not None:
        if not fields:
            raise SystemExit("--form-holdout-family needs a configured form-scale holdout")
        fields["form_holdout_family"] = family
    return fields


if __name__ == "__main__":
    raise SystemExit(main())
