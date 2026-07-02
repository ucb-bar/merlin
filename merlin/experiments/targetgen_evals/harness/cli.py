"""Entry point: python -m harness.cli <subcommand> [args]"""

import argparse
import sys
from pathlib import Path


def _root() -> Path:
    """Return the targetgen-evals/ root (parent of this file's package)."""
    return Path(__file__).parent.parent


def _add_tracking_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tracking",
        default="local",
        choices=("local", "mlflow", "full", "debug"),
        help="Tracking mode (default: local)",
    )
    p.add_argument("--mlflow-tracking-uri", default=None, metavar="URI",
                   help="MLflow tracking server URI (e.g. http://localhost:5000)")
    p.add_argument("--experiment-name", default=None, metavar="NAME",
                   help="MLflow experiment name")
    p.add_argument("--otel-endpoint", default=None, metavar="URL",
                   help="OpenTelemetry OTLP endpoint (e.g. http://localhost:4318/v1/traces)")


def cmd_init_run(args: argparse.Namespace) -> int:
    from harness.materialize_run import materialize
    return materialize(
        root=_root(),
        target=args.target,
        method=args.method,
        seed=args.seed,
        force=args.force,
        is_smoke_test=args.smoke,
        budget=args.budget,
        tracking_mode=args.tracking,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        otel_endpoint=args.otel_endpoint,
    )


def cmd_validate(args: argparse.Namespace) -> int:
    from harness.run_experiment import validate_run
    return validate_run(
        run_path=Path(args.run_path),
        root=_root(),
        tracking_mode=args.tracking,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        otel_endpoint=args.otel_endpoint,
    )


def cmd_compare(args: argparse.Namespace) -> int:
    from harness.compare_runs import compare
    output_dir = Path(args.output_dir) if args.output_dir else _root() / "reports" / args.target
    return compare(root=_root(), target=args.target, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m harness.cli",
        description="targetgen-evals harness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-run", help="Initialise an isolated experiment run directory")
    p_init.add_argument("--target", required=True, help="Target name (e.g. gemmini)")
    p_init.add_argument("--method", required=True, help="Method name (e.g. v0_naive_claude)")
    p_init.add_argument("--seed", type=int, required=True, help="Random seed integer")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing run directory")
    p_init.add_argument("--smoke", action="store_true", default=True,
                        help="Mark as smoke test (default: true; use --no-smoke for real baseline)")
    p_init.add_argument("--no-smoke", dest="smoke", action="store_false")
    p_init.add_argument("--budget", default="cheap_smoke",
                        help="Budget profile name (default: cheap_smoke)")
    _add_tracking_args(p_init)

    p_val = sub.add_parser("validate", help="Validate a run directory")
    p_val.add_argument("run_path", help="Path to the run directory")
    _add_tracking_args(p_val)

    p_cmp = sub.add_parser("compare", help="Aggregate all runs for a target")
    p_cmp.add_argument("--target", required=True, help="Target name (e.g. gemmini)")
    p_cmp.add_argument("--output-dir", default=None, help="Override output directory for reports")

    args = parser.parse_args()
    dispatch = {"init-run": cmd_init_run, "validate": cmd_validate, "compare": cmd_compare}
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
