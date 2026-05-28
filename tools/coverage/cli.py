"""`./merlin coverage-check` — per-dispatch accelerator coverage of a VMFB."""

from pathlib import Path


def setup_parser(parser):
    parser.add_argument("vmfb", type=Path, help=".vmfb file to inspect")
    parser.add_argument("--csv", type=Path, default=None, help="Write per-function CSV")


def main(args) -> int:
    # Pass through to the standalone implementation.
    import sys

    from coverage import check

    sys.argv = ["coverage-check", str(args.vmfb)]
    if args.csv is not None:
        sys.argv.extend(["--csv", str(args.csv)])
    return check.main()
