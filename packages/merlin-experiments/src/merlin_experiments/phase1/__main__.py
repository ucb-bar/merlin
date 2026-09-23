"""Installed functional compiler execution with explicit operator inputs."""

from __future__ import annotations

import sys
from pathlib import Path

from .options import RunOptions, build_parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = "Run functional compiler authoring from explicit experiment inputs."
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="operator root for declared resources")
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--oracle-timing", type=Path, required=True, help="operator-owned timing record path")
    parser.add_argument("--public-root", type=Path, help="explicit public grading corpus override")
    parser.add_argument("--language", default="", help="task language selection")
    parser.add_argument("--treatment", choices=("baseline", "rtlchecks"), default="baseline")
    arguments = list(sys.argv[1:] if argv is None else argv)
    values = vars(parser.parse_args(arguments))
    if not values["bundle"]:
        parser.error("installed execution requires --bundle (the authored bundle identity)")
    if values["treatment"] == "rtlchecks":
        from .feedback.rtlchecks import prepare_arguments

        try:
            arguments = prepare_arguments(arguments, bundle_manifest=values["bundle_manifest"].expanduser().resolve())
        except ValueError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 4
        values = vars(parser.parse_args(arguments))
    treatment_name = values.pop("treatment")
    descriptor, repo = values.pop("descriptor"), values.pop("repo")
    manifest = values.pop("bundle_manifest").expanduser().resolve()
    timing = values.pop("oracle_timing").expanduser().resolve()
    public = values.pop("public_root")
    language = values.pop("language")
    if not manifest.is_file():
        parser.error(f"bundle manifest is not a readable file: {manifest}")
    options = RunOptions(**values)

    # Context must precede environment-sensitive execution imports. No native
    # launcher, checkout discovery or machine-specific library default is used.
    from .context import load_context

    context = load_context(descriptor, repo=repo)
    from .controller import run

    treatment = None
    if treatment_name == "rtlchecks":
        from .feedback.rtlchecks import treatment as rtl_treatment

        treatment = rtl_treatment(context)

    return run(
        context,
        options,
        bundle_manifest=manifest,
        bundle_id=options.bundle,
        oracle_timing=timing,
        launcher_argv=tuple(arguments),
        language=language,
        public_root=None if public is None else public.expanduser().resolve(),
        treatment=treatment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
