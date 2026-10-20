"""Installed functional compiler execution with explicit operator inputs."""

from __future__ import annotations

import sys
from pathlib import Path

from .options import RunOptions, build_parser


def main(argv: list[str] | None = None) -> int:
    from merlin_experiments.phase2.compiler_library_state import add_selection_arguments, load_selection

    parser = build_parser()
    add_selection_arguments(parser)
    parser.description = "Run functional compiler authoring from explicit experiment inputs."
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="operator root for declared resources")
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--oracle-timing", type=Path, required=True, help="operator-owned timing record path")
    parser.add_argument("--public-root", type=Path, help="explicit public grading corpus override")
    parser.add_argument("--language", default="", help="task language selection")
    parser.add_argument(
        "--require-shared-passes",
        action="store_true",
        help="new authoring only: require reviewed shared-pass interfaces and deterministic defaults",
    )
    parser.add_argument("--treatment", choices=("baseline", "rtlchecks"), default="baseline")
    from .levels import LEVELS, materialize_level

    parser.add_argument(
        "--level",
        choices=tuple(level["id"] for level in LEVELS),
        help="readable experiment level; derives the legacy --arm and --treatment keys",
    )
    arguments = list(sys.argv[1:] if argv is None else argv)
    values = vars(parser.parse_args(arguments))
    if (
        values["codex_canary"] or values["codex_binary"] or values["codex_auth_source"] or values["codex_home_root"]
    ) and not any(arg == "--model" or arg.startswith("--model=") for arg in arguments):
        parser.error("explicit Codex runtime requires an explicit --model selection")
    if not values["bundle"]:
        parser.error("installed execution requires --bundle (the authored bundle identity)")
    selected_level = values.pop("level")
    if selected_level:
        selection = {"level": selected_level, "bundle": values["bundle"]}
        for key in ("arm", "treatment"):
            flag = "--" + key
            if any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
                selection[key] = values[key]
        try:
            materialize_level(selection)
        except ValueError as exc:
            parser.error(str(exc))
        for key in ("arm", "treatment"):
            flag = "--" + key
            values[key] = selection[key]
            if not any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
                arguments += [flag, str(selection[key])]
    if values["treatment"] == "rtlchecks":
        from .feedback.rtlchecks import prepare_arguments

        try:
            arguments = prepare_arguments(arguments, bundle_manifest=values["bundle_manifest"].expanduser().resolve())
        except ValueError as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            return 4
        values = vars(parser.parse_args(arguments))
        values.pop("level")
    treatment_name = values.pop("treatment")
    descriptor, repo = values.pop("descriptor"), values.pop("repo")
    manifest = values.pop("bundle_manifest").expanduser().resolve()
    timing = values.pop("oracle_timing").expanduser().resolve()
    public = values.pop("public_root")
    language = values.pop("language")
    require_shared_passes = values.pop("require_shared_passes")
    if not manifest.is_file():
        parser.error(f"bundle manifest is not a readable file: {manifest}")
    library = load_selection(
        root=values.pop("compiler_library_root"),
        contract_path=values.pop("compiler_library_contract"),
        contract_sha256=values.pop("compiler_library_contract_sha256"),
    )
    options = RunOptions(**values)

    # Context must precede environment-sensitive execution imports. No native
    # launcher, checkout discovery or machine-specific library default is used.
    from .context import load_context

    context_kwargs = {"readback_policy": options.readback_policy} if options.readback_policy else {}
    context = load_context(descriptor, repo=repo, **context_kwargs)
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
        compiler_library_selection=library,
        **({"require_shared_passes": True} if require_shared_passes else {}),
    )


if __name__ == "__main__":
    raise SystemExit(main())
