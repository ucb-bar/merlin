"""Host-owned Phase 0 __main__ implementation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generation import generate_target
from .profiles import build_comparison_manifest, profile_targets, validate_profile_inputs, write_comparison_manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified descriptor-driven capsule-corpus generator.", allow_abbrev=False)
    ap.add_argument("--target", default=None, help="one target (default: every target with a profile)")
    ap.add_argument(
        "--profiles-root",
        type=Path,
        default=None,
        help="complete external profile directory, including shared template and available sidecars",
    )
    ap.add_argument("--recipe", type=Path, help="explicit public authored recipe; never discover sibling files")
    ap.add_argument(
        "--performance-template", type=Path, help="sole shared performance template; required with --recipe"
    )
    ap.add_argument("--synth-profile", type=Path, help="explicit optional generated synthesis sidecar")
    ap.add_argument("--smt-profile", type=Path, help="explicit optional solver-generated sidecar")
    ap.add_argument("--hidden-profile", type=Path, help="explicit optional private holdout sidecar (host only)")
    ap.add_argument(
        "--descriptor",
        type=Path,
        default=None,
        help="explicit target descriptor, including out-of-tree paths; requires --target profile selector",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="explicit artifact destination for capsules and MANIFEST; requires --target; "
        "the descriptor's source corpus is never an implicit output",
    )
    ap.add_argument(
        "--comparison-manifest",
        action="store_true",
        help="also emit the cross-target op-comparison manifest under out/artifacts/compare/",
    )
    a = ap.parse_args(argv)
    from merlin.common.paths import checkout_root

    profile_inputs = {
        name: getattr(a, name)
        for name in (
            "profiles_root",
            "recipe",
            "performance_template",
            "synth_profile",
            "smt_profile",
            "hidden_profile",
        )
    }
    try:
        validate_profile_inputs(**profile_inputs)
    except ValueError as exc:
        ap.error(str(exc))
    if a.recipe is None and a.profiles_root is None:
        ap.error(
            "explicit --recipe and --performance-template or --profiles-root are required; "
            "use merlin experiment run with a definition"
        )
    if a.recipe is not None and a.comparison_manifest:
        ap.error(
            "use merlin experiment corpus compare DEFINITION DEFINITION for explicit recipes; "
            "--comparison-manifest requires a legacy --profiles-root collection"
        )
    if a.comparison_manifest and not profile_targets(profiles_root=a.profiles_root):
        ap.error("no legacy public profiles found; use merlin experiment corpus compare with explicit definitions")
    if checkout_root() is None and (a.descriptor is None or (a.profiles_root is None and a.recipe is None)):
        ap.error("installed Phase 0 requires --profiles-root or --recipe, --descriptor and --output-root")
    if a.descriptor is not None and a.target is None:
        ap.error("--descriptor requires --target so exactly one profile is selected")
    if a.output_root is not None and a.target is None:
        ap.error("--output-root requires --target so corpora cannot overwrite one another")
    targets = [a.target] if a.target else profile_targets(profiles_root=a.profiles_root)
    for t in targets:
        options = {}
        if a.descriptor is not None:
            options["descriptor"] = a.descriptor
        if a.output_root is not None:
            options["output_root"] = a.output_root
        options.update({name: value for name, value in profile_inputs.items() if value is not None})
        written = generate_target(t, **options)
        print(f"{t}: wrote {len(written)} capsules -> {written[0].parent.parent if written else '(none)'}")
    if a.comparison_manifest or not a.target:
        allt = profile_targets(profiles_root=a.profiles_root)
        m = write_comparison_manifest(allt, profiles_root=a.profiles_root)
        count = len(build_comparison_manifest(allt, profiles_root=a.profiles_root)["comparison_sets"])
        print(f"comparison manifest: {m} ({count} sets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
