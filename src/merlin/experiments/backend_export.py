"""Export an existing phase champion and inspect its preserved service metadata.

The champion owner retains evidence validation, source custody and publication.
This command neither creates a compiler adapter nor installs or publishes code.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


def service_declarations(root: Path) -> dict:
    """Read ordinary compiler declarations without importing or validating them."""
    result = {
        "scope": "preserved_project_metadata_only",
        "compatibility": "not_checked",
        "installed_by_export": False,
        "code_loaded_by_inspection": False,
        "providers": [],
    }
    path = root / "pyproject.toml"
    if not path.exists():
        return {**result, "status": "missing_metadata"}
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        if "project" not in document:
            return {**result, "status": "missing_project_metadata"}
        project = document.get("project", {})
        if not isinstance(project, dict):
            raise ValueError("project must be a table")
        dynamic = project.get("dynamic", [])
        if not isinstance(dynamic, list) or any(not isinstance(value, str) for value in dynamic):
            raise ValueError("project.dynamic must be a list of field names")
        groups = project.get("entry-points", {})
        if not isinstance(groups, dict):
            raise ValueError("project.entry-points must be a table")
        if "entry-points" in dynamic:
            if "entry-points" in project:
                raise ValueError("entry points cannot be both static and dynamic")
            return {**result, "status": "dynamic"}
        providers = groups.get("merlin.backends", {})
        if not isinstance(providers, dict) or any(
            not isinstance(name, str) or not name.strip() or not isinstance(value, str) or not value.strip()
            for name, value in providers.items()
        ):
            raise ValueError("merlin.backends must map nonempty names to nonempty entry-point strings")
        return {
            **result,
            "status": "declared" if providers else "absent",
            "providers": [{"name": name, "entry_point": providers[name]} for name in sorted(providers)],
        }
    except (OSError, ValueError):
        return {**result, "status": "invalid_metadata"}


def _object(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate evidence key: {key}")
        document[key] = value
    return document


def _record(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError(f"invalid JSON numeric constant: {value}")

    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object, parse_constant=invalid_constant)
    if not isinstance(document, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="merlin experiment export",
        description="Export the selected frozen/best compiler history using its existing champion evidence.",
        epilog="Preserves existing packaging; does not synthesize adapters, install code or publish remotely.",
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--repo", type=Path, required=True, help="the run's existing OOT Git history")
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True)
    parser.add_argument("--package-id", required=True)
    for name in ("provenance", "certification", "measurements", "isa-prohibition"):
        parser.add_argument("--" + name, type=Path, required=True, help="existing scoped evidence JSON record")
    parser.add_argument("--rev", help="selected revision; defaults to the phase's frozen/best tag")
    parser.add_argument("--run-dir", type=Path, help="owning run directory; defaults to the OOT history's parent")
    parser.add_argument("--stage-root", type=Path, help="explicit retained export staging parent")
    parser.add_argument("--artifacts-root", type=Path, help="artifact root containing targets/<target>/champions")
    args = parser.parse_args(argv)

    from merlin.targetgen.champions import ChampionError, export_champion

    try:
        records = {
            name: _record(getattr(args, name))
            for name in ("provenance", "certification", "measurements", "isa_prohibition")
        }
        destination = export_champion(
            args.target,
            args.repo,
            package_id=args.package_id,
            phase=args.phase,
            rev=args.rev,
            phase2_run=args.run_dir,
            stage_root=args.stage_root,
            artifacts_root=args.artifacts_root,
            **records,
        )
    except (ChampionError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"directory": str(destination), "services": service_declarations(destination)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
