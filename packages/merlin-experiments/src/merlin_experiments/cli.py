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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin experiment", description=__doc__)
    parser.add_argument("--catalog", type=Path, help="catalog YAML; paths inside it are relative to that file")
    commands = parser.add_subparsers(dest="verb", required=True)
    commands.add_parser("list", help="list the versioned experiment catalog")
    stored = commands.add_parser("runs", help="discover stored phase orchestrations (read-only)")
    stored.add_argument("--root", type=Path, help="run root; defaults to the configured out/runs")
    stored.add_argument("--target", help="filter by exact target identity")
    stored.add_argument("--experiment", help="filter by exact experiment identity")
    for verb in ("inspect", "preflight", "run"):
        child = commands.add_parser(verb)
        child.add_argument("spec", help="definition path or catalog id")
        child.add_argument("--phase", choices=("0", "1", "2", "all"), default="all")
        child.add_argument("--run-dir", type=Path, help="explicit output; otherwise use the configured run root")
        child.add_argument("--corpus-seal", type=Path, help="reviewed Phase 0 release seal for Phase 1")
        child.add_argument("--bundle-manifest", type=Path, help="reviewed replacement Phase 1 input bundle")
    commands.add_parser("status").add_argument("run_dir", type=Path)
    commands.add_parser("lineage", help="read frozen phase inputs and handoffs without executing engines").add_argument(
        "run_dir", type=Path
    )
    child = commands.add_parser("resume")
    child.add_argument("run_dir", type=Path)
    child.add_argument("--checkpoint", type=Path, help="sealed native checkpoint for a new model_portfolio segment")
    corpus = commands.add_parser(
        "corpus", help="derive capsule groups, or prepare, inspect and review a corpus release"
    )
    operations = corpus.add_subparsers(dest="operation", required=True)
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
    prepare = operations.add_parser("prepare")
    prepare.add_argument("run_dir", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    operations.add_parser("inspect").add_argument("release", type=Path)
    seal = operations.add_parser("seal")
    seal.add_argument("release", type=Path)
    seal.add_argument("--expected-digest", required=True)
    seal.add_argument("--reviewed-by", required=True)
    seal.add_argument("--review-note", required=True)
    args = parser.parse_args(argv)
    try:
        if args.verb == "corpus":
            if args.operation == "groups":
                return group_capsules.run_from_args(args)
            if args.operation == "compare":
                from .phase0.comparison import build

                print(json.dumps(build([_source(value, args.catalog) for value in args.definitions]), indent=2))
                return 0
            from .corpus import release as corpus_release

            if args.operation == "prepare":
                result = corpus_release.prepare(args.run_dir, args.output)
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
                    }
                )
        elif args.verb == "runs":
            from .history import runs

            result = runs(root=args.root, target=args.target, experiment=args.experiment)
        elif args.verb == "status":
            result = runner.status(args.run_dir)
        elif args.verb == "lineage":
            from .history import lineage

            result = lineage(args.run_dir)
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
