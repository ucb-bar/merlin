"""Explicit input/output entry point for declarative instruction dialect generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen.isa_dialect_gen import generate_package
from merlin.targetgen.isa_spec import IsaSpecError, resolve_isa


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--software-spec", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = yaml.safe_load(args.software_spec.read_bytes())
        if (
            not isinstance(spec, dict)
            or spec.get("schema") != "merlin.software_spec.v1"
            or spec.get("target") != args.target
        ):
            raise IsaSpecError("software spec must explicitly select the requested target and schema")
        raw = args.facts.read_bytes()
        model = resolve_isa(
            spec.get("isa"), json.loads(raw), target=args.target, facts_sha256=hashlib.sha256(raw).hexdigest()
        )
        manifest = generate_package(model, args.out)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f"ISA generation refused: {exc}\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
