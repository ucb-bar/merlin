"""Explicit input/output entry point for declarative instruction dialect generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen.isa_dialect_gen import generate_package
from merlin.targetgen.isa_spec import IsaSpecError, resolve_isa


def inspect_inputs(*, software_spec: Path, facts: Path, target: str) -> dict:
    """Inspect explicit declaration readiness without generating or executing code.

    A resolvable declaration establishes mechanical generation readiness only.
    It cannot establish that the declared encoding/semantics match selected RTL,
    or that a generated operation reaches an executable compiler path.
    """
    spec_bytes = Path(software_spec).read_bytes()
    facts_bytes = Path(facts).read_bytes()
    result = {
        'schema': 'merlin.isa_scaffold_readiness.v1',
        'target': target,
        'software_spec_sha256': hashlib.sha256(spec_bytes).hexdigest(),
        'facts_sha256': hashlib.sha256(facts_bytes).hexdigest(),
        'status': 'invalid',
        'generation_ready': False,
        'independent_rtl_correspondence': 'UNKNOWN',
        'compiler_execution': 'UNKNOWN',
    }
    try:
        spec = yaml.safe_load(spec_bytes)
        facts_record = json.loads(facts_bytes)
        if (not isinstance(spec, dict) or spec.get('schema') != 'merlin.software_spec.v1'
                or spec.get('target') != target):
            raise IsaSpecError('software spec must explicitly select the requested target and schema')
        if 'isa' not in spec:
            result.update(status='missing_declaration', reason='selected software specification has no isa declaration')
            return result
        model = resolve_isa(spec['isa'], facts_record, target=target,
                            facts_sha256=result['facts_sha256'])
        result.update(
            status='ready_for_generation', generation_ready=True, model_sha256=model.identity(),
            instruction_count=len(model.instructions),
            unknown_semantics=[i.name for i in model.instructions if json.loads(i.semantics_json).get('status') == 'unknown'],
            unknown_effects=[i.name for i in model.instructions if json.loads(i.effects_json).get('status') == 'unknown'],
            reason='resolved typed declarations; independent encoding/RTL and execution checks remain required',
        )
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        result['reason'] = f'{type(exc).__name__}: {str(exc)[:512]}'
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--software-spec", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--target", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out", type=Path)
    action.add_argument("--inspect", action="store_true", help="report readiness and source identities without writing files")
    args = parser.parse_args(argv)
    try:
        if args.inspect:
            report = inspect_inputs(software_spec=args.software_spec, facts=args.facts, target=args.target)
            print(json.dumps(report, sort_keys=True))
            return 0 if report['generation_ready'] else 2
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
