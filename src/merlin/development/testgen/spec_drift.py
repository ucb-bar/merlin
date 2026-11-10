"""Phase 0 spec-vs-facts drift: the authored software spec against the selected fact-derived capability.

The comparison itself is :mod:`merlin.targetgen.spec_fact_drift`. This module feeds it the views a
Phase 0 selection already observed (never a live registry read), so the report is bound to the same
bytes the corpus was derived from, and blocking findings become readiness blockers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from merlin.targetgen import spec_fact_drift as D


def from_views(
    *,
    target: str,
    software_spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    raw_facts: Mapping[str, Any] | None,
    readout_facets: Iterable[Mapping[str, Any]],
    quantization_candidates: Iterable[Mapping[str, Any]],
    taxonomy: Mapping[str, Any] | None,
    prohibited_roles: Iterable[str] = (),
) -> dict[str, Any]:
    from merlin.targetgen.rtl.facts import observed_facts
    from merlin.targetgen.target_registry import observed_contract

    # Capability derivation still contains name-based geometry readers. They
    # must consume these selected documents, including an explicitly absent
    # facts document, rather than escaping to a live registry or extractor.
    with observed_contract(target, dict(contract)), observed_facts(target, dict(raw_facts or {})):
        facts = D.fact_capabilities(
            target=target,
            contract=contract,
            raw_facts=raw_facts,
            readout_facets=readout_facets,
            quantization_candidates=quantization_candidates,
            taxonomy=taxonomy,
            prohibited_roles=prohibited_roles,
        )
    # A selection's spec is already resolved; an authored one is resolved against the same facts, so
    # the comparison always sees the values admission will use.
    resolved, _ = D.resolve_spec(software_spec, facts)
    report = D.spec_fact_drift(resolved, facts, target=target)
    report["prohibited_instruction_roles"] = sorted(set(prohibited_roles))
    return report


def from_selection(selection, *, prohibited_roles: Iterable[str] = ()) -> dict[str, Any]:
    """The drift report for an :class:`EvidenceSelection`'s observed views."""
    return from_views(
        target=selection.target,
        software_spec=selection.software_spec or {},
        contract=selection.contract or {},
        raw_facts=json.loads(selection.raw_facts) if selection.raw_facts is not None else None,
        readout_facets=selection.readout_facets or (),
        quantization_candidates=(selection.quantization_snapshot or {}).get("quantization_candidates") or (),
        taxonomy=selection.isa_taxonomy,
        prohibited_roles=prohibited_roles,
    )


def from_evidence_dir(root: Path, *, prohibited_roles: Iterable[str] = ()) -> dict[str, Any]:
    """The drift report for an exported Phase 0 ``evidence/`` directory."""

    def load(name: str):
        return json.loads((root / name).read_bytes())

    software = load("software/software-spec.json")
    raw = root / "hardware/circt/facts.json"
    return from_views(
        target=software["target"],
        software_spec=software,
        contract=load("software/contract.json"),
        raw_facts=json.loads(raw.read_bytes()) if raw.is_file() else None,
        readout_facets=load("hardware/effective-views/readout-facets.json"),
        quantization_candidates=load("hardware/effective-views/quantization.json").get("quantization_candidates") or (),
        taxonomy=load("hardware/effective-views/isa-taxonomy.json"),
        prohibited_roles=prohibited_roles,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="an exported Phase 0 evidence/ directory")
    parser.add_argument("--software-spec", type=Path, help="compare this authored spec instead of the selected one")
    parser.add_argument("--prohibited-role", action="append", default=[], help="experiment-prohibited role")
    args = parser.parse_args(argv)
    report = from_evidence_dir(args.evidence, prohibited_roles=args.prohibited_role)
    if args.software_spec is not None:
        from merlin.targetgen.software_spec import load_software_spec

        root = args.evidence
        report = from_views(
            target=report["target"],
            software_spec=load_software_spec(args.software_spec),
            contract=json.loads((root / "software/contract.json").read_bytes()),
            raw_facts=json.loads((root / "hardware/circt/facts.json").read_bytes())
            if (root / "hardware/circt/facts.json").is_file()
            else None,
            readout_facets=json.loads((root / "hardware/effective-views/readout-facets.json").read_bytes()),
            quantization_candidates=json.loads((root / "hardware/effective-views/quantization.json").read_bytes()).get(
                "quantization_candidates"
            )
            or (),
            taxonomy=json.loads((root / "hardware/effective-views/isa-taxonomy.json").read_bytes()),
            prohibited_roles=args.prohibited_role,
        )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 1 if report["n_blocking"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
