"""``merlin experiment census``: exact, unpriced censuses of one observed execution, from records on disk.

Two analyses whose inputs come from a trace reader or a TARGET PROVIDER, never from the core:

    merlin experiment census locality ADDRESSES.json --granule BYTES --max-requests N [--capacity C ...]
    merlin experiment census boundaries RECORDS.json [--entry FUNCTION] [--admit DOMAIN.json --domain-sha256 S]
    merlin experiment census boundary-domain SUMMARY.json ... --entry FUNCTION --pointer P ... --domain-sha256 S

``locality`` reads an ordered JSON list of requested addresses and prints the exact recurrence census
of :func:`merlin.analysis.perf.address_locality.address_locality` -- first touches and the distinct-intervening-
region distance histogram. The granule and the request budget are the caller's: nothing here knows a
line size, and a trace longer than the budget is refused rather than truncated. Its producer is
``merlin experiment inspect --group gN --trace``, which records the group program's committed memory
requests as ``requested_addresses.json``; this re-takes the census at another granule without a rerun.

``boundaries`` reads a provider's decoded function extents and instruction records (the provider owns
instruction decoding, the ABI and the digests of the program, census and its own artifacts) and prints
:func:`merlin.analysis.perf.execution_boundaries.summarize_boundaries`. With ``--entry`` it adds the features
normalised to that function's invocations; with ``--admit`` it checks those features against a
boundary domain derived from training summaries (``boundary-domain``), which never approves a ranking.
No provider writes those records yet: call kinds, callees and stack access widths are host-ABI
decoding a target's support owns, and none of the vendored providers declares a decoder for them.

Every command prints one JSON document and exits 0, or names its refusal on stderr and exits 2. Unknown
facts stay ``null``; nothing is read as zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from merlin.analysis.perf import address_locality as AL
from merlin.analysis.perf import execution_boundaries as EB

#: Exit status of a refused input: malformed JSON, a record the analysis rejects, a budget exceeded.
REFUSED = 2


def _load(path: str) -> Any:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return json.loads(text)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def locality(args: argparse.Namespace) -> dict[str, Any]:
    addresses = _load(args.addresses)
    _require(isinstance(addresses, list), "the address trace must be a JSON list of nonnegative integers")
    result = AL.address_locality(
        addresses, granule=args.granule, max_requests=args.max_requests, capacities=tuple(args.capacity or ())
    )
    return {"schema": "address_locality_v1", **asdict(result)}


def _records(document: Any) -> tuple[list[EB.FunctionExtent], list[EB.BoundaryInstruction], dict[str, str]]:
    _require(isinstance(document, dict), "provider records must be a JSON object")
    functions, instructions = document.get("functions"), document.get("instructions")
    _require(isinstance(functions, list) and isinstance(instructions, list), "functions and instructions lists")
    digests = {}
    for key in ("program_sha256", "census_sha256", "provider_sha256"):
        _require(isinstance(document.get(key), str), f"provider records name no {key}")
        digests[key] = document[key]
    try:
        extents = [EB.FunctionExtent(**row) for row in functions]
        decoded = [EB.BoundaryInstruction(**row) for row in instructions]
    except TypeError as exc:
        raise ValueError(f"a provider record does not have the declared fields: {exc}") from exc
    return extents, decoded, digests


def _domain(document: Any) -> EB.BoundaryDomain:
    _require(isinstance(document, dict), "a boundary domain must be a JSON object")
    bounds = document.get("bounds")
    _require(isinstance(bounds, list) and all(isinstance(b, list) and len(b) == 3 for b in bounds), "domain bounds")
    return EB.BoundaryDomain(
        str(document.get("domain_sha256")), tuple(tuple(b) for b in bounds), str(document.get("evidence_sha256"))
    )


def boundaries(args: argparse.Namespace) -> dict[str, Any]:
    extents, decoded, digests = _records(_load(args.records))
    summary = EB.summarize_boundaries(extents, decoded, **digests)
    out: dict[str, Any] = {"summary": summary}
    if args.entry:
        out["features"] = EB.boundary_features(summary, entry_function=args.entry)
    if args.admit:
        _require(bool(args.entry) and bool(args.domain_sha256), "--admit needs --entry and --domain-sha256")
        out["admission"] = _domain(_load(args.admit)).admit(out["features"], domain_sha256=args.domain_sha256)
    return out


def boundary_domain(args: argparse.Namespace) -> dict[str, Any]:
    domain = EB.derive_boundary_domain(
        [_load(path) for path in args.summaries],
        pointers=list(args.pointer),
        entry_function=args.entry,
        domain_sha256=args.domain_sha256,
    )
    return {
        "domain_sha256": domain.domain_sha256,
        "bounds": [list(bound) for bound in domain.bounds],
        "evidence_sha256": domain.evidence_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merlin experiment census", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    loc = sub.add_parser("locality", help="exact recurrence census of an ordered requested-address trace")
    loc.add_argument("addresses", help="JSON list of addresses in request order ('-' reads stdin)")
    loc.add_argument("--granule", type=int, required=True, help="region size in address units")
    loc.add_argument("--max-requests", type=int, required=True, help="refuse a trace longer than this")
    loc.add_argument("--capacity", type=int, action="append", help="count recurrences at or above this distance")
    loc.set_defaults(run=locality)
    bnd = sub.add_parser("boundaries", help="call and stack boundary summary of provider-decoded records")
    bnd.add_argument("records", help="JSON object: digests, functions, instructions ('-' reads stdin)")
    bnd.add_argument("--entry", help="normalise the features to this function's invocations")
    bnd.add_argument("--admit", help="a boundary domain (JSON) to check the features against")
    bnd.add_argument("--domain-sha256", help="the domain identity the features were observed under")
    bnd.set_defaults(run=boundaries)
    dom = sub.add_parser("boundary-domain", help="derive an unpriced boundary domain from training summaries")
    dom.add_argument("summaries", nargs="+", help="summary JSON documents (the `summary` of `boundaries`)")
    dom.add_argument("--entry", required=True)
    dom.add_argument("--pointer", action="append", required=True, help="a /boundaries/ feature to bound")
    dom.add_argument("--domain-sha256", required=True)
    dom.set_defaults(run=boundary_domain)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = args.run(args)
    except (OSError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return REFUSED
    print(json.dumps(document, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
