#!/usr/bin/env python3
"""Write the Phase 0 spec-validation case selection for the Core ATen suites.

Core ATen suites are spec-validation evidence, not capsules. This writes only
the identities of the cases the reference backend must run: no recipe, cohorts,
lanes, expectations or descriptor copy. Every case of each given suite is kept.

Output (one JSON file, ``sort_keys``, indent 2, trailing newline):

    {
      "schema": "merlin.phase0.spec_validation_selection.v1",
      "suites": {
        "<source>": {                       # "corpus_193" and/or "bounded"
          "file_sha256": "<sha256 of the suite file bytes>",
          "denominator_sha256": "<suite value, or null>",
          "suite_sha256": "<suite value, or null>",
          "case_count": <int>,
          "case_ids": ["<case_id>", ...]    # sorted, unique within the suite
        }
      },
      "case_count": <total over suites>,
      "selection_sha256": "<see below>"
    }

``corpus_193`` is the canonical one-case-per-overload corpus document (key
``cases``); ``bounded`` is the bounded suite (key ``selected_cases``). A case's
identity is its ``case_id``, or its ``overload`` when it has none (the 193
corpus). Spec-validation rows are identified by ``(source, case_id)``.
``selection_sha256`` is the SHA-256 of every other field, encoded as
``json.dumps(..., sort_keys=True, separators=(",", ":")) + "\\n"``. Suite paths
are not recorded; a consumer is given the suite files and checks
``file_sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.phase0.spec_validation_selection.v1"
#: Selection source -> the suite key holding its cases.
SOURCES = {"corpus_193": "cases", "bounded": "selected_cases"}


def selection_digest(document: dict) -> str:
    payload = {key: value for key, value in document.items() if key != "selection_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(encoded.encode()).hexdigest()


def suite_selection(source: str, raw: bytes) -> dict:
    suite = json.loads(raw)
    cases = suite.get(SOURCES[source]) if isinstance(suite, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{source}: suite must contain a non-empty {SOURCES[source]!r} list")
    ids = [case.get("case_id") or case["overload"] for case in cases]
    if any(not isinstance(identity, str) or not identity for identity in ids):
        raise ValueError(f"{source}: every case needs a case_id or overload string")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{source}: duplicate case identities")
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "denominator_sha256": suite.get("denominator_sha256"),
        "suite_sha256": suite.get("suite_sha256"),
        "case_count": len(ids),
        "case_ids": sorted(ids),
    }


def build(suites: dict[str, Path]) -> dict:
    if not suites or set(suites) - set(SOURCES):
        raise ValueError(f"select at least one suite from {sorted(SOURCES)}")
    selected = {source: suite_selection(source, Path(path).read_bytes()) for source, path in sorted(suites.items())}
    document = {
        "schema": SCHEMA,
        "suites": selected,
        "case_count": sum(item["case_count"] for item in selected.values()),
    }
    document["selection_sha256"] = selection_digest(document)
    return document


def write(document: dict, output: Path) -> None:
    with Path(output).open("x") as handle:  # Never replace an existing selection.
        handle.write(json.dumps(document, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-193", type=Path, help="canonical Core ATen corpus document (cases)")
    parser.add_argument("--bounded", type=Path, help="bounded Core ATen suite (selected_cases)")
    parser.add_argument("--output", type=Path, required=True, help="new selection JSON file")
    args = parser.parse_args()
    suites = {source: path for source, path in (("corpus_193", args.corpus_193), ("bounded", args.bounded)) if path}
    if not suites:
        parser.error("give --corpus-193 and/or --bounded")
    document = build(suites)
    write(document, args.output)
    print(json.dumps({source: item["case_count"] for source, item in document["suites"].items()}, sort_keys=True))
    print(document["selection_sha256"])


if __name__ == "__main__":
    main()
