"""Prepare or inspect an authored scheduling declaration without target discovery.

Uses the same core API offline and inside an explicitly selected author library.
No backend import, target code, runtime selection, qualification or simulator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from merlin.compile.scheduling.selected_contracts import inspect_scheduling_draft, scheduling_spec_draft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("draft", "inspect"))
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--software", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--isa", type=Path, help="Explicit candidate ISA declaration, draft action only")
    parser.add_argument("--dense-schema", choices=("merlin.dense_schedule.v1", "merlin.dense_schedule.v2"))
    parser.add_argument("--require-dense", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory; never overwrite selected inputs")
    args = parser.parse_args()
    if args.action == "inspect" and (args.isa is not None or args.dense_schema is not None):
        parser.error("inspect reads the supplied document unchanged")
    facts = args.facts.read_bytes()
    software = args.software.read_bytes()
    if args.action == "draft":
        draft = scheduling_spec_draft(yaml.safe_load(software), target=args.target,
                                     isa=json.loads(args.isa.read_bytes()) if args.isa else None,
                                     dense_schema=args.dense_schema)
        software = yaml.safe_dump(draft, sort_keys=False).encode()
    report = inspect_scheduling_draft(facts, software, target=args.target, require_dense=args.require_dense)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.action == "draft":
        (args.output / "software-draft.yaml").write_bytes(software)
    (args.output / "requirements.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "qualification": report["qualification"],
                      "issues": len(report["issues"]), "unknown_fields": len(report["unknown_fields"]),
                      "report": str(args.output / "requirements.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
