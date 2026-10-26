#!/usr/bin/env python3
"""Run Phase 0 spec validation over a Core ATen case selection and seal its record.

The reference backend executes every selected case through
``run_bounded_core_aten_suite.py`` under ``--numeric-policy spec_int8`` (resumable:
completed shards with the same case digest and provenance stamp are reused). The
selection resolves cases from byte-pinned ``--corpus-193`` and/or ``--bounded``
suite files. The shards are then judged against the spec's own admission decisions and written as
``merlin.phase0.spec_validation.v1``. Nothing here writes capsules.

Subsets (``--source corpus_193``, ``--case-id``, ``--limit``) run and record only the
named cases; their record is at most ``incomplete`` and cannot satisfy derivation.
Arguments after ``--`` go to the suite runner unchanged (``--repo``, ``--m2m-dir``,
``--m2m-python``, ``--target``, ``--device-package``, ``--facts``, ``--workers``...).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from merlin_experiments.phase0 import spec_validation as SV


def _subset(selection, args):
    keys = [(row["source"], SV.case_key(row["case"])) for row in selection["cases"]]
    if args.source:
        keys = [key for key in keys if key[0] in set(args.source)]
    if args.case_id:
        unknown = sorted(set(args.case_id) - {key[1] for key in keys})
        if unknown:
            raise SystemExit(f"unknown --case-id values: {unknown[:5]}")
        keys = [key for key in keys if key[1] in set(args.case_id)]
    if args.limit:
        keys = [keys[int(i * len(keys) / args.limit)] for i in range(min(args.limit, len(keys)))]
    subset = bool(args.source or args.case_id or args.limit)
    return keys if subset else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, required=True, help="v1 case selection JSON")
    parser.add_argument("--corpus-193", type=Path, help="canonical Core ATen corpus suite file")
    parser.add_argument("--bounded", type=Path, help="bounded Core ATen suite file")
    parser.add_argument("--output", type=Path, required=True, help="runner output root (captures and shards)")
    parser.add_argument("--label", default="spec-validation")
    parser.add_argument("--record", type=Path, help="record path (default <output>/<label>/record.json)")
    parser.add_argument("--software-spec", type=Path, required=True)
    parser.add_argument("--host-capabilities", type=Path, required=True)
    parser.add_argument("--source", action="append", choices=("corpus_193", "bounded"))
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--limit", type=int, help="evenly spread smoke subset")
    parser.add_argument("--record-only", action="store_true", help="judge existing shards; run nothing")
    parser.add_argument("--receipt-policy", choices=SV.RECEIPT_POLICIES, default="execution_evidence")
    parser.add_argument("--no-reverify-evidence", action="store_true")
    parser.add_argument("--truth-audit", action="store_true", help="add the non-gating functional-truth column")
    parser.add_argument("--quantization-scheme", default=SV.DEFAULT_QUANTIZATION_SCHEME)
    args, runner_args = parser.parse_known_args(argv)
    if runner_args[:1] == ["--"]:
        runner_args = runner_args[1:]
    admission_parser = argparse.ArgumentParser(add_help=False)
    admission_parser.add_argument("--target", required=True)
    admission_parser.add_argument("--facts", type=Path, required=True)
    admission_args, runner_args = admission_parser.parse_known_args(runner_args)
    suites = {source: path for source, path in (("corpus_193", args.corpus_193), ("bounded", args.bounded)) if path}
    selection = SV.load_selection(args.selection, suites)
    keys = _subset(selection, args)
    base = args.output / args.label
    base.mkdir(parents=True, exist_ok=True)
    suite = base / "selection-suite.json"
    payload = json.dumps(SV.runner_suite(selection), sort_keys=True).encode()
    if not suite.is_file() or suite.read_bytes() != payload:
        suite.write_bytes(payload)
    if not args.record_only:
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_bounded_core_aten_suite.py"),
            "--suite",
            str(suite),
            "--output",
            str(args.output),
            "--label",
            args.label,
            "--numeric-policy",
            SV.NUMERIC_POLICY,
            "--software-spec",
            str(args.software_spec),
            "--host-capabilities",
            str(args.host_capabilities),
            "--target",
            admission_args.target,
            "--facts",
            str(admission_args.facts),
            "--skip-evaluation",
            *runner_args,
        ]
        if keys is not None:
            (base / "selected-case-ids.json").write_text(
                json.dumps([f"{source}:{case_id}" for source, case_id in keys]) + "\n"
            )
            command += ["--case-ids", str(base / "selected-case-ids.json")]
        subprocess.run(command, check=True, env=dict(os.environ))
    record = SV.prepare(
        selection,
        args.output,
        label=args.label,
        software_spec=args.software_spec,
        host_capabilities=args.host_capabilities,
        rtl_facts=admission_args.facts,
        target=admission_args.target,
        case_keys=keys,
        receipt_policy=args.receipt_policy,
        reverify_evidence=not args.no_reverify_evidence,
        with_truth_audit=args.truth_audit,
        quantization_scheme=args.quantization_scheme,
    )
    destination = SV.write(record, args.record or base / "record.json")
    print(
        json.dumps(
            {
                "record": str(destination),
                "sha256": record["sha256"],
                "verdict": record["verdict"],
                "verdict_counts": record["verdict_counts"],
                "mismatches": len(record["mismatches"]),
                "unmodeled": len(record["unmodeled"]),
            },
            indent=2,
        )
    )
    return 0 if record["verdict"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
