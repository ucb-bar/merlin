#!/usr/bin/env python3
"""Run Phase 0 spec validation over a Core ATen case selection and seal its record.

The reference backend executes every selected case through
``run_bounded_core_aten_suite.py`` under ``--numeric-policy spec_int8`` (resumable:
completed shards with the same case digest and provenance stamp are reused).
``--component-host-capabilities`` is required for suite execution and per-case
expectations. ``--host-capabilities`` selects the descriptor host lane bound into
the record for derivation; both document paths and digests are recorded. The
selection resolves cases from byte-pinned ``--corpus-193`` and/or ``--bounded``
suite files. The shards are then judged against the spec's own admission decisions and written as
``merlin.phase0.spec_validation.v1``. Nothing here writes capsules.

MERLIN_SPEC_VALIDATION_ATTESTATION selects operator-authorized shard reuse,
ATTESTED, not proven. It probes only the stamp, never launches suite work (also
with --record-only), and seals the receipt and its disclosed residual risks.

Subsets (``--source corpus_193``, ``--case-id``, ``--limit``) run and record only the
named cases; their record is at most ``incomplete`` and cannot satisfy derivation.
Arguments after ``--`` go to the suite runner unchanged (``--repo``, ``--m2m-dir``,
``--m2m-python``, ``--target``, ``--device-package``, ``--facts``, ``--workers``...).
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
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


class _StampCaptured(BaseException):
    def __init__(self, block):
        self.block = block


def _probe_stamp(runner, argv):
    """Compute the current stamp, then stop before provenance writes or execution."""
    from merlin_experiments.phase0.evidence_attestation import AttestationRefused

    old_argv = sys.argv
    try:
        spec = importlib.util.spec_from_file_location("_spec_attestation_runner", runner)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original = module.stamp

        def capture(args):
            raise _StampCaptured(original(args))

        module.stamp = capture
        sys.argv = [str(runner), *argv]
        try:
            if inspect.signature(module.main).parameters:
                module.main(argv)
            else:
                module.main()
        except _StampCaptured as captured:
            return captured.block
        finally:
            module.stamp = original
        raise AttestationRefused("runner main returned without calling stamp")
    except SystemExit as exc:
        raise AttestationRefused(f"runner exited before stamp sentinel: SystemExit({exc.code!r})") from exc
    finally:
        sys.argv = old_argv


def _attestation_reuse(command, attestation, base):
    from merlin_experiments.phase0 import evidence_attestation as EA

    provenance = base / "provenance.json"
    EA.require(provenance.is_file(), f"{provenance}: attestation runner provenance is missing")
    raw = provenance.read_bytes()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", type=Path, required=True)
    forwarded, _ = parser.parse_known_args(command[2:])
    try:
        current = _probe_stamp(Path(command[1]), command[2:])
    finally:
        if not provenance.is_file() or provenance.read_bytes() != raw:
            provenance.write_bytes(raw)
            raise EA.AttestationRefused("stamp probe modified provenance.json; saved bytes restored")
    receipt = EA.decide(json.loads(raw), current, repo=forwarded.repo, attestation=attestation)
    path = base / "evidence-attestation.json"
    SV.write(receipt, path)
    print(f"ATTESTED, not proven: reuse shards at {receipt['shard_commit']} for record at {receipt['current_commit']}")
    return path


def _bind_attestation(record, path):
    """Seal the reviewer-visible receipt binding without changing producer imports."""
    from merlin_experiments.phase0 import evidence_attestation as EA

    raw = path.read_bytes()
    receipt = json.loads(raw)
    EA.require(receipt.get("schema") == EA.RECEIPT_SCHEMA and receipt.get("decision") == "attested_reuse_of_shards"
               and receipt.get("status") == "ATTESTED_NOT_PROVEN",
               "invalid evidence attestation receipt")
    EA.require(receipt.get("saved_stamp_sha256") == record.get("runner_provenance_sha256"),
               "evidence attestation receipt differs from record runner stamp")
    record["evidence_attestation"] = {
        "status": "ATTESTED_NOT_PROVEN",
        "receipt_sha256": EA.sha256(raw),
        "shard_commit": receipt["shard_commit"],
        "current_commit": receipt["current_commit"],
    }
    record["sha256"] = SV.digest(record)


def _runner_help_requested(arguments):
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if argument.startswith("-h") and not argument.startswith("--"):
            return True
        if option.startswith("--") and len(option) > 2 and "--help".startswith(option):
            return True
    return False


def _retained_input(path, payload, active):
    if active:
        from merlin_experiments.phase0.evidence_attestation import require

        require(path.is_file() and path.read_bytes() == payload,
                f"{path}: attestation requires the identical retained runner input")
    elif not path.is_file() or path.read_bytes() != payload:
        path.write_bytes(payload)


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, required=True, help="v1 case selection JSON")
    parser.add_argument("--corpus-193", type=Path, help="canonical Core ATen corpus suite file")
    parser.add_argument("--bounded", type=Path, help="bounded Core ATen suite file")
    parser.add_argument("--output", type=Path, required=True, help="runner output root (captures and shards)")
    parser.add_argument("--label", default="spec-validation")
    parser.add_argument("--record", type=Path, help="record path (default <output>/<label>/record.json)")
    parser.add_argument("--software-spec", type=Path, required=True)
    parser.add_argument(
        "--host-capabilities", type=Path, required=True, help="descriptor host lane bound for derivation"
    )
    parser.add_argument(
        "--component-host-capabilities",
        type=Path,
        required=True,
        help="component host lane used by the suite and expectations",
    )
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
    attestation = os.environ.get("MERLIN_SPEC_VALIDATION_ATTESTATION")
    active = attestation is not None
    if active:
        from merlin_experiments.phase0.evidence_attestation import require

        require(not _runner_help_requested(runner_args),
                "attestation refuses runner help flags")
    if any(arg.split("=", 1)[0] == "--host-capabilities" for arg in runner_args):
        parser.error("use --component-host-capabilities before -- to select the suite host lane")
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
    receipt = None
    if active:
        from merlin_experiments.phase0.evidence_attestation import require

        require((base / "provenance.json").is_file(), f"{base / 'provenance.json'}: attestation runner provenance is missing")
        controlled = {"--suite", "--output", "--label", "--numeric-policy", "--software-spec",
                      "--case-ids", "--worker", "--case", "--evaluate-only", "--help", "-h"}
        require(not any(arg.startswith("--") and any(option.startswith(arg.split("=", 1)[0])
                        for option in controlled) for arg in runner_args),
                "attestation refuses runner control overrides")
    _retained_input(suite, payload, active)
    if not args.record_only or active:
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
            str(args.component_host_capabilities),
            "--target",
            admission_args.target,
            "--facts",
            str(admission_args.facts),
            "--skip-evaluation",
            *runner_args,
        ]
        if keys is not None:
            _retained_input(
                base / "selected-case-ids.json",
                (json.dumps([f"{source}:{case_id}" for source, case_id in keys]) + "\n").encode(),
                active,
            )
            command += ["--case-ids", str(base / "selected-case-ids.json")]
        if active:
            receipt = _attestation_reuse(command, attestation, base)
        else:
            subprocess.run(command, check=True, env=dict(os.environ))
    record = SV.prepare(
        selection,
        args.output,
        label=args.label,
        software_spec=args.software_spec,
        host_capabilities=args.host_capabilities,
        component_host_capabilities=args.component_host_capabilities,
        rtl_facts=admission_args.facts,
        target=admission_args.target,
        case_keys=keys,
        receipt_policy=args.receipt_policy,
        reverify_evidence=not args.no_reverify_evidence,
        with_truth_audit=args.truth_audit,
        quantization_scheme=args.quantization_scheme,
    )
    if receipt is not None:
        _bind_attestation(record, receipt)
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


def main(argv=None) -> int:
    if "MERLIN_SPEC_VALIDATION_ATTESTATION" not in os.environ:
        return _main(argv)
    from merlin_experiments.phase0.evidence_attestation import AttestationRefused

    try:
        return _main(argv)
    except (AttestationRefused, OSError, ValueError, TypeError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
