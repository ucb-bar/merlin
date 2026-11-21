"""Produce certificates consumed by ``merlin.development.backend.optimization.measurement.gsim_gate``.

Legacy cross-validation JSON is intentionally not upgraded: it did not record the shared ELF digest or
the simulator/model/FIRRTL identities.  Cases must be captured again through :func:`capture_case`, which
builds one ELF and passes that same path to both registered backend engines.  Certificate production is
then a pure validation/assembly step over those v1 captures and an independently sealed model-build
receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.measurement import gsim_workload as WORKLOAD

# ---- gsim_certificate --------------------------------------------------------------------------


def capture_case(
    *,
    target: str,
    capsule_manifest: str | Path,
    artifact_dir: str | Path,
    workdir: str | Path,
    artifacts: ArtifactPaths,
    timeout: int = 3600,
    reference_timeout: int | None = None,
    backend: Any = None,
    build_elf: Callable[..., Path] | None = None,
    readback: str | None = None,
) -> dict[str, Any]:
    """Build one ELF and run that exact file on Verilator and GSIM through the backend seam.

    ``readback`` (default ``MERLIN_PHASE2_GSIM_READBACK``, itself ``digest``): with ``digest`` the shared
    ELF's grader harness prints one XXH64 per output instead of every value -- a full-value readback on
    the reference engine (about 210 cycles/s) is days for a performance-scale output. The values are read
    once on Spike from the full-value build of the same program, must match the reference there, and
    BOTH engines' digests must equal the digest of those values. ``full`` keeps the historical text frame.
    Each engine's ``cycles`` are recorded either way (engine qualification compares them).

    THE TWO ENGINES NEED DIFFERENT DEADLINES, and giving them one is how a capture dies on a member
    that was working. The reference engine executes a small multiple of a hundred cycles a second
    while the candidate engine is more than an order of magnitude faster, so a deadline sized for
    the fast one kills the slow one on exactly the deep members a certificate is most wanted for:
    measured here, a deep-K member ran to the 3600 s default on the reference leg, under an outer
    cap of six hours that never got to apply. ``reference_timeout`` sizes the reference leg on its
    own; left None it falls back to ``timeout``, which is the previous behaviour and is stated
    rather than silently reintroduced. Keeping the candidate leg tight stays useful: a run there
    that takes an hour is hung, not slow.

    Injection points keep the smoke test offline.  The default path uses the same contract compiler and
    backend ``run_elf``/``parse_output`` methods as capsule grading.
    """
    from merlin.development.backend.optimization.measurement.gsim_certificate_receipts import (
        CAPTURE_SCHEMA,
        ArtifactPaths,
        _semantic_oracle,
        _sha_file,
    )

    manifest_path = Path(capsule_manifest).resolve(strict=True)
    artifact_path = Path(artifact_dir).resolve(strict=True)
    cb_path, llvm_path = artifact_path / "command_buffer.json", artifact_path / "lowered.llvm.mlir"
    if not cb_path.is_file() or not llvm_path.is_file():
        raise WORKLOAD.ProducerError(f"{artifact_path}: command_buffer.json/lowered.llvm.mlir are required")
    cb, expected, matches_expected, semantic_reference = _semantic_oracle(
        manifest_path, WORKLOAD._load_mapping(cb_path)
    )
    llvm_text = llvm_path.read_text(encoding="utf-8")
    case_work = Path(workdir)
    case_work.mkdir(parents=True, exist_ok=True)
    from merlin.development.backend.optimization.measurement import gsim_workload as DIGEST
    from merlin.target.contract.harness import readback_policy as RB

    if readback is None:
        import inspect

        # An injected builder that cannot build the digest harness can only take the full-value capture.
        accepts = build_elf is None or "readback_policy" in inspect.signature(build_elf).parameters
        readback = DIGEST.mode() if accepts else "full"
    if readback not in ("digest", "full"):
        raise WORKLOAD.ProducerError(f"capture readback must be 'digest' or 'full', got {readback!r}")
    digest_policy = RB.ReadbackPolicy(RB.OUT_DIGEST_V1)
    if build_elf is None:
        from merlin.development.targetgen.execution.runner.oracle_run_compile import compile_lowered_to_elf

        def build_elf(buffer: Mapping[str, Any], llvm: str, destination: Path, readback_policy=None) -> Path:
            destination.mkdir(parents=True, exist_ok=True)
            return Path(
                compile_lowered_to_elf(buffer, llvm, destination, target=target, readback_policy=readback_policy)
            )

    if readback == "digest":
        elf = Path(build_elf(cb, llvm_text, case_work / "digest", readback_policy=digest_policy)).resolve(strict=True)
    else:
        elf = Path(build_elf(cb, llvm_text, case_work)).resolve(strict=True)
    elf_digest = _sha_file(elf)
    if backend is None:
        from merlin.runtime.backends import registry as backends

        backend = backends.get_backend(target)
    from merlin.analysis.perf.measurement import execution_traces as STORE
    from merlin.runtime.backends import registry as backends
    from merlin.runtime.engine.command_buffer import declared_output_dtypes

    pins = artifacts.pinned()
    # A CAPTURE IS A PURE FUNCTION OF THE ELF AND THE ENGINES, so one already taken for these exact
    # bytes and pins answers this call. The reference leg is the expensive half by more than an order
    # of magnitude -- a 90-case certificate cost over ninety minutes of serial reference simulation in
    # front of a run that could not begin without it -- and every second of it was being re-paid.
    #
    # The workload identity is re-derived and compared before a hit is used: two capsules can compile
    # to one ELF, and a stored document must not answer for a capsule whose declared workload differs
    # from the one being asked about. The identity fields are then taken from THIS call, because the
    # measurement is about the ELF while the manifest binding is about which capsule asked.
    _workload = WORKLOAD.derive_workload(manifest_path)
    _identity = GATE.workload_sha256(_workload)
    _hit = STORE.lookup(
        target, elf_sha256=elf_digest, pins=pins, workload_sha256=_identity, semantic_reference=semantic_reference
    )
    if _hit is not None:
        return {
            **_hit,
            "target": target,
            "capsule": str(WORKLOAD._load_mapping(manifest_path, yaml_input=True).get("name") or ""),
            "capsule_manifest_path": str(manifest_path),
            "capsule_manifest_sha256": _sha_file(manifest_path),
            "workload": _workload,
            "workload_sha256": _identity,
        }
    values = None
    readback_record: dict[str, Any] = {"mode": readback}
    if readback == "digest":
        # THE VALUES, read once where reading them is cheap, from the full-value build of this program.
        values_elf = Path(build_elf(cb, llvm_text, case_work / "spike_full_values")).resolve(strict=True)
        if not backend.available("spike"):
            raise WORKLOAD.ProducerError("spike is unavailable; digest capture needs its full-value readback")
        try:
            values, _ = backend.parse_output(backend.run_elf(values_elf, simulator="spike", timeout=timeout))
        except Exception as exc:  # noqa: BLE001 - backend result is untrusted evidence
            raise WORKLOAD.ProducerError(f"spike full-value readback failed: {exc}") from exc
        values = backends.decode_float_readback(values, declared_output_dtypes(cb))
        if not matches_expected(values):
            raise WORKLOAD.ProducerError("spike did not produce the reference output")
        readback_record.update(
            policy=digest_policy.record(), values_engine="spike", values_elf_sha256=_sha_file(values_elf)
        )
    runs = {}
    for side, engine, binary_pin, firrtl_pin in (
        ("reference", GATE.REFERENCE_ENGINE, "verilator_binary", "verilator_firrtl"),
        ("candidate", GATE.GSIM_ENGINE, "gsim_binary", "gsim_firrtl"),
    ):
        if not backend.available(engine):
            raise WORKLOAD.ProducerError(f"{engine} is unavailable; absent execution is not agreement")
        if _sha_file(elf) != elf_digest:
            raise WORKLOAD.ProducerError("shared ELF changed before the second engine ran")
        deadline = (
            timeout if side != "reference" else (timeout if reference_timeout is None else int(reference_timeout))
        )
        console = backend.run_elf(elf, simulator=engine, timeout=deadline)
        if _sha_file(elf) != elf_digest:
            raise WORKLOAD.ProducerError(f"shared ELF changed while {engine} ran")
        try:
            outputs, raw_metrics = backend.parse_output(console)
        except Exception as exc:  # noqa: BLE001 - backend result is untrusted evidence
            raise WORKLOAD.ProducerError(f"{engine} console is not gradeable: {exc}") from exc
        digests = None
        if values is not None:
            try:
                roster = RB.require_digest_roster(cb, str(console), outputs)
                mismatched = RB.digest_mismatches(roster, values)
            except ValueError as exc:
                raise WORKLOAD.ProducerError(f"{engine} digest readback is not gradeable: {exc}") from exc
            if mismatched:
                raise WORKLOAD.ProducerError(f"{engine} output digests differ from the Spike-verified values")
            digests = {name: row["digest"] for name, row in roster.items()}
            outputs = values
        else:
            # Bare-metal harnesses print a float destination's exact container word. Decode from the
            # command buffer's declared dtype before applying the capsule's numeric policy; integer
            # outputs and backends that already return floats remain byte-identical.
            outputs = backends.decode_float_readback(outputs, declared_output_dtypes(cb))
            if not matches_expected(outputs):
                raise WORKLOAD.ProducerError(f"{engine} did not produce the reference output")
        output_digest, output_rows = WORKLOAD.encode_declared_outputs(outputs, cb)
        runs[side] = {
            "engine": engine,
            "ran": True,
            "verdict": "pass",
            "elf_sha256": elf_digest,
            "binary_sha256": pins[binary_pin]["sha256"],
            "firrtl_sha256": pins[firrtl_pin]["sha256"],
            "derived_from_rtl": True,
            "cycle_accurate": True,
            "output_sha256": output_digest,
            "output_encoding": WORKLOAD.OUTPUT_ENCODING,
            "output_tensors": output_rows,
            "console_sha256": WORKLOAD._sha_bytes(str(console).encode("utf-8")),
            "cycles": raw_metrics.get("cycles") if isinstance(raw_metrics, Mapping) else None,
            "readback": {**readback_record, **({"output_digests": digests} if digests is not None else {})},
        }
        if engine == GATE.GSIM_ENGINE:
            runs[side]["model_sha256"] = pins["gsim_model"]["sha256"]
    if runs["reference"]["output_sha256"] != runs["candidate"]["output_sha256"]:
        raise WORKLOAD.ProducerError("GSIM and Verilator output bytes differ")
    workload = WORKLOAD.derive_workload(manifest_path)
    document = {
        "schema_version": CAPTURE_SCHEMA,
        "target": target,
        "capsule": str(WORKLOAD._load_mapping(manifest_path, yaml_input=True).get("name") or ""),
        "capsule_manifest_path": str(manifest_path),
        "capsule_manifest_sha256": _sha_file(manifest_path),
        "workload": workload,
        "workload_sha256": GATE.workload_sha256(workload),
        "semantic_reference": semantic_reference,
        "elf_sha256": elf_digest,
        "agreement": "AGREE",
        "evidence": GATE.STRONG_EVIDENCE,
        "bytes_match": True,
        **runs,
    }
    # File it so the next run that presents these bytes and these engines does not pay again. The
    # store REFUSES a document that does not answer for the key it is filed under, so a failure here
    # loses a cache entry and nothing else -- the capture is returned either way.
    STORE.store(target, elf_sha256=elf_digest, pins=pins, document=document)
    return document


def validate_capture(path: str | Path, *, target: str, pins: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    from merlin.development.backend.optimization.measurement.gsim_certificate_receipts import CAPTURE_SCHEMA, _sha_file

    doc = dict(WORKLOAD._load_mapping(Path(path)))
    if doc.get("schema_version") != CAPTURE_SCHEMA or doc.get("target") != target:
        raise WORKLOAD.ProducerError(f"{path}: not a v1 capture for target {target!r}")
    identity = GATE.workload_sha256(doc.get("workload"))
    if doc.get("workload_sha256") != identity:
        raise WORKLOAD.ProducerError(f"{path}: workload commitment is invalid")
    # Reuse the gate's definitive same-ELF/pin/evidence validator rather than maintain a second policy.
    _, member = GATE._validate_member(doc, pins=pins, index=0)
    manifest_sha = doc.get("capsule_manifest_sha256")
    if not GATE._is_sha256(manifest_sha):
        raise WORKLOAD.ProducerError(f"{path}: capsule manifest digest is absent")
    manifest_path = Path(str(doc.get("capsule_manifest_path") or ""))
    if manifest_path.is_symlink() or not manifest_path.is_file() or _sha_file(manifest_path) != manifest_sha:
        raise WORKLOAD.ProducerError(f"{path}: frozen capsule manifest is absent or changed")
    if WORKLOAD.derive_workload(manifest_path) != doc.get("workload"):
        raise WORKLOAD.ProducerError(f"{path}: workload was not derived from the bound capsule manifest")
    reference_output = (doc.get("reference") or {}).get("output_sha256")
    candidate_output = (doc.get("candidate") or {}).get("output_sha256")
    if not GATE._is_sha256(reference_output) or reference_output != candidate_output:
        raise WORKLOAD.ProducerError(f"{path}: engines lack identical output-byte digests")
    reference_rows = (doc.get("reference") or {}).get("output_tensors")
    candidate_rows = (doc.get("candidate") or {}).get("output_tensors")
    if (
        (doc.get("reference") or {}).get("output_encoding") != WORKLOAD.OUTPUT_ENCODING
        or (doc.get("candidate") or {}).get("output_encoding") != WORKLOAD.OUTPUT_ENCODING
        or not isinstance(reference_rows, list)
        or reference_rows != candidate_rows
    ):
        raise WORKLOAD.ProducerError(f"{path}: engines lack matching declared-tensor byte encodings")
    return member


def produce_certificate(
    *, target: str, captures: Sequence[str | Path], artifacts: ArtifactPaths, build_receipt: str | Path
) -> dict[str, Any]:
    """Assemble a certificate only from new v1 captures and a complete build-lineage receipt."""
    from merlin.development.backend.optimization.measurement.gsim_certificate_receipts import (
        ArtifactPaths,
        validate_build_receipt,
    )

    if not captures:
        raise WORKLOAD.ProducerError("no v1 cross-validation captures were supplied")
    pins = artifacts.pinned()
    receipt = validate_build_receipt(build_receipt, pins=pins)
    members = [validate_capture(path, target=target, pins=pins) for path in captures]
    identities = [member["workload_sha256"] for member in members]
    if len(set(identities)) != len(identities):
        raise WORKLOAD.ProducerError("v1 captures contain duplicate workload identities")
    return {
        "schema_version": GATE.SCHEMA_VERSION,
        "status": "certified",
        "target": target,
        "fidelity": GATE.FIDELITY,
        "primary_engine": GATE.GSIM_ENGINE,
        "reference_engine": GATE.REFERENCE_ENGINE,
        "pins": pins,
        "build_binding": receipt,
        "members": sorted(members, key=lambda item: item["workload_sha256"]),
        "unresolved": [],
    }


def smoke_legacy_evidence(
    *,
    target: str,
    legacy_root: str | Path,
    v1_capture_root: str | Path | None,
    artifacts: ArtifactPaths,
    build_receipt: str | Path | None,
) -> dict[str, Any]:
    """Offline readiness report.  It never promotes legacy rows into v1 captures."""
    from merlin.development.backend.optimization.measurement.gsim_certificate_receipts import (
        CAPTURE_SCHEMA,
        REFUSAL_SCHEMA,
        ArtifactPaths,
        _firrtl_circuit,
        model_top,
        validate_build_receipt,
    )

    issues = []
    legacy = [row.to_dict() for row in GATE.discover_cross_validation_reports([legacy_root], target=target)]
    capture_paths = []
    if v1_capture_root is not None and Path(v1_capture_root).is_dir():
        capture_paths = sorted(Path(v1_capture_root).rglob("*.json"), key=str)
        capture_paths = [
            path for path in capture_paths if WORKLOAD._load_mapping(path).get("schema_version") == CAPTURE_SCHEMA
        ]
    if not capture_paths:
        issues.append("no v1 same-ELF capture records; legacy xval JSON cannot supply missing fields")
    try:
        pins = artifacts.pinned()
    except (OSError, WORKLOAD.ProducerError) as exc:
        pins = {}
        issues.append(f"artifact pin failure: {exc}")
    if pins:
        firrtl_top = _firrtl_circuit(Path(pins["gsim_firrtl"]["path"]))
        generated_top = model_top(pins["gsim_model"]["path"])
        if firrtl_top is not None and generated_top is not None and firrtl_top != generated_top:
            issues.append(
                f"GSIM FIRRTL top {firrtl_top!r} does not match sealed generated-model top "
                f"{generated_top!r}; no build receipt may bind this pair"
            )
    if build_receipt is None or not Path(build_receipt).is_file():
        issues.append("no sealed GSIM build receipt binding FIRRTL + model manifest + binary")
    elif pins:
        try:
            validate_build_receipt(build_receipt, pins=pins)
        except WORKLOAD.ProducerError as exc:
            issues.append(str(exc))
    return {
        "schema_version": REFUSAL_SCHEMA,
        "status": "ready" if not issues else "refused",
        "target": target,
        "legacy_evidence": legacy,
        "artifact_pins": pins,
        "v1_capture_count": len(capture_paths),
        "issues": issues,
        "rule": "GSIM remains primary final timing; Verilator is correctness corroboration only",
    }


def _add_artifact_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gsim-firrtl", required=True)
    parser.add_argument("--verilator-firrtl", required=True)
    parser.add_argument("--gsim-model-manifest", required=True)
    parser.add_argument("--gsim-binary", required=True)
    parser.add_argument("--verilator-binary", required=True)


def _parser() -> argparse.ArgumentParser:
    from merlin.development.backend.optimization.measurement.gsim_certificate_receipts import (
        FIRRTL_BOUNDARY_ADOPTED,
        FIRRTL_BOUNDARY_ELABORATED,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="action", required=True)
    manifest = commands.add_parser("model-manifest", help="seal an explicit generated-source set")
    manifest.add_argument("--model-root", required=True)
    manifest.add_argument("--file", action="append", required=True)
    manifest.add_argument("--output", required=True)

    receipt = commands.add_parser("build-receipt", help="seal complete native-build lineage")
    receipt.add_argument("--firrtl", required=True)
    receipt.add_argument("--model-manifest", required=True)
    receipt.add_argument("--binary", required=True)
    receipt.add_argument("--emitter", required=True)
    receipt.add_argument("--cxx-wrapper", required=True)
    receipt.add_argument("--cxx-compiler", required=True)
    receipt.add_argument("--input", action="append", required=True, metavar="ROLE=PATH")
    receipt.add_argument("--commands", required=True, help="JSON list of ordered {stage,cwd,argv} records")
    receipt.add_argument(
        "--firrtl-boundary",
        choices=(FIRRTL_BOUNDARY_ELABORATED, FIRRTL_BOUNDARY_ADOPTED),
        default=FIRRTL_BOUNDARY_ELABORATED,
        help="whether this transcript elaborated FIRRTL or adopted pinned existing bytes",
    )
    receipt.add_argument("--output", required=True)

    capture = commands.add_parser("capture", help="run one same-ELF GSIM/Verilator capture")
    capture.add_argument("--target", required=True)
    capture.add_argument("--capsule-manifest", required=True)
    capture.add_argument("--artifact-dir", required=True)
    capture.add_argument("--workdir", required=True)
    capture.add_argument("--timeout", type=int, default=3600, help="deadline for the candidate (fast) engine")
    capture.add_argument(
        "--reference-timeout",
        type=int,
        default=None,
        help="deadline for the reference (cycle-accurate, much slower) engine; defaults to --timeout",
    )
    capture.add_argument(
        "--readback",
        choices=("digest", "full"),
        default=None,
        help="output readback on both engines (default MERLIN_PHASE2_GSIM_READBACK, itself digest)",
    )
    _add_artifact_args(capture)
    capture.add_argument("--output", required=True)

    certificate = commands.add_parser("certificate", help="assemble strict v1 certificate")
    certificate.add_argument("--target", required=True)
    certificate.add_argument("--capture", action="append", required=True)
    certificate.add_argument("--build-receipt", required=True)
    _add_artifact_args(certificate)
    certificate.add_argument("--output", required=True)

    smoke = commands.add_parser("smoke", help="fail-closed legacy/readiness report")
    smoke.add_argument("--target", required=True)
    smoke.add_argument("--legacy-root", required=True)
    smoke.add_argument("--v1-capture-root")
    _add_artifact_args(smoke)
    smoke.add_argument("--build-receipt")
    smoke.add_argument("--output", required=True)
    return parser


def _artifact_paths(args: argparse.Namespace) -> ArtifactPaths:
    from merlin.development.backend.optimization.measurement.gsim_certificate_receipts import ArtifactPaths

    return ArtifactPaths(
        *(
            Path(value)
            for value in (
                args.gsim_firrtl,
                args.verilator_firrtl,
                args.gsim_model_manifest,
                args.gsim_binary,
                args.verilator_binary,
            )
        )
    )


def _role_path(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise WORKLOAD.ProducerError(f"build input must be ROLE=PATH, got {value!r}")
    return role, Path(path)


def main(argv: list[str] | None = None) -> int:
    from merlin.development.backend.optimization.measurement.gsim_certificate_receipts import (
        write_build_receipt,
        write_model_manifest,
    )

    values = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original smoke-only CLI while exposing explicit operational subcommands.
    if values and values[0] not in {
        "model-manifest",
        "build-receipt",
        "capture",
        "certificate",
        "smoke",
        "-h",
        "--help",
    }:
        values.insert(0, "smoke")
    args = _parser().parse_args(values)
    if args.action == "model-manifest":
        write_model_manifest(args.model_root, args.file, args.output)
        return 0
    if args.action == "build-receipt":
        command_doc = WORKLOAD._load_mapping(Path(args.commands))
        command_rows = command_doc.get("commands")
        if not isinstance(command_rows, list):
            raise WORKLOAD.ProducerError("command transcript JSON must contain a commands list")
        write_build_receipt(
            output=args.output,
            firrtl=args.firrtl,
            model_manifest=args.model_manifest,
            binary=args.binary,
            emitter=args.emitter,
            cxx_wrapper=args.cxx_wrapper,
            cxx_compiler=args.cxx_compiler,
            inputs=[_role_path(value) for value in args.input],
            commands=command_rows,
            firrtl_boundary=args.firrtl_boundary,
        )
        return 0
    artifacts = _artifact_paths(args)
    if args.action == "capture":
        report = capture_case(
            target=args.target,
            capsule_manifest=args.capsule_manifest,
            artifact_dir=args.artifact_dir,
            workdir=args.workdir,
            artifacts=artifacts,
            timeout=args.timeout,
            reference_timeout=args.reference_timeout,
            readback=args.readback,
        )
    elif args.action == "certificate":
        report = produce_certificate(
            target=args.target, captures=args.capture, artifacts=artifacts, build_receipt=args.build_receipt
        )
    else:
        report = smoke_legacy_evidence(
            target=args.target,
            legacy_root=args.legacy_root,
            v1_capture_root=args.v1_capture_root,
            artifacts=artifacts,
            build_receipt=args.build_receipt,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(GATE.canonical_json(report) + "\n", encoding="utf-8")
    print(GATE.canonical_json(report))
    return 0 if args.action != "smoke" or report["status"] == "ready" else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
