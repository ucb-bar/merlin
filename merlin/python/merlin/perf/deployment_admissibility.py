"""Fail-closed, target-neutral checks for deploying one emitted program.

The compiler search loop may optimize without this optional evidence.  Once a deployment profile is
explicitly selected, however, promotion must bind the target contract, configuration, runtime
interface, and deployable image to exact bytes; prove that every emitted egress can be represented by
the physical readout path; and prove that the measurement wrapper orders warmup, completion, counter
reset, measurement, completion, and validation correctly.

This module interprets only the supplied profile and evidence.  It contains no target identities,
operation names, geometries, or encoding assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.common.schemas import validate_or_raise


_REQUIRED_ARTIFACT_ROLES = ("contract", "config", "runtime_header", "bitstream")
_CORE_WRAPPER_EVENTS = ("warm", "reset", "start", "measured", "end")
_HEX = frozenset("0123456789abcdef")


class DeploymentAdmissibilityError(ValueError):
    """Raised when a requested deployment is not proved admissible."""


def _canonical_json(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode("utf-8")


def _file_sha256(path: Path) -> str:
    """Hash an artifact without materializing a potentially large deployment image in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_profile_sha256(profile: Mapping[str, Any]) -> str:
    """Return the stable content identity used to pin a deployment profile."""
    return hashlib.sha256(_canonical_json(profile)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in _HEX for character in value))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _artifact_check(role: str, raw: object, *, root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    spec = _mapping(raw)
    declared_path = spec.get("path")
    expected = spec.get("sha256")
    check: dict[str, Any] = {"role": role, "path": declared_path, "expected_sha256": expected}
    if not isinstance(declared_path, str) or not declared_path.strip() or not _is_sha256(expected):
        return ({**check, "status": "refused"}, {
            "priority": 10, "status": "refused", "code": "artifact_binding_malformed",
            "message": f"deployment artifact binding {role!r} lacks a path or exact SHA-256",
            "action": f"record the exact {role} file path and SHA-256 in the deployment profile",
        })
    path = Path(declared_path)
    path = path if path.is_absolute() else root / path
    check["resolved_path"] = str(path.resolve(strict=False))
    if path.is_symlink():
        return ({**check, "status": "refused"}, {
            "priority": 11, "status": "refused", "code": "artifact_binding_is_symlink",
            "message": f"deployment artifact {role!r} is a mutable symlink",
            "action": f"bind {role} to a real immutable file",
        })
    if not path.is_file():
        return ({**check, "status": "UNKNOWN"}, {
            "priority": 12, "status": "UNKNOWN", "code": "artifact_unavailable",
            "message": f"deployment artifact {role!r} is unavailable for identity verification",
            "action": f"make the pinned {role} file readable and rerun admissibility",
        })
    try:
        actual = _file_sha256(path)
    except OSError as exc:
        return ({**check, "status": "UNKNOWN", "reason": str(exc)}, {
            "priority": 12, "status": "UNKNOWN", "code": "artifact_unreadable",
            "message": f"deployment artifact {role!r} could not be read",
            "action": f"make the pinned {role} file readable and rerun admissibility",
        })
    check["actual_sha256"] = actual
    if actual != expected:
        return ({**check, "status": "refused"}, {
            "priority": 1, "status": "refused", "code": "artifact_identity_mismatch",
            "message": f"deployment artifact {role!r} does not match its pinned SHA-256",
            "action": f"use the pinned {role} bytes or issue and review a new deployment profile",
        })
    return {**check, "status": "verified"}, None


def _identity_diagnostic(raw: object, expected: Mapping[str, str], *, source: str
                         ) -> dict[str, Any] | None:
    observed = _mapping(raw)
    if observed != expected:
        return {
            "priority": 2, "status": "refused", "code": "emission_identity_mismatch",
            "message": f"{source} evidence names different emitted program bytes",
            "action": "regenerate the evidence from the exact candidate and emitted artifacts under review",
        }
    return None


def _egress_check(profile: Mapping[str, Any], evidence: object, *, profile_sha256: str,
                  expected_identity: Mapping[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    supported: set[tuple[str, int]] = set()
    malformed_capabilities = 0
    for raw in _rows(profile.get("supported_physical_egress")):
        capability = _mapping(raw)
        encoding, width = capability.get("encoding"), capability.get("width_bits")
        if (not isinstance(encoding, str) or not encoding.strip()
                or not isinstance(width, int) or isinstance(width, bool) or width <= 0):
            malformed_capabilities += 1
            continue
        supported.add((encoding, width))
    if malformed_capabilities or not supported:
        diagnostics.append({
            "priority": 13, "status": "refused", "code": "egress_capabilities_malformed",
            "message": "deployment profile has no complete physical egress capability set",
            "action": "declare each supported physical egress encoding and positive bit width",
        })

    document = _mapping(evidence)
    rows: list[dict[str, Any]] = []
    if not document:
        diagnostics.append({
            "priority": 30, "status": "UNKNOWN", "code": "egress_evidence_missing",
            "message": "no emitted representation/readout evidence was supplied",
            "action": "derive physical egress evidence from the exact emitted artifact and readout path",
        })
    else:
        if document.get("schema") != "physical_egress_evidence_v1":
            diagnostics.append({
                "priority": 14, "status": "refused", "code": "egress_evidence_schema_mismatch",
                "message": "physical egress evidence uses an unsupported schema",
                "action": "emit physical_egress_evidence_v1 from the host-owned artifact analyzer",
            })
        if document.get("derivation_status") != "verified":
            diagnostics.append({
                "priority": 30, "status": "UNKNOWN", "code": "egress_derivation_unverified",
                "message": "physical egress rows were not structurally derived from the emitted artifact",
                "action": "derive every egress from the exact host-owned emitted-artifact analysis",
            })
        if document.get("coverage_status") != "complete":
            diagnostics.append({
                "priority": 30, "status": "UNKNOWN", "code": "egress_coverage_incomplete",
                "message": "physical egress evidence does not prove complete externally visible result coverage",
                "action": "enumerate every externally visible emitted result before deployment promotion",
            })
        if document.get("profile_sha256") != profile_sha256:
            diagnostics.append({
                "priority": 2, "status": "refused", "code": "egress_profile_identity_mismatch",
                "message": "physical egress evidence was derived for a different deployment profile",
                "action": "regenerate egress evidence against the exact pinned deployment profile",
            })
        mismatch = _identity_diagnostic(
            document.get("emission_identity"), expected_identity, source="physical egress")
        if mismatch is not None:
            diagnostics.append(mismatch)
        raw_egresses = _rows(document.get("egresses"))
        if not raw_egresses:
            diagnostics.append({
                "priority": 30, "status": "UNKNOWN", "code": "egress_rows_missing",
                "message": "no physical program egresses were verified",
                "action": "enumerate every externally visible emitted result and its physical readout",
            })
        seen_names: set[str] = set()
        for index, raw in enumerate(raw_egresses):
            row = _mapping(raw)
            declared_name = row.get("name")
            name = (declared_name if isinstance(declared_name, str) and declared_name
                    else f"egress[{index}]")
            representation = _mapping(row.get("emitted_representation"))
            readout = _mapping(row.get("physical_readout"))
            representation_pair = (representation.get("encoding"), representation.get("width_bits"))
            readout_pair = (readout.get("encoding"), readout.get("width_bits"))
            complete = isinstance(declared_name, str) and bool(declared_name) and all(
                isinstance(encoding, str) and encoding.strip()
                and isinstance(width, int) and not isinstance(width, bool) and width > 0
                for encoding, width in (representation_pair, readout_pair))
            row_result = {"name": name, "emitted_representation": dict(representation),
                          "physical_readout": dict(readout)}
            if row.get("status") != "verified" or not complete:
                row_result["status"] = "UNKNOWN"
                diagnostics.append({
                    "priority": 31, "status": "UNKNOWN", "code": "egress_row_unverified",
                    "message": f"physical representation/readout for {name!r} is incomplete",
                    "action": "verify both the emitted representation and the physical readout encoding/width",
                })
            elif representation_pair != readout_pair:
                row_result["status"] = "refused"
                diagnostics.append({
                    "priority": 4, "status": "refused",
                    "code": "egress_representation_readout_mismatch",
                    "message": f"emitted representation and physical readout disagree for {name!r}",
                    "action": "emit a representation the readout path consumes without an unproved reinterpretation",
                })
            elif readout_pair not in supported:
                row_result["status"] = "refused"
                diagnostics.append({
                    "priority": 3, "status": "refused", "code": "unsupported_physical_egress",
                    "message": f"physical egress {name!r} uses an encoding/width absent from the deployment profile",
                    "action": "narrow or convert the emitted result, or prove and pin a supporting readout path",
                })
            else:
                row_result["status"] = "verified"
            if name in seen_names:
                row_result["status"] = "refused"
                diagnostics.append({
                    "priority": 4, "status": "refused", "code": "duplicate_physical_egress",
                    "message": f"physical egress name {name!r} is duplicated",
                    "action": "emit one uniquely named evidence row for every external result",
                })
            seen_names.add(name)
            rows.append(row_result)
    status = _component_status(diagnostics)
    return ({"status": "verified" if status == "admitted" else status,
             "supported": [{"encoding": encoding, "width_bits": width}
                           for encoding, width in sorted(supported)], "egresses": rows}, diagnostics)


def _event_names(raw_events: object) -> tuple[list[str], bool]:
    names: list[str] = []
    malformed = False
    for raw in _rows(raw_events):
        value = raw.get("event") if isinstance(raw, Mapping) else raw
        if not isinstance(value, str) or not value:
            malformed = True
        else:
            names.append(value)
    return names, malformed


def _wrapper_check(evidence: object, *, profile_sha256: str,
                   expected_identity: Mapping[str, str], root: Path
                   ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    document = _mapping(evidence)
    artifact: dict[str, Any] = {"status": "UNKNOWN"}
    names: list[str] = []
    if not document:
        diagnostics.append({
            "priority": 40, "status": "UNKNOWN", "code": "wrapper_evidence_missing",
            "message": "no measurement-wrapper event proof was supplied",
            "action": "derive a wrapper event trace from the exact wrapper artifact",
        })
    else:
        if document.get("schema") != "wrapper_event_evidence_v1":
            diagnostics.append({
                "priority": 15, "status": "refused", "code": "wrapper_evidence_schema_mismatch",
                "message": "wrapper event evidence uses an unsupported schema",
                "action": "emit wrapper_event_evidence_v1 from the host-owned wrapper analyzer",
            })
        if document.get("profile_sha256") != profile_sha256:
            diagnostics.append({
                "priority": 2, "status": "refused", "code": "wrapper_profile_identity_mismatch",
                "message": "wrapper event evidence was derived for a different deployment profile",
                "action": "regenerate wrapper evidence against the exact pinned deployment profile",
            })
        mismatch = _identity_diagnostic(
            document.get("emission_identity"), expected_identity, source="wrapper")
        if mismatch is not None:
            diagnostics.append(mismatch)
        artifact, artifact_diagnostic = _artifact_check(
            "wrapper", document.get("wrapper_artifact"), root=root)
        if artifact_diagnostic is not None:
            diagnostics.append(artifact_diagnostic)
        if document.get("derivation_status") != "verified":
            diagnostics.append({
                "priority": 41, "status": "UNKNOWN", "code": "wrapper_derivation_unverified",
                "message": "wrapper events were not structurally verified against the wrapper artifact",
                "action": "parse the wrapper artifact and bind the derived events to its exact SHA-256",
            })
        names, malformed = _event_names(document.get("events"))
        if malformed or not names:
            diagnostics.append({
                "priority": 42, "status": "UNKNOWN", "code": "wrapper_events_unavailable",
                "message": "wrapper event sequence is absent or malformed",
                "action": "emit the ordered wrapper event sequence from structural analysis",
            })
        else:
            positions = {name: [index for index, event in enumerate(names) if event == name]
                         for name in (*_CORE_WRAPPER_EVENTS, "completion", "validation")}
            ambiguous = [name for name in _CORE_WRAPPER_EVENTS if len(positions[name]) != 1]
            if ambiguous:
                diagnostics.append({
                    "priority": 5, "status": "refused", "code": "wrapper_core_event_ambiguous",
                    "message": f"wrapper needs exactly one of each core event; invalid {ambiguous}",
                    "action": "emit one warm, reset, start, measured invocation, and end marker",
                })
            else:
                warm, reset, start, measured, end = (
                    positions[name][0] for name in _CORE_WRAPPER_EVENTS)
                warm_completions = [index for index in positions["completion"]
                                    if warm < index < reset]
                measured_completions = [index for index in positions["completion"]
                                        if measured < index < end]
                if not warm_completions:
                    diagnostics.append({
                        "priority": 5, "status": "refused", "code": "warm_without_completion",
                        "message": "warm invocation is not completed before counter reset",
                        "action": "place a proven completion after warm invocation and before reset",
                    })
                if not (warm < reset < start < measured < end):
                    diagnostics.append({
                        "priority": 5, "status": "refused", "code": "wrapper_event_order_invalid",
                        "message": "wrapper core events are not ordered warm, reset, start, measured, end",
                        "action": "order warmup before reset and enclose only the measured invocation in the window",
                    })
                if not measured_completions:
                    after_end = any(index > end for index in positions["completion"])
                    diagnostics.append({
                        "priority": 5, "status": "refused",
                        "code": ("measured_end_before_completion" if after_end
                                 else "measured_without_completion"),
                        "message": "measurement window ends before measured work has proven completion",
                        "action": "place a proven completion after measured invocation and before end",
                    })
                if not positions["validation"]:
                    diagnostics.append({
                        "priority": 43, "status": "UNKNOWN", "code": "wrapper_validation_missing",
                        "message": "wrapper evidence does not locate output validation",
                        "action": "record validation events and keep them outside the measured window",
                    })
                elif any(start <= index <= end for index in positions["validation"]):
                    diagnostics.append({
                        "priority": 5, "status": "refused", "code": "validation_inside_measurement_window",
                        "message": "output validation is inside the measured compute window",
                        "action": "move validation before start or after end",
                    })
    status = _component_status(diagnostics)
    return ({"status": "verified" if status == "admitted" else status,
             "wrapper_artifact": artifact, "events": names,
             "required_order": ["warm", "completion", "reset", "start", "measured",
                                "completion", "end"],
             "validation_scope": "outside_start_end"}, diagnostics)


def _component_status(diagnostics: Sequence[Mapping[str, Any]]) -> str:
    if any(row.get("status") == "refused" for row in diagnostics):
        return "refused"
    if any(row.get("status") == "UNKNOWN" for row in diagnostics):
        return "UNKNOWN"
    return "admitted"


def assess_deployment_admissibility(
        profile: Mapping[str, Any], *, expected_profile_sha256: str,
        expected_emission_identity: Mapping[str, str],
        egress_evidence: Mapping[str, Any] | None,
        wrapper_evidence: Mapping[str, Any] | None,
        profile_root: Path | None = None, wrapper_root: Path | None = None) -> dict[str, Any]:
    """Assess one exact deployment without guessing missing target or wrapper facts.

    ``expected_emission_identity`` is an opaque host-owned mapping of artifact roles to SHA-256
    values.  Evidence must reproduce it exactly, which lets each compiler/runtime integration choose
    the necessary artifact roles without teaching this generic checker about a target.
    """
    diagnostics: list[dict[str, Any]] = []
    root = Path.cwd() if profile_root is None else Path(profile_root)
    wrapper_base = root if wrapper_root is None else Path(wrapper_root)
    try:
        validate_or_raise(dict(profile), "deployment_profile")
    except (ValueError, FileNotFoundError) as exc:
        diagnostics.append({
            "priority": 10, "status": "refused", "code": "deployment_profile_malformed",
            "message": str(exc), "action": "provide a complete deployment_profile schema instance",
        })
    if profile.get("schema") != "deployment_profile_v1":
        diagnostics.append({
            "priority": 10, "status": "refused", "code": "deployment_profile_schema_mismatch",
            "message": "deployment profile uses an unsupported schema",
            "action": "provide a deployment_profile_v1 instance",
        })
    profile_sha256 = deployment_profile_sha256(profile)
    if not _is_sha256(expected_profile_sha256) or profile_sha256 != expected_profile_sha256:
        diagnostics.append({
            "priority": 1, "status": "refused", "code": "deployment_profile_identity_mismatch",
            "message": "deployment profile bytes do not match the host-pinned identity",
            "action": "use the pinned profile or issue and review a new profile identity",
        })

    expected_identity = dict(expected_emission_identity)
    if (not expected_identity or any(not isinstance(key, str) or not key
            or not _is_sha256(value) for key, value in expected_identity.items())):
        diagnostics.append({
            "priority": 2, "status": "refused", "code": "expected_emission_identity_malformed",
            "message": "host expected-emission identity is absent or contains a non-SHA value",
            "action": "bind every selected emitted artifact role to an exact SHA-256",
        })

    artifact_checks: dict[str, Any] = {}
    artifacts = _mapping(profile.get("artifacts"))
    for role in _REQUIRED_ARTIFACT_ROLES:
        check, diagnostic = _artifact_check(role, artifacts.get(role), root=root)
        artifact_checks[role] = check
        if diagnostic is not None:
            diagnostics.append(diagnostic)

    egress, egress_diagnostics = _egress_check(
        profile, egress_evidence, profile_sha256=profile_sha256,
        expected_identity=expected_identity)
    wrapper, wrapper_diagnostics = _wrapper_check(
        wrapper_evidence, profile_sha256=profile_sha256,
        expected_identity=expected_identity, root=wrapper_base)
    diagnostics.extend(egress_diagnostics)
    diagnostics.extend(wrapper_diagnostics)
    diagnostics.sort(key=lambda row: (int(row["priority"]), str(row["code"]), str(row["message"])))
    ranked = [{"rank": index, **row} for index, row in enumerate(diagnostics, start=1)]
    status = _component_status(ranked)
    return {
        "schema": "deployment_admissibility_v1",
        "status": status,
        "admitted": status == "admitted",
        "profile": {"sha256": profile_sha256,
                    "expected_sha256": expected_profile_sha256,
                    "artifact_bindings": artifact_checks},
        "emission_identity": expected_identity,
        "physical_egress": egress,
        "wrapper_event_order": wrapper,
        "ranked_actionable_diagnostics": ranked,
        "promotion_scope": ("deployment-specific final readiness; compiler authoring remains allowed "
                            "while evidence is refused or UNKNOWN"),
    }


def require_deployment_admissible(result: Mapping[str, Any], *, expected_profile_sha256: str) -> None:
    """Fail final promotion when an explicitly requested deployment remains unproved."""
    profile = _mapping(result.get("profile"))
    if (result.get("schema") != "deployment_admissibility_v1"
            or profile.get("expected_sha256") != expected_profile_sha256
            or profile.get("sha256") != expected_profile_sha256):
        raise DeploymentAdmissibilityError("deployment admissibility result has a stale profile binding")
    if result.get("status") != "admitted" or result.get("admitted") is not True:
        diagnostics = _rows(result.get("ranked_actionable_diagnostics"))
        codes = [str(_mapping(row).get("code")) for row in diagnostics[:5]]
        raise DeploymentAdmissibilityError(
            "deployment is not admissible: " + ", ".join(codes or [str(result.get("status"))]))


def _load_json(path: Path) -> Mapping[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"{path} does not contain a JSON object")
    return document


def main(argv: list[str] | None = None) -> int:
    """Standalone host-side entrypoint used before a deployment-specific final seal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--emission-identity", type=Path, required=True)
    parser.add_argument("--egress-evidence", type=Path, required=True)
    parser.add_argument("--wrapper-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    profile = _load_json(args.profile)
    result = assess_deployment_admissibility(
        profile, expected_profile_sha256=args.profile_sha256,
        expected_emission_identity=dict(_load_json(args.emission_identity)),
        egress_evidence=_load_json(args.egress_evidence),
        wrapper_evidence=_load_json(args.wrapper_evidence),
        profile_root=args.profile.parent, wrapper_root=args.wrapper_evidence.parent)
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0 if result["status"] == "admitted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
