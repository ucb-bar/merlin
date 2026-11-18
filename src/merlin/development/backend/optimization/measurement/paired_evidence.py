"""Paired GSIM measurement evidence shared by producers, adoption and reporting.

This module verifies existing receipts; it does not create a seal or discover runs.
Callers own candidate admission and pass explicit expected identities.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.digest import is_sha256 as _is_sha256
from merlin.development.backend.optimization.campaign import compiler_library_state as LIBRARY
from merlin.development.backend.optimization.campaign import functional_boundary as PC
from merlin.development.backend.optimization.campaign import pass_preparation as PREPARATION
from merlin.development.targetgen.execution.host.public_program_timing import (
    SCHEMA,
    is_public_program,
    verify_selection,
)

# ---- measurement_evidence ----------------------------------------------------------------------

ARMS = ("baseline", "candidate")

REPLICATES = ("r000", "r001")

SIMULATORS = ("spike", "gsim")

PHASES = ("tuning", "held_out")

TARGET_SELECTED_ANALYZERS = frozenset({"perf_pk_claim.analyze_pk_claim/v4", "perf_pr_claim.analyze_pr_claim/v2"})


class MeasurementEvidenceError(ValueError):
    """Paired measurement evidence differs from the caller's declared inputs."""


def require_supported_oracle_selection(capsules: Sequence[object]) -> None:
    """Refuse a new frozen oracle selection this GSIM paired executor cannot honor.

    Historical PK v3 and PR v1 declarations retain their recorded interpretation.
    A new target-selected declaration cannot be silently measured with the old
    Spike/GSIM producer or certified by its GSIM-only equivalence gate.
    """
    for member in capsules:
        descriptor = getattr(member, "descriptor", None)
        performance = descriptor.get("performance") if isinstance(descriptor, Mapping) else None
        acceptance = performance.get("acceptance") if isinstance(performance, Mapping) else None
        if not isinstance(acceptance, Mapping):
            continue
        evidence = acceptance.get("evidence")
        selected = acceptance.get("analyzer") in TARGET_SELECTED_ANALYZERS or (
            isinstance(evidence, Mapping) and "resolved_from" in evidence
        )
        if not selected:
            continue
        fit = acceptance.get("fit")
        label = f"{getattr(member, 'family', '?')}/{getattr(member, 'capsule', '?')}"
        if not isinstance(evidence, Mapping) or not isinstance(fit, Mapping):
            raise MeasurementEvidenceError(f"{label}: target-selected oracle contract is incomplete")
        correctness, timing = evidence.get("correctness_simulator"), evidence.get("timing_simulator")
        if (
            not isinstance(correctness, str)
            or not isinstance(timing, str)
            or evidence.get("timing_oracle_kind") != f"rtl_{timing}"
            or fit.get("dependent_metric") != f"{timing}_L3_cycles"
            or evidence.get("resolved_from") is None
        ):
            raise MeasurementEvidenceError(f"{label}: target-selected oracle contract is malformed")
        if (correctness, timing) != SIMULATORS:
            raise MeasurementEvidenceError(
                f"{label}: frozen oracles select {correctness} at L2 and {timing} at L3; "
                "the installed paired executor and certificate gate support only Spike/GSIM"
            )


@dataclass(frozen=True, order=True)
class ResultIdentity:
    phase: str
    arm: str
    family: str
    capsule: str
    simulator: str
    replicate: str

    @property
    def label(self) -> str:
        return "/".join((self.phase, self.arm, self.family, self.capsule, self.simulator, self.replicate))


def _simulator_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and value[0].isalpha()
        and all(character.isalnum() or character == "_" for character in value)
    )


def completion_report(
    results: Sequence[Mapping[str, Any]],
    expected: Sequence[ResultIdentity],
    *,
    correctness_simulator: str = SIMULATORS[0],
    timing_simulator: str = SIMULATORS[1],
) -> dict[str, Any]:
    if (
        not _simulator_name(correctness_simulator)
        or not _simulator_name(timing_simulator)
        or correctness_simulator == timing_simulator
        or timing_simulator in {"spike", "unknown", "elaborated_rtl"}
    ):
        raise PC.CampaignGateError("paired oracle selection must name two distinct concrete simulators")
    simulators = (correctness_simulator, timing_simulator)
    wanted = tuple(expected)
    if not wanted or len(set(wanted)) != len(wanted):
        raise PC.CampaignGateError("expected identities are empty or duplicated")
    observed: dict[ResultIdentity, Mapping[str, Any]] = {}
    for row in results:
        identity = ResultIdentity(
            *(str(row.get(key) or "") for key in ("phase", "arm", "family", "capsule", "simulator", "replicate"))
        )
        if (
            identity.phase not in PHASES
            or identity.arm not in ARMS
            or identity.simulator not in simulators
            or identity.replicate not in REPLICATES
        ):
            raise PC.CampaignGateError(f"invalid result identity: {identity.label}")
        if identity in observed:
            raise PC.CampaignGateError(f"duplicate result: {identity.label}")
        observed[identity] = row
    extras = sorted(set(observed) - set(wanted))
    if extras:
        raise PC.CampaignGateError(f"unexpected results: {[item.label for item in extras]}")
    passed = failed = 0
    for identity in wanted:
        row = observed.get(identity)
        if row is None:
            continue
        if identity.simulator == correctness_simulator:
            valid = row.get("correct") is True and row.get("cycles") is None and not row.get("citable")
        else:
            provenance, cycles = row.get("provenance"), row.get("cycles")
            valid = (
                row.get("correct") is True
                and isinstance(cycles, int)
                and not isinstance(cycles, bool)
                and cycles > 0
                and row.get("citable") is True
                and isinstance(provenance, Mapping)
                and provenance.get("tier") == "L3"
                and provenance.get("simulator") == identity.simulator
                and provenance.get("oracle_kind") == f"rtl_{identity.simulator}"
                and provenance.get("derived_from_rtl") is True
                and provenance.get("cycle_accurate") is True
                and _is_sha256(provenance.get("elf_sha256"))
            )
            valid = (
                valid and isinstance(row.get("qualification"), Mapping) and row["qualification"].get("admitted") is True
            )
        passed += int(valid)
        failed += int(not valid)
    missing = len(wanted) - len(observed)
    return {
        "expected": len(wanted),
        "reported": len(observed),
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "complete": missing == failed == 0 and passed == len(wanted),
    }


@dataclass(frozen=True)
class MeasurementBinding:
    phase: str
    functional_run_id: str
    functional_submission_sha256: str
    candidate_record_sha256: str
    candidate_sha256: str
    corpus_manifest_sha256: str
    corpus_capsules_sha256: str
    certificate_sha256: str
    compiler_library_binding: Mapping[str, Any] | None = None
    compiler_pass_preparation: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class VerifiedPairedMeasurement:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]

    def receipt(self) -> dict[str, str]:
        return {"path": str(self.manifest_path), "sha256": self.manifest_sha256}


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _read_measurement(manifest_path: Path, expected_sha256: str | None = None, *, require_current_schema: bool = False):
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise MeasurementEvidenceError("paired measurement manifest is absent or linked")
    payload = manifest_path.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and (not _is_sha256(expected_sha256) or observed != expected_sha256):
        raise MeasurementEvidenceError("paired measurement manifest digest changed")
    manifest = json.loads(payload)
    if not isinstance(manifest, Mapping):
        raise MeasurementEvidenceError("paired measurement manifest must be a mapping")
    if require_current_schema and manifest.get("schema") not in {
        "paired_arm4_performance_campaign_v2",
        "paired_arm4_prepared_performance_campaign_v1",
    }:
        raise MeasurementEvidenceError(
            "unsupported paired campaign schema (expected paired_arm4_performance_campaign_v2)"
        )
    if manifest.get("status") != "GO":
        raise MeasurementEvidenceError(f"paired measurement did not reach GO: {manifest_path}")
    results = manifest.get("raw_results") or {}
    path = Path(str(results.get("paired_cells") or ""))
    if path.is_symlink() or not path.is_file():
        raise MeasurementEvidenceError("paired result cells are absent or changed")
    result_payload = path.read_bytes()
    if hashlib.sha256(result_payload).hexdigest() != results.get("paired_cells_sha256"):
        raise MeasurementEvidenceError("paired result cells are absent or changed")
    result = json.loads(result_payload)
    if not isinstance(result, Mapping):
        raise MeasurementEvidenceError("paired result document must be a mapping")
    if require_current_schema and result.get("schema") != "paired_arm4_result_cells_v2":
        raise MeasurementEvidenceError("unsupported paired result schema (expected paired_arm4_result_cells_v2)")
    cells = result.get("cells") or []
    if not isinstance(cells, list) or any(not isinstance(row, Mapping) for row in cells):
        raise MeasurementEvidenceError("paired result cells must be mappings")
    return manifest, cells, observed


def statistics_rows(
    results: Sequence[Mapping[str, Any]],
    *,
    trial: str,
    timing_simulator: str = SIMULATORS[1],
    identity_evidence: bool = False,
) -> list[dict[str, Any]]:
    """Project every selected timing cell; this projection alone does not admit evidence."""
    if not _simulator_name(timing_simulator) or timing_simulator in {"spike", "unknown", "elaborated_rtl"}:
        raise MeasurementEvidenceError("statistics projection lacks a concrete timing simulator")
    rows = []
    for cell in results:
        if cell.get("simulator") != timing_simulator:
            continue
        provenance = cell.get("provenance") or {}
        row = {
            "identity": {
                "trial": trial,
                "subject": cell.get("arm"),
                "family": f"{cell.get('phase')}:{cell.get('family')}",
                "capsule": cell.get("capsule"),
                "simulator": timing_simulator,
                "replicate": cell.get("replicate"),
            },
            "tier": provenance.get("tier"),
            "correct": cell.get("correct"),
            "cycle_accurate": provenance.get("cycle_accurate"),
            "cycles": cell.get("cycles"),
            "oracle": {
                "kind": provenance.get("oracle_kind"),
                "derived_from_rtl": provenance.get("derived_from_rtl"),
            },
        }
        if identity_evidence:
            row["oracle"].update(
                {
                    "simulator_binary_sha256": provenance.get("simulator_binary_sha256"),
                    "elaborated_firrtl_sha256": provenance.get("elaborated_firrtl_sha256"),
                    "program_elf_sha256": provenance.get("elf_sha256"),
                }
            )
            row["qualification"] = cell.get("qualification")
        rows.append(row)
    return rows


def read_statistics_rows(
    manifest_path: Path,
    *,
    trial: str,
    manifest_sha256: str | None = None,
    timing_simulator: str = SIMULATORS[1],
    identity_evidence: bool = False,
) -> list[dict[str, Any]]:
    """Rehash recorded result bytes before projection; callers separately admit the manifest."""
    manifest, cells, _digest = _read_measurement(manifest_path, manifest_sha256)
    if timing_simulator != SIMULATORS[1] or identity_evidence:
        plan = manifest.get("measurement_plan") or {}
        if not isinstance(plan, Mapping) or plan.get("primary_simulator") != timing_simulator:
            raise MeasurementEvidenceError("statistics timing simulator differs from the pinned measurement plan")
    return statistics_rows(cells, trial=trial, timing_simulator=timing_simulator, identity_evidence=identity_evidence)


def verify_paired_measurement(
    manifest_path: Path, *, expected: MeasurementBinding, manifest_sha256: str | None = None
) -> VerifiedPairedMeasurement:
    """Recompute actual paired evidence against explicit trusted candidate/corpus bindings.

    Adoption can omit the manifest hash before checkpointing a fresh child; replay and
    reporting must supply the hash already bound by their parent receipt.
    """
    if (
        expected.phase not in PHASES
        or not isinstance(expected.functional_run_id, str)
        or not expected.functional_run_id
        or any(
            not _is_sha256(value)
            for value in (
                expected.functional_submission_sha256,
                expected.candidate_record_sha256,
                expected.candidate_sha256,
                expected.corpus_manifest_sha256,
                expected.corpus_capsules_sha256,
                expected.certificate_sha256,
            )
        )
    ):
        raise MeasurementEvidenceError("paired measurement expected binding is incomplete")
    manifest, raw_cells, observed_sha = _read_measurement(manifest_path, manifest_sha256, require_current_schema=True)
    certificate = manifest.get("gsim_certificate") or {}
    corpus = manifest.get("frozen_corpus") or {}
    completion = manifest.get("completion") or {}
    engine_policy = manifest.get("engine_policy") or {}
    plan = manifest.get("measurement_plan") or {}

    try:
        validate_expected_treatments(manifest, expected, raw_cells)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise MeasurementEvidenceError(str(error)) from error
    if manifest.get("measurement_plan_sha256") != hashlib.sha256(_canonical_bytes(plan)).hexdigest():
        raise MeasurementEvidenceError("paired measurement plan digest is invalid")
    try:
        expected_results = tuple(ResultIdentity(**row) for row in plan.get("expected_results") or [])
        recomputed_completion = completion_report(raw_cells, expected_results)
    except Exception as exc:
        raise MeasurementEvidenceError(f"paired raw evidence cannot be revalidated: {exc}") from exc
    if recomputed_completion != completion:
        raise MeasurementEvidenceError("paired completion does not match raw evidence")
    identities = {
        "phase": expected.phase,
        "functional_run_id": expected.functional_run_id,
        "functional_submission_sha256": expected.functional_submission_sha256,
        "candidate_record_sha256": expected.candidate_record_sha256,
        "candidate_sha256": expected.candidate_sha256,
    }
    if any(manifest.get(key) != value for key, value in identities.items()):
        raise MeasurementEvidenceError("paired measurement manifest identity differs from its trial declaration")
    if (
        certificate.get("sha256") != expected.certificate_sha256
        or corpus.get("manifest_sha256") != expected.corpus_manifest_sha256
        or corpus.get("capsules_sha256") != expected.corpus_capsules_sha256
        or corpus.get("visibility") != expected.phase
    ):
        raise MeasurementEvidenceError("paired measurement certificate/corpus identity differs from declaration")
    expected_cells = completion.get("expected")
    if (
        isinstance(expected_cells, bool)
        or not isinstance(expected_cells, int)
        or expected_cells <= 0
        or completion.get("reported") != expected_cells
        or completion.get("passed") != expected_cells
        or completion.get("failed") != 0
        or completion.get("missing") != 0
        or completion.get("complete") is not True
        or engine_policy.get("rtl_execution_backends") != ["gsim"]
        or engine_policy.get("timing_authority") != "gsim"
        or engine_policy.get("verilator") != "prelaunch_certificate_qualification_only"
        or manifest.get("identity_before") != manifest.get("identity_after")
        or (manifest.get("fork_before") or {}).get("ok") is not True
        or (manifest.get("fork_after") or {}).get("ok") is not True
    ):
        raise MeasurementEvidenceError("paired measurement lacks complete GSIM-only identity/fork evidence")
    return VerifiedPairedMeasurement(manifest_path.resolve(), observed_sha, manifest, tuple(raw_cells))


# ---- measurement_library_binding ---------------------------------------------------------------
# Explicit compared-treatment bindings for paired evidence consumption.
#
# Pure receipt checks issue no execution or timing authority. Callers still obtain
# candidate handoffs from the original verifier with independently selected owners.


def binding_arguments(handoff):
    """Reopen the original handoff owners before projecting expected data."""
    library = LIBRARY.handoff_selection(handoff)
    passes = PREPARATION.handoff_selection(handoff)
    return dict(
        compiler_library_binding=copy.deepcopy(handoff.compiler_library_binding) if library is not None else None,
        compiler_pass_preparation=copy.deepcopy(handoff.compiler_pass_preparation) if passes is not None else None,
    )


def validate_expected_treatments(manifest, expected, cells):
    """Require opt-in exact releases and actual prepared-arm identities."""
    library, prepared = expected.compiler_library_binding, expected.compiler_pass_preparation
    if library is not None:
        library = LIBRARY.validate_binding(library)
    if prepared is not None:
        if library is None:
            raise ValueError("prepared paired evidence lacks its independently selected library")
        prepared = PREPARATION.validate_pair(prepared, functional_sha256=expected.functional_submission_sha256)
        if prepared["selection"]["binding"]["library"] != library["snapshot"]["contract_sha256"]:
            raise ValueError("prepared paired evidence selected different library releases")
    plan = manifest.get("measurement_plan") or {}
    schema = (
        "paired_arm4_measurement_plan_v5"
        if prepared
        else ("paired_arm4_measurement_plan_v4" if library else "paired_arm4_measurement_plan_v3")
    )
    campaign = "paired_arm4_prepared_performance_campaign_v1" if prepared else "paired_arm4_performance_campaign_v2"
    if plan.get("schema") != schema or manifest.get("schema") != campaign:
        raise ValueError("unsupported paired measurement plan/campaign schema for the explicitly selected treatments")
    for name, value in (("additional_compiler_library", library), ("compiler_pass_preparation", prepared)):
        if (name in plan) != (value is not None) or plan.get(name) != value:
            raise ValueError("paired measurement changed its explicitly selected " + name)
    if library is None:
        if any(name in manifest for name in ("compiler_passes", "compiler_preparation", "comparison_baseline_sha256")):
            raise ValueError("legacy paired evidence cannot inherit prepared compiler treatments")
        return
    identities = manifest.get("identity_before") or {}
    if identities.get("compiler_library_binding_sha256") != library["sha256"]:
        raise ValueError("paired evidence does not retain its exact selected library identity")
    baseline = expected.functional_submission_sha256
    if prepared is not None:
        baseline = prepared["baseline"]["sha256"]
        if (
            manifest.get("comparison_baseline_sha256") != baseline
            or manifest.get("compiler_passes") != prepared["selection"]
            or manifest.get("compiler_preparation") != prepared["preparation"]
            or identities.get("compiler_pass_preparation_sha256") != prepared["sha256"]
        ):
            raise ValueError("paired evidence lost its original preparation or actual comparison baseline")
    elif any(name in manifest for name in ("compiler_passes", "compiler_preparation", "comparison_baseline_sha256")):
        raise ValueError("unprepared paired evidence cannot inherit a prepared comparison")
    if identities.get("baseline_sha256") != baseline or identities.get("candidate_sha256") != expected.candidate_sha256:
        raise ValueError("paired evidence changed its independently selected compared compiler bytes")
    schedule = plan.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("selected-library paired evidence lacks its actual arm schedule")
    digests = {"baseline": baseline, "candidate": expected.candidate_sha256}
    for row in schedule:
        if (
            not isinstance(row, dict)
            or row.get("arm") not in digests
            or row.get("package_sha256") != digests[row["arm"]]
        ):
            raise ValueError("paired schedule differs from its actual independently selected compiler arm")
    _validate_raw_treatments(schedule, cells, library, prepared)


def _validate_raw_treatments(schedule, cells, library, prepared):
    """Reopen same-execution source/library bindings, not copied cycle labels."""
    fields = ("phase", "arm", "family", "capsule", "replicate")
    selected = {tuple(row.get(key) for key in fields): row for row in schedule}
    if len(selected) != len(schedule):
        raise ValueError("paired schedule repeats a selected compiler execution")
    reopened = {}
    consumed = set()
    for cell in cells:
        key = tuple(cell.get(name) for name in fields)
        path = Path(cell.get("raw_result_path") or "")
        digest = cell.get("raw_result_sha256")
        if (
            key not in selected
            or not path.is_absolute()
            or path.resolve() != path
            or not path.is_file()
            or path.stat().st_size > 32 << 20
        ):
            raise ValueError("selected-library paired evidence lacks a bounded original execution product")
        if (str(path), digest) not in reopened:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("selected-library paired raw execution bytes changed")
            reopened[str(path), digest] = json.loads(raw)
        raw = reopened[str(path), digest]
        if (
            raw.get("execution") != selected[key]
            or raw.get("compiler_library_binding_sha256") != library["sha256"]
            or ("compiler_pass_preparation_sha256" in raw) != (prepared is not None)
            or raw.get("compiler_pass_preparation_sha256") != (prepared["sha256"] if prepared else None)
        ):
            raise ValueError("paired raw execution changed its actual treatment or selected library")
        consumed.add(key)
    if consumed != set(selected):
        raise ValueError("paired raw evidence omits selected compiler executions")


# ---- public_program_measurement ----------------------------------------------------------------
# Original native public-program receipts projected into normal paired rows.
#
# This does not execute a second evaluator or issue an engine certificate. The
# original model grade must already pass. Actual native source/ELF/caller/console
# and engine receipt bytes are reopened before the ordinary certificate gate.


def selection(capsule, rtl_identity, *, simulator="gsim"):
    """The candidate-whole-program selection, or ``None`` when the capsule is not that route's.

    The candidate emits the whole program on this route, host code included, so only a capsule whose
    frozen Phase 0 placement keeps no compute on the host may take it. A capsule whose placement
    does returns ``None`` and takes the owned-host split, where Merlin supplies the host part and the
    candidate cannot be scored on it (:mod:`.metric_scope`). A capsule with no frozen placement, or
    none on the accelerator, has no scorable scope and is refused.
    """
    if not is_public_program(capsule):
        return None
    from merlin.development.backend.optimization.author_interface.prompt_inputs import declared_scope

    if not declared_scope(capsule)["candidate_whole_program_admissible"]:
        return None
    pin = dict(rtl_identity.get("rtl_facts") or {})
    path = Path(str(pin.get("path") or ""))
    if not path.is_file() or path.stat().st_size > 16 << 20:
        raise ValueError("public program requires the selected ordinary RTL facts")
    document = json.loads(path.read_bytes())
    body = document.get("facts") or {}
    config = (body.get("source") or {}).get("config")
    if "simulation_correspondence" in body:
        from merlin.target.rtl.extraction.unit_correspondence import execution_binding

        simulation = (body["simulation_correspondence"].get("configs") or {}).get("simulation") or {}
        config = execution_binding(document, config=simulation.get("config"))["config"]
    value = {
        "schema": SCHEMA,
        "simulator": simulator,
        "rtl_facts": pin,
        "board_config": config,
        "complete_call_timing": simulator == "gsim",
    }
    verify_selection(capsule, value)
    return value


def _reopen(pin):
    if not isinstance(pin, Mapping) or not {"path", "sha256"} <= set(pin):
        raise ValueError("actual public native product has no original byte pin")
    path = Path(pin["path"])
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("actual public native product is unavailable or indirect")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != pin["sha256"] or ("size_bytes" in pin and path.stat().st_size != pin["size_bytes"]):
        raise ValueError("actual public native product changed")
    return path


def project(capsule, grade, *, target, readback_policy=None):
    """Return actual timing evidence only after the original complete grade."""
    if not is_public_program(capsule) or grade.get("status") != "pass":
        raise ValueError("public native timing requires the original complete passing public grade")
    verify_selection(capsule, grade.get("public_program_execution"))
    if (grade.get("model_execution_check") or {}).get("kind") != "candidate_whole_program_execution":
        raise ValueError("public timing cannot borrow a runner-owned host-graph verdict")
    native = grade.get("candidate_native_execution") or {}
    timing = native.get("complete_call_timing") or {}
    from merlin.runtime.backends import registry as backends
    from merlin.runtime.harness.call_timing import COMPLETE_CALL_SCHEMA as TIMER_SCHEMA

    backend = backends.get_backend(target)
    resolver = getattr(backend, "complete_call_timing", None)
    plan = resolver() if callable(resolver) else None
    if plan is None or timing.get("schema") != TIMER_SCHEMA:
        raise ValueError("public timing has no selected original complete-call counter/clock")
    expected = plan.verify()
    if {key: timing.get(key) for key in expected} != expected:
        raise ValueError("public timing selection changed after the original run")
    if timing.get("elf") != native.get("elf"):
        raise ValueError("complete-call timer does not bind the actual candidate ELF")
    products = native.get("build_artifacts")
    required_products = {
        "kernel.llvm.mlir",
        "kernel.ll",
        "kernel.o",
        "kernel.stack_frame.json",
        "kernel.abi.json",
        "harness.c",
    }
    if type(products) is not dict or not required_products <= set(products) or len(products) > 4096:
        raise ValueError("public timing lacks its bounded actual object/caller/build product roster")
    for pin in [
        *products.values(),
        *native.get("source", {}).values(),
        *native.get("host_build_sources", []),
        native.get("elf"),
        native.get("console"),
        timing.get("caller"),
        *grade.get("candidate_emission", {}).values(),
    ]:
        _reopen(pin)
    tier = native.get("tiers", {}).get("L3") or {}
    if tier.get("status") != "pass" or tier.get("engine") != "gsim":
        raise ValueError("public timing did not execute the original selected RTL tier")
    citation = tier.get("engine_citation") or {}
    from merlin.target.simulators import gsim_emulator as GE

    current = GE.citation(target, env_var=getattr(backend, "GSIM_EMU_ENV", None))
    if current != citation or current.get("receipt_status") != "bound":
        raise ValueError("public timing engine bytes or original receipt changed")
    receipt = citation.get("receipt") or {}
    if receipt.get("schema_version") != GE.STRICT_RECEIPT_SCHEMA:
        raise ValueError("public timing has no original bound engine receipt")
    console_path = _reopen(native["console"])
    cb_path = _reopen(grade["candidate_emission"]["command_buffer"])
    cb = json.loads(cb_path.read_bytes())
    if console_path.stat().st_size > 64 << 20:
        raise ValueError("complete public output console exceeds its normal bounded projection")
    from merlin.target.contract.harness.readback_policy import read_console, selected

    readback_policy = selected(readback_policy)
    console = read_console(console_path, policy=readback_policy)
    # Coherent memory readout has separately selected complete output objects.
    # Until they have a fixed normal replay producer, this projection refuses it.
    if native.get("readback_memory_l3") is not None:
        raise ValueError("public complete timing projection lacks a coherent-output replay producer")
    observed, metrics = backend.parse_output(console)
    from merlin.development.backend.optimization.measurement.gsim_workload import (
        OUTPUT_ENCODING,
        encode_declared_outputs,
    )
    from merlin.runtime.engine.command_buffer import declared_output_dtypes

    observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
    from merlin.development.targetgen.grading.golden import compare
    from merlin.development.targetgen.grading.golden_store import load_golden

    declaration_path = _reopen(native["source"]["capsule_declaration"])
    import yaml

    declaration = yaml.safe_load(declaration_path.read_text())
    from merlin.development.backend.optimization.author_interface.prompt_inputs import (
        ACCELERATOR_ONLY,
        bound_scope,
        scored_cycles,
    )

    # The frozen placement of the exact pinned declaration, bound to its program bytes: the complete
    # call is a candidate score only when that placement keeps no compute on the host.
    scope = bound_scope(declaration, declaration_path.parent)
    if scope["kind"] != ACCELERATOR_ONLY:
        raise ValueError("public whole-program timing would score Merlin-supplied host compute")
    golden = load_golden(declaration_path.parent)
    from merlin.development.targetgen.execution.native.candidate_model import checked_source_output_values

    source_observed = checked_source_output_values(cb, native, golden, observed)
    if set(source_observed) != set(golden["outputs"]) or set(observed) != set(cb["kernel_abi"]["outputs"]):
        raise ValueError("public native output names differ from the original complete result roster")
    comparison = compare(
        golden["outputs"],
        source_observed,
        declaration["numeric_policy"],
        golden_source=str(golden.get("golden_source") or "independent_capsule"),
    )
    if comparison.get("status") != "pass":
        raise ValueError("actual complete public outputs no longer pass the original numerical contract")
    output_sha, output_tensors = encode_declared_outputs(observed, cb)
    if native.get("output_observation") != {
        "sha256": output_sha,
        "encoding": OUTPUT_ENCODING,
        "tensors": output_tensors,
    }:
        raise ValueError("public native complete output bytes or roster changed")
    cycles = metrics.get("cycles")
    if (
        type(cycles) is not int
        or not 0 < cycles <= plan.max_delta_cycles
        or cycles != tier.get("metrics", {}).get("cycles")
    ):
        raise ValueError("public native console lacks its actual complete-call cycle delta")
    return {
        "engine": "gsim",
        "status": "pass",
        "elf": str(native["elf"]["path"]),
        "elf_sha256": native["elf"]["sha256"],
        "cycles": cycles,
        "output_sha256": output_sha,
        "output_encoding": OUTPUT_ENCODING,
        "output_tensors": output_tensors,
        "binary_sha256": citation["binary_sha256"],
        "firrtl_sha256": receipt["firrtl_sha256"],
        "model_sha256": receipt["model_manifest_sha256"],
        "observed_engine_binaries": {"status": "observed", "digests": [citation["binary_sha256"]]},
        "derived_from_rtl": True,
        "cycle_accurate": True,
        "reused_measurement": False,
        "oracle": {"kind": "candidate_whole_program_same_elf", "engine_citation": citation},
        "complete_call_timing": timing,
        "actual_console": native["console"],
        "execution_invocations": native.get("execution_invocations"),
        "actual_source": native["source"],
        "actual_candidate": grade["candidate_emission"],
        "metric_scope": scope,
        "scored": scored_cycles(scope, cycles),
    }
