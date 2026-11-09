"""Explicit live service-effect selection for ordinary descriptive feedback.

The original target owner supplies effects; this module only checks custody and
delegates arithmetic. Emitted schedule work never becomes compulsory workload
work, measured latency, statistical qualification or promotion authority.
"""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import asdict
from pathlib import Path
from types import MethodType

from merlin_timing import observations as O

from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding, ReviewedPassCapability
from merlin.perf import capsule_roofline as R
from merlin.runtime import _native_files as F

from .contracts import StageGateError, document_sha256

MECHANISM = "emitted_schedule_service_effects"
SCHEMA = "merlin.original_emitted_service_selection.v1"
_EFFECT_FIELDS = {
    "schema", "source_sha256", "elf_sha256", "scope_sha256", "device_sha256", "scope",
    "resources", "events", "completion_chains", "unresolved", "provenance", "interpretation",
    "attainment", "measurement_comparability", "convergence_eligible",
}


def _callback(owner, name):
    callback = getattr(owner, name, None)
    if (type(callback) is not MethodType or callback.__self__ is not owner
            or callback.__func__ is not getattr(type(owner), name, None)):
        raise StageGateError("service effects require the original bound owner callback")
    source = Path(inspect.getsourcefile(callback.__func__) or "")
    if not source.is_absolute():
        raise StageGateError("service effect callback has no original source")
    return callback.__func__, callback.__func__.__code__, F.pin(source, max_bytes=4 << 20)


def _source_digest(value):
    return type(value) is str and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


class SelectedServiceEffects:
    """Original live host choice; records and callback names cannot issue it."""

    def __init__(self, capability, binding):
        if type(capability) is not ReviewedPassCapability or type(binding) is not PassCapabilityBinding:
            raise StageGateError("service effects need an original live capability and exact binding")
        capability.revalidate()
        if capability.mechanism != MECHANISM or capability.binding != binding:
            raise StageGateError("service effects selected a foreign mechanism or binding")
        self._capability, self._binding = capability, binding
        self._owner = capability.owner
        self._callback = _callback(self._owner, "evidence")
        self._binding_callback = _callback(self._owner, "selected_binding")
        self._record = {
            "schema": SCHEMA,
            "capability": capability.record(),
            "callback": self._callback[2],
            "binding_callback": self._binding_callback[2],
            "scope": "emitted_schedule",
            "authority": "descriptive necessary service only; no qualification or objective floor",
        }
        self._identity = document_sha256(self._record)
        self.revalidate()

    @property
    def identity(self):
        return SCHEMA + ":" + self._identity

    def revalidate(self):
        self._capability.revalidate()
        if (
            self._capability.mechanism != MECHANISM
            or self._capability.binding != self._binding
            or self._capability.owner is not self._owner
            or self._capability.record() != self._record["capability"]
            or _callback(self._owner, "evidence") != self._callback
            or _callback(self._owner, "selected_binding") != self._binding_callback
            or self._owner.selected_binding() != asdict(self._binding)
            or document_sha256(self._record) != self._identity
        ):
            raise StageGateError("original service effect owner or callback changed")

    def record(self):
        self.revalidate()
        return copy.deepcopy(self._record)

    def project(self, *, member_source_sha256, compiled, report_path, invocation_path, counter_review_path):
        """Consume current original products, leaving all performance gates alone."""
        self.revalidate()
        if not _source_digest(member_source_sha256):
            raise StageGateError("service effects require the original frozen member source")
        diagnostics = compiled.get("candidate_diagnostics") or {}
        artifacts = diagnostics.get("artifacts") or {}
        join = diagnostics.get("source_emission_join") or {}
        if (compiled.get("status") != "observed" or join.get("status") != "verified"
                or not _source_digest(join.get("source_sha256"))
                or join["source_sha256"] != (artifacts.get("source") or {}).get("sha256")):
            return unknown("current compiled product has no original complete source/emission join")
        report_path = Path(report_path)
        pin = F.pin(report_path, max_bytes=16 << 20)
        report = json.loads(report_path.read_bytes())
        if (F.pin(report_path, max_bytes=16 << 20) != pin or not O.feature_complete(report)
                or report.get("schema") != O.REPORT_V2
                or report.get("elf_sha256") != compiled.get("elf_sha256")
                or report.get("input_sha256") != document_sha256(
                    {"kind": "original_frozen_public_capsule", "sha256": member_source_sha256})
                or report.get("model_sha256") != self._binding.runtime):
            raise StageGateError("service effects changed their current original count report or ELF")
        try:
            evidence = self._owner.evidence(
                report_path, Path(invocation_path), Path(counter_review_path),
                source_sha256=member_source_sha256,
            )
        except (ValueError, OSError) as error:
            self.revalidate()
            return unknown("original service owner refused this scope: " + str(error)[:240])
        self.revalidate()
        events = evidence.get("events") if isinstance(evidence, dict) else None
        if (not isinstance(evidence, dict)
                or not set(evidence) <= _EFFECT_FIELDS
                or evidence.get("schema") != "merlin.phase2.service_effect_evidence.v1"
                or evidence.get("scope") != "emitted_schedule"
                or evidence.get("source_sha256") != member_source_sha256
                or evidence.get("elf_sha256") != compiled["elf_sha256"]
                or any(evidence.get(key) != report.get(key) for key in ("scope_sha256", "device_sha256"))
                or evidence.get("attainment") is not None
                or evidence.get("convergence_eligible") is not False
                or type(events) is not list
                or any(not isinstance(row, dict) or row.get("compulsory") is not False for row in events)):
            raise StageGateError("service owner returned foreign or compulsory-workload effects")
        try:
            floor = R.compose_service_floor(
                evidence.get("resources"), evidence.get("events"),
                completion_chains=evidence.get("completion_chains", ()),
                unresolved={**(evidence.get("unresolved") or {}),
                            "fixed_workload": "emitted effects do not prove a compulsory workload floor"},
                scope="emitted_schedule",
            )
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise StageGateError("invalid original service inventory: " + str(error)) from error
        if F.pin(report_path, max_bytes=16 << 20) != pin:
            raise StageGateError("original count report changed during service projection")
        return {
            "status": "partial_emitted_service",
            "selection_sha256": self._identity,
            "source_binding": {"member_sha256": member_source_sha256,
                               "interface_sha256": join["source_sha256"]},
            "evidence": copy.deepcopy(evidence),
            "floor": floor,
            "measured_cycles": None,
            "attainment": None,
            "objective_floor_eligible": False,
            "convergence_eligible": False,
            "promotion_authority": "NONE",
        }


def unknown(reason):
    return {"status": "UNKNOWN", "reason": reason, "evidence": None, "floor": None,
            "attainment": None, "convergence_eligible": False, "promotion_authority": "NONE"}


def select(capability, binding):
    """Absent means unavailable; copied records never enable a target callback."""
    return None if capability is None else SelectedServiceEffects(capability, binding)
