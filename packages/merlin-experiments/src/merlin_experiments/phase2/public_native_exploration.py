"""Explicit native predictions for public diagnostics, never measured promotion.

The original public archive and live semantics owners establish the selected
domain. Each prediction still needs the queued arm's complete numerical receipt
and fresh original boxed counts. Defaults and formal search remain unchanged.
"""

from __future__ import annotations

import json
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding, ReviewedPassCapability
from merlin.runtime import _native_files as F

from . import _archive_feedback as HISTORY
from . import _archive_source_domain as DOMAIN
from . import _public_native_feedback as FEEDBACK
from . import _public_native_observation as OBSERVATION
from . import archive_count_observation as COUNTS
from . import archive_native_screen as NATIVE
from . import archive_screen_review as REVIEW
from .archive_screen_selection import NativeArchiveScreenSelection
from .archive_spike_runtime import ArchiveSpikeRuntime
from .contracts import StageGateError, canonical_json, document_sha256

SCHEMA = "merlin.public_native_exploration.v1"
_ISSUER = object()
_ISSUED = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class PublicNativeExploration:
    selection: NativeArchiveScreenSelection
    reviewed_json: bytes
    catalog_json: bytes
    files: tuple
    _issuer: object

    def verify(self):
        issued = _ISSUED.get(self)
        if (
            type(self) is not PublicNativeExploration
            or self._issuer is not _ISSUER
            or issued is None
            or self.selection is not issued[0]
            or (self.reviewed_json, self.catalog_json, self.files) != issued[1:4]
            or any(
                getattr(owner, name) is not function or function.__code__ is not code
                for owner, name, function, code in issued[4]
            )
        ):
            raise StageGateError("public native exploration lost its original issued owners")
        self.selection.runtime.verify()
        self.selection.correspondence.revalidate()
        NATIVE.verify_semantics(self.selection.native_semantics, self.selection.correspondence.binding)
        if (self.selection.correspondence.identity, self.selection.native_semantics.identity) != issued[5]:
            raise StageGateError("public native exploration semantics correspondence changed")
        for path, digest in self.files:
            if F.pin(Path(path), max_bytes=256 << 20)["sha256"] != digest:
                raise StageGateError("public native exploration original evidence changed")

    def record(self):
        self.verify()
        reviewed = json.loads(self.reviewed_json)
        return {
            "schema": SCHEMA,
            "runtime_sha256": self.selection.runtime.sha256,
            "facts_sha256": self.selection.expected_rtl_facts_sha256,
            "software_sha256": self.selection.expected_observed_contract_sha256,
            "reviewed_archive_sha256": self.selection.archive_sha256,
            "qualification_sha256": document_sha256(reviewed),
            "catalog_sha256": document_sha256(json.loads(self.catalog_json)),
            "source_files_sha256": document_sha256(self.files),
            "native_semantics_identity": self.selection.native_semantics.identity,
            "prediction_kind": "event_timing",
            "scope": "qualified-domain public exploration only",
            "measured_cycles": None,
            "promotion_authority": "NONE",
        }

    def verify_for(self, policy):
        self.verify()
        if (
            F.digest(policy.feedback_evaluator.rtl_facts_path) != self.selection.expected_rtl_facts_sha256
            or policy._compiler_pass_selection.binding.hardware != self.selection.expected_rtl_facts_sha256
            or policy._compiler_pass_selection.binding.software != self.selection.expected_observed_contract_sha256
        ):
            raise StageGateError("public native exploration selected another hardware/software domain")

    def observe_member(self, policy, member, workspace, value, *, sandbox, remaining_s):
        self.verify_for(policy)
        result = OBSERVATION.observe(
            self,
            policy,
            member,
            Path(workspace),
            value,
            sandbox=sandbox,
            remaining_s=remaining_s,
        )
        self.verify_for(policy)
        return result


def prepare_public_native_exploration(selection):
    """Reconstruct original public gates; a saved report cannot issue this owner."""
    if (
        type(selection) is not NativeArchiveScreenSelection
        or selection.screen_path is not None
        or type(selection.runtime) is not ArchiveSpikeRuntime
    ):
        raise StageGateError("public native exploration requires its explicit original native selection")
    selection.runtime.verify()
    model = json.loads(selection.runtime.model_json)
    binding = PassCapabilityBinding(
        hardware=selection.expected_rtl_facts_sha256,
        software=selection.expected_observed_contract_sha256,
        numerical_policy=model["domain"]["feature_schema_sha256"],
        runtime=F.digest(selection.runtime.model_path),
        library=document_sha256(model["libraries"]),
    )
    if (
        type(selection.correspondence) is not ReviewedPassCapability
        or selection.correspondence.mechanism != "timing_model_correspondence"
        or selection.correspondence.binding != binding
    ):
        raise StageGateError("public native exploration requires exact live model correspondence")
    reviewed = selection.preflight()
    catalog = HISTORY.prepare(
        archive=selection.archive,
        archive_sha256=selection.archive_sha256,
        binding_path=selection.binding_path,
        dataset=reviewed["dataset"],
    )
    modules = (HISTORY, DOMAIN, *DOMAIN.source_modules(), COUNTS, NATIVE, REVIEW, OBSERVATION, FEEDBACK)
    paths = tuple(
        dict.fromkeys(
            (
                Path(selection.archive),
                Path(selection.binding_path),
                Path(selection.observations_path),
                *(Path(module.__file__).resolve() for module in modules),
                Path(HISTORY.source_observation.__code__.co_filename).resolve(),
                Path(__file__).resolve(),
            )
        )
    )
    files = tuple((str(path), F.pin(path, max_bytes=256 << 20)["sha256"]) for path in paths)
    result = PublicNativeExploration(selection, canonical_json(reviewed), canonical_json(catalog), files, _ISSUER)
    functions = tuple(
        (owner, name, getattr(owner, name), getattr(owner, name).__code__)
        for owner, names in (
            (PublicNativeExploration, ("verify", "record", "verify_for", "observe_member")),
            (OBSERVATION, ("observe", "validate_projection")),
            (NATIVE, ("inference_case", "compare", "verify_semantics")),
            (COUNTS, ("observe",)),
            (FEEDBACK, ("join", "validate")),
            (DOMAIN, ("current", "train_support", "typed_profile")),
            (REVIEW, ("review_archive", "domain_reason")),
            (HISTORY, ("prepare",)),
        )
        for name in names
    )
    _ISSUED[result] = (
        selection,
        result.reviewed_json,
        result.catalog_json,
        result.files,
        functions,
        (selection.correspondence.identity, selection.native_semantics.identity),
    )
    try:
        result.verify()
    except BaseException:
        del _ISSUED[result]
        raise
    return result
