"""Explicit native predictions for public diagnostics, never measured promotion.

The original public archive and live semantics owners establish the selected
domain. Each prediction still needs the queued arm's complete numerical receipt
and fresh original boxed counts. Defaults and formal search remain unchanged.
"""

from __future__ import annotations

import inspect
import json
import sys
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

from merlin.llvmlower import reviewed_metadata as METADATA
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
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
_JOINT_REGISTRIES = weakref.WeakKeyDictionary()
BRIDGE_MECHANISM = "compiler_source_native_correspondence"
JOINT_METADATA_PROTOCOL = "merlin.public_native_metadata.v1"


def _registry_roster(registry, expected):
    if (
        type(registry) is not ReviewedPassCapabilities
        or type(expected) is not tuple
        or type(registry._capabilities) is not tuple
        or len(registry._capabilities) != len(expected)
        or any(
            type(cap) is not ReviewedPassCapability or actual is not cap
            for actual, cap in zip(registry._capabilities, expected, strict=True)
        )
    ):
        raise StageGateError("native correspondence requires its original exact bridge registry")


def _metadata_roster(issued, selection, bridge):
    registry, roster, protocol = issued[7:10]
    expected = (bridge,)
    if protocol == JOINT_METADATA_PROTOCOL:
        expected = (selection.correspondence, selection.native_semantics, bridge)
    elif protocol is not None:
        raise StageGateError("native metadata protocol changed after original issuance")
    if len(roster) != len(expected) or any(a is not b for a, b in zip(roster, expected, strict=True)):
        raise StageGateError("native metadata lost its original selected capability roster")
    _registry_roster(registry, roster)
    if protocol == JOINT_METADATA_PROTOCOL and _JOINT_REGISTRIES.get(registry) is not roster:
        raise StageGateError("joint native metadata lost its original issuance")
    return roster


def _bridge_verification(bridge, native_binding, registry, *, expected_roster=None):
    if type(bridge) is not ReviewedPassCapability or type(registry) is not ReviewedPassCapabilities:
        raise StageGateError("native correspondence requires the original exact reviewed bridge")
    roster = (bridge,) if expected_roster is None else expected_roster
    if not roster or roster[-1] is not bridge:
        raise StageGateError("native correspondence requires its original exact bridge registry")
    if len(roster) != 1 and (len(roster) != 3 or _JOINT_REGISTRIES.get(registry) is not roster):
        raise StageGateError("joint native metadata requires its original exact issued roster")
    _registry_roster(registry, roster)
    # Only historical-owner metadata getters are coalesced. The original owner
    # runs at both boundaries, and a failed final recheck prevents return.
    with METADATA.scoped_registry_metadata_verification(registry):
        if bridge.mechanism != BRIDGE_MECHANISM or bridge.binding != native_binding:
            raise StageGateError("native correspondence requires the original exact reviewed bridge")
        owner = bridge.owner
        callback = getattr(owner, "verify_compiler_binding", None)
        if type(callback) is not MethodType or callback.__self__ is not owner:
            raise StageGateError("native correspondence requires its original bound verification method")
        source = inspect.getsourcefile(callback.__func__)
        if source is None or Path(source).is_symlink() or not Path(source).is_file():
            raise StageGateError("native correspondence verification source is unavailable")
        path = Path(source).resolve()
        return bridge, bridge.identity, callback.__func__, callback.__func__.__code__, str(path), F.digest(path)


@dataclass(frozen=True, eq=False)
class PublicNativeExploration:
    selection: NativeArchiveScreenSelection
    reviewed_json: bytes
    catalog_json: bytes
    files: tuple
    _issuer: object
    binding_correspondence: ReviewedPassCapability | None = None

    @contextmanager
    def _metadata_operation(self):
        """Bound one trusted selection-record operation, never target work.

        Original owner checks bracket the operation. Native runtime/semantics
        checks and the actual compiler-binding callback still run on every use.
        """
        issued = _ISSUED.get(self)
        if type(self) is not PublicNativeExploration or issued is None or self._issuer is not _ISSUER:
            raise StageGateError("public native metadata requires its original live owner")
        if issued[7] is None:
            yield
            return
        _metadata_roster(issued, self.selection, self.binding_correspondence)
        with METADATA.scoped_registry_metadata_verification(issued[7]):
            self.verify()
            primary = None
            try:
                yield
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    self.verify()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note("Public native final metadata verification failed: " + str(error))

    def verify(self):
        issued = _ISSUED.get(self)
        if (
            type(self) is not PublicNativeExploration
            or self._issuer is not _ISSUER
            or issued is None
            or self.selection is not issued[0]
            or self.binding_correspondence is not (issued[6][0] if issued[6] is not None else None)
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
        if (
            self.binding_correspondence is not None
            and _bridge_verification(
                self.binding_correspondence,
                self.selection.correspondence.binding,
                issued[7],
                expected_roster=_metadata_roster(issued, self.selection, self.binding_correspondence),
            )
            != issued[6]
        ):
            raise StageGateError("public native exploration original binding correspondence changed")
        for path, digest in self.files:
            if F.pin(Path(path), max_bytes=256 << 20)["sha256"] != digest:
                raise StageGateError("public native exploration original evidence changed")

    def record(self):
        self.verify()
        reviewed = json.loads(self.reviewed_json)
        return {
            "schema": SCHEMA if self.binding_correspondence is None else "merlin.public_native_exploration.v2",
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
            **(
                {"compiler_source_native_correspondence": self.binding_correspondence.record()}
                if self.binding_correspondence is not None
                else {}
            ),
        }

    def verify_for(self, policy):
        self.verify()
        if (
            F.digest(policy.feedback_evaluator.rtl_facts_path) != self.selection.expected_rtl_facts_sha256
            or policy._compiler_pass_selection.binding.hardware != self.selection.expected_rtl_facts_sha256
            or self.binding_correspondence is None
            and policy._compiler_pass_selection.binding.software != self.selection.expected_observed_contract_sha256
        ):
            raise StageGateError("public native exploration selected another hardware/software domain")
        if self.binding_correspondence is not None:
            try:
                self.binding_correspondence.owner.verify_compiler_binding(
                    policy._compiler_pass_selection.binding,
                    native_binding=self.selection.correspondence.binding,
                )
            finally:
                self.verify()

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


def prepare_public_native_exploration(selection, *, binding_correspondence=None, metadata_protocol=None):
    """Reconstruct public gates, optionally joining distinct binding namespaces.

    The optional exact reviewed bridge must be issued by original live owners.
    Its bound verifier checks the complete compiler and native bindings; neither
    a record nor equalized software hashes can establish that correspondence.
    Default V1 selection and records retain their original direct comparison.
    The optional original joint-metadata protocol coalesces only original Count,
    native-semantics and bridge getters during one selection-record operation.
    It preserves V2 record semantics and never brackets current target work.
    """
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
    if metadata_protocol is not None and (
        type(metadata_protocol) is not str or metadata_protocol != JOINT_METADATA_PROTOCOL
    ):
        raise StageGateError("native metadata requires the exact original opt-in protocol")
    if metadata_protocol is not None and binding_correspondence is None:
        raise StageGateError("joint native metadata requires an explicit original bridge")
    bridge_registry, metadata_roster = None, ()
    if binding_correspondence is not None:
        if type(binding_correspondence) is not ReviewedPassCapability:
            raise StageGateError("native correspondence requires the original exact reviewed bridge")
        metadata_roster = (binding_correspondence,)
        if metadata_protocol == JOINT_METADATA_PROTOCOL:
            NATIVE.verify_semantics(selection.native_semantics, binding)
            metadata_roster = (selection.correspondence, selection.native_semantics, binding_correspondence)
        bridge_registry = ReviewedPassCapabilities(metadata_roster)
        if metadata_protocol == JOINT_METADATA_PROTOCOL:
            _JOINT_REGISTRIES[bridge_registry] = metadata_roster
    bridge = (
        _bridge_verification(binding_correspondence, binding, bridge_registry, expected_roster=metadata_roster)
        if binding_correspondence is not None
        else None
    )
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
                *(
                    (Path(METADATA.__file__).resolve(), Path(inspect.getsourcefile(ReviewedPassCapabilities)).resolve())
                    if bridge_registry is not None
                    else ()
                ),
            )
        )
    )
    files = tuple((str(path), F.pin(path, max_bytes=256 << 20)["sha256"]) for path in paths)
    result = PublicNativeExploration(
        selection, canonical_json(reviewed), canonical_json(catalog), files, _ISSUER, binding_correspondence
    )
    functions = tuple(
        (owner, name, getattr(owner, name), getattr(owner, name).__code__)
        for owner, names in (
            (PublicNativeExploration, ("verify", "record", "verify_for", "observe_member", "_metadata_operation")),
            (PublicNativeExploration._metadata_operation, ("__wrapped__",)),
            (OBSERVATION, ("observe", "validate_projection")),
            (NATIVE, ("inference_case", "compare", "verify_semantics")),
            (COUNTS, ("observe",)),
            (FEEDBACK, ("join", "validate")),
            (DOMAIN, ("current", "train_support", "typed_profile")),
            (REVIEW, ("review_archive", "domain_reason")),
            (HISTORY, ("prepare",)),
            (sys.modules[__name__], ("_bridge_verification", "_registry_roster", "_metadata_roster")),
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
        bridge,
        bridge_registry,
        metadata_roster,
        metadata_protocol,
    )
    try:
        result.verify()
    except BaseException:
        del _ISSUED[result]
        if bridge_registry is not None:
            _JOINT_REGISTRIES.pop(bridge_registry, None)
        raise
    return result
