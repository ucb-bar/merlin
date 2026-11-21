"""Sealed original-host compiler capability transport into a budgeted grader.

Only the trusted grader receives the readonly sealed FD. Candidate commands do
not inherit it. Original archives, source identity and actual kernel controls
are reopened before installing the ordinary per-call package/proof executor.
This transports existing authority; it supplies no numerical or timing verdict.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import fcntl
import hashlib
import json
import os
import sys
import threading
import weakref
import zlib
from pathlib import Path

import yaml

from merlin.common.digest import sha256_file
from merlin.common.process import cpu_pool
from merlin.common.process import invocation_records as invocation_record
from merlin.common.storage.bundle_snapshots import verified_snapshot_scope
from merlin.development.backend.feedback.execution import profile_executor as _merlin_profile_executor_1
from merlin.development.backend.feedback.execution import profile_runtime as R
from merlin.development.backend.optimization.campaign import external_resources as resources
from merlin.development.backend.optimization.campaign import functional_boundary as C
from merlin.development.backend.optimization.contracts import StageGateError
from merlin.development.backend.optimization.search.retained_revisions import source_inventory
from merlin.development.targetgen.authoring import profile_admission as API
from merlin.development.targetgen.authoring.library_admission import (
    active_compiler_library,
    scoped_compiler_library,
)
from merlin.development.targetgen.authoring.pass_selection import (
    CompilerPassSelection,
    active_compiler_passes,
    scoped_compiler_passes,
)
from merlin.development.targetgen.execution.host.common_transforms import CommonTransformService
from merlin.development.targetgen.packaging import loading as P
from merlin.evaluation.capsule.evidence import source_custody as capsule_identity
from merlin.evaluation.capsule.models import candidates as capsule_model_candidates
from merlin.evaluation.capsule.models import grading as capsule_model
from merlin.evaluation.capsule.models import runtime as capsule_processes
from merlin.target.registry.descriptor_loading import load_target_experiment
from merlin.target.registry.experiment_descriptor import TargetExperiment

# ---- compiler_profile_child --------------------------------------------------------------------

SCHEMA = "merlin.original_model_child.v1"

LIBRARY_SCHEMA = "merlin.original_model_child.v2"

PASSES_SCHEMA = "merlin.original_model_child.v3"

CHECKED_SOURCE_SCHEMA = "merlin.original_model_child.v4"

DELEGATED_PASSES_SCHEMA = "merlin.original_model_child.v5"

CHECKPOINT_SCHEMA = "merlin.original_model_child_case_checkpoint.v1"

_RESTORED_CHILD = contextvars.ContextVar("original_restored_child_checkpoint", default=None)

_RESTORED_ISSUED = weakref.WeakSet()

MAX_BYTES = 256 << 10

MAX_SELECTION_BYTES = 4 << 20

_COMPRESSED_SELECTION = b"merlin.original_model_child.zlib.v1\0"

_SEALS, _ADD_SEALS, _GET_SEALS = 0x000F, 1033, 1034


def _memfd():
    if sys.platform != "linux":
        raise StageGateError("original model child transport requires Linux sealed descriptors")
    if hasattr(os, "memfd_create"):
        return os.memfd_create("merlin-original-model-child", 0x0001 | 0x0002)
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    function = library.memfd_create
    function.argtypes, function.restype = (ctypes.c_char_p, ctypes.c_uint), ctypes.c_int
    fd = function(b"merlin-original-model-child", 0x0001 | 0x0002)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "original model child sealed descriptor creation failed")
    return fd


def _data(value):
    if dataclasses.is_dataclass(value):
        return _data(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if type(value) in (tuple, list):
        return [_data(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _data(item) for key, item in value.items()}
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise StageGateError("model child transport admits original plain typed selections only")


def _digest(value):
    return hashlib.sha256(json.dumps(_data(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _encode_selection(document):
    raw = json.dumps(_data(document), sort_keys=True, allow_nan=False).encode()
    if len(raw) > MAX_SELECTION_BYTES:
        raise StageGateError("original model child selection exceeds its decoded transport bound")
    encoded = raw if len(raw) <= MAX_BYTES else _COMPRESSED_SELECTION + zlib.compress(raw)
    if len(encoded) > MAX_BYTES:
        raise StageGateError("original model child selection exceeds its bounded transport size")
    return encoded


def _decode_selection(raw):
    if len(raw) > MAX_BYTES:
        raise StageGateError("model child capability exceeds its bounded transport size")
    if raw.startswith(_COMPRESSED_SELECTION):
        decoder = zlib.decompressobj()
        try:
            decoded = decoder.decompress(raw[len(_COMPRESSED_SELECTION) :], MAX_SELECTION_BYTES + 1)
        except zlib.error as error:
            raise StageGateError("model child compressed selection is invalid") from error
        if len(decoded) > MAX_SELECTION_BYTES or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise StageGateError("model child compressed selection exceeds its closed transport bound")
        raw = decoded
    return json.loads(raw)


def _plain(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path or any(parent.is_symlink() for parent in (path, *path.parents)):
        raise StageGateError("model child requires canonical original ownership")
    return path


def _pin(value):
    path = _plain(value)
    if not path.is_file():
        raise StageGateError("model child original input is absent")
    return {"path": str(path), "sha256": sha256_file(path)}


def _source(candidate):
    return source_inventory(_plain(candidate), max_bytes=32 << 20, max_files=8192, exclude_generated=True)["sha256"]


def _process(pid):
    stat = (Path("/proc") / str(pid) / "stat").read_text().rpartition(") ")[2].split()
    return {"pid": pid, "start_time": stat[19], "mount_namespace": (Path("/proc") / str(pid) / "ns/mnt").stat().st_ino}


def _original_executor():
    executor, admission = P.active_package_executor(), API.active_compiler_imports()
    if executor is None and admission is None and resources.active_resource_boundary() is None:
        return None
    if (
        type(executor) is not _merlin_profile_executor_1._Executor
        or admission is None
        or type(executor.owner) is not CommonTransformService
    ):
        raise StageGateError("budgeted model cannot drop a foreign or incomplete original execution owner")
    service = executor.owner
    if any(name in vars(service) for name in ("compiler_imports", "_verify", "check", "apply", "_toolchain")):
        raise StageGateError("model child original owner cannot supply instance policy callbacks")
    CommonTransformService._verify(service)
    if (service.workspace, service.bundle_manifest, service.repo) != (
        admission.workspace,
        admission.bundle_manifest,
        admission.repo,
    ):
        raise StageGateError("budgeted model compiler owner differs from original import admission")
    return executor


def model_child_temporary_root():
    """Keep budget products under the original private/capped execution owner."""
    from merlin.development.backend.feedback.execution.snapshot_evaluator import active_snapshot

    executor = active_snapshot() or _original_executor()
    return None if executor is None else _plain(executor.work_root)


def _inputs_record(inputs):
    if type(inputs) is C.PackageSandboxInputs:
        return {"kind": "explicit", "values": _data(inputs)}
    if type(inputs) is C.FrozenPackageSandboxInputs:
        from merlin.development.backend.optimization.campaign.package_closure import restore

        if restore(inputs.root, inputs.record) != inputs:
            raise StageGateError("model child frozen package inputs changed")
        return {"kind": "frozen", "root": str(inputs.root), "record": inputs.record}
    raise StageGateError("model child requires the exact original package sandbox selection")


def _restore_inputs(record):
    from merlin.development.targetgen.sandbox import toolchain as TC
    from merlin.development.targetgen.sandbox.answer_surfaces import AnswerSurface

    if record["kind"] == "frozen":
        from merlin.development.backend.optimization.campaign.package_closure import restore

        return restore(_plain(record["root"]), record["record"])
    if record["kind"] != "explicit":
        raise StageGateError("model child has no original package input type")
    values = record["values"]
    paths = dict(values["paths"])
    paths["repo"] = _plain(paths["repo"])
    if paths["python_import_roots"] is not None:
        paths["python_import_roots"] = tuple(paths["python_import_roots"])
    sim = dict(values["sim"])
    for name in ("bind_paths", "path_dirs", "ld_dirs"):
        sim[name] = tuple(sim[name])
    sim["probes"] = tuple(TC.ToolProbe(**row) for row in sim["probes"])
    surfaces = tuple(
        AnswerSurface(**{**row, "path": Path(row["path"]), "grantable": tuple(row["grantable"])})
        for row in values["surfaces"]
    )
    return C.PackageSandboxInputs(TC.ToolchainPaths(**paths), TC.SimToolchain(**sim), values["harness"], surfaces)


def _closure(*, library=False, passes=False, checked_source=False, parent_delegation=False, source_optimizations=False):
    """Direct invoked host owners; this makes no transitive dependency claim."""
    from merlin.common.paths import module_source_path

    names = (
        __name__,
        R.__name__,
        CommonTransformService.__module__,
        API.__name__,
        P.__name__,
        C.__name__,
        resources.__name__,
        resources.kernel.__name__,
        "merlin.evaluation.capsule.runner",
        "merlin.evaluation.capsule.models.runtime",
        "merlin.evaluation.capsule.evidence.source_custody",
        "merlin.development.backend.feedback.execution.operator_child",
        "merlin.development.backend.feedback.execution.scale_probe_child",
        "merlin.development.backend.feedback.execution.scale_probe_batch",
        "merlin.development.backend.feedback.execution.compile_only",
        "merlin.development.targetgen.oracles.certificates",
        "merlin.development.targetgen.authoring.pass_selection",
        "merlin.development.targetgen.authoring.shared_pass_contract",
        "merlin.compiler.lower.device.target",
        "merlin.compiler.lower.device.target",
        "merlin.development.backend.feedback.certification.visible_model_gate",
        "merlin.compiler.lower.host.completed_host_storage",
        "merlin.compiler.lower.host.host_storage_policy",
        "merlin.compiler.lower.host.borrowed_arena",
        "merlin.development.targetgen.execution.host.source_composition_preferences",
        "merlin.development.targetgen.execution.host.model_execution",
        "merlin.development.targetgen.execution.host.dispatch_checks",
    )
    if library:
        names += (
            "merlin.development.targetgen.authoring.library_admission",
            "merlin.development.backend.optimization.campaign.compiler_library_state",
            "merlin.development.backend.optimization.campaign.compiler_libraries",
        )
    if passes:
        names += (
            "merlin.development.targetgen.authoring.pass_selection",
            "merlin.compiler.lower.source.reviewed",
            "merlin.compiler.lower.source.reviewed",
        )
    if checked_source:
        names += (
            "merlin.development.targetgen.execution.host.checked_source_diagnostics",
            "merlin.development.targetgen.execution.host.checked_source_diagnostics",
            "merlin.development.testgen.sealed_capture.connected_integer_qualification",
        )
    if parent_delegation:
        names += (
            "merlin.development.backend.feedback.execution.compiler_pass_delegation",
            "merlin.development.backend.feedback.execution.compiler_pass_parent_proof",
        )
    rows = [_pin(module_source_path(name)) for name in names]
    from merlin.common.digest import python_members

    members = python_members(module_source_path("merlin.evaluation.capsule").parent, label="capsule evaluation")
    for path in members.values():
        row = _pin(path)
        if row not in rows:
            rows.append(row)
    if source_optimizations:
        from merlin.development.targetgen.authoring.profile_admission import implementation_sources

        for path in implementation_sources():
            row = _pin(path)
            if row not in rows:
                rows.append(row)
    return rows


def _selected_closure(document):
    source_optimizations = "source_optimizations_v1" in (document.get("compiler_passes") or {}).get("mechanisms", [])
    if document["schema"] == DELEGATED_PASSES_SCHEMA:
        return _closure(
            library="compiler_library" in document,
            passes=True,
            source_optimizations=source_optimizations,
            checked_source="checked_source_diagnostic" in document,
            parent_delegation=True,
        )
    if document["schema"] == CHECKED_SOURCE_SCHEMA:
        return _closure(
            library="compiler_library" in document,
            passes=True,
            checked_source=True,
            source_optimizations=source_optimizations,
        )
    return (
        _closure(library="compiler_library" in document, passes=True, source_optimizations=source_optimizations)
        if document["schema"] == PASSES_SCHEMA
        else _closure(library=True)
        if document["schema"] == LIBRARY_SCHEMA
        else _closure()
    )


def _owner_record(service):
    from merlin.development.targetgen.sandbox import grants as bwrap

    CommonTransformService._verify(service)
    return {
        "workspace": str(service.workspace),
        "bundle_manifest": _pin(service.bundle_manifest),
        "repo": str(service.repo),
        "snapshot_sha256": service.snapshot_digest,
        "snapshot_marker": _pin(bwrap.bundle_snapshot_root(service.workspace) / "snapshot.json"),
        "profile_sha256": _digest(service.profile),
    }


class ModelChildTransport:
    def __init__(self, *, fd, document, executor, boundary):
        self.fd, self.document, self.executor, self.boundary = fd, document, executor, boundary
        from merlin.development.backend.feedback.certification.visible_model_gate import ModelStorageReturn

        self.storage_return = ModelStorageReturn(document["host_sources"], document.get("storage_completion_key"))
        self.compiler_library_selection = active_compiler_library()
        self.compiler_pass_selection = active_compiler_passes()
        from merlin.development.targetgen.execution.host.checked_source_diagnostics import (
            active_checked_source_diagnostic,
        )

        self.checked_source_diagnostic = active_checked_source_diagnostic()
        from merlin.development.backend.feedback.execution.compiler_pass_delegation import (
            active_original_parent_pass_delegation,
        )

        self.parent_pass_delegation = active_original_parent_pass_delegation()

    @property
    def argv(self):
        return ["--original-profile-fd", str(self.fd)]

    @property
    def pass_fds(self):
        return (self.fd,)

    def _verify_original(self):
        from merlin.development.targetgen.authoring.pass_selection import active_shared_pass_requirement

        original = self.document
        if active_shared_pass_requirement() != original.get("shared_pass_requirement"):
            raise StageGateError("original model child shared-pass requirement changed")
        if (
            _pin(original["spec"]["path"]) != original["spec"]
            or _source(original["candidate"]) != original["candidate_sha256"]
            or _owner_record(self.executor.owner) != original["owner"]
            or _pin(original["target"]["descriptor"]["path"]) != original["target"]["descriptor"]
            or _digest(self.executor.target) != original["target"]["parsed_sha256"]
            or _selected_closure(original) != original["host_sources"]
            or invocation_record.environment_identity(os.environ) != original["parent_environment"]
        ):
            raise StageGateError("original model child source or execution owner changed")
        if active_compiler_library() is not self.compiler_library_selection or (
            None if self.compiler_library_selection is None else self.compiler_library_selection.record()
        ) != original.get("compiler_library"):
            raise StageGateError("original model child additional compiler library owner changed")
        if active_compiler_passes() is not self.compiler_pass_selection or (
            None if self.compiler_pass_selection is None else self.compiler_pass_selection.record()
        ) != original.get("compiler_passes"):
            raise StageGateError("original model child compiler pass owner changed")
        if "checked_source_diagnostic" in original:
            from merlin.development.targetgen.execution.host.checked_source_diagnostics import (
                active_checked_source_diagnostic,
            )

            if (
                active_checked_source_diagnostic() is not self.checked_source_diagnostic
                or self.checked_source_diagnostic.record() != original["checked_source_diagnostic"]
            ):
                raise StageGateError("original model child checked source diagnostic owner changed")
        if "parent_pass_delegation" in original:
            from merlin.development.backend.feedback.execution.compiler_pass_delegation import (
                active_original_parent_pass_delegation,
            )

            if (
                active_original_parent_pass_delegation() is not self.parent_pass_delegation
                or self.parent_pass_delegation.record() != original["parent_pass_delegation"]
            ):
                raise StageGateError("original model child live parent qualification changed")
        if self.boundary is not None:
            if self.boundary.verify() != original["resources"]["boundary_sha256"]:
                raise StageGateError("original model child kernel resource owner changed")

    def launched(self, process, command):
        self.storage_return.launched(process, command)

    def verify_result(self, result, *, completed_process=None):
        original = self.document
        self._verify_original()
        if result.get("original_compiler_child") != _receipt(original):
            raise StageGateError("model child did not complete under its original restored execution owner")
        return self.storage_return.verify_result(result, completed_process)

    def verify_checkpoint(self, receipt, *, worker):
        """Reopen parent custody for one closed case, never an entire batch."""
        self._verify_original()
        return self._verify_checkpoint_fields(receipt, worker=worker)

    def _verify_checkpoint_fields(self, receipt, *, worker):
        """Validate bound fields only; supplies no source/owner freshness authority."""
        if (
            type(receipt) is not dict
            or set(receipt) != {"schema", "original_compiler_child", "worker"}
            or receipt["schema"] != CHECKPOINT_SCHEMA
            or receipt["original_compiler_child"] != _receipt(self.document)
            or receipt["worker"] != worker
        ):
            raise StageGateError("original case checkpoint lost its sealed worker/owner selection")
        return receipt


def _receipt(document):
    return {
        "schema": document["schema"],
        "transport_sha256": _digest(document),
        "snapshot_sha256": document["owner"]["snapshot_sha256"],
        "profile_sha256": document["owner"]["profile_sha256"],
        "candidate_sha256": document["candidate_sha256"],
        "environment_sha256": document["environment"]["sha256"],
        "scope": "restored original compiler execution selections; no numerical or timing authority",
        **(
            {"compiler_library_sha256": _digest(document["compiler_library"])} if "compiler_library" in document else {}
        ),
        **({"compiler_passes_sha256": _digest(document["compiler_passes"])} if "compiler_passes" in document else {}),
        **(
            {
                "inherited_pass_qualification": {
                    "schema": "merlin.original_parent_pass_qualification.v1",
                    "delegation_sha256": _digest(document["parent_pass_delegation"]),
                    "fresh_pair_evaluation": False,
                }
            }
            if "parent_pass_delegation" in document
            else {}
        ),
    }


class _RestoredChild:
    def __init__(self, document, executor, boundary):
        from merlin.verification.lean import emission_custody as emission

        self.document, self.executor, self.boundary = document, executor, boundary
        self.admission, self.emission = API._ACTIVE.get(), emission._ACTIVE.get()
        self.modules = self.admission.modules
        self.worker, self.thread = _process(os.getpid()), threading.current_thread()
        self.compiler_library_selection = active_compiler_library()
        self.compiler_pass_selection = active_compiler_passes()

    def verify(self):
        """Check live ownership and small selections; parent reopens snapshot bytes.

        Actual command invocations already close independent snapshot operations.
        This checkpoint does not issue a reusable snapshot verification capability
        or retain a verification operation across commands.
        """
        from merlin.development.targetgen.authoring.pass_selection import active_shared_pass_requirement
        from merlin.verification.lean import emission_custody as emission

        document, owner = self.document, self.executor.owner
        if active_shared_pass_requirement() != document.get("shared_pass_requirement"):
            raise StageGateError("restored model child shared-pass requirement changed")
        if (
            self not in _RESTORED_ISSUED
            or _RESTORED_CHILD.get() is not self
            or threading.current_thread() is not self.thread
            or _process(os.getpid()) != self.worker
            or P.active_package_executor() is not self.executor
            or API._ACTIVE.get() is not self.admission
            or self.admission not in API._ISSUED
            or self.admission.modules != self.modules
            or emission._ACTIVE.get() is not self.emission
            or (
                self.emission is not None
                and (
                    self.emission not in emission._ISSUED
                    or not self.emission.live
                    or self.emission.admission is not self.admission
                    or self.emission.service is not owner
                )
            )
            or resources.active_resource_boundary() is not self.boundary
        ):
            raise StageGateError("original case checkpoint requires its live restored execution owner")
        selected = document["owner"]
        if (
            tuple(map(str, (owner.workspace, owner.bundle_manifest, owner.repo)))
            != (selected["workspace"], selected["bundle_manifest"]["path"], selected["repo"])
            or owner.snapshot_digest != selected["snapshot_sha256"]
            or _digest(owner.profile) != selected["profile_sha256"]
            or (self.admission.workspace, self.admission.bundle_manifest, self.admission.repo)
            != (owner.workspace, owner.bundle_manifest, owner.repo)
            or self.admission.profile != owner.profile
            or self.admission.snapshot_sha256 != owner.snapshot_digest
            or _pin(owner.bundle_manifest) != selected["bundle_manifest"]
            or _pin(selected["snapshot_marker"]["path"]) != selected["snapshot_marker"]
            or _pin(document["spec"]["path"]) != document["spec"]
            or _source(document["candidate"]) != document["candidate_sha256"]
            or _pin(document["target"]["descriptor"]["path"]) != document["target"]["descriptor"]
            or _digest(self.executor.target) != document["target"]["parsed_sha256"]
            or _selected_closure(document) != document["host_sources"]
            or invocation_record.environment_identity(os.environ) != document["environment"]
        ):
            raise StageGateError("original case checkpoint changed its sealed source/owner selection")
        if active_compiler_library() is not self.compiler_library_selection or (
            None if self.compiler_library_selection is None else self.compiler_library_selection.record()
        ) != document.get("compiler_library"):
            raise StageGateError("original case checkpoint changed its additional compiler library owner")
        if active_compiler_passes() is not self.compiler_pass_selection or (
            None if self.compiler_pass_selection is None else self.compiler_pass_selection.record()
        ) != document.get("compiler_passes"):
            raise StageGateError("original case checkpoint changed its compiler pass owner")
        if (
            self.boundary is not None
            and self.boundary.verify(owner_pids=(document["parent"]["pid"],))
            != document["resources"]["boundary_sha256"]
        ):
            raise StageGateError("original case checkpoint changed its kernel resource owner")
