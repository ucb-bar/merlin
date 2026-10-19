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

from merlin.common import cpu_pool, invocation_record
from merlin.common.bundle_snapshot import verified_snapshot_scope
from merlin.common.digest import sha256_file
from merlin.targetgen import compiler_profile_admission as API
from merlin.targetgen import package_runtime as P
from merlin.targetgen.compiler_library import active_compiler_library, scoped_compiler_library
from merlin.targetgen.compiler_pass_selection import active_compiler_passes, scoped_compiler_passes
from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as C
from merlin_experiments.phase2 import external_resources as resources
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.retained_revisions import source_inventory

SCHEMA = "merlin.original_model_child.v1"
LIBRARY_SCHEMA = "merlin.original_model_child.v2"
PASSES_SCHEMA = "merlin.original_model_child.v3"
CHECKED_SOURCE_SCHEMA = "merlin.original_model_child.v4"
CHECKPOINT_SCHEMA = "merlin.original_model_child_case_checkpoint.v1"
_RESTORED_CHILD = contextvars.ContextVar("original_restored_child_checkpoint", default=None)
_RESTORED_ISSUED = weakref.WeakSet()
MAX_BYTES = 256 << 10
MAX_SELECTION_BYTES = 4 << 20
_COMPRESSED_SELECTION = b"merlin.original_model_child.zlib.v1\0"
# Linux UAPI linux/fcntl.h: some otherwise compatible CPython builds omit these
# names and os.memfd_create. Kernel enforcement is identical in either binding.
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
    if type(executor) is not R._Executor or admission is None or type(executor.owner) is not CommonTransformService:
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
    executor = _original_executor()
    return None if executor is None else _plain(executor.work_root)


def _inputs_record(inputs):
    if type(inputs) is C.PackageSandboxInputs:
        return {"kind": "explicit", "values": _data(inputs)}
    if type(inputs) is C.FrozenPackageSandboxInputs:
        from merlin_experiments.phase2.qualification_policy import restore

        if restore(inputs.root, inputs.record) != inputs:
            raise StageGateError("model child frozen package inputs changed")
        return {"kind": "frozen", "root": str(inputs.root), "record": inputs.record}
    raise StageGateError("model child requires the exact original package sandbox selection")


def _restore_inputs(record):
    from merlin.targetgen.sandbox import toolchain as TC
    from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface

    if record["kind"] == "frozen":
        from merlin_experiments.phase2.qualification_policy import restore

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


def _closure(*, library=False, passes=False, checked_source=False):
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
        "merlin.targetgen.capsule_runner",
        "merlin.targetgen.capsule_candidate_context",
        "merlin.targetgen.capsule_source_identity",
        "merlin_experiments.phase1.feedback.operator_child",
        "merlin_experiments.phase1.feedback.scale_probe_child",
        "merlin_experiments.phase1.feedback.scale_probe_batch",
        "merlin_experiments.phase1.feedback.compile_only",
        "merlin.targetgen.simulation_certificate",
    )
    if library:
        names += (
            "merlin.targetgen.compiler_library",
            "merlin_experiments.phase2.compiler_library_state",
            "merlin_experiments.phase2.compiler_libraries",
        )
    if passes:
        names += (
            "merlin.targetgen.compiler_pass_selection",
            "merlin.llvmlower.reviewed_pass_capabilities",
        )
    if checked_source:
        names += (
            "merlin.targetgen.checked_source_model_diagnostic",
            "merlin.targetgen.checked_source_model_diagnostic_execution",
            "merlin_experiments.phase0.connected_integer_qualification",
        )
    return [_pin(module_source_path(name)) for name in names]


def _selected_closure(document):
    if document["schema"] == CHECKED_SOURCE_SCHEMA:
        return _closure(library="compiler_library" in document, passes=True, checked_source=True)
    return (
        _closure(library="compiler_library" in document, passes=True)
        if document["schema"] == PASSES_SCHEMA
        else _closure(library=True) if document["schema"] == LIBRARY_SCHEMA else _closure()
    )


def _owner_record(service):
    from merlin.targetgen.sandbox import bwrap

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
        self.compiler_library_selection = active_compiler_library()
        self.compiler_pass_selection = active_compiler_passes()
        from merlin.targetgen.checked_source_model_diagnostic import active_checked_source_diagnostic

        self.checked_source_diagnostic = active_checked_source_diagnostic()

    @property
    def argv(self):
        return ["--original-profile-fd", str(self.fd)]

    @property
    def pass_fds(self):
        return (self.fd,)

    def _verify_original(self):
        original = self.document
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
            from merlin.targetgen.checked_source_model_diagnostic import active_checked_source_diagnostic

            if (active_checked_source_diagnostic() is not self.checked_source_diagnostic
                    or self.checked_source_diagnostic.record() != original["checked_source_diagnostic"]):
                raise StageGateError("original model child checked source diagnostic owner changed")
        if self.boundary is not None:
            if self.boundary.verify() != original["resources"]["boundary_sha256"]:
                raise StageGateError("original model child kernel resource owner changed")

    def verify_result(self, result):
        original = self.document
        self._verify_original()
        if result.get("original_compiler_child") != _receipt(original):
            raise StageGateError("model child did not complete under its original restored execution owner")
        return result

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
    }


class _RestoredChild:
    def __init__(self, document, executor, boundary):
        from merlin.verify import lean_emission_context as emission

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
        from merlin.verify import lean_emission_context as emission

        document, owner = self.document, self.executor.owner
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


@contextlib.contextmanager
def restored_case_checkpoint():
    """Publishable only after this actual restored worker's case scope closes."""
    state = _RESTORED_CHILD.get()
    if type(state) is not _RestoredChild:
        raise StageGateError("original case checkpoint requires an inherited restored worker")
    state.verify()
    receipt = {}
    yield receipt
    state.verify()
    receipt.update(schema=CHECKPOINT_SCHEMA, original_compiler_child=_receipt(state.document), worker=state.worker)


def require_restored_model_context(field, record, *, package_dir, target):
    """Require this exact context in the actual sealed v4 worker's source spec.

    A plain dictionary, an issued pass owner in another process, or a copied
    worker checkpoint does not select a diagnostic execution route.
    """
    state = _RESTORED_CHILD.get()
    if type(state) is not _RestoredChild:
        raise StageGateError("checked model context requires the original restored sealed worker")
    state.verify()
    document = state.document
    spec = json.loads(Path(document["spec"]["path"]).read_bytes())
    if (document["schema"] != CHECKED_SOURCE_SCHEMA or field != "checked_source_diagnostic"
            or document.get(field) != record
            or (spec.get("candidate_context") or {}).get(field) != record
            or spec.get("package_dir") != str(package_dir)
            or spec.get("target") != target
            or document["candidate"] != str(package_dir)):
        raise StageGateError("checked model context differs from its original sealed parent/spec")
    return copy_record(record)


def copy_record(record):
    return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))


@contextlib.contextmanager
def prepare_model_child(*, spec_path, environment, package_dir):
    """Issue a sealed FD only from the actual original parent execution scope.

    ``environment`` is the exact mapping the child is launched with; build it
    with :func:`merlin.common.cpu_pool.child_environment` so the launcher adds
    nothing after it is sealed.
    """
    library = active_compiler_library()
    passes = active_compiler_passes()
    from merlin.targetgen import checked_source_model_diagnostic as diagnostic

    spec = json.loads(_plain(spec_path).read_bytes())
    checked = (spec.get("candidate_context") or {}).get(diagnostic.CONTEXT_FIELD)
    selected_diagnostic = diagnostic.active_checked_source_diagnostic()
    if checked is not None:
        if (type(selected_diagnostic) is not diagnostic.CheckedSourceModelDiagnostic
                or selected_diagnostic.record() != checked):
            raise StageGateError("checked model child requires the actual original host diagnostic selection")
        selected_diagnostic.verify_for(spec["capsule"], target=spec["target"])
    executor = _original_executor()
    if executor is None:
        if library is not None or passes is not None or checked is not None:
            raise StageGateError("additional compiler library/pass model child requires its original execution owner")
        yield None
        return
    if passes is not None and passes.factory is None:
        raise StageGateError("compiler pass model child requires its original independent admission factory")
    candidate = _plain(package_dir)
    if _source(candidate) != executor.source_sha256:
        raise StageGateError("model child candidate is another compiler source revision")
    te = executor.target
    if type(te) is not TargetExperiment or _pin(te.path)["sha256"] != te.descriptor_sha256:
        raise StageGateError("model child target is not its original parsed descriptor")
    inputs = executor.inputs if executor.inputs is not None else C.select_package_sandbox_inputs(te)
    boundary = resources.active_resource_boundary()
    if boundary is not None and type(boundary) is not resources.ExternalResourceBoundary:
        raise StageGateError("model child resources require the exact live kernel-issued owner")
    if cpu_pool.child_environment(environment) != dict(environment):
        # Its launcher would add the class after sealing; the child would then refuse its environment.
        raise StageGateError("model child environment lacks the caller's CPU lease class")
    document = {
        "schema": CHECKED_SOURCE_SCHEMA if checked is not None else (
            PASSES_SCHEMA if passes is not None else LIBRARY_SCHEMA if library is not None else SCHEMA),
        **({diagnostic.CONTEXT_FIELD: checked} if checked is not None else {}),
        **({"compiler_library": library.record()} if library is not None else {}),
        **({"compiler_passes": passes.record()} if passes is not None else {}),
        "parent": _process(os.getpid()),
        "spec": _pin(spec_path),
        "environment": invocation_record.environment_identity(environment),
        "parent_environment": invocation_record.environment_identity(os.environ),
        "candidate": str(candidate),
        "candidate_sha256": executor.source_sha256,
        "owner": _owner_record(executor.owner),
        "target": {
            "descriptor": _pin(te.path),
            "source_root": str(te.source_root) if te.source_root else None,
            "parsed_sha256": _digest(te),
        },
        "inputs": _inputs_record(inputs),
        "host_sources": (
            _closure(library=library is not None, passes=True, checked_source=checked is not None)
            if passes is not None
            else _closure(library=library is not None)
        ),
        "python": _pin(Path(sys.executable).resolve()),
        "work_root": str(_plain(spec_path).parent / "compiler"),
        "resources": None,
    }
    if boundary is not None:
        document["resources"] = {
            "cpu_ids": boundary.cpu_ids,
            "ram_bytes": boundary.ram_bytes,
            "disk_bytes": boundary.disk_bytes,
            "writable_roots": [str(path) for path in boundary.writable_roots],
            "boundary_sha256": boundary.verify(writable_roots=(_plain(spec_path).parent,)),
        }
    with _sealed_transport(document, executor=executor, boundary=boundary) as transport:
        yield transport


@contextlib.contextmanager
def _sealed_transport(document, *, executor, boundary):
    raw = _encode_selection(document)
    fd = _memfd()
    readonly = None
    try:
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(raw)
        fcntl.fcntl(fd, _ADD_SEALS, _SEALS)
        readonly = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
        os.close(fd)
        fd = None
        yield ModelChildTransport(fd=readonly, document=_data(document), executor=executor, boundary=boundary)
    finally:
        for descriptor in (fd, readonly):
            if descriptor is not None:
                os.close(descriptor)


@contextlib.contextmanager
def prepare_model_children(*, spec_paths, environment, package_dir):
    """Issue distinct case capabilities from one live original setup operation.

    Common selections are checked once, while every specification remains a
    separately pinned, sealed descriptor with its own restored child namespace.
    The caller must close its exact snapshot operation before launching children;
    every child's restoration and each transport's final verifier stay unchanged.
    """
    from merlin.common import bundle_snapshot as BW

    paths = tuple(_plain(path) for path in spec_paths)
    roots = tuple(path.parent / "compiler" for path in paths)
    if not paths or len(set(paths)) != len(paths) or len(set(roots)) != len(roots):
        raise StageGateError("original child batch requires distinct specifications and work roots")
    issued = BW._ACTIVE_VERIFICATION.get()
    executor = P.active_package_executor()
    if type(executor) is not R._Executor or type(executor.owner) is not CommonTransformService:
        raise StageGateError("original child batch requires its exact original execution owner")
    owner = executor.owner

    def operation():
        if (
            BW._ACTIVE_VERIFICATION.get() is not issued
            or not BW._current_verification(issued)
            or issued.root != BW.bundle_snapshot_root(owner.workspace)
            or issued.repo != owner.repo
            or issued.policy_bytes != BW._policy_bytes(owner.bundle)
        ):
            raise StageGateError("original child batch requires its live exact-owner snapshot operation")

    operation()
    with contextlib.ExitStack() as lifetime:
        first = lifetime.enter_context(
            prepare_model_child(spec_path=paths[0], environment=environment, package_dir=package_dir)
        )
        if first is None or first.executor is not executor:
            raise StageGateError("original child batch lost its live execution owner")
        if (
            first.boundary is not None
            and first.boundary.verify(writable_roots=tuple(path.parent for path in paths))
            != first.document["resources"]["boundary_sha256"]
        ):
            raise StageGateError("original child batch changed its live resource boundary")
        transports = [first]
        for path, root in zip(paths[1:], roots[1:], strict=True):
            document = {**first.document, "spec": _pin(path), "work_root": str(root)}
            transports.append(
                lifetime.enter_context(_sealed_transport(document, executor=executor, boundary=first.boundary))
            )
        if len({transport.fd for transport in transports}) != len(transports):
            raise StageGateError("original child batch reused a capability descriptor")
        # Reopen shared source/closure/owner selections once at the end of this
        # issuance operation; a mutated later spec cannot stale the first seal.
        first._verify_original()
        if invocation_record.environment_identity(environment) != first.document["environment"]:
            raise StageGateError("original child batch changed its selected child environment")
        for transport in transports:
            if _pin(transport.document["spec"]["path"]) != transport.document["spec"]:
                raise StageGateError("original child batch specification changed during issuance")
        operation()
        yield tuple(transports)


@contextlib.contextmanager
def restore_model_child(fd, *, spec_path, spec_bytes):
    """Reopen sealed parent provenance and enter original execution before work."""
    if (
        type(fd) is not int
        or fd < 3
        or P.active_package_executor() is not None
        or API.active_compiler_imports() is not None
        or active_compiler_library() is not None
        or active_compiler_passes() is not None
    ):
        raise StageGateError("model child requires a fresh interpreter and original inherited descriptor")
    if fcntl.fcntl(fd, _GET_SEALS) & _SEALS != _SEALS:
        raise StageGateError("model child capability descriptor is not immutable and sealed")
    if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        raise StageGateError("model child capability descriptor retains writable authority")
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    document = _decode_selection(raw)
    if (
        document.get("schema") not in (SCHEMA, LIBRARY_SCHEMA, PASSES_SCHEMA, CHECKED_SOURCE_SCHEMA)
        or (document.get("schema") == SCHEMA and "compiler_library" in document)
        or (document.get("schema") == LIBRARY_SCHEMA and "compiler_library" not in document)
        or (document.get("schema") not in (PASSES_SCHEMA, CHECKED_SOURCE_SCHEMA) and "compiler_passes" in document)
        or (document.get("schema") in (PASSES_SCHEMA, CHECKED_SOURCE_SCHEMA) and "compiler_passes" not in document)
        or (document.get("schema") != CHECKED_SOURCE_SCHEMA and "checked_source_diagnostic" in document)
        or (document.get("schema") == CHECKED_SOURCE_SCHEMA and "checked_source_diagnostic" not in document)
        or document["parent"] != _process(os.getppid())
        or document["spec"] != _pin(spec_path)
        or type(spec_bytes) is not bytes
        or hashlib.sha256(spec_bytes).hexdigest() != document["spec"]["sha256"]
        or document["environment"] != invocation_record.environment_identity(os.environ)
        or document["python"] != _pin(Path(sys.executable).resolve())
    ):
        raise StageGateError("model child lost original parent/spec/environment/interpreter provenance")
    for pin in document["host_sources"]:
        if _pin(pin["path"]) != pin:
            raise StageGateError("model child original invoked host source changed")
    if (
        _selected_closure(document) != document["host_sources"]
        or _source(document["candidate"]) != document["candidate_sha256"]
    ):
        raise StageGateError("model child resolved another host or compiler source revision")
    owner = document["owner"]
    workspace, archive, repo = (
        _plain(owner[key]["path"] if key == "bundle_manifest" else owner[key])
        for key in ("workspace", "bundle_manifest", "repo")
    )
    bundle = yaml.safe_load(archive.read_bytes())
    with contextlib.ExitStack() as execution:
        library = None
        if "compiler_library" in document:
            from merlin_experiments.phase2.compiler_library_state import restore_selection

            library = restore_selection(document["compiler_library"])
            execution.enter_context(scoped_compiler_library(library))
        passes = None
        if document["schema"] in (PASSES_SCHEMA, CHECKED_SOURCE_SCHEMA):
            from merlin.targetgen.compiler_pass_selection import restore_original_selection

            _plain(document["work_root"]).mkdir(parents=True, exist_ok=True)
            passes = restore_original_selection(
                document["compiler_passes"], work_root=_plain(document["work_root"]) / "pass-admission"
            )
            execution.enter_context(scoped_compiler_passes(passes))
        # Batch only exact-owner setup. Close its full final payload read before
        # yielding the grader capability; actual calls and final checks stay fresh.
        with verified_snapshot_scope(workspace, bundle, repo=repo):
            service = CommonTransformService(
                workspace=workspace,
                bundle_manifest=archive,
                repo=repo,
            )
            if _owner_record(service) != owner:
                raise StageGateError("model child original frozen archive/profile changed")
            target = document["target"]
            if _pin(target["descriptor"]["path"]) != target["descriptor"]:
                raise StageGateError("model child original target descriptor changed")
            te = load_target_experiment(
                target["descriptor"]["path"], source_root=Path(target["source_root"]) if target["source_root"] else None
            )
            if _digest(te) != target["parsed_sha256"]:
                raise StageGateError("model child target differs from original parsed selection")
            spec = json.loads(spec_bytes)
            if spec.get("package_dir") != document["candidate"] or spec.get("target") != te.target:
                raise StageGateError("model child job selects another original compiler or target")
            selected = document["resources"]
            boundary = None
            if selected is not None:
                boundary = resources.prepare_external_resource_boundary(
                    cpu_ids=tuple(selected["cpu_ids"]),
                    ram_bytes=selected["ram_bytes"],
                    disk_bytes=selected["disk_bytes"],
                    writable_roots=tuple(_plain(value) for value in selected["writable_roots"]),
                )
                if boundary.verify(owner_pids=(os.getppid(),)) != selected["boundary_sha256"]:
                    raise StageGateError("model child did not inherit the original live kernel resource boundary")
            execution.enter_context(resources.executing_resource_boundary(boundary))
            execution.enter_context(
                R._original_execution_lifetime(
                    service,
                    target_experiment=te,
                    candidate=_plain(document["candidate"]),
                    work_root=_plain(document["work_root"]),
                    sandbox_inputs=_restore_inputs(document["inputs"]),
                )
            )
            if _owner_record(service) != owner:
                raise StageGateError("model child original execution owner changed during setup")
        state = _RestoredChild(document, R.active_original_execution(), boundary)
        _RESTORED_ISSUED.add(state)
        token = _RESTORED_CHILD.set(state)
        try:
            if document["schema"] == CHECKED_SOURCE_SCHEMA:
                from merlin.targetgen import checked_source_model_diagnostic as diagnostic

                checked = diagnostic.restore_checked_source_diagnostic(
                    document[diagnostic.CONTEXT_FIELD], capsule=spec["capsule"], target=spec["target"],
                    package_dir=spec["package_dir"],
                )
                with diagnostic.scoped_checked_source_diagnostic(checked):
                    yield _receipt(document)
            else:
                yield _receipt(document)
        finally:
            _RESTORED_CHILD.reset(token)
            _RESTORED_ISSUED.discard(state)
    if _pin(spec_path) != document["spec"] or _source(document["candidate"]) != document["candidate_sha256"]:
        raise StageGateError("model child original source changed during grading")
    if _owner_record(service) != owner:
        raise StageGateError("model child original execution owner changed during grading")
    if (
        _pin(target["descriptor"]["path"]) != target["descriptor"]
        or _selected_closure(document) != document["host_sources"]
        or invocation_record.environment_identity(os.environ) != document["environment"]
    ):
        raise StageGateError("model child original target/host sources/environment changed during grading")
    if library is not None and library.record() != document["compiler_library"]:
        raise StageGateError("model child additional compiler library changed during grading")
    if passes is not None and passes.record() != document["compiler_passes"]:
        raise StageGateError("model child compiler pass selection changed during grading")


def run_model_grade(argv):
    """The original runner CLI child entry, inside the restored host scope."""
    import argparse

    from merlin.targetgen import capsule_runner as runner

    sub = argparse.ArgumentParser(add_help=False)
    sub.add_argument("--model-grade", required=True)
    sub.add_argument("--model-grade-out", required=True)
    sub.add_argument("--original-profile-fd", type=int)
    sa, _ = sub.parse_known_args(argv)
    runner._die_with_parent()
    spec_bytes = Path(sa.model_grade).read_bytes()
    spec = json.loads(spec_bytes)
    from contextlib import nullcontext

    scope = (
        nullcontext(None)
        if sa.original_profile_fd is None
        else restore_model_child(sa.original_profile_fd, spec_path=Path(sa.model_grade), spec_bytes=spec_bytes)
    )
    with scope as child_receipt:
        if spec.get("candidate_context") is not None:
            res = runner._grade_candidate_model_capsule_inline(
                spec["capsule"],
                target=spec["target"],
                timeout=int(spec["timeout"]),
                package_dir=spec["package_dir"],
                context=spec["candidate_context"],
            )
        else:
            res = runner._grade_model_capsule_inline(
                spec["capsule"],
                target=spec.get("target"),
                timeout=int(spec["timeout"]),
                package_dir=spec.get("package_dir"),
                numeric_policy=spec.get("numeric_policy", runner._MODEL_NUMERIC_UNSET),
                numeric_policy_identity=spec.get("numeric_policy_identity"),
            )
    if child_receipt is not None:
        res["original_compiler_child"] = child_receipt
    Path(sa.model_grade_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0
