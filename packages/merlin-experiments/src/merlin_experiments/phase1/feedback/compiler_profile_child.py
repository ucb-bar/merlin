"""Sealed original-host compiler capability transport into a budgeted grader.

Only the trusted grader receives the readonly sealed FD. Candidate commands do
not inherit it. Original archives, source identity and actual kernel controls
are reopened before installing the ordinary per-call package/proof executor.
This transports existing authority; it supplies no numerical or timing verdict.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

import yaml

from merlin.common import invocation_record
from merlin.common.bundle_snapshot import verified_snapshot_scope
from merlin.common.digest import sha256_file
from merlin.targetgen import compiler_profile_admission as API
from merlin.targetgen import package_runtime as P
from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment
from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as C
from merlin_experiments.phase2 import external_resources as resources
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.retained_revisions import source_inventory

SCHEMA = "merlin.original_model_child.v1"
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
            decoded = decoder.decompress(raw[len(_COMPRESSED_SELECTION):], MAX_SELECTION_BYTES + 1)
        except zlib.error as error:
            raise StageGateError("model child compressed selection is invalid") from error
        if (len(decoded) > MAX_SELECTION_BYTES or not decoder.eof
            or decoder.unused_data or decoder.unconsumed_tail):
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


def _closure():
    """Direct invoked host owners; this makes no transitive dependency claim."""
    from merlin.common.paths import module_source_path

    return [
        _pin(module_source_path(name))
        for name in (
            __name__,
            R.__name__,
            CommonTransformService.__module__,
            API.__name__,
            P.__name__,
            C.__name__,
            resources.__name__,
            resources.kernel.__name__,
            "merlin.targetgen.capsule_runner",
            "merlin_experiments.phase1.feedback.operator_child",
        )
    ]


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

    @property
    def argv(self):
        return ["--original-profile-fd", str(self.fd)]

    @property
    def pass_fds(self):
        return (self.fd,)

    def verify_result(self, result):
        original = self.document
        if (
            _pin(original["spec"]["path"]) != original["spec"]
            or _source(original["candidate"]) != original["candidate_sha256"]
            or _owner_record(self.executor.owner) != original["owner"]
            or _pin(original["target"]["descriptor"]["path"]) != original["target"]["descriptor"]
            or _digest(self.executor.target) != original["target"]["parsed_sha256"]
            or _closure() != original["host_sources"]
            or invocation_record.environment_identity(os.environ) != original["parent_environment"]
        ):
            raise StageGateError("original model child source or execution owner changed")
        if self.boundary is not None:
            if self.boundary.verify() != original["resources"]["boundary_sha256"]:
                raise StageGateError("original model child kernel resource owner changed")
        if result.get("original_compiler_child") != _receipt(original):
            raise StageGateError("model child did not complete under its original restored execution owner")
        return result


def _receipt(document):
    return {
        "schema": SCHEMA,
        "transport_sha256": _digest(document),
        "snapshot_sha256": document["owner"]["snapshot_sha256"],
        "profile_sha256": document["owner"]["profile_sha256"],
        "candidate_sha256": document["candidate_sha256"],
        "environment_sha256": document["environment"]["sha256"],
        "scope": "restored original compiler execution selections; no numerical or timing authority",
    }


@contextlib.contextmanager
def prepare_model_child(*, spec_path, environment, package_dir):
    """Issue a sealed FD only from the actual original parent execution scope."""
    executor = _original_executor()
    if executor is None:
        yield None
        return
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
    document = {
        "schema": SCHEMA,
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
        "host_sources": _closure(),
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
def restore_model_child(fd, *, spec_path, spec_bytes):
    """Reopen sealed parent provenance and enter original execution before work."""
    if (
        type(fd) is not int
        or fd < 3
        or P.active_package_executor() is not None
        or API.active_compiler_imports() is not None
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
        document.get("schema") != SCHEMA
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
    if _closure() != document["host_sources"] or _source(document["candidate"]) != document["candidate_sha256"]:
        raise StageGateError("model child resolved another host or compiler source revision")
    owner = document["owner"]
    workspace, archive, repo = (_plain(owner[key]["path"] if key == "bundle_manifest" else owner[key])
                                for key in ("workspace", "bundle_manifest", "repo"))
    bundle = yaml.safe_load(archive.read_bytes())
    with contextlib.ExitStack() as execution:
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
            execution.enter_context(R._original_execution_lifetime(
                service,
                target_experiment=te,
                candidate=_plain(document["candidate"]),
                work_root=_plain(document["work_root"]),
                sandbox_inputs=_restore_inputs(document["inputs"]),
            ))
            if _owner_record(service) != owner:
                raise StageGateError("model child original execution owner changed during setup")
        yield _receipt(document)
    if _pin(spec_path) != document["spec"] or _source(document["candidate"]) != document["candidate_sha256"]:
        raise StageGateError("model child original source changed during grading")
    if _owner_record(service) != owner:
        raise StageGateError("model child original execution owner changed during grading")
    if (
        _pin(target["descriptor"]["path"]) != target["descriptor"]
        or _closure() != document["host_sources"]
        or invocation_record.environment_identity(os.environ) != document["environment"]
    ):
        raise StageGateError("model child original target/host sources/environment changed during grading")


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
