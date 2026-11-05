"""Sealed independent model evaluator transport for snapshot_external_v1.

Only trusted host metadata crosses this descriptor. Candidate entrypoints remain
inside SnapshotExecutor boxes; these receipts assert execution identity, not
original-profile pass proofs or successful model numerics.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

from merlin.common import invocation_record
from merlin.targetgen.compiler_library import active_compiler_library, scoped_compiler_library
from . import compiler_profile_child as C, snapshot_execution as S
from merlin_experiments.phase1 import full_compiler_workspace as full

SCHEMA = "merlin.snapshot_external_model_child.v1"
FIELD = "snapshot_external_model_child"


def receipt(document):
    return {"schema": SCHEMA, "execution_mode": S.MODE,
            "selection_sha256": hashlib.sha256(C._encode_selection(document)).hexdigest(),
            "compiler_sha256": document["candidate_sha256"],
            "spec_sha256": document["spec"]["sha256"]}


class Transport:
    def __init__(self, fd, document, executor):
        self.fd, self.document, self.executor = fd, document, executor
        from .model_storage_completion import ModelStorageReturn

        self.storage_return = ModelStorageReturn(document['host_sources'], document.get('storage_completion_key'))

    @property
    def argv(self):
        return ["--snapshot-external-fd", str(self.fd)]

    @property
    def pass_fds(self):
        return (self.fd,)

    def launched(self, process, command):
        self.storage_return.launched(process, command)

    def verify_result(self, result, *, completed_process=None):
        d = self.document
        self.executor.owner._verify()
        if (S.active_snapshot() is not self.executor
                or S.source_identity(d["candidate"]) != d["candidate_sha256"]
                or C._pin(d["spec"]["path"]) != d["spec"]
                or self.executor.owner.record() != d['owner']
                or result.get(FIELD) != receipt(d)):
            raise ValueError("external model result lost its admitted snapshot/spec/receipt")
        return self.storage_return.verify_result(result, completed_process)


@contextlib.contextmanager
def prepare(*, spec_path, environment, package_dir):
    executor = S.active_snapshot()
    if executor is None or Path(package_dir).resolve() != executor.candidate:
        raise ValueError("external model child requires its admitted compiler snapshot")
    target = executor.target
    spec = json.loads(Path(spec_path).read_bytes())
    if spec.get("package_dir") != str(executor.candidate) or spec.get("target") != target.target:
        raise ValueError("external model job selects another compiler or target")
    if environment.get(S.ENV) != S.MODE:
        raise ValueError("external model child lost its explicit execution mode")
    library = active_compiler_library()
    editable = full.selected()
    boundary = C.resources.active_resource_boundary()
    from .model_storage_completion import implementation_pins

    document = {"schema": SCHEMA, "parent": C._process(os.getpid()),
        "spec": C._pin(spec_path), "environment": invocation_record.environment_identity(environment),
        "candidate": str(executor.candidate), "candidate_sha256": executor.source_sha256,
        "owner": executor.owner.record(),
        "target": {"descriptor": C._pin(target.path), "parsed_sha256": C._digest(target),
                   "source_root": str(target.source_root) if target.source_root else None},
        "inputs": C._inputs_record(executor.inputs if executor.inputs is not None
                                    else C.C.select_package_sandbox_inputs(target)),
        "compiler_library": None if library is None else library.record(),
        "full_compiler": None if editable is None else editable.record(),
        "python": C._pin(Path(sys.executable).resolve()),
        "host_sources": [C._pin(Path(module.__file__)) for module in (C, S, sys.modules[__name__])] + implementation_pins(),
        "storage_completion_key": os.urandom(32).hex(),
        "work_root": str(Path(spec_path).resolve().parent / "compiler"), "resources": None}
    if boundary is not None:
        document["resources"] = {"cpu_ids": boundary.cpu_ids, "ram_bytes": boundary.ram_bytes,
            "disk_bytes": boundary.disk_bytes, "writable_roots": list(map(str, boundary.writable_roots)),
            "boundary_sha256": boundary.verify(writable_roots=(Path(spec_path).resolve().parent,))}
    fd = C._memfd()
    readonly = None
    try:
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(C._encode_selection(document))
        fcntl.fcntl(fd, C._ADD_SEALS, C._SEALS)
        readonly = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
        os.close(fd); fd = None
        yield Transport(readonly, document, executor)
    finally:
        for descriptor in (fd, readonly):
            if descriptor is not None:
                os.close(descriptor)


def read_selection(fd, *, spec_path, spec_bytes):
    if (not S.selected() or type(fd) is not int or fd < 3
            or S.P.active_package_executor() is not None
            or fcntl.fcntl(fd, C._GET_SEALS) & C._SEALS != C._SEALS
            or fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
        raise ValueError("external model requires an original sealed read-only child descriptor")
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(fd, "rb") as stream:
        document = C._decode_selection(stream.read(C.MAX_BYTES + 1))
    if (document.get("schema") != SCHEMA or document["parent"] != C._process(os.getppid())
            or document["spec"] != C._pin(spec_path)
            or hashlib.sha256(spec_bytes).hexdigest() != document["spec"]["sha256"]
            or document["environment"] != invocation_record.environment_identity(os.environ)
            or document["python"] != C._pin(Path(sys.executable).resolve())):
        raise ValueError("external model child parent/spec/environment identity changed")
    from .model_storage_completion import implementation_pins

    expected = [C._pin(Path(module.__file__)) for module in (C, S, sys.modules[__name__])] + implementation_pins()
    if document["host_sources"] != expected or S.source_identity(document["candidate"]) != document["candidate_sha256"]:
        raise ValueError("external model child host or compiler source changed")
    return document


@contextlib.contextmanager
def restore(fd, *, spec_path, spec_bytes):
    document = read_selection(fd, spec_path=spec_path, spec_bytes=spec_bytes)
    owner = document["owner"]
    from .snapshot_inputs import SnapshotInputs
    service = SnapshotInputs(workspace=Path(owner["workspace"]),
        bundle_manifest=Path(owner["bundle_manifest"]["path"]), repo=Path(owner["repo"]))
    if service.record() != owner:
        raise ValueError("external model selected input owner changed")
    selected = document["target"]
    if C._pin(selected["descriptor"]["path"]) != selected["descriptor"]:
        raise ValueError("external model target descriptor changed")
    target = C.load_target_experiment(selected["descriptor"]["path"],
        source_root=Path(selected["source_root"]) if selected["source_root"] else None)
    spec = json.loads(spec_bytes)
    if (C._digest(target) != selected["parsed_sha256"]
            or spec.get("target") != target.target or spec.get("package_dir") != document["candidate"]):
        raise ValueError("external model job changed target or compiler")
    with contextlib.ExitStack() as stack:
        from .model_storage_completion import signing_scope

        stack.enter_context(signing_scope(document.get('storage_completion_key')))
        editable = document["full_compiler"]
        selected_full = None if editable is None else full.select(
            editable["root"], editable["manifest_path"], editable["manifest_sha256"])
        if selected_full is not None and selected_full.record() != editable:
            raise ValueError("external model full compiler baseline changed")
        stack.enter_context(full.scoped_full_compiler(selected_full))
        if document["compiler_library"] is not None:
            from merlin_experiments.phase2.compiler_library_state import restore_selection
            stack.enter_context(scoped_compiler_library(restore_selection(document["compiler_library"])))
        record = document["resources"]
        boundary = None
        if record is not None:
            boundary = C.resources.prepare_external_resource_boundary(cpu_ids=tuple(record["cpu_ids"]),
                ram_bytes=record["ram_bytes"], disk_bytes=record["disk_bytes"],
                writable_roots=tuple(map(Path, record["writable_roots"])))
            if boundary.verify(owner_pids=(os.getppid(),)) != record["boundary_sha256"]:
                raise ValueError("external model resource owner changed")
        stack.enter_context(C.resources.executing_resource_boundary(boundary))
        stack.enter_context(S.execution(service, target_experiment=target,
            candidate=Path(document["candidate"]), work_root=Path(document["work_root"]),
            sandbox_inputs=C._restore_inputs(document["inputs"])))
        yield receipt(document)
    if C._pin(spec_path) != document["spec"]:
        raise ValueError("external model specification changed while grading")
