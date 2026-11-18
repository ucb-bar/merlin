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


@contextlib.contextmanager
def restored_case_checkpoint():
    """Publishable only after this actual restored worker's case scope closes."""
    from merlin.development.backend.feedback.execution.compiler_profile_child import (
        _RESTORED_CHILD,
        CHECKPOINT_SCHEMA,
        _receipt,
        _RestoredChild,
    )

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
    from merlin.development.backend.feedback.execution.compiler_profile_child import (
        _RESTORED_CHILD,
        CHECKED_SOURCE_SCHEMA,
        DELEGATED_PASSES_SCHEMA,
        _RestoredChild,
    )

    state = _RESTORED_CHILD.get()
    if type(state) is not _RestoredChild:
        raise StageGateError("checked model context requires the original restored sealed worker")
    state.verify()
    document = state.document
    spec = json.loads(Path(document["spec"]["path"]).read_bytes())
    if (
        document["schema"] not in (CHECKED_SOURCE_SCHEMA, DELEGATED_PASSES_SCHEMA)
        or field != "checked_source_diagnostic"
        or document.get(field) != record
        or (spec.get("candidate_context") or {}).get(field) != record
        or spec.get("package_dir") != str(package_dir)
        or spec.get("target") != target
        or document["candidate"] != str(package_dir)
    ):
        raise StageGateError("checked model context differs from its original sealed parent/spec")
    return copy_record(record)


def copy_record(record):
    return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))


@contextlib.contextmanager
def prepare_model_child(*, spec_path, environment, package_dir):
    """Complete original metadata checks before exposing the child transport.

    An opted-in selection bounds only document construction. Its final full
    owner check completes before the caller can launch or grade the child;
    no registry metadata scope survives the yield, wait or result checks.
    """
    from merlin.development.backend.feedback.execution import (
        snapshot_evaluator as snapshot_execution,
    )
    from merlin.development.backend.feedback.execution import (
        snapshot_model_child,
    )
    from merlin.development.backend.feedback.execution.compiler_profile_child import _original_executor

    if snapshot_execution.selected():
        with snapshot_model_child.prepare(
            spec_path=spec_path, environment=environment, package_dir=package_dir
        ) as transport:
            yield transport
        return
    from merlin.development.targetgen.authoring.pass_selection import active_shared_pass_requirement

    if active_shared_pass_requirement() is not None and _original_executor() is None:
        raise StageGateError("shared-pass model child requires its original execution owner")
    from merlin.compiler.lower.source import reviewed as reviewed_metadata

    if reviewed_metadata._ACTIVE.get() is not None:
        raise StageGateError("finish the original registry metadata operation before preparing a model child")
    passes = active_compiler_passes()
    operation = passes.metadata_operation() if type(passes) is CompilerPassSelection else contextlib.nullcontext()
    with contextlib.ExitStack() as transport:
        with operation:
            selected = transport.enter_context(
                _prepare_model_child(spec_path=spec_path, environment=environment, package_dir=package_dir)
            )
        yield selected


@contextlib.contextmanager
def _prepare_model_child(*, spec_path, environment, package_dir):
    """Issue a sealed FD only from the actual original parent execution scope.

    ``environment`` is the exact mapping the child is launched with; build it
    with :func:`merlin.common.process.cpu_pool.child_environment` so the launcher adds
    nothing after it is sealed.
    """
    from merlin.development.backend.feedback.execution.compiler_profile_child import (
        CHECKED_SOURCE_SCHEMA,
        DELEGATED_PASSES_SCHEMA,
        LIBRARY_SCHEMA,
        PASSES_SCHEMA,
        _closure,
        _digest,
        _inputs_record,
        _original_executor,
        _owner_record,
        _pin,
        _plain,
        _process,
        _source,
    )
    from merlin.development.backend.feedback.execution.snapshot_model_child import SCHEMA

    library = active_compiler_library()
    passes = active_compiler_passes()
    from merlin.development.targetgen.authoring.pass_selection import active_shared_pass_requirement

    shared_requirement = active_shared_pass_requirement()
    from merlin.development.backend.feedback.execution.compiler_pass_delegation import (
        active_original_parent_pass_delegation,
    )
    from merlin.development.targetgen.execution.host import checked_source_diagnostics as diagnostic

    spec = json.loads(_plain(spec_path).read_bytes())
    checked = (spec.get("candidate_context") or {}).get(diagnostic.CONTEXT_FIELD)
    selected_diagnostic = diagnostic.active_checked_source_diagnostic()
    delegated = active_original_parent_pass_delegation()
    if delegated is not None and delegated.selection is not passes:
        raise StageGateError("parent qualification does not belong to the active original pass selection")
    if checked is not None:
        if (
            type(selected_diagnostic) is not diagnostic.CheckedSourceModelDiagnostic
            or selected_diagnostic.record() != checked
        ):
            raise StageGateError("checked model child requires the actual original host diagnostic selection")
        selected_diagnostic.verify_for(spec["capsule"], target=spec["target"])
    executor = _original_executor()
    if executor is None:
        if (
            library is not None
            or passes is not None
            or checked is not None
            or delegated is not None
            or shared_requirement is not None
        ):
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
        "schema": DELEGATED_PASSES_SCHEMA
        if delegated is not None
        else (
            CHECKED_SOURCE_SCHEMA
            if checked is not None
            else (PASSES_SCHEMA if passes is not None else LIBRARY_SCHEMA if library is not None else SCHEMA)
        ),
        **({"parent_pass_delegation": delegated.record()} if delegated is not None else {}),
        **({diagnostic.CONTEXT_FIELD: checked} if checked is not None else {}),
        **({"compiler_library": library.record()} if library is not None else {}),
        **({"compiler_passes": passes.record()} if passes is not None else {}),
        "shared_pass_requirement": shared_requirement,
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
            _closure(
                library=library is not None,
                passes=True,
                checked_source=checked is not None,
                source_optimizations="source_optimizations_v1" in passes.mechanisms,
                parent_delegation=delegated is not None,
            )
            if passes is not None
            else _closure(library=library is not None)
        ),
        "storage_completion_key": os.urandom(32).hex(),
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
    from merlin.development.backend.feedback.execution.compiler_profile_child import (
        _ADD_SEALS,
        _SEALS,
        DELEGATED_PASSES_SCHEMA,
        ModelChildTransport,
        _data,
        _encode_selection,
        _memfd,
    )

    fd = _memfd()
    readonly = None
    try:
        document = copy_record(document)
        if document["schema"] == DELEGATED_PASSES_SCHEMA:
            readonly = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
            document["parent_delegation_fd"] = readonly
        raw = _encode_selection(document)
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(raw)
        fcntl.fcntl(fd, _ADD_SEALS, _SEALS)
        if readonly is None:
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
    from merlin.common.storage import bundle_snapshots as BW
    from merlin.development.backend.feedback.execution.compiler_profile_child import _pin, _plain

    paths = tuple(_plain(path) for path in spec_paths)
    roots = tuple(path.parent / "compiler" for path in paths)
    if not paths or len(set(paths)) != len(paths) or len(set(roots)) != len(roots):
        raise StageGateError("original child batch requires distinct specifications and work roots")
    issued = BW._ACTIVE_VERIFICATION.get()
    executor = P.active_package_executor()
    if type(executor) is not _merlin_profile_executor_1._Executor or type(executor.owner) is not CommonTransformService:
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
    from merlin.development.backend.feedback.execution.compiler_profile_child import (
        _GET_SEALS,
        _RESTORED_CHILD,
        _RESTORED_ISSUED,
        _SEALS,
        CHECKED_SOURCE_SCHEMA,
        DELEGATED_PASSES_SCHEMA,
        LIBRARY_SCHEMA,
        MAX_BYTES,
        PASSES_SCHEMA,
        _decode_selection,
        _digest,
        _owner_record,
        _pin,
        _plain,
        _process,
        _receipt,
        _restore_inputs,
        _RestoredChild,
        _selected_closure,
        _source,
    )
    from merlin.development.backend.feedback.execution.snapshot_model_child import SCHEMA
    from merlin.development.targetgen.authoring.pass_selection import (
        active_shared_pass_requirement,
        scoped_shared_pass_requirement,
    )

    if (
        active_shared_pass_requirement() is not None
        or type(fd) is not int
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
    inherited_stat = os.fstat(fd)
    inherited_identity = (inherited_stat.st_dev, inherited_stat.st_ino)
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    document = _decode_selection(raw)
    if (
        document.get("schema")
        not in (SCHEMA, LIBRARY_SCHEMA, PASSES_SCHEMA, CHECKED_SOURCE_SCHEMA, DELEGATED_PASSES_SCHEMA)
        or (document.get("schema") == SCHEMA and "compiler_library" in document)
        or (document.get("schema") == LIBRARY_SCHEMA and "compiler_library" not in document)
        or (
            document.get("schema") not in (PASSES_SCHEMA, CHECKED_SOURCE_SCHEMA, DELEGATED_PASSES_SCHEMA)
            and "compiler_passes" in document
        )
        or (
            document.get("schema") in (PASSES_SCHEMA, CHECKED_SOURCE_SCHEMA, DELEGATED_PASSES_SCHEMA)
            and "compiler_passes" not in document
        )
        or (
            document.get("schema") not in (CHECKED_SOURCE_SCHEMA, DELEGATED_PASSES_SCHEMA)
            and "checked_source_diagnostic" in document
        )
        or (document.get("schema") == CHECKED_SOURCE_SCHEMA and "checked_source_diagnostic" not in document)
        or (
            (document.get("schema") == DELEGATED_PASSES_SCHEMA)
            != ("parent_pass_delegation" in document and "parent_delegation_fd" in document)
        )
        or (
            document.get("schema") != DELEGATED_PASSES_SCHEMA
            and ("parent_pass_delegation" in document or "parent_delegation_fd" in document)
        )
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
        # Restored only from the sealed, source-bound original parent document.
        # The ordinary compiler entrypoints enforce this same mandatory contract.
        execution.enter_context(scoped_shared_pass_requirement(document.get("shared_pass_requirement")))
        if document["schema"] == DELEGATED_PASSES_SCHEMA:
            from merlin.development.backend.feedback.execution.compiler_pass_delegation import (
                restored_original_parent_qualification,
            )
            from merlin.development.backend.feedback.execution.compiler_pass_parent_proof import (
                _verified_parent_scope,
            )

            proof = execution.enter_context(_verified_parent_scope(document, inherited_identity))
            execution.enter_context(restored_original_parent_qualification(proof))
        library = None
        if "compiler_library" in document:
            from merlin.development.backend.optimization.campaign.compiler_library_state import (
                restore_selection,
            )

            library = restore_selection(document["compiler_library"])
            execution.enter_context(scoped_compiler_library(library))
        passes = None
        if document["schema"] in (PASSES_SCHEMA, CHECKED_SOURCE_SCHEMA, DELEGATED_PASSES_SCHEMA):
            from merlin.development.targetgen.authoring.pass_selection import restore_original_selection

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
        from merlin.development.backend.feedback.certification.visible_model_gate import signing_scope

        execution.enter_context(signing_scope(document.get("storage_completion_key")))
        state = _RestoredChild(document, R.active_original_execution(), boundary)
        _RESTORED_ISSUED.add(state)
        token = _RESTORED_CHILD.set(state)
        try:
            if "checked_source_diagnostic" in document:
                from merlin.development.targetgen.execution.host import (
                    checked_source_diagnostics as diagnostic,
                )

                checked = diagnostic.restore_checked_source_diagnostic(
                    document[diagnostic.CONTEXT_FIELD],
                    capsule=spec["capsule"],
                    target=spec["target"],
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


def _verify_parent_delegation_inputs(document):
    """Reopen original source/spec bindings before and throughout redemption."""
    from merlin.development.backend.feedback.execution.compiler_pass_delegation import PROTOCOL
    from merlin.development.backend.feedback.execution.compiler_pass_delegation import (
        SCHEMA as DELEGATION_SCHEMA,
    )
    from merlin.development.backend.feedback.execution.compiler_profile_child import (
        DELEGATED_PASSES_SCHEMA,
        _pin,
        _selected_closure,
        _source,
    )

    delegated = document.get("parent_pass_delegation")
    if (
        document["schema"] != DELEGATED_PASSES_SCHEMA
        or type(delegated) is not dict
        or delegated.get("schema") != DELEGATION_SCHEMA
        or delegated.get("protocol") != PROTOCOL
        or delegated.get("selection") != document.get("compiler_passes")
        or delegated["selection"]["factory"]["inputs"].get("parent_delegation_protocol") != PROTOCOL
        or _pin(document["spec"]["path"]) != document["spec"]
        or invocation_record.environment_identity(os.environ) != document["environment"]
        or _selected_closure(document) != document["host_sources"]
        or _source(document["candidate"]) != document["candidate_sha256"]
        or _pin(Path(sys.executable).resolve()) != document["python"]
    ):
        raise StageGateError("original parent qualification changed its sealed source/spec selection")
    factory = delegated["selection"]["factory"]
    for pin in (factory["callback"], *factory["sources"], delegated["exporter"], delegated["redeemer"]):
        if _pin(pin["path"]) != {key: pin[key] for key in ("path", "sha256")}:
            raise StageGateError("original parent qualification implementation source changed")


def verified_parent_pass_delegation(proof):
    from merlin.development.backend.feedback.execution.compiler_pass_parent_proof import (
        verified_parent_pass_delegation as verify,
    )

    return verify(proof)


def _model_child_command(native_argv, *, environment):
    """Carry exact selected sources/environment without delaying the death guard."""
    from merlin.evaluation.capsule.models.runtime import _CHILD_GUARD
    from merlin.experiments import frozen_python as FP

    native = list(native_argv)
    command = FP.inherited_python_command(native)
    if command == native:
        return command
    if len(command) < 5 or command[-2] != "--execute" or "-c" not in native[1:]:
        raise StageGateError("model child frozen source transport has an unknown entry shape")
    context = json.loads(command[-1])["context"]
    verifier = FP._load(FP._checked(context["verifier"]), "_model_child_source_verifier", context["verifier"]["sha256"])
    root = Path(context["snapshot"])
    with cpu_pool.shared_compute_pool(timeout_s=120):
        receipt = verifier.verify(root)
        provider_fields = (
            set(verifier.provider_environment(root, receipt)) if hasattr(verifier, "provider_environment") else set()
        )
    if provider_fields - {
        "MERLIN_TARGET_PATH",
        "MERLIN_TARGET_CONTRACT",
        "MERLIN_RTL_FACTS",
        "MERLIN_TARGETS_DIR",
        "MERLIN_CONTRACT_DIR",
        "MERLIN_SCHEMAS_DIR",
    }:
        raise StageGateError("model child source bootstrap changed its public selector fields")
    # Bootstrap owns import guards, never the caller's already-sealed selectors.
    # Restore only fields the original bootstrap actually writes, including
    # absence. PATH/PYTHONPATH/LD paths, credentials and other grants stay intact.
    fields = provider_fields | {FP.CONTEXT, "MERLIN_REPO_ROOT", "PYTHONDONTWRITEBYTECODE"}
    selected = {key: environment.get(key) for key in sorted(fields)}
    if any(value is not None and type(value) is not str for value in selected.values()):
        raise StageGateError("model child selected public environment fields must be strings")
    index = native.index("-c") + 1
    native[index] = (
        "import os;_merlin_selected_environment=" + repr(selected) + ";\n"
        "for _key,_value in _merlin_selected_environment.items():\n"
        " if _value is None: os.environ.pop(_key,None)\n"
        " else: os.environ[_key]=_value\n" + native[index]
    )
    command = FP.inherited_python_command(native)
    # Preserve the original interpreter flags and exact bootstrap arguments.
    # There is no extra process: original sealed-FD parent identity stays exact.
    return [
        *command[:-3],
        "-c",
        _CHILD_GUARD + "import runpy;sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')",
        *command[-3:],
    ]


def run_model_grade(argv):
    """The original runner CLI child entry, inside the restored host scope."""
    import argparse

    from merlin.evaluation.capsule import runner

    sub = argparse.ArgumentParser(add_help=False)
    sub.add_argument("--model-grade", required=True)
    sub.add_argument("--model-grade-out", required=True)
    sub.add_argument("--original-profile-fd", type=int)
    sub.add_argument("--snapshot-external-fd", type=int)
    sa, _ = sub.parse_known_args(argv)
    capsule_processes._die_with_parent()
    spec_bytes = Path(sa.model_grade).read_bytes()
    spec = json.loads(spec_bytes)
    from contextlib import nullcontext

    scope = (
        nullcontext(None)
        if sa.original_profile_fd is None
        else restore_model_child(sa.original_profile_fd, spec_path=Path(sa.model_grade), spec_bytes=spec_bytes)
    )
    from merlin.development.backend.feedback.execution import (
        snapshot_evaluator as snapshot_execution,
    )
    from merlin.development.backend.feedback.execution import (
        snapshot_model_child,
    )

    if snapshot_execution.selected():
        if sa.snapshot_external_fd is None or sa.original_profile_fd is not None:
            raise StageGateError("external model grade requires its own sealed child transport")
        scope = snapshot_model_child.restore(
            sa.snapshot_external_fd, spec_path=Path(sa.model_grade), spec_bytes=spec_bytes
        )
    elif sa.snapshot_external_fd is not None:
        raise StageGateError("external child transport requires explicit external execution mode")
    with scope as child_receipt:
        if spec.get("candidate_context") is not None:
            res = capsule_model_candidates._grade_candidate_model_capsule_inline(
                spec["capsule"],
                target=spec["target"],
                timeout=int(spec["timeout"]),
                package_dir=spec["package_dir"],
                context=spec["candidate_context"],
            )
        else:
            res = capsule_model._grade_model_capsule_inline(
                spec["capsule"],
                target=spec.get("target"),
                timeout=int(spec["timeout"]),
                package_dir=spec.get("package_dir"),
                numeric_policy=spec.get("numeric_policy", capsule_identity._MODEL_NUMERIC_UNSET),
                numeric_policy_identity=spec.get("numeric_policy_identity"),
            )
        if child_receipt is not None:
            res[snapshot_model_child.FIELD if snapshot_execution.selected() else "original_compiler_child"] = (
                child_receipt
            )
            from merlin.development.backend.feedback.certification.visible_model_gate import export_return

            export_return(res)
    Path(sa.model_grade_out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0
