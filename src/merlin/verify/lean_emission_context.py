"""Live original-profile custody for ordinary source and object compilation.

Only the original host may enter this scope. Its finite records are shared by
copied worker contexts and expire on exit. Saved summaries cannot restore a
completion. Formal relationships never establish numerics or machine semantics.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import os
import threading
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record

from . import lean_emission_chain as E

_ACTIVE = contextvars.ContextVar("original_emission_checks", default=None)
_ISSUED = weakref.WeakSet()
_MAX_PENDING = 4096


@dataclass(frozen=True)
class _Pending:
    package: object
    source: Path
    compiler_input: Path
    artifact: Path
    call: object
    artifact_pin: dict
    manifest_sha256: str
    timeout_s: float


def _manifest_identity(package):
    return hashlib.sha256(json.dumps(package.manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()


class _Scope:
    def __init__(self, service, admission):
        self.service, self.admission = service, admission
        self.selection = E.EmissionChainSelection(service)
        self.records = {}
        self.claimed = set()
        self.lock = threading.RLock()
        self.live = True

    def verify(self):
        from merlin.targetgen.compiler_profile_admission import active_compiler_imports

        if self not in _ISSUED or not self.live or active_compiler_imports() is not self.admission:
            raise ValueError("ordinary emission scope has no current original host admission")
        admission, service = self.admission, self.service
        admission.verify()
        if any(
            getattr(admission, field) != getattr(service, field) for field in ("workspace", "bundle_manifest", "repo")
        ) or (admission.profile != service.profile or admission.snapshot_sha256 != service.snapshot_digest):
            raise ValueError("ordinary emission scope changed its original archive/profile/snapshot")

    def retain(self, pending):
        with self.lock:
            self.verify()
            if len(self.records) >= _MAX_PENDING:
                raise ValueError("ordinary emission scope exceeds its bounded live product inventory")
            self.records[id(pending)] = pending

    def matching(self, source):
        digest = hashlib.sha256(source.encode()).hexdigest()
        with self.lock:
            self.verify()
            matches = [pending for pending in self.records.values() if pending.artifact_pin["sha256"] == digest]
            if len(matches) > 1:
                # Equal artifact bytes cannot select which original source
                # obligation this consumer owns. Preserve all records and
                # refuse before either proof checking or native compilation.
                raise ValueError("ordinary object input has ambiguous original emission ownership")
            if matches:
                pending = matches[0]
                if id(pending) in self.claimed:
                    raise ValueError("ordinary emission has an incomplete or already claimed object consumer")
                self.claimed.add(id(pending))
                return pending
        return None

    def complete(self, pending):
        with self.lock:
            self.verify()
            if self.records.get(id(pending)) is not pending or id(pending) not in self.claimed:
                raise ValueError("ordinary emission completion lost its current claimed source")
            self.records.pop(id(pending))
            self.claimed.remove(id(pending))

    def has_pending(self, source):
        digest = hashlib.sha256(source.encode()).hexdigest()
        with self.lock:
            self.verify()
            return any(pending.artifact_pin["sha256"] == digest for pending in self.records.values())

    def close(self):
        with self.lock:
            self.live = False
            self.records.clear()
            self.claimed.clear()


@contextlib.contextmanager
def original_emission_checks(service):
    """Enter only inside the matching genuine original API import admission."""
    from merlin.targetgen.common_transforms import CommonTransformService
    from merlin.targetgen.compiler_profile_admission import active_compiler_imports
    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3

    if type(service) is not CommonTransformService:
        raise TypeError("ordinary emission checks require the exact original common service")
    admission = active_compiler_imports()
    if admission is None:
        raise ValueError("ordinary emission checks require live original API admission")
    if any(
        getattr(admission, field) != getattr(service, field) for field in ("workspace", "bundle_manifest", "repo")
    ) or (admission.profile != service.profile or admission.snapshot_sha256 != service.snapshot_digest):
        raise ValueError("ordinary emission checks selected a different original owner")
    if service.profile.get("common_substrate") != COMMON_SUBSTRATE_V3:
        yield
        return
    scope = _Scope(service, admission)
    _ISSUED.add(scope)
    token = _ACTIVE.set(scope)
    try:
        scope.verify()
        yield
        scope.verify()
    finally:
        scope.close()
        _ACTIVE.reset(token)
        _ISSUED.discard(scope)


def active_emission_checks():
    scope = _ACTIVE.get()
    if scope is not None:
        if type(scope) is not _Scope:
            raise ValueError("ordinary emission checks have no original issuer")
        scope.verify()
    return scope


def requires_fresh_object(source):
    """Only a pending actual trace disables artifact-only legacy cache reuse."""
    scope = active_emission_checks()
    return scope is not None and scope.has_pending(source)


def explicit_emission_transport(function):
    """Existing explicit original chain owns its check, without double issuance."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def selected(*args, **kwargs):
        scope = active_emission_checks()
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        selection = bound.arguments["emission_verification"]
        if scope is None or selection is None:
            return function(*args, **kwargs)
        if (
            type(selection) is not E.EmissionChainSelection
            or selection.service is not scope.service
            or (selection.original_selection != scope.selection.original_selection)
        ):
            raise ValueError("explicit emission transport differs from the live original selection")
        # This adapter is applied only to the host-owned existing compile-only
        # transport. Author requests have no observation or coverage flag.
        token = _ACTIVE.set(None)
        try:
            output = function(*args, **kwargs)
            scope.verify()
            return output
        finally:
            _ACTIVE.reset(token)

    return selected


def _remaining(deadline):
    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("ordinary source/proof transport exhausted its aggregate budget")
        return left

    return remaining


def _package_sources(package):
    """Observe bounded author sources; builds remain separately observed products."""
    root = Path(package.directory)
    sources, size, entries = [], 0, 0
    for directory, directories, files in os.walk(root, followlinks=False):
        entries += len(directories) + len(files)
        if entries > 32768:
            raise ValueError("ordinary compiler source observation exceeds its entry budget")
        directories[:] = sorted(name for name in directories if name not in {".git", "build", "__pycache__"})
        if any((Path(directory) / name).is_symlink() for name in directories):
            raise ValueError("ordinary compiler source ownership contains a directory alias")
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix == ".pyc":
                continue
            size += path.stat().st_size
            if size > 32 << 20 or len(sources) >= 8192:
                raise ValueError("ordinary compiler source observation exceeds its bounded inventory")
            E._pin(path)
            sources.append(path)
    return tuple(sources)


def _save(root, report):
    path = root / "emission_proof_chain.json"
    path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def observe_emission(function):
    """Observe the actual ordinary lower-interface call, without saved replay."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def observed(*args, **kwargs):
        scope = active_emission_checks()
        if scope is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        package, source, generated = (
            values["pkg"],
            Path(values["interface"]).absolute(),
            Path(values["generated"]).absolute(),
        )
        generated.mkdir(parents=True, exist_ok=True)
        evidence = generated / ("emission-custody-" + uuid.uuid4().hex)
        evidence.mkdir(mode=0o700)
        declared = "emit_transformation_trace" in (package.manifest.get("commands") or {})
        # A memoized reply cannot issue a fresh original trace completion.
        # Untraced compilers retain their original reuse, with no formal credit.
        if declared:
            values["memo"] = None
        artifact = generated / values["artifact_name"]
        products = (generated / "input.interface.mlir", generated / "command_buffer.json", artifact)
        with invocation_record.observe_call(
            evidence,
            stage="ordinary_source_emission",
            function=observed,
            arguments={"contract": str(values["contract"])},
            inputs=(source,),
            outputs=products,
            dependencies=(*_package_sources(package), Path(inspect.getsourcefile(function))),
        ) as call:
            output = function(*bound.args, **bound.kwargs)
            call.returned(stdout=output[1])
        scope.verify()
        pin = E._pin(artifact)
        if pin["sha256"] != hashlib.sha256(output[1].encode()).hexdigest():
            raise ValueError("ordinary emitted product differs from its actual caller result")
        _save(
            evidence,
            {
                "schema": "merlin.emission_pending.v1",
                "status": "pending_consumer" if declared else "not_declared",
                "source": E._pin(source),
                "emitted_artifact": pin,
                "compiler_source_invocation": E._pin(call.path),
                "original_profile_sha256": scope.selection.original_selection[2],
                "numerics": "not_established",
                "machine_semantics": "not_established",
                "formal_relationship": "not_established",
            },
        )
        if declared:
            # The original ABI's actual dispatch entry is selected only at
            # object production. Standalone emission cannot assert that entry
            # from a candidate-authored trace or a guessed helper function.
            scope.retain(
                _Pending(
                    package, source, products[0], artifact, call, pin, _manifest_identity(package), values["timeout"]
                )
            )
        return output

    return observed


def observe_object(function):
    """Bind a live matching emission to the actual ordinary object consumer."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def observed(*args, **kwargs):
        scope = active_emission_checks()
        if scope is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        source, work = values["lowered_mlir_text"], Path(values["workdir"])
        work.mkdir(parents=True, exist_ok=True)
        evidence = work / ("emission-consumer-" + uuid.uuid4().hex)
        evidence.mkdir(mode=0o700)
        pending = scope.matching(source)
        result = None
        report = {
            "schema": E.RESULT_SCHEMA,
            "status": "not_matched",
            "gap": "no current original emitted chain matches this complete consumer input",
            "numerics": "not_established",
            "machine_semantics": "not_established",
        }
        try:
            inputs = ()
            if pending is not None:
                from merlin.targetgen.host_optimization_admission import _original_entry

                entry, entry_source = _original_entry(scope.admission, values["target"], values["_build_service"])
                if _manifest_identity(pending.package) != pending.manifest_sha256:
                    raise ValueError("ordinary emission changed its original package declaration")
                invocation_record.verify(pending.call.path)
                if E._pin(pending.artifact) != pending.artifact_pin:
                    raise ValueError("ordinary emitted product changed before its actual consumer")
                from merlin.common.execution_deadline import selected_deadline

                parent = selected_deadline(
                    seconds=values["build_timeout_s"],
                    parent=values["execution_deadline"],
                    build_service=values["_build_service"],
                )
                if parent is not None:
                    # The ordinary producer consumes the same aggregate build
                    # deadline, including proof work, without resetting it.
                    values["build_timeout_s"], values["execution_deadline"] = None, parent
                remaining = (
                    parent.remaining
                    if parent is not None
                    else _remaining(time.monotonic() + (values["build_timeout_s"] or pending.timeout_s))
                )
                result = E.check_emission_chain(
                    selection=scope.selection,
                    package=pending.package,
                    original_source=pending.source,
                    compiler_input=pending.compiler_input,
                    emitted_artifact=pending.artifact,
                    emission_invocation=pending.call,
                    target_entry=entry,
                    evidence_root=evidence / "proof",
                    remaining=remaining,
                )
                report = result.record()
                if E.reread_emitted_product(result, selection=scope.selection) != source:
                    raise ValueError("actual ordinary object input changed its live emitted product")
                inputs = (pending.artifact,)
            with invocation_record.observe_call(
                evidence,
                stage="ordinary_emission_object_consumer",
                function=observed,
                arguments={"actual_input_sha256": hashlib.sha256(source.encode()).hexdigest()},
                inputs=inputs,
                outputs=(work / "kernel.llvm.mlir", work / "kernel.o"),
                dependencies=(Path(inspect.getsourcefile(function)),),
            ) as call:
                obj = function(*bound.args, **bound.kwargs)
                call.returned(stdout=str(obj))
            scope.verify()
            if result is not None:
                report = E.consume_emission_chain(
                    result,
                    selection=scope.selection,
                    object_invocation=call,
                    object_path=obj,
                    consumer_source=work / "kernel.llvm.mlir",
                )
                if _original_entry(scope.admission, values["target"], values["_build_service"]) != (
                    entry,
                    entry_source,
                ):
                    raise ValueError("ordinary object consumer changed its original ABI entry")
                report["original_dispatch_entry"] = entry
                report["source_relationship_scope"] = "one unambiguous current observed source"
                # Generic host transformations and terminal translation keep
                # their own provenance; this source relationship proves neither.
                if (work / "kernel.host.mlir").is_file():
                    report["selected_host_source"] = E._pin(work / "kernel.host.mlir")
                    report["subsequent_host_transformations"] = "outside_this_emission_chain"
            _save(evidence, report)
            if pending is not None:
                scope.complete(pending)
            return obj
        finally:
            if result is not None:
                E.discard_emission_chain(result)

    return observed
