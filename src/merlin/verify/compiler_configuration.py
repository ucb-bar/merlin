"""Fresh compiler gate for actual current configuration instruction buffers.

The complete-entry theorem preserves error and terminal state under explicit
selected effect/ISA premises. This API neither installs an instruction model nor
turns a caller's qualification label into independent implementation approval.
"""

from __future__ import annotations

import json
import os
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

from . import configuration_elision_runner as runner
from .compiler_configuration_intake import (
    DOMAIN,
    RELATION,
    CurrentCandidate,
    prepare,
    proof_sources,
    reopen,
    request,
    revalidate,
)
from .configuration_elision_bridge import canonical, digest, file_ref
from .configuration_elision_emit import emit
from .configuration_elision_gate import NotEstablished, Unavailable, _exact
from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

SELECTOR = "configuration_write_elision"
CHECK = "compiler_configuration_refinement"
_LIVE = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class Completion:
    identity: str
    report: str

    def __reduce_ex__(self, protocol):
        raise TypeError("live compiler configuration completion cannot be serialized")

    def __copy__(self):
        return Completion(self.identity, self.report)

    def __deepcopy__(self, memo):
        return self.__copy__()


def ready(candidate):
    if type(candidate) is not CurrentCandidate or candidate.domain != DOMAIN or candidate.relation != RELATION:
        raise Unavailable("exact current compiler instruction configuration candidate required")
    generated = emit(json.loads(candidate.projection_json))
    if candidate.query != generated["query"] or list(candidate.roots) != generated["roots"]:
        raise Unavailable("canonical complete configuration query/root roster differs")
    if json.loads(candidate.proof_sources_json) != proof_sources():
        raise Unavailable("canonical configuration proof closure changed")
    try:
        revalidate(candidate)
    except (ValueError, OSError) as error:
        raise Unavailable("compiler configuration inputs or proof sources changed") from error


def check_candidate(candidate, *, lean, destination, timeout=120):
    ready(candidate)
    report, _ = runner.prove(candidate, lean=lean, destination=destination, timeout=timeout, validate=ready)
    report.update(schema="merlin.compiler-configuration.checked.v1", domain=DOMAIN)
    path = Path(destination).resolve() / "REPORT.json"
    path.write_text(canonical(report))
    completion = Completion(candidate.identity, str(path))
    _LIVE[completion] = (os.getpid(), threading.get_ident(), candidate, file_ref(path), report, ())
    validate_completion(completion, candidate)
    return completion


def validate_completion(completion, candidate):
    if type(completion) is not Completion or completion not in _LIVE:
        raise Unavailable("saved/constructed/copied configuration completion has no authority")
    pid, thread, original, report_ref, report, origins = _LIVE[completion]
    if (pid, thread) != (os.getpid(), threading.get_ident()) or original is not candidate:
        raise Unavailable("compiler configuration completion belongs to a different live invocation")
    if completion.identity != candidate.identity or file_ref(completion.report) != report_ref:
        raise Unavailable("compiler configuration report/candidate identity changed")
    ready(candidate)
    if origins:
        _read_bound_buffers(origins)
    library = Path(report["library_root"])
    if any(path.is_symlink() for path in library.rglob("*")):
        raise Unavailable("private configuration proof objects may not be symlinks")
    if {str(path.resolve()) for path in library.rglob("*") if path.is_file()} != {
        row["path"] for row in report["library"]
    }:
        raise Unavailable("private configuration proof object tree membership changed")
    for row in [
        *report["buffers"],
        *report["mapping_records"],
        report["query"],
        report["object"],
        report["tool"],
        report["auditor"],
        *report["imports"],
        *report["library"],
        *report["process_evidence"],
    ]:
        _exact(row)
    return json.loads(canonical(report))


def deliver(completion, candidate):
    validate_completion(completion, candidate)
    return candidate.final


def _read_bound_buffers(rows):
    """Read exactly the saved endpoint bytes; never normalize their newlines."""
    if type(rows) not in (list, tuple) or len(rows) != 2:
        raise Unavailable("exact two bound configuration replay buffers required")
    result = []
    try:
        for row in rows:
            if type(row) is not dict or set(row) != {"path", "sha256", "bytes"}:
                raise Unavailable("exact configuration replay file reference required")
            _exact(row)
            raw = Path(row["path"]).read_bytes()
            if digest(raw) != row["sha256"] or len(raw) != row["bytes"]:
                raise Unavailable("configuration replay bytes differ from bound original")
            result.append(raw.decode("utf-8"))
            _exact(row)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as error:
        raise Unavailable("configuration replay original bytes unavailable or changed") from error
    return tuple(result)


def replay(report_path, *, lean, destination, timeout=120):
    saved = json.loads(Path(report_path).read_bytes().decode("utf-8"))
    if saved.get("schema") != "merlin.compiler-configuration.checked.v1" or len(saved.get("buffers", [])) != 2:
        raise Unavailable("current complete compiler configuration replay record required")
    origins = saved["buffers"]
    raw = _read_bound_buffers(origins)
    selection = reopen(saved["source_request"])
    candidate = prepare(
        *raw,
        selection=selection,
        representation=saved["candidate"]["representation"],
    )
    if (candidate.source, candidate.final) != raw:
        raise Unavailable("configuration replay preparation changed original bytes")
    # Replay intentionally uses current canonical proof/host sources. A saved
    # report locates the source pair, never approves the reconstructed result.
    completion = check_candidate(candidate, lean=lean, destination=destination, timeout=timeout)
    if _read_bound_buffers(origins) != raw:
        raise Unavailable("configuration replay originals changed during proof")
    record = _LIVE.get(completion)
    if record is None:
        raise Unavailable("configuration replay did not obtain an owned live completion")
    _LIVE[completion] = (*record[:5], tuple(json.loads(canonical(origins))))
    validate_completion(completion, candidate)
    return completion, candidate


def job_params(selection, *, sdk_bin, representation="instruction_mlir"):
    return {
        "schema": "merlin.compiler-configuration.job.v1",
        "sdk_bin": str(sdk_bin),
        "selection": request(selection),
        "representation": representation,
    }


def check(job):
    params = job.get("params")
    if (
        type(params) is not dict
        or set(params) != {"schema", "sdk_bin", "selection", "representation"}
        or params["schema"] != "merlin.compiler-configuration.job.v1"
    ):
        raise GateUnsupported("exact original compiler configuration selection and tool required")
    try:
        for key in ("facts", "software"):
            _exact(params["selection"][key])
        selection = reopen(params["selection"])
        originals = [file_ref(job[key]) for key in ("before", "after")]
        candidate = prepare(
            Path(job["before"]).read_text(),
            Path(job["after"]).read_text(),
            selection=selection,
            representation=params["representation"],
        )
        for row, text in zip(originals, (candidate.source, candidate.final), strict=True):
            if row["sha256"] != digest(text):
                raise Unavailable("actual compiler configuration boundary changed while reading")
        completion = check_candidate(
            candidate, lean=Path(params["sdk_bin"]) / "lean", destination=Path(job["directory"]) / "configuration-proof"
        )
        report = validate_completion(completion, candidate)
        deliver(completion, candidate)
        for row in originals:
            _exact(row)
    except NotEstablished as error:
        raise GateRejected(str(error)) from error
    except (Unavailable, OSError) as error:
        raise GateUnavailable(str(error)) from error
    except (ValueError, TypeError, KeyError) as error:
        raise GateUnsupported(str(error)) from error
    return {
        "schema": "merlin.compiler-configuration.gate.v1",
        "status": "checked",
        "claim": RELATION,
        "domain": DOMAIN,
        "representation": candidate.representation,
        "relation": RELATION,
        "receipt": completion.report,
        "theorems": list(candidate.roots),
        "axioms": report["audit"]["axioms_used"],
        "premises": candidate.describe()["premises"],
        "entries": candidate.describe()["entries"],
        "hardware_model_approved": False,
        "independently_qualified": False,
        "native_fidelity": False,
        "compiler_enforcement": True,
        "before": originals[0],
        "after": originals[1],
    }


def observe(selection, original, result):
    """Strict admitted boundary; source exclusion belongs to the original wrapper."""
    from merlin.transforms.certificates import ir_text

    from . import pass_validation
    from .compiler_configuration_source_profile import ELISION_ID
    from .compiler_coverage import boundary

    if not pass_validation.selected(SELECTOR):
        return result
    result.revalidate()
    before, after = ir_text(original), ir_text(result.module)
    if before == after:
        boundary(SELECTOR, before, after, profile_id=ELISION_ID)
        return result
    validation = pass_validation.current()
    pass_validation.require_canonical_handler(validation, CHECK)
    job = validation.gate.submit(
        CHECK,
        before=before,
        after=after,
        label=SELECTOR,
        params=job_params(selection, sdk_bin=validation.tools.sdk_bin),
    )
    boundary(SELECTOR, before, after, profile_id=ELISION_ID, job=job)
    return result


def validated_apply(selection, module, validation):
    """Run the real original producer and hold its result through the owned gate."""
    from merlin.compile.scheduling.configuration_ir import _eliminate
    from merlin.transforms.certificates import ir_text

    from . import pass_validation

    if type(validation) is not pass_validation.PassValidation or SELECTOR not in validation.passes:
        raise ValueError("configuration proof requires its original explicit PassValidation selection")
    pass_validation.require_canonical_handler(validation, CHECK)

    def produce():
        from .compiler_configuration_source_profile import ELISION_ID, admit_elision_source
        from .compiler_coverage import boundary
        from .compiler_profile import SourceOutsideProfile

        original = module.clone()
        before = ir_text(original)
        exclusion = None
        try:
            admit_elision_source(before, selection=selection, representation="instruction_mlir")
        except SourceOutsideProfile as error:
            if error.profile_id != ELISION_ID:
                raise ValueError("configuration source exclusion belongs to a different profile") from error
            exclusion = error
        result = _eliminate(module, selection)
        if ir_text(result.source) != before or ir_text(original) != before:
            raise ValueError("configuration producer source differs from original compiler input")
        result.revalidate()
        if exclusion is None:
            observe(selection, original, result)
        else:
            boundary(SELECTOR, before, ir_text(result.module), profile_id=ELISION_ID, exclusion=exclusion)
        return result

    active = pass_validation.current()
    if active is not None:
        if active is not validation:
            raise ValueError("configuration validation differs from active original compiler owner")
        result = produce()
        validation.gate.join()
        result.revalidate()
        return result
    if validation.gate.guarding:
        raise ValueError("configuration proof must execute on its original active compiler owner thread")
    with pass_validation.validating(validation):
        result = produce()
    return result


def validated_kernel_result(
    selected_contracts, qualification_identity, original, result, validation, *, max_calls=4096, enabled=True
):
    """Hold the actual schedule producer result before its ordinary emitter consumes it."""
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection

    from . import pass_validation

    if type(validation) is not pass_validation.PassValidation or SELECTOR not in validation.passes:
        raise ValueError("Kernel configuration proof requires original explicit PassValidation")
    pass_validation.require_canonical_handler(validation, CHECK)
    active = pass_validation.current()
    if active is not None and active is not validation:
        raise ValueError("Kernel configuration validation differs from active original owner")
    if active is None and validation.gate.guarding:
        raise ValueError("configuration Kernel must execute on the original active owner thread")
    selection = ConfigurationIRSelection(
        selected_contracts, qualification_identity, max_operations=max_calls, enabled=enabled
    )
    before, after = original.text(), result.kernel.text()

    def submit():
        from .compiler_configuration_source_profile import ELISION_ID, admit_elision_source
        from .compiler_coverage import boundary
        from .compiler_profile import SourceOutsideProfile

        exclusion = None
        try:
            admit_elision_source(before, selection=selection, representation="schedule_kernel")
        except SourceOutsideProfile as error:
            if error.profile_id != ELISION_ID:
                raise ValueError("configuration source exclusion belongs to a different profile") from error
            exclusion = error
        if exclusion is not None:
            boundary(SELECTOR, before, after, profile_id=ELISION_ID, exclusion=exclusion)
        elif before == after:
            boundary(SELECTOR, before, after, profile_id=ELISION_ID)
        else:
            job = validation.gate.submit(
                CHECK,
                before=before,
                after=after,
                label=SELECTOR,
                params=job_params(selection, sdk_bin=validation.tools.sdk_bin, representation="schedule_kernel"),
            )
            boundary(SELECTOR, before, after, profile_id=ELISION_ID, job=job)
            validation.gate.join()
        selection.revalidate()
        if original.text() != before or result.kernel.text() != after:
            raise ValueError("actual complete configuration Kernel changed before delivery")

    if active is not None:
        submit()
    else:
        with pass_validation.validating(validation):
            submit()
    return result
