"""Fresh current compiler gate for scoped complete Kernel configuration hoisting.

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
from .compiler_configuration_hoist_intake import DOMAIN as DOMAIN
from .compiler_configuration_hoist_intake import (
    DOMAINS,
    RELATION,
    CurrentHoistCandidate,
    emit_profile,
    prepare,
    proof_sources,
    reopen,
    request,
    revalidate,
)
from .configuration_elision_bridge import canonical, digest, file_ref
from .configuration_elision_gate import NotEstablished, Unavailable, _exact
from .configuration_hoist_emit import MODULE
from .gate_errors import GateRejected, GateUnavailable, GateUnsupported

SELECTOR = "configuration_loop_hoisting"
CHECK = "compiler_configuration_hoist_refinement"
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
    if (
        type(candidate) is not CurrentHoistCandidate
        or type(candidate.domain) is not str
        or candidate.domain not in DOMAINS
        or candidate.relation != RELATION
    ):
        raise Unavailable("exact current compiler instruction configuration candidate required")
    generated = emit_profile(json.loads(candidate.projection_json), candidate.domain)
    if candidate.query != generated["query"] or list(candidate.roots) != generated["roots"]:
        raise Unavailable("canonical complete configuration query/root roster differs")
    if json.loads(candidate.proof_sources_json) != proof_sources(candidate.domain):
        raise Unavailable("canonical configuration proof closure changed")
    try:
        revalidate(candidate)
    except (ValueError, OSError) as error:
        raise Unavailable("compiler configuration inputs or proof sources changed") from error


def check_candidate(candidate, *, lean, destination, timeout=120):
    ready(candidate)
    report, _ = runner.prove(
        candidate,
        lean=lean,
        destination=destination,
        timeout=timeout,
        validate=ready,
        query_module=MODULE,
        source_plan=lambda: proof_sources(candidate.domain),
        relation=RELATION,
    )
    report.update(schema="merlin.compiler-configuration-hoist.checked.v1", domain=candidate.domain)
    path = Path(destination).resolve() / "REPORT.json"
    path.write_text(canonical(report))
    completion = Completion(candidate.identity, str(path))
    _LIVE[completion] = (os.getpid(), threading.get_ident(), candidate, file_ref(path), report)
    validate_completion(completion, candidate)
    return completion


def validate_completion(completion, candidate):
    if type(completion) is not Completion or completion not in _LIVE:
        raise Unavailable("saved/constructed/copied configuration completion has no authority")
    pid, thread, original, report_ref, report = _LIVE[completion]
    if (pid, thread) != (os.getpid(), threading.get_ident()) or original is not candidate:
        raise Unavailable("compiler configuration completion belongs to a different live invocation")
    if completion.identity != candidate.identity or file_ref(completion.report) != report_ref:
        raise Unavailable("compiler configuration report/candidate identity changed")
    ready(candidate)
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


def replay(report_path, *, lean, destination, timeout=120):
    saved = json.loads(Path(report_path).read_text())
    if saved.get("schema") != "merlin.compiler-configuration-hoist.checked.v1" or len(saved.get("buffers", [])) != 2:
        raise Unavailable("current complete compiler configuration replay record required")
    for row in saved["buffers"]:
        _exact(row)
    selection = reopen(saved["source_request"])
    candidate = prepare(
        *(Path(row["path"]).read_text() for row in saved["buffers"]),
        selection=selection,
    )
    # Replay intentionally uses current canonical proof/host sources. A saved
    # report locates the source pair, never approves the reconstructed result.
    completion = check_candidate(candidate, lean=lean, destination=destination, timeout=timeout)
    return completion, candidate


def job_params(selection, *, sdk_bin):
    return {
        "schema": "merlin.compiler-configuration-hoist.job.v1",
        "sdk_bin": str(sdk_bin),
        "selection": request(selection),
    }


def check(job):
    params = job.get("params")
    if (
        type(params) is not dict
        or set(params) != {"schema", "sdk_bin", "selection"}
        or params["schema"] != "merlin.compiler-configuration-hoist.job.v1"
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
        "schema": "merlin.compiler-configuration-hoist.gate.v1",
        "status": "checked",
        "claim": RELATION,
        "domain": candidate.domain,
        "representation": "schedule_kernel",
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


def validated_hoist(
    kernel,
    adapter,
    *,
    selected_contracts,
    validation,
    enabled=True,
    max_calls=4096,
    max_fields=256,
    max_depth=64,
):
    """Observe actual original input before the producer and hold its result."""
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.compile.scheduling.configuration_loops import _hoist_loop_configuration

    from . import pass_validation

    if type(validation) is not pass_validation.PassValidation or SELECTOR not in validation.passes:
        raise ValueError("hoisting proof requires original explicit PassValidation")
    pass_validation.require_canonical_handler(validation, CHECK)
    active = pass_validation.current()
    if active is not None and active is not validation:
        raise ValueError("hoisting validation differs from active original compiler owner")
    if active is None and validation.gate.guarding:
        raise ValueError("hoisting proof must execute on original active owner thread")
    selection = ConfigurationIRSelection(
        selected_contracts,
        adapter.qualification_identity,
        enabled=enabled,
        max_operations=max_calls,
        max_fields=max_fields,
    )
    if adapter.configuration_identity != selection.selected_contracts.hardware_identity:
        raise ValueError("hoisting adapter differs from original selected hardware inputs")
    before = kernel.text()

    def produce():
        result = _hoist_loop_configuration(
            kernel,
            adapter,
            enabled=enabled,
            max_calls=max_calls,
            max_fields=max_fields,
            max_depth=max_depth,
        )
        after = result.kernel.text()
        if kernel.text() != before:
            raise ValueError("hoisting producer changed original compiler input")
        if before != after:
            validation.gate.submit(
                CHECK,
                before=before,
                after=after,
                label=SELECTOR,
                params=job_params(selection, sdk_bin=validation.tools.sdk_bin),
            )
            validation.gate.join()
        selection.revalidate()
        if kernel.text() != before or result.kernel.text() != after:
            raise ValueError("hoisting source/result changed before actual compiler delivery")
        return result

    if active is not None:
        return produce()
    with pass_validation.validating(validation):
        result = produce()
    return result
