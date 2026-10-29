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
from .compiler_configuration import _read_bound_buffers
from .compiler_configuration_hoist_intake import DOMAIN as DOMAIN
from .compiler_configuration_hoist_intake import (
    DOMAINS,
    RELATION,
    CurrentHoistCandidate,
    emit_profile,
    prepare,
    proof_sources,
    reopen_request,
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
    report.update(
        schema="merlin.compiler-configuration-hoist.checked.v2"
        if json.loads(candidate.request_json).get("schema") == "merlin.compiler-configuration-hoist.request.v2"
        else "merlin.compiler-configuration-hoist.checked.v1",
        domain=candidate.domain,
    )
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


def replay(report_path, *, lean, destination, timeout=120):
    saved = json.loads(Path(report_path).read_bytes().decode("utf-8"))
    if (
        saved.get("schema")
        not in {"merlin.compiler-configuration-hoist.checked.v1", "merlin.compiler-configuration-hoist.checked.v2"}
        or len(saved.get("buffers", [])) != 2
    ):
        raise Unavailable("current complete compiler configuration replay record required")
    origins = saved["buffers"]
    raw = _read_bound_buffers(origins)
    current_request = saved["source_request"]
    bound_consumer = current_request.get("schema") == "merlin.compiler-configuration-hoist.request.v2"
    if bound_consumer != (saved["schema"] == "merlin.compiler-configuration-hoist.checked.v2"):
        raise Unavailable("checked hoist report/request schema or C-consumer binding differs")
    selection, c_consumer = reopen_request(current_request)
    candidate = prepare(
        *raw,
        selection=selection,
        c_consumer=c_consumer,
    )
    if (candidate.source, candidate.final) != raw:
        raise Unavailable("configuration hoist replay preparation changed original bytes")
    # Replay intentionally uses current canonical proof/host sources. A saved
    # report locates the source pair, never approves the reconstructed result.
    completion = check_candidate(candidate, lean=lean, destination=destination, timeout=timeout)
    if _read_bound_buffers(origins) != raw:
        raise Unavailable("configuration hoist replay originals changed during proof")
    record = _LIVE.get(completion)
    if record is None:
        raise Unavailable("configuration hoist replay did not obtain an owned live completion")
    _LIVE[completion] = (*record[:5], tuple(json.loads(canonical(origins))))
    validate_completion(completion, candidate)
    return completion, candidate


def job_params(selection, *, sdk_bin, c_consumer=None):
    if c_consumer is not None:
        from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer

        if type(c_consumer) is not ConfigurationCConsumer:
            raise ValueError("original typed C consumer required")
        return {
            "schema": "merlin.compiler-configuration-hoist.job.v2",
            "sdk_bin": str(sdk_bin),
            "selection": request(selection),
            "c_consumer": c_consumer.record(),
        }
    return {
        "schema": "merlin.compiler-configuration-hoist.job.v1",
        "sdk_bin": str(sdk_bin),
        "selection": request(selection),
    }


def check(job):
    params = job.get("params")
    legacy = type(params) is dict and params.get("schema") == "merlin.compiler-configuration-hoist.job.v1"
    expected = {"schema", "sdk_bin", "selection"} if legacy else {"schema", "sdk_bin", "selection", "c_consumer"}
    if (
        type(params) is not dict
        or set(params) != expected
        or (not legacy and params.get("schema") != "merlin.compiler-configuration-hoist.job.v2")
    ):
        raise GateUnsupported("exact original compiler configuration/C-consumer selection and tool required")
    try:
        for key in ("facts", "software"):
            _exact(params["selection"][key])
        selection, c_consumer = reopen_request(
            params["selection"]
            if legacy
            else {
                "schema": "merlin.compiler-configuration-hoist.request.v2",
                "selection": params["selection"],
                "c_consumer": params["c_consumer"],
            }
        )
        originals = [file_ref(job[key]) for key in ("before", "after")]
        candidate = prepare(
            Path(job["before"]).read_text(),
            Path(job["after"]).read_text(),
            selection=selection,
            c_consumer=c_consumer,
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
        "schema": "merlin.compiler-configuration-hoist.gate.v1"
        if legacy
        else "merlin.compiler-configuration-hoist.gate.v2",
        **({"c_consumer": c_consumer.record()} if not legacy else {}),
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


def _joined_c_delivery(validation, job, candidate, c_consumer, params):
    """Bind one actual live gate join to this owner's frozen endpoints/domain.

    This is transport verification after the original canonical gate ran, not
    acceptance derived from a saved report or child consumer object.
    """
    from .speculative_gate import Job

    if type(job) is not Job or job.check != CHECK or job.params != params:
        raise ValueError("original C-bound gate job differs from its frozen request")
    rows = [row for row in validation.gate.verdicts if row.get("job") == job.id]
    if len(rows) != 1 or rows[0].get("verdict") != "accepted":
        raise ValueError("original C-bound job has no unique actual accepted delivery")
    detail = rows[0].get("detail", {})
    if (
        detail.get("schema") != "merlin.compiler-configuration-hoist.gate.v2"
        or detail.get("c_consumer") != c_consumer.record()
        or detail.get("domain") != candidate.domain
        or detail.get("relation") != candidate.relation
    ):
        raise ValueError("actual accepted gate delivery differs from original C consumer/candidate")
    request_path = job.directory / "request.json"
    frozen = json.loads(request_path.read_text())
    expected = {
        "check": CHECK,
        "label": SELECTOR,
        "params": params,
        "directory": str(job.directory),
        "before": str(job.directory / "before.mlir"),
        "after": str(job.directory / "after.mlir"),
    }
    if frozen != expected:
        raise ValueError("actual C-bound gate request changed before delivery")
    for key, text in (("before", candidate.source), ("after", candidate.final)):
        if Path(frozen[key]).read_text() != text or detail.get(key) != file_ref(frozen[key]):
            raise ValueError("actual accepted gate source/output differs from original candidate")
    report_path = Path(detail["receipt"])
    report = json.loads(report_path.read_text())
    if (
        report.get("schema") != "merlin.compiler-configuration-hoist.checked.v2"
        or report.get("source_identity") != candidate.identity
        or report.get("candidate") != candidate.describe()
        or report.get("source_request") != json.loads(candidate.request_json)
        or report.get("sources") != json.loads(candidate.sources_json)
        or [row.get("sha256") for row in report.get("buffers", [])]
        != [digest(candidate.source), digest(candidate.final)]
    ):
        raise ValueError("actual accepted report differs from original C-bound proof inputs")
    refs = [file_ref(path) for path in (request_path, frozen["before"], frozen["after"], report_path)]
    # All child products were checked by the live canonical join. Preserve that
    # exact product/source/output binding until this owner's controlled emission.
    for key in ("buffers", "mapping_records", "imports", "library", "process_evidence"):
        refs.extend(report[key])
    refs.extend(report[key] for key in ("query", "object", "tool", "auditor"))
    for row in refs:
        _exact(row)
    return tuple({row["path"]: row for row in refs}.values())


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
    c_consumer=None,
):
    """Observe actual original input before the producer and hold its result."""
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.compile.scheduling.configuration_loops import _hoist_loop_configuration

    from . import pass_validation

    if type(validation) is not pass_validation.PassValidation or SELECTOR not in validation.passes:
        raise ValueError("hoisting proof requires original explicit PassValidation")
    pass_validation.require_canonical_handler(validation, CHECK)
    if c_consumer is not None:
        if type(validation.index_bits) is not int or validation.index_bits != c_consumer.index_bits:
            raise ValueError("original C consumer differs from the selected proof index domain")
        c_consumer.revalidate()
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
        from .compiler_configuration_source_profile import HOIST_ID, admit_hoist_source
        from .compiler_coverage import boundary
        from .compiler_profile import SourceOutsideProfile

        exclusion = None
        try:
            admit_hoist_source(kernel, selection=selection, c_consumer=c_consumer)
        except SourceOutsideProfile as error:
            if error.profile_id != HOIST_ID:
                raise ValueError("hoisting source exclusion belongs to a different profile") from error
            exclusion = error
        if kernel.text() != before:
            raise ValueError("hoisting source changed during source-profile admission")
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
        proof_inputs, transport_refs = (), ()
        if exclusion is not None:
            # Ordinary changed output remains explicitly UNVERIFIED. Empty proof
            # inputs confer no theorem claim; original C-domain/owner custody
            # is independently retained by ConfigurationCConsumer._issue.
            boundary(SELECTOR, before, after, profile_id=HOIST_ID, exclusion=exclusion)
        elif before != after:
            if c_consumer is not None:
                bound = prepare(before, after, selection=selection, c_consumer=c_consumer)
                proof_inputs = tuple(json.loads(bound.sources_json))
            original_params = job_params(selection, sdk_bin=validation.tools.sdk_bin, c_consumer=c_consumer)
            job = validation.gate.submit(
                CHECK,
                before=before,
                after=after,
                label=SELECTOR,
                params=original_params,
            )
            boundary(SELECTOR, before, after, profile_id=HOIST_ID, job=job)
            validation.gate.join()
            for row in proof_inputs:
                _exact(row)
            if c_consumer is not None:
                transport_refs = _joined_c_delivery(validation, job, bound, c_consumer, original_params)
        else:
            boundary(SELECTOR, before, after, profile_id=HOIST_ID)
        selection.revalidate()
        if kernel.text() != before or result.kernel.text() != after:
            raise ValueError("hoisting source/result changed before actual compiler delivery")
        return (result, proof_inputs, transport_refs) if c_consumer is not None else result

    if active is not None:
        return produce()
    with pass_validation.validating(validation):
        result = produce()
    return result


def replay_to_c_consumer(report_path, *, c_consumer, lean, destination, timeout=120):
    """Fresh proof/ownership operation; a saved record only locates input buffers."""
    from merlin.compile.scheduling.configuration import ConfigurationResult
    from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer, _canonical
    from merlin.sched.ir import parse_kernel

    if type(c_consumer) is not ConfigurationCConsumer:
        raise Unavailable("original typed C consumer required for controlled replay emission")
    c_consumer.revalidate()
    saved = json.loads(Path(report_path).read_bytes().decode("utf-8"))
    if saved.get("schema") != "merlin.compiler-configuration-hoist.checked.v2" or len(saved.get("buffers", [])) != 2:
        raise Unavailable("exact C-bound descriptive replay record required")
    selection, requested = reopen_request(saved["source_request"])
    if requested is None or requested.record() != c_consumer.record():
        raise Unavailable("fresh original C consumer differs from replay input domain")
    origins = saved["buffers"]
    texts = _read_bound_buffers(origins)
    original = parse_kernel(texts[0])
    selected_record = _canonical(selection.selected_contracts.record())
    before = prepare(*texts, selection=selection, c_consumer=c_consumer)
    proof_inputs = tuple(json.loads(before.sources_json))
    completion, fresh = replay(report_path, lean=lean, destination=destination, timeout=timeout)
    if fresh != before or deliver(completion, fresh) != texts[1]:
        raise Unavailable("fresh replay source/target/consumer inputs changed")
    if _read_bound_buffers(origins) != texts:
        raise Unavailable("C-consumer replay originals changed during fresh proof")
    for row in (*proof_inputs, *saved["buffers"]):
        _exact(row)
    report = validate_completion(completion, fresh)
    refs = [file_ref(completion.report), *saved["buffers"]]
    for key in ("buffers", "mapping_records", "imports", "library", "process_evidence"):
        refs.extend(report[key])
    refs.extend(report[key] for key in ("query", "object", "tool", "auditor"))
    for row in refs:
        _exact(row)
    actual = ConfigurationResult(parse_kernel(texts[1]), {"scope": "fresh actual endpoint replay; no producer replay"})
    return c_consumer._issue(
        actual,
        source=original,
        source_text=texts[0],
        selected_contracts=selection.selected_contracts,
        selected_record=selected_record,
        max_calls=selection.max_operations,
        max_fields=selection.max_fields,
        max_depth=64,
        proof_inputs=proof_inputs,
        transport_refs=tuple({row["path"]: row for row in refs}.values()),
    )
