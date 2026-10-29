"""Fresh compiler enforcement for actual logical host dispatch fusion.

Saved reports locate input pairs for a new check and never grant delivery.
The existing isolated canonical-source runner compiles and audits the proof.
"""

from __future__ import annotations

import json
import os
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

from . import configuration_elision_runner as runner
from .configuration_elision_bridge import canonical, digest, file_ref
from .configuration_elision_gate import NotEstablished, Unavailable, _exact
from .gate_errors import GateRejected, GateUnavailable, GateUnsupported
from .host_graph_fusion_intake import DOMAIN, MODULE, RELATION, prepare, proof_sources, ready

SELECTOR = "host_graph_fusion"
CHECK = "compiler_host_graph_fusion"
_LIVE = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class Completion:
    identity: str
    report: str

    def __reduce_ex__(self, protocol):
        raise TypeError("live host graph completion cannot be serialized")

    def __copy__(self):
        return Completion(self.identity, self.report)

    def __deepcopy__(self, memo):
        return self.__copy__()


def check_boundary(before, after, destination, *, lean, timeout=120):
    candidate = prepare(before, after)
    try:
        report, _ = runner.prove(
            candidate,
            lean=lean,
            destination=destination,
            timeout=timeout,
            validate=ready,
            query_module=MODULE,
            source_plan=proof_sources,
            relation=RELATION,
        )
    except NotEstablished as error:
        raise GateRejected("public-output preservation was not established for actual host graph fusion") from error
    except (Unavailable, OSError, ValueError) as error:
        raise GateUnavailable("host graph proof unavailable: " + str(error)) from error
    report.update(
        schema="merlin.host-graph-fusion.checked.v1", domain=DOMAIN, compiler_enforcement=False, native_fidelity=False
    )
    path = Path(destination).resolve() / "REPORT.json"
    path.write_text(canonical(report))
    completion = Completion(candidate.identity, str(path))
    _LIVE[completion] = (os.getpid(), threading.get_ident(), candidate, file_ref(path))
    validate_completion(completion)
    return completion


def validate_completion(completion):
    if type(completion) is not Completion or completion not in _LIVE:
        raise GateUnavailable("saved, copied or constructed host fusion completion has no authority")
    pid, thread, candidate, report_ref = _LIVE[completion]
    if (pid, thread) != (os.getpid(), threading.get_ident()) or completion.identity != candidate.identity:
        raise GateUnavailable("host fusion completion belongs to a different invocation")
    try:
        ready(candidate)
        if file_ref(completion.report) != report_ref:
            raise ValueError("host fusion report changed")
        report = json.loads(Path(completion.report).read_text())
        library = Path(report["library_root"])
        if any(path.is_symlink() for path in library.rglob("*")):
            raise ValueError("host fusion private objects contain a symlink")
        if {str(p.resolve()) for p in library.rglob("*") if p.is_file()} != {row["path"] for row in report["library"]}:
            raise ValueError("host fusion private proof membership changed")
        for row in (
            *report["buffers"],
            *report["mapping_records"],
            report["query"],
            report["object"],
            report["tool"],
            report["auditor"],
            *report["imports"],
            *report["library"],
            *report["process_evidence"],
        ):
            _exact(row)
    except (Unavailable, OSError, ValueError) as error:
        raise GateUnavailable("host fusion completion is stale: " + str(error)) from error
    return report


def deliver(completion):
    validate_completion(completion)
    return _LIVE[completion][2].final


def _check_input_refs(refs, paths):
    try:
        if any(file_ref(path) != ref for path, ref in zip(paths, refs, strict=True)):
            raise ValueError("original input reference changed")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GateUnavailable("host fusion source changed") from error


def _read_bound_inputs(refs, paths):
    _check_input_refs(refs, paths)
    try:
        raw = tuple(path.read_bytes() for path in paths)
        if any(digest(data) != ref["sha256"] or len(data) != ref["bytes"] for data, ref in zip(raw, refs, strict=True)):
            raise ValueError("actual input bytes differ from the captured references")
        # Decode the exact bytes without universal-newline normalization.
        return tuple(data.decode("utf-8") for data in raw)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GateUnavailable("host fusion source changed during independent read") from error


def replay(report_path, destination, *, lean, timeout=120):
    report = json.loads(Path(report_path).read_text())
    if report.get("schema") != "merlin.host-graph-fusion.checked.v1" or len(report.get("buffers", [])) != 2:
        raise GateUnavailable("host fusion replay requires its actual source pair")
    try:
        refs = tuple(report["buffers"])
        paths = tuple(Path(row["path"]) for row in refs)
    except (KeyError, TypeError, ValueError) as error:
        raise GateUnavailable("host fusion replay source changed") from error
    # Never restore a saved success. Reproject and kernel-check with current sources.
    before, after = _read_bound_inputs(refs, paths)
    completion = check_boundary(before, after, destination, lean=lean, timeout=timeout)
    if deliver(completion) != after:
        raise GateRejected("host fusion replay delivery differs from the actual target")
    _check_input_refs(refs, paths)
    return completion


def check(job):
    params = job["params"]
    if params.get("index_bits") != 64:
        raise GateUnsupported("host graph compiler gate requires index64")
    lean = Path(params["sdk_bin"]) / "lean"
    if not lean.is_file():
        raise GateUnavailable("selected Lean executable is unavailable")
    try:
        paths = tuple(Path(job[key]) for key in ("before", "after"))
        refs = tuple(file_ref(path) for path in paths)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise GateUnavailable("host fusion source is unavailable") from error
    before, after = _read_bound_inputs(refs, paths)
    completion = check_boundary(before, after, Path(job["directory"]) / "proof", lean=lean)
    report = validate_completion(completion)
    if deliver(completion) != after:
        raise GateRejected("host fusion delivery differs from the actual target")
    _check_input_refs(refs, paths)
    return {
        "schema": "merlin.host-graph-fusion.gate.v1",
        "status": "checked",
        "domain": DOMAIN,
        "relation": RELATION,
        "compiler_enforcement": True,
        "native_fidelity": False,
        "theorems": list(report["statements"]),
        "entry": report["candidate"]["entry"],
        "fused_stages": report["candidate"]["fused_stages"],
        "report": completion.report,
        "source_sha256": report["candidate"]["source_sha256"],
        "final_sha256": report["candidate"]["final_sha256"],
        "premises": report["candidate"]["premises"],
    }
