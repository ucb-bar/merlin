"""Standalone exact-integer graph boundary validation; no shared broker edits.

Kernel acceptance has the disclosed canonical-entry/operation-contract meaning.
Neither capture diagnostics nor host tests supply native or review approval.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from merlin.common.owned_process import run_owned
from merlin.common.paths import build_dir, repo_root

from . import lean_graph_state as stable
from .lean_graph_state_integer_emit import emit
from .lean_graph_state_integer_kernel import BUILD, guard
from .lean_graph_state_integer_proof import check_query, sources
from .lean_graph_state_pipeline_kernel import write


def configuration(*, factor=2, tile=3, index_bits=32):
    if any(type(v) is not int for v in (factor, tile, index_bits)):
        raise ValueError("actual integer configuration required; bool refused")
    if not 2 <= factor <= 4 or not 2 <= tile <= 16 or index_bits not in (32, 64):
        raise ValueError("supported factor2..4/tile2..16/index32|64 profile required")
    return {"factor": factor, "tile": tile, "index_bits": index_bits}


def _unavailable(destination, reason):
    record = {
        "status": "UNAVAILABLE_PRODUCER",
        "reason": reason,
        "cause": "UNKNOWN",
        "retry_permitted": False,
        "stdout": stable.file_ref(destination / "producer.stdout")
        if (destination / "producer.stdout").is_file()
        else None,
        "stderr": stable.file_ref(destination / "producer.stderr")
        if (destination / "producer.stderr").is_file()
        else None,
        "native_or_setup_failure_counted_as_refusal": False,
    }
    write(destination / "UNAVAILABLE.json", record)
    blocker = build_dir() / BUILD / "PRODUCER_BLOCKER.json"
    if not blocker.exists():
        write(blocker, record)
    raise RuntimeError("producer unavailable; context held, no retry: " + str(destination))


def intake(reference, destination, *, source=None, buffers=None, chain=False, settings=None):
    guard()
    settings = configuration(**(settings or {}))
    if (source is None) == (buffers is None):
        raise ValueError("select producer input OR independent buffers")
    if buffers is not None and len(buffers) not in (2, 6):
        raise ValueError("two standalone or six composed original buffers required")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    script = repo_root() / "src/merlin/verify/lean_graph_state_integer_capture.py"
    command = [
        sys.executable,
        str(script),
        "--reference",
        str(reference),
        "--destination",
        str(destination / "actual"),
        "--factor",
        str(settings["factor"]),
        "--tile",
        str(settings["tile"]),
        "--bits",
        str(settings["index_bits"]),
    ]
    if source is not None:
        command += ["--source", str(source)]
        if chain:
            command += ["--chain"]
    else:
        command += ["--buffers", *map(str, buffers)]
    try:
        with (destination / "producer.stdout").open("x") as out, (destination / "producer.stderr").open("x") as err:
            result = run_owned(
                command,
                cwd=reference,
                env={**os.environ, "PYTHONPATH": str(Path(reference) / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=out,
                stderr=err,
                timeout=60,
                check=False,
            )
    except (subprocess.TimeoutExpired, OSError) as error:
        return _unavailable(destination, str(error) + str(getattr(error, "__notes__", [])))
    if result.returncode:
        rejection = destination / "actual/REJECTION.json"
        if result.returncode != 1 or not rejection.is_file():
            return _unavailable(destination, "signal/setup/unknown producer exit " + str(result.returncode))
        try:
            refused = json.loads(rejection.read_text())
        except json.JSONDecodeError:
            return _unavailable(destination, "malformed producer refusal record")
        if (
            type(refused) is not dict
            or set(refused) != {"schema", "status", "reason", "kind", "qualified"}
            or refused["schema"] != "merlin.graph-state-integer.host-domain-refusal.v1"
            or refused["status"] != "unsupported"
            or type(refused["qualified"]) is not bool
            or refused["qualified"] is not False
            or type(refused["reason"]) is not str
            or refused["kind"] not in ("ValueError", "ParseError", "VerifyException")
        ):
            return _unavailable(destination, "unconfirmed producer failure record")
        raise ValueError("structured actual-IR domain refusal; retained logs: " + str(destination))
    captured = destination / "actual/CAPTURE.json"
    if not captured.is_file():
        return _unavailable(destination, "successful exit without actual capture")
    try:
        return json.loads(captured.read_text())
    except json.JSONDecodeError:
        return _unavailable(destination, "successful exit with malformed capture")


def roots(capture):
    base = "Merlin.GraphStateIntegerInstance."
    names = [base + "prefix_actual"]
    if capture["pipeline"]:
        names += [base + "complete", base + "exactMiddle", base + "Suffix.complete"]
    names += [
        "Merlin.GraphState.Integer.programs_sound",
        "Merlin.GraphState.Integer.Relation.sound",
        "Merlin.GraphState.Integer.typed_result",
        "Merlin.GraphState.Integer.typed_reached",
        "Merlin.GraphState.Integer.canonical_iff",
    ]
    for index, function in enumerate(capture["prefix"]["functions"]):
        if not function["declaration"]:
            names += [base + f"F{index}." + suffix for suffix in ("actual", "reached", "localRewrites")]
    return names


def author(capture, mode="graph"):
    selected = roots(capture)
    if mode == "graph":
        return emit(capture["prefix"], pipeline_capture=capture["pipeline"]), selected, None
    if mode == "initialized-symbolic-provider-v1":
        from .lean_graph_state_integer_models import witness

        text, added, model = witness(capture)
        return text, [*selected, *added], model
    raise ValueError("unknown exact integer checker/model mode")


def check_boundary(buffers, destination, reference, *, settings=None, mode="graph"):
    settings = configuration(**(settings or {}))
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    original = [stable.file_ref(path) for path in buffers]
    code = sources()
    try:
        capture = intake(reference, destination / "intake", buffers=buffers, settings=settings)
        text, selected_roots, model = author(capture, mode)
    except ValueError as error:
        report = {
            "schema": "merlin.graph-state-integer.boundary.v1",
            "status": "unsupported",
            "reason": str(error),
            "originals": original,
            "sources": code,
            "settings": settings,
            "mode": mode,
            "qualified": False,
            "review_approved": False,
            "native_failure_counted_as_refusal": False,
        }
        write(destination / "report.json", report)
        return report
    proof = check_query(text, destination / "lean", selected_roots)
    if any(stable.file_ref(row["path"]) != row for row in [*original, *code]):
        raise ValueError("original or validator source changed during boundary proof")
    report = {
        "schema": "merlin.graph-state-integer.boundary.v1",
        "status": proof["status"],
        "originals": original,
        "capture": stable.file_ref(destination / "intake/actual/CAPTURE.json"),
        "settings": settings,
        "reference": str(Path(reference).resolve()),
        "sources": code,
        "mode": mode,
        "model": model,
        "proof": stable.file_ref(destination / "lean/report.json"),
        "changed": capture["each_stage_changed"],
        "qualified": False,
        "review_approved": False,
        "native_fidelity_proved": False,
        "poison_freedom_proved": False,
        "claim": "canonical-entry complete returned/error/world correspondence under local operation contracts",
    }
    write(destination / "report.json", report)
    return report
