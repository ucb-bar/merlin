"""Public standalone whole-context schedule checker; no shared broker edits.

Only explicit input-domain exceptions are refusals. Identity/process/runtime/
author failures are unavailable. Old sealed integer host limitations stay
documented and unchanged; this leaf never runs its old capture/main wrapper.
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
from .lean_graph_state_pipeline_kernel import write
from .lean_graph_state_schedule_emit import emit
from .lean_graph_state_schedule_errors import ScheduleDomainRefusal, ScheduleUnavailable, require
from .lean_graph_state_schedule_kernel import BUILD, guard
from .lean_graph_state_schedule_proof import check_query, sources


def configuration(*, factor=2, tile=3, index_bits=32, count=1, minimum_depth=1):
    require(all(type(v) is int for v in (factor, tile, index_bits, count, minimum_depth)), "exact integer schedule configuration required")
    require(2 <= factor <= 4 and 2 <= tile <= 16 and index_bits in (32, 64) and 1 <= count <= 8 and minimum_depth >= 1,
            "selected schedule factor/tile/index/count/depth domain required")
    return {"factor": factor, "tile": tile, "index_bits": index_bits, "count": count, "minimum_depth": minimum_depth}


def unavailable(destination, reason, *, hold):
    record = {"status": "UNAVAILABLE", "reason": reason, "qualified": False, "semantic_refusal": False,
              "context_hold": hold, "cause": "UNKNOWN",
              "stdout": stable.file_ref(destination / "producer.stdout") if (destination / "producer.stdout").is_file() else None,
              "stderr": stable.file_ref(destination / "producer.stderr") if (destination / "producer.stderr").is_file() else None}
    write(destination / "UNAVAILABLE.json", record)
    if hold:
        path = build_dir() / BUILD / "PRODUCER_BLOCKER.json"
        if not path.exists():
            write(path, record)
    raise ScheduleUnavailable(reason + "; retained at " + str(destination))


def intake(reference, destination, *, source=None, buffers=None, orders=None, requests=None, chain=False,
           settings=None, explicit_orders=None):
    guard()
    settings = configuration(**(settings or {}))
    require((source is None) != (buffers is None), "select actual producer input OR complete submitted buffers")
    if buffers is not None:
        require(len(buffers) in (2, 7) and orders is not None and requests is not None,
                "complete pair/seven-buffer boundary and current order trace required")
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(repo_root() / "src/merlin/verify/lean_graph_state_schedule_capture.py"),
               "--reference", str(reference), "--destination", str(destination / "actual"),
               "--factor", str(settings["factor"]), "--tile", str(settings["tile"]), "--bits", str(settings["index_bits"]),
               "--count", str(settings["count"]), "--minimum-depth", str(settings["minimum_depth"])]
    if source is not None:
        command += ["--source", str(source)]
        if chain:
            command += ["--chain"]
        if explicit_orders is not None:
            command += ["--explicit-orders", str(explicit_orders)]
    else:
        command += ["--buffers", *map(str, buffers), "--orders", *map(str, orders), "--requests", str(requests)]
    try:
        with (destination / "producer.stdout").open("x") as out, (destination / "producer.stderr").open("x") as err:
            run = run_owned(command, cwd=reference,
                            env={**os.environ, "PYTHONPATH": str(Path(reference) / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
                            stdout=out, stderr=err, timeout=60, check=False)
    except (subprocess.TimeoutExpired, OSError) as error:
        return unavailable(destination, "producer process/setup failure: " + str(error) + str(getattr(error, "__notes__", [])), hold=True)
    if run.returncode == 1:
        path = destination / "actual/REJECTION.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return unavailable(destination, "exit1 without a well-formed explicit domain refusal", hold=True)
        if (type(data) is dict and set(data) == {"schema", "status", "kind", "reason", "qualified"}
            and data["schema"] == "merlin.graph-state-schedule.explicit-domain-refusal.v1"
            and data["status"] == "unsupported" and data["kind"] == "ScheduleDomainRefusal"
            and type(data["qualified"]) is bool and data["qualified"] is False and type(data["reason"]) is str):
            raise ScheduleDomainRefusal(data["reason"])
        return unavailable(destination, "unconfirmed generic/identity/runtime error is not an input-domain refusal", hold=True)
    if run.returncode:
        diagnostic = destination / "actual/UNAVAILABLE.json"
        detail = json.loads(diagnostic.read_text()) if diagnostic.is_file() else {}
        hold = run.returncode < 0 or detail.get("kind") == "ScheduleIdentityUnavailable" or not detail
        return unavailable(destination, "producer unavailable: " + str(run.returncode) + " " + str(detail), hold=hold)
    path = destination / "actual/CAPTURE.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return unavailable(destination, "success exit without a well-formed complete actual capture", hold=True)


def check_boundary(buffers, orders, requests, destination, reference, *, settings=None, mode="graph"):
    settings = configuration(**(settings or {}))
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    originals = [stable.file_ref(path) for path in buffers]
    original_orders = [stable.file_ref(path) for path in orders]
    journal = stable.file_ref(requests)
    code = sources()
    try:
        capture = intake(reference, destination / "intake", buffers=buffers, orders=orders, requests=requests, settings=settings)
        query, roots, model = emit(capture, mode=mode)
    except ScheduleDomainRefusal as error:
        result = {"schema": "merlin.graph-state-schedule.boundary.v1", "status": "unsupported", "reason": str(error),
                  "originals": originals, "orders": original_orders, "requests": journal,
                  "sources": code, "settings": settings, "mode": mode, "qualified": False,
                  "native_or_identity_failure_counted_as_refusal": False}
        write(destination / "report.json", result)
        return result
    # Unexpected ValueError/KeyError/TypeError is an author/transport failure,
    # never caught as an ordinary unsupported program.
    proof = check_query(query, destination / "lean", roots)
    if any(stable.file_ref(row["path"]) != row for row in [*originals, *original_orders, journal, *code]):
        return unavailable(destination, "original/journal/author identity changed during complete check", hold=True)
    result = {"schema": "merlin.graph-state-schedule.boundary.v1", "status": proof["status"], "originals": originals,
              "orders": original_orders, "requests": journal, "settings": settings, "mode": mode, "model": model,
              "sources": code, "capture": stable.file_ref(destination / "intake/actual/CAPTURE.json"),
              "proof": stable.file_ref(destination / "lean/report.json"), "changed": capture["each_stage_changed"],
              "orders_changed": capture["each_order_changed"], "qualified": False, "review_approved": False,
              "native_or_poison_fidelity_proved": False, "broker_selected": False,
              "claim": "complete return/error/terminal-world preservation under total primitive and anchored opaque/region contracts"}
    write(destination / "report.json", result)
    return result
