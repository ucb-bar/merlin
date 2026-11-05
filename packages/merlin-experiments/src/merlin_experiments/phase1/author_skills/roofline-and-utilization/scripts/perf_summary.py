#!/usr/bin/env python3
"""Summarize the performance evidence already in this workspace; read-only, stdlib only.

Run from the workspace root (or pass ``--workspace``). It reads, and never writes:

* ``qa/latest_completed_grade.json``: per-case rows (``per_capsule``) with their ``performance`` record,
  and the performance cohorts of ``joint_test_selection``;
* ``qa/tier_state.json``: the newest L2 and L3 execution digest of each case;
* ``.qa_channel/resp_*.json`` and ``simresp_*.json``: self-check and simulator responses;
* ``.model_check_channel/resp_*.json``: whole-model check reports;
* ``qa/idle_tuning.json``: idle-time timing experiments;
* ``isa/<case>/capsule.yaml``: the declared operation of a case, when its declaration is public.

The definitions repeat the harness's own feedback writers, so the numbers match the ``performance_brief``
and ``development_headroom`` blocks the author already receives:

* bound = the largest of ``structural_compute_floor_cycles`` and the read/write
  ``logical_request_ideal_service.*.ideal_requested_service_cycles`` of ``structural_roofline``; its name is
  ``bound_by`` (``compute``, ``read requests``, ``write requests``);
* array utilization % = 100 * ``useful_macs`` / (cycles * ``structural_peak_macs_per_cycle``);
* measured = a gSIM response row (``sim: gsim``, ``barrier_tier: L3``, ``pass: true``): ``barrier_cycles``;
* modeled = a ``performance`` record with status PROVISIONAL or QUALIFIED and complete event accounting:
  ``estimated_whole_soc_cycles``;
* busiest unit = the ``scoped_resource_feedback.resources`` row with the largest
  ``capacity_normalized_activity`` (a modeled reservation share, not a hardware counter);
* modeled-vs-measured ratio = measured / modeled for the same case and the same ``execution_digest``.

A case counts as passing when the newest completed grade says its L2 tier passed, or, for a case that
grade did not report, when a ``submission/`` self-check did. Per case the evidence is, in order: the gSIM
row of the certified (or current) executable, the newest gSIM row, the newest modeled estimate of
``submission/`` (grade row or self-check). Variant and knob-copy responses never describe ``submission/``.
The busiest unit of a measured case is read from the measured row itself, as the request-level brief does.

Everything printed is advisory and comes from files the author can already read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

MAX_BYTES = 64 << 20
#: Newest response files read per channel; older ones describe compilers long gone.
MAX_FILES = 800
MODELED_STATUSES = ("PROVISIONAL", "QUALIFIED")


# ------------------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------------------

def _positive(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _load(path: Path):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
            return None
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and document.get("result_index"):
            for parent in Path(path).resolve().parents:
                helper = parent / "qa_result_details.py"
                if helper.is_file():
                    namespace = {"__name__": "_public_result_details"}
                    exec(compile(helper.read_bytes(), str(helper), "exec"), namespace)
                    return namespace["hydrate"](document, workspace=parent)
        return document
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _newest(directory: Path, prefixes: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.suffix == ".json" and p.name.startswith(prefixes)]
    files.sort(key=_mtime, reverse=True)
    return files[:MAX_FILES]


def _tier_status(value):
    return value.get("status") if isinstance(value, dict) else value


def _passes(row: dict) -> bool:
    tiers = row.get("tiers")
    if isinstance(tiers, dict) and "L2" in tiers:
        return _tier_status(tiers["L2"]) == "pass"
    return row.get("pass") is True or row.get("numerical_execution_passed") is True


def _name(row: dict) -> str:
    return str(row.get("capsule") or row.get("name") or "")


def _rows(document) -> list[dict]:
    if not isinstance(document, dict):
        return []
    rows = document.get("per_capsule") or document.get("cases") or []
    return [row for row in rows if isinstance(row, dict) and _name(row)]


def _is_submission(document: dict) -> bool:
    """Responses about ``submission/`` itself (variants and knob copies carry a ``variant`` block)."""
    return not document.get("variant") and document.get("selfcheck_submission") in (None, "submission")


# ------------------------------------------------------------------------------------------------
# Definitions (mirroring the feedback writers)
# ------------------------------------------------------------------------------------------------

def bound(roofline) -> tuple[float | None, str | None]:
    """``(cycles, bound_by)``: the largest positive structural term of a ``structural_roofline``."""
    roofline = roofline if isinstance(roofline, dict) else {}
    terms = {"compute": roofline.get("structural_compute_floor_cycles")}
    for side, service in (roofline.get("logical_request_ideal_service") or {}).items():
        if isinstance(service, dict):
            terms[f"{side} requests"] = service.get("ideal_requested_service_cycles")
    terms = {name: value for name, value in terms.items() if _positive(value)}
    if not terms:
        return None, None
    name = max(terms, key=terms.get)
    return terms[name], name


def utilization_pct(roofline, cycles) -> float | None:
    """100 * useful MACs / (cycles * structural peak MACs per cycle), or None when a term is missing."""
    roofline = roofline if isinstance(roofline, dict) else {}
    useful, peak = roofline.get("useful_macs"), roofline.get("structural_peak_macs_per_cycle")
    if _positive(useful) and _positive(peak) and _positive(cycles):
        return round(100.0 * useful / (cycles * peak), 2)
    return None


def busiest(performance) -> dict | None:
    resources = ((performance or {}).get("scoped_resource_feedback") or {}).get("resources") or ()
    ranked = [r for r in resources if isinstance(r, dict) and _positive(r.get("capacity_normalized_activity"))]
    if not ranked:
        return None
    top = max(ranked, key=lambda r: r["capacity_normalized_activity"])
    return {"name": top.get("name"), "modeled_activity": round(top["capacity_normalized_activity"], 3)}


def modeled_cycles(performance):
    performance = performance if isinstance(performance, dict) else {}
    cycles = performance.get("estimated_whole_soc_cycles")
    if (performance.get("status") in MODELED_STATUSES and performance.get("complete_event_accounting") is True
            and _positive(cycles)):
        return cycles
    return None


def measured_cycles(document: dict, row: dict):
    cycles = row.get("barrier_cycles")
    if (document.get("sim") == "gsim" and row.get("barrier_tier") == "L3" and row.get("pass") is True
            and _positive(cycles)):
        return cycles
    return None


def _quartiles(values: list[float]) -> dict | None:
    values = sorted(values)
    if not values:
        return None
    if len(values) >= 4:
        q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
    else:
        q1, median, q3 = values[0], statistics.median(values), values[-1]
    return {"n": len(values), "min": round(values[0], 3), "q1": round(q1, 3), "median": round(median, 3),
            "q3": round(q3, 3), "max": round(values[-1], 3)}


def work_class(roofline) -> str:
    """Order of magnitude of the case's useful MACs: a shape-size class read from the roofline."""
    useful = (roofline or {}).get("useful_macs") if isinstance(roofline, dict) else None
    if not _positive(useful):
        return "no MAC work"
    decade = int(math.floor(math.log10(useful)))
    return f"1e{decade}-1e{decade + 1} MACs"


# ------------------------------------------------------------------------------------------------
# Collection
# ------------------------------------------------------------------------------------------------

def _declared_operation(workspace: Path, name: str) -> str | None:
    """``operation.op`` of ``isa/<name>/capsule.yaml`` by a line scan (no YAML dependency)."""
    path = workspace / "isa" / name / "capsule.yaml"
    try:
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    except (OSError, UnicodeDecodeError):
        return None
    inside = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            inside = line.split("#", 1)[0].strip() == "operation:"
            continue
        if inside:
            key, _, value = line.strip().partition(":")
            if key == "op" and value.strip():
                return value.strip().strip("'\"")
    return None


#: Dialects and op names that carry data or structure rather than the computation; skipped when naming
#: the operations of a public source.
_STRUCTURAL_DIALECTS = frozenset({"arith", "tensor", "func", "builtin", "memref", "scf", "affine", "cf",
                                  "bufferization", "math", "index"})
_STRUCTURAL_OPS = frozenset({"tensor", "resident_pack", "commit", "evict", "yield", "return", "constant",
                             "empty", "fill"})
#: More computing ops than this and a source is grouped as one connected program.
_CONNECTED_OPS = 3


def _source_operations(workspace: Path, name: str) -> dict[str, int]:
    """``{op: count}`` of the computing ops of ``public_sources/<name>.mlir`` (empty when absent)."""
    path = workspace / "public_sources" / f"{name}.mlir"
    try:
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            return {}
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    found: dict[str, int] = {}
    for line in text.splitlines():
        body = line.split("=", 1)[1] if "=" in line.split("{", 1)[0] else line
        words = body.strip().split(None, 1)
        token = words[0].strip('"').split("(", 1)[0] if words else ""
        dialect, dot, op = token.partition(".")
        if not dot or not dialect.isidentifier() or not op.replace(".", "_").isidentifier():
            continue
        if dialect in _STRUCTURAL_DIALECTS or op in _STRUCTURAL_OPS:
            continue
        found[op] = found.get(op, 0) + 1
    return found


def _family_resolver(workspace: Path):
    """The workspace's own semantic-family mapping (``semantic_families.py``), else the declared op name."""
    mapping = None
    source = workspace / "semantic_families.py"
    if source.is_file():  # often a link to the staged tool
        # Loaded by path: no sys.path change, no module cache entry, no bytecode file in the workspace.
        # (Registered only while it executes: dataclasses look their module up there.)
        before, private = sys.dont_write_bytecode, "_workspace_semantic_families"
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location(private, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[private] = module
            spec.loader.exec_module(module)
            mapping = getattr(module, "from_op", None)
        except Exception:  # noqa: BLE001 -- the declared op name is a fine fallback
            mapping = None
        finally:
            sys.modules.pop(private, None)
            sys.dont_write_bytecode = before

    def family(op: str) -> str:
        try:
            return (mapping(op) if mapping else None) or op
        except Exception:  # noqa: BLE001
            return op

    def resolve(name: str) -> str:
        op = _declared_operation(workspace, name)
        if op is not None:
            return family(op)
        found = _source_operations(workspace, name)
        if not found:
            return "undeclared"
        names = sorted({family(op) for op in found})
        if sum(found.values()) > _CONNECTED_OPS:
            shown = "+".join(names[:_CONNECTED_OPS]) + ("+..." if len(names) > _CONNECTED_OPS else "")
            return f"connected program ({shown})"
        return "+".join(names)

    return resolve


def collect(workspace: Path) -> dict:
    """Everything the report needs, gathered from the workspace's own files."""
    qa, channel = workspace / "qa", workspace / ".qa_channel"
    grade_path = qa / "latest_completed_grade.json"
    grade = _load(grade_path) or {}
    tier_state = _load(qa / "tier_state.json") or {}
    tier_state = tier_state if isinstance(tier_state, dict) else {}
    passing: dict[str, bool] = {}
    modeled: dict[str, list] = {}     # name -> [(time, cycles, performance, source)]
    measured: dict[str, list] = {}    # name -> [(time, cycles, digest, performance)]
    pair_model: dict[tuple, tuple] = {}
    pair_measure: dict[tuple, tuple] = {}
    latency: dict[str, list] = {}
    grade_time = _mtime(grade_path)
    for row in _rows(grade):
        name = _name(row)
        passing[name] = _passes(row)
        cycles = modeled_cycles(row.get("performance"))
        if cycles is not None:
            modeled.setdefault(name, []).append((grade_time, cycles, row["performance"], "grade"))
    newest_response = 0.0
    for path in _newest(channel, ("resp_", "simresp_")):
        document = _load(path)
        if not isinstance(document, dict):
            continue
        when = _mtime(path)
        newest_response = max(newest_response, when)
        _latency(path, document, latency)
        submission = _is_submission(document)
        for row in _rows(document):
            name = _name(row)
            performance = row.get("performance") if isinstance(row.get("performance"), dict) else {}
            digest = row.get("execution_digest")
            model = modeled_cycles(performance)
            measure = measured_cycles(document, row)
            if isinstance(digest, str) and digest:
                if model is not None:
                    pair_model.setdefault((name, digest), (when, model, performance))
                if measure is not None:
                    pair_measure.setdefault((name, digest), (when, measure))
            if not submission:
                continue
            if name not in passing and _passes(row):
                passing[name] = True
            if measure is not None:
                measured.setdefault(name, []).append((when, measure, digest, performance))
            elif model is not None and _passes(row):
                modeled.setdefault(name, []).append((when, model, performance, "self-check"))
    for rows in list(modeled.values()) + list(measured.values()):
        rows.sort(key=lambda item: item[0], reverse=True)
    selection = grade.get("joint_test_selection") or {}
    cohorts = selection.get("performance_cohorts") or {}
    # Cases declared ``performance_scope: excluded`` stay graded for correctness but are not ranked.
    excluded = (selection.get("performance_scope") or {}).get("excluded") or []
    membership = {}
    if isinstance(cohorts, dict):
        for lane, names in cohorts.items():
            for name in names if isinstance(names, list) else ():
                membership.setdefault(str(name), lane)
    return {
        "workspace": workspace, "grade": grade, "grade_time": grade_time, "tier_state": tier_state,
        "passing": passing, "modeled": modeled, "measured": measured, "pair_model": pair_model,
        "pair_measure": pair_measure, "latency": latency, "cohort": membership,
        "excluded": {str(name) for name in excluded if isinstance(name, str)},
        "newest_response": newest_response, "family": _family_resolver(workspace),
    }


def _latency(path: Path, document: dict, latency: dict) -> None:
    """Seconds from request to result, grouped by kind of request."""
    start, end = document.get("requested_at_unix_ns"), document.get("graded_at_unix_ns")
    seconds = None
    if type(start) is int and type(end) is int and end > start:
        seconds = (end - start) / 1e9
    elif path.name.startswith("simresp_"):
        request = path.with_name("simreq_" + path.name[len("simresp_"):])
        if request.is_file() and _mtime(path) > _mtime(request):
            seconds = _mtime(path) - _mtime(request)
    if seconds is None:
        return
    rid = str(document.get("selfcheck_request_id") or path.stem)
    origin = "operator idle" if "op_idle" in rid else "author"
    if document.get("sim") == "gsim":
        kind = "gSIM (L3) check"
    elif "scalability" in str(document.get("schema") or ""):
        kind = "Spike timing (scalability suite)"
    elif document.get("sim") == "spike":
        kind = "Spike core check"
    else:
        kind = "other check"
    if document.get("variant"):
        kind += ", variant"
    latency.setdefault(f"{kind} [{origin}]", []).append(seconds)


def _digests(state: dict) -> tuple[str | None, str | None, bool]:
    """``(current L2 digest, certified L3 digest, certificate describes the current executable)``.

    As in the grade's headroom block: a pass or joined L3 ledger slot for the current L2 executable
    means the certified cycles still describe the current program.
    """
    l2, l3 = state.get("L2"), state.get("L3")
    current = l2.get("execution_digest") if isinstance(l2, dict) else None
    certified = l3.get("execution_digest") if isinstance(l3, dict) and l3.get("status") == "pass" else None
    slots = ((state.get("<certs>") or {}).get("L3") or {}) if isinstance(state.get("<certs>"), dict) else {}
    joined = bool(current) and any(isinstance(slot, dict) and slot.get("execution_digest") == current
                                   and slot.get("status") in ("pass", "joined") for slot in slots.values())
    return current, certified, joined or (bool(current) and certified == current)


def entries(data: dict) -> tuple[list[dict], list[str]]:
    """One entry per passing case with evidence, and the passing cases without any."""
    out, missing = [], []
    tier_state = data["tier_state"]
    excluded = data.get("excluded") or set()
    for name in sorted(n for n, ok in data["passing"].items() if ok and n not in excluded):
        state = tier_state.get(name) if isinstance(tier_state.get(name), dict) else {}
        current_l2, certified, cert_current = _digests(state)
        chosen = None
        for when, cycles, digest, performance in data["measured"].get(name, ()):
            chosen = chosen or (cycles, performance, "measured: gSIM", digest)
            if digest in (certified, current_l2) and digest:
                chosen = (cycles, performance, "measured: gSIM", digest)
                break
        if chosen is None and data["modeled"].get(name):
            _, cycles, performance, source = data["modeled"][name][0]
            chosen = (cycles, performance, f"modeled: Spike timing model ({source})", None)
        if chosen is None:
            missing.append(name)
            continue
        cycles, performance, evidence, digest = chosen
        roofline = performance.get("structural_roofline") or {}
        limit, limiter = bound(roofline)
        if limit is None or cycles < limit:
            missing.append(name)
            continue
        entry = {"test_case": name, "cycles": cycles, "evidence": evidence,
                 "roofline_bound_cycles": round(limit, 1), "bound_by": limiter,
                 "gap_ratio": round(cycles / limit, 2), "headroom_cycles": round(cycles - limit, 1),
                 "array_utilization_pct": utilization_pct(roofline, cycles),
                 "busiest_modeled_unit": busiest(performance),
                 "family": data["family"](name), "cohort": data["cohort"].get(name, "-"),
                 "work_class": work_class(roofline),
                 "has_gsim": bool(data["measured"].get(name))}
        if evidence.startswith("measured") and current_l2 and digest != current_l2 and not (
                cert_current and digest == certified):
            entry["measured_on_earlier_executable"] = True
        entry["traffic"] = _traffic(performance, roofline)
        entry["_performance"] = performance
        out.append(entry)
    out.sort(key=lambda e: (-e["headroom_cycles"], -e["gap_ratio"], e["test_case"]))
    return out, missing


def _traffic(performance: dict, roofline: dict) -> dict:
    """Requested bytes the program issued against the workload's logical bytes, per direction."""
    counts = performance.get("feature_counts") if isinstance(performance.get("feature_counts"), dict) else {}
    service = roofline.get("logical_request_ideal_service") or {}
    found = {}
    for side in ("read", "write"):
        logical = (service.get(side) or {}).get("requested_bytes") if isinstance(service.get(side), dict) else None
        issued = counts.get(f"target.requested_{side}_bytes")
        if _positive(logical) and type(issued) is int:
            found[side] = round(issued / logical, 2)
    commands = counts.get("target.commands")
    if type(commands) is int:
        found["commands"] = commands
    return found


def calibration(data: dict) -> dict:
    pairs = []
    for key in sorted(set(data["pair_model"]) & set(data["pair_measure"])):
        when, model, performance = data["pair_model"][key]
        measure = data["pair_measure"][key][1]
        _, limiter = bound((performance or {}).get("structural_roofline"))
        pairs.append({"test_case": key[0], "execution_digest": key[1][:12], "modeled": model, "measured": measure,
                      "ratio": round(measure / model, 3), "bound_by": limiter,
                      "when": max(when, data["pair_measure"][key][0])})
    newest = {}
    for pair in sorted(pairs, key=lambda p: p["when"], reverse=True):
        newest.setdefault(pair["test_case"], pair)
    return {"pairs": len(pairs), "ratio": _quartiles([p["ratio"] for p in pairs]),
            "per_case": sorted(newest.values(), key=lambda p: (-abs(math.log(p["ratio"])), p["test_case"])),
            "beyond_2x": sorted(p["test_case"] for p in newest.values() if p["ratio"] > 2 or p["ratio"] < 0.5)}


def groups(found: list[dict], key: str) -> list[dict]:
    total = sum(e["headroom_cycles"] for e in found) or 1.0
    by: dict[str, list[dict]] = {}
    for entry in found:
        by.setdefault(str(entry.get(key)), []).append(entry)
    out = []
    for name, members in by.items():
        util = [e["array_utilization_pct"] for e in members if e["array_utilization_pct"] is not None]
        headroom = sum(e["headroom_cycles"] for e in members)
        out.append({"group": name, "cases": len(members),
                    "median_utilization_pct": round(statistics.median(util), 2) if util else None,
                    "headroom_cycles": round(headroom), "headroom_share_pct": round(100 * headroom / total, 1)})
    out.sort(key=lambda g: (-g["headroom_cycles"], g["group"]))
    return out


def units(found: list[dict]) -> list[dict]:
    """Per modeled unit: in how many cases it is the busiest, and its median activity where reported."""
    seen: dict[str, list[float]] = {}
    top: dict[str, int] = {}
    for entry in found:
        unit = entry["busiest_modeled_unit"]
        if unit:
            top[unit["name"]] = top.get(unit["name"], 0) + 1
        resources = (entry.get("_performance", {}).get("scoped_resource_feedback") or {}).get("resources") or ()
        for resource in resources:
            if isinstance(resource, dict) and type(resource.get("capacity_normalized_activity")) in (int, float):
                seen.setdefault(str(resource.get("name")), []).append(resource["capacity_normalized_activity"])
    out = [{"unit": name, "busiest_in_cases": top.get(name, 0), "cases_reported": len(values),
            "median_modeled_activity": round(statistics.median(values), 3)} for name, values in seen.items()]
    out.sort(key=lambda u: (-u["busiest_in_cases"], -u["median_modeled_activity"], u["unit"]))
    return out


def models(workspace: Path) -> list[dict]:
    """Newest whole-model check per model: per-kernel cycle share weighted by its source regions."""
    newest: dict[str, dict] = {}
    for path in _newest(workspace / ".model_check_channel", ("resp_",)):
        document = _load(path)
        if not isinstance(document, dict) or not document.get("model") or document["model"] in newest:
            continue
        report = document.get("report") if isinstance(document.get("report"), dict) else {}
        kernels, fractions = [], {}
        for program in report.get("programs") or ():
            if not isinstance(program, dict):
                continue
            census = program.get("census") if isinstance(program.get("census"), dict) else {}
            if census.get("device_mac_fraction") is not None:
                fractions[str(program.get("program"))] = census["device_mac_fraction"]
            for kernel in program.get("kernels") or ():
                cycles = kernel.get("cycles") if isinstance(kernel, dict) else None
                if not isinstance(cycles, dict) or not _positive(cycles.get("max")):
                    continue
                regions = len(kernel.get("source_region_ids") or ()) or 1
                kernels.append({"program": program.get("program"), "form": kernel.get("form"),
                                "symbol": kernel.get("symbol"), "cycles_kind": cycles.get("kind"),
                                "cycles_per_call": cycles["max"], "source_regions": regions,
                                "weighted_cycles": cycles["max"] * regions})
        total = sum(k["weighted_cycles"] for k in kernels) or 1
        for kernel in kernels:
            kernel["share_pct"] = round(100 * kernel["weighted_cycles"] / total, 1)
        kernels.sort(key=lambda k: -k["weighted_cycles"])
        newest[document["model"]] = {"model": document["model"], "status": document.get("status"),
                                     "source_sha256": str(document.get("source_sha256") or "")[:12],
                                     "device_mac_fraction": fractions, "kernels": kernels}
    return list(newest.values())


def tuning(workspace: Path) -> dict | None:
    ledger = _load(workspace / "qa" / "idle_tuning.json")
    if not isinstance(ledger, dict):
        return None
    rows = [r for r in ledger.get("experiments") or () if isinstance(r, dict)]
    rows.sort(key=lambda r: (r.get("delta_pct") is None, r.get("delta_pct") or 0.0, str(r.get("test_case"))))
    return {"source_sha256": str(ledger.get("source_sha256") or "")[:12], "count": len(rows),
            "experiments": [{k: r.get(k) for k in ("test_case", "mechanism", "delta_pct", "numerics", "how_to_apply")}
                            for r in rows]}


def report(workspace: Path, *, top: int = 10) -> dict:
    data = collect(workspace)
    found, missing = entries(data)
    unit_rows = units(found)
    for entry in found:
        entry.pop("_performance", None)
    util = [e["array_utilization_pct"] for e in found if e["array_utilization_pct"] is not None]
    grade = data["grade"] if isinstance(data["grade"], dict) else {}
    pointer = grade.get("latest_completed_grade") if isinstance(grade.get("latest_completed_grade"), dict) else grade
    now = time.time()

    def age(stamp):
        return round((now - stamp) / 60, 1) if stamp else None

    measured_util = [e["array_utilization_pct"] for e in found
                     if e["evidence"].startswith("measured") and e["array_utilization_pct"] is not None]
    return {
        "workspace": str(workspace),
        "freshness": {
            "grade_graded_at": pointer.get("graded_at"),
            "grade_snapshot_current": grade.get("snapshot_current"),
            "grade_file_age_min": age(data["grade_time"]),
            "newest_response_age_min": age(data["newest_response"]),
        },
        "counts": {"passing": sum(data["passing"].values()), "with_evidence": len(found),
                   "measured": sum(e["evidence"].startswith("measured") for e in found),
                   "modeled": sum(e["evidence"].startswith("modeled") for e in found),
                   "without_evidence": len(missing),
                   "performance_scope_excluded": sum(
                       ok for name, ok in data["passing"].items() if name in data.get("excluded", ()))},
        "utilization_pct": _quartiles(util),
        "utilization_pct_measured": _quartiles(measured_util),
        "by_family": groups(found, "family"), "by_work_class": groups(found, "work_class"),
        "by_bound": groups(found, "bound_by"), "by_cohort": groups(found, "cohort"),
        "largest_headroom": found[:top],
        "lowest_utilization": sorted((e for e in found if e["array_utilization_pct"] is not None),
                                     key=lambda e: (e["array_utilization_pct"], -e["headroom_cycles"]))[:top],
        "units": unit_rows[:top],
        "model_vs_measured": calibration(data),
        "without_gsim": sorted(e["test_case"] for e in found if not e["has_gsim"]),
        "without_evidence": missing,
        "connected_models": models(workspace),
        "idle_tuning": tuning(workspace),
        "latency_s": {kind: _quartiles(values) for kind, values in sorted(data["latency"].items())},
    }


# ------------------------------------------------------------------------------------------------
# Text output
# ------------------------------------------------------------------------------------------------

def _q(stats: dict | None, unit: str = "") -> str:
    if not stats:
        return "none"
    return (f"n={stats['n']} min={stats['min']}{unit} q1={stats['q1']}{unit} median={stats['median']}{unit} "
            f"q3={stats['q3']}{unit} max={stats['max']}{unit}")


def _names(names: list[str], limit: int = 12) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" ... (+{len(names) - limit})" if len(names) > limit else "") if names else "-"


def render(doc: dict, *, top: int = 10) -> str:
    lines = [f"# performance summary of {doc['workspace']} (read-only; advisory)"]
    fresh = doc["freshness"]
    lines.append(f"grade graded_at={fresh['grade_graded_at']} snapshot_current={fresh['grade_snapshot_current']} "
                 f"file_age_min={fresh['grade_file_age_min']}; "
                 f"newest response age_min={fresh['newest_response_age_min']}")
    c = doc["counts"]
    lines.append(f"passing={c['passing']} with_evidence={c['with_evidence']} (measured={c['measured']}, "
                 f"modeled={c['modeled']}) without_evidence={c['without_evidence']}")
    lines += ["", "## array utilization % (100 * useful_macs / (cycles * structural_peak_macs_per_cycle))",
              f"all evidence: {_q(doc['utilization_pct'])}",
              f"gSIM-measured only: {_q(doc['utilization_pct_measured'])}"]
    for title, key in (("family (declared operation, else the ops of its public source)", "by_family"),
                       ("work class (useful MACs)", "by_work_class"), ("bound_by", "by_bound"),
                       ("performance cohort", "by_cohort")):
        lines += ["", f"## by {title}: cases, median utilization %, headroom cycles (share)"]
        for g in doc[key][:top]:
            lines.append(f"- {g['group']}: {g['cases']} cases, util {g['median_utilization_pct']}%, "
                         f"headroom {g['headroom_cycles']} ({g['headroom_share_pct']}%)")
    lines += ["", "## largest headroom (cycles - roofline bound)"]
    for e in doc["largest_headroom"]:
        unit = e["busiest_modeled_unit"]
        unit_text = f"{unit['name']} {unit['modeled_activity']}" if unit else "-"
        stale = " [measured on an earlier executable]" if e.get("measured_on_earlier_executable") else ""
        lines.append(f"- {e['test_case']}: {e['cycles']} cyc vs bound {e['roofline_bound_cycles']} ({e['bound_by']}), "
                     f"gap {e['gap_ratio']}x, util {e['array_utilization_pct']}%, busiest {unit_text}, "
                     f"traffic {e['traffic'] or '-'}; {e['evidence']}{stale}")
    lines += ["", "## lowest utilization"]
    for e in doc["lowest_utilization"]:
        lines.append(f"- {e['test_case']}: util {e['array_utilization_pct']}% of {e['cycles']} cyc, "
                     f"bound_by {e['bound_by']}; {e['evidence']}")
    lines += ["", "## modeled units (busiest_in_cases, median capacity_normalized_activity where reported)"]
    for u in doc["units"]:
        lines.append(f"- {u['unit']}: busiest in {u['busiest_in_cases']}, median {u['median_modeled_activity']} "
                     f"over {u['cases_reported']} cases")
    cal = doc["model_vs_measured"]
    lines += ["", "## modeled vs measured (gSIM / Spike model, same case and execution_digest)",
              f"pairs={cal['pairs']} ratio: {_q(cal['ratio'])}",
              f"newest pair per case off by more than 2x: {_names(cal['beyond_2x'])}"]
    for p in cal["per_case"][:top]:
        lines.append(f"- {p['test_case']}: measured {p['measured']} / modeled {p['modeled']} = {p['ratio']} "
                     f"({p['bound_by']})")
    lines += ["", f"## passing cases without gSIM evidence ({len(doc['without_gsim'])}): {_names(doc['without_gsim'])}",
              f"## passing cases without any usable evidence ({len(doc['without_evidence'])}): "
              f"{_names(doc['without_evidence'])}"]
    if doc["connected_models"]:
        lines += ["", "## whole-model checks (newest per model; kernel share = cycles per call * source regions)"]
        for m in doc["connected_models"]:
            lines.append(f"- {m['model']} status={m['status']} source={m['source_sha256']} "
                         f"device_mac_fraction={m['device_mac_fraction'] or '-'}")
            for k in m["kernels"][:5]:
                lines.append(f"    {k['share_pct']}% {k['program']} {k['form']} ({k['cycles_kind']} cycles "
                             f"{k['cycles_per_call']} x {k['source_regions']} regions)")
    if doc["idle_tuning"]:
        t = doc["idle_tuning"]
        lines += ["", f"## idle tuning (qa/idle_tuning.json, source {t['source_sha256']}, {t['count']} results; "
                      "cycles are Spike model estimates)"]
        for r in t["experiments"][:top]:
            lines.append(f"- {r['test_case']}: {r['mechanism']} delta {r['delta_pct']}%; {r['numerics']}")
    if doc["latency_s"]:
        lines += ["", "## request latency in seconds (request to result, your own history)"]
        for kind, stats in doc["latency_s"].items():
            lines.append(f"- {kind}: {_q(stats, 's')}")
    return "\n".join(lines) + "\n"


# The operator console's per-test-case view (``tracking/console/sources/case_history.py``) loads this file
# by path, so the author and the operator read the same evidence by one set of definitions.
passes = _passes
is_submission = _is_submission
case_name = _name
case_rows = _rows
digests = _digests
traffic = _traffic


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only summary of this workspace's performance evidence.")
    parser.add_argument("--workspace", default=".", help="workspace root (default: current directory)")
    parser.add_argument("--top", type=int, default=10, help="rows per ranked section")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    doc = report(Path(args.workspace).resolve(), top=max(1, args.top))
    sys.stdout.write(json.dumps(doc, indent=1, default=str) + "\n" if args.json else render(doc, top=max(1, args.top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
