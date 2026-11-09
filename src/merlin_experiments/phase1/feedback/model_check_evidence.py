"""Validate the evidence required by a selected sealed model-check mode."""

from __future__ import annotations


def assess(body, programs, *, mode, shape_counts):
    """Keep original programs distinct from the additional sealed variant rows."""
    body = body if isinstance(body, dict) else {}
    rows = [row for row in body.get("programs", []) if isinstance(row, dict)]
    primary = [row for row in rows if not row.get("source")]
    observed = {row.get("program"): row for row in primary}
    problems = []
    if set(observed) != set(programs) or len(primary) != len(programs):
        problems.append("original program roster is missing or duplicated")
    metrics = body.get("model_metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("worst"), dict):
        problems.append("model metrics unavailable")
    if mode == "fast":
        gates = body.get("gates") or {}
        if any((gates.get(name) or {}).get("status") != "pass"
               or (gates.get(name) or {}).get("partial", False)
               for name in ("build", "kernel_exact", "model_numerics")):
            problems.append("sealed kernel build, exactness or model gate incomplete")
    for program in programs:
        item = observed.get(program, {})
        build, census = item.get("build") or {}, item.get("census") or {}
        kernels = item.get("kernels") or []
        device, denominator = census.get("device"), census.get("denominator")
        if (type(device) is not int or type(denominator) is not int or not 0 <= device <= denominator):
            problems.append(f"{program}: invalid offload census")
        if device and not any(isinstance(k, dict) and isinstance(k.get("source_region_id"), str) for k in kernels):
            problems.append(f"{program}: original kernel identities unavailable")
        if mode == "full":
            if build.get("status") != "linked" or not any(
                isinstance(run, dict) and run.get("engine") == "spike" and run.get("status") == "completed"
                for run in item.get("runs", [])
            ):
                problems.append(f"{program}: linked whole-model Spike execution incomplete")
            continue
        if (type(build.get("built")) is not int or build["built"] < 0
                or build.get("refused") != 0 or any(k.get("status") != "pass" for k in kernels)):
            problems.append(f"{program}: kernel compilation or replay incomplete")
        if not device:
            continue
        expected = shape_counts.get(program)
        varied = [row for row in rows if row.get("source") == "sealed_variants" and row.get("program") == program]
        if type(expected) is not int or expected <= 0 or len(varied) != 1:
            problems.append(f"{program}: sealed shape variants unavailable")
            continue
        cases = varied[0].get("kernels") or []
        if (not cases or any(k.get("status") != "pass" or type(k.get("cases")) is not int
                             or k["cases"] <= 0 for k in cases)
                or sum(k.get("cases", 0) for k in cases) != expected):
            problems.append(f"{program}: sealed shape variant replay incomplete")
    return observed, problems


def rtl_requirement_satisfied(record):
    """An explicit unrequested model RTL gate needs its honest empty status."""
    if record.get("rtl_required") is False:
        return (record.get("rtl_status") == "not_requested" and record.get("rtl") == []
                and record.get("rtl_complete") is False)
    return (record.get("rtl_complete") is True
            and record.get("unique_kernel_count") == len(record.get("rtl") or []))
