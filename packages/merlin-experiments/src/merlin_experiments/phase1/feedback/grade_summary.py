"""A one-line-per-test-case text view of ``qa/latest_completed_grade.json``, rewritten with it.

The completed-grade file carries every row's performance record, compiler diagnostics and brief, so it
reached 4.8 MB on a live joint run, and the author queried it with ad hoc scripts again and again to
learn which cases pass, which tiers ran and how the failures group. This view answers those questions
in a few kilobytes: the newest grade's identity and counts, every failure cluster with its members and
reproduce command, and one line per test case (status, tiers, failure plane/category, performance
status and estimated cycles, and the grade a row came from when it is not the newest).

It opens with that document's ``performance_headline`` (correctness in one line, utilization and the
largest-headroom cases), so the first lines say where the cycles are.

A row whose numerics pass while its structural trace evidence is failed or unavailable says so on its
line (``trace=<status>: <reason>``) and in its own section: the final closure requires that evidence
(:func:`merlin.targetgen.trace_check.requirements_complete`), so a passing row that lacks it would be
withheld there, and the author should learn that while still able to act on it.

It is derived only from the document it sits beside, which the author can already read, so it adds no
information and hides none: the full file stays where it was and every line says where its detail is.
Writing it never affects the grade or the full file.
"""

from __future__ import annotations

from pathlib import Path

from .completed_grade import SUMMARY_FILENAME as FILENAME

_MAX_DETAIL = 200


def _text(value, limit: int = _MAX_DETAIL) -> str:
    text = " ".join(str(value).split()) if value is not None else ""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _tiers(row: dict) -> str:
    tiers = row.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        return "-"
    out = []
    for name in sorted(tiers):
        value = tiers[name]
        status = value.get("status") if isinstance(value, dict) else value
        out.append(f"{name}={status}")
    return ",".join(out)


def _performance(row: dict) -> str:
    performance = row.get("performance")
    if not isinstance(performance, dict):
        return ""
    cycles = performance.get("estimated_whole_soc_cycles")
    status = performance.get("status")
    return f" perf={status}" + (f" cycles={cycles}" if cycles is not None else "")


def trace_withheld(row: dict) -> str | None:
    """``"<status>: <reason>"`` when the row's trace evidence would withhold final closure, else ``None``."""
    status, requirement = row.get("trace_status"), row.get("trace_requirement_status")
    if status is None and requirement is None:
        return None  # no instruction-trace obligation on this row
    if status == "pass" and requirement in (None, "pass"):
        return None
    violations = [str(v) for v in row.get("trace_violations") or ()]
    unavailable = [str(v) for v in row.get("trace_unavailable_requirements") or ()]
    if violations:
        why = violations[0] + (f" (+{len(violations) - 1} more)" if len(violations) > 1 else "")
    elif unavailable:
        why = "evidence unavailable for " + ", ".join(unavailable)
    else:
        why = "no structural evidence recorded"
    return f"{requirement or status}: {_text(why)}"


def _case_line(row: dict, newest_token) -> str:
    plane, category = row.get("failure_plane"), row.get("failure_category")
    failure = f"{plane}/{category}" if plane or category else "-"
    graded = row.get("graded") if isinstance(row.get("graded"), dict) else {}
    origin = "" if str(graded.get("token")) == str(newest_token) else f" graded={graded.get('token')}"
    if graded.get("currency") == "older_version":  # reconciled: the bytes this case reaches changed since
        origin += f" older_version={str(graded.get('graded_source_sha256'))[:12]}"
    withheld = trace_withheld(row)
    trace = f" trace={withheld}" if withheld else ""
    return f"{row.get('capsule')} {row.get('status')} {_tiers(row)} {failure}{_performance(row)}{trace}{origin}"


def render(document: dict) -> str:
    """The text view of one completed-grade document."""
    from .completed_grade import FILENAME as FULL
    from .completed_grade import POINTER, SUMMARY
    from .failure_clusters import KEY as CLUSTERS

    pointer = document.get(POINTER) if isinstance(document.get(POINTER), dict) else {}
    counts = document.get(SUMMARY) if isinstance(document.get(SUMMARY), dict) else {}
    rows = [row for row in document.get("per_capsule") or () if isinstance(row, dict) and row.get("capsule")]
    from .performance_headline import KEY as HEADLINE
    from .performance_headline import render as render_headline

    lines = [
        *render_headline(document.get(HEADLINE)),
        f"# One line per test case from qa/{FULL} (rewritten with it). Full rows: each per_capsule detail.path;",
        f"# failure groups: {CLUSTERS}; ranked performance: feedback_brief.performance_priority.",
        f"newest grade token={pointer.get('token')} graded_at={pointer.get('graded_at')} "
        f"source_sha256={pointer.get('graded_source_sha256')} snapshot_current={document.get('snapshot_current')}",
        f"current_source_sha256={counts.get('current_source_sha256')} rows={counts.get('rows', len(rows))} "
        f"executed={counts.get('executed')} passed={counts.get('passed')} "
        f"from_newest_grade={counts.get('from_newest_grade')} from_current_source={counts.get('from_current_source')}",
        f"reconciled with your current source: current_passed={counts.get('current_passed')} "
        f"current_rows={counts.get('current_rows')} older_version_rows={counts.get('older_version_rows')} "
        f"(older_version = a result for the older source it names; rerun queued on the newest grade)",
        f"newest grade only: n_passed={document.get('n_passed')} n_capsules={document.get('n_capsules')} "
        f"all_pass={document.get('all_pass')} tiers_graded={document.get('tiers_graded')} "
        f"tiers_not_run={document.get('tiers_not_run')}",
        "",
    ]
    section = document.get(CLUSTERS) if isinstance(document.get(CLUSTERS), dict) else None
    clusters = (section or {}).get("clusters") or []
    lines.append(
        f"## {CLUSTERS}: {(section or {}).get('clusters_total', 0)} total, "
        f"{(section or {}).get('failing_test_cases', 0)} failing test cases"
    )
    for index, cluster in enumerate(clusters, 1):
        signature = f"{cluster.get('failure_plane')}/{cluster.get('failure_category')}"
        detail = _text(cluster.get("example_failure_detail"))
        lines += [
            f"{index}. {cluster.get('count')} case(s) {signature}: {detail}",
            f"   reproduce: {cluster.get('reproduce')}",
            f"   members: {', '.join(str(name) for name in cluster.get('test_cases') or ())}",
        ]
    if (section or {}).get("clusters_not_listed"):
        lines.append(f"({section['clusters_not_listed']} smaller cluster(s) not listed; see per_capsule)")
    withheld = [(row, trace_withheld(row)) for row in rows if row.get("status") == "pass" and trace_withheld(row)]
    if withheld:
        lines += [
            "",
            f"## trace evidence: {len(withheld)} passing case(s) lack the structural trace evidence the "
            "final closure requires (trace_violations, trace_unavailable_requirements in per_capsule)",
            *(f"- {row.get('capsule')}: {why}" for row, why in withheld),
        ]
    lines += [
        "",
        "## test cases: name status tiers failure_plane/category [perf=status cycles=N] "
        "[graded=token when not the newest grade]",
    ]
    lines += [_case_line(row, pointer.get("token")) for row in rows]
    return "\n".join(lines) + "\n"


def write(workspace, document: dict) -> Path | None:
    """Rewrite the view beside the full file; a failure leaves the full file untouched."""
    path = Path(workspace) / "qa" / FILENAME
    try:
        temporary = path.with_suffix(".next.txt")
        temporary.write_text(render(document))
        temporary.replace(path)
    except Exception as error:  # noqa: BLE001 -- a view must never fail the grade it mirrors
        print(f"[first-grade] grade summary unavailable: {type(error).__name__}", flush=True)
        return None
    return path
