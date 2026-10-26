"""Group failing test cases by one redacted failure signature, so each group is a unit of work.

The verdict lists failures one row per test case. Measured on a live joint run: 44 rows carried the
same elaboration error (only the generated loop name differed: ``tile5``, ``tile3``, ``tile2``), an
earlier grade had 296 rows failing one link error whose text differed only in the per-case object
path and a code offset, and three rows were vetoed for host compute. The operator console showed those
groups; the author, reading rows, kept one or two helpers busy while sixteen were available. A group
that one fix can clear is exactly what a lead can hand to one helper, with one case to reproduce it.

A signature is ``(failure_plane, failure_category, normalized failure_detail)``. The detail is the
grader's already redacted text (numbers scrubbed to ``#``); normalization compares it structurally:
whitespace runs are one separator, a token naming a path keeps only its last component (per-case work
directories differ), the row's own test-case name is replaced, and every digit run -- decimal or a
``0x`` offset -- becomes ``#`` so ``tile5`` and ``tile3`` agree. A detail cut at the redaction cap,
or clipped in the middle by the diagnostic clipper, loses the partial tokens at the cut and joins the
signature it is part of: a longer case name leaves less of the same message before the cut.

Only rows that carry a failed result are grouped (a gated, deferred or not-run row says why it did
not run, not what failed). With a public capability inventory in the workspace, only its members are
grouped, the same names the joint brief may show; without one, the rows are the verdict's own,
already visible to the author. The section is information only: ordering is by size, and nothing here
assigns, requires or limits work.
"""

from __future__ import annotations

import json
from pathlib import Path

KEY = "failure_clusters"
SCHEMA = "merlin.phase1.failure_clusters.v1"
#: Clusters listed in full. Bounds the published file; the remaining clusters are counted.
LIMIT = 12
EXAMPLES = 3
_CASE = "<case>"
_HEX = frozenset("0123456789abcdefABCDEF")
NOTE = ("Failing test cases grouped by the same redacted failure (plane, category and detail compared "
        "with numbers, paths and the case's own name normalized). One fix often clears a whole cluster, "
        "so each cluster is a self-contained unit a helper can take: its member names, one example's "
        "detail and a command that reproduces it. Ordered by size; per-case rows stay in `per_capsule`.")


def _no_result_statuses() -> frozenset:
    from .completed_grade import NO_RESULT

    return NO_RESULT | {"pass"}


def _detail_cap() -> int:
    from .qa import DETAIL_LIMIT

    return DETAIL_LIMIT


def _numbers(token: str) -> str:
    """Every run of digits (a ``0x`` offset included) as ``#``; adjacent markers merge."""
    out: list[str] = []
    i, n = 0, len(token)
    while i < n:
        ch = token[i]
        if ch.isascii() and ch.isdigit():
            j = i + 1
            if ch == "0" and j < n and token[j] in "xX" and j + 1 < n and token[j + 1] in _HEX:
                j += 1
                while j < n and token[j] in _HEX:
                    j += 1
            while j < n and token[j].isascii() and token[j].isdigit():
                j += 1
            if not (out and out[-1] == "#"):
                out.append("#")
            i = j
            continue
        if not (ch == "#" and out and out[-1] == "#"):
            out.append(ch)
        i += 1
    return "".join(out)


def _token(token: str) -> str:
    if "/" in token:
        # A path names one case's work tree; its last component says what the message is about.
        stripped = token.rstrip("/")
        token = stripped.rsplit("/", 1)[-1] if stripped else token
    return _numbers(token)


def _elision() -> str:
    from merlin.targetgen.capsule_diagnostics import ELISION

    return ELISION


def normalize(detail, case: str | None = None) -> tuple[str, bool]:
    """``(structural form of a redacted detail, whether the grader cut it at its cap)``.

    A diagnostic clipped in the middle keeps its elision marker; the partial tokens on both sides of
    it are dropped, since where the clip fell depends on the case's own path and name lengths.
    """
    text = str(detail or "")
    cut = len(text) >= _detail_cap()
    marker = _elision()
    case_form = _numbers(case) if case else ""
    tokens = []
    for raw in text.split():
        token = _token(raw.replace(case, _CASE) if case else raw)
        tokens.append(token.replace(case_form, _CASE) if case_form else token)
    if cut and tokens:
        tokens = tokens[:-1]
    if marker in tokens:
        at = tokens.index(marker)
        tokens = tokens[:max(at - 1, 0)] + [marker] + tokens[at + 2:]
    return " ".join(tokens), cut


def _covers(host: str, partial: str, cut: bool) -> bool:
    """Whether a complete signature ``host`` is the message ``partial`` was clipped or cut from."""
    head, marker, tail = partial.partition(f" {_elision()} ")
    if not head or len(host) <= len(partial) or not host.startswith(head):
        return False
    if not marker:
        return cut
    rest = host[len(head):]
    return (tail in rest) if cut else rest.endswith(tail)


def _failed(row) -> bool:
    return (isinstance(row, dict) and isinstance(row.get("capsule"), str)
            and row.get("status") not in _no_result_statuses())


def _plane(row) -> str:
    return str(row.get("failure_plane") or row.get("first_failure_plane") or row.get("plane") or "unknown")


def _category(row) -> str:
    return str(row.get("failure_category") or row.get("failure_code") or row.get("status") or "unknown")


def _inventory(workspace) -> dict | None:
    from ..capability_inventory import FILENAME, SCHEMA as INVENTORY_SCHEMA

    if workspace is None:
        return None
    try:
        document = json.loads((Path(workspace) / FILENAME).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != INVENTORY_SCHEMA:
        return None
    members = document.get("members")
    return members if isinstance(members, dict) else None


def _tier_status(row, tier: str):
    value = (row.get("tiers") or {}).get(tier)
    return value.get("status") if isinstance(value, dict) else value


def reproduce_command(row, member: dict | None = None) -> str:
    """The ordinary self-check that re-runs this one test case on the engine where it failed."""
    from ..capability_inventory import SCALABILITY_COHORT

    name = row["capsule"]
    suite = " --suite scalability" if (member or {}).get("cohort") == SCALABILITY_COHORT else ""
    engines = (member or {}).get("development_engines")
    certifying_only = _tier_status(row, "L3") == "fail" and _tier_status(row, "L2") == "pass"
    sim = " --sim gsim" if certifying_only and (engines is None or "gsim" in engines) else ""
    return f"python agent_selfcheck.py --submission submission{suite} --capsules {name}{sim} --wait-budget 5"


def _merge_partial(groups: dict, partial: dict) -> dict:
    """Fold each cut or clipped signature into the largest group whose message it is part of."""
    for key in sorted(partial, key=lambda key: len(key[2])):
        hosts = [other for other in groups if other != key and other[:2] == key[:2]
                 and _covers(other[2], key[2], partial[key])]
        if key in groups and hosts:
            host = max(hosts, key=lambda other: (other not in partial, len(groups[other]), other[2]))
            groups[host] += groups.pop(key)
    return groups


def build(rows, *, workspace=None, limit: int = LIMIT) -> dict | None:
    """The cluster section for a verdict's ``per_capsule`` rows; None when no row failed."""
    members = _inventory(workspace)
    failed = [row for row in rows or () if _failed(row) and (members is None or row["capsule"] in members)]
    if not failed:
        return None
    groups: dict[tuple, list] = {}
    partial: dict[tuple, bool] = {}
    for row in failed:
        detail, cut = normalize(row.get("failure_detail"), row["capsule"])
        key = (_plane(row), _category(row), detail)
        groups.setdefault(key, []).append(row)
        if cut or f" {_elision()} " in f" {detail} ":
            partial[key] = partial.get(key, False) or cut
    groups = _merge_partial(groups, partial)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1], item[0][2]))
    clusters = []
    for (plane, category, _), members_rows in ordered[:limit]:
        members_rows = sorted(members_rows, key=lambda row: row["capsule"])
        example = members_rows[0]
        entry = {
            "count": len(members_rows),
            "failure_plane": plane,
            "failure_category": category,
            "examples": [row["capsule"] for row in members_rows[:EXAMPLES]],
            "test_cases": [row["capsule"] for row in members_rows],
            "example_failure_detail": example.get("failure_detail"),
            "reproduce": reproduce_command(example, (members or {}).get(example["capsule"])),
        }
        graded = [str((row.get("graded") or {}).get("graded_at")) for row in members_rows
                  if isinstance(row.get("graded"), dict) and (row["graded"].get("graded_at"))]
        if graded:
            entry["newest_graded_at"] = max(graded)
        if workspace is not None and (Path(workspace) / "selfcheck_out" / example["capsule"]
                                      / "command_buffer.json").is_file():
            entry["inspect"] = (f"python isa_tools.py debug selfcheck_out/{example['capsule']}/command_buffer.json"
                                f" --capsule {example['capsule']}")
        clusters.append(entry)
    return {
        "schema": SCHEMA,
        "failing_test_cases": len(failed),
        "clusters_total": len(groups),
        **({"clusters_not_listed": len(groups) - len(clusters)} if len(groups) > len(clusters) else {}),
        "clusters": clusters,
        "note": NOTE,
    }


def attach(workspace, verdict: dict) -> dict:
    """Set (or clear) the verdict's cluster section from its own rows; never fails a verdict."""
    try:
        section = build(verdict.get("per_capsule"), workspace=workspace)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return verdict
    if section is None:
        verdict.pop(KEY, None)
    else:
        verdict[KEY] = section
    return verdict


def _brief_text(value, limit: int) -> str:
    text = " ".join(str(value or "").split()).replace("`", "'")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render(section: dict | None, *, detail_chars: int = 240) -> list[str]:
    """Markdown lines for a round brief; empty when nothing failed."""
    if not isinstance(section, dict) or not section.get("clusters"):
        return []
    lines = ["## Failure clusters: test cases that fail the same way", "",
             "Each cluster below shares one redacted failure, so one fix often clears all of it. A cluster "
             "is a self-contained unit you can hand to a helper with its reproduce command; full member "
             f"lists are under `{KEY}` in `qa/verdict.json`.", ""]
    for cluster in section["clusters"]:
        examples = ", ".join(f"`{name}`" for name in cluster["examples"])
        lines += [f"- {cluster['count']} case(s), `{cluster['failure_plane']}` / `{cluster['failure_category']}`: "
                  f"e.g. {examples}.",
                  f"  Detail: {_brief_text(cluster.get('example_failure_detail'), detail_chars)}",
                  f"  Reproduce: `{cluster['reproduce']}`"]
    if section.get("clusters_not_listed"):
        lines.append(f"- {section['clusters_not_listed']} smaller cluster(s) not listed; see `per_capsule`.")
    return lines + [""]
