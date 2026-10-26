"""Cross-round MEMORY brief for the agentic loop.

Each round is a FRESH agent session — its only memory of prior rounds is what lives on disk (its
``submission/`` code + ``docs/iteration_notes.md``) plus whatever the harness injects. Left to itself a
weaker model lets its own notes go stale and re-derives (or undoes) what it already figured out. This
module builds a compact **round brief** the harness writes every round and feeds into the next one, so
cross-round learning does not depend on the model remembering to journal.

Target-agnostic, oracle-free, leaks NO golden: it is assembled only from the REDACTED per-round verdicts
the grader already wrote (``run_dir/qa_history/verdict_round_*.json`` — pass/fail + failure plane +
mismatch COUNT, never values) and the agent's OWN notes. No target name, no ``re``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

#: The brief's own lock in ``qa/``: every writer of ``round_brief.md`` holds it (round end, pre-launch,
#: the background refresh after each published grade, the RESUME banner) and replaces the file
#: atomically, so a reader never sees a half-written brief and no writer drops another's banner.
LOCK = ".round-brief.lock"
FILENAME = "round_brief.md"


@contextmanager
def locked(ws: Path):
    qa = Path(ws) / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    with (qa / LOCK).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield qa / FILENAME
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def replace_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a sibling temporary file and an atomic rename."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _headline_section(ws: Path) -> list[str]:
    """The newest grade's performance headline, first in the brief (:mod:`.performance_headline`)."""
    from .performance_headline import load, render

    try:
        return render(load(ws))
    except Exception:  # noqa: BLE001 -- the rest of the brief stands without it
        return []


def _load_verdicts(run_dir: Path) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    d = run_dir / "qa_history"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("verdict_round_*.json")):
        try:
            out.append((int(f.stem.split("_")[-1]), json.loads(f.read_text())))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
    return out


def _row(n: int, v: dict) -> tuple:
    planes = v.get("first_failure_planes") or {}
    ms = [e.get("mismatch_count") for e in (v.get("per_capsule") or []) if isinstance(e.get("mismatch_count"), int)]
    best = min(ms) if ms else None
    plane_s = ", ".join(f"{k}:{val}" for k, val in planes.items()) or "-"
    return (n, v.get("n_passed"), v.get("n_capsules"), plane_s, best)


def _focus_section(v: dict) -> list[str]:
    """From the latest redacted verdict, a crisp FAILING-set focus list + the PASSING set as locked-don't-
    break. Prioritization only (the grader still grades every capsule; a regression on a passing one still
    surfaces via the mismatch delta) — it does not restrict what the agent may touch. Golden-free: uses only
    the per-capsule status/plane/mismatch_count the grader already redacts."""
    per = v.get("per_capsule") or []
    if not per:
        return []

    def _st(e):
        return str(e.get("status", "")).lower()

    passing = [e.get("capsule") for e in per if _st(e) == "pass" and e.get("capsule")]
    failing = [e for e in per if _st(e) and _st(e) != "pass"]
    if not failing:
        return []  # nothing failing → no focus needed
    L = ["## Correctness repair — shared failing boundaries", ""]
    for e in failing:
        pl = e.get("failure_plane") or e.get("plane") or (e.get("first_failure_plane")) or "?"
        mc = e.get("mismatch_count")
        L.append(
            f"- `{e.get('capsule')}` — {_st(e)}, plane `{pl}`"
            + (f", mismatch_count {mc}" if isinstance(mc, int) else "")
        )
    L += [
        "",
        f"{len(passing)} capsule(s) already PASS on the recorded source. Preserve their numerical behavior "
        "while improving reusable passes and performance alongside correctness repair. Retain a working "
        "checkpoint and check affected boundaries before integrating changes.",
    ]
    if passing:
        L += ["", "Recorded passing cases: " + ", ".join(f"`{c}`" for c in passing) + "."]
    return L + [""]


def _passing(v: dict) -> set:
    return {
        e.get("capsule")
        for e in (v.get("per_capsule") or [])
        if str(e.get("status", "")).lower() == "pass" and e.get("capsule")
    }


def _public_build_section(run_dir: Path) -> list[str]:
    """Pair the latest archived round with its exact-key public advisory.

    Continuous rounds use keys such as ``r0000_t000007``. The historical
    numeric progress reader intentionally ignores those keys, but an advisory
    must still follow the latest actual archived candidate, never an older
    projection with a coincidentally matching numeric round.

    A record without a timezone-aware ``graded_at`` was not stamped by a
    grader that orders continuous rounds (the historical numeric rounds), so
    it takes no part in the ordering. An archived verdict that cannot be READ
    is different: it may be the newest round, and skipping it would pair the
    advisory with an older candidate. The advisory is then withheld, and the
    section names the records that stopped it.
    """
    history = run_dir / "qa_history"
    rounds = []
    unreadable: list[str] = []
    for path in sorted(history.glob("verdict_round_*.json")):
        try:
            verdict = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            unreadable.append(f"`{path.name}` ({type(exc).__name__})")
            continue
        graded_at = verdict.get("graded_at") if isinstance(verdict, dict) else None
        if not isinstance(graded_at, str):
            continue
        try:
            stamp = datetime.fromisoformat(graded_at)
        except ValueError:
            unreadable.append(f"`{path.name}` (graded_at {graded_at!r} is not ISO-8601)")
            continue
        if stamp.tzinfo is None:
            continue
        rounds.append((stamp, path.name, verdict))
    if unreadable:
        return [
            "## Public object-build scalability (advisory only)",
            "",
            "Withheld: the latest graded round cannot be determined because these archived verdicts "
            "could not be read: " + ", ".join(unreadable) + ".",
            "",
        ]
    if not rounds:
        return []
    _stamp, name, verdict = max(rounds)
    candidate = verdict.get("graded_candidate_sha256")
    if not isinstance(candidate, str) or len(candidate) != 64:
        return []
    key = name.removeprefix("verdict_round_").removesuffix(".json")
    path = history / f"codegen_scalability_public_round_{key}.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if (
        not isinstance(record, dict)
        or record.get("schema") != "merlin.public_object_build_feedback.v1"
        or record.get("candidate_sha256") != candidate
        or record.get("scope") != "public_emit_and_build_only_advisory"
    ):
        return []
    rows = record.get("samples")
    if record.get("ran") is not True or not isinstance(rows, list) or len(rows) != 4:
        return []
    lines = [
        "## Public object-build scalability (advisory only)",
        "",
        "These public fact-derived samples compile emitted objects only; they do not execute or certify them.",
        "",
        "| scale | emit | object build | emitted bytes | object bytes | compile seconds |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row, scale in zip(rows, (1, 2, 4, 8), strict=True):
        if not isinstance(row, dict) or row.get("extent_scale") != scale:
            return []
        emit, status = row.get("emit_outcome"), row.get("build_status")
        if emit not in ("lowered", "empty", "declined", "error") or status not in (
            "compiled",
            "compile_error",
            "not_emitted",
            "selection_changed",
            "source_exceeds_probe_budget",
        ):
            return []
        size = row.get("emitted_artifact_bytes")
        obj = row.get("object_bytes")
        wall = row.get("compile_wall_s")
        if any(type(value) is not int or value < 0 for value in (size, obj) if value is not None):
            return []
        if wall is not None and (type(wall) not in (int, float) or not 0 <= wall < 1_000_000):
            return []
        lines.append(
            f"| {scale} | {emit} | {status} | {size if size is not None else '-'} | "
            f"{obj if obj is not None else '-'} | {wall if wall is not None else '-'} |"
        )
    return lines + [""]


def _regression_section(verdicts: list[tuple[int, dict]]) -> list[str]:
    """Name the capsules LOST since the best round, and which round to diff against.

    The existing delta line watches the lowest mismatch_count, which can hold steady (or improve) while
    whole capsules are lost -- a round that trades five passing capsules for progress on one shows up as
    "improved". Measured: a run went 38/40 -> 33/40 while chasing two capsules that were failing for a
    harness reason, and nothing in the brief said "you dropped these five"; the agent never restored them.
    Pass count is the graded quantity, so regressions in it are the loudest thing a brief can carry.
    Redacted-verdict-only (status + capsule name), so no golden and no target name."""
    if len(verdicts) < 2:
        return []
    cur_n, cur_v = verdicts[-1]
    best_n, best_v = max(verdicts, key=lambda nv: len(_passing(nv[1])))
    lost = sorted(_passing(best_v) - _passing(cur_v))
    if not lost or best_n == cur_n:
        return []
    return [
        f"## ⚠️ REGRESSION — you are {len(lost)} capsule(s) BELOW your best round (round {best_n})",
        "",
        f"Round {best_n} passed {len(_passing(best_v))}; round {cur_n} passed {len(_passing(cur_v))}. "
        "These passed then and fail now:",
        "",
        *[f"- `{c}`" for c in lost],
        "",
        "Recovering these is worth MORE than new work: they are known-solvable by code you already "
        "wrote. Your `submission/` tree is the one that regressed, so read "
        "`docs/iteration_notes.md` for what you changed after the good round and undo the part that "
        "broke them. If a change was needed for something else, make it ADDITIVE instead of a "
        "replacement.",
        "",
    ]


def _errata_section(run_dir: Path) -> list[str]:
    """Operator-authored RETRACTIONS of this harness's own prior output, if any.

    Exists because a harness advisory can be WRONG, and a fresh session has no way to know it. Measured
    2026-09-01 on merlincirct_arm4_func_20260901_codex1: the `out_stride_bytes` advisory derived its
    `intended_value` from a packed row stride that did not match how the graded harness allocates the
    buffer. The agent emitted the correct value, the advisory contradicted it, and the agent complied --
    recording the change in `docs/iteration_notes.md`, where it then reads as settled fact to every later
    round. Correcting the check is not enough on a resumed run: the false premise is already written into
    the agent's own carried memory, and the brief is the only channel that can retract it.

    Deliberately NOT a hint channel. An erratum states that a specific piece of OUR output was wrong and
    must be re-derived; it never supplies the value. Placed before the progress log so it is read ahead of
    the notes it contradicts.
    """
    errata = run_dir / "ERRATA.md"
    if not errata.is_file():
        return []
    body = errata.read_text(encoding="utf-8", errors="replace").strip()
    if not body:
        return []
    return [body, ""]


def build(run_dir: Path, ws: Path, rnd: int, *, notes_stale: bool = False) -> str:
    """Markdown brief for the round ABOUT TO START (``rnd`` = the round just graded)."""
    from .capability_brief import build_if_selected

    headline = _headline_section(ws)
    joint = build_if_selected(run_dir, ws, errata=_errata_section(run_dir), notes_stale=notes_stale)
    if joint is not None:
        return "\n".join(headline) + ("\n" if headline else "") + joint
    verdicts = _load_verdicts(run_dir)
    L = [
        *headline,
        "# Round brief — your progress so far (READ THIS FIRST)",
        "",
        "You are a FRESH session. Your ONLY memory of prior rounds is this brief, your `submission/` "
        "code, and `docs/iteration_notes.md`. Build on them — do not re-derive what you already worked "
        "out, and do NOT undo a change that improved a previous round.",
        "",
    ]

    L += _errata_section(run_dir)

    rows = [_row(n, v) for n, v in verdicts]
    if rows:
        L += [
            "## Progress log (from the official grader — redacted, no goldens)",
            "",
            "| round | passed | failure planes | lowest mismatch_count |",
            "|---|---|---|---|",
        ]
        for n, npass, ncap, planes, best in rows:
            L.append(f"| {n} | {npass}/{ncap} | {planes} | {best if best is not None else '-'} |")
        L.append("")
        if len(rows) >= 2:
            (_, _, _, pplanes, pbest), (_, _, _, cplanes, cbest) = rows[-2], rows[-1]
            delta = []
            if pplanes != cplanes:
                delta.append(f"failure planes moved `{pplanes}` → `{cplanes}`")
            if isinstance(pbest, int) and isinstance(cbest, int) and cbest != pbest:
                if cbest > pbest:
                    delta.append(
                        f"⚠️ lowest mismatch got WORSE ({pbest} → {cbest}) — you likely "
                        "REGRESSED; diff against the code that produced the better round and "
                        "restore what worked"
                    )
                else:
                    delta.append(f"lowest mismatch improved ({pbest} → {cbest}) — continue this direction")
            if delta:
                L += ["**Since last round:** " + "; ".join(delta) + ".", ""]
        L += _regression_section(verdicts)

    if verdicts:
        from .failure_clusters import build as clusters
        from .failure_clusters import render as render_clusters

        L += _focus_section(verdicts[-1][1])
        L += render_clusters(clusters(verdicts[-1][1].get("per_capsule"), workspace=ws))
    L += _public_build_section(run_dir)

    notes = ws / "submission" / "docs" / "iteration_notes.md"
    body = notes.read_text(encoding="utf-8", errors="replace") if notes.is_file() else ""
    if body.strip():
        L += ["## Your iteration_notes.md (carried from prior rounds)", "", "```markdown", body[-4000:], "```", ""]
        if notes_stale:
            L += [
                "> ⚠️ You did NOT update `docs/iteration_notes.md` last round — you are losing your own "
                "findings across the round boundary. THIS ROUND: after each change, append what you "
                "changed, what the verdict showed, and your next hypothesis.",
                "",
            ]
    else:
        L += [
            "## Your iteration_notes.md is EMPTY",
            "",
            "Create `submission/docs/iteration_notes.md` and append to it EVERY round: what you changed, "
            "what the grader verdict said, and your next hypothesis. It is your only cross-round memory.",
            "",
        ]
    return "\n".join(L) + "\n"


def _notes_hash(ws: Path) -> str:
    notes = ws / "submission" / "docs" / "iteration_notes.md"
    if not notes.is_file():
        return ""
    txt = notes.read_text(encoding="utf-8", errors="replace")
    return hashlib.sha256(txt.encode("utf-8")).hexdigest() if txt.strip() else ""


def write(run_dir: Path, ws: Path, rnd: int) -> Path:
    """Build + persist ``ws/qa/round_brief.md`` for the next round. Detects whether the agent updated its
    notes during the round just graded (content hash vs the stamp left last round) and flags staleness."""
    with locked(ws) as p:
        stamp = p.parent / ".notes_hash"
        cur = _notes_hash(ws)
        prev = stamp.read_text().strip() if stamp.is_file() else ""
        stale = bool(prev) and bool(cur) and prev == cur  # unchanged since last round → not journaling
        replace_text(p, build(run_dir, ws, rnd, notes_stale=stale))
        stamp.write_text(cur)
    return p


_RESUME_NOTE_BEGIN = "<!-- merlin:resume-note:begin -->"
_RESUME_NOTE_END = "<!-- merlin:resume-note:end -->"


def _leading_resume_note(existing: str) -> str:
    """Return the one harness-owned RESUME prefix, including legacy unmarked briefs.

    ``write`` intentionally replaces a completed round's old banner. This parser is only used by the
    pre-launch refresh: there the banner describes the partial turn about to resume and must survive
    rebuilding the rest of the brief from current operator input.
    """
    if existing.startswith("> ## RESUME") and existing.find(_RESUME_NOTE_BEGIN) >= 0:
        end = existing.find(_RESUME_NOTE_END, existing.find(_RESUME_NOTE_BEGIN))
        if end >= 0:
            end += len(_RESUME_NOTE_END)
            return existing[:end].rstrip() + "\n\n"
    # Compatibility for briefs written before the marker pair existed. A generated brief starts at the
    # first top-level Round brief heading, so this cannot absorb iteration notes or arbitrary markdown.
    if existing.startswith("> ## RESUME"):
        # A refreshed brief opens with the performance headline; it belongs to the rebuilt part.
        ends = [at for at in (existing.find("\n# Round brief"), existing.find("\n## Performance headline")) if at >= 0]
        if ends:
            return existing[: min(ends)].rstrip() + "\n\n"
    return ""


def refresh_before_launch(run_dir: Path, ws: Path, rnd: int) -> Path:
    """Refresh mutable harness guidance immediately before an agent turn.

    TASK.md and treatment inputs stay sealed across ``--resume``. The round brief is deliberately an
    operator/harness feedback channel, so a late ``ERRATA.md`` must be visible even if no new grade ran.
    Preserve only a recognized leading RESUME banner and rebuild everything else. Unlike :func:`write`,
    this does not advance the notes-staleness stamp merely because the driver relaunched. The
    background grader calls it after each published grade too (:mod:`.brief_refresh`).
    """
    with locked(ws) as p:
        existing = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        prefix = _leading_resume_note(existing)
        replace_text(p, prefix + build(run_dir, ws, rnd))
    return p


def verified_status_line(verdict: dict, *, pass_line: str) -> str:
    gate = verdict.get("model_gate")
    if not isinstance(gate, dict):
        return pass_line
    from ..brokers.model_check import PROGRAMS

    rows = gate.get("programs") or {}
    names = [f"{model}/{program}" for model, programs in PROGRAMS.items() for program in programs]
    statuses = [f"{name}={((rows.get(name) or {}).get('status') or 'pending')}" for name in names]
    return "Visible models: " + "; ".join(statuses) + ". Sealed official grade pending."


def stamp_report_status(report: Path, pass_line: str, *, status_lines: tuple[str, ...]) -> bool:
    """Last-resort guarantee: ensure REPORT.md's final status line states the VERIFIED result. The
    multi-round relaunch grades AFTER the agent exits, so the converging round's agent never saw a
    passing verdict and may leave a stale 'not yet passing' line. Returns True if it had to rewrite."""
    if not report.exists():
        report.write_text(f"# REPORT\n\n## Final status line\n{pass_line}\n")
        return True
    txt = report.read_text()
    if pass_line in txt:
        return False
    new = txt
    for sl in status_lines:
        if sl != pass_line and sl in new:
            new = new.replace(sl, pass_line)
    if new == txt:  # no known status line present -> append one
        new = txt.rstrip() + f"\n\n## Final status line (verified)\n{pass_line}\n"
    report.write_text(new)
    return True
