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

import hashlib
import json
from pathlib import Path


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
    ms = [e.get("mismatch_count") for e in (v.get("per_capsule") or [])
          if isinstance(e.get("mismatch_count"), int)]
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
        return []                                          # nothing failing → no focus needed
    L = ["## Focus THIS round — spend your turns on the FAILING capsules", ""]
    for e in failing:
        pl = e.get("failure_plane") or e.get("plane") or (e.get("first_failure_plane")) or "?"
        mc = e.get("mismatch_count")
        L.append(f"- `{e.get('capsule')}` — {_st(e)}, plane `{pl}`"
                 + (f", mismatch_count {mc}" if isinstance(mc, int) else ""))
    L += ["",
          f"{len(passing)} capsule(s) already PASS. Do NOT re-work them — but do NOT delete or break the "
          "code that makes them pass (a regression shows up as the lowest mismatch getting worse above). "
          "Put this round's effort into the failing set listed here."]
    if passing:
        L += ["", "Passing (locked): " + ", ".join(f"`{c}`" for c in passing) + "."]
    return L + [""]


def build(run_dir: Path, ws: Path, rnd: int, *, notes_stale: bool = False) -> str:
    """Markdown brief for the round ABOUT TO START (``rnd`` = the round just graded)."""
    verdicts = _load_verdicts(run_dir)
    L = ["# Round brief — your progress so far (READ THIS FIRST)", "",
         "You are a FRESH session. Your ONLY memory of prior rounds is this brief, your `submission/` "
         "code, and `docs/iteration_notes.md`. Build on them — do not re-derive what you already worked "
         "out, and do NOT undo a change that improved a previous round.", ""]

    rows = [_row(n, v) for n, v in verdicts]
    if rows:
        L += ["## Progress log (from the official grader — redacted, no goldens)", "",
              "| round | passed | failure planes | lowest mismatch_count |",
              "|---|---|---|---|"]
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
                    delta.append(f"⚠️ lowest mismatch got WORSE ({pbest} → {cbest}) — you likely "
                                 "REGRESSED; diff against the code that produced the better round and "
                                 "restore what worked")
                else:
                    delta.append(f"lowest mismatch improved ({pbest} → {cbest}) — continue this direction")
            if delta:
                L += ["**Since last round:** " + "; ".join(delta) + ".", ""]

    if verdicts:
        L += _focus_section(verdicts[-1][1])

    notes = ws / "submission" / "docs" / "iteration_notes.md"
    body = notes.read_text(encoding="utf-8", errors="replace") if notes.is_file() else ""
    if body.strip():
        L += ["## Your iteration_notes.md (carried from prior rounds)", "",
              "```markdown", body[-4000:], "```", ""]
        if notes_stale:
            L += ["> ⚠️ You did NOT update `docs/iteration_notes.md` last round — you are losing your own "
                  "findings across the round boundary. THIS ROUND: after each change, append what you "
                  "changed, what the verdict showed, and your next hypothesis.", ""]
    else:
        L += ["## Your iteration_notes.md is EMPTY", "",
              "Create `submission/docs/iteration_notes.md` and append to it EVERY round: what you changed, "
              "what the grader verdict said, and your next hypothesis. It is your only cross-round memory.",
              ""]
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
    qa = ws / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    stamp = qa / ".notes_hash"
    cur = _notes_hash(ws)
    prev = stamp.read_text().strip() if stamp.is_file() else ""
    stale = bool(prev) and bool(cur) and prev == cur          # unchanged since last round → not journaling
    p = qa / "round_brief.md"
    p.write_text(build(run_dir, ws, rnd, notes_stale=stale), encoding="utf-8")
    stamp.write_text(cur)
    return p
