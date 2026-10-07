#!/usr/bin/env python3
"""Gate: on an evidence path, absent must not read as a value.

The recurring defect in this repo is a reading that CANNOT DISCRIMINATE the cases it is used to
decide, whose ambiguous answer is the PASSING one. The cheapest way to build one is a silent default:
the datum is missing, the code substitutes something, and the verdict is computed from the substitute
with nothing recording that it was substituted. Four shapes have each produced an incident:

* ``.get("explicit_cycles", 0)`` -- a measurement reader that cannot tell *absent* from *zero*. The
  zero flowed into a ratio and the ratio read as a result.
* ``except ...: pass`` around a load -- a file that failed to parse and a file that said nothing
  become the same empty answer, and the empty answer is the clean one.
* a ``subprocess`` result nobody inspects -- "the command ran" and "the command was killed" are the
  same silence. A timed-out run's empty log was read as a clean log.
* ``shell=True`` over a pipeline without ``pipefail`` -- the shell reports the LAST stage's status, so
  a producer that died mid-pipe exits 0. This is the same defect one layer down.

None of these is wrong everywhere. ``config.get("mode", "auto")`` is a default, not a silent one,
because a wrong answer changes a preference rather than a verdict. So this gate is scoped to the
paths where a wrong answer BECOMES a verdict -- gates, graders, provenance, measurement readers --
declared in ``EVIDENCE_ROOTS`` below and extended by a reviewed edit to it, never by the scan.

What the gate does NOT claim: it cannot see a constant a producer writes that a consumer then trusts
(``cycle_accurate: true``), and it cannot see an ambiguity built from control flow rather than from a
default. Those need their own checks; this one is deliberately about the four shapes above, each
detected STRUCTURALLY over the ``ast`` (a text search cannot tell an ``except: pass`` from the words
in a docstring, and this repo does not use regex).

Known debt lives in ``silent_default_ratchet.txt`` beside this file, one ``<path>::<scope>::<shape>::
<subject>`` key per line (a repeated subject in one scope takes a ``~2`` ordinal) -- keyed by the SUBJECT, not the line number, so the ledger survives an edit
above it. It may only shrink (``check_ratchets_shrink.py`` holds every ``*_ratchet.txt``), and an
entry whose finding is gone fails this gate until it is removed, so the ledger cannot rot into an
allowlist.

    python build_tools/scripts/check_silent_defaults.py           # exit 1 on a new silent default
    python build_tools/scripts/check_silent_defaults.py --list    # every finding, ratcheted or not
    python build_tools/scripts/check_silent_defaults.py --write   # regenerate the ledger (review it)
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "build_tools" / "scripts" / "silent_default_ratchet.txt"

#: Where a wrong answer becomes a verdict. A path enters this list by review, and the gate never
#: widens itself: a scan that decides its own scope would quietly stop covering a file that moved.
#: A directory is scanned whole; a glob or a file is scanned as spelled. Deliberately NOT the whole
#: tree: a plotting script that labels a missing bar "0" is a cosmetic bug, and a ledger that mixes
#: those in with a grader's fabricated cycle count buys nobody anything. Measured 2026-10-05, the
#: roster below covers 101 files and holds 145 findings, all recorded as debt in the ledger. Widening
#: the roster is a reviewed edit HERE, in one place.
EVIDENCE_ROOTS: tuple[str, ...] = (
    # The gates themselves: a gate that cannot tell absent from passing is the defect, recursively.
    "build_tools/scripts/check_*.py",
    # Which hardware a result is about, and where a product was written.
    "src/merlin/common/provenance.py",
    "src/merlin/common/artifacts.py",
    # Verification: the tier verdicts and their oracles.
    "src/merlin/verify",
    # Phase 1: the controller modules that PRODUCE a verdict -- grading, certification, promotion,
    # freeze, the audit and the conformance check -- not the providers, brokers or task staging that
    # merely carry one.
    "packages/merlin-experiments/src/merlin_experiments/phase1/feedback",
    "packages/merlin-experiments/src/merlin_experiments/phase1/audit.py",
    "packages/merlin-experiments/src/merlin_experiments/phase1/conformance.py",
    "packages/merlin-experiments/src/merlin_experiments/phase1/tools/await_verdict.py",
    # The retained native capsule-bench drivers that still grade or audit a run.
    "merlin/experiments/capsule_bench/harness/conformance.py",
    "merlin/experiments/capsule_bench/harness/freeze_run.py",
    "merlin/experiments/capsule_bench/harness/freeze_state.py",
    "merlin/experiments/capsule_bench/harness/full_suite_audit.py",
    "merlin/experiments/capsule_bench/harness/generalization_difftest.py",
    "merlin/experiments/capsule_bench/harness/grade_agent_run.py",
    "merlin/experiments/capsule_bench/harness/qa_check.py",
    "merlin/experiments/capsule_bench/harness/qa_check_rtlchecks.py",
    "merlin/experiments/capsule_bench/harness/regrade_run_snapshot.py",
    "merlin/experiments/capsule_bench/harness/verify_no_cheat.py",
    # Phase 2: what chains a candidate to a certified whole-model number -- the measured mode's gates
    # and the champion export that records it.
    "packages/merlin-experiments/src/merlin_experiments/phase2/whole_model_measured/gates.py",
    "src/merlin/targetgen/champions.py",
)

#: Not scanned even under an evidence root: the test suite (a test's job is to construct the awkward
#: case, and ``except: pass`` inside one is usually the point), caches and vendored trees.
EXCLUDED_PARTS = ("__pycache__", "tests", "_qa_ws", "_data", "third_party")

#: A default that cannot hide an absence. ``None`` is the explicit "not there"; an empty container
#: iterates to nothing, so a missing key and an empty list reach the same *emptiness*, which the
#: reader can still see. A number, a bool or a non-empty string is a fabricated datum.
_INNOCUOUS_CONSTANTS = (None, "")

#: THE EXEMPTION THAT IS THE POINT. A fallback that itself SAYS "I did not read this" is not a silent
#: default -- it is the third state this repo asks for, and ``d.get("status", "unmeasured")`` is the
#: fix for ``d.get("status", "ok")``, not another instance of it. So a string default drawn from this
#: vocabulary passes, and every other string default is debt. Compared case-folded and stripped of
#: surrounding punctuation, because the same marker is spelled `?`, `UNKNOWN` and `(unknown)`.
_UNKNOWN_MARKERS = frozenset(
    {
        "?",
        "-",
        "--",
        "—",
        "–",
        "n/a",
        "na",
        "nan",
        "unknown",
        "unmeasured",
        "unset",
        "unverified",
        "undeclared",
        "undetermined",
        "unresolved",
        "unattributed",
        "missing",
        "absent",
        "none",
        "not measured",
    }
)

#: Subprocess entry points whose result carries the only evidence that the command worked.
_SUBPROCESS_CALLS = ("run", "call", "Popen")

SHAPES = ("get-default", "swallowed-error", "unchecked-subprocess", "shell-pipe-status")


class Finding(NamedTuple):
    """One reading whose ambiguous answer is the passing one."""

    path: str
    scope: str
    shape: str
    subject: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.scope}::{self.shape}::{self.subject}"

    def explain(self) -> str:
        return {
            "get-default": (
                f"reads {self.subject!r} with a fabricated fallback: an absent datum and a real one "
                f"produce the same value, and the verdict cannot tell them apart. Read it without a "
                f"default and handle the absence explicitly (record UNKNOWN)."
            ),
            "swallowed-error": (
                f"discards a {self.subject} without recording it: a load that FAILED and a load that "
                f"found nothing become the same empty answer, and empty is the passing answer."
            ),
            "unchecked-subprocess": (
                f"never inspects the result of {self.subject}: a command that was killed and a "
                f"command that succeeded leave the same silence. Pass check=True or read .returncode."
            ),
            "shell-pipe-status": (
                f"runs a pipeline through the shell ({self.subject}) with no 'set -o pipefail': the "
                f"exit status is the LAST stage's, so a producer that died mid-pipe reports success."
            ),
        }[self.shape]


def is_unknown_marker(value: str) -> bool:
    """Does this string say 'nobody read this' rather than assert a value?"""
    return value.strip().strip("()[]<>").strip().casefold() in _UNKNOWN_MARKERS


def _is_fabricated(node: ast.expr) -> bool:
    """True when this default substitutes a datum an absent key did not supply."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str) and is_unknown_marker(node.value):
            return False
        return not any(node.value is c or node.value == c for c in _INNOCUOUS_CONSTANTS if type(c) is type(node.value))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.keys)
    # A call, a name or an expression as the default is a deliberate computation, not a fabricated
    # literal; flagging it would drown the ledger in `.get(k, self._derive(k))`, which is a
    # derivation and exactly what this gate wants people to write instead.
    return False


def _render(node: ast.expr | None) -> str:
    """A short, stable name for what the finding is ABOUT.

    Every ledger entry has to name its subject, so a computed key is spelled out rather than reduced
    to "<computed>": two different computed keys in one function would otherwise collide into one
    entry and a fix to either would look like a fix to both. `#` is replaced because the ledger
    format treats it as the start of a comment.
    """
    if node is None:
        return "?"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        spelled = node.value
    elif isinstance(node, ast.Name):
        spelled = node.id
    elif isinstance(node, ast.Attribute):
        spelled = node.attr
    else:
        try:
            spelled = ast.unparse(node)
        except (AttributeError, ValueError):  # pragma: no cover - unparse covers every expr node
            spelled = type(node).__name__
    spelled = " ".join(spelled.split()).replace("#", "<hash>")
    return spelled if len(spelled) <= 48 else spelled[:45] + "..."


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_true(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _callee(call: ast.Call) -> tuple[str, str]:
    """(receiver, attribute) for ``a.b()``; ("", name) for a bare call."""
    if isinstance(call.func, ast.Attribute):
        return _render(call.func.value), call.func.attr
    if isinstance(call.func, ast.Name):
        return "", call.func.id
    return "", ""


def _scope_of(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names: list[str] = []
    cursor: ast.AST | None = node
    while cursor is not None:
        if isinstance(cursor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(cursor.name)
        cursor = parents.get(cursor)
    return ".".join(reversed(names)) or "<module>"


def _statement_of(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    cursor: ast.AST | None = node
    while cursor is not None and not isinstance(cursor, ast.stmt):
        cursor = parents.get(cursor)
    return cursor


def _result_is_inspected(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    """Does anything read the status this call returns?

    Deliberately generous: passing the result on, returning it, or reading any attribute of it all
    count. The gate is about the case where the result goes NOWHERE -- that is the one where a killed
    command is indistinguishable from a finished one.
    """
    statement = _statement_of(call, parents)
    if statement is None:
        return True
    if not isinstance(statement, ast.Expr):
        # Assigned, returned, awaited, passed as an argument, used in a condition: somebody has it.
        return True
    # A bare expression statement: the result is dropped on the floor.
    return False


def findings_for_source(text: str, relpath: str) -> list[Finding]:
    """Every silent-default finding in one file's source. Pure, so a test can mutate the input."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        # Fail CLOSED and loudly rather than returning "no findings": an unparseable evidence file is
        # exactly the case this gate exists to refuse to paper over.
        return [Finding(relpath, "<module>", "swallowed-error", "unparseable source")]
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    raw: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            inert = all(isinstance(s, (ast.Pass, ast.Continue)) for s in body)
            if inert:
                caught = _render(node.type) if node.type is not None else "bare except"
                if isinstance(node.type, ast.Tuple):
                    caught = "/".join(_render(e) for e in node.type.elts)
                raw.append(Finding(relpath, _scope_of(node, parents), "swallowed-error", caught))
        elif isinstance(node, ast.Call):
            receiver, attribute = _callee(node)
            if attribute == "get" and len(node.args) == 2 and _is_fabricated(node.args[1]):
                raw.append(Finding(relpath, _scope_of(node, parents), "get-default", _render(node.args[0])))
            elif attribute in _SUBPROCESS_CALLS and receiver in ("subprocess", "sp"):
                if not _is_true(_keyword(node, "check")) and not _result_is_inspected(node, parents):
                    raw.append(
                        Finding(relpath, _scope_of(node, parents), "unchecked-subprocess", f"subprocess.{attribute}")
                    )
                if _is_true(_keyword(node, "shell")):
                    command = node.args[0] if node.args else None
                    spelled = _render(command)
                    if "|" in spelled and "pipefail" not in spelled:
                        raw.append(
                            Finding(
                                relpath, _scope_of(node, parents), "shell-pipe-status", spelled.split()[0] or "pipeline"
                            )
                        )

    # Two findings of the same shape on the same subject in the same scope get stable ordinals, so a
    # ledger entry keeps naming the same one when a sibling is fixed.
    counted: Counter[str] = Counter()
    out: list[Finding] = []
    for finding in raw:
        counted[finding.key] += 1
        seen = counted[finding.key]
        out.append(
            finding if seen == 1 else Finding(finding.path, finding.scope, finding.shape, f"{finding.subject}~{seen}")
        )
    return out


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    missing: list[str] = []
    for entry in EVIDENCE_ROOTS:
        base = ROOT / entry
        if base.is_file():
            files.append(base)
        elif base.is_dir():
            files.extend(
                path
                for path in sorted(base.rglob("*.py"))
                if not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
            )
        else:
            matched = sorted(ROOT.glob(entry))
            if not matched:
                # FAIL CLOSED. A roster entry that matches nothing means the file moved and this gate
                # silently stopped covering it -- the gate's own instance of the defect it is for.
                missing.append(entry)
            files.extend(matched)
    if missing:
        raise FileNotFoundError(
            "EVIDENCE_ROOTS names path(s) that match nothing, so they are no longer scanned: "
            + ", ".join(missing)
            + ". Point the roster at where the code moved rather than leaving it unscanned."
        )
    return sorted(set(files))


def scan() -> list[Finding]:
    found: list[Finding] = []
    for path in _scanned_files():
        relative = str(path.relative_to(ROOT))
        found.extend(findings_for_source(path.read_text(encoding="utf-8", errors="replace"), relative))
    return found


def _ledger() -> list[str]:
    if not LEDGER.is_file():
        return []
    return [
        line.split("#", 1)[0].strip()
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.split("#", 1)[0].strip()
    ]


def verdict(found: list[Finding], known: set[str]) -> tuple[list[Finding], list[str]]:
    """(new findings, stale ledger entries). Pure, so a test can drive both directions."""
    keys = {finding.key for finding in found}
    new = [finding for finding in found if finding.key not in known]
    stale = sorted(known - keys)
    return new, stale


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    found = scan()
    if "--list" in arguments:
        for finding in found:
            print(finding.key)
        return 0
    if "--write" in arguments:
        header = (
            "# Silent defaults on the declared evidence paths: a reading whose ambiguous answer is\n"
            "# the passing one. Keyed <path>::<scope>::<shape>::<subject> so a line move does not\n"
            "# rewrite the ledger. May only shrink: read the datum without a fallback and record the\n"
            "# absence, or move the code off an evidence path.\n"
            "# growth-accepted: the gate is new; this is debt it discovered, not debt added.\n"
        )
        LEDGER.write_text(header + "".join(f"{f.key}\n" for f in found), encoding="utf-8")
        print(f"wrote {LEDGER.relative_to(ROOT)} ({len(found)} entries)")
        return 0
    new, stale = verdict(found, set(_ledger()))
    for finding in new:
        print(f"[FAIL] silent default: {finding.path} ({finding.scope}) {finding.explain()}")
    for key in stale:
        print(f"[FAIL] stale ledger entry: {key} is gone -- remove it from {LEDGER.relative_to(ROOT)}.")
    if new or stale:
        print(
            f"\n{len(new)} new, {len(stale)} stale. An evidence path may not substitute a datum it did "
            f"not read. Scope is declared in EVIDENCE_ROOTS; widening it is a reviewed edit."
        )
        return 1
    print(f"[  ok] silent defaults: {len(found)} known on {len(_scanned_files())} evidence file(s); no new one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
