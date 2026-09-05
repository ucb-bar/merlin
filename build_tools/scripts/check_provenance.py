#!/usr/bin/env python3
"""Gate: a result that claims a hardware verdict must record which hardware revision it came from.

WHY. A session certified a microkernel against the only saturn revision containing the outer-product unit
while believing it was the revision named for the tapeout, which does not contain that unit at all.
Nothing in the artifact recorded which revision the numbers belonged to, so the mistake was invisible
until someone happened to name a commit in conversation. A result attributed to the wrong device is worse
than no result.

WHAT IS CHECKED, in increasing severity:

1. The pin registry loads and every pin declares a full sha (always enforced -- a malformed registry means
   nothing downstream can be verified).
2. Tracked reports that CLAIM a verdict (``certified: true``) carry a ``provenance`` block naming the
   revision. Enforced for reports not in the ratchet list below.
3. Where the checkout is reachable, pins are verified and material drift is reported.
4. For a pin that verifies its read set by CONTENT (a nested submodule's ISA headers, say), the per-FILE
   verdict is printed and an UNDETERMINABLE one FAILS the gate. Added after a measured miss: the headers
   every int8 dtype claim about the systolic target is derived from sit in a nested submodule off the
   gitlink its container records, and the containing pin verified CLEAN because its own three Scala files
   were clean. An OFF_PIN file is a loud note (the fix is in someone else's checkout and this gate cannot
   make it); an UNDETERMINABLE one is a failure, because a check that could not run must never report
   success.

ADVISORY BY DEFAULT, RATCHETED. Existing reports predate this convention and are listed in
``provenance_ratchet.txt``; the list may only shrink. New claims are enforced immediately, which is the
direction that matters -- the point is to stop producing unattributable results, not to retro-fit old ones.

Usage:
  check_provenance.py                 # tracked reports + registry
  check_provenance.py --staged        # staged files + the untracked `out/` scan (pre-commit)
  check_provenance.py --stop-hook     # session gate; same checks, hook-shaped output
  check_provenance.py --verify-pins   # additionally verify pins against live checkouts
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "merlin" / "python"))

RATCHET = _HERE.parent / "provenance_ratchet.txt"

#: Reports live under the generated-output root; only these are candidates.
_REPORT_SUFFIXES = (".json",)

#: Purgeable/huge subtrees that never hold verdicts. Skipped by name so the scan stays cheap; the
#: recaptures tree alone is ~130 GB and the cache is regenerable by definition.
_SKIP_DIRS = frozenset({"cache", "recaptures", "build", ".git", "__pycache__"})

#: Bound on files examined. Reported when hit rather than silently truncating -- a gate that quietly
#: stopped looking would read as "nothing to report".
_SCAN_CAP = 20000


def _tracked(staged: bool) -> list[Path]:
    """The tracked half of the work list. RAISES when `git` fails -- it must never return "nothing".

    Returning ``[]`` on a non-zero `git` made an unreadable index indistinguishable from a clean one:
    the gate printed OK having examined nothing. "We could not look" is not "there is nothing to
    find" (see check_no_answer_keys.py, which fixed the same shape first).
    """
    args = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged
            else ["git", "ls-files"])
    got = subprocess.run(args, capture_output=True, text=True, cwd=_ROOT, check=True)
    return [_ROOT / line for line in got.stdout.splitlines() if line.strip()]


# Directories this gate could not read (a live run's chmod-000 answer surface, most often). Surfaced in
# the summary so "no findings" never silently means "could not look".
_UNREADABLE: list[str] = []


def _scan_out() -> list[Path]:
    """Verdict-bearing report candidates under the generated-output root.

    The reports this gate exists for are UNTRACKED -- `out/` is gitignored -- so scanning only tracked
    files checked nothing at all, which is how the first version of this gate passed while an
    unattributed certification sat on disk.
    """
    try:
        from merlin.common.paths import artifacts_dir
        root = Path(artifacts_dir())
    except Exception:                                     # noqa: BLE001 — no output root yet
        return []
    if not root.is_dir():
        return []
    found: list[Path] = []
    stack = [root]
    seen = 0
    while stack:
        d = stack.pop()
        try:
            for entry in d.iterdir():
                seen += 1
                if seen > _SCAN_CAP:
                    print(f"  NOTE: scan capped at {_SCAN_CAP} entries; some reports were not examined",
                          file=sys.stderr)
                    return found
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name not in _SKIP_DIRS:
                        stack.append(entry)
                elif entry.suffix in _REPORT_SUFFIXES:
                    found.append(entry)
        except OSError:
            # Unreadable directory (commonly a run's chmod-000 answer surface). Recorded, not silent:
            # a report this gate could not examine must be visible, never counted as "clean".
            _UNREADABLE.append(str(d))
            continue
    return found


def _ratcheted() -> set[str]:
    if not RATCHET.is_file():
        return set()
    return {l.strip() for l in RATCHET.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}


def _claims_a_verdict(payload: object) -> bool:
    """True when a report asserts a hardware result, i.e. when attribution matters.

    Two shapes qualify. The first is an explicit boolean claim (``certified`` / ``correct`` / ``passed``).

    The second is a CAPSULE SCORE, which asserts a hardware result without ever using those words: it
    says N of M capsules passed on named oracle tiers. Keying only on the booleans made the primary
    result artifact of the whole capsule bench -- ``score_capsule.json`` -- structurally invisible to this
    gate, on every target. The mechanism existed and pointed away from the thing it should have been
    checking, which is worse than not having it: the reports looked gated and were not.

    A score is recognised by its own field shape, not by filename, so a renamed or relocated score is
    still caught. It counts as a claim only when it actually graded something -- an empty suite asserts
    nothing about any hardware.
    """
    if not isinstance(payload, dict):
        return False
    for key in ("certified", "correct", "passed"):
        if payload.get(key) is True:
            return True
    # A capsule SCORE is identified by its own field shape (`functional_pass`/`task` beside the counts),
    # not by filename, so a renamed or relocated score is still caught. Deliberately NOT every artifact
    # carrying pass counts: a per-round QA verdict has the same counts but is an intermediate written
    # every round inside a run dir, and the run's final score is what gets published and cited. Demanding
    # a block on each round would flag hundreds of historical files and train people to bypass the gate.
    if ("n_passed" in payload and "n_capsules" in payload
            and ("functional_pass" in payload or "task" in payload)):
        try:
            return int(payload.get("n_capsules") or 0) > 0
        except (TypeError, ValueError):
            return False
    return False


def _surface_sources(prov, pins: dict, got, notes: list[str], seen: set[str]) -> list[str]:
    """Print the per-file verdict for a pin that verifies its read set by CONTENT, and fail the gate on
    any UNDETERMINABLE one. Recurses into covered pins.

    WHY THIS IS SEPARATE FROM THE DRIFT LINE. A drift line is a NOTE here — advisory by design, because
    two pins in this registry are documented as permanently drifting on a development checkout and a
    permanently-red line nobody can distinguish from a regression stops being read. But the finding this
    surfaces is per-FILE and directional, and one of its three states must not be advisory:

    * PINNED           — the bytes are the pin's commit's bytes. A claim from them is a pinned claim.
    * OFF_PIN          — loud note. The bytes are known and are the wrong revision's; the fix is in
                         someone else's checkout, and this gate deliberately cannot make it.
    * UNDETERMINABLE   — PROBLEM. Nobody could tell which revision the bytes belong to, and a check that
                         could not run must never report success. This repo has been bitten by that shape
                         repeatedly (a codegen smoke check that was n/a and reported true burned 101
                         minutes); reporting "could not determine" as OK is the same bug.

    Only when the checkout is PRESENT. An absent checkout means this host simply does not have the
    hardware sources, which is a note, not a failure — otherwise the gate fails for everyone who has not
    cloned every external repo.
    """
    problems: list[str] = []
    declared = pins.get(got.pin)
    if declared is None:                                  # a covered pin that vanished mid-run
        notes.append(f"pin {got.pin}: verified but no longer declared; per-file verdict not surfaced")
        return problems
    # A covered pin is reached twice -- once on its own turn through the registry, once through its
    # container -- and printing the same per-file verdict twice reads as two separate findings.
    if got.pin in seen:
        return problems
    seen.add(got.pin)
    if declared.nested_in:
        rec = got.nested_recorded or "UNDETERMINABLE"
        notes.append(f"pin {got.pin}: nested in {declared.nested_in} at {declared.nested_path} — "
                     f"container records {rec[:12]}, pin declares {declared.commit[:12]}, checkout is at "
                     f"{got.observed.commit[:12]}")
        if got.observed.present and not got.nested_recorded:
            problems.append(f"pin {got.pin}: the revision {declared.nested_in} records at "
                            f"{declared.nested_path} is UNDETERMINABLE, so nothing can say which revision "
                            "the sources read from this nested checkout belong to. Not a pass.")
    for s in got.sources:
        if s.status == prov.PINNED:
            notes.append(f"pin {got.pin}: {s.rel} PINNED by content ({s.digest[:12]})")
        elif s.status == prov.OFF_PIN:
            notes.append(f"pin {got.pin}: {s.rel} OFF-PIN — {s.reason} (read {s.digest[:12]}, pinned "
                         f"revision {s.pinned_digest[:12]}). A claim derived from this file is NOT a "
                         "pinned claim and must not be reported as one.")
        else:
            problems.append(f"pin {got.pin}: {s.rel} is UNDETERMINABLE against its pin — {s.reason}. A "
                            "claim derived from it can be neither confirmed nor refuted, which is not the "
                            "same as clean.")
    for child in got.covered:
        problems.extend(_surface_sources(prov, pins, child, notes, seen))
    return problems


def _hook_result(stop_hook: bool, reason: str | None, lines: list[str] | None = None) -> int:
    """Emit the refusal (or the all-clear) in the dialect the caller speaks.

    A Claude Code Stop hook signals BLOCK through ``{"decision": "block"}`` on stdout; a non-zero exit
    is a NON-blocking error there. ``--stop-hook`` was parsed here but only ever decorated the success
    line, so every provenance failure exited 1 and the session stopped regardless -- the gate reported
    and could not enforce. Routing both dialects through one helper keeps them from drifting apart.
    """
    if not stop_hook:
        return 0 if reason is None else 1
    if reason is None:
        print(json.dumps({}))
    else:
        body = reason + ("\n- " + "\n- ".join(lines) if lines else "")
        print(json.dumps({"decision": "block", "reason": body}))
    return 0  # stop-hook signals via JSON, not exit code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    staged = "--staged" in argv
    stop_hook = "--stop-hook" in argv
    verify_pins = "--verify-pins" in argv
    problems: list[str] = []
    notes: list[str] = []

    # 1. The registry itself.
    try:
        from merlin.common import provenance as P
        pins = P.load_pins()
    except Exception as exc:                              # noqa: BLE001 — any failure is fatal here
        print(f"provenance: FAILED — pin registry unusable: {exc}", file=sys.stderr)
        return _hook_result(stop_hook, f"provenance: pin registry unusable: {exc}")
    notes.append(f"{len(pins)} pin(s) declared: {', '.join(sorted(pins))}")

    # 2. Tracked reports that claim a verdict.
    allow = _ratcheted()
    checked = 0
    # --staged SCANS `out/` TOO. The untracked scan is the whole reason this gate exists -- verdict-
    # claiming reports live untracked under the generated-output root -- and skipping it in --staged
    # meant the pre-commit hook, the ONLY place --staged is used, never ran that check once: measured
    # 0 reports checked under --staged against 86 for the bare invocation. Affordability was the
    # implied excuse and it does not hold: measured 1.6 s for the full scan against 0.09 s for staged-
    # only, on 86 verdict-claiming reports out of a capped 20000-entry walk.
    try:
        candidates = _tracked(staged) + _scan_out()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"provenance: FAILED — could not list the files to examine ({exc}); NOTHING was "
              f"examined, which is not the same as clean", file=sys.stderr)
        return _hook_result(stop_hook,
                            f"provenance: could not list the files to examine ({exc}); nothing "
                            f"was examined, which is not the same as clean")
    unreadable: list[str] = []
    for path in candidates:
        if path.suffix not in _REPORT_SUFFIXES:
            continue
        # A candidate can become unreadable BETWEEN the scan and this stat: an experiment locks its
        # answer surfaces with chmod 000 at launch, and this gate deliberately scans the same
        # ``out/artifacts`` tree those surfaces live in. Crashing there takes out the pre-commit and
        # Stop hooks for as long as a run holds the lock. Fail CLOSED but stay alive: record the path as
        # unexamined and surface it, so an unreadable report is visible rather than silently uncounted.
        try:
            if not path.is_file():
                continue
        except OSError:
            unreadable.append(str(path))
            continue
        try:
            rel = path.relative_to(_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not _claims_a_verdict(payload):
            continue
        checked += 1
        prov = payload.get("provenance") if isinstance(payload, dict) else None
        if prov:
            continue
        if rel in allow:
            notes.append(f"ratcheted (pre-dates the convention): {rel}")
            continue
        problems.append(f"{rel}: claims a verdict but records no `provenance` block, so the result "
                        f"cannot be attributed to a hardware revision. Record one via "
                        f"merlin.common.provenance.record(), or add the path to "
                        f"{RATCHET.name} with a reason.")

    # 3. Optional live verification.
    if verify_pins:
        # Built artifacts first: a bitstream or a prebuilt simulator has no revision of its own, so it is
        # verified by CONTENT. An artifact present with no declared digest is reported as a gap rather than
        # passing quietly -- one that verifies against nothing certifies itself.
        try:
            arts = P.load_artifacts()
        except Exception as exc:                          # noqa: BLE001 — a malformed section is fatal
            print(f"provenance: FAILED — artifact registry unusable: {exc}", file=sys.stderr)
            return _hook_result(stop_hook, f"provenance: artifact registry unusable: {exc}")
        for name in sorted(arts):
            got = P.verify_artifact(name)
            if got.ok:
                notes.append(f"artifact {name}: ok ({got.digest[:12]})")
            else:
                notes.append(f"artifact {name}: NOT VERIFIED — {'; '.join(got.gaps)}")
        surfaced: set[str] = set()
        for name in sorted(pins):
            got = P.verify(name)
            if got.ok:
                notes.append(f"pin {name}: ok at {got.observed.commit[:12]}")
            else:
                detail = "; ".join([*got.drift,
                                    *([f"missing {list(got.missing_paths)}"] if got.missing_paths else []),
                                    *([f"forbidden present {list(got.forbidden_present)}"]
                                      if got.forbidden_present else [])])
                notes.append(f"pin {name}: DRIFT — {detail}")
            problems.extend(_surface_sources(P, pins, got, notes, surfaced))

    # Surface what could not be examined. A gate that says OK while it silently skipped reports is the
    # failure mode this whole convention exists to prevent, so the count is always printed when non-zero.
    skipped = len(unreadable) + len(_UNREADABLE)
    if skipped:
        notes.append(f"{skipped} path(s) NOT EXAMINED (unreadable — commonly a live run's chmod-000 "
                     f"answer surface): " + ", ".join(
                         [Path(x).name for x in (_UNREADABLE + unreadable)][:5])
                     + (" ..." if skipped > 5 else ""))

    for n in notes:              # stdout must be JSON ONLY in hook mode
        print(f"  {n}", file=sys.stderr if stop_hook else sys.stdout)
    if problems:
        print("provenance: FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return _hook_result(stop_hook, f"Hardware-provenance violations ({len(problems)}):", problems)
    if stop_hook:
        return _hook_result(stop_hook, None)
    print(f"provenance: OK ({checked} verdict-claiming report(s) checked"
          + (f", {skipped} unreadable" if skipped else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
