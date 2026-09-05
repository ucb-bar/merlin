#!/usr/bin/env python3
"""PreToolUse guard: deny pattern-matched process kills and self-matching wait loops.

This host runs MULTIPLE concurrent agent sessions across different projects, plus long paid
experiment runs. Two distinct failure modes have each bitten repeatedly, and both are documented in
the `no-broad-pkill-shared-host` memory — which did not stop them, because a memory is advice and
this is a gate.

**1. `pkill -f <pattern>` kills the caller.** The wrapper runs the whole script as
`bash -c '<entire script text>'`, so the pattern you are matching on is ALWAYS present in the calling
shell's own argv. `pkill` excludes itself but not its parent. Measured three times (2026-07-14,
2026-08-28, 2026-09-05), each presenting as a misleading `exit 144` that reads like a timeout or a
resource cap rather than a kill. The last one used the pattern `land.sh` — about as narrow as a
pattern gets — and still self-terminated.

**2. `pgrep -f <pattern>` in a wait loop hangs forever.** Same root cause, opposite symptom: the loop
waits on a process that is itself, so it never exits. Measured 2026-08-29 — it spun 34 minutes,
produced no output, and looked exactly like a slow job.

**3. Killing by bare process name hits other sessions.** `pkill python`, `pkill pytest`, `pkill z3`
and friends reach across projects and kill work that is not yours, including paid runs.

Do this instead:
  * capture the PID at launch — `nohup … & echo $!` — and `kill <pid>`;
  * wait on `ps -p <pid>` or a marker file the job writes on exit, never on a pattern;
  * use the harness task tools (TaskStop by task id) for jobs the harness launched;
  * inspect with a bare `pgrep -af <pattern>` (allowed) and kill the specific PIDs you meant.

Contract: exit 0 = allow; exit 2 + stderr message = deny.
Escape hatch: `MERLIN_ALLOW_PROCESS_KILL=1` for a genuine one-off you have reasoned about.
"""
from __future__ import annotations

import json
import os
import sys

#: Commands that kill by PATTERN. Always denied: the pattern is in the caller's own argv.
PATTERN_KILLERS = {"pkill"}

#: Commands that kill by NAME across the whole host, reaching other sessions' work.
NAME_KILLERS = {"killall"}

#: Process names common enough that killing them by name hits another session or a paid run.
SHARED_NAMES = {
    "python", "python3", "pytest", "z3", "node", "java", "verilator", "spike",
    "mlir-opt", "mlir-translate", "llvm-lit", "firesim", "make", "ninja", "cargo",
}

#: Shell tokens that start a new command, so a word after one is in command position.
SEPARATORS = {"&&", "||", ";", "|", "&", "(", ")", "{", "}", "\n"}

#: Loop keywords — a pattern match inside one of these is the hang, not the kill.
LOOP_WORDS = {"until", "while"}

#: Words that may PRECEDE a command without being one: `until ! pgrep …`, `while ! ps …`,
#: `time pytest …`. Without stripping these the real command is never in head position and the guard
#: silently allows exactly the case it exists to catch — which is how the wait-loop hang first got
#: through this hook's own test.
COMMAND_PREFIXES = {"until", "while", "if", "then", "do", "!", "time", "nohup", "exec", "sudo"}


def _segments(command: str) -> list[list[str]]:
    """Split a shell command into rough command segments, without regex.

    Deliberately approximate: the goal is to tell an INVOCATION from a MENTION. A guard that fired on
    the mere substring "pkill" would block writing this very file, or any note explaining the rule.
    Splitting on separators and looking only at the first word of each segment gets that right for
    ordinary scripts, and errs toward allowing — text inside a heredoc or a quoted string is almost
    never the first word of a segment.
    """
    words = command.replace("\n", " \n ").split()
    out: list[list[str]] = []
    current: list[str] = []
    for word in words:
        stripped = word.strip("'\"")
        if stripped in SEPARATORS or word in SEPARATORS:
            if current:
                out.append(current)
            current = []
            continue
        current.append(word)
    if current:
        out.append(current)
    return out


def _basename(token: str) -> str:
    return token.strip("'\"").rsplit("/", 1)[-1]


def _deny(reason: str, remedy: str) -> None:
    print(f"blocked: {reason}\n\n{remedy}\n\n"
          f"(guard: .claude/hooks/guard_process_kills.py; genuine one-off: "
          f"MERLIN_ALLOW_PROCESS_KILL=1)", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    if os.environ.get("MERLIN_ALLOW_PROCESS_KILL"):
        return 0
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    command = str((payload.get("tool_input") or {}).get("command") or "")
    if not command.strip():
        return 0

    segments = _segments(command)
    in_loop = any(_basename(seg[0]) in LOOP_WORDS for seg in segments if seg)

    for seg in segments:
        # Strip prefixes so the actual command reaches head position.
        idx = 0
        while idx < len(seg) and _basename(seg[idx]) in COMMAND_PREFIXES:
            idx += 1
        seg = seg[idx:]
        if not seg:
            continue
        head = _basename(seg[0])
        rest = [w.strip("'\"") for w in seg[1:]]

        if head in PATTERN_KILLERS:
            _deny(
                f"`{head}` matches against every process's FULL command line, and this wrapper runs "
                f"your whole script as `bash -c '<script>'` — so the pattern is always present in the "
                f"calling shell's own argv and {head} kills the caller. Measured three times, each "
                f"time surfacing as a misleading `exit 144`.",
                "Instead: capture the PID at launch (`nohup … & echo $!`) and `kill <pid>`; or run "
                "`pgrep -af <pattern>` first, read what it actually matched, and kill those PIDs; or "
                "use TaskStop for a harness-launched job.")

        if head in NAME_KILLERS:
            _deny(
                f"`{head}` kills every process with that name, across every session on this shared "
                f"host — including other projects' work and live paid runs.",
                "Instead: `pgrep -af <name>` to see whose processes exist, then kill only the PIDs "
                "you launched yourself.")

        if head == "kill":
            for arg in rest:
                if arg and not arg.lstrip("-").isdigit() and not arg.startswith("-"):
                    _deny(
                        f"`kill {arg}` does not look like a PID. Killing by anything but an exact PID "
                        f"you launched risks another session's work on this shared host.",
                        "Instead: kill the PID you captured at launch, or `pgrep -af` first and kill "
                        "the specific PIDs.")

        if head == "pgrep" and in_loop:
            _deny(
                "`pgrep -f` inside a wait loop waits on ITSELF: the pattern appears in the enclosing "
                "shell's argv, so the loop never exits. Measured 2026-08-29 — it spun 34 minutes with "
                "no output and looked exactly like a slow job.",
                "Instead: poll an exact PID with `ps -p <pid>`, or wait for a marker file the job "
                "writes when it exits, or let the harness notify you.")

        if head in ("pkill", "killall") or (head == "pgrep" and any(a in SHARED_NAMES for a in rest)):
            continue  # already handled above; kept explicit so the intent is readable

    return 0


if __name__ == "__main__":
    sys.exit(main())
