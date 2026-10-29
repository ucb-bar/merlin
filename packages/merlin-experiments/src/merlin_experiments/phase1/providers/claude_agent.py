"""Claude CLI author: the same kickoff and continuation text as the Codex author, resumed for the wall budget.

The ordinary Phase 1 controller drives the ``claude`` CLI as one ``claude --print`` turn. This module builds the
sandboxed shell script for a Claude author that matches the Codex author's contract:

* kickoff bytes are ``authoring_prompt()`` plus the delegation text (``codex_agent.delegation_user_prompt``) exactly
  as the Codex provider composes them; ``TASK.md`` and the workspace are unchanged;
* under ``--schedule continuous`` the session is kept alive with ``claude --resume <session-id>`` and the standing
  continuation prompt (``codex_agent.DEFAULT_CONTINUATION_PROMPT``) until the wall budget is spent, a turn fails, or
  the turn cap is reached (``codex_agent._CONTINUE_*``), mirroring the Codex thread keep-alive;
* the loop runs INSIDE the one sandbox launch: the sandbox stages a fresh Claude home per launch, and the session
  store lives in that home, so resuming from a second launch would find no session. No sandbox policy changes.

The claude CLI has no concurrent-sub-agent flag; the cap is the delegation prompt's "at most N helper sub-agent(s)".
"""

from __future__ import annotations

import shlex
import uuid


def kickoff_prompt(limit: int | None) -> str:
    from .authoring_prompt import authoring_prompt
    from .codex_agent import delegation_user_prompt

    return delegation_user_prompt(authoring_prompt(), limit)


def continuation_prompt(limit: int | None) -> str:
    from .codex_agent import DEFAULT_CONTINUATION_PROMPT, delegation_user_prompt

    return delegation_user_prompt(DEFAULT_CONTINUATION_PROMPT, limit)


def session_script(
    *, cli_model: str, effort: str, ws, limit: int | None, continuous: bool, timeout: int, session_id: str | None = None,
    kickoff: str | None = None, continuation: str | None = None,
) -> str:
    """The ``inner`` command for ``sandbox_command``: one turn, or a resume loop bounded by ``timeout`` seconds."""
    from .codex_agent import _CONTINUE_MAX_TURNS, _CONTINUE_MIN_S

    sid = session_id or str(uuid.uuid4())
    flags = (
        f"--print --model {shlex.quote(cli_model)} --effort {shlex.quote(effort)} "
        f"--permission-mode bypassPermissions --add-dir {shlex.quote(str(ws))} "
        "--output-format stream-json --verbose"
    )
    first = kickoff if kickoff is not None else kickoff_prompt(limit)
    if not continuous:
        return f"claude {flags} --session-id {sid} {shlex.quote(first)} < /dev/null"
    cont = continuation if continuation is not None else continuation_prompt(limit)
    return "\n".join(
        [
            f"K={shlex.quote(first)}",
            f"C={shlex.quote(cont)}",
            f"D=$(( $(date +%s) + {int(timeout)} )); T=0",
            "while :; do",
            f'  if [ "$T" -eq 0 ]; then claude {flags} --session-id {sid} "$K" < /dev/null;'
            f' else claude {flags} --resume {sid} "$C" < /dev/null; fi',
            "  rc=$?; T=$((T+1)); R=$(( D - $(date +%s) ))",
            "  S=",
            f'  [ "$rc" -ne 0 ] && S="the turn did not complete"',
            f'  [ -z "$S" ] && [ "$R" -le {_CONTINUE_MIN_S} ] && S="wall budget spent"',
            f'  [ -z "$S" ] && [ "$T" -ge {_CONTINUE_MAX_TURNS} ] && S="turn cap {_CONTINUE_MAX_TURNS}"',
            '  printf \'{"type":"claude_session_turn","turn":%s,"rc":%s,"remaining_s":%s,"continuing":%s,'
            '"stopped_because":"%s","session_id":"' + sid + '"}\\n\' "$T" "$rc" "$R" '
            '"$([ -z "$S" ] && echo true || echo false)" "$S"',
            '  [ -n "$S" ] && exit "$rc"',
            "done",
        ]
    )
