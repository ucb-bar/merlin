"""Join a literal all-capsule request to its own later completed attach response."""

from __future__ import annotations

import json
from pathlib import Path


def _arguments(call, tokenize):
    if call.name.lower() not in {"bash", "shell", "exec_command", "run_shell", "shell_command", "run_command"}:
        return None
    command = call.input.get("command") or call.input.get("cmd")
    if not isinstance(command, str) or any(marker in command for marker in ("\n", "\r", "$(", "`")):
        return None
    tokens = tokenize(command)
    for _ in range(4):
        if not tokens:
            return None
        if Path(tokens[0]).name not in {"bash", "sh", "dash", "zsh", "ksh"}:
            break
        if len(tokens) != 3 or tokens[1] not in {"-c", "-lc", "-cl"}:
            return None
        if any(marker in tokens[2] for marker in ("\n", "\r", "$(", "`")):
            return None
        tokens = tokenize(tokens[2])
    if not tokens or any(token in {";", "&&", "||", "|", "&", ">", ">>", "<", "<<"} for token in tokens):
        return None
    if Path(tokens[0]).name in {"python", "python3"}:
        tokens = tokens[1:]
        while tokens and tokens[0] in {"-u", "-B"}:
            tokens = tokens[1:]
    if not tokens or Path(tokens[0]).name != "agent_selfcheck.py":
        return None
    return tokens[1:]


def _option(arguments, name, default=None):
    values = [arguments[i + 1] for i, value in enumerate(arguments[:-1]) if value == name]
    values.extend(value.partition("=")[2] for value in arguments if value.startswith(name + "="))
    return values[0] if len(values) == 1 else default if not values and name not in arguments else None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate selfcheck JSON field")
        result[key] = value
    return result


def _response(text):
    lines = text.strip().splitlines()
    while lines and lines[0].startswith("[self-check]"):
        lines.pop(0)
    try:
        text = "\n".join(lines)
        value, end = json.JSONDecoder(object_pairs_hook=_unique).raw_decode(text)
        if text[end:].strip():
            note = json.loads(text[end:], object_pairs_hook=_unique)
            if not isinstance(note, dict) or set(note) != {"note"} or not isinstance(note["note"], str):
                return {}
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def all_invocation(call, *, tokenize):
    arguments = _arguments(call, tokenize)
    return (
        arguments is not None
        and _option(arguments, "--capsules") == "all"
        and not any(value == "--attach" or value.startswith("--attach=") for value in arguments)
    )


def _completed(body, request_id):
    count, passed, rows = body.get("n_capsules"), body.get("n_passed"), body.get("per_capsule")
    if (
        body.get("selfcheck_request_id") != request_id
        or type(count) is not int
        or count <= 0
        or type(passed) is not int
        or not 0 <= passed <= count
        or type(body.get("all_pass")) is not bool
        or body.get("status") not in {None, "completed", "done"}
        or body.get("partial", False) is not False
        or body.get("error")
        or not isinstance(rows, list)
        or len(rows) != count
        or body.get("n_result_rows", count) != count
    ):
        return False
    names = [row.get("capsule") for row in rows if isinstance(row, dict)]
    if len(names) != count or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != count:
        return False
    stamps = [body.get(key) for key in ("requested_at_unix_ns", "grade_started_at_unix_ns", "graded_at_unix_ns")]
    return (
        all(type(value) is int and value > 0 for value in stamps)
        and stamps[0] < stamps[1] < stamps[2]
        and all(
            isinstance(body.get(key), str) and len(body[key]) == 64 and all(c in "0123456789abcdef" for c in body[key])
            for key in ("submission_sha256", "requested_submission_sha256")
        )
    )


def completed_all_request(calls, *, tokenize):
    """Workflow evidence only: a nonzero completed grade still counts, never a partial response."""
    pending = {}
    ambiguous = set()
    for call in calls:
        arguments = _arguments(call, tokenize)
        if arguments is None or not call.result_present:
            continue
        body = _response(call.result_text)
        attach = _option(arguments, "--attach")
        if attach:
            earlier = pending.get(attach)
            if attach not in ambiguous and earlier is not None:
                events = (earlier.start_event, earlier.end_event, call.start_event, call.end_event)
                if (
                    all(type(value) is int for value in events)
                    and events[0] < events[1] < events[2] < events[3]
                    and _completed(body, attach)
                ):
                    return True
        elif (
            all_invocation(call, tokenize=tokenize)
            and body.get("status") == "in_progress"
            and body.get("submission") == _option(arguments, "--submission", "submission")
        ):
            request_id = body.get("request_id")
            if isinstance(request_id, str) and request_id and not any(c.isspace() for c in request_id):
                if request_id in pending:
                    ambiguous.add(request_id)
                pending[request_id] = call
    return False
