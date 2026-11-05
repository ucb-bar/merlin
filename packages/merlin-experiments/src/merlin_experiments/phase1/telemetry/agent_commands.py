"""Which harness clients a recorded author shell command invoked, and with what arguments.

The author's commands are free-form shell: multi-line scripts, heredocs that carry Python, pipes,
redirections and wrappers (``timeout 600 python3 agent_selfcheck.py ... > log``). A substring test is
not enough -- measured on a real run, ``rg --files -g 'agent_selfcheck.py'`` and
``sed -n '1,130p' agent_selfcheck.py`` both mention the client without running it. So the script is
parsed structurally: heredoc bodies are dropped, each line is tokenized with :mod:`shlex`, split into
simple commands at shell operators, environment assignments and common wrappers are peeled off, and
only a command whose program (or interpreter's script) IS the client counts as an invocation.

This reads command text the author wrote; it never executes it and never opens answer material.
"""

from __future__ import annotations

import shlex

SELFCHECK = "agent_selfcheck.py"
SIMJOB = "simjob.py"
CLIENTS = (SELFCHECK, SIMJOB)

_OPERATOR_CHARS = frozenset("();<>|&")
_REDIRECTS = frozenset({">", ">>", "<", ">&", "&>", ">|", "<&", "&>>", "<>"})
#: Wrappers that run their argument as a command, with how many leading option words they take.
_WRAPPERS = frozenset({"env", "timeout", "nice", "nohup", "stdbuf", "taskset", "time", "exec", "command"})
_VALUE_FLAGS = frozenset(
    {
        "--submission",
        "--capsules",
        "--sim",
        "--tiers",
        "--attach",
        "--workers",
        "--timeout",
        "--out",
        "--wait-budget",
        "--caller-layout",
        "--job-id",
        "--debug",
    }
)
_PROBE_FLAGS = (
    "--compile-only",
    "--scale-probe",
    "--candidate-diagnostics",
    "--shape-coverage",
    "--offload-census",
    "--model-layers",
    "--dynamic-census",
)


def _heredoc_delimiters(line: str) -> list[str]:
    """Delimiters of every ``<<WORD`` / ``<<-'WORD'`` opened on one line (``<<<`` is a here-string)."""
    found = []
    index = line.find("<<")
    while index >= 0:
        rest = line[index + 2 :]
        if rest.startswith("<"):
            index = line.find("<<", index + 3)
            continue
        rest = rest.lstrip("-").lstrip()
        word = []
        for ch in rest:
            if ch.isspace() or ch in ";|&<>()":
                break
            word.append(ch)
        delimiter = "".join(word).strip("'\"")
        if delimiter:
            found.append(delimiter)
        index = line.find("<<", index + 2)
    return found


def _lines(script: str) -> list[str]:
    """Script lines with every heredoc body removed (its text is data, not commands)."""
    out: list[str] = []
    pending: list[str] = []
    for line in script.split("\n"):
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        out.append(line)
        pending.extend(_heredoc_delimiters(line))
    return out


def _tokens(line: str) -> list[str]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return list(lexer)
    except ValueError:  # Unbalanced quoting: fall back to whitespace words rather than dropping the line.
        return line.split()


def simple_commands(script: str) -> list[list[str]]:
    """Every simple command's words, operators and redirection targets removed."""
    commands: list[list[str]] = []
    for line in _lines(script):
        current: list[str] = []
        skip_next = False
        for token in _tokens(line):
            if skip_next:
                skip_next = False
                continue
            if token in _REDIRECTS:
                skip_next = True
                continue
            if token and set(token) <= _OPERATOR_CHARS:
                if current:
                    commands.append(current)
                current = []
                continue
            current.append(token)
        if current:
            commands.append(current)
    return commands


def _assignment(word: str) -> bool:
    name, sep, _ = word.partition("=")
    return bool(sep) and name.isidentifier()


def _peel(words: list[str]) -> list[str]:
    """Drop leading ``NAME=value`` assignments and wrapper programs with their own options."""
    i = 0
    while i < len(words):
        word = words[i]
        base = word.rsplit("/", 1)[-1]
        if _assignment(word):
            i += 1
        elif base in _WRAPPERS:
            i += 1
            while i < len(words) and (words[i].startswith("-") or _assignment(words[i])):
                takes_value = words[i] in ("-n", "-c", "-s", "-k", "-u", "--signal", "--kill-after", "-o", "-e")
                i += 2 if takes_value else 1
            if base in ("timeout", "taskset") and i < len(words):
                i += 1  # the duration / CPU list
        else:
            break
    return words[i:]


def invocation(words: list[str]) -> tuple[str, list[str]] | None:
    """``(client, args)`` when this simple command runs a harness client, else ``None``."""
    words = _peel(words)
    if not words:
        return None
    program = words[0].rsplit("/", 1)[-1]
    if program in CLIENTS:
        return program, words[1:]
    if not program.startswith("python"):
        return None
    i = 1
    while i < len(words) and words[i].startswith("-"):
        if words[i] in ("-m", "-c", "-"):
            return None  # a module, inline program or stdin script is not the staged client file
        i += 2 if words[i] in ("-X", "-W") else 1
    if i < len(words) and words[i].rsplit("/", 1)[-1] in CLIENTS:
        return words[i].rsplit("/", 1)[-1], words[i + 1 :]
    return None


def _options(args: list[str]) -> tuple[dict, list[str]]:
    values: dict[str, str] = {}
    positional: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        flag, sep, inline = arg.partition("=")
        if flag in _VALUE_FLAGS:
            if sep:
                values[flag] = inline
                i += 1
            else:
                values[flag] = args[i + 1] if i + 1 < len(args) else ""
                i += 2
            continue
        if arg.startswith("-"):
            values[arg] = ""
        else:
            positional.append(arg)
        i += 1
    return values, positional


def selfcheck_request(args: list[str]) -> dict:
    """The request one ``agent_selfcheck.py`` invocation submits (or the request it re-attaches to)."""
    values, _ = _options(args)
    probes = [flag[2:] for flag in _PROBE_FLAGS if flag in values]
    if "-h" in values or "--help" in values:
        mode = "help"
    elif "--attach" in values:
        mode = "attach"
    elif probes:
        mode = "+".join(probes)
    else:
        mode = "check"
    submission = values.get("--submission", "submission") or "submission"
    parts = [part for part in submission.split("/") if part not in ("", ".")]
    return {
        "mode": mode,
        "submits_request": mode not in ("help", "attach"),
        "submission": submission,
        "variant": parts[1] if len(parts) >= 2 and parts[0] == "variants" else None,
        "capsules": values.get("--capsules", "all") or "all",
        "sim": values.get("--sim") or None,
        "tiers": values.get("--tiers") or None,
        "attach": values.get("--attach") or None,
        "caller_layout": values.get("--caller-layout") or None,
    }


def simjob_request(args: list[str]) -> dict:
    values, positional = _options(args)
    action = positional[0] if positional else None
    return {
        "action": action,
        "submits_request": action == "submit",
        "capsules": values.get("--capsules", "all") or "all",
        "sim": values.get("--sim") or None,
        "job_id": values.get("--job-id") or None,
    }


_SHELLS = frozenset({"bash", "sh", "zsh", "dash"})


def _calls(commands: list[list[str]], depth: int) -> list[dict]:
    calls = []
    for words in commands:
        peeled = _peel(words)
        # ``bash -lc SCRIPT`` (how Codex records every command) runs SCRIPT: parse it, bounded.
        if (
            depth < 3
            and len(peeled) >= 3
            and peeled[0].rsplit("/", 1)[-1] in _SHELLS
            and peeled[1].startswith("-")
            and peeled[1].endswith("c")
        ):
            calls.extend(_calls(simple_commands(peeled[2]), depth + 1))
            continue
        found = invocation(peeled)
        if found is None:
            continue
        client, args = found
        request = selfcheck_request(args) if client == SELFCHECK else simjob_request(args)
        calls.append({"client": client, **request})
    return calls


def client_calls(command) -> list[dict]:
    """Harness-client invocations of one recorded command (argv list or shell string)."""
    if isinstance(command, list):
        return _calls([[str(part) for part in command]], 0)
    if not isinstance(command, str):
        return []
    try:
        whole = shlex.split(command)
    except ValueError:
        whole = []
    # The exec stream renders the argv as one string (``/bin/bash -lc '...'``) whose quoted script may
    # span lines; split it as a whole first so the script stays one word.
    if len(whole) >= 3 and whole[0].rsplit("/", 1)[-1] in _SHELLS and whole[1].startswith("-"):
        return _calls([whole], 0)
    return _calls(simple_commands(command), 0)
