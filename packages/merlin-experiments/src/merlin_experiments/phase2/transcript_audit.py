"""Read-only Phase 2 transcript admission; no command execution or target policy.

Checks broker use and answer reconnaissance in translated/native agent events.
The audit is defense in depth, never a replacement for answer-masked mounts.
"""

from __future__ import annotations

import ast
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from merlin.common.access import audit_token_in
from merlin.common.digest import sha256_bytes as _sha256
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import audit_tokens
from merlin.targetgen.target_experiment import TargetExperiment

from .broker import BROKER_NAME, BrokerAction
from .contracts import StageGateError
from .contracts import canonical_json as _canonical_json


def _heredoc_delimiter(line: str) -> str | None:
    """The delimiter word of a heredoc redirection on this line, or None.

    Structural, not pattern-matched: locate ``<<``, skip the ``<<<`` here-string form, drop an
    optional ``-`` (tab-stripping form), and take the next word with its quoting removed.
    """
    marker = line.find("<<")
    if marker < 0 or line[marker : marker + 3] == "<<<":
        return None
    rest = line[marker + 2 :].lstrip()
    if rest.startswith("-"):
        rest = rest[1:].lstrip()
    fields = rest.split()
    if not fields:
        return None
    return fields[0].strip("'\"") or None


def _split_heredocs(text: str) -> tuple[list[str], str]:
    """Peel heredoc bodies out of a shell payload.

    A heredoc body is DATA to the shell but SOURCE to the interpreter that reads it, so it has to be
    audited as source -- and it must NOT be lexed as further commands. Its lines are not commands,
    and counting them as such corrupts the brokered/total ratio that decides whether a payload mixed
    brokered and unbrokered work.
    """
    bodies: list[str] = []
    kept: list[str] = []
    lines = text.replace("\r", "\n").split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        delimiter = _heredoc_delimiter(line)
        index += 1
        if delimiter is None:
            continue
        body: list[str] = []
        while index < len(lines) and lines[index].strip() != delimiter:
            body.append(lines[index])
            index += 1
        index += 1  # consume the terminator line (past-the-end is fine: unterminated heredoc)
        bodies.append("\n".join(body))
    return bodies, "\n".join(kept)


def audit_codex_transcript(
    path: Path,
    target_experiment: TargetExperiment,
    candidate: Path,
    actions: Sequence[BrokerAction] = (),
    *,
    audit_token_set: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Reject answer reconnaissance and direct execution in translated or native Codex JSONL.

    This audit is a second line of defence.  Answer bytes are absent from the outer mount table, and
    target tools are available only to the inner broker; an audit hit therefore makes the candidate
    non-consumable even if the attempted command failed.  Native Codex emits both ``item.started`` and
    ``item.completed`` for one command, so those envelopes are validated independently but counted once.
    """
    tokens = audit_tokens(target_experiment) if audit_token_set is None else audit_token_set
    if (
        not isinstance(tokens, Mapping)
        or set(tokens) != {"answer", "grader", "oracle_subpath"}
        or any(
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or any(not isinstance(value, str) or not value for value in values)
            for values in tokens.values()
        )
    ):
        raise StageGateError("transcript audit token set is malformed")
    answer_tokens = tuple(value.lower() for values in tokens.values() for value in values if value)
    entry_tokens: set[str] = set()
    manifest = candidate / "manifest.yaml"
    if manifest.is_file():
        document = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        entries = document.get("entrypoints") if isinstance(document, dict) else {}
        if isinstance(entries, dict):
            entry_tokens.update(Path(str(value)).name.lower() for value in entries.values() if isinstance(value, str))
    entry_tokens -= {"python", "python3", "bash", "sh", "env"}
    target_tool_tokens: set[str] = set()
    ordinary_shell = {
        "awk",
        "basename",
        "bash",
        "cat",
        "command",
        "cut",
        "dirname",
        "echo",
        "env",
        "false",
        "find",
        "grep",
        "head",
        "ls",
        "printf",
        "pwd",
        "python",
        "python3",
        "readlink",
        "sed",
        "sh",
        "tail",
        "test",
        "true",
        "wc",
        "which",
    }
    for probe in TC.required_tool_probes(target_experiment):
        first = probe.cmd.strip().split(maxsplit=1)[0] if probe.cmd.strip() else ""
        if first and Path(first).name.lower() not in ordinary_shell:
            target_tool_tokens.add(Path(first).name.lower())
        # Probe labels begin with an ASCII tool token, optionally followed by prose.
        # Stop at the first delimiter; do not search later words or accept Unicode
        # lookalikes as tool-name characters.
        end = 0
        for character in probe.label:
            if not (character.isascii() and (character.isalnum() or character in "+_.-")):
                break
            end += 1
        label_token = probe.label[:end].lower()
        if label_token and label_token not in ordinary_shell:
            target_tool_tokens.add(label_token)
    hits: list[dict[str, str]] = []
    broker_invocations: list[dict[str, Any]] = []
    action_names = {action.name for action in actions}
    commands_seen = 0
    current_thread = "no-thread"
    native_commands: dict[tuple[str, str], str] = {}
    known_native_items = {
        "agent_message",
        "command_execution",
        "error",
        "file_change",
        "mcp_tool_call",
        "reasoning",
        "web_search",
    }

    def audit_command(command: str, line_number: int) -> None:
        nonlocal commands_seen
        commands_seen += 1
        lowered = command.lower()
        command_words = {Path(word.strip("'\";|&()[]{}<>")).name for word in lowered.replace("\n", " ").split()}
        try:
            outer_words = shlex.split(command)
        except ValueError:
            hits.append(
                {
                    "kind": "malformed_shell_command",
                    "line": str(line_number),
                    "command_sha256": _sha256(command.encode("utf-8")),
                }
            )
            outer_words = []
        words: list[str] = []
        payload_text = command
        if len(outer_words) == 3 and Path(outer_words[0]).name in ("bash", "sh") and outer_words[1] in ("-c", "-lc"):
            payload_text = outer_words[2]
        try:
            lexer = shlex.shlex(payload_text, posix=True, punctuation_chars=";&|<>()")
            lexer.whitespace_split = True
            lexer.commenters = ""
            words = list(lexer)
        except ValueError:
            words = []
            hits.append(
                {
                    "kind": "malformed_shell_command",
                    "line": str(line_number),
                    "command_sha256": _sha256(command.encode("utf-8")),
                }
            )
        # DECOMPOSE COMPOUND COMMANDS, then apply the SAME test to each simple command.
        #
        # This is a more precise parse, not a looser rule. The previous version tested the whole
        # payload as one string, which produced two false accusations against a correctly-behaving
        # agent (measured 2026-09-03, 6 hits on a 28-command round):
        #
        #  * a payload containing "\n" was rejected wholesale, so a batch of TEN one-per-line broker
        #    calls counted as zero invocations while the host wrote ten receipts -- surfacing as
        #    "host broker receipts do not exactly match transcript invocations" (12 vs 22);
        #  * `BROKER_NAME in lowered` fired whenever the broker path merely APPEARED, so
        #    `ls -l /perf-control/perf_tool.py` was an "invalid broker invocation" -- while the prompt
        #    ORDERS the agent to stop with NO-GO unless it verifies that exact file exists. The audit
        #    punished the behaviour the prompt demands.
        #
        # Splitting on shell separators keeps every guarantee: each simple command must still be a
        # clean, single, well-formed invocation with known action and unique bindings, and the
        # `$`/backtick ban still applies WITHIN each one. What changes is that a mention of the broker
        # as an ARGUMENT to another program is data, not an invocation -- unless that program could
        # execute it, which is what the obfuscation check below still catches.
        _SEPARATORS = ("&&", "||", ";", "|", "\n", "\r")
        # DENY BY DEFAULT. A simple command may NAME the broker only as an argument to one of these
        # read-only inspection verbs; anything else that mentions it is an invalid invocation. An
        # allowlist is the safe direction here: a deny-list of `eval`/`$(`/backtick lets
        # `cp <broker> /tmp/x && python3 /tmp/x ...` and
        # `python3 -c 'exec(open("<broker>").read())'` straight through, both of which are pinned as
        # must-fail by test_wrapped_broker_compound_rename_and_python_exec_forms_fail_closed.
        _READ_ONLY_VERBS = {
            "ls",
            "stat",
            "cat",
            "head",
            "tail",
            "wc",
            "sed",
            "grep",
            "find",
            "rg",
            "ripgrep",
            "file",
            "readlink",
            "test",
            "diff",
            "sha256sum",
            "md5sum",
            "cksum",
            "du",
            "basename",
            "dirname",
            "realpath",
        }

        def _simple_commands(text: str) -> list[list[str]]:
            """Split into simple commands as TOKEN LISTS, quote-aware.

            Splitting the raw string on separators is wrong: it cuts inside quotes and leaves
            fragments the lexer then reports as `malformed_shell_command` (measured: 8 spurious hits
            on a real round). Lex each shell LOGICAL LINE first -- a quoted jq/awk program or a
            backslash continuation may span physical lines -- while retaining an unquoted newline as
            a command boundary. Then split the resulting tokens on separator TOKENS, which the lexer
            has already distinguished from separator characters appearing inside quotes.
            """
            groups: list[list[str]] = []
            pending: list[str] = []

            def lex_logical_line(raw_line: str) -> list[str] | None:
                try:
                    line_lexer = shlex.shlex(raw_line, posix=True, punctuation_chars=";&|<>()")
                    line_lexer.whitespace_split = True
                    line_lexer.commenters = ""
                    return list(line_lexer)
                except ValueError as error:
                    # These two errors mean a physical line is incomplete, not malformed. The
                    # enclosing payload has already passed a whole-command shlex parse above, so
                    # accumulating the next physical line cannot turn invalid outer shell into a
                    # valid audited command. V14 line 114 was a valid multi-line single-quoted jq
                    # program and previously received two false malformed-command hits here.
                    if str(error) in {"No closing quotation", "No escaped character"}:
                        return None
                    raise

            for raw_line in text.replace("\r", "\n").split("\n"):
                if not pending and not raw_line.strip():
                    continue
                pending.append(raw_line)
                logical_line = "\n".join(pending)
                try:
                    line_words = lex_logical_line(logical_line)
                except ValueError:
                    hits.append(
                        {
                            "kind": "malformed_shell_command",
                            "line": str(line_number),
                            "command_sha256": _sha256(command.encode("utf-8")),
                        }
                    )
                    pending = []
                    continue
                if line_words is None:
                    continue
                pending = []
                # A REDIRECT TARGET IS DATA, NOT A COMMAND. Splitting on `<`/`>` as if they were
                # command separators turned `broker ... > out.mlir` into TWO simple commands: a valid
                # invocation plus a bare filename. The filename tripped the mixing rule AND resolved
                # under the candidate, so it was also reported as candidate execution outside the
                # broker. Measured 2026-09-03 on perf_agentic_20260903T184101Z__trial_00: 13 such
                # lines, 26 hits, a refused trial -- for the agent doing exactly what the prompt asks,
                # capturing emitted code to diff it. Only `;`, `&&`, `||`, `|` and `&` start a new
                # command; a redirection operator consumes its target as data.
                current: list[str] = []
                skip_target = False
                for token in line_words:
                    if skip_target:
                        skip_target = False
                        continue
                    if token and set(token) <= set("<>&") and ("<" in token or ">" in token):
                        skip_target = True
                        continue
                    if token and all(character in ";&|" for character in token):
                        if current:
                            groups.append(current)
                        current = []
                    else:
                        current.append(token)
                if current:
                    groups.append(current)
            if pending:
                hits.append(
                    {
                        "kind": "malformed_shell_command",
                        "line": str(line_number),
                        "command_sha256": _sha256(command.encode("utf-8")),
                    }
                )
            return groups

        simple_total = 0
        simple_brokered = 0

        def _audit_simple(sub_words: list[str]) -> None:
            nonlocal simple_total, simple_brokered
            simple_total += 1
            simple = " ".join(sub_words)
            invokes = len(sub_words) >= 2 and sub_words[0] in ("python", "python3") and sub_words[1] == BROKER_NAME
            if invokes:
                # A USAGE PROBE IS NOT AN INVOCATION. `python3 <broker> --help` names no action and
                # carries no bindings, so it executes nothing; the broker refuses it as an undeclared
                # action and now records that refusal. Treating interface discovery as tool-access
                # misuse refused a whole trial on 2026-09-03 (perf_agentic_..._trial_02) for one
                # `--help` among 32 commands, while its other 21 invocations were clean.
                if len(sub_words) == 3 and sub_words[2] in ("--help", "-h", "--usage"):
                    return
                if len(sub_words) < 3:
                    hits.append(
                        {
                            "kind": "invalid_broker_invocation",
                            "line": str(line_number),
                            "command_sha256": _sha256(command.encode("utf-8")),
                        }
                    )
                    return
                action = sub_words[2]
                bindings = sub_words[3:]
                binding_names = [value.split("=", 1)[0] for value in bindings if "=" in value]
                if (
                    action in action_names
                    and len(binding_names) == len(bindings)
                    and len(binding_names) == len(set(binding_names))
                    and all(binding_names)
                    and not any(value and all(character in ";&|<>()" for character in value) for value in sub_words)
                    # Substitution is banned across the WHOLE payload, not just this simple
                    # command: `input_mlir=$(pwd)/x.mlir` must fail closed even though the lexer
                    # may have already expanded or split it away from this group.
                    and not any(token in payload_text for token in ("`", "$"))
                ):
                    simple_brokered += 1
                    broker_invocations.append(
                        {
                            "line": line_number,
                            "action": action,
                            "bindings_sha256": _sha256(_canonical_json(sorted(bindings))),
                        }
                    )
                else:
                    hits.append(
                        {
                            "kind": "invalid_broker_invocation",
                            "line": str(line_number),
                            "command_sha256": _sha256(command.encode("utf-8")),
                        }
                    )
                return
            # Not an invocation. Naming the broker is allowed ONLY as an argument to a read-only
            # inspection verb -- which the prompt requires, since the agent must verify the broker
            # exists before it will proceed. Every other mention (copy, rename, link, interpreter
            # -c, unknown verb) is an invalid invocation.
            if BROKER_NAME.lower() in simple.lower():
                verb = Path(sub_words[0]).name.lower() if sub_words else ""
                # The existing quote-aware shell parser identifies the executable
                # position. A search operand mentioning the broker is not a call.
                # Ripgrep's preprocessor, however, executes another program: it
                # cannot receive this read-only exemption (including --pre=...).
                search_exec = verb in {"rg", "ripgrep"} and (
                    any(value == "--pre" or value.startswith("--pre=") for value in sub_words[1:])
                    or any(value in payload_text for value in ("`", "$"))
                )
                if verb not in _READ_ONLY_VERBS or BROKER_NAME in sub_words[:1] or search_exec:
                    hits.append(
                        {
                            "kind": "invalid_broker_invocation",
                            "line": str(line_number),
                            "command_sha256": _sha256(command.encode("utf-8")),
                        }
                    )

        # Heredoc bodies are peeled off FIRST: they are interpreter source, not further commands.
        heredoc_bodies, command_text = _split_heredocs(payload_text)
        simple_groups = _simple_commands(command_text)
        for _simple in simple_groups:
            _audit_simple(_simple)
        # `brokered` suppresses the candidate-execution and target-tool checks below, so it must mean
        # EVERY simple command was a clean broker invocation -- not merely that one of them was.
        # `python3 <broker> candidate-parse ... ; ./target-opt x.mlir` has a valid invocation AND an
        # unbrokered target-tool run; treating that as brokered would switch off exactly the check that
        # catches it (pinned by test_wrapped_broker_compound_rename_and_python_exec_forms_fail_closed).
        brokered = simple_total > 0 and simple_brokered == simple_total
        # A broker invocation must stand alone. Mixing one with unbrokered work in a single shell
        # command -- `python3 <broker> candidate-parse ... ; ./target-opt x.mlir` -- is how brokered
        # and unbrokered execution get laundered into one audited line, so it is refused even though
        # the invocation half is well formed. A batch that is ENTIRELY broker calls is not mixing and
        # stays legal, which is what lets the agent issue its probe set one call per line.
        if simple_brokered and simple_brokered != simple_total:
            hits.append(
                {
                    "kind": "invalid_broker_invocation",
                    "line": str(line_number),
                    "command_sha256": _sha256(command.encode("utf-8")),
                }
            )
        # A NEGATED SEARCH PREDICATE NAMES WHAT MUST NOT BE READ.  Treating
        # `find /perf-corpus -not -name golden.yaml` as reconnaissance refused a clean PR trial
        # whose command did exactly what the answer-surface policy requires.  Ripgrep expresses
        # the same exclusion as `--glob '!golden.yaml'`; that spelling refused the direct PQ run
        # even though the excluded oracle file was never searched.  Remove only the pattern
        # operand of an immediately negated find predicate or an explicitly negated rg glob;
        # every positive predicate, every other command, and every heredoc body remains
        # fail-closed.
        reconnaissance_texts: list[str] = []
        negatable_find_predicates = {
            "-name",
            "-iname",
            "-path",
            "-ipath",
            "-wholename",
            "-iwholename",
            "-regex",
            "-iregex",
        }
        reconnaissance_groups = list(simple_groups)
        nested_shell_sources: set[str] = set()
        for group in reconnaissance_groups:
            if group and Path(group[0]).name in {"bash", "sh", "dash", "zsh", "ksh"}:
                for position, value in enumerate(group[1:], 1):
                    if value in {"-c", "-lc", "-lic", "-ic"} and position + 1 < len(group):
                        source = group[position + 1]
                        if source not in nested_shell_sources:
                            nested_shell_sources.add(source)
                            reconnaissance_groups.extend(_simple_commands(source))
                        break
            filtered: list[str] = []
            index = 0
            is_find = bool(group) and Path(group[0]).name.lower() == "find"
            is_rg = bool(group) and Path(group[0]).name.lower() in ("rg", "ripgrep")
            while index < len(group):
                if (
                    is_find
                    and group[index] in ("-not", "!")
                    and index + 2 < len(group)
                    and group[index + 1].lower() in negatable_find_predicates
                ):
                    filtered.extend(group[index : index + 2])
                    index += 3
                    continue
                if (
                    is_rg
                    and group[index] in ("-g", "--glob")
                    and index + 1 < len(group)
                    and group[index + 1].startswith("!")
                ):
                    filtered.append(group[index])
                    index += 2
                    continue
                if is_rg and (group[index].startswith("--glob=!") or group[index].startswith("-g!")):
                    index += 1
                    continue
                filtered.append(group[index])
                index += 1
            # Shape matching applies to parsed operands, never the whole shell
            # sentence: a private package named 'feedback' is not the public
            # action 'tuning-gsim-feedback', but 'cat feedback' still names it.
            reconnaissance_texts.extend(filtered)
            # Attached option/env/broker bindings still name exact path values;
            # their left-hand flag spelling is not itself a private module read.
            reconnaissance_texts.extend(word.partition("=")[2] for word in filtered if "=" in word)
        reconnaissance_texts.extend(body.lower() for body in heredoc_bodies)
        source_words: list[str] = []
        for text in reconnaissance_texts:
            # The shell parser retains quoted Python source as one word. Recover
            # actual import targets and literal paths with its grammar, including
            # spaced/parenthesized from-imports and strings passed to open/exec.
            pending = [text]
            seen: set[str] = set()
            while pending:
                source = pending.pop()
                if source in seen:
                    continue
                seen.add(source)
                try:
                    tree = ast.parse(source)
                except (SyntaxError, ValueError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        source_words.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        source_words.append(node.module or "")
                        source_words.extend(alias.name for alias in node.names)
                        source_words.extend(f"{node.module}.{alias.name}" for alias in node.names if node.module)
                    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                        source_words.append(node.value)
                        pending.append(node.value)
        if any(audit_token_in(text, answer_tokens) for text in (*reconnaissance_texts, *source_words)):
            hits.append(
                {
                    "kind": "answer_reconnaissance",
                    "line": str(line_number),
                    "command_sha256": _sha256(command.encode("utf-8")),
                }
            )

        candidate_root = candidate.resolve(strict=True)

        def candidate_path(value: str, *, must_exist: bool) -> Path | None:
            raw = Path(value)
            possible_paths = [raw] if raw.is_absolute() else [candidate.parent / raw, candidate / raw]
            for possible in possible_paths:
                try:
                    resolved = possible.resolve(strict=must_exist)
                    resolved.relative_to(candidate_root)
                except (OSError, ValueError):
                    continue
                if not must_exist or resolved.exists():
                    return resolved
            return None

        # A candidate copied to an untracked host path and executed from there would evade the
        # ordinary "path is under candidate" check.  Reject the copy-out itself, regardless of whether
        # the later segment ran successfully.
        segments: list[list[str]] = [[]]
        for value in words:
            if value and all(character in ";&|" for character in value):
                segments.append([])
            else:
                segments[-1].append(value)
        for segment in segments:
            if not segment or Path(segment[0]).name not in ("cp", "install", "mv"):
                continue
            operands = [value for value in segment[1:] if not value.startswith("-")]
            if len(operands) < 2:
                continue
            destination = operands[-1]
            if (
                any(candidate_path(source, must_exist=True) is not None for source in operands[:-1])
                and candidate_path(destination, must_exist=False) is None
            ):
                hits.append(
                    {
                        "kind": "candidate_code_copied_outside",
                        "line": str(line_number),
                        "command_sha256": _sha256(command.encode("utf-8")),
                    }
                )
                break

        # WHICH SIMPLE COMMAND OWNS THE FLAG. These checks used to run against `words` -- the
        # flattened token list of the WHOLE payload -- which mis-attributes flags across command
        # boundaries. Measured 2026-09-03 on perf_stage_20260903T151801Z: the agent's own integrity
        # self-check, `python3 - <<'PY' ... PY` followed by `stat -c '%n %s %a' <control files>`, was
        # reported as `candidate_execution_outside_broker` because `"-c" in words` found STAT's flag,
        # took `'%n %s %a'` to be the Python source, failed to parse it, and took the fail-closed
        # branch. Two such lines refused an otherwise clean 54-command round in which every one of the
        # five required broker actions had been invoked. Ownership of a flag is a property of the
        # simple command it appears in, so the test has to be applied there.
        def _python_source_reads_candidate(source: str) -> bool:
            """True if this Python source opens candidate bytes -- or cannot be cleared at all."""
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                return True  # unparseable source cannot be cleared; fail closed
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "open"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and candidate_path(node.args[0].value, must_exist=True) is not None
                ):
                    return True
            return False

        reads_candidate_source = False
        direct_candidate = False
        for sub_words in simple_groups:
            if not sub_words:
                continue
            verb = Path(sub_words[0]).name.lower()
            if verb in ("python", "python3"):
                # `python -c` has no script argv for the structural check below, so inspect its AST
                # and reject code that opens candidate bytes, including exec(open(...).read()).
                if "-c" in sub_words:
                    code_index = sub_words.index("-c") + 1
                    if code_index >= len(sub_words) or _python_source_reads_candidate(sub_words[code_index]):
                        reads_candidate_source = True
                # `python3 - <<EOF` takes its source from stdin, where a heredoc body is the script.
                # Auditing `-c` while ignoring `<<` would leave the same execution one keystroke away.
                if any(_python_source_reads_candidate(body) for body in heredoc_bodies):
                    reads_candidate_source = True
            execution_token: str | None = None
            if verb in ("python", "python3", "bash", "sh"):
                for value in sub_words[1:]:
                    if value.startswith("-"):
                        if value in ("-c", "-lc", "-m"):
                            break
                        continue
                    execution_token = value
                    break
            else:
                execution_token = sub_words[0]
            if execution_token:
                resolved_execution = candidate_path(execution_token, must_exist=True)
                if resolved_execution is not None and resolved_execution.is_file():
                    direct_candidate = True
                if Path(execution_token).name.lower() in entry_tokens:
                    direct_candidate = True
        if reads_candidate_source:
            hits.append(
                {
                    "kind": "candidate_execution_outside_broker",
                    "line": str(line_number),
                    "command_sha256": _sha256(command.encode("utf-8")),
                }
            )
        if not brokered and direct_candidate and not reads_candidate_source:
            hits.append(
                {
                    "kind": "candidate_execution_outside_broker",
                    "line": str(line_number),
                    "command_sha256": _sha256(command.encode("utf-8")),
                }
            )
        if not brokered and target_tool_tokens & command_words:
            hits.append(
                {
                    "kind": "target_tool_outside_broker",
                    "line": str(line_number),
                    "command_sha256": _sha256(command.encode("utf-8")),
                }
            )

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            record = json.loads(line)
        except ValueError:
            hits.append({"kind": "malformed_transcript", "line": str(line_number)})
            continue
        if not isinstance(record, dict):
            hits.append({"kind": "malformed_transcript", "line": str(line_number)})
            continue
        event_type = record.get("type")
        if event_type == "codex_unparsed":
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
        if event_type == "codex_summary":
            unknown_types = record.get("unknown_types")
            if not isinstance(unknown_types, list):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            elif unknown_types:
                hits.append({"kind": "unknown_command_event", "line": str(line_number)})
        if event_type == "thread.started":
            thread_id = record.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            else:
                current_thread = thread_id

        item = record.get("item")
        if event_type in ("item.started", "item.completed"):
            if not isinstance(item, dict):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            elif item.get("type") not in known_native_items:
                hits.append({"kind": "unknown_command_event", "line": str(line_number)})
            elif item.get("type") == "command_execution":
                item_id, command = item.get("id"), item.get("command")
                if not isinstance(item_id, str) or not item_id or not isinstance(command, str) or not command.strip():
                    hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                else:
                    key = (current_thread, item_id)
                    previous = native_commands.get(key)
                    if previous is not None and previous != command:
                        hits.append({"kind": "conflicting_command_event", "line": str(line_number)})
                    elif previous is None:
                        native_commands[key] = command
                        audit_command(command, line_number)
        elif isinstance(item, dict) and item.get("type") == "command_execution":
            hits.append({"kind": "unknown_command_event", "line": str(line_number)})

        message = record.get("message")
        if message is not None and not isinstance(message, dict):
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            continue
        content = message.get("content") if isinstance(message, dict) else []
        if content is None:
            content = []
        if not isinstance(content, list):
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                continue
            command = tool_input.get("command")
            if command is None:
                # File edits and other non-command tools legitimately have no command field.
                continue
            if not isinstance(command, str) or not command.strip():
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                continue
            audit_command(command, line_number)
    if commands_seen <= 0:
        hits.append({"kind": "no_command_evidence", "line": "0"})
    return {
        "clean": not hits,
        "hits": hits,
        "commands_seen": commands_seen,
        "candidate": str(candidate),
        "broker_required": BROKER_NAME,
        "broker_invocations": broker_invocations,
    }
