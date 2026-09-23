"""POST-HOC READ AUDIT: what did the agent actually read, and did any of it intersect the answer key?

WHY DETECTION IS NEEDED AS WELL AS PREVENTION. A clean room (:mod:`merlin.targetgen.sandbox.cleanroom`)
guarantees a property of the TREE — no answer surface is reachable by walking down from the workspace.
It cannot guarantee a property of the PROCESS. An agent that opens an absolute path elsewhere on the
filesystem is not stopped by a tidy workspace, and on a host where the isolation sandbox cannot build a
namespace nothing stops it: preventing that needs kernel isolation or a separate uid, and this repo has
neither available right now. So the second line of defence is to read back what the agent DID, and to
say so.

THE SINGLE MOST IMPORTANT PROPERTY OF THIS MODULE is that ``UNKNOWN`` never reads as clean. A run whose
event log is missing, truncated, unparseable, or full of event kinds this module does not understand has
not been shown to be clean — it has not been audited. Every accessor here is phrased so that the only
affirmative answer is :data:`CLEAN`: :attr:`ReadAudit.is_clean` is true for one verdict and
:meth:`ReadAudit.require_clean` raises for the other two. There is deliberately no ``not contaminated``
accessor, because that is the spelling under which an unauditable run gets cited as a clean one.

WHAT THE EVENT LOG ACTUALLY CONTAINS. The driver writes one JSONL line per emitted event, wrapped as
``{"seq", "arrived_at", "event"}`` — or ``{"seq", "arrived_at", "unparsed"}`` when the bytes were not
JSON, which is exactly the case this module must not silently drop. The raw sidecar holds the bare event
object, and both spellings are accepted. There is NO structured file-read event: every read is a shell
command inside a ``command_execution`` item, whose ``command`` is one flat string (``/bin/bash -lc
"<script>"``), so the paths have to be recovered by parsing that script. ``file_change`` items carry
absolute paths and a verb, and are audited as writes.

DERIVED, NEVER HARD-CODED. The withheld set is
:func:`merlin.targetgen.sandbox.answer_surfaces.answer_surfaces` plus the audit TOKENS derived from the
same descriptor and registry, so this module knows nothing about any particular target, and a target
that grows a new answer surface is covered the day its descriptor says so. The advisory/violation split
reuses :func:`merlin.targetgen.sandbox.answer_surfaces.audit_hit_is_violation`, which fails closed: a hit
kind that has not been deliberately declared advisory counts as a violation.

NO REGEX. Commands are parsed structurally with :mod:`shlex`, for the reason the repo bans regex in
library code: a pattern narrow enough to be precise silently drops the spellings it did not anticipate,
and a read this module fails to see is reported as CLEAN.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from merlin.common.access import audit_token_in
from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces, audit_hit_is_violation, audit_tokens
from merlin.targetgen.target_experiment import TargetExperiment

#: Hit kinds meaning "a withheld path was NAMED but we cannot tell whether bytes came back". They are
#: neither a clearance nor a finding, and they drive UNKNOWN.
#:
#: This third outcome exists because forcing every hit into clean-or-dirty produces a dishonest verdict
#: in the commonest real case: a withheld name inside a COMPOUND command, where the captured output
#: belongs to the whole pipeline and cannot be attributed to the one simple command that named it. The
#: truthful answer there is "go and look", not an accusation and not a clearance.
INDETERMINATE_KINDS: frozenset[str] = frozenset({"indeterminate_outcome", "unresolved_expansion"})

#: The three verdicts. There is no fourth, and no boolean shorthand for "not contaminated".
CLEAN = "CLEAN"
CONTAMINATED = "CONTAMINATED"
UNKNOWN = "UNKNOWN"

#: Event envelope kinds the driver emits. DECLARED, so an envelope this module has never seen forces
#: UNKNOWN rather than being skipped — an unrecognised envelope may be carrying a read we cannot see.
KNOWN_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "thread.started",
        "turn.started",
        "turn.completed",
        "turn.failed",
        "item.started",
        "item.updated",
        "item.completed",
        "error",
    }
)

#: Item kinds the driver emits. Same rule, same reason: an unknown item kind is UNKNOWN, never ignored.
#: ``todo_list`` is here because the tool emits it even though the driver's own constants omit it.
KNOWN_ITEM_TYPES: frozenset[str] = frozenset(
    {
        "command_execution",
        "agent_message",
        "reasoning",
        "error",
        "file_change",
        "mcp_tool_call",
        "web_search",
        "todo_list",
        "collab_tool_call",
    }
)

#: ``collab_tool_call`` carries a ``tool`` field with its own open vocabulary, so accepting the ITEM
#: kind outright would accept every tool that kind ever grows. These are the tools observed to touch no
#: filesystem; any other forces UNKNOWN, which keeps the fail-closed rule one level deeper rather than
#: abandoning it at the item boundary.
FILESYSTEM_FREE_COLLAB_TOOLS: frozenset[str] = frozenset({"wait"})

#: Filenames a run's event log is written under, most informative first. Declared here so "where is the
#: log" is one answer rather than a glob repeated at each call site.
EVENT_LOG_GLOBS: tuple[str, ...] = (
    "**/*.codex_events.timestamped.jsonl",
    "**/*.codex_events.raw.jsonl",
    "**/events.*.raw.jsonl",
)

#: Shell words that introduce a nested script rather than a path.
_SHELL_PROGRAMS = frozenset({"bash", "sh", "dash", "zsh", "ksh"})
_SHELL_SCRIPT_FLAGS = frozenset({"-c", "-lc", "-lic", "-ic"})
#: Programs whose FIRST operand is a pattern, not a path (so naming a withheld token there is a search,
#: not a read). Getting this wrong in either direction is costly: treat a pattern as a path and every
#: agent that greps its own task card is accused; treat a path as a pattern and a real read is missed.
_PATTERN_FIRST = frozenset({"grep", "egrep", "fgrep", "rg", "ag", "ack", "pt"})
#: Programs whose first operand is a script, not a path.
_SCRIPT_FIRST = frozenset({"sed", "awk", "gawk", "mawk", "perl", "jq"})
#: Flags and operators whose FOLLOWING word is a pattern to match, not a file to open. ``find ... !
#: -name golden.yaml`` and ``rg -g '!**/golden.yaml'`` both EXCLUDE the answer key from a search; an
#: audit that reads those as reads of the answer key accuses an agent of the opposite of what it did.
_PATTERN_VALUE_FLAGS = frozenset(
    {
        "!",
        "-not",
        "-name",
        "-iname",
        "-path",
        "-ipath",
        "-wholename",
        "-regex",
        "-iregex",
        "-lname",
        "-g",
        "--glob",
        "-e",
        "--regexp",
        "--include",
        "--exclude",
        "--exclude-dir",
    }
)
#: Programs whose operands are text to emit, not files to open. An agent that PRINTS the name of a
#: withheld file has not read it, and accusing it of one is how an audit teaches its readers to skim.
_TEXT_OPERANDS = frozenset({"echo", "printf"})
#: Programs that surface names without surfacing contents.
_LISTING_PROGRAMS = frozenset({"ls", "find", "fd", "stat", "file", "wc", "basename", "dirname", "readlink", "tree"})
#: Separators that end one simple command inside a compound one.
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "\n"})


class EventLogUnreadable(RuntimeError):
    """The log could not be read at all. Surfaced as UNKNOWN, never swallowed."""


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class ObservedRead:
    """One path the agent's own commands named, with how it was named and how it turned out."""

    path: str
    command_sha: str
    program: str
    role: str  # "path" | "pattern" | "script" | "unsplit"
    exit_code: int | None
    produced_output: bool
    seq: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "program": self.program,
            "role": self.role,
            "exit_code": self.exit_code,
            "produced_output": self.produced_output,
            "seq": self.seq,
            "command_sha256": self.command_sha,
        }


@dataclass(frozen=True)
class AuditHit:
    """A read that intersects the withheld set. ``kind`` is the declared audit vocabulary."""

    kind: str
    path: str
    surface: str
    detail: str = ""
    seq: int = 0

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "path": self.path, "surface": self.surface, "detail": self.detail, "seq": self.seq}


@dataclass(frozen=True)
class ReadAudit:
    """The verdict for one run, and the evidence behind it."""

    verdict: str
    reason: str
    logs: tuple[str, ...] = ()
    reads: tuple[ObservedRead, ...] = ()
    hits: tuple[AuditHit, ...] = ()
    events_parsed: int = 0
    events_unparsed: int = 0
    unknown_kinds: tuple[str, ...] = ()
    incomplete_reasons: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        """The ONLY affirmative accessor. UNKNOWN and CONTAMINATED are both false."""
        return self.verdict == CLEAN

    def __bool__(self) -> bool:
        # ``if audit:`` means "this run was shown to be clean". An unauditable run is falsy.
        return self.is_clean

    @property
    def violations(self) -> tuple[AuditHit, ...]:
        """Hits that demonstrate withheld content reached the agent.

        An INDETERMINATE hit is deliberately not one: it is reported through
        :attr:`incomplete_reasons`, where it makes the verdict UNKNOWN instead of quietly counting as
        either a finding or a clearance.
        """
        return tuple(
            hit for hit in self.hits if hit.kind not in INDETERMINATE_KINDS and audit_hit_is_violation(hit.as_dict())
        )

    @property
    def indeterminate(self) -> tuple[AuditHit, ...]:
        """Hits where a withheld path was named and the outcome is not recoverable."""
        return tuple(hit for hit in self.hits if hit.kind in INDETERMINATE_KINDS)

    def require_clean(self, context: str = "") -> None:
        """Raise unless the verdict is CLEAN. The raising form exists so a caller cannot reach for a
        'well, it was not CONTAMINATED' reading of an unaudited run."""
        if self.is_clean:
            return
        prefix = f"{context}: " if context else ""
        raise AnswerKeyExposure(f"{prefix}{self.describe()}", self)

    def describe(self) -> str:
        head = f"{self.verdict} ({self.reason})"
        if self.verdict == CONTAMINATED:
            rows = "; ".join(f"{h.kind}:{h.path}->{h.surface}" for h in self.violations[:8])
            return f"{head}: {rows}"
        if self.verdict == UNKNOWN:
            why = "; ".join(self.incomplete_reasons[:8]) or self.reason
            return f"{head}: {why} — this run has NOT been shown to be clean"
        return f"{head}: {self.events_parsed} events, {len(self.reads)} path reads, no withheld path touched"

    def as_record(self) -> dict[str, object]:
        """The block a run artifact embeds. ``verdict`` is a string on purpose: a boolean here is how
        UNKNOWN becomes clean in someone's downstream summary."""
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "logs": list(self.logs),
            "events_parsed": self.events_parsed,
            "events_unparsed": self.events_unparsed,
            "unknown_kinds": list(self.unknown_kinds),
            "incomplete_reasons": list(self.incomplete_reasons),
            "n_reads": len(self.reads),
            "hits": [hit.as_dict() for hit in self.hits],
            "violations": [hit.as_dict() for hit in self.violations],
        }


class AnswerKeyExposure(RuntimeError):
    """Raised by :meth:`ReadAudit.require_clean`. Carries the audit so the caller records the verdict."""

    def __init__(self, message: str, audit: ReadAudit) -> None:
        super().__init__(message)
        self.audit = audit


# --------------------------------------------------------------------------- command parsing
def _simple_commands(script: str) -> list[list[str]]:
    """Split a shell script into simple commands, structurally.

    A compound command tested as one string is what produced false accusations in the earlier transcript
    audit — a batch of ten one-per-line calls counted as one suspicious blob. Each simple command is
    classified on its own, and a nested ``bash -c`` payload is descended into so that wrapping a read
    in a subshell does not hide it.
    """
    try:
        lexer = shlex.shlex(script, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        words = list(lexer)
    except ValueError:
        return []
    out: list[list[str]] = []
    current: list[str] = []
    for word in words:
        if word in _SEPARATORS:
            if current:
                out.append(current)
            current = []
            continue
        current.append(word)
    if current:
        out.append(current)
    # Descend into nested shell payloads.
    expanded: list[list[str]] = []
    for command in out:
        expanded.append(command)
        program = Path(command[0]).name if command else ""
        if program in _SHELL_PROGRAMS:
            for index, word in enumerate(command[1:], start=1):
                if word in _SHELL_SCRIPT_FLAGS and index + 1 < len(command):
                    expanded += _simple_commands(command[index + 1])
                    break
    return expanded


def _operand_roles(command: Sequence[str]) -> list[tuple[str, str]]:
    """``(word, role)`` for each operand of one simple command.

    Roles are ``path`` (a file the command reads), ``pattern`` (a search expression) and ``script`` (an
    editor/interpreter program text). Options and their attached values are skipped; a bare ``--``
    ends option parsing.
    """
    if not command:
        return []
    program = Path(command[0]).name
    roles: list[tuple[str, str]] = []
    operand_index = 0
    options_done = False
    previous = ""
    for word in command[1:]:
        if not options_done and word == "--":
            options_done = True
            previous = word
            continue
        if previous in _PATTERN_VALUE_FLAGS:
            roles.append((word, "pattern"))
            previous = word
            continue
        if not options_done and word.startswith("-") and len(word) > 1:
            previous = word
            continue  # an option; its attached value is either glued on or the next operand we still see
        previous = word
        if program in _TEXT_OPERANDS:
            roles.append((word, "text"))
        elif word.startswith("!"):
            # A negation: `rg -g '!**/golden.yaml'` EXCLUDES the answer key from a search. Reading that
            # as a read of the answer key accuses an agent of the precise opposite of what it did.
            roles.append((word, "pattern"))
        elif operand_index == 0 and program in _PATTERN_FIRST:
            roles.append((word, "pattern"))
        elif operand_index == 0 and program in _SCRIPT_FIRST:
            roles.append((word, "script"))
        else:
            roles.append((word, "path"))
        operand_index += 1
    return roles


def _command_payload(command_string: str) -> tuple[str, list[list[str]]]:
    """The script the driver actually ran, and its simple commands.

    The driver always emits ``/bin/bash -lc "<script>"``; unwrapping that first is what makes the inner
    program (``sed``, ``cat``, ``rg``) the thing classified, rather than ``bash`` every time.
    """
    try:
        outer = shlex.split(command_string)
    except ValueError:
        return command_string, []
    if len(outer) >= 3 and Path(outer[0]).name in _SHELL_PROGRAMS and outer[1] in _SHELL_SCRIPT_FLAGS:
        script = outer[2]
    else:
        script = command_string
    return script, _simple_commands(script)


# --------------------------------------------------------------------------- the withheld set
class _Withheld:
    """The withheld set in both the shapes a transcript can reveal it.

    A transcript says two different kinds of thing about a path, so the set is consulted two ways and
    neither alone suffices. An ABSOLUTE path is matched against the derived answer surfaces by prefix,
    which is exact but only works for a spelling that exists on this host. A TOKEN (``golden.yaml``, a
    hidden corpus's trailing components, an oracle module's dotted stem) is matched as a substring,
    which catches a path spelled through a symlink, a path inside a clean room whose root differs, and
    a path on a host this audit is not running on. Both come from the same descriptor-derived
    registry, so they cannot disagree about what an answer is.
    """

    def __init__(self, te: TargetExperiment, *, repo: Path | None = None) -> None:
        self.repo = (repo or repo_root()).absolute()
        self.surfaces = [s.path.absolute() for s in answer_surfaces(te)]
        self._surface_set = set(self.surfaces)
        tokens = audit_tokens(te)
        self.tokens = tuple(
            value.lower() for key in ("answer", "grader", "oracle_subpath") for value in tokens.get(key, ())
        )

    def surface_for(self, candidate: Path) -> str | None:
        for ancestor in (candidate, *candidate.parents):
            if ancestor in self._surface_set:
                return str(ancestor)
        return None

    def token_in(self, text: str) -> str | None:
        """A withheld token named by ``text``, or None — matched by SHAPE, not by bare substring.

        A token carrying a ``/`` or a ``.`` is specific enough to match anywhere in a word
        (``golden.yaml``, ``capsules/hidden``, ``merlin/runtime/reference``). A BARE STEM is not: the
        grader registry contributes stems like ``decode`` and ``trace_check``, and substring-matching
        those against arbitrary shell text flags every ``jq`` filter that mentions a ``decoded`` field.
        Measured on the archived transcripts, that single rule produced 30 of 34 accusations, all of
        them against agents doing exactly what they were asked to do. A bare stem therefore has to BE a
        path component (optionally with an extension), which is the only way it names a module.
        """
        return audit_token_in(text, self.tokens)


def _resolve(word: str, cwd: Path | None, repo: Path) -> list[Path]:
    """Candidate absolute spellings of one command word. A relative word is resolved against the run's
    working directory AND the checkout, because a transcript rarely records which one was in force."""
    path = Path(word)
    if path.is_absolute():
        return [path]
    bases = [base for base in (cwd, repo) if base is not None]
    return [base / path for base in bases]


# --------------------------------------------------------------------------- classification
def _classify(
    *,
    withheld: _Withheld,
    surface: str,
    role: str,
    program: str,
    word: str,
    exit_code: int | None,
    produced_output: bool,
    allowed: Sequence[Path],
    workspace: Path | None,
    compound: bool,
) -> str:
    """Which declared audit kind this hit is. Advisory kinds mean the protection worked; the rest are
    violations, and anything not deliberately declared advisory counts as one."""
    if role in ("pattern", "script", "text"):
        # A search expression, an interpreter program, or text being printed. None of these is a file
        # the command opened, and treating them as one is the documented way an audit becomes noise.
        return "pattern_mention"
    if "$" in word or "`" in word:
        # An unexpanded variable or command substitution: the literal part names something withheld,
        # but what the shell actually opened is not recoverable from the transcript.
        return "unresolved_expansion"
    candidate = Path(word)
    if workspace is not None and candidate.is_absolute():
        if candidate == workspace or workspace in candidate.parents:
            return "owned_read"
    for grant in allowed:
        if candidate.is_absolute() and (candidate == grant or grant in candidate.parents):
            # A grant DEEPER than the surface out-ranks it, the same longest-prefix rule the mount
            # policy and the clean room use.
            if len(grant.parts) > len(Path(surface).parts):
                return "granted_read"
    if exit_code is not None and exit_code != 0:
        return "blocked_probe"
    if exit_code is not None and not produced_output:
        return "blocked_probe"
    if program in _LISTING_PROGRAMS:
        return "recon_probe"
    if compound:
        # The captured output belongs to the whole pipeline, so it cannot be attributed to this one
        # simple command. Saying so is the honest answer; picking either verdict would be a guess.
        return "indeterminate_outcome"
    return "path_read"


# --------------------------------------------------------------------------- the log reader
def _iter_events(path: Path) -> Iterable[tuple[int, object, str | None]]:
    """``(seq, event_or_None, unparsed_or_None)`` per line, in both envelope spellings.

    Streamed rather than read whole: one command's captured output reaches tens of kilobytes and a
    single round's log reaches megabytes.
    """
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for index, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                document = json.loads(text)
            except ValueError:
                yield index, None, text
                continue
            if not isinstance(document, dict):
                yield index, None, text
                continue
            if "unparsed" in document:
                yield int(document.get("seq", index)), None, str(document["unparsed"])
                continue
            if "event" in document:
                yield int(document.get("seq", index)), document["event"], None
                continue
            if "type" in document:  # the raw sidecar: the bare event object
                yield index, document, None
                continue
            yield index, None, text


# --------------------------------------------------------------------------- the audit
def audit_event_logs(
    logs: Sequence[Path],
    te: TargetExperiment,
    *,
    bundle: Mapping[str, object] | None = None,
    workspace: Path | None = None,
    cwd: Path | None = None,
    repo: Path | None = None,
) -> ReadAudit:
    """Audit one or more event logs for reads that intersect the derived answer surfaces.

    Returns CLEAN only when every line parsed, every event kind was recognised, and no read landed on
    a withheld path. Anything else is CONTAMINATED (a positive finding) or UNKNOWN (no finding either
    way, which is not the same as a negative one).
    """
    repo = (repo or repo_root()).absolute()
    logs = [Path(p) for p in logs]
    if not logs:
        return ReadAudit(verdict=UNKNOWN, reason="no_event_log", incomplete_reasons=("no event log was given",))

    try:
        withheld = _Withheld(te, repo=repo)
    except Exception as exc:  # noqa: BLE001 — an underivable withheld set cannot yield a clean verdict
        return ReadAudit(
            verdict=UNKNOWN,
            reason="withheld_set_underivable",
            logs=tuple(str(p) for p in logs),
            incomplete_reasons=(f"answer surfaces could not be derived: {exc}",),
        )

    allowed: list[Path] = []
    if bundle:
        from merlin.targetgen.sandbox.bwrap import resolve_grant

        for entry in bundle.get("allowed") or []:
            if isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
                allowed.append(resolve_grant(str(entry["path"]), repo).absolute())

    reads: list[ObservedRead] = []
    hits: list[AuditHit] = []
    unknown_kinds: set[str] = set()
    incomplete: list[str] = []
    parsed = 0
    unparsed = 0
    # A command's outcome arrives on ``item.completed``; its ``item.started`` twin carries the same
    # text. Pair by item id so a command is audited once, with its outcome.
    commands: dict[str, dict[str, object]] = {}
    order: list[str] = []

    for log in logs:
        if not log.is_file():
            incomplete.append(f"event log not found: {log}")
            continue
        try:
            stream = list(_iter_events(log))
        except OSError as exc:
            incomplete.append(f"event log unreadable: {log} ({exc})")
            continue
        if not stream:
            incomplete.append(f"event log is empty: {log}")
            continue
        for seq, event, raw in stream:
            if raw is not None:
                unparsed += 1
                incomplete.append(f"unparseable line at seq {seq} in {log.name}")
                continue
            if not isinstance(event, dict):
                unparsed += 1
                incomplete.append(f"event is not an object at seq {seq} in {log.name}")
                continue
            parsed += 1
            kind = event.get("type")
            if not isinstance(kind, str) or kind not in KNOWN_EVENT_TYPES:
                unknown_kinds.add(f"event.type={kind!r}")
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if not isinstance(item_type, str) or item_type not in KNOWN_ITEM_TYPES:
                unknown_kinds.add(f"item.type={item_type!r}")
                continue
            if item_type == "collab_tool_call":
                tool = item.get("tool")
                if not isinstance(tool, str) or tool not in FILESYSTEM_FREE_COLLAB_TOOLS:
                    unknown_kinds.add(f"collab_tool_call.tool={tool!r}")
                continue
            if item_type == "command_execution":
                key = str(item.get("id") or f"{log}:{seq}")
                record = commands.setdefault(key, {"seq": seq})
                if key not in order:
                    order.append(key)
                if isinstance(item.get("command"), str):
                    record["command"] = item["command"]
                if kind == "item.completed":
                    record["exit_code"] = item.get("exit_code")
                    record["output"] = bool(str(item.get("aggregated_output") or ""))
                    record["completed"] = True
            elif item_type == "file_change":
                for change in item.get("changes") or []:
                    if not isinstance(change, Mapping) or not isinstance(change.get("path"), str):
                        continue
                    written = Path(str(change["path"]))
                    surface = withheld.surface_for(written) or (
                        str(withheld.token_in(str(written))) if withheld.token_in(str(written)) else None
                    )
                    if surface:
                        hits.append(
                            AuditHit(
                                kind="answer_surface_write",
                                path=str(written),
                                surface=surface,
                                detail=str(change.get("kind") or "change"),
                                seq=seq,
                            )
                        )

    for key in order:
        record = commands[key]
        command_string = str(record.get("command") or "")
        if not command_string:
            continue
        seq = int(record.get("seq") or 0)
        exit_code = record.get("exit_code") if record.get("completed") else None
        produced = bool(record.get("output")) if record.get("completed") else False
        sha = _short_sha(command_string)
        script, simple = _command_payload(command_string)
        if not simple:
            # The command did not lex — a nested heredoc, an unbalanced quote. Its raw text is still
            # checked for a withheld token. A command naming none cannot have opened one by any
            # spelling of its operands, which is the same standard applied everywhere else here, so it
            # does not make the run unauditable; recording every such command as UNKNOWN would spend
            # the word on shell quoting and leave nothing for a real gap.
            token = withheld.token_in(script)
            if token:
                incomplete.append(f"command at seq {seq} names {token} and could not be parsed ({sha})")
                hits.append(AuditHit("indeterminate_outcome", script[:200], token, "unparseable command", seq))
            continue
        for command in simple:
            if not command:
                continue
            program = Path(command[0]).name
            for word, role in _operand_roles(command):
                surface: str | None = None
                matched = word
                for candidate in _resolve(word, cwd, repo):
                    surface = withheld.surface_for(candidate)
                    if surface:
                        matched = str(candidate)
                        break
                if surface is None:
                    token = withheld.token_in(word)
                    surface = token
                if surface is None:
                    continue
                reads.append(
                    ObservedRead(
                        path=matched,
                        command_sha=sha,
                        program=program,
                        role=role,
                        exit_code=exit_code if isinstance(exit_code, int) else None,
                        produced_output=produced,
                        seq=seq,
                    )
                )
                if not record.get("completed"):
                    # We saw the command start and never saw it finish, so we cannot say whether it
                    # returned withheld bytes. That is UNKNOWN, not "probably fine".
                    incomplete.append(f"command touching a withheld path at seq {seq} has no outcome ({sha})")
                hits.append(
                    AuditHit(
                        kind=_classify(
                            withheld=withheld,
                            surface=surface,
                            role=role,
                            program=program,
                            word=matched,
                            exit_code=exit_code if isinstance(exit_code, int) else None,
                            produced_output=produced,
                            allowed=allowed,
                            workspace=workspace,
                            compound=len(simple) > 1,
                        ),
                        path=matched,
                        surface=surface,
                        detail=program,
                        seq=seq,
                    )
                )

    indeterminate = [hit for hit in hits if hit.kind in INDETERMINATE_KINDS]
    incomplete = [
        *incomplete,
        *(f"{hit.kind} at seq {hit.seq}: {hit.path[:120]} names {hit.surface}" for hit in indeterminate),
    ]
    violations = [hit for hit in hits if hit.kind not in INDETERMINATE_KINDS and audit_hit_is_violation(hit.as_dict())]
    if violations:
        verdict, reason = CONTAMINATED, "withheld content reached the agent"
    elif unparsed or unknown_kinds or incomplete or not parsed:
        verdict, reason = UNKNOWN, "the event log could not be audited completely"
    else:
        verdict, reason = CLEAN, "no read intersected the derived answer surfaces"

    if unknown_kinds:
        incomplete = [*incomplete, *(f"unrecognised event kind: {kind}" for kind in sorted(unknown_kinds))]
    if not parsed and not incomplete:
        incomplete = ["no events were parsed"]

    return ReadAudit(
        verdict=verdict,
        reason=reason,
        logs=tuple(str(p) for p in logs),
        reads=tuple(reads),
        hits=tuple(hits),
        events_parsed=parsed,
        events_unparsed=unparsed,
        unknown_kinds=tuple(sorted(unknown_kinds)),
        incomplete_reasons=tuple(dict.fromkeys(incomplete)),
    )


def find_event_logs(run_dir: Path) -> list[Path]:
    """Every event log under a run directory, preferring the timestamped envelope.

    Only ONE family is returned: the timestamped and raw streams are the same events, and auditing both
    would double-count every read.
    """
    run_dir = Path(run_dir)
    for pattern in EVENT_LOG_GLOBS:
        found = sorted(p for p in run_dir.glob(pattern) if p.is_file())
        if found:
            return found
    return []


def audit_run(
    run_dir: Path,
    te: TargetExperiment,
    *,
    bundle: Mapping[str, object] | None = None,
    workspace: Path | None = None,
    repo: Path | None = None,
) -> ReadAudit:
    """Audit a whole run directory. A run with no event log is UNKNOWN — the case this exists for."""
    run_dir = Path(run_dir)
    logs = find_event_logs(run_dir)
    if not logs:
        return ReadAudit(
            verdict=UNKNOWN,
            reason="no_event_log",
            incomplete_reasons=(f"no agent event log under {run_dir}; this run cannot be audited",),
        )
    return audit_event_logs(logs, te, bundle=bundle, workspace=workspace, repo=repo)


def _short_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
