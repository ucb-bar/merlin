#!/usr/bin/env python3
"""Capsule-bench agent driver for the **Codex CLI** (``codex exec --json``).

Same contract as :mod:`opencode_agent` and :mod:`bedrock_agent`: ``run_round``
drives ONE round inside the prepared workspace and returns
``(rc, transcript_path)``, writing a transcript in the Claude-stream JSONL shape
the rest of the harness already reads (trajectory, token accounting, transcript
audit). Nothing downstream needs to learn a second schema.

Four things here are deliberate and easy to get wrong.

**1. Token subsets.** Codex reports ``input_tokens`` with ``cached_input_tokens``
and ``cache_write_input_tokens`` already *inside* it, and
``reasoning_output_tokens`` already inside ``output_tokens``. The Claude stream
shape this harness consumes means the opposite by ``input_tokens`` — it is the
*uncached* part, with cache reads counted separately. So the mapping subtracts
rather than copies; copying would inflate input by the cache-hit rate, which on
a warm run is most of the prompt.

**2. Usage only arrives on ``turn.completed``.** A ``turn.failed`` carries no
usage at all, so its tokens are *unknown*, not zero. The transcript records the
turn with usage omitted and a ``codex_usage_unreported`` marker, so a spend
figure derived from it is visibly a lower bound instead of a confident total.

**3. The events carry no timestamps.** Every time in the transcript is this
reader's arrival time, recorded as the line arrives. Capture therefore tees line
by line to ``codex_events.raw.jsonl`` before interpreting anything, so a timeout
or a kill still leaves the evidence (and the token counts) on disk.

**4. Instruction parity between arms.** Codex reads ``AGENTS.md`` from the
workspace; Claude Code reads ``CLAUDE.md``/``AGENT.md``. An arm that silently
gets extra instructions is not the same arm. This driver does not author either
file — it records which instruction files the prepared workspace actually
contains into the transcript's init record, so an asymmetry is visible in the
artifact rather than discovered later.

Sandboxing: at ``--sandbox bwrap`` the whole ``codex`` process runs inside the
harness's existing bwrap wrapper (which masks goldens and hidden inputs), while
model-generated commands run inside a second, deny-by-default Codex permission
profile. The second boundary keeps the writable credential available to the
Codex parent for token refresh, but unavailable to its candidate commands.
At ``--sandbox none`` Codex's own ``workspace-write`` sandbox is used.
"""

from __future__ import annotations

import json
import os
import selectors
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from merlin.common.paths import module_source_path

# --- The measured event contract (codex-cli 0.147.0) ------------------------
# Verified by capturing a live run, not read from documentation. Envelope types
# and item types are two separate vocabularies and are matched exactly; anything
# unrecognized is preserved in the transcript rather than dropped.
EVENT_THREAD_STARTED = "thread.started"
EVENT_TURN_STARTED = "turn.started"
EVENT_TURN_COMPLETED = "turn.completed"
EVENT_TURN_FAILED = "turn.failed"
EVENT_ITEM_STARTED = "item.started"
EVENT_ITEM_UPDATED = "item.updated"
EVENT_ITEM_COMPLETED = "item.completed"
EVENT_ERROR = "error"

ITEM_COMMAND_EXECUTION = "command_execution"
ITEM_AGENT_MESSAGE = "agent_message"
ITEM_REASONING = "reasoning"
ITEM_ERROR = "error"
ITEM_FILE_CHANGE = "file_change"
ITEM_MCP_TOOL_CALL = "mcp_tool_call"
ITEM_WEB_SEARCH = "web_search"
# Emitted by the tool and observed in captured sessions, though this driver handles neither: a plan
# update (``todo_list``, carried by ``item.updated``) and a sub-agent call (``collab_tool_call``).
ITEM_TODO_LIST = "todo_list"
ITEM_COLLAB_TOOL_CALL = "collab_tool_call"
_TOOL_ITEMS = (ITEM_COMMAND_EXECUTION, ITEM_FILE_CHANGE, ITEM_MCP_TOOL_CALL, ITEM_WEB_SEARCH)

#: The whole measured envelope and item vocabularies, ONE declaration. The read audit fails closed on
#: anything outside them, so a second copy that drifted would turn a real event into UNKNOWN (or, the
#: dangerous way round, admit one this driver never saw).
ENVELOPE_TYPES: frozenset[str] = frozenset(
    (
        EVENT_THREAD_STARTED,
        EVENT_TURN_STARTED,
        EVENT_TURN_COMPLETED,
        EVENT_TURN_FAILED,
        EVENT_ITEM_STARTED,
        EVENT_ITEM_UPDATED,
        EVENT_ITEM_COMPLETED,
        EVENT_ERROR,
    )
)
ITEM_TYPES: frozenset[str] = frozenset(
    (
        ITEM_COMMAND_EXECUTION,
        ITEM_AGENT_MESSAGE,
        ITEM_REASONING,
        ITEM_ERROR,
        ITEM_FILE_CHANGE,
        ITEM_MCP_TOOL_CALL,
        ITEM_WEB_SEARCH,
        ITEM_TODO_LIST,
        ITEM_COLLAB_TOOL_CALL,
    )
)

# How this driver's runs are billed, DECLARED so the cost ledger asks the driver instead of guessing
# from the model id. ChatGPT auth consumes a subscription seat: any USD figure downstream is notional.
BILLING_MODE = "subscription_notional"

#: Default model when no mapping applies. The ChatGPT-auth account's own default;
#: an API-only slug (e.g. a ``-codex-max`` alias) fails the request outright, so
#: the fallback must be a slug this auth mode accepts.
DEFAULT_CODEX_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.6-sol")

#: Instruction files whose presence in the workspace changes what an agent is
#: told. Recorded, never authored, so arms stay comparable.
_INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md", "AGENT.md", "TASK.md")

_POLL_S = 0.25
_CANDIDATE_PERMISSION_PROFILE = "merlin-candidate"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_model(model: str) -> str:
    """Map a harness model alias onto a Codex slug.

    ``CODEX_MODEL_MAP`` (``alias=slug,alias=slug``) lets a campaign pin the
    mapping explicitly; otherwise an alias that already looks like a Codex slug
    passes through and anything else falls back to :data:`DEFAULT_CODEX_MODEL`.
    Both the requested and the resolved id are recorded by the caller — an alias
    can change what it points at, and a result must say which model actually ran.
    """
    raw = (model or "").strip()
    mapping = {}
    for pair in (os.environ.get("CODEX_MODEL_MAP") or "").split(","):
        alias, sep, slug = pair.partition("=")
        if sep and alias.strip():
            mapping[alias.strip()] = slug.strip()
    if raw in mapping:
        return mapping[raw]
    # A model reached through the LiteLLM bridge keeps the proxy's model_name -- it is NOT a Codex slug
    # and must not fall through to DEFAULT_CODEX_MODEL, which would silently run OpenAI's default model
    # while the manifest claimed the run measured nemotron.
    from merlin_experiments.phase1.providers import agent_bridge as _BR

    bridged = _BR.bridged_name(raw, "codex")
    if bridged:
        return bridged
    if raw.startswith("gpt-") or raw.startswith("codex-") or raw.startswith("o3"):
        return raw
    return DEFAULT_CODEX_MODEL


def _effort_arg(effort: str) -> list[str]:
    """Reasoning effort as a config override, empty when unset."""
    value = (effort or "").strip()
    if not value:
        return []
    return ["-c", f"model_reasoning_effort={json.dumps(value)}"]


def build_cmd(
    ws: Path,
    *,
    model: str,
    effort: str,
    final_path: Path,
    sandbox: str,
    codex_bin: str = "codex",
) -> list[str]:
    """Assemble the ``codex exec`` argv.

    ``--skip-git-repo-check`` because the prepared workspace is a copy, not a
    checkout. The prompt arrives on stdin (``-``) so its exact bytes are an
    artifact rather than an argv fragment mangled by quoting.
    """
    cmd = [
        codex_bin,
        "exec",
        "--json",
        "--color",
        "never",
        "--skip-git-repo-check",
        "--model",
        model,
        "-C",
        str(ws),
        "-o",
        str(final_path),
    ]
    cmd += _effort_arg(effort)
    if sandbox == "bwrap":
        # Do not pass --sandbox: the legacy setting overrides permission profiles.
        # --strict-config fails closed on a CLI too old to understand the profile.
        cmd += [
            "--strict-config",
            "-c",
            f'default_permissions="{_CANDIDATE_PERMISSION_PROFILE}"',
            "-c",
            "approval_policy=never",
        ]
    else:
        # NOT ``--ask-for-approval``: that flag does not exist in 0.147.0 (the
        # CLI offers --approve-for-me / --dangerously-bypass-approvals-and-sandbox),
        # so passing it aborts the launch. The policy is a config override.
        cmd += ["--sandbox", "workspace-write", "-c", "approval_policy=never"]
    cmd.append("-")
    return cmd


def usage_to_claude_shape(usage: dict) -> tuple[dict, bool]:
    """Translate a Codex ``turn.completed`` usage payload to the Claude shape.

    Returns ``(usage_dict, reported)``. Codex's ``input_tokens`` is a TOTAL that
    already contains the cache reads and writes; the Claude shape's
    ``input_tokens`` is the uncached remainder. Subtracting is therefore the
    correct translation, clamped at zero so a provider inconsistency cannot
    produce a negative count. When nothing was reported, ``reported`` is False
    and the caller must omit usage rather than emit zeros.
    """

    def _int(key: str) -> int | None:
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value)

    total_in = _int("input_tokens")
    cache_read = _int("cached_input_tokens")
    cache_write = _int("cache_write_input_tokens")
    out = _int("output_tokens")
    reasoning = _int("reasoning_output_tokens")
    if all(v is None for v in (total_in, cache_read, cache_write, out, reasoning)):
        return {}, False

    uncached = None if total_in is None else max(total_in - (cache_read or 0) - (cache_write or 0), 0)
    shaped = {
        "input_tokens": uncached or 0,
        "output_tokens": out or 0,
        "cache_read_input_tokens": cache_read or 0,
        "cache_creation_input_tokens": cache_write or 0,
    }
    # Kept alongside, not folded in: reasoning is a subset of output_tokens and
    # adding it again would double-count the most expensive bucket.
    if reasoning is not None:
        shaped["reasoning_output_tokens"] = reasoning
    if total_in is not None:
        shaped["codex_input_tokens_total"] = total_in
    return shaped, True


#: The frozen per-experiment Codex config. Deliberately minimal: the user's own
#: config.toml carries per-project trust levels and notice state that have nothing
#: to do with the experiment and would differ between machines.
_FROZEN_CONFIG = (
    "model = {model}\nmodel_reasoning_effort = {effort}\n"
    f'default_permissions = "{_CANDIDATE_PERMISSION_PROFILE}"\n'
    "{provider}\n{profile}"
)


def _candidate_permission_config(codex_home: Path) -> str:
    """Freeze the candidate's filesystem policy into the isolated Codex config.

    The outer bwrap masks /scratch and /scratch2 before remounting only declared
    inputs and tools. Granting those mount destinations read-only here therefore
    cannot reveal an unmounted host path; the writable workspace gets its own
    narrower rule. The even narrower CODEX_HOME denial is essential: its auth
    file is writable to the Codex *parent* for refresh, never to candidate tools.
    """
    if not codex_home.is_absolute():
        raise ValueError("isolated CODEX_HOME must be absolute")
    launcher_dir = Path.home() / ".local" / "bin"
    package_roots = dict.fromkeys((real_codex_home() / "packages", Path.home() / ".codex" / "packages"))
    runtime_grants = "".join(
        f'{json.dumps(str(path))} = "read"\n' for path in (launcher_dir, *package_roots) if path.exists()
    )
    return (
        f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}]\n"
        'extends = ":workspace"\n'
        f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}.filesystem]\n"
        '":root" = "deny"\n'
        '":minimal" = "read"\n'
        '"/scratch" = "read"\n'
        '"/scratch2" = "read"\n'
        f"{runtime_grants}"
        f'{json.dumps(str(codex_home))} = "deny"\n'
        f'[permissions.{_CANDIDATE_PERMISSION_PROFILE}.filesystem.":workspace_roots"]\n'
        '"." = "write"\n'
        f"[permissions.{_CANDIDATE_PERMISSION_PROFILE}.network]\n"
        "enabled = false\n"
    )


#: ``codex --version`` per binary, asked once. The CLI is a hot dependency and its event names are
#: what this module parses, so a run that does not record which CLI produced its stream cannot be
#: attributed to a contract later -- the same reason a hardware verdict records its RTL revision.
_CLI_VERSION: dict = {}


def cli_version(codex_bin: str = "codex") -> str | None:
    """The CLI's self-reported version, or ``None`` when it cannot be asked.

    ``None`` is a real answer and is recorded as such: guessing a version would put a fabricated
    provenance stamp on every run made with that binary, which is worse than an absent one. The
    string is stored VERBATIM (e.g. ``codex-cli 0.153.0``) and never parsed here -- a comparison is
    somebody else's job, and a parser is one more thing to drift.
    """
    if codex_bin in _CLI_VERSION:
        return _CLI_VERSION[codex_bin]
    out = None
    try:
        proc = subprocess.run([codex_bin, "--version"], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            text = (proc.stdout or proc.stderr or "").strip().splitlines()
            out = text[0].strip() if text else None
    except (OSError, subprocess.SubprocessError):
        out = None
    _CLI_VERSION[codex_bin] = out or None
    return _CLI_VERSION[codex_bin]


def real_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def prepare_codex_home(dest: Path, *, model: str, effort: str) -> dict:
    """Build an ISOLATED ``CODEX_HOME`` at *dest* and describe it.

    Why not just bind the real ``~/.codex``: it contains ``sessions/`` — every
    prior Codex conversation on this host — plus history and state databases.
    Exposing those to a graded agent is an answer-leak surface of exactly the
    kind this bench has already been bitten by, and none of it is needed to run.

    What the isolated home holds is a frozen ``config.toml`` and nothing else;
    Codex creates its own ``sessions/``, ``state_*.sqlite`` and caches inside it.
    **The credential is never copied here** — :func:`codex_runtime_binds`
    bind-mounts the real ``auth.json`` over this path in the outer sandbox. A
    separately enforced Codex permission profile denies the entire isolated
    home to candidate commands, while the Codex parent can refresh its token.

    Measured caveat: a fresh home has no warm prompt cache, so the cached-token
    share differs from a run using the user's own home. Every arm must therefore
    build its home the same way, or the cache-hit rate varies between arms for a
    reason that has nothing to do with the treatment.
    """
    dest.mkdir(parents=True, exist_ok=True)
    # A non-OpenAI model reaches codex-cli only through the LiteLLM bridge: codex 0.147 speaks the
    # Responses API and nothing else, so the provider block points it at our proxy and declares the
    # measured context window (without it codex budgets against fallback metadata). Empty for a native
    # model, which keeps the existing gpt-5.6-sol arms byte-identical to their previous runs.
    from merlin_experiments.phase1.providers import agent_bridge as _BR

    provider = _BR.codex_config_fragment(model)
    config = _FROZEN_CONFIG.format(
        model=json.dumps(_BR.codex_model_name(model)),
        effort=json.dumps(effort or "high"),
        profile=_candidate_permission_config(dest),
        provider=provider,
    )
    config_path = dest / "config.toml"
    config_path.write_text(config)
    auth = real_codex_home() / "auth.json"
    import hashlib

    return {
        "codex_home": str(dest),
        "config_sha256": hashlib.sha256(config.encode()).hexdigest(),
        "auth_source": str(auth),
        "auth_present": auth.is_file(),
        "auth_copied": False,  # bind-mounted; never written to disk here
        "isolated_from_real_home": True,
        "bridge": _BR.record(model, harness="codex"),
    }


def access_token_remaining_s() -> float | None:
    """Seconds until the mounted credential's access token expires, or None if unreadable.

    Reads ONLY the ``exp`` claim; no token material is returned, logged or recorded anywhere.

    This exists because the sandbox mounts ``auth.json`` READ-ONLY on purpose -- a graded agent must not
    be able to rewrite the operator's shared credential -- and the documented consequence was that "a
    refresh attempt fails loudly". In practice it failed as five stderr lines: a run whose token expired
    mid-flight logged `Failed to refresh token: Read-only file system`, then 401ed every call, and kept
    its process alive for five more hours producing nothing while the score sat unchanged. Loud enough to
    read afterwards, far too quiet to act on.

    Knowing the remaining lifetime up front turns that into a decision at t=0."""
    import base64
    import json as _json
    import time

    auth = real_codex_home() / "auth.json"
    if not auth.is_file():
        return None
    try:
        tok = (_json.loads(auth.read_text()).get("tokens") or {}).get("access_token") or ""
        payload = tok.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = _json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        return float(exp) - time.time() if exp else None
    except Exception:  # noqa: BLE001 — unreadable is not fatal
        return None


def check_token_outlasts_run(planned_s: float) -> dict:
    """Whether the credential will still be valid when a run of ``planned_s`` finishes.

    Returned, not raised: the caller decides whether a short-lived token is a refusal or a warning. The
    sandbox cannot refresh (by design), so a token that expires mid-run takes the run with it."""
    rem = access_token_remaining_s()
    if rem is None:
        return {
            "known": False,
            "ok": None,
            "detail": "credential lifetime unreadable; a mid-run expiry cannot be ruled out",
        }
    ok = rem > planned_s
    return {
        "known": True,
        "ok": ok,
        "remaining_s": int(rem),
        "planned_s": int(planned_s),
        "detail": (
            f"token has {rem / 3600:.1f} h left against a planned {planned_s / 3600:.1f} h run"
            + (
                ""
                if ok
                else " -- it will expire MID-RUN, and the sandbox mounts the "
                "credential read-only so codex cannot refresh it in place. "
                "Refresh on the host first (codex login status), then relaunch."
            )
        ),
    }


def codex_runtime_binds(codex_home: Path) -> list[str]:
    """bwrap args that make ``codex`` runnable and authenticated inside the sandbox.

    Runtime mounts, each for a reason:

    * ``~/.local/bin`` and ``~/.codex/packages`` RO — the former normally
      contains a ``codex`` symlink into the latter. Neither needs Claude's
      credential or configuration mounts.
    * *codex_home* writable — Codex must create sessions/state/caches somewhere.
    * the real ``auth.json`` WRITABLE **onto** ``<codex_home>/auth.json``.

      This bind was read-only, on the reasoning that a refresh attempt should
      fail loudly rather than silently rewrite a shared credential. Measured,
      that reasoning inverts: OAuth refresh tokens are SINGLE-USE and rotate, so
      Codex spends the old token server-side, cannot write the new pair back,
      and every subsequent run dies ``401 refresh_token_reused``. The failure is
      not loud where it matters either -- rounds complete in seconds and the
      score is a small constant, which reads as a bad agent, not a dead
      credential. ``codex login`` only postpones it to the next rotation, so the
      read-only bind was the defect rather than the safeguard.

      Writable permits token rotation. The native Codex permission profile is
      mandatory before untrusted execution: without it, generated shell commands
      share this mount namespace and can read or rewrite the credential.

    Note what is NOT bound: ``~/.codex`` itself. Inside the sandbox that
    directory therefore contains only ``packages/``, so no prior session,
    history file or state database is reachable. The canary asserts this.
    """
    home = real_codex_home()
    binds: list[str] = []
    # The Codex path no longer inherits Claude's broad runtime mounts. Keep its
    # launcher reachable explicitly; the usual ~/.local/bin/codex is a symlink
    # into the separately mounted packages directory.
    launcher_dir = Path.home() / ".local" / "bin"
    if launcher_dir.exists():
        binds += ["--ro-bind", str(launcher_dir), str(launcher_dir)]
        # base_argv(clearenv=True) intentionally drops the operator PATH. Keep
        # only this launcher directory plus system bins; toolchain env prepends
        # the experiment's frozen Python/LLVM/clang paths later.
        binds += ["--setenv", "PATH", f"{launcher_dir}:/usr/bin:/bin"]
    package_roots = (home / "packages", Path.home() / ".codex" / "packages")
    for packages in dict.fromkeys(package_roots):
        if packages.exists():
            binds += ["--ro-bind", str(packages), str(packages)]
    binds += ["--bind", str(codex_home), str(codex_home)]
    auth = home / "auth.json"
    if auth.is_file():
        # Writable, so a single-use refresh token can rotate in place; see the docstring.
        binds += ["--bind", str(auth), str(codex_home / "auth.json")]
    binds += ["--setenv", "CODEX_HOME", str(codex_home)]
    return binds


def _instruction_files(ws: Path) -> dict[str, int]:
    """Which instruction files the workspace carries, and their sizes."""
    found = {}
    for name in _INSTRUCTION_FILES:
        path = ws / name
        if path.is_file():
            found[name] = path.stat().st_size
    return found


_TOOL_OUTPUT_CAP = 20000


def _clip_tool_output(text: str, cap: int = _TOOL_OUTPUT_CAP) -> str:
    """Bound a recorded tool result WITHOUT discarding its tail.

    Head-only truncation destroys evidence silently. A discovery command that prints a large
    structure and THEN its summary -- ``print(facts); print('DERIVED_LEVERS', levers)`` -- loses
    exactly the closing lines the conformance checks read, so the run is graded as though the agent
    never ran the tool. Measured on the atlas arm-4 run of 2026-09-04: 29 of 467 results were cut at
    the cap, and the one carrying the arm-4 discovery evidence was among them. Keeping both ends
    preserves the structure and the summary inside the same budget.
    """
    if len(text) <= cap:
        return text
    head = cap * 2 // 3
    tail = cap - head
    return f"{text[:head]}\n...[{len(text) - head - tail} chars elided by the harness]...\n{text[-tail:]}"


def _tool_block(item: dict, tool_use_id: str) -> dict:
    """Render a Codex tool item as a Claude ``tool_use`` block.

    ``tool_use_id`` is REQUIRED and must be the same string the paired ``tool_result`` carries: it is
    the only key joining a call to its result, so every consumer of the transcript keys on it. Emitting
    the block WITHOUT an id silently costs the run its entire tool telemetry -- aet's stream parser
    records a call only ``if tc.tool_use_id``, so the atlas round of 2026-09-04 reported
    ``tool_call_count=0`` and ``unique_tools_used=[]`` for a round that made 125 tool calls, and the
    per-call latency (measured as the gap between a tool_use and its tool_result) had nothing to pair.
    """
    itype = item.get("type")
    if itype == ITEM_COMMAND_EXECUTION:
        return {"type": "tool_use", "id": tool_use_id, "name": "Bash", "input": {"command": item.get("command", "")}}
    if itype == ITEM_FILE_CHANGE:
        changes = item.get("changes")
        return {
            "type": "tool_use",
            "id": tool_use_id,
            "name": "Edit",
            "input": {"changes": changes if isinstance(changes, list) else []},
        }
    return {
        "type": "tool_use",
        "id": tool_use_id,
        "name": str(itype or "tool"),
        "input": {k: v for k, v in item.items() if k not in ("id", "type")},
    }


class _Transcript:
    """Writes the harness's Claude-stream JSONL, one flushed line at a time."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w")
        self.path = path

    def emit(self, obj: dict) -> None:
        self._f.write(json.dumps(obj) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except OSError:
            pass


def _terminate_process_group(pgid: int, proc: subprocess.Popen | None = None) -> None:
    """Terminate a saved child process group even after its leader has exited."""
    if pgid <= 1:
        raise ValueError("refusing to signal an unsafe process-group id")
    for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except (PermissionError, OSError):
            if proc is not None:
                try:
                    proc.kill()
                except OSError:
                    return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc is not None:
                proc.poll()  # reap an exited group leader so an empty group disappears
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)


def _kill_tree(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the child's saved process group."""
    _terminate_process_group(proc.pid, proc)


def _link_for_aet(run_dir: Path, raw_path: Path, rnd: int) -> Path | None:
    """Expose this round's RAW codex stream under ``<run_dir>/agent/`` for ``aet import --format codex``.

    aet's codex importer takes a directory and globs ``**/*.jsonl`` recursively, replaying files in NAME
    order as consecutive rounds. Pointing it at ``rounds/`` would therefore also swallow
    ``round_NN.transcript.jsonl`` (the translated Claude-shape stream) and the ``timestamped`` wrapper
    shape, double-counting the round and polluting the trajectory. So the raw streams — and only those —
    are linked into their own directory under a zero-padded, sortable name.

    A hard link keeps one copy of the bytes; a symlink is the cross-device fallback. Failure here is
    never fatal: the raw file is already written, and losing the convenience link must not kill a round.
    """
    try:
        agent_dir = run_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        dest = agent_dir / f"events.{rnd:02d}.raw.jsonl"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            os.link(raw_path, dest)
        except OSError:
            dest.symlink_to(os.path.relpath(raw_path, agent_dir))
        return dest
    except Exception:
        return None


def last_message_path(ws: Path, final_path: Path, sandbox: str) -> Path:
    """Where codex may actually WRITE its last message (``-o``).

    ``final_path`` lives under the run directory, which is on /scratch -- and the sandbox tmpfs-hides
    /scratch* by design, so codex could not create it and reported ``Failed to write last message file
    ... (os error 2)`` on every sandboxed round. The read afterwards is guarded, so the loss was
    SILENT: the round's ``result`` was simply empty. Inside the sandbox the workspace is the one
    writable tree, so the file goes there and is copied out (and removed) once the process exits.
    """
    if sandbox != "bwrap":
        return final_path
    return ws / ".codex_out" / final_path.name


#: A continuation needs enough budget left to be worth a turn; below this the session stops cleanly
#: rather than spawning a process that will be killed mid-thought.
_CONTINUE_MIN_S = 120
#: A backstop so a pathological loop cannot spin against the API for the whole budget.
_CONTINUE_MAX_TURNS = 200
# Transient capacity refusals may resume the same continuous session, never
# switch its model or start another grader. Keep retries within its wall budget.
_CAPACITY_MAX_RETRIES = 3
_CAPACITY_BACKOFF_S = 20


def build_resume_cmd(
    ws: Path, *, model: str, effort: str, final_path: Path, sandbox: str, thread_id: str, codex_bin: str = "codex"
) -> list[str]:
    """Assemble ``codex exec resume <SESSION_ID> -`` for continuing an existing thread.

    Built from scratch rather than by rewriting the first turn's argv, because ``resume`` accepts a
    SMALLER option set than ``exec``. The 0.158.0 CLI accepts ``--json``, ``--model``, ``-o``,
    ``--skip-git-repo-check`` and ``-c``, but NOT ``--color``, ``--sandbox`` or ``-C``.
    Patching the exec argv therefore produced

        Usage: codex exec resume --json --dangerously-bypass-approvals-and-sandbox <SESSION_ID> [PROMPT]

    and the continuation died on its first attempt with 14772s of budget left.

    Dropping ``-C`` is safe: the child is spawned with ``cwd=ws`` already, so the working directory is
    the workspace either way. The session id precedes the trailing ``-`` because the grammar is
    ``[OPTIONS] [SESSION_ID] [PROMPT]`` and ``-`` is the prompt (read from stdin, so its exact bytes
    stay an artifact rather than an argv fragment)."""
    cmd = [codex_bin, "exec", "resume", "--json", "--skip-git-repo-check", "--model", model, "-o", str(final_path)]
    cmd += _effort_arg(effort)
    if sandbox == "bwrap":
        cmd += [
            "--strict-config",
            "-c",
            f'default_permissions="{_CANDIDATE_PERMISSION_PROFILE}"',
            "-c",
            "approval_policy=never",
        ]
    else:
        # `resume` has no --sandbox option; its config override is the same
        # workspace-write policy the first unsandboxed turn selects.
        cmd += ["-c", 'sandbox_mode="workspace-write"', "-c", "approval_policy=never"]
    cmd += [str(thread_id), "-"]
    return cmd


def _sandbox_script(rounds: Path, rnd: int, turn: int, command: str) -> list[str]:
    """Run a bwrap command from a FILE rather than as a ``bash -c`` argument.

    execve refuses any single argument over MAX_ARG_STRLEN (128 KiB) with E2BIG, and the error names
    only the interpreter -- `Argument list too long: 'bash'` -- so a size refusal is indistinguishable
    from a missing shell. Measured 2026-09-06 on the perf campaign: the composed sandbox command was
    178,303 bytes and every trial died before its first round. The bind list has a size escape already
    (``bwrap --args`` from a file descriptor), but the shell string that carries it does not, and the
    mask set grows with the corpus, so the ceiling is reachable by adding capsules.

    A file has no such limit. It lives beside the round's other artifacts and NOT in the workspace:
    the agent must not be able to read or edit the command that sandboxes it.
    """
    script = rounds / f"round_{rnd:02d}.sandbox.turn{turn:02d}.sh"
    # A completed launch seals this file 0500.  A checkpoint resume may retry the same round/turn and
    # therefore the same path; opening the sealed inode for writing fails before Codex starts.  Replace
    # it atomically with a newly sealed inode instead of temporarily making the old launch record
    # writable.  The behaviour is identical for every driver invocation and independent of target.
    rounds.mkdir(parents=True, exist_ok=True)
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=rounds, prefix=f".{script.name}.", suffix=".tmp", delete=False
        ) as handle:
            handle.write(command)
            staged = Path(handle.name)
        staged.chmod(0o500)
        os.replace(staged, script)
        staged = None
    finally:
        if staged is not None:
            staged.unlink(missing_ok=True)
    return ["bash", str(script)]


def _verify_frozen_config(codex_home: Path, expected_sha256: str) -> None:
    """Refuse an altered policy before the first turn and every continuation."""
    import hashlib

    actual = hashlib.sha256((codex_home / "config.toml").read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError("isolated Codex config changed after it was frozen")


def _preflight_candidate_sandbox(
    ws: Path,
    codex_home: Path,
    codex_bin: str,
    bundle: dict,
    sandbox_command: Callable[..., str],
    rounds: Path,
    rnd: int,
) -> None:
    """No-model proof that the installed CLI enforces the frozen profile.

    The probe runs through the *same* outer mount policy as the paid turn. A
    missing or unsupported permission profile, visible credential, inaccessible
    workspace, or failure to execute common tools is a launch failure, not a
    degraded mode that silently falls back to bypassing Codex's sandbox.
    """
    probe = (
        'test -n "$CODEX_HOME" && '
        'test ! -r "$CODEX_HOME/auth.json" && test ! -w "$CODEX_HOME/auth.json" && '
        'for proc_auth in /proc/[0-9]*/root"$CODEX_HOME/auth.json"; do '
        'test ! -r "$proc_auth" || exit 1; done && '
        f"test -r {shlex.quote(str(ws))} && test -w {shlex.quote(str(ws))} && "
        "command -v python3 >/dev/null && python3 --version >/dev/null"
    )
    auth = shlex.quote(str(codex_home / "auth.json"))
    native_probe = shlex.join(
        (
            codex_bin,
            "sandbox",
            "--permission-profile",
            _CANDIDATE_PERMISSION_PROFILE,
            "-C",
            str(ws),
            "--",
            "/bin/sh",
            "-c",
            probe,
        )
    )
    inner = f"test -r {auth} && test -w {auth} && {native_probe}"
    script = _sandbox_script(
        rounds,
        rnd,
        -1,
        sandbox_command(inner, ws, bundle, extra_binds=codex_runtime_binds(codex_home)),
    )
    result = subprocess.run(script, cwd=str(ws), capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(
            "Codex candidate permission profile preflight failed; refusing untrusted launch "
            f"(rc={result.returncode}, stderr={result.stderr[-1200:].strip()!r})"
        )


def run_round(
    ws: Path,
    run_dir: Path,
    model: str,
    bundle: dict,
    te,
    sandbox: str,
    rnd: int,
    timeout: int,
    *,
    subagent_model: str = "",
    background_model: str = "",
    effort: str = "",
    prompt: str | None = None,
    continue_session: bool = False,
    effective_model: str | None = None,
    sandbox_command: Callable[..., str] | None = None,
    codex_binary: str | Path | None = None,
    codex_home_root: Path | None = None,
    **_ignored,
) -> tuple[int, Path]:
    """Drive ONE capsule-bench round via ``codex exec``. Returns ``(rc, transcript_path)``.

    ``codex_binary`` selects the executable without changing ``CODEX_BIN``.
    ``codex_home_root`` selects the parent of the isolated per-round home for
    bwrap execution. Omitted values retain the environment and cache defaults;
    both selections remain fixed across this round's session continuations.

    ``continue_session`` makes ONE session span the whole ``timeout`` budget. It exists because
    ``codex exec`` is one TURN, not one session: it returns as soon as the model stops, which the
    harness reads as "the agent session ended". Measured on atlas arm-4, three runs in a row ended
    that way -- the last after 83 minutes with 23890s of its 28901s budget unspent, having emitted
    exactly one ``turn.started``/``turn.completed`` pair. Continuation re-invokes
    ``codex exec resume <thread_id>``, so the SAME thread (and its accumulated understanding) keeps
    working; ``keep_launching``'s "one invocation per process" contract is preserved, because this is
    still one session -- just delivered over several processes. Off by default: in ROUNDS mode a
    round ending is how the harness re-grades and re-prompts, and continuing would break that cadence.

    Tier-within-agent (``subagent_model`` / ``background_model``) has no Codex
    equivalent — ``codex exec`` exposes no subagent mechanism — so a request for
    one is recorded in the init record and otherwise ignored. Silently accepting
    it would make an arm look tiered when it was not.
    """
    # Explicit round inputs avoid changing another concurrent round's executable
    # or credential-isolated home placement. Defaults retain historical callers.
    if codex_binary is not None and (not isinstance(codex_binary, (str, Path)) or not str(codex_binary).strip()):
        raise ValueError("explicit codex_binary must be a nonempty executable path or name")
    codex_bin = str(codex_binary) if codex_binary is not None else os.environ.get("CODEX_BIN", "codex")
    # A rigorous caller may preflight model resolution and pass the content-addressed result.
    # In that mode do NOT consult CODEX_MODEL_MAP a second time between the declaration and
    # the process launch: a remap landing in between would run a model the run did not
    # declare, and the transcript would name the declared one.
    resolved = str(effective_model).strip() if effective_model is not None else resolve_model(model)
    if not resolved:
        raise ValueError("effective Codex model must be non-empty")
    if sandbox == "bwrap":
        from merlin_experiments.phase1.providers import agent_bridge as _BR

        if _BR.bridged_name(resolved, "codex"):
            # The proxy's master key belongs to the host Codex process, not the
            # candidate shell. A future broker can deliver it without a shell
            # environment leak; clearenv intentionally does not do that today.
            raise RuntimeError("bridged Codex needs a host-side proxy credential broker under bwrap")
    rounds = run_dir / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    tpath = rounds / f"round_{rnd:02d}.transcript.jsonl"
    raw_path = rounds / f"round_{rnd:02d}.codex_events.raw.jsonl"
    stamped_path = rounds / f"round_{rnd:02d}.codex_events.timestamped.jsonl"
    stderr_path = rounds / f"round_{rnd:02d}.codex_stderr.log"
    prompt_path = rounds / f"round_{rnd:02d}.prompt.txt"
    final_path = rounds / f"round_{rnd:02d}.final.txt"

    tr = _Transcript(tpath)
    tr.emit(
        {
            "type": "system",
            "subtype": "init",
            "driver": "codex",
            "round": rnd,
            "model": resolved,
            "model_requested": model,
            "codex_bin": codex_bin,
            "cli_version": cli_version(codex_bin),
            "sandbox": sandbox,
            # Instruction parity between arms is an artifact-level fact, not a hope.
            "workspace_instruction_files": _instruction_files(ws),
            "tiering_requested_but_unsupported": bool(subagent_model or background_model),
            "started_at": _now(),
        }
    )

    # The graded instruction. ``prompt=`` overrides it only for out-of-band uses
    # (the sandbox canary states its own task); a measured arm always gets this
    # text, so two arms cannot silently differ in what they were asked to do.
    msg = (
        prompt
        if prompt is not None
        else (
            "Read TASK.md and qa/verdict.json (if present) in your workspace, then build or repair the target "
            "backend under submission/ per those instructions. During iteration, check the smallest affected capsule "
            "or coherent comma-separated capsule cluster with `python3 agent_selfcheck.py --submission submission "
            "--sim spike --capsules <names>`; do not repeatedly grade the whole corpus. Run `--capsules all` only "
            "after the focused checks improve and the candidate is ready for a regression sweep. Do not edit "
            "submission/ while a self-check is running, because that makes its result stale. Goldens are withheld; "
            "iterate until the complete corpus passes. THE FIRST GRADE NEEDS YOUR SUBMISSION FIRST: the harness "
            "grades submission/, so while submission/manifest.yaml does not exist there is NOTHING to grade and no "
            "verdict can ever arrive -- an absent qa/verdict.json is not a queue you wait in, it means you have not "
            "submitted yet. MEASURED: two runs each burned their whole first round blocked on await_verdict.py "
            "reporting 'the waiter is healthy but has received nothing' while the harness logged 'no "
            "submission/manifest.yaml to grade yet' every 30s -- a mutual wait that consumes the round timeout. "
            "Build something minimal and write submission/manifest.yaml FIRST; only once a verdict exists does "
            "waiting for the next one make sense (then use await_verdict.py rather than a poll loop, and do not "
            "launch `--capsules all` merely to refresh it). Use exact capsule directory names for focused checks. "
            "Begin now."
        )
    )
    prompt_path.write_text(msg)

    inner_final = last_message_path(ws, final_path, sandbox)
    inner_final.parent.mkdir(parents=True, exist_ok=True)
    run_cmd = build_cmd(ws, model=resolved, effort=effort, final_path=inner_final, sandbox=sandbox, codex_bin=codex_bin)
    home_info: dict = {}
    if sandbox == "bwrap":
        # The caller owns sandbox policy; a provider must not import a native controller.
        if sandbox_command is None:
            raise ValueError("bwrap provider execution requires the caller's sandbox_command")
        from merlin.common.artifacts import cache_dir

        # An isolated CODEX_HOME per run: the real ~/.codex holds every prior
        # session on this host, which a graded agent must not be able to read.
        # PURGEABLE cache, and no credential is written into it — the real
        # auth.json is bind-mounted writable for refresh, behind Codex's native
        # candidate-command sandbox (see codex_runtime_binds).
        home_root = Path(codex_home_root) if codex_home_root is not None else cache_dir("codex_home")
        codex_home = home_root / f"{run_dir.name}_r{rnd:02d}"
        if codex_home.resolve().is_relative_to(ws.resolve()) or ws.resolve().is_relative_to(codex_home.resolve()):
            raise ValueError("isolated CODEX_HOME must not overlap the candidate workspace")
        home_info = prepare_codex_home(codex_home, model=resolved, effort=effort)
        _verify_frozen_config(codex_home, home_info["config_sha256"])
        _preflight_candidate_sandbox(ws, codex_home, codex_bin, bundle, sandbox_command, rounds, rnd)
        inner = " ".join(shlex.quote(c) for c in run_cmd)
        cmd = _sandbox_script(
            rounds, rnd, 0, sandbox_command(inner, ws, bundle, extra_binds=codex_runtime_binds(codex_home))
        )
    else:
        cmd = run_cmd

    started = time.monotonic()
    deadline = started + max(int(timeout), 1)
    turns_started = turns_reported = 0
    pending_tools: dict[str, dict] = {}
    unknown: list[str] = []
    errors: list[str] = []
    recovered_errors: set[int] = set()
    capacity_retries = 0
    thread_id = None
    timed_out = False
    seq = 0

    raw_f = open(raw_path, "wb")
    stamped_f = open(stamped_path, "w")
    #: Sent when resuming. Deliberately the SAME standing instruction, never a hint: an arm that got
    #: extra guidance mid-session would not be comparable to one that did not.
    _CONTINUE_MSG = (
        "Continue. Re-read qa/verdict.json for the latest grade, then keep repairing the "
        "backend under submission/. With agent_selfcheck, re-check only the smallest affected "
        "capsule or coherent cluster after each edit; use `--capsules all` only after focused "
        "checks improve, and never edit submission/ while a self-check is running."
    )
    turn_index = 0
    rc = 0
    active_proc: subprocess.Popen | None = None
    active_pgid: int | None = None
    try:
        while True:
            turn_errors_start = len(errors)
            turn_failed = turn_completed = False
            cur_prompt = (
                prompt_path
                if turn_index == 0
                else prompt_path.with_name(f"{prompt_path.stem}.cont{turn_index:02d}{prompt_path.suffix}")
            )
            if turn_index:
                # A previous attempt's final is not this resumed turn's answer.
                if inner_final.is_file():
                    previous_final = rounds / f"round_{rnd:02d}.turn{turn_index - 1:02d}.final.txt"
                    previous_final.write_text(inner_final.read_text())
                    inner_final.unlink()
                cur_prompt.write_text(_CONTINUE_MSG)
                if sandbox == "bwrap":
                    _verify_frozen_config(codex_home, home_info["config_sha256"])
                resume_argv = build_resume_cmd(
                    ws,
                    model=resolved,
                    effort=effort,
                    final_path=inner_final,
                    sandbox=sandbox,
                    thread_id=thread_id,
                    codex_bin=codex_bin,
                )
                if sandbox == "bwrap":
                    # Re-wrap exactly as the first turn was, so the continuation runs inside the SAME
                    # sandbox with the same binds. Rewriting the wrapped string instead would have to
                    # edit shell syntax; rebuilding it cannot drift from the first turn's wrapping.
                    _inner = " ".join(shlex.quote(c) for c in resume_argv)
                    cmd = _sandbox_script(
                        rounds,
                        rnd,
                        turn_index,
                        sandbox_command(_inner, ws, bundle, extra_binds=codex_runtime_binds(codex_home)),
                    )
                else:
                    cmd = resume_argv
            with open(stderr_path, "ab") as err_f, open(cur_prompt, "rb") as in_f:
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=in_f,
                        stdout=subprocess.PIPE,
                        stderr=err_f,
                        cwd=str(ws),
                        env=dict(os.environ),
                        start_new_session=True,
                    )
                    active_proc, active_pgid = proc, proc.pid
                    # Resource telemetry is sampled OUTSIDE the graded sandbox and follows the complete
                    # bwrap/Codex/tool descendant tree. One file per continuation turn prevents ambiguous
                    # counter resets when Codex is resumed in a new process.
                    _resource_path = rounds / f"round_{rnd:02d}.turn{turn_index:02d}.resource_samples.jsonl"
                    try:
                        subprocess.Popen(
                            [
                                sys.executable,
                                str(module_source_path("merlin_experiments.phase1.providers.resource_sampler")),
                                "--pid",
                                str(proc.pid),
                                "--output",
                                str(_resource_path),
                            ],
                            cwd=str(ws),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                    except OSError as _e:
                        tr.emit({"type": "resource_sampler_unavailable", "reason": str(_e), "arrived_at": _now()})
                except (OSError, ValueError) as exc:
                    # E2BIG NAMES `bash` AND NOTHING ELSE, so a spawn refused for size looks identical to
                    # a missing interpreter. execve bounds argv AND envp together, and the environment
                    # here is inherited from whatever launched the stage -- so record both sides, and the
                    # largest single contributor of each, or the next reader guesses like this one did.
                    _env = dict(os.environ)
                    _argv_bytes = sum(len(part.encode("utf-8", "replace")) + 1 for part in cmd)
                    _env_bytes = sum(
                        len(k.encode("utf-8", "replace")) + len(v.encode("utf-8", "replace")) + 2
                        for k, v in _env.items()
                    )
                    _biggest_env = sorted(
                        ((len(v.encode("utf-8", "replace")), k) for k, v in _env.items()), reverse=True
                    )[:5]
                    tr.emit(
                        {
                            "type": "result",
                            "subtype": "error",
                            "is_error": True,
                            "result": f"codex spawn failed: {type(exc).__name__}: {exc}",
                            "spawn_sizes": {
                                "argv_bytes": _argv_bytes,
                                "argv_parts": [len(part) for part in cmd],
                                "env_bytes": _env_bytes,
                                "env_count": len(_env),
                                "largest_env": [{"name": k, "bytes": n} for n, k in _biggest_env],
                                "total_bytes": _argv_bytes + _env_bytes,
                            },
                        }
                    )
                    tr.close()
                    return 127, tpath

                buf = bytearray()
                with selectors.DefaultSelector() as sel:
                    sel.register(proc.stdout.fileno(), selectors.EVENT_READ)
                    while True:
                        if time.monotonic() >= deadline:
                            timed_out = True
                            break
                        wait = min(_POLL_S, max(deadline - time.monotonic(), 0.0))
                        if not sel.select(timeout=wait):
                            if proc.poll() is not None and not sel.select(timeout=0):
                                break
                            continue
                        try:
                            chunk = os.read(proc.stdout.fileno(), 65536)
                        except OSError:
                            break
                        if not chunk:
                            break
                        buf += chunk
                        while True:
                            nl = buf.find(b"\n")
                            if nl < 0:
                                break
                            line = bytes(buf[: nl + 1])
                            del buf[: nl + 1]
                            seq += 1
                            arrived = _now()
                            # Durable FIRST, interpreted second.
                            raw_f.write(line)
                            raw_f.flush()
                            text = line.decode("utf-8", errors="replace").rstrip("\n")
                            try:
                                event = json.loads(text)
                                if not isinstance(event, dict):
                                    event = None
                            except json.JSONDecodeError:
                                event = None
                            stamped_f.write(
                                json.dumps(
                                    {
                                        "seq": seq,
                                        "arrived_at": arrived,
                                        **({"event": event} if event is not None else {"unparsed": text}),
                                    }
                                )
                                + "\n"
                            )
                            stamped_f.flush()
                            if event is None:
                                tr.emit(
                                    {"type": "codex_unparsed", "seq": seq, "arrived_at": arrived, "line": text[:500]}
                                )
                                continue

                            etype = event.get("type")
                            if etype == EVENT_THREAD_STARTED:
                                announced = event.get("thread_id")
                                if (
                                    not isinstance(announced, str)
                                    or not announced
                                    or (thread_id is not None and announced != thread_id)
                                ):
                                    errors.append("codex session thread identity changed or is missing")
                                    _kill_tree(proc)
                                else:
                                    thread_id = announced
                                tr.emit({"type": "codex_thread", "thread_id": thread_id, "arrived_at": arrived})
                            elif etype == EVENT_TURN_STARTED:
                                turns_started += 1
                            elif etype == EVENT_TURN_COMPLETED:
                                turn_completed = True
                                shaped, reported = usage_to_claude_shape(event.get("usage") or {})
                                if reported:
                                    turns_reported += 1
                                message: dict[str, Any] = {
                                    "id": f"codex_{rnd}_{turns_started}",
                                    "model": resolved,
                                    "content": [],
                                }
                                if reported:
                                    message["usage"] = shaped
                                record = {"type": "assistant", "message": message, "arrived_at": arrived}
                                if not reported:
                                    record["codex_usage_unreported"] = True
                                tr.emit(record)
                            elif etype == EVENT_TURN_FAILED:
                                turn_failed = True
                                # No usage is carried here: unmeasured, not free.
                                errors.append(_error_text(event.get("error")))
                                tr.emit(
                                    {
                                        "type": "assistant",
                                        "message": {
                                            "id": f"codex_{rnd}_{turns_started}",
                                            "model": resolved,
                                            "content": [],
                                        },
                                        "codex_usage_unreported": True,
                                        "codex_turn_failed": True,
                                        "arrived_at": arrived,
                                    }
                                )
                            elif etype == EVENT_ERROR:
                                errors.append(_error_text(event.get("message") or event))
                            elif etype in (EVENT_ITEM_STARTED, EVENT_ITEM_COMPLETED):
                                item = event.get("item")
                                if not isinstance(item, dict):
                                    continue
                                itype = item.get("type")
                                item_id = str(item.get("id") or f"anon_{seq}")
                                if itype == ITEM_AGENT_MESSAGE and etype == EVENT_ITEM_COMPLETED:
                                    tr.emit(
                                        {
                                            "type": "assistant",
                                            "message": {
                                                "id": f"codex_msg_{item_id}",
                                                "model": resolved,
                                                "content": [{"type": "text", "text": item.get("text") or ""}],
                                            },
                                            "arrived_at": arrived,
                                        }
                                    )
                                elif itype == ITEM_REASONING and etype == EVENT_ITEM_COMPLETED:
                                    tr.emit(
                                        {
                                            "type": "assistant",
                                            "message": {
                                                "id": f"codex_think_{item_id}",
                                                "model": resolved,
                                                "content": [{"type": "thinking", "thinking": item.get("text") or ""}],
                                            },
                                            "arrived_at": arrived,
                                        }
                                    )
                                elif itype == ITEM_ERROR:
                                    errors.append(_error_text(item.get("message") or item))
                                elif itype in _TOOL_ITEMS:
                                    if etype == EVENT_ITEM_STARTED:
                                        pending_tools[item_id] = item
                                        tr.emit(
                                            {
                                                "type": "assistant",
                                                "message": {
                                                    "id": f"codex_tool_{item_id}",
                                                    "model": resolved,
                                                    "content": [_tool_block(item, f"codex_tool_{item_id}")],
                                                },
                                                "arrived_at": arrived,
                                            }
                                        )
                                    else:
                                        pending_tools.pop(item_id, None)
                                        tr.emit(
                                            {
                                                "type": "user",
                                                "message": {
                                                    "content": [
                                                        {
                                                            "type": "tool_result",
                                                            "tool_use_id": f"codex_tool_{item_id}",
                                                            "content": _clip_tool_output(
                                                                item.get("aggregated_output") or ""
                                                            ),
                                                            "is_error": bool(item.get("exit_code")),
                                                        }
                                                    ]
                                                },
                                                "arrived_at": arrived,
                                            }
                                        )
                                else:
                                    unknown.append(str(itype))
                            else:
                                unknown.append(str(etype))

                if buf:
                    raw_f.write(bytes(buf))
                    raw_f.flush()
                if timed_out:
                    _kill_tree(proc)
                rc = proc.wait()
                # A successful leader exit is not proof that its compiler/tool descendants exited.
                _terminate_process_group(proc.pid, proc)
                active_proc, active_pgid = None, None
                try:
                    proc.stdout.close()
                except OSError:
                    pass
            # CONTINUE only while every reason to stop is absent, and say which one stopped it. A silent
            # stop here would look identical to the defect this exists to fix.
            turn_index += 1
            remaining = deadline - time.monotonic()
            turn_errors = errors[turn_errors_start:]
            retry_capacity = (
                continue_session
                and thread_id
                and turn_failed
                and not turn_completed
                and not timed_out
                and rc != 0
                and turn_errors
                and all(error.lower().startswith("selected model is at capacity") for error in turn_errors)
                and capacity_retries < _CAPACITY_MAX_RETRIES
                and turn_index < _CONTINUE_MAX_TURNS
                and remaining > _CAPACITY_BACKOFF_S + _CONTINUE_MIN_S
            )
            if retry_capacity:
                capacity_retries += 1
                tr.emit(
                    {
                        "type": "codex_capacity_retry",
                        "attempt": capacity_retries,
                        "thread_id": thread_id,
                        "model": resolved,
                        "backoff_s": _CAPACITY_BACKOFF_S,
                        "arrived_at": _now(),
                    }
                )
                time.sleep(_CAPACITY_BACKOFF_S)
                recovered_errors.update(range(turn_errors_start, len(errors)))
                continue
            stop = (
                ("" if continue_session else "continuation not enabled")
                or ("no thread id to resume" if not thread_id else "")
                or ("wall budget spent" if remaining <= _CONTINUE_MIN_S else "")
                or ("the turn did not complete" if timed_out or rc != 0 else "")
                or (f"turn cap {_CONTINUE_MAX_TURNS}" if turn_index >= _CONTINUE_MAX_TURNS else "")
            )
            tr.emit(
                {
                    "type": "codex_session_turn",
                    "turn": turn_index,
                    "continuing": not stop,
                    "stopped_because": stop,
                    "remaining_s": round(max(remaining, 0.0), 1),
                    "thread_id": thread_id,
                    "arrived_at": _now(),
                }
            )
            if stop:
                break
    finally:
        if active_proc is not None and active_pgid is not None:
            _terminate_process_group(active_pgid, active_proc)
        for handle in (raw_f, stamped_f):
            try:
                os.fsync(handle.fileno())
            except (OSError, ValueError):
                pass
            try:
                handle.close()
            except OSError:
                pass

    # Copy the last message out of the workspace, then drop it: the agent's own final message is not
    # part of its next round's inputs, and leaving it in the workspace would make it one.
    if inner_final != final_path and inner_final.is_file():
        try:
            final_path.write_text(inner_final.read_text())
            inner_final.unlink()
        except OSError as _e:
            print(f"[codex] could not recover the final message: {_e}", file=sys.stderr)
    final_text = final_path.read_text() if final_path.is_file() else ""
    usage_complete = turns_started > 0 and turns_reported >= turns_started
    unrecovered_errors = [error for index, error in enumerate(errors) if index not in recovered_errors]
    summary = {
        "thread_id": thread_id,
        "turns_started": turns_started,
        "turns_usage_reported": turns_reported,
        "usage_complete": usage_complete,
        "unknown_types": sorted(set(unknown)),
        "errors": errors[:10],
        "capacity_retries": capacity_retries,
        "unrecovered_errors": unrecovered_errors[:10],
        "timed_out": timed_out,
        "exit_code": rc,
        # Which CLI parsed this stream. `unknown_types` above DETECTS contract drift; this says what
        # to compare a drifted run against. None means the binary could not be asked, never a guess.
        "cli_version": cli_version(codex_bin),
        "wall_s": round(time.monotonic() - started, 3),
        # ChatGPT-auth runs consume a subscription, not metered dollars. Any USD
        # figure downstream is notional and must not enter a money budget.
        "billing_mode": BILLING_MODE,
        "codex_home": home_info,
        "artifacts": {
            "raw": str(raw_path),
            "timestamped": str(stamped_path),
            "stderr": str(stderr_path),
            "prompt": str(prompt_path),
            "final": str(final_path),
        },
    }
    (rounds / f"round_{rnd:02d}.codex_summary.json").write_text(json.dumps(summary, indent=2))
    tr.emit({"type": "codex_summary", **summary})
    _link_for_aet(run_dir, raw_path, rnd)

    if timed_out:
        tr.emit(
            {"type": "result", "subtype": "error", "is_error": True, "result": f"codex exec timed out after {timeout}s"}
        )
        tr.close()
        return 124, tpath
    if rc != 0 or unrecovered_errors:
        tr.emit(
            {
                "type": "result",
                "subtype": "error",
                "is_error": True,
                "result": (unrecovered_errors[0] if unrecovered_errors else f"codex exited {rc}")[:500],
            }
        )
        tr.close()
        return (rc or 1), tpath
    tr.emit({"type": "result", "subtype": "success", "is_error": False, "result": final_text[:2000]})
    tr.close()
    return 0, tpath


def _error_text(value: Any) -> str:
    """Render an error payload without losing the message nested inside it."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        message = value.get("message")
        if isinstance(message, str):
            return message
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)
