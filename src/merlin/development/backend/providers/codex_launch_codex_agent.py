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
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from merlin.common.paths import module_source_path
from merlin.development.backend.providers import codex_capacity as _CAP  # noqa: E402
from merlin.development.backend.providers import codex_context as _CTX
from merlin.development.backend.providers.authoring_prompt import (
    authoring_prompt,
    profile_authoring_prompt,
)
from merlin.development.backend.providers.codex_context import delegation_user_prompt as delegation_user_prompt
from merlin.development.backend.providers.codex_runtime import (
    _kill_tree,
    _process_group_quiescent,
    _terminate_process_group,
)
from merlin.development.backend.providers.codex_transcript import (
    _INSTRUCTION_FILES as _INSTRUCTION_FILES,
)
from merlin.development.backend.providers.codex_transcript import (
    _TOOL_OUTPUT_CAP as _TOOL_OUTPUT_CAP,
)
from merlin.development.backend.providers.codex_transcript import (
    ITEM_COMMAND_EXECUTION as ITEM_COMMAND_EXECUTION,
)
from merlin.development.backend.providers.codex_transcript import (
    ITEM_FILE_CHANGE as ITEM_FILE_CHANGE,
)
from merlin.development.backend.providers.codex_transcript import (
    _clip_tool_output as _clip_tool_output,
)
from merlin.development.backend.providers.codex_transcript import (
    _error_text as _error_text,
)
from merlin.development.backend.providers.codex_transcript import (
    _instruction_files as _instruction_files,
)
from merlin.development.backend.providers.codex_transcript import (
    _tool_block as _tool_block,
)
from merlin.development.backend.providers.codex_transcript import (
    _Transcript as _Transcript,
)
from merlin.development.backend.providers.codex_transcript import (
    last_message_path as last_message_path,
)
from merlin.development.backend.providers.codex_usage import usage_to_claude_shape as usage_to_claude_shape
from merlin.development.backend.providers.execution import sandbox_script as _sandbox_script
from merlin.development.backend.telemetry import author_agents as _START

# ---- codex_agent -------------------------------------------------------------------------------

EVENT_THREAD_STARTED = "thread.started"

EVENT_TURN_STARTED = "turn.started"

EVENT_TURN_COMPLETED = "turn.completed"

EVENT_TURN_FAILED = "turn.failed"

EVENT_ITEM_STARTED = "item.started"

EVENT_ITEM_UPDATED = "item.updated"

EVENT_ITEM_COMPLETED = "item.completed"

EVENT_ERROR = "error"

ITEM_AGENT_MESSAGE = "agent_message"

ITEM_REASONING = "reasoning"

ITEM_ERROR = "error"

ITEM_MCP_TOOL_CALL = "mcp_tool_call"

ITEM_WEB_SEARCH = "web_search"

ITEM_TODO_LIST = "todo_list"

ITEM_COLLAB_TOOL_CALL = "collab_tool_call"

_TOOL_ITEMS = (ITEM_COMMAND_EXECUTION, ITEM_FILE_CHANGE, ITEM_MCP_TOOL_CALL, ITEM_WEB_SEARCH)

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

BILLING_MODE = "subscription_notional"

DEFAULT_CODEX_MODEL = os.environ.get("CODEX_MODEL", "gpt-5.6-sol")

_POLL_S = 0.25

_CONTROL_POLL_S = 1.0

OPERATOR_INTERRUPTED_EXIT_CODE = 130

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
    from merlin.development.backend.providers import agent_bridge as _BR

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


_FROZEN_CONFIG = (
    "model = {model}\nmodel_reasoning_effort = {effort}\n"
    f'default_permissions = "{_CANDIDATE_PERMISSION_PROFILE}"\n'
    "{provider}\n{profile}{trust}"
)


def _candidate_permission_config(codex_home: Path, *, read_paths: tuple[str, ...] | None = None) -> str:
    from merlin.development.backend.providers.codex_permissions import _candidate_permission_config as render

    return render(codex_home, read_paths=read_paths, real_home=real_codex_home)


_CLI_VERSION: dict = {}


def cli_version(codex_bin: str = "codex") -> str | None:
    """The verbatim CLI version, or ``None`` when it cannot be observed."""
    from merlin.development.backend.providers.codex_runtime import cached_cli_version

    return cached_cli_version(codex_bin, _CLI_VERSION)


def real_codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))


def prepare_codex_home(
    dest: Path,
    *,
    model: str,
    effort: str,
    workspace: Path,
    candidate_read_paths: tuple[str, ...] | None = None,
    auth_source: Path | None = None,
    max_concurrent_subagents: int | None = None,
    helper_effort: str | None = None,
    author_context: dict | None = None,
) -> dict:
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
    if max_concurrent_subagents is not None and (
        type(max_concurrent_subagents) is not int or max_concurrent_subagents < 1
    ):
        raise ValueError("Codex subagent concurrency must be an explicit positive integer")
    if not workspace.is_absolute() or not workspace.is_dir() or workspace.resolve(strict=True) != workspace:
        raise ValueError("Codex workspace must be an existing canonical absolute directory")
    if auth_source is not None:
        from merlin.development.backend.providers.codex_runtime import canonical_file

        auth_source = canonical_file(str(auth_source), "client credential")
    dest.mkdir(parents=True, exist_ok=True)
    # A non-OpenAI model reaches codex-cli only through the LiteLLM bridge: codex 0.147 speaks the
    # Responses API and nothing else, so the provider block points it at our proxy and declares the
    # measured context window (without it codex budgets against fallback metadata). Empty for a native
    # model, which keeps the existing gpt-5.6-sol arms byte-identical to their previous runs.
    from merlin.development.backend.providers import agent_bridge as _BR

    provider = _BR.codex_config_fragment(model)
    config = _FROZEN_CONFIG.format(
        model=json.dumps(_BR.codex_model_name(model)),
        effort=json.dumps(effort or "high"),
        profile=_candidate_permission_config(dest, read_paths=candidate_read_paths),
        provider=provider,
        trust=f'[projects.{json.dumps(str(workspace))}]\ntrust_level = "trusted"\n',
    )
    from merlin.development.backend.authoring import author_subagents as _AS

    # Root-level dotted keys precede permission/provider tables; the CLI's strict-config preflight must accept them.
    config = _AS.home_config(max_concurrent_subagents, helper_effort) + _CTX.home_config(author_context) + config
    config_path = dest / "config.toml"
    config_path.write_text(config)
    auth = real_codex_home() / "auth.json" if auth_source is None else auth_source
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


_CONTINUE_MIN_S = 120

_CONTINUE_MAX_TURNS = 200


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
    *,
    runtime_binds: Callable[[Path], list[str]] | None = None,
    prepared_plan=None,
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
    from merlin.development.backend.providers.execution import CodexSandboxCommand
    from merlin.development.backend.session.inputs import public_contract_preflight

    expected_public = public_contract_preflight.expected_reads(
        ws, bundle, repo=sandbox_command.context.repo if type(sandbox_command) is CodexSandboxCommand else None
    )
    if expected_public is not None:
        probe += " && " + public_contract_preflight.command(expected_public)
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
        prepared_plan.command(inner)
        if prepared_plan is not None
        else sandbox_command(
            inner, ws, bundle, extra_binds=(codex_runtime_binds if runtime_binds is None else runtime_binds)(codex_home)
        ),
    )
    result = subprocess.run(script, cwd=str(ws), capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(
            "Codex candidate permission profile preflight failed; refusing untrusted launch "
            f"(rc={result.returncode}, stderr={result.stderr[-1200:].strip()!r})"
        )
    if expected_public is not None:
        public_contract_preflight.record_success(expected_public, result.stdout, rounds, rnd)
