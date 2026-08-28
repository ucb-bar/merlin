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
harness's existing bwrap wrapper (the boundary that masks goldens and hidden
capsules), and Codex's own approval prompts are bypassed *because* that outer
boundary is what the isolation claim rests on. At ``--sandbox none`` there is no
outer boundary, so Codex's own ``workspace-write`` sandbox is used instead of
bypassing it.
"""

from __future__ import annotations

import json
import os
import selectors
import shlex
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- The measured event contract (codex-cli 0.147.0) ------------------------
# Verified by capturing a live run, not read from documentation. Envelope types
# and item types are two separate vocabularies and are matched exactly; anything
# unrecognized is preserved in the transcript rather than dropped.
EVENT_THREAD_STARTED = "thread.started"
EVENT_TURN_STARTED = "turn.started"
EVENT_TURN_COMPLETED = "turn.completed"
EVENT_TURN_FAILED = "turn.failed"
EVENT_ITEM_STARTED = "item.started"
EVENT_ITEM_COMPLETED = "item.completed"
EVENT_ERROR = "error"

ITEM_COMMAND_EXECUTION = "command_execution"
ITEM_AGENT_MESSAGE = "agent_message"
ITEM_REASONING = "reasoning"
ITEM_ERROR = "error"
ITEM_FILE_CHANGE = "file_change"
ITEM_MCP_TOOL_CALL = "mcp_tool_call"
ITEM_WEB_SEARCH = "web_search"
_TOOL_ITEMS = (ITEM_COMMAND_EXECUTION, ITEM_FILE_CHANGE, ITEM_MCP_TOOL_CALL, ITEM_WEB_SEARCH)

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    import agent_bridge as _BR
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
    cmd = [codex_bin, "exec", "--json", "--color", "never", "--skip-git-repo-check",
           "--model", model, "-C", str(ws), "-o", str(final_path)]
    cmd += _effort_arg(effort)
    if sandbox == "bwrap":
        # The outer bwrap IS the boundary; Codex's own sandbox would only add a
        # second, weaker one that cannot see the masked answer surfaces.
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
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
_FROZEN_CONFIG = 'model = {model}\nmodel_reasoning_effort = {effort}\n{provider}'


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
    read-only *bind-mounts* the real ``auth.json`` over this path inside the
    sandbox, so no secret is written to the artifact tree.

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
    import agent_bridge as _BR
    provider = _BR.codex_config_fragment(model)
    config = _FROZEN_CONFIG.format(model=json.dumps(_BR.codex_model_name(model)),
                                  effort=json.dumps(effort or "high"),
                                  provider=provider)
    config_path = dest / "config.toml"
    config_path.write_text(config)
    auth = real_codex_home() / "auth.json"
    import hashlib

    return {
        "codex_home": str(dest),
        "config_sha256": hashlib.sha256(config.encode()).hexdigest(),
        "auth_source": str(auth),
        "auth_present": auth.is_file(),
        "auth_copied": False,  # bind-mounted read-only; never written to disk here
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
    except Exception:                                    # noqa: BLE001 — unreadable is not fatal
        return None


def check_token_outlasts_run(planned_s: float) -> dict:
    """Whether the credential will still be valid when a run of ``planned_s`` finishes.

    Returned, not raised: the caller decides whether a short-lived token is a refusal or a warning. The
    sandbox cannot refresh (by design), so a token that expires mid-run takes the run with it."""
    rem = access_token_remaining_s()
    if rem is None:
        return {"known": False, "ok": None,
                "detail": "credential lifetime unreadable; a mid-run expiry cannot be ruled out"}
    ok = rem > planned_s
    return {"known": True, "ok": ok, "remaining_s": int(rem), "planned_s": int(planned_s),
            "detail": (f"token has {rem/3600:.1f} h left against a planned {planned_s/3600:.1f} h run"
                       + ("" if ok else " -- it will expire MID-RUN, and the sandbox mounts the "
                                       "credential read-only so codex cannot refresh it in place. "
                                       "Refresh on the host first (codex login status), then relaunch."))}


def codex_runtime_binds(codex_home: Path) -> list[str]:
    """bwrap args that make ``codex`` runnable and authenticated inside the sandbox.

    Three binds, each for a reason:

    * ``~/.codex/packages`` RO — ``~/.local/bin/codex`` is a SYMLINK into it, so
      binding ``~/.local/bin`` alone (which the shared claude binds already do)
      leaves the launcher pointing at nothing. Contains no conversation state.
    * *codex_home* writable — Codex must create sessions/state/caches somewhere.
    * the real ``auth.json`` RO **onto** ``<codex_home>/auth.json`` — auth works,
      the token cannot be modified, and nothing secret is written to disk. A
      refresh attempt fails loudly rather than silently rewriting a shared
      credential.

    Note what is NOT bound: ``~/.codex`` itself. Inside the sandbox that
    directory therefore contains only ``packages/``, so no prior session,
    history file or state database is reachable. The canary asserts this.
    """
    home = real_codex_home()
    binds: list[str] = []
    packages = home / "packages"
    if packages.exists():
        binds += ["--ro-bind", str(packages), str(packages)]
    binds += ["--bind", str(codex_home), str(codex_home)]
    auth = home / "auth.json"
    if auth.is_file():
        binds += ["--ro-bind", str(auth), str(codex_home / "auth.json")]
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


def _tool_block(item: dict) -> dict:
    """Render a Codex tool item as a Claude ``tool_use`` block."""
    itype = item.get("type")
    if itype == ITEM_COMMAND_EXECUTION:
        return {"type": "tool_use", "name": "Bash",
                "input": {"command": item.get("command", "")}}
    if itype == ITEM_FILE_CHANGE:
        changes = item.get("changes")
        return {"type": "tool_use", "name": "Edit",
                "input": {"changes": changes if isinstance(changes, list) else []}}
    return {"type": "tool_use", "name": str(itype or "tool"),
            "input": {k: v for k, v in item.items() if k not in ("id", "type")}}


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


def _kill_tree(proc: subprocess.Popen) -> None:
    """SIGTERM then SIGKILL the child's process group."""
    for sig, grace in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                return
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


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


def run_round(ws: Path, run_dir: Path, model: str, bundle: dict, te, sandbox: str, rnd: int,
              timeout: int, *, subagent_model: str = "", background_model: str = "",
              effort: str = "", prompt: str | None = None, **_ignored) -> tuple[int, Path]:
    """Drive ONE capsule-bench round via ``codex exec``. Returns ``(rc, transcript_path)``.

    Tier-within-agent (``subagent_model`` / ``background_model``) has no Codex
    equivalent — ``codex exec`` exposes no subagent mechanism — so a request for
    one is recorded in the init record and otherwise ignored. Silently accepting
    it would make an arm look tiered when it was not.
    """
    codex_bin = os.environ.get("CODEX_BIN", "codex")
    resolved = resolve_model(model)
    rounds = run_dir / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    tpath = rounds / f"round_{rnd:02d}.transcript.jsonl"
    raw_path = rounds / f"round_{rnd:02d}.codex_events.raw.jsonl"
    stamped_path = rounds / f"round_{rnd:02d}.codex_events.timestamped.jsonl"
    stderr_path = rounds / f"round_{rnd:02d}.codex_stderr.log"
    prompt_path = rounds / f"round_{rnd:02d}.prompt.txt"
    final_path = rounds / f"round_{rnd:02d}.final.txt"

    tr = _Transcript(tpath)
    tr.emit({
        "type": "system", "subtype": "init", "driver": "codex", "round": rnd,
        "model": resolved, "model_requested": model,
        "codex_bin": codex_bin, "sandbox": sandbox,
        # Instruction parity between arms is an artifact-level fact, not a hope.
        "workspace_instruction_files": _instruction_files(ws),
        "tiering_requested_but_unsupported": bool(subagent_model or background_model),
        "started_at": _now(),
    })

    # The graded instruction. ``prompt=`` overrides it only for out-of-band uses
    # (the sandbox canary states its own task); a measured arm always gets this
    # text, so two arms cannot silently differ in what they were asked to do.
    msg = prompt if prompt is not None else (
        "Read TASK.md and qa/verdict.json (if present) in your workspace, then build or repair the target "
        "backend under submission/ per those instructions. Run `python3 agent_selfcheck.py --submission "
        "submission --sim spike --capsules all` with your shell after each build to grade against the "
        "real oracle (goldens withheld), and iterate until capsules pass. Begin now.")
    prompt_path.write_text(msg)

    run_cmd = build_cmd(ws, model=resolved, effort=effort, final_path=final_path,
                        sandbox=sandbox, codex_bin=codex_bin)
    home_info: dict = {}
    if sandbox == "bwrap":
        # Imported here rather than at module scope: it is needed only for the
        # bwrap wrapper, and keeping it lazy lets the driver be exercised (and
        # its command contract tested) without pulling in the whole QA loop.
        import run_baseline_qa_loop as _R  # the same integrity wrapper the claude path uses
        from merlin.common.artifacts import cache_dir

        # An isolated CODEX_HOME per run: the real ~/.codex holds every prior
        # session on this host, which a graded agent must not be able to read.
        # PURGEABLE cache, and no credential is written into it — the real
        # auth.json is bind-mounted read-only (see codex_runtime_binds).
        codex_home = cache_dir("codex_home") / f"{run_dir.name}_r{rnd:02d}"
        home_info = prepare_codex_home(codex_home, model=resolved, effort=effort)
        inner = " ".join(shlex.quote(c) for c in run_cmd)
        cmd = ["bash", "-c", _R.bwrap_cmd(inner, ws, bundle,
                                         extra_binds=codex_runtime_binds(codex_home))]
    else:
        cmd = run_cmd

    started = time.monotonic()
    deadline = started + max(int(timeout), 1)
    turns_started = turns_reported = 0
    pending_tools: dict[str, dict] = {}
    unknown: list[str] = []
    errors: list[str] = []
    thread_id = None
    timed_out = False
    seq = 0

    raw_f = open(raw_path, "wb")
    stamped_f = open(stamped_path, "w")
    try:
        with open(stderr_path, "wb") as err_f, open(prompt_path, "rb") as in_f:
            try:
                proc = subprocess.Popen(cmd, stdin=in_f, stdout=subprocess.PIPE, stderr=err_f,
                                        cwd=str(ws), env=dict(os.environ), start_new_session=True)
            except (OSError, ValueError) as exc:
                tr.emit({"type": "result", "subtype": "error", "is_error": True,
                         "result": f"codex spawn failed: {type(exc).__name__}: {exc}"})
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
                        line = bytes(buf[:nl + 1])
                        del buf[:nl + 1]
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
                        stamped_f.write(json.dumps(
                            {"seq": seq, "arrived_at": arrived,
                             **({"event": event} if event is not None else {"unparsed": text})}) + "\n")
                        stamped_f.flush()
                        if event is None:
                            tr.emit({"type": "codex_unparsed", "seq": seq, "arrived_at": arrived,
                                     "line": text[:500]})
                            continue

                        etype = event.get("type")
                        if etype == EVENT_THREAD_STARTED:
                            thread_id = event.get("thread_id")
                            tr.emit({"type": "codex_thread", "thread_id": thread_id,
                                     "arrived_at": arrived})
                        elif etype == EVENT_TURN_STARTED:
                            turns_started += 1
                        elif etype == EVENT_TURN_COMPLETED:
                            shaped, reported = usage_to_claude_shape(event.get("usage") or {})
                            if reported:
                                turns_reported += 1
                            message: dict[str, Any] = {
                                "id": f"codex_{rnd}_{turns_started}", "model": resolved,
                                "content": [],
                            }
                            if reported:
                                message["usage"] = shaped
                            record = {"type": "assistant", "message": message,
                                      "arrived_at": arrived}
                            if not reported:
                                record["codex_usage_unreported"] = True
                            tr.emit(record)
                        elif etype == EVENT_TURN_FAILED:
                            # No usage is carried here: unmeasured, not free.
                            errors.append(_error_text(event.get("error")))
                            tr.emit({"type": "assistant",
                                     "message": {"id": f"codex_{rnd}_{turns_started}",
                                                 "model": resolved, "content": []},
                                     "codex_usage_unreported": True,
                                     "codex_turn_failed": True, "arrived_at": arrived})
                        elif etype == EVENT_ERROR:
                            errors.append(_error_text(event.get("message") or event))
                        elif etype in (EVENT_ITEM_STARTED, EVENT_ITEM_COMPLETED):
                            item = event.get("item")
                            if not isinstance(item, dict):
                                continue
                            itype = item.get("type")
                            item_id = str(item.get("id") or f"anon_{seq}")
                            if itype == ITEM_AGENT_MESSAGE and etype == EVENT_ITEM_COMPLETED:
                                tr.emit({"type": "assistant", "message": {
                                    "id": f"codex_msg_{item_id}", "model": resolved,
                                    "content": [{"type": "text", "text": item.get("text") or ""}]},
                                    "arrived_at": arrived})
                            elif itype == ITEM_REASONING and etype == EVENT_ITEM_COMPLETED:
                                tr.emit({"type": "assistant", "message": {
                                    "id": f"codex_think_{item_id}", "model": resolved,
                                    "content": [{"type": "thinking",
                                                 "thinking": item.get("text") or ""}]},
                                    "arrived_at": arrived})
                            elif itype == ITEM_ERROR:
                                errors.append(_error_text(item.get("message") or item))
                            elif itype in _TOOL_ITEMS:
                                if etype == EVENT_ITEM_STARTED:
                                    pending_tools[item_id] = item
                                    tr.emit({"type": "assistant", "message": {
                                        "id": f"codex_tool_{item_id}", "model": resolved,
                                        "content": [_tool_block(item)]}, "arrived_at": arrived})
                                else:
                                    pending_tools.pop(item_id, None)
                                    tr.emit({"type": "user", "message": {"content": [{
                                        "type": "tool_result",
                                        "tool_use_id": f"codex_tool_{item_id}",
                                        "content": (item.get("aggregated_output") or "")[:20000],
                                        "is_error": bool(item.get("exit_code")),
                                    }]}, "arrived_at": arrived})
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
            try:
                proc.stdout.close()
            except OSError:
                pass
    finally:
        for handle in (raw_f, stamped_f):
            try:
                os.fsync(handle.fileno())
            except (OSError, ValueError):
                pass
            try:
                handle.close()
            except OSError:
                pass

    final_text = final_path.read_text() if final_path.is_file() else ""
    usage_complete = turns_started > 0 and turns_reported >= turns_started
    summary = {
        "thread_id": thread_id, "turns_started": turns_started,
        "turns_usage_reported": turns_reported, "usage_complete": usage_complete,
        "unknown_types": sorted(set(unknown)), "errors": errors[:10],
        "timed_out": timed_out, "exit_code": rc,
        "wall_s": round(time.monotonic() - started, 3),
        # ChatGPT-auth runs consume a subscription, not metered dollars. Any USD
        # figure downstream is notional and must not enter a money budget.
        "billing_mode": BILLING_MODE,
        "codex_home": home_info,
        "artifacts": {"raw": str(raw_path), "timestamped": str(stamped_path),
                      "stderr": str(stderr_path), "prompt": str(prompt_path),
                      "final": str(final_path)},
    }
    (rounds / f"round_{rnd:02d}.codex_summary.json").write_text(json.dumps(summary, indent=2))
    tr.emit({"type": "codex_summary", **summary})
    _link_for_aet(run_dir, raw_path, rnd)

    if timed_out:
        tr.emit({"type": "result", "subtype": "error", "is_error": True,
                 "result": f"codex exec timed out after {timeout}s"})
        tr.close()
        return 124, tpath
    if rc != 0 or errors:
        tr.emit({"type": "result", "subtype": "error", "is_error": True,
                 "result": (errors[0] if errors else f"codex exited {rc}")[:500]})
        tr.close()
        return (rc or 1), tpath
    tr.emit({"type": "result", "subtype": "success", "is_error": False,
             "result": final_text[:2000]})
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
