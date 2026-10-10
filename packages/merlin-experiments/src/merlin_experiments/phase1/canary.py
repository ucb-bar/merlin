"""Finite fresh-client observation through the ordinary sealed preparation.

The same explicit runtime selection serves ordinary authoring. Raw provider tool
completion, public/tool positives and protected synthetic negatives are required;
source tests or a canary result never certify a compiler, runtime or phase.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
from pathlib import Path

from . import preflight, tooling_readiness
from .providers import codex_agent, execution

_MAX_BYTES = 2 * 1024 * 1024
_PROBE = "client_canary_probe.py"
_REPORT = "merlin_client_canary.result"
_PUBLIC = "MERLIN_PUBLIC_READ_OK\n"
_DONE = "MERLIN_CLIENT_PROBE_COMPLETED\n"


def _read(path: Path) -> bytes:
    try:
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("canary selected data must be regular")
        with path.open("rb") as stream:
            data = stream.read(_MAX_BYTES + 1)
    except OSError:
        raise ValueError("canary selected data is unavailable") from None
    if len(data) > _MAX_BYTES:
        raise ValueError("canary selected data exceeds its bound")
    return data


def _write(path: Path, data: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(data)


def _children(path: Path, remaining: int) -> list[Path]:
    children = []
    try:
        for child in path.iterdir():
            if len(children) >= remaining:
                raise ValueError("canary file selection exceeds its bound")
            children.append(child)
    except OSError:
        raise ValueError("canary selected directory is unavailable") from None
    return sorted(children, reverse=True)


def _submission(ws: Path) -> tuple:
    root = ws / "submission"
    if not root.exists():
        return ()
    entries, total, rows = 0, 0, []
    pending = [root]
    while pending:
        path = pending.pop()
        entries += 1
        if entries > 4096 or path.is_symlink():
            raise ValueError("canary initial submission is unsupported")
        if path.is_dir():
            rows.append((str(path.relative_to(root)), "directory"))
            pending.extend(_children(path, 4096 - entries - len(pending)))
        else:
            data = _read(path)
            total += len(data)
            if total > 4 * _MAX_BYTES:
                raise ValueError("canary initial submission exceeds its bound")
            rows.append((str(path.relative_to(root)), hashlib.sha256(data).hexdigest()))
    return tuple(rows)


def _public_control(prepared) -> tuple[str, str]:
    """Join the admitted public capsule through its existing original-root copy."""
    from merlin.targetgen.sandbox import bwrap as BW

    from . import corpus_inputs as CI

    public = prepared.public_root
    if public is None:
        raise ValueError("canary requires the admitted public interface view")
    commitment = json.loads(_read(public.parent / "source_commitments.json"))
    if (
        not isinstance(commitment, dict)
        or type(commitment.get("version")) is not int
        or commitment["version"] != 1
        or commitment.get("mode") != "descriptor_cohort"
    ):
        raise ValueError("canary requires the reviewed original corpus commitments")
    roots = []
    for row in commitment.get("sources", ()):
        if row.get("role") != "corpus":
            continue
        staged = Path(row["staged"])
        original = Path(row["original"])
        if staged.is_absolute() or ".." in staged.parts or not original.is_absolute():
            raise ValueError("canary corpus commitment is malformed")
        if staged.parts and staged.parts[0] == "policy":
            roots.append((public.parent / staged, original))
    public_caps = CI.discover_capsules(public, labels={"public", "dev"}, contract=prepared.contract_root)
    policy_caps = CI.discover_capsules(
        [staged for staged, _ in roots], labels={"public", "dev"}, contract=prepared.contract_root
    )
    if not public_caps or len({cap["name"] for cap in public_caps}) != len(public_caps):
        raise ValueError("canary public capsule identity is missing or duplicated")
    selected = public_caps[0]
    matched = [cap for cap in policy_caps if cap["name"] == selected["name"]]
    if len(matched) != 1:
        raise ValueError("canary public capsule lacks a unique original policy member")
    policy = matched[0]
    relative = Path(selected.get("interface_mlir", ""))
    if (
        not selected.get("interface_mlir")
        or relative.is_absolute()
        or ".." in relative.parts
        or policy.get("interface_mlir") != selected["interface_mlir"]
    ):
        raise ValueError("canary public interface declaration changed")
    directory = Path(policy["__dir__"])
    owners = [(staged, original) for staged, original in roots if directory.is_relative_to(staged)]
    if len(owners) != 1:
        raise ValueError("canary public policy member has no unique original root")
    staged, original = owners[0]
    member = original / directory.relative_to(staged) / relative
    _, grants = BW._snapshot_grants(prepared.workspace, prepared.bundle, prepared.request.context.repo)
    if not any(member == destination or member.is_relative_to(destination) for _, destination, _ in grants):
        raise ValueError("reviewed public interface has no exact frozen candidate grant")
    [frozen] = BW.snapshot_input_paths(
        prepared.workspace, prepared.bundle, [member], repo=prepared.request.context.repo
    )
    data = _read(Path(selected["__dir__"]) / relative)
    if not data or _read(directory / relative) != data or _read(frozen) != data:
        raise ValueError("canary public/source/frozen interface bytes changed")
    return str(member), hashlib.sha256(data).hexdigest()


def _command_matches(actual: object, command: str) -> bool:
    if not isinstance(actual, str):
        return False
    if actual == command:
        return True
    try:
        parts = shlex.split(actual)
        return (
            len(parts) == 3
            and parts[0] in {"bash", "/bin/bash", "sh", "/bin/sh"}
            and parts[1] in {"-c", "-lc"}
            and parts[2] == command
        )
    except ValueError:
        return False


def completed_probe(rows: list, command: str) -> str:
    """Require one paired, ordered, explicitly successful raw command item."""
    started = None
    completed = None
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed raw canary event")
        item = row.get("item")
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"reasoning", "agent_message"}:
            continue
        if item.get("type") != "command_execution" or not _command_matches(item.get("command"), command):
            raise ValueError("canary turn contains another tool or changed command")
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("canary command item lacks identity")
        if row.get("type") == "item.started":
            if started is not None or completed is not None:
                raise ValueError("duplicate or reordered canary command")
            started = (item_id, item["command"])
        elif row.get("type") == "item.completed":
            if (
                started != (item_id, item["command"])
                or completed is not None
                or item.get("status") != "completed"
                or type(item.get("exit_code")) is not int
                or item["exit_code"] != 0
            ):
                raise ValueError("canary command lacks paired successful completion")
            completed = item_id
    if completed is None:
        raise ValueError("canary command completion is missing")
    return completed


def _probe(
    public: tuple[str, str], protected: tuple[Path, ...], tool_probe: str, python_paths: tuple[str, ...]
) -> bytes:
    # Only synthetic protected paths are exposed in this issued probe; no answer
    # paths, credential content or private metadata enters the client prompt.
    lines = [
        "import hashlib, os, sys",
        "from pathlib import Path",
        f"public = Path({public[0]!r})",
        "data = public.read_bytes()",
        f"assert hashlib.sha256(data).hexdigest() == {public[1]!r}",
        f"protected = {tuple(str(path) for path in protected)!r}",
        "for name in protected:",
        "    try:",
        "        Path(name).read_bytes()",
        "    except (PermissionError, FileNotFoundError):",
        "        pass",
        "    else:",
        "        raise AssertionError('protected synthetic input was readable')",
        "home = Path(os.environ['CODEX_HOME'])",
        "assert not os.access(home / 'auth.json', os.R_OK)",
        "assert not os.access(home / 'auth.json', os.W_OK)",
        "assert not os.access(home / 'config.toml', os.R_OK)",
        f"sys.path[:0] = {python_paths!r}",
        f"exec(compile({tool_probe!r}, '<selected-public-tool-probe>', 'exec'))",
        f"Path({_REPORT!r}).write_text({_PUBLIC + _DONE!r})",
    ]
    return ("\n".join(lines) + "\n").encode()


def execute(prepared) -> int:
    from .session import validate_preflight_options

    options = prepared.request.options
    if not options.codex_canary or options.preflight_only:
        raise ValueError("canary continuation requires its explicit mode")
    validate_preflight_options(options)
    if not 0 < options.round_timeout <= 600:
        raise ValueError("canary round budget must be 1..600 seconds")
    identity = preflight.verify_prepared_inputs(prepared)
    runtime = prepared.selected_codex_runtime
    if runtime is None:
        raise ValueError("canary requires the ordinary explicit client runtime")
    tools = tuple(prepared.request.resolved_tools())
    ws, run_dir = prepared.workspace, prepared.run_dir
    task, submission = _read(ws / "TASK.md"), _submission(ws)
    public = _public_control(prepared)
    control = run_dir / "client_canary_controls"
    control.mkdir(mode=0o700)
    protected = (control / "answer", control / "history")
    for path in protected:
        _write(path, b"synthetic protected isolation control\n")
    probe_path, report_path = run_dir / _PROBE, ws / _REPORT
    if report_path.exists() or report_path.is_symlink():
        raise ValueError("canary refuses an existing probe result")
    record = {
        "schema": "phase1_client_canary.v1",
        "mode": "codex_canary",
        "formal_complete": False,
        "bundle_manifest_sha256": identity,
        "runtime_selection": runtime.record(),
        "tools": list(tools),
        "scope": "selected ordinary client configuration only; no compiler/runtime/phase verdict",
        "observed_ok": False,
        "provider_started": False,
    }
    try:
        with tooling_readiness.public_probe_session(prepared.request.context, ws, prepared.bundle, tools) as tool_probe:
            probe = _probe(public, protected, tool_probe, runtime.python_paths)
            _write(probe_path, probe)
            # The literal body is immutable in the provider-owned raw command.
            # A workspace script would permit modify/execute/restore.
            command = "python3 -I -B -c " + shlex.quote(probe.decode())
            preflight.verify_prepared_inputs(prepared)
            record["provider_started"] = True
            rc, _ = codex_agent.run_round(
                ws,
                run_dir,
                options.model,
                prepared.bundle,
                None,
                "bwrap",
                0,
                options.round_timeout,
                effort=options.effort,
                continue_session=False,
                prompt="Run exactly one command in Bash. Do not edit files or invoke other tools. "
                + "Do not implement a compiler. "
                + "Stop after its result. The complete command is on the final line:\n"
                + command,
                **execution.codex_call_kwargs(
                    runtime, context=prepared.request.context, run_dir=run_dir, model=options.model
                ),
            )
            rows = [
                json.loads(line) for line in _read(run_dir / "rounds" / "round_00.codex_events.raw.jsonl").splitlines()
            ]
            summary = json.loads(_read(run_dir / "rounds" / "round_00.codex_summary.json"))
            item = completed_probe(rows, command)
            if (
                type(rc) is not int
                or rc != 0
                or summary.get("timed_out") is not False
                or type(summary.get("exit_code")) is not int
                or summary["exit_code"] != 0
                or summary.get("usage_complete") is not True
                or type(summary.get("turns_usage_reported")) is not int
                or summary["turns_usage_reported"] < 1
            ):
                raise ValueError("canary provider observation is incomplete")
            if (
                _read(probe_path) != probe
                or _read(report_path) != (_PUBLIC + _DONE).encode()
                or _read(ws / "TASK.md") != task
                or _submission(ws) != submission
            ):
                raise ValueError("canary owned probe, task, output or submission changed")
            if tuple(prepared.request.resolved_tools()) != tools:
                raise ValueError("canary selected tools changed")
            preflight.verify_prepared_inputs(prepared)
            record.update(completed_command_item=item, probe_sha256=hashlib.sha256(probe).hexdigest())
        preflight.verify_prepared_inputs(prepared)
        record["observed_ok"] = True
    except Exception as exc:
        # Full provider transcripts stay private. Feedback contains no selected
        # private paths, credential material or raw tool errors.
        record["failure_type"] = type(exc).__name__
    _write(run_dir / "client_canary_result.json", json.dumps(record, sort_keys=True, indent=2).encode())
    return 0 if record["observed_ok"] else 1
