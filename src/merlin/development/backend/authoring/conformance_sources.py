"""Bounded transcript source custody and reachable imported discovery calls.

This is an observer, never a shell/Python interpreter. Opaque writes, overlapping calls,
failed results and ambiguous execution refuse attribution; current files are never replayed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path, PurePosixPath
from time import monotonic

# ---- conformance_sources -----------------------------------------------------------------------

MAX_SOURCE_BYTES = 256 * 1024

MAX_SCRIPTS = 64


def execution_context(workspace, target, selected_facts, aliases=()):
    """Bind existing admitted runtime inputs before the client starts; expose metadata only."""
    facts = None
    if selected_facts is not None:
        selected = Path(selected_facts)
        digest = hashlib.sha256(selected.read_bytes()).hexdigest()
        paths = []
        for path in (selected, *map(Path, aliases)):
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                paths.append(str(path.absolute()))
        facts = {"sha256": digest, "paths": sorted(set(paths))}
    return {
        "schema": "merlin.phase1.conformance_execution.v1",
        "workspace": str(Path(workspace).absolute()),
        "target": target,
        "selected_facts": facts,
    }


def validate_context(value):
    if value is None:
        return None  # Historical inline-only observations have no native binding.
    if not isinstance(value, dict) or value.get("schema") != "merlin.phase1.conformance_execution.v1":
        return {"invalid": True}
    workspace, target, facts = value.get("workspace"), value.get("target"), value.get("selected_facts")
    if (
        not isinstance(workspace, str)
        or not _path(workspace, None)
        or (target is not None and not isinstance(target, str))
    ):
        return {"invalid": True}
    if facts is not None and (
        not isinstance(facts, dict)
        or not isinstance(facts.get("sha256"), str)
        or len(facts["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in facts["sha256"])
        or not isinstance(facts.get("paths"), list)
        or not facts["paths"]
        or any(not isinstance(path, str) or not _path(path, None) for path in facts["paths"])
    ):
        return {"invalid": True}
    return value


@dataclass(frozen=True)
class Attribution:
    sources: tuple[str, ...] = ()
    script: str | None = None
    source_sha256: str | None = None
    reason: str | None = None
    cwd: str | None = None


def _path(value, cwd):
    if not isinstance(value, str) or not value or any(c in value for c in "$`*?[]~\x00"):
        return None
    path = PurePosixPath(value)
    if ".." in path.parts:
        return None
    if not path.is_absolute():
        if not cwd or not PurePosixPath(cwd).is_absolute():
            return None
        path = PurePosixPath(cwd) / path
    return str(path)


def _commands(command, tokenize):
    """Recognize simple sequential statements and quoted heredocs, retaining their source."""
    if "`" in command or "$(" in command:
        return None
    lines, commands, i = command.splitlines(), [], 0
    while i < len(lines):
        raw = lines[i]
        tokens = tokenize(raw)
        if not tokens and raw.strip():
            end = next((j for j in range(i + 1, len(lines) + 1) if tokenize("\n".join(lines[i:j]))), None)
            if end is None:
                return None
            tokens = tokenize("\n".join(lines[i:end]))
            i = end - 1
        if any(t in {"|", "||", "&", "(", ")", ";;"} for t in tokens):
            return None
        body = None
        if "<<" in tokens:
            at = tokens.index("<<")
            if len(tokens) != at + 2:
                return None
            delimiter = tokens[at + 1]
            suffix = raw.split("<<", 1)[1].lstrip()
            if not suffix.startswith(("'" + delimiter + "'", '"' + delimiter + '"')):
                return None
            end = next((j for j in range(i + 1, len(lines)) if lines[j] == delimiter), None)
            if end is None:
                return None
            body = "\n".join(lines[i + 1 : end]) + "\n"
            tokens = tokens[:at]
            i = end
        groups = [[]]
        for token in tokens:
            if token in {";", "&&"}:
                groups.append([])
            else:
                groups[-1].append(token)
        groups = [group for group in groups if group]
        for n, group in enumerate(groups):
            commands.append((group, body if n == len(groups) - 1 else None))
        i += 1
    return commands


def _strict_unwrap(command, tokenize):
    for _ in range(4):
        tokens = tokenize(command)
        if not tokens or Path(tokens[0]).name not in {"bash", "sh", "dash", "zsh", "ksh"}:
            return command
        if len(tokens) != 3 or tokens[1] not in {"-c", "-lc"}:
            return None
        command = tokens[2]
    return None


def _python_source(tokens, body, files, cwd):
    if not tokens or Path(tokens[0]).name not in {"python", "python3"}:
        return None
    args = tokens[1:]
    while args and args[0] in {"-B", "-I", "-u"}:
        args = args[1:]
    if args == ["-"] and body is not None:
        return body, None
    if len(args) == 2 and args[0] == "-c" and body is None:
        return args[1], None
    if len(args) == 1 and body is None:
        path = _path(args[0], cwd)
        if path in files:
            return files[path], path
    return None


def attribute(calls, *, unwrap, tokenize):
    """One source attribution per call, using only prior completed literal write evidence."""
    files = {}
    result = []
    intervals = sorted(
        (call.start_event, call.end_event if call.end_event is not None else float("inf"), i)
        for i, call in enumerate(calls)
        if call.start_event is not None
    )
    overlaps, frontier = set(), -1
    for position, (start, end, index) in enumerate(intervals):
        if frontier > start or (position + 1 < len(intervals) and intervals[position + 1][0] < end):
            overlaps.add(index)
        frontier = max(frontier, end)
    readers = {"cat", "ls", "pwd", "head", "tail", "wc", "stat", "test", "true", "echo", "readlink", "realpath"}
    for index, call in enumerate(calls):
        cwd = None  # A round workspace does not prove a tool's per-command cwd override.
        overlap = index in overlaps
        if not call.succeeded:
            files.clear()
            result.append(Attribution(reason="failed_or_missing_result"))
            continue
        if overlap:
            files.clear()
        command = str(call.input.get("command") or call.input.get("cmd") or "")
        unwrapped = _strict_unwrap(command, tokenize)
        commands = _commands(unwrapped, tokenize) if unwrapped else None
        if commands is None:
            files.clear()
            result.append(Attribution(reason="unattributable_command"))
            continue
        if commands and commands[0][0][:1] == ["cd"]:
            prefix = tokenize(unwrapped.split("&&", 1)[0].strip())
            if len(prefix) == 2 and prefix == commands[0][0] and _path(prefix[1], None):
                cwd = _path(prefix[1], None)
                commands = commands[1:]
            else:
                files.clear()
                result.append(Attribution(reason="unattributable_working_directory"))
                continue
        # The final process's return code is observable. Earlier Python failures may be masked.
        final = commands[-1] if commands else ([], None)
        source = _python_source(*final, files, cwd)
        if source is not None and len(source[0].encode()) <= MAX_SOURCE_BYTES:
            # Earlier statements may change scripts or interpreter state; only harmless directory creation is allowed.
            if all(
                body is None
                and (
                    tokens[:2] == ["mkdir", "-p"]
                    or (
                        tokens
                        and Path(tokens[0]).name in {"python", "python3"}
                        and len(tokens) > 1
                        and Path(tokens[1]).name in {"cca_contract.py", "action_catalog.py"}
                    )
                )
                for tokens, body in commands[:-1]
            ):
                text, script = source
                result.append(Attribution((text,), script, hashlib.sha256(text.encode()).hexdigest(), cwd=cwd))
                files.clear()  # Arbitrary executed Python can mutate files; no assumed future custody.
                continue
        written = False
        if (
            not overlap
            and commands
            and all(tokens[:2] == ["mkdir", "-p"] and body is None for tokens, body in commands[:-1])
        ):
            tokens, body = final
            if len(tokens) == 3 and tokens[:2] == ["cat", ">"] and body is not None:
                path = _path(tokens[2], cwd)
                if path and len(body.encode()) <= MAX_SOURCE_BYTES and len(files) < MAX_SCRIPTS:
                    files[path] = body
                    written = True
            elif len(tokens) == 4 and tokens[:2] == ["sed", "-i"] and body is None:
                expression, path = tokens[2], _path(tokens[3], cwd)
                parts = expression.split("/")
                if (
                    len(parts) == 4
                    and parts[0] == "s"
                    and parts[3] == "g"
                    and parts[1]
                    and all(c.isalnum() or c in " _-" for c in parts[1] + parts[2])
                    and path in files
                ):
                    files[path] = files[path].replace(parts[1], parts[2])
                    written = True
        if not written and not all(
            tokens and tokens[0] in readers and body is None and not any(t in {">", ">>", "<"} for t in tokens)
            for tokens, body in commands
        ):
            files.clear()
        result.append(Attribution(reason="script_write_recorded" if written else "no_attributable_python_execution"))
    return tuple(result)


_FACTS = {"merlin.target.rtl.facts.load_facts", "rtl.facts.load_facts"}

_LEVERS = {"merlin.target.registry.plugins.derived_levers", "rtl_backend.derived_levers"}

_PROFILES = {"merlin.target.registry.plugins.target_profile", "rtl_backend.target_profile"}

_BIJECTIONS = {"merlin.analysis.kernels.cca.action_contract.check_bijection", "cca_contract.check_bijection"}

_LADDERS = {"merlin.analysis.kernels.cca.action_catalog.escalation_ladder", "action_catalog.escalation_ladder"}

_GENERATORS = {
    f"{prefix}.{module}.{operation}"
    for prefix in ("merlin.development.targetgen.generate", "generate")
    for module, operation in (
        ("mlir_scaffold", "generate"),
        ("xdsl", "generate"),
        ("llvm_plan", "generate"),
        ("target_repo", "generate_skeleton"),
        ("target_repo", "emit_contracts"),
    )
}


def discovery(source, context=None):
    """Only reachable calls through admitted imports count; opaque control flow supplies no evidence."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return frozenset()
    aliases, values, found, shadowed = {}, {}, set(), set()
    opaque = False
    context = context or {}
    if context.get("invalid"):
        return frozenset()
    selected_target = context.get("target")
    selected_facts = context.get("selected_facts")

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, "")
        if isinstance(node, ast.Attribute):
            base = qualified(node.value)
            return base + "." + node.attr if base else ""
        return ""

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.Call) and qualified(node.func) in {"pathlib.Path", "Path"} and len(node.args) == 1:
            return literal(node.args[0])
        return None

    def target_ok(node):
        target = literal(node)
        return target is not None and (selected_target is None or target == selected_target)

    def argument(node, keyword, position=0):
        return (
            node.args[position]
            if len(node.args) > position
            else next((k.value for k in node.keywords if k.arg == keyword), None)
        )

    def expression(node):
        nonlocal opaque
        if opaque:
            return None
        if isinstance(node, (ast.Lambda, ast.GeneratorExp)):
            opaque = True
            return None
        if isinstance(node, ast.Call):
            if any(keyword.arg is None for keyword in node.keywords) or any(
                isinstance(argument, ast.Starred) for argument in node.args
            ):
                opaque = True
                return None
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call):
                expression(node.func.value)
            # Arguments execute before the callee. Comprehensions are handled separately below.
            for child in (*node.args, *(k.value for k in node.keywords)):
                expression(child)
            if opaque:
                return None
            name = qualified(node.func)
            if name in {"builtins.exec", "builtins.eval", "exec", "eval"} or (
                isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "__import__"}
            ):
                aliases.clear()
                opaque = True
                return None
            target = argument(node, "target")
            backend_parameter = "backend" if name.startswith("merlin.analysis.kernels.") else "target"
            backend = argument(node, backend_parameter)
            if name in _BIJECTIONS and backend is not None and target_ok(backend):
                found.add("cca_bijection")
            if name in _LADDERS and argument(node, "axis") is not None:
                backend = argument(node, backend_parameter, 1)
                if selected_target is None or (backend is not None and target_ok(backend)):
                    found.add("cca_ladder")
            if name in _FACTS and target is not None and target_ok(target):
                explicit = next((k.value for k in node.keywords if k.arg == "explicit"), None)
                path = literal(explicit) if explicit is not None else None
                default = explicit is None or isinstance(explicit, ast.Constant) and explicit.value is None
                if selected_target is None or (
                    isinstance(selected_facts, dict)
                    and (default or _path(path, context.get("command_cwd")) in selected_facts.get("paths", ()))
                ):
                    found.add("facts")
            if name in _PROFILES and target is not None and target_ok(target):
                return ("profile", literal(target))
            arg = argument(node, "profile")
            if name in _LEVERS and arg is not None:
                profile = values.get(arg.id) if isinstance(arg, ast.Name) else expression(arg)
                if isinstance(profile, tuple) and profile[0] == "profile":
                    found.add("derived")
            if name in _GENERATORS:
                kind = (
                    "target"
                    if name.endswith((".generate_skeleton", ".emit_contracts"))
                    else ("llvm_extension_plan" if ".llvm_plan." in name else "dialect_plan")
                )
                plan_arg = argument(node, kind)
                plan = (
                    values.get(plan_arg.id)
                    if isinstance(plan_arg, ast.Name)
                    else (expression(plan_arg) if plan_arg is not None else None)
                )
                if (
                    selected_target is None
                    or (isinstance(plan, tuple) and plan == ("plan", selected_target))
                    or (plan_arg is not None and target_ok(plan_arg))
                ):
                    found.add("scaffold")
            known = _FACTS | _PROFILES | _LEVERS | _GENERATORS | _BIJECTIONS | _LADDERS | {"pathlib.Path", "json.dumps"}
            pure = {
                "print",
                "list",
                "dict",
                "tuple",
                "set",
                "str",
                "repr",
                "len",
                "bool",
                "int",
                "float",
                "sorted",
                "range",
                "enumerate",
                "zip",
                "min",
                "max",
                "isinstance",
                "hasattr",
            }
            if name not in known and not (
                isinstance(node.func, ast.Name) and node.func.id in pure and node.func.id not in shadowed
            ):
                opaque = True
            return None
        if isinstance(node, ast.Dict):
            plan = None
            for key, value in zip(node.keys, node.values):
                if key is not None:
                    expression(key)
                expression(value)
                if isinstance(key, ast.Constant) and key.value == "target":
                    plan = ("plan", literal(value))
            return plan
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            # The outer iterable is evaluated; the body may run zero times.
            if node.generators:
                expression(node.generators[0].iter)
            if any(
                isinstance(child, (ast.Call, ast.NamedExpr))
                for child in ast.walk(node)
                if child is not node.generators[0].iter
            ):
                # The outer iterable has already executed; the body/filters may mutate imports.
                body = [node.elt] if hasattr(node, "elt") else [node.key, node.value]
                body += [condition for generator in node.generators for condition in generator.ifs]
                if any(isinstance(child, (ast.Call, ast.NamedExpr)) for expr in body for child in ast.walk(expr)):
                    opaque = True
            return None
        if isinstance(node, ast.BoolOp):
            if node.values:
                expression(node.values[0])
            if any(
                isinstance(child, (ast.Call, ast.NamedExpr)) for tail in node.values[1:] for child in ast.walk(tail)
            ):
                opaque = True
            return None
        if isinstance(node, ast.IfExp):
            expression(node.test)
            if isinstance(node.test, ast.Constant):
                expression(node.body if node.test.value else node.orelse)
            else:
                opaque = True
            return None
        for child in ast.iter_child_nodes(node):
            expression(child)
        return literal(node)

    def block(statements):
        nonlocal opaque
        for statement in statements:
            if opaque:
                break
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    local = alias.asname or alias.name.split(".")[0]
                    aliases[local] = alias.name if alias.asname else alias.name.split(".")[0]
                    shadowed.add(local)
            elif isinstance(statement, ast.ImportFrom):
                if statement.level or not statement.module:
                    aliases.clear()
                    continue
                for alias in statement.names:
                    if alias.name == "*":
                        aliases.clear()
                    else:
                        aliases[alias.asname or alias.name] = statement.module + "." + alias.name
                        shadowed.add(alias.asname or alias.name)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = expression(statement.value) if statement.value is not None else None
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        aliases.pop(target.id, None)
                        shadowed.add(target.id)
                        values[target.id] = value if value is not None else literal(statement.value)
                    else:
                        opaque = True  # Attribute mutation may replace an imported API.
            elif isinstance(statement, ast.Expr):
                expression(statement.value)
            elif isinstance(statement, ast.Assert):
                expression(statement.test)
            elif isinstance(statement, ast.If):
                expression(statement.test)
                if isinstance(statement.test, ast.Constant):
                    block(statement.body if statement.test.value else statement.orelse)
                else:
                    opaque = True
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                opaque = True  # Classes, defaults/decorators and function dispatch are intentionally opaque.
            elif isinstance(statement, (ast.For, ast.While, ast.Try, ast.With, ast.Match)):
                opaque = True
            elif not isinstance(statement, (ast.Pass, ast.Global, ast.Nonlocal)):
                opaque = True

    block(tree.body)
    return frozenset(found)


def mutates_submission(source):
    """Keep script-contained writes in the same conservative first-edit boundary."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    paths = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(value, ast.Constant) and isinstance(value.value, str) and "submission/" in value.value
            for value in ast.walk(node.value)
        ):
            paths.update(target.id for target in node.targets if isinstance(target, ast.Name))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else (node.func.id if isinstance(node.func, ast.Name) else "")
        )
        if name not in {"write_text", "write_bytes", "mkdir", "unlink", "rename", "open"}:
            continue
        target = node.func.value if isinstance(node.func, ast.Attribute) else (node.args[0] if node.args else None)
        if target is not None and any(
            (isinstance(value, ast.Constant) and isinstance(value.value, str) and "submission/" in value.value)
            or (isinstance(value, ast.Name) and value.id in paths)
            for value in ast.walk(target)
        ):
            return True
    return False


# ---- authoring_conformance ---------------------------------------------------------------------
# Conformance summaries used by the Phase 1 authoring continuation.

_MAX_ADVISORY_REUSE_S = 60


def conformance_ever(rounds) -> dict:
    """Fold the run-level gate: true if ever satisfied, None if always inapplicable.

    Resume rebuilds this same view from persisted round records.
    """
    ever: dict = {}
    for entry in rounds or []:
        checks = ((entry or {}).get("conformance") or {}).get("checks") or {}
        for key, value in checks.items():
            if value is None:
                ever.setdefault(key, None)
            else:
                ever[key] = bool(ever.get(key)) or bool(value)
    return ever


def workflow_conformance(
    tpath: Path, submission_dir: Path, arm: str, endpoint_kind: str, resolved_tools
) -> tuple[dict, bool]:
    """Recompute current workflow evidence from resolved tools; checker errors fail closed."""
    try:
        from merlin.development.backend.authoring import conformance as _CONF
        from merlin.development.backend.authoring.library.compiler_workspace import selected as full_selection
        from merlin.development.backend.authoring.profiles import current as current_author_profile
        from merlin.development.backend.feedback.execution.snapshot_evaluator import (
            selected as external_selection,
        )

        treatment = {}
        if external_selection():
            treatment = {
                "compiler_access": "full_compiler" if full_selection() is not None else "selected_library",
                "compiler_execution": "snapshot_external_v1",
            }
        profile = current_author_profile()
        if profile is not None:
            treatment["author_profile"] = profile.name
        conf = _CONF.compute(tpath, submission_dir, arm, endpoint_kind, resolved_tools=resolved_tools, **treatment)
    except Exception as exc:  # noqa: BLE001 — conformance is a formal gate; unavailable means false
        conf = {"conformant": False, "error": f"{type(exc).__name__}: {exc}"}
    return conf, conf.get("conformant") is True


def verify_live_selection(options, environment: dict) -> None:
    """This opt-in author feedback selection is immutable across continuation/resume."""
    recorded = environment.get("run_config") or {}
    prior = recorded.get("live_conformance_feedback", False)
    if type(prior) is not bool or prior != options.live_conformance_feedback:
        raise RuntimeError("resume refused: live workflow conformance feedback selection changed")


def _public_projection(conf: dict, submission: Path, *, running: bool) -> dict:
    """Publish no transcript, raw tool results, private paths or exception strings."""
    checks = conf.get("checks")
    available = isinstance(checks, dict) and not any(key in conf for key in ("error", "regex_scan_error"))
    public_checks = {
        key: value
        for key, value in (checks.items() if isinstance(checks, dict) else ())
        if isinstance(key, str) and (type(value) is bool or value is None)
    }
    hits = []
    for hit in conf.get("regex_hits") or []:
        if not isinstance(hit, dict) or not isinstance(hit.get("file"), str):
            continue
        path, line = Path(hit["file"]), hit.get("line")
        if path.is_absolute() or ".." in path.parts or type(line) is not int or line < 1:
            continue
        if not (submission / path).resolve().is_relative_to(submission.resolve()):
            continue
        hits.append({"file": str(path), "line": line})
    return {
        "schema_version": 1,
        "advisory": True,
        "available": available,
        "conformant": available and conf.get("conformant") is True,
        "checks": public_checks,
        "regex_hits": hits,
        "discovery_source_attribution": {
            key: value
            for key, value in (conf.get("discovery_source_attribution") or {}).items()
            if isinstance(key, str) and type(value) is int
        },
        "running": running,
        "updated_at": datetime.now(UTC).isoformat(),
    }


def _publish(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".workflow-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _path_state(path: Path) -> tuple:
    try:
        link, target = path.lstat(), path.stat()
    except FileNotFoundError:
        return (str(path), None)
    return (str(path),) + tuple(
        (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        for value in (link, target)
    )


def _observation_state(transcript: Path, submission: Path, tools) -> tuple:
    """Cheap invalidation for advisory reuse only; never a formal evidence identity."""
    # Include membership and symlink targets. The real scanner also considers
    # vendored Python, so a directory name must not exempt it from invalidation.
    return (
        tools,
        _path_state(transcript),
        _path_state(submission),
        tuple(_path_state(path) for path in sorted(submission.rglob("*.py"))),
    )


@contextmanager
def live_feedback(
    workspace: Path,
    run_dir: Path,
    round_index: int,
    arm: str,
    endpoint_kind: str,
    resolved_tools,
    *,
    interval_s: float = 5,
):
    """Observe the same actual checker during one launch; never run a capsule grade.

    This view is advisory and current-round only. Formal completion independently
    recomputes its existing gate. Missing startup evidence remains false while work
    is in progress, and a discovery-order failure cannot be repaired by late calls.
    """
    stop = threading.Event()
    transcript = run_dir / "rounds" / f"round_{round_index:02d}.transcript.jsonl"
    submission = workspace / "submission"
    destination = workspace / "qa" / "workflow_conformance.json"
    last_error = None
    last_state = None
    last_checked = 0.0

    def private_error(value: str) -> None:
        nonlocal last_error
        if value == last_error:
            return
        last_error = value
        try:
            descriptor = os.open(
                run_dir / "workflow_conformance_errors.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            with os.fdopen(descriptor, "a") as stream:
                stream.write(json.dumps({"round": round_index, "error": value}) + "\n")
        except OSError:
            pass  # Advisory publication must not replace the formal completion gate.

    def refresh(*, running: bool) -> None:
        nonlocal last_state, last_checked
        try:
            selected = resolved_tools()
            tools = None if selected is None else tuple(sorted(str(value) for value in selected))
            before = _observation_state(transcript, submission, tools)
            if (
                running
                and last_state is not None
                and monotonic() - last_checked < _MAX_ADVISORY_REUSE_S
                and (before, _path_state(destination)) == last_state
            ):
                return
            last_state = None
            conf, _ = workflow_conformance(transcript, submission, arm, endpoint_kind, tools)
            for key in ("error", "regex_scan_error"):
                if key in conf:
                    private_error(str(conf[key]))
            after = _observation_state(transcript, submission, tools)
            public = _public_projection(conf if before == after else {}, submission, running=running)
            _publish(destination, public)
            if before == after and public["available"]:
                last_state = (after, _path_state(destination))
                last_checked = monotonic()
        except Exception as exc:  # noqa: BLE001 — errors remain private; final gate is independent
            last_state = None
            private_error(f"{type(exc).__name__}: {exc}")
            try:
                _publish(destination, _public_projection({}, submission, running=running))
            except OSError:
                pass

    def observe() -> None:
        while not stop.wait(interval_s):
            refresh(running=True)

    refresh(running=True)
    thread = threading.Thread(target=observe, name="workflow-conformance-feedback", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        refresh(running=False)


def live_launch(launch, enabled: bool, workspace: Path, run_dir: Path, arm: str, endpoint_kind: str, resolved_tools):
    """Wrap the shared author launch once, including normal, continuous and L3 repair paths."""
    if not enabled:
        return launch

    @wraps(launch)
    def invoke(*args, **kwargs):
        with live_feedback(workspace, run_dir, args[6], arm, endpoint_kind, resolved_tools):
            return launch(*args, **kwargs)

    return invoke
