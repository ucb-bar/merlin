"""Bounded transcript source custody and reachable imported discovery calls.

This is an observer, never a shell/Python interpreter. Opaque writes, overlapping calls,
failed results and ambiguous execution refuse attribution; current files are never replayed.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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
    return {"schema": "merlin.phase1.conformance_execution.v1", "workspace": str(Path(workspace).absolute()),
            "target": target, "selected_facts": facts}


def validate_context(value):
    if value is None:
        return None  # Historical inline-only observations have no native binding.
    if not isinstance(value, dict) or value.get("schema") != "merlin.phase1.conformance_execution.v1":
        return {"invalid": True}
    workspace, target, facts = value.get("workspace"), value.get("target"), value.get("selected_facts")
    if not isinstance(workspace, str) or not _path(workspace, None) or (
        target is not None and not isinstance(target, str)
    ):
        return {"invalid": True}
    if facts is not None and (
        not isinstance(facts, dict) or not isinstance(facts.get("sha256"), str)
        or len(facts["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in facts["sha256"])
        or not isinstance(facts.get("paths"), list) or not facts["paths"]
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
            body = "\n".join(lines[i + 1:end]) + "\n"
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
    intervals = sorted((call.start_event, call.end_event if call.end_event is not None else float("inf"), i)
                       for i, call in enumerate(calls) if call.start_event is not None)
    overlaps, frontier = set(), -1
    for position, (start, end, index) in enumerate(intervals):
        if frontier > start or (position + 1 < len(intervals) and intervals[position + 1][0] < end):
            overlaps.add(index)
        frontier = max(frontier, end)
    readers = {"cat", "ls", "pwd", "head", "tail", "wc", "stat", "test",
               "true", "echo", "readlink", "realpath"}
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
            if all(body is None and (
                tokens[:2] == ["mkdir", "-p"] or (
                    tokens and Path(tokens[0]).name in {"python", "python3"} and len(tokens) > 1
                    and Path(tokens[1]).name in {"cca_contract.py", "action_catalog.py"}
                )
            ) for tokens, body in commands[:-1]):
                text, script = source
                result.append(Attribution((text,), script, hashlib.sha256(text.encode()).hexdigest(), cwd=cwd))
                files.clear()  # Arbitrary executed Python can mutate files; no assumed future custody.
                continue
        written = False
        if not overlap and commands and all(
            tokens[:2] == ["mkdir", "-p"] and body is None for tokens, body in commands[:-1]
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
                if (len(parts) == 4 and parts[0] == "s" and parts[3] == "g"
                        and parts[1] and all(c.isalnum() or c in " _-" for c in parts[1] + parts[2])
                        and path in files):
                    files[path] = files[path].replace(parts[1], parts[2])
                    written = True
        if not written and not all(
            tokens and tokens[0] in readers and body is None
            and not any(t in {">", ">>", "<"} for t in tokens)
            for tokens, body in commands
        ):
            files.clear()
        result.append(Attribution(reason="script_write_recorded" if written else "no_attributable_python_execution"))
    return tuple(result)


_FACTS = {"merlin.targetgen.rtl.facts.load_facts", "rtl.facts.load_facts"}
_LEVERS = {"merlin.targetgen.rtl_backend.derived_levers", "rtl_backend.derived_levers"}
_PROFILES = {"merlin.targetgen.rtl_backend.target_profile", "rtl_backend.target_profile"}
_BIJECTIONS = {"merlin.kernels.cca_contract.check_bijection", "cca_contract.check_bijection"}
_LADDERS = {"merlin.kernels.action_catalog.escalation_ladder", "action_catalog.escalation_ladder"}
_GENERATORS = {f"{prefix}.{module}.{operation}" for prefix in ("merlin.targetgen.generate", "generate")
               for module, operation in (("mlir_scaffold", "generate"), ("xdsl", "generate"),
                                         ("llvm_plan", "generate"), ("target_repo", "generate_skeleton"),
                                         ("target_repo", "emit_contracts"))}


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
        return node.args[position] if len(node.args) > position else next(
            (k.value for k in node.keywords if k.arg == keyword), None)

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
            backend_parameter = "backend" if name.startswith("merlin.kernels.") else "target"
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
                    isinstance(selected_facts, dict) and (
                        default or _path(path, context.get("command_cwd")) in selected_facts.get("paths", ())
                    )
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
                kind = "target" if name.endswith((".generate_skeleton", ".emit_contracts")) else (
                    "llvm_extension_plan" if ".llvm_plan." in name else "dialect_plan"
                )
                plan_arg = argument(node, kind)
                plan = values.get(plan_arg.id) if isinstance(plan_arg, ast.Name) else (
                    expression(plan_arg) if plan_arg is not None else None)
                if selected_target is None or (
                    isinstance(plan, tuple) and plan == ("plan", selected_target)
                ) or (plan_arg is not None and target_ok(plan_arg)):
                    found.add("scaffold")
            known = _FACTS | _PROFILES | _LEVERS | _GENERATORS | _BIJECTIONS | _LADDERS | {
                "pathlib.Path", "json.dumps"}
            pure = {"print", "list", "dict", "tuple", "set", "str", "repr", "len", "bool", "int", "float",
                    "sorted", "range", "enumerate", "zip", "min", "max", "isinstance", "hasattr"}
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
            if any(isinstance(child, (ast.Call, ast.NamedExpr)) for child in ast.walk(node)
                   if child is not node.generators[0].iter):
                # The outer iterable has already executed; the body/filters may mutate imports.
                body = [node.elt] if hasattr(node, "elt") else [node.key, node.value]
                body += [condition for generator in node.generators for condition in generator.ifs]
                if any(isinstance(child, (ast.Call, ast.NamedExpr)) for expr in body for child in ast.walk(expr)):
                    opaque = True
            return None
        if isinstance(node, ast.BoolOp):
            if node.values:
                expression(node.values[0])
            if any(isinstance(child, (ast.Call, ast.NamedExpr))
                   for tail in node.values[1:] for child in ast.walk(tail)):
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
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else ""
        )
        if name not in {"write_text", "write_bytes", "mkdir", "unlink", "rename", "open"}:
            continue
        target = node.func.value if isinstance(node.func, ast.Attribute) else (node.args[0] if node.args else None)
        if target is not None and any(
            (isinstance(value, ast.Constant) and isinstance(value.value, str) and "submission/" in value.value)
            or (isinstance(value, ast.Name) and value.id in paths) for value in ast.walk(target)
        ):
            return True
    return False
