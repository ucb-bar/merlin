"""Narrow transcript syntax recognition; never execute commands or grant directories."""

from __future__ import annotations

import ast
import shlex
from pathlib import Path

from .conformance_sources import _commands, _strict_unwrap


def _tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return []


def _data_assignment(statement: ast.stmt) -> bool:
    # Name assignment of inert containers cannot itself open a file. Exclude calls,
    # interpolation, overloaded operations, unpacking and attribute/subscript stores.
    if not isinstance(statement, ast.Assign) or not all(isinstance(t, ast.Name) for t in statement.targets):
        return False
    for node in ast.walk(statement.value):
        if not isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Constant, ast.Name, ast.Load)):
            return False
        if isinstance(node, ast.Dict) and any(not isinstance(key, ast.Constant) for key in node.keys):
            return False
    return True


def _only_output_uses(tree: ast.Module, statement: ast.Assign) -> bool:
    """Keep data bindings unless unused or consumed by one literal-path report write.

    This is deliberately not a general Python dataflow analysis. Aliases, unknown
    readers, execution, rebinding or control flow keep the original text available
    to the conservative audit. In particular a constant path used by open(p) must
    never disappear merely because its assignment is inert.
    """
    names = {target.id for target in statement.targets}
    own = set(ast.walk(statement))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    uses = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in names and node not in own]
    # Only known standard-library report serialization is recognized. Rebinding
    # these names or modifying attributes invalidates that syntactic recognition.
    sensitive = {"Path", "json", "str"}
    path_import = json_import = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return False
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in sensitive:
            return False
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(node.ctx, (ast.Store, ast.Del)):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "exec", "eval", "compile", "globals", "locals", "vars", "setattr", "getattr", "__import__",
        }:
            return False
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound = alias.asname or alias.name
                if bound in sensitive:
                    if node.module != "pathlib" or node.level or alias.name != "Path" or alias.asname:
                        return False
                    path_import = True
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                if bound in sensitive:
                    if alias.name != "json" or alias.asname:
                        return False
                    json_import = True
    output_calls = set()
    for use in uses:
        if not isinstance(use.ctx, ast.Load):
            return False
        serialize = parents.get(use)
        if not isinstance(serialize, ast.Call) or serialize.args != [use]:
            return False
        fn = serialize.func
        if not (
            isinstance(fn, ast.Name) and fn.id == "str" and not serialize.keywords
            or json_import and isinstance(fn, ast.Attribute) and fn.attr == "dumps"
            and isinstance(fn.value, ast.Name) and fn.value.id == "json"
            and all(keyword.arg is not None and isinstance(keyword.value, ast.Constant) for keyword in serialize.keywords)
        ):
            return False
        write = parents.get(serialize)
        if not isinstance(write, ast.Call) or write.args != [serialize] or write.keywords:
            return False
        if not isinstance(write.func, ast.Attribute) or write.func.attr != "write_text":
            return False
        destination = write.func.value
        if not (path_import and isinstance(destination, ast.Call) and isinstance(destination.func, ast.Name)
                and destination.func.id == "Path" and len(destination.args) == 1 and not destination.keywords
                and isinstance(destination.args[0], ast.Constant) and isinstance(destination.args[0].value, str)):
            return False
        if not isinstance(parents.get(write), ast.Expr) or parents.get(parents.get(write)) is not tree:
            return False
        output_calls.update(node for node in ast.walk(write) if isinstance(node, ast.Call))
    # Even without a lexical name use, opaque later code may consume bindings
    # through globals/aliases. Only the report writes above and literal prints
    # may follow a discarded binding; otherwise retain it conservatively.
    for later in tree.body:
        if later.lineno <= statement.end_lineno:
            continue
        if not isinstance(later, ast.Expr):
            return False
        for node in ast.walk(later):
            if isinstance(node, ast.Call) and node not in output_calls:
                if not (isinstance(node.func, ast.Name) and node.func.id == "print" and not node.keywords
                        and all(isinstance(arg, ast.Constant) for arg in node.args)):
                    return False
    return True


def read_view(command: str) -> str:
    """Remove only proved inert Python data statements from content-read recognition.

    Literal quoted heredocs have no shell expansion. Unknown shell/Python, unquoted
    heredocs and command substitutions retain the original conservative scan. Actual
    Python reads and all other statements remain, including code after a data assignment.
    This projection is never used for oracle-import/use auditing.
    """
    if len(command) > 256 * 1024:
        return command
    unwrapped = _strict_unwrap(command, _tokens)
    commands = _commands(unwrapped, _tokens) if unwrapped is not None else None
    if commands is None:
        return command
    projected = unwrapped
    for tokens, body in commands:
        if len(commands) != 1 or body is None or not tokens or Path(tokens[0]).name not in {"python", "python3"}:
            continue
        if tokens[1:] != ["-"]:
            continue
        # Bind the body position, not a substring that might also occur in an
        # earlier shell word. Composite heredoc commands remain conservative.
        start = unwrapped.find("\n") + 1
        if not start or not unwrapped[start:].startswith(body):
            continue
        try:
            tree = ast.parse(body)
        except (SyntaxError, ValueError, RecursionError):
            continue
        lines = body.splitlines(keepends=True)
        for statement in tree.body:
            if not _data_assignment(statement) or not _only_output_uses(tree, statement):
                continue
            first, last = statement.lineno - 1, statement.end_lineno - 1
            # AST columns are UTF-8 byte offsets. Only whole standalone statements
            # are removed; semicolon neighbours remain conservative.
            if lines[first].encode()[:statement.col_offset].strip():
                continue
            if lines[last].encode()[statement.end_col_offset:].strip():
                continue
            lines[first:last + 1] = ["\n"] * (last - first + 1)
        # The legacy shell recognizer's word boundary after "open(" misses a
        # quoted first argument. Mark actual parsed calls, never prose mentions.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open":
                lines.append("\nread_text " + ast.get_source_segment(body, node).replace("\n", " ") + "\n")
        projected = unwrapped[:start] + "".join(lines) + unwrapped[start + len(body):]
    return projected


def public_feedback_context(command: str) -> bool:
    """Recognize simple content/listing commands that cannot change their own cwd."""
    unwrapped = _strict_unwrap(command, _tokens)
    if unwrapped is None or "\n" in unwrapped or "$(" in unwrapped or "`" in unwrapped:
        return False
    tokens = _tokens(unwrapped)
    if not tokens or any(token in {"&", "(", ")", "<<", ";;"} for token in tokens):
        return False
    commands = [[]]
    for token in tokens:
        if token in {";", "&&", "||", "|"}:
            commands.append([])
        else:
            commands[-1].append(token)
    readers = {"cat", "head", "tail", "less", "more", "sed", "awk", "xxd", "od", "grep",
               "find", "ls", "locate", "which", "whereis", "pwd", "wc", "stat"}
    return all(tokens and Path(tokens[0]).name in readers for tokens in commands)


def published_feedback(word: str, workspace: Path | None) -> bool:
    """Only the exact host-published public workflow path, never a same-named file."""
    if workspace is None:
        return False
    raw = word.strip().strip("'\"")
    if not raw or any(c in raw for c in "$`*?[]\x00"):
        return False
    candidate = Path(raw)
    if ".." in candidate.parts:
        return False
    try:
        root = Path(workspace).resolve(strict=True)
        expected = root / "qa" / "workflow_conformance.json"
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate == expected and candidate.is_file() and candidate.resolve(strict=True) == expected
    except (OSError, RuntimeError):
        return False
