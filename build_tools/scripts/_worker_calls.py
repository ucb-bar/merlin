"""Static sibling-script calls made through subprocess, including local argv-forwarding helpers.

Workers use interpreters whose MLIR/framework dependencies differ from the caller's. Importing the
worker in the caller is therefore not a valid way to establish its wiring. Only executable
siblings used as the script argument of a process launch count; dormant path constants do not.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (base := _name(node.value)):
        return f"{base}.{node.attr}"
    return None


def _executable(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and _name(node.test.left) == "__name__"
        and len(node.test.ops) == len(node.test.comparators) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
        and any(isinstance(call, ast.Call) for statement in node.body for call in ast.walk(statement))
        for node in tree.body
    )


def script_calls(path: Path, tree: ast.AST) -> set[Path]:
    """Resolve literal ``Path(__file__).with_name(...)`` script arguments without executing code.

    Like import/name matching in the wiring gate, this establishes a static caller, not proof of
    runtime execution. Dynamic paths and argv transformations remain outside this conservative scan.
    """
    launches: set[str] = set()
    operations = {"run", "Popen", "call", "check_call", "check_output"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    launches.update(f"{alias.asname or alias.name}.{operation}" for operation in operations)
        elif isinstance(node, ast.ImportFrom) and node.module == "subprocess" and not node.level:
            launches.update(alias.asname or alias.name for alias in node.names if alias.name in operations)
    if not launches:
        return set()

    # A local helper counts only when it forwards its first argv parameter to a process launcher.
    # Follow chains of such helpers; merely accepting a list or naming a script is insufficient.
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    while True:
        previous = set(launches)
        for function in functions:
            parameters = [*function.args.posonlyargs, *function.args.args]
            if parameters and any(
                isinstance(call, ast.Call)
                and _name(call.func) in launches
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == parameters[0].arg
                for call in ast.walk(function)
            ):
                launches.add(function.name)
        if launches == previous:
            break

    bindings: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = node.value

    def sibling(node: ast.AST, seen: frozenset[str] = frozenset()) -> Path | None:
        if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
            return sibling(bindings[node.id], seen | {node.id})
        if isinstance(node, ast.Call) and _name(node.func) == "str" and len(node.args) == 1:
            return sibling(node.args[0], seen)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "with_name"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and isinstance(node.func.value, ast.Call)
            and _name(node.func.value.func) in {"Path", "pathlib.Path"}
            and len(node.func.value.args) == 1
            and _name(node.func.value.args[0]) == "__file__"
        ):
            filename = node.args[0].value
            if Path(filename).name == filename and filename.endswith(".py"):
                return path.with_name(filename)
        return None

    found: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _name(node.func) in launches and node.args:
            argv = node.args[0]
            if isinstance(argv, (ast.List, ast.Tuple)) and len(argv.elts) >= 2:
                script = sibling(argv.elts[1])
                if script is not None and _executable(script):
                    found.add(script)
    return found
