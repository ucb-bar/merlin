"""Resolve selected finite lazy Python exports without importing or executing source.

Only literal export tables and bounded literal comprehensions are evaluated. The getter must bind
the requested symbol through that table and pass the selected module to ``import_module``. This
accounts implementation bytes; it is not a capability grant and does not override answer masks.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LazyExportResolution:
    status: str
    module: str | None = None
    reason: str = ""


class _Unsupported(ValueError):
    pass


def _literal(node: ast.AST, variables: dict[str, Any], budget: list[int]) -> Any:
    budget[0] -= 1
    if budget[0] < 0:
        raise _Unsupported("lazy export table exceeds bounded static evaluation")
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, bool, type(None))):
        return node.value
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_literal(item, variables, budget) for item in node.elts)
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise _Unsupported("unpacked lazy export tables are not statically resolved")
        return {_literal(key, variables, budget): _literal(value, variables, budget)
                for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.DictComp):
        result = {}

        def bind(target, value, scope):
            if isinstance(target, ast.Name):
                scope[target.id] = value
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
                if len(target.elts) != len(value):
                    raise _Unsupported("lazy export comprehension unpacking does not match")
                for item, element in zip(target.elts, value):
                    bind(item, element, scope)
            else:
                raise _Unsupported("unsupported lazy export comprehension target")

        def expand(index, scope):
            if index == len(node.generators):
                result[_literal(node.key, scope, budget)] = _literal(node.value, scope, budget)
                return
            generator = node.generators[index]
            if generator.is_async or generator.ifs:
                raise _Unsupported("conditional or async lazy exports are unresolved")
            values = _literal(generator.iter, scope, budget)
            if not isinstance(values, (tuple, list)):
                raise _Unsupported("lazy export generator is not a literal sequence")
            for value in values:
                child = dict(scope)
                bind(generator.target, value, child)
                expand(index + 1, child)

        expand(0, variables)
        return result
    if isinstance(node, ast.JoinedStr):
        chunks = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                chunks.append(part.value)
            elif isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None:
                value = _literal(part.value, variables, budget)
                if not isinstance(value, str):
                    raise _Unsupported("lazy import module interpolation is not a string")
                chunks.append(value)
            else:
                raise _Unsupported("unsupported lazy import module interpolation")
        return "".join(chunks)
    raise _Unsupported(f"unsupported static lazy export expression: {type(node).__name__}")


def resolve_lazy_export(source: bytes, *, package: str, symbol: str) -> LazyExportResolution:
    """Resolve one requested public export, never every sibling export of a lazy package."""
    if len(source) > 131072:
        return LazyExportResolution("unresolved", reason="package initializer exceeds static parsing policy")
    tree = ast.parse(source)
    # An eagerly bound attribute never invokes the package's __getattr__.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            return LazyExportResolution("not_lazy")
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split(".")[0]) == symbol for alias in node.names):
            return LazyExportResolution("not_lazy")
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and any(
                isinstance(target, ast.Name) and target.id == symbol
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])):
            return LazyExportResolution("not_lazy")
    getters = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"]
    if not getters:
        return LazyExportResolution("not_lazy")
    if len(getters) != 1 or len(getters[0].args.args) != 1:
        return LazyExportResolution("unresolved", reason="lazy getter signature is not statically supported")
    getter = getters[0]
    requested = getter.args.args[0].arg
    table_name = module_variable = None
    for node in ast.walk(getter):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "get"
                and isinstance(node.value.func.value, ast.Name) and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Name) and node.value.args[0].id == requested):
            table_name, module_variable = node.value.func.value.id, node.targets[0].id
    if table_name is None:
        return LazyExportResolution("unresolved", reason="lazy getter has no recognized finite export table")
    assignments = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                   and any(isinstance(target, ast.Name) and target.id == table_name
                           for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
    if len(assignments) != 1:
        return LazyExportResolution("unresolved", reason="lazy export table is absent or reassigned")
    try:
        table = _literal(assignments[0].value, {}, [20000])
        if not isinstance(table, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                             for k, v in table.items()):
            raise _Unsupported("lazy export table must map symbol names to module names")
        if symbol == "*":
            raise _Unsupported("wildcard lazy exports require explicit selected-symbol imports")
        if symbol not in table:
            return LazyExportResolution("not_exported")
        module_calls = []
        for node in ast.walk(getter):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            name = (node.func.id if isinstance(node.func, ast.Name) else
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if name != "import_module" or not any(isinstance(item, ast.Name) and item.id == module_variable
                                                 for item in ast.walk(node.args[0])):
                continue
            module_calls.append(node)
        if len(module_calls) != 1:
            raise _Unsupported("selected lazy module does not reach one supported import_module call")
        call = module_calls[0]
        module = _literal(call.args[0], {module_variable: table[symbol]}, [20000])
        if not isinstance(module, str):
            raise _Unsupported("lazy import module is not a static string")
        if module.startswith("."):
            if (len(call.args) != 2 or not isinstance(call.args[1], ast.Name)
                    or call.args[1].id != "__name__"):
                raise _Unsupported("relative lazy import does not bind the declaring package")
            dots = len(module) - len(module.lstrip("."))
            components = package.split(".")
            if dots > len(components):
                raise _Unsupported("relative lazy import escapes the package hierarchy")
            module = ".".join((*components[:len(components) - dots + 1], module[dots:]))
        if not module or any(not component.isidentifier() for component in module.split(".")):
            raise _Unsupported("lazy import is not a Python module name")
        return LazyExportResolution("resolved", module=module)
    except _Unsupported as error:
        return LazyExportResolution("unresolved", reason=str(error))


def imported_attribute_paths(tree: ast.AST, bindings: dict[str, set[str]]) -> set[str]:
    """Find selected package attributes through imported aliases and constant getattr calls."""
    def paths(node):
        if isinstance(node, ast.Name):
            return bindings.get(node.id, set())
        if isinstance(node, ast.Attribute):
            return {base + "." + node.attr for base in paths(node.value)}
        return set()

    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            result.update(paths(node))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
            result.update(base + "." + node.args[1].value for base in paths(node.args[0]))
    return result
