"""Print a topological order of Merlin/MerlinVerify modules reachable from the given closure JSONs."""
import json
import sys
from pathlib import Path

modules = {}
for name in sys.argv[1:]:
    for row in json.loads(Path(name).read_text())["files"]:
        modules[row["module"]] = row["path"]


def imports(path):
    result = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "import":
            result.extend(p for p in parts[1:] if p in modules)
    return result


order, state = [], {}


def visit(module):
    if state.get(module) == 2:
        return
    if state.get(module) == 1:
        raise SystemExit(f"cycle at {module}")
    state[module] = 1
    for dependency in imports(modules[module]):
        visit(dependency)
    state[module] = 2
    order.append(module)


for module in sorted(modules):
    visit(module)
for module in order:
    print(module, modules[module])
