from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

from merlin.common.paths import repo_root

GATE_PATH = repo_root() / "build_tools" / "checks" / "check_target_package_duplication.py"
ORIGINAL = """
def pick_tiles(rows, cols, depth, budget):
    best = None
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = i * depth + j * depth
            if cost <= budget and (best is None or cost > best[0]):
                best = (cost, i, j)
    return best
"""


def _gate():
    spec = importlib.util.spec_from_file_location("_dup_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        path = tmp_path / "experiments/inputs/target-inputs" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


def test_the_same_function_in_two_target_packages_is_flagged(tmp_path):
    gate = _gate()
    root = _tree(tmp_path, {"alpha/backend/sched.py": ORIGINAL, "beta/backend/sched.py": ORIGINAL})
    problems, rc = gate.verdict(gate.shapes(root), set())
    assert rc == 1, "a verbatim copy across target packages was not flagged"
    assert any("alpha/sched.py:pick_tiles" in p and "beta/sched.py:pick_tiles" in p for p in problems)
    assert any("move it to the core" in p for p in problems), "the finding does not say what to do"
