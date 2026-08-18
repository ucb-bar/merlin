"""The workload-specific-constant scanner (verify_no_cheat.scan_workload_constants) — a frozen
compiler must generalize, so shape-equality dispatch and model-name branches are flagged while legal
target-derived thresholds pass."""
from __future__ import annotations

import sys
import types
from pathlib import Path

from merlin.common.paths import repo_root


def _load_scanner():
    """Import verify_no_cheat with its _common bootstrap stubbed (the scanner itself needs none)."""
    harness = repo_root() / "merlin" / "experiments" / "capsule_bench" / "harness"
    stub = types.ModuleType("_common")
    stub.EXP = harness           # any dir; the unit test calls the scanner directly
    stub.REPO = repo_root()
    stub.experiment_conditions = lambda: []
    sys.modules.setdefault("_common", stub)
    sys.path.insert(0, str(harness))
    import verify_no_cheat as V
    return V


def test_flags_cheats_and_passes_legal_code(tmp_path: Path):
    V = _load_scanner()
    (tmp_path / "cheat_shape.py").write_text("def f(M):\n    if M == 512:\n        return 1\n")
    (tmp_path / "cheat_name.py").write_text("def f(model_name):\n    if model_name == 'llama':\n        return 1\n")
    (tmp_path / "cheat_member.py").write_text("def f(K):\n    if K in (512, 4096):\n        return 1\n")
    (tmp_path / "legal.py").write_text(
        "def f(K, tile, dim, M):\n"
        "    if K >= tile * 4:\n"        # legal target-derived threshold
        "        return 1\n"
        "    if dim == 0:\n"             # legal small structural guard
        "        return 2\n"
        "    if M == 2:\n"               # legal small structural guard
        "        return 3\n"
    )
    hits = V.scan_workload_constants(tmp_path)
    files = {h.split(":")[0] for h in hits}
    assert "cheat_shape.py" in files
    assert "cheat_name.py" in files
    assert "cheat_member.py" in files
    assert "legal.py" not in files
    assert len(hits) == 3
