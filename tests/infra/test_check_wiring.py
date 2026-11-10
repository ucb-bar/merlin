from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from merlin.common.paths import repo_root


def _gate(root: Path):
    spec = importlib.util.spec_from_file_location(
        "check_wiring_under_test", repo_root() / "build_tools/checks/check_wiring.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT, module.PACKAGE_ROOT = (root, root / "merlin" / "python")
    module.LEDGER = root / "build_tools" / "checks" / "unwired_ratchet.txt"
    module.SYMBOL_LEDGER = root / "build_tools" / "checks" / "unwired_symbols_ratchet.txt"
    return module


def _tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


_PERF = "merlin/python/merlin/perf"


def test_new_debt_fails_and_a_stale_entry_fails(tmp_path: Path, capsys) -> None:
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{_PERF}/__init__.py": "",
                f"{_PERF}/orphan.py": "X = 1\n",
                f"{_PERF}/wired.py": "X = 1\n",
                "build_tools/use.py": "from merlin.perf import wired\n",
            },
        )
    )
    assert gate.main([]) == 1 and "unwired" in capsys.readouterr().out
    gate.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    gate.LEDGER.write_text(f"{_PERF}/orphan.py\n{_PERF}/wired.py  # was debt once\n", encoding="utf-8")
    assert gate.main([]) == 1 and "stale ledger entry" in capsys.readouterr().out
    gate.LEDGER.write_text(f"# comment\n{_PERF}/orphan.py\n", encoding="utf-8")
    assert gate.main([]) == 0
