"""No library module may grow past the size limit unless it is recorded, shrink-only debt.

Measured 2026-09-16 on formatted code, 26 modules were over 1,500 lines, led by
targetgen/capsule_runner.py at 5,145. Past a size nobody can hold in their head, the second copy of a
helper gets written beside the first -- the duplication this repo spent a week removing.
"""

from __future__ import annotations

import importlib.util

from merlin.common.paths import repo_root

SCRIPT = repo_root() / "build_tools" / "scripts" / "check_structure.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_structure_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path, sizes: dict[str, int]):
    pkg = tmp_path / "merlin" / "python" / "merlin"
    for rel, n in sizes.items():
        path = pkg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n" * n, encoding="utf-8")
    return tmp_path


def _run(mod, root, ratchet_lines=()):
    mod.ROOT = str(root)
    ratchet = root / "module_size_ratchet.txt"
    ratchet.write_text("# debt\n" + "".join(f"{r}\n" for r in ratchet_lines), encoding="utf-8")
    mod.MODULE_SIZE_RATCHET = str(ratchet)
    errors: list[str] = []
    mod.check_module_size(errors)
    return errors


def test_a_new_module_over_the_limit_fails(tmp_path):
    mod = _module()
    root = _tree(tmp_path, {"small.py": 10, "huge.py": mod.MODULE_SIZE_LIMIT + 1})
    errors = _run(mod, root)
    assert len(errors) == 1 and "merlin/python/merlin/huge.py" in errors[0]


def test_a_module_at_the_limit_passes(tmp_path):
    mod = _module()
    root = _tree(tmp_path, {"exact.py": mod.MODULE_SIZE_LIMIT})
    assert _run(mod, root) == []


def test_recorded_debt_is_excused_and_only_that(tmp_path):
    mod = _module()
    big = mod.MODULE_SIZE_LIMIT + 50
    root = _tree(tmp_path, {"old.py": big, "new.py": big})
    errors = _run(mod, root, ["merlin/python/merlin/old.py  # recorded"])
    assert len(errors) == 1 and "new.py" in errors[0]


def test_the_build_copy_under_data_is_not_counted(tmp_path):
    """`_data/` is a build-generated copy; counting it would report the same module twice."""
    mod = _module()
    root = _tree(tmp_path, {"_data/copy.py": mod.MODULE_SIZE_LIMIT + 1})
    assert _run(mod, root) == []


def test_every_ledger_entry_still_exists():
    """A ledger naming a module that was deleted or split away can hide a regression under its name."""
    ledger = repo_root() / "build_tools" / "scripts" / "module_size_ratchet.txt"
    entries = [ln.split("#", 1)[0].strip() for ln in ledger.read_text().splitlines()]
    missing = [e for e in entries if e and not (repo_root() / e).is_file()]
    assert not missing, missing
