"""Count every active pytest case, including parameterization, before execution.

Discover owned collectors from Git rather than a default testpaths selection.
An import error, collection-time skip, empty suite, or hidden selection is a
failure. Frozen third-party inputs remain inputs, not an active regression suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MAXIMUM = 150
ROOT = Path(__file__).resolve().parents[2]
# Immutable companion inputs recorded by each example's SOURCE.yaml. These are
# retained reproduction bytes, not maintained Merlin regression collectors.
ARCHIVED = {
    "examples/atlas/support/tests/test_program_emitter_migration.py": (  # target-ok: frozen input identity
        "dd01b6d6bfc48bc7d629c94da5e34807734043c985ed99a9785b5302dbe5eae3"
    ),
    "examples/atlas/support/tests/test_support_contract.py": (  # target-ok: frozen input identity
        "ae074ea9a8236ff46fc41070d9ec7417582cbc8839b7bd78ea20b003161cccfb"
    ),
    "examples/muon/support/tests/test_codegen_smoke.py": (  # target-ok: frozen input identity
        "890cabf6befa65d27778caa6f1e01a43badaed725d93d1cfc86eb84c1cfaf9cd"
    ),
    "examples/rvv/support/tests/test_support_contract.py": (  # target-ok: frozen input identity
        "769601d79d98aa18ea5c07ea553104c6c522946b7f5ef5f69d671e5a60a9227f"
    ),
    "examples/saturn/support/tests/test_matrix_lowering.py": (  # target-ok: frozen input identity
        "0859ef19143a19afcd89ac20a310d2dd548a152c405dfeb21db704d31abb95c7"
    ),
    "examples/saturn/support/tests/test_migration.py": (  # target-ok: frozen input identity
        "a9f11081c34572ed9d35497ead6f4a59736fe0a3fc7bc7b348b3b52f9b8078d2"
    ),
}


def collectors(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = []
    for name in sorted(set(result.stdout.decode("utf-8").split("\0")) - {""}):
        path = Path(name)
        if not (path.name.startswith("test_") or path.name.endswith("_test.py")):
            continue
        full = root / path
        if not full.is_file():
            continue  # Git still lists locally removed files before staging.
        if name in ARCHIVED:
            if hashlib.sha256(full.read_bytes()).hexdigest() != ARCHIVED[name]:
                raise ValueError(f"archived companion input changed: {name}")
            continue
        paths.append(path.as_posix())
    return paths


class Collection:
    def __init__(self):
        self.nodeids: list[str] = []
        self.errors: list[str] = []
        self.skipped: list[str] = []
        self.deselected: list[str] = []

    def pytest_collection_finish(self, session):
        self.nodeids = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors.append(str(report.longrepr))
        elif report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)


def validate(report: dict) -> list[str]:
    errors = []
    if report["exit_code"] != 0 or report["collection_errors"]:
        errors.append("collection must complete without errors")
    if report["collection_skips"] or report["deselected"]:
        errors.append("collection must include every case without skips or deselection")
    count = len(report["nodeids"])
    if not 0 < count <= MAXIMUM:
        errors.append(f"active suite has {count} cases; allowed range is 1..{MAXIMUM}")
    if count != len(set(report["nodeids"])):
        errors.append("duplicate collected case identities")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.environ.get("PYTEST_ADDOPTS"):
        parser.error("PYTEST_ADDOPTS must be unset for the complete collection budget")
    import pytest

    files = collectors(ROOT)
    if not files:
        parser.error("no active collectors discovered")
    plugin = Collection()
    status = pytest.main(["--collect-only", "-q", "-o", "addopts=", *files], plugins=[plugin])
    report = {
        "schema": "merlin.test-budget.v1",
        "maximum": MAXIMUM,
        "archived_inputs": ARCHIVED,
        "collectors": files,
        "exit_code": int(status),
        "nodeids": plugin.nodeids,
        "collection_errors": plugin.errors,
        "collection_skips": plugin.skipped,
        "deselected": plugin.deselected,
    }
    report["errors"] = validate(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Active pytest cases: {len(plugin.nodeids)}/{MAXIMUM}")
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
