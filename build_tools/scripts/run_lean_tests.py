"""Run the retained compiler verification regressions with a 200-case ceiling."""

from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import repo_root


class CaseBudget:
    def pytest_collection_modifyitems(self, items):
        if not items or len(items) > 200:
            raise pytest.UsageError(f"Lean suite requires 1 through 200 cases; collected {len(items)}")


def main():
    root = repo_root()
    sys.path.insert(0, str(root))
    suites = json.loads((root / "merlin/tests/data/lean_suite.json").read_text())
    return pytest.main([*[str(root / path) for path in suites], *sys.argv[1:]], plugins=[CaseBudget()])


if __name__ == "__main__":
    raise SystemExit(main())
