#!/usr/bin/env python3
"""Validate a generated Merlin target repository.

Checks that a ``merlin-target-<name>/`` repo produced by ``merlin.targetgen`` has:
  - the five contract plans (target_contract, dialect_plan, runtime_adapter_plan,
    zephyr_plan, llvm_extension_plan), each schema-valid,
  - docs/evidence_report.md,
  - the per-layer directories xdsl/ runtime/ zephyr/ llvm/ tests/,
  - AGENT.md coverage in every directory.

Delegates to ``merlin.validation.generated_target.check_generated_target``.

Usage:
    python build_tools/scripts/check_generated_target.py <path-to-generated-repo>
"""
from __future__ import annotations

import os
import sys

# Make the in-tree package importable without installation.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "merlin", "python"))

from merlin.validation.generated_target import check_generated_target  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_generated_target.py <path-to-generated-repo>")
        return 2
    target = argv[0]
    problems = check_generated_target(target)
    if problems:
        print(f"[FAIL] {target}")
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"[ ok ] {target}")
    print("\nGenerated target repo checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
