#!/usr/bin/env python3
"""Validate the merlin repository scaffold.

Checks:
  1. Required directories exist.
  2. Every tracked directory contains an AGENT.md (tmp/ is exempt; it uses AGENTS.md).
  3. Required schema files exist and are non-empty.
  4. Required docs exist.
  5. Required semantic-memory benchmark YAML files exist.

Pure stdlib. Exits non-zero on any failure.

Usage:
    python build_tools/scripts/check_structure.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_DIRS = [
    "build_tools/scripts",
    "docs",
    "third_party",
    "tools",
    "merlin/compiler/include/merlin/Dialect/Contract",
    "merlin/compiler/include/merlin/Dialect/Schedule",
    "merlin/compiler/include/merlin/Dialect/Interface",
    "merlin/compiler/include/merlin/Dialect/Runtime",
    "merlin/compiler/lib/Dialect",
    "merlin/compiler/tools/merlin-opt",
    "merlin/compiler/tools/merlin-translate",
    "merlin/compiler/tests/lit",
    "merlin/compiler/tests/unit",
    "merlin/python/merlin/contracts",
    "merlin/python/merlin/xdsl_dialects",
    "merlin/python/merlin/targetgen",
    "merlin/python/merlin/kernels",
    "merlin/python/merlin/design_pressure",
    "merlin/python/merlin/dse",
    "merlin/python/merlin/runtime",
    "merlin/python/merlin/common",
    "merlin/runtime/common",
    "merlin/runtime/command_buffer",
    "merlin/runtime/simulator",
    "merlin/runtime/baremetal",
    "merlin/runtime/zephyr",
    "merlin/integrations/xnnpack",
    "merlin/integrations/autocomp",
    "merlin/integrations/exo",
    "merlin/integrations/triton",
    "merlin/integrations/xdsl",
    "merlin/integrations/iree",
    "merlin/integrations/cuda_tile",
    "merlin/integrations/hexagon_mlir",
    "merlin/targets/toy_npu/docs",
    "merlin/targets/toy_npu/contracts",
    "merlin/targets/toy_npu/examples",
    "merlin/targets/toy_npu/generated",
    "merlin/targets/toy_npu/tests",
    "merlin/targets/example_vector",
    "merlin/schemas",
    "merlin/benchmarks/kernels",
    "merlin/benchmarks/semantic_memory",
    "merlin/benchmarks/design_pressure",
    "merlin/benchmarks/models",
    "merlin/experiments/targetgen_toy",
    "merlin/experiments/kernel_policy",
    "merlin/experiments/semantic_memory",
    "merlin/experiments/interface_dse",
    "merlin/tests/integration",
    "merlin/tests/conformance",
    "merlin/tests/golden",
    "merlin/tests/data",
]

REQUIRED_SCHEMAS = [
    "target_contract", "dialect_plan", "kernel_record", "abstraction_candidate",
    "policy_rule", "workload_region", "design_pressure", "interface_candidate",
    "dse_result", "exploitability_report",
]

REQUIRED_DOCS = [
    "architecture", "repo_structure", "contracts", "dialects", "targetgen",
    "kernel_mining", "design_pressure", "dse", "runtime", "integrations",
    "xdsl", "parallel_workstreams", "adding_a_target",
]

REQUIRED_BENCHMARKS = [
    "repeated_rhs_matmul", "matmul_bias_requant_relu",
    "no_reuse_matmul", "capacity_stress_reuse",
]

# Directories whose contents are gitignored / exempt from the AGENT.md walk.
SKIP_DIRS = {".git", "build", "output", "tmp", "__pycache__",
             ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def check_required_dirs(errors):
    for d in REQUIRED_DIRS:
        if not os.path.isdir(os.path.join(ROOT, d)):
            errors.append(f"missing required directory: {d}")


def check_agent_md(errors):
    for dirpath, dirnames, _ in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and not d.endswith(".egg-info")
                       and not (d.startswith(".") and d != ".")]
        rel = os.path.relpath(dirpath, ROOT)
        if rel == "." or rel.split(os.sep)[0] in SKIP_DIRS:
            top = os.path.join(ROOT, "AGENT.md")
            if rel == "." and not os.path.isfile(top):
                errors.append("missing AGENT.md: (root)")
            continue
        if not os.path.isfile(os.path.join(dirpath, "AGENT.md")):
            errors.append(f"missing AGENT.md: {rel}")


def check_schemas(errors):
    for s in REQUIRED_SCHEMAS:
        p = os.path.join(ROOT, "merlin", "schemas", f"{s}.schema.yaml")
        if not os.path.isfile(p):
            errors.append(f"missing schema: merlin/schemas/{s}.schema.yaml")
        elif os.path.getsize(p) == 0:
            errors.append(f"empty schema: merlin/schemas/{s}.schema.yaml")


def check_docs(errors):
    for d in REQUIRED_DOCS:
        p = os.path.join(ROOT, "docs", f"{d}.md")
        if not os.path.isfile(p):
            errors.append(f"missing doc: docs/{d}.md")


def check_benchmarks(errors):
    for b in REQUIRED_BENCHMARKS:
        p = os.path.join(ROOT, "merlin", "benchmarks", "semantic_memory", f"{b}.yaml")
        if not os.path.isfile(p):
            errors.append(f"missing benchmark: merlin/benchmarks/semantic_memory/{b}.yaml")


def main():
    errors: list[str] = []
    checks = [
        ("required directories", check_required_dirs),
        ("AGENT.md coverage", check_agent_md),
        ("schemas", check_schemas),
        ("docs", check_docs),
        ("benchmarks", check_benchmarks),
    ]
    for label, fn in checks:
        before = len(errors)
        fn(errors)
        status = "FAIL" if len(errors) > before else "ok"
        print(f"[{status:>4}] {label}")

    if errors:
        print(f"\n{len(errors)} problem(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll structure checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
