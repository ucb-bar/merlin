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
    "merlin/tests",
    "merlin/python/merlin/xdsl_dialects",
    "merlin/python/merlin/targetgen",
    "merlin/python/merlin/kernels",
    "merlin/python/merlin/design_pressure",
    "merlin/python/merlin/dse",
    "merlin/python/merlin/runtime",
    "merlin/python/merlin/common",
    "merlin/python/merlin/validation",
    "merlin/runtime/c",
    "merlin/runtime/abi",
    "merlin/runtime/baremetal",
    "merlin/targets/toy_npu/docs",
    "merlin/targets/toy_npu/contracts",
    "merlin/targets/toy_npu/examples",
    "merlin/targets/toy_npu/generated",
    "merlin/schemas",
    "merlin/benchmarks/kernels",
    "merlin/benchmarks/semantic_memory",
    "merlin/benchmarks/design_pressure",
    "merlin/benchmarks/models",
    "merlin/experiments/targetgen_toy",
    "merlin/experiments/kernel_policy",
    "merlin/experiments/semantic_memory",
    "merlin/tests/data",
]

REQUIRED_SCHEMAS = [
    "target_contract", "dialect_plan", "kernel_record", "abstraction_candidate",
    "policy_rule", "workload_region", "design_pressure", "interface_candidate",
    "dse_result", "exploitability_report", "compilation_strategy", "search_space",
    # Runtime + TargetGen plans (Merlin-owned runtime model; targets adapt it).
    "runtime_adapter_plan", "zephyr_plan", "llvm_extension_plan", "evidence_report",
    "target_source_manifest", "command_buffer", "runtime_abi", "metrics", "trace",
    # Kernel-mining L6/L8 outputs (feed TargetGen's dialect_plan / llvm_extension_plan).
    "runtime_candidate", "dialect_requirement", "llvm_requirement",
]

REQUIRED_DOCS = [
    "architecture", "repo_structure", "contracts", "dialects", "targetgen",
    "kernel_mining", "design_pressure", "dse", "runtime", "integrations",
    "xdsl", "parallel_workstreams", "adding_a_target",
    "compilation_strategies", "search",
    # Core-dialect + runtime + generated-target documentation.
    "core_dialects", "zephyr", "llvm_integration", "generated_target_repos",
    "implementation_milestones",
]

REQUIRED_BENCHMARKS = [
    "repeated_rhs_matmul", "matmul_bias_requant_relu",
    "no_reuse_matmul", "capacity_stress_reuse",
]

# Directories whose contents are gitignored / exempt from the AGENT.md walk.
# runs/ and artifacts/ are the gitignored generated-output roots (see CLAUDE.md
# "Generated-output convention"); only their top-level AGENT.md is tracked.
# generated_targets/ is retired (folded into artifacts/targets/, no symlink). artifacts/ is a
# skip-root; its top-level AGENT.md is enough.
SKIP_DIRS = {".git", "build", "output", "runs", "artifacts", "results",
             "_qa_ws", "tmp", "__pycache__",
             ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def check_required_dirs(errors):
    for d in REQUIRED_DIRS:
        if not os.path.isdir(os.path.join(ROOT, d)):
            errors.append(f"missing required directory: {d}")


def check_agent_md(errors):
    # AGENT.md is required at MEANINGFUL levels only:
    #   - the repo root,
    #   - each top-level area  merlin/<area>,
    #   - each experiment      merlin/experiments/<exp>,
    #   - each test bucket      merlin/tests/<bucket>.
    # The importable package tree (merlin/python/merlin/<pkg>) is owned by gen_package_docs.py
    # (--check enforces AGENT.md coverage + freshness there). Everything deeper — experiment
    # scripts/inputs, test fixtures, frozen archives, capsule/schema data — is exempt.
    root_md = os.path.join(ROOT, "AGENT.md")
    if not os.path.isfile(root_md):
        errors.append("missing AGENT.md: (root)")

    def require(rel):
        if not os.path.isfile(os.path.join(ROOT, rel, "AGENT.md")):
            errors.append(f"missing AGENT.md: {rel}")

    merlin = os.path.join(ROOT, "merlin")
    for area in sorted(os.listdir(merlin)):
        ap = os.path.join(merlin, area)
        if not os.path.isdir(ap) or area in SKIP_DIRS or area == "python":
            continue
        require(f"merlin/{area}")
        if area in ("experiments", "tests"):
            for sub in sorted(os.listdir(ap)):
                sp = os.path.join(ap, sub)
                if os.path.isdir(sp) and sub not in SKIP_DIRS and sub not in ("fixtures", "data"):
                    require(f"merlin/{area}/{sub}")


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


TEST_BUCKETS = {"kernels", "rvv", "dse", "gemmini", "targetgen", "ir", "runtime", "infra"}


def check_test_layout(errors):
    """Every test lives in merlin/tests/<bucket>/test_*.py (bucket in the fixed set); none at root."""
    tdir = os.path.join(ROOT, "merlin", "tests")
    if not os.path.isdir(tdir):
        return
    for name in os.listdir(tdir):
        if name.startswith("test_") and name.endswith(".py"):
            errors.append(f"test at merlin/tests/ root (must be in a subsystem bucket): {name}")
    for b in sorted(os.listdir(tdir)):
        bp = os.path.join(tdir, b)
        if not os.path.isdir(bp) or b in {"fixtures", "data", "__pycache__"}:
            continue
        if b not in TEST_BUCKETS:
            has_tests = any(f.startswith("test_") and f.endswith(".py") for f in os.listdir(bp))
            if has_tests:
                errors.append(f"unknown test bucket merlin/tests/{b} (allowed: {sorted(TEST_BUCKETS)})")


def check_cli_docs(errors):
    """docs/cli.md must be in sync with pyproject [project.scripts] (single CLI source of truth)."""
    import subprocess
    gen = os.path.join(ROOT, "build_tools", "scripts", "gen_cli_docs.py")
    r = subprocess.run([sys.executable, gen, "--check"], capture_output=True, text=True)
    if r.returncode != 0:
        errors.append("docs/cli.md stale vs pyproject — run python build_tools/scripts/gen_cli_docs.py")


def check_package_docs(errors):
    """docs/module_index.md fresh + every package has a non-stale AGENT.md (living package docs)."""
    import subprocess
    gen = os.path.join(ROOT, "build_tools", "scripts", "gen_package_docs.py")
    r = subprocess.run([sys.executable, gen, "--check"], capture_output=True, text=True)
    if r.returncode != 0:
        for ln in (r.stderr or "").splitlines():
            if ln.strip().startswith("- "):
                errors.append(ln.strip()[2:])
        if not any("package" in e or "module_index" in e for e in errors):
            errors.append("package docs stale — run python build_tools/scripts/gen_package_docs.py")


def main():
    errors: list[str] = []
    checks = [
        ("required directories", check_required_dirs),
        ("AGENT.md coverage", check_agent_md),
        ("schemas", check_schemas),
        ("docs", check_docs),
        ("benchmarks", check_benchmarks),
        ("cli docs", check_cli_docs),
        ("package docs", check_package_docs),
        ("test layout", check_test_layout),
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
