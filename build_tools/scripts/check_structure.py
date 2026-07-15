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
    # Canonical per-target shape: contracts/ (the target definition) + generated/ (its output dir)
    # are REQUIRED for every target; docs/ and examples/ are present when there's content (no empty
    # stubs — see the WS4 de-pin cleanup). toy_npu is the fully-populated reference instance.
    "merlin/targets/toy_npu/contracts",
    "merlin/targets/toy_npu/generated",
    "merlin/targets/toy_npu/docs",
    "merlin/targets/toy_npu/examples",
    "merlin/targets/gemmini/contracts",
    "merlin/targets/gemmini/generated",
    "merlin/targets/saturn/contracts",
    "merlin/targets/saturn/generated",
    "merlin/schemas",
    "merlin/benchmarks/semantic_memory",
    "merlin/experiments/kernel_policy",
    "merlin/tests/data",
]

REQUIRED_SCHEMAS = [
    "target_contract", "dialect_plan", "kernel_record", "abstraction_candidate",
    "policy_rule", "workload_region", "design_pressure", "interface_candidate",
    "dse_result", "exploitability_report", "compilation_strategy", "search_space",
    # Runtime + TargetGen plans (Merlin-owned runtime model; targets adapt it).
    "runtime_adapter_plan", "zephyr_plan", "llvm_extension_plan", "evidence_report",
    "target_source_manifest", "command_buffer", "metrics", "trace",
    # Kernel-mining L6/L8 outputs (feed TargetGen's dialect_plan / llvm_extension_plan).
    "runtime_candidate", "dialect_requirement", "llvm_requirement",
    # DSE-guidance + rvvgen subsystem schemas (were used but unlisted).
    "baseline_cost", "cpu_coupling", "dse_axis_triage", "temporal_workload_metadata",
    "rvv_package_manifest", "rvv_result",
    # Quantization-format registry entry schema (merlin.common.quant_formats).
    "quant_format",
]

REQUIRED_DOCS = [
    # reference/ — durable, code-derived facts
    "reference/architecture", "reference/repo_structure", "reference/contracts",
    "reference/dialects", "reference/core_dialects", "reference/runtime", "reference/xdsl",
    "reference/generated_target_repos",
    # guides/ — task-oriented how-tos
    "guides/getting_started",
    "guides/targetgen", "guides/kernel_mining", "guides/design_pressure", "guides/dse",
    "guides/integrations", "guides/adding_a_target", "guides/compilation_strategies",
    "guides/search", "guides/zephyr", "guides/llvm_integration",
    # design/ — rationale
    "design/parallel_workstreams",
]

REQUIRED_BENCHMARKS = [
    "repeated_rhs_matmul", "matmul_bias_requant_relu",
    "no_reuse_matmul", "capacity_stress_reuse",
]

# Directories whose contents are gitignored / exempt from the AGENT.md walk.
# out/ is the single gitignored generated-output root (out/{runs,artifacts,build}; see CLAUDE.md
# "Generated-output convention"); only its top-level AGENT.md skeletons are tracked.
# generated_targets/ is retired (folded into out/artifacts/targets/, no symlink).
SKIP_DIRS = {".git", "out", "build", "output", "runs", "artifacts", "results",
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


def check_doc_paths(errors):
    """No doc/AGENT.md references a RETIRED repo path (deny-list; see check_doc_paths.py)."""
    import subprocess
    chk = os.path.join(ROOT, "build_tools", "scripts", "check_doc_paths.py")
    r = subprocess.run([sys.executable, chk, "--check"], capture_output=True, text=True)
    if r.returncode != 0:
        for ln in (r.stderr or "").splitlines():
            if ln.strip().startswith("- "):
                errors.append(ln.strip()[2:])
        if not any("retired" in e or "deprecated output" in e for e in errors):
            errors.append("docs reference retired paths — run python build_tools/scripts/check_doc_paths.py")


# Misleading "scaffold-era" phrasing that must never reappear in the top-level entry docs
# (the repo has working end-to-end pipelines; see Phase-0 of the docs restructure).
ROOT_STALE_PHRASES = ("placeholder modules", "not working compiler",
                      "do not implement major algorithms", "currently a scaffold",
                      "status: **scaffold**")


def check_root_docs(errors):
    """Root README.md/AGENT.md must not describe the repo as an empty scaffold."""
    for name in ("README.md", "AGENT.md"):
        p = os.path.join(ROOT, name)
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as fh:
            low = fh.read().lower()
        for ph in ROOT_STALE_PHRASES:
            if ph in low:
                errors.append(f"{name}: stale scaffold-era phrase {ph!r} (the repo is active)")


def check_docs_freshness(errors):
    """docs/ front-matter is schema-valid (drift is a soft signal; see check_docs_freshness.py)."""
    import subprocess
    chk = os.path.join(ROOT, "build_tools", "scripts", "check_docs_freshness.py")
    r = subprocess.run([sys.executable, chk, "--check"], capture_output=True, text=True)
    if r.returncode != 0:
        for ln in (r.stderr or "").splitlines():
            if ln.strip().startswith("- "):
                errors.append(ln.strip()[2:])
        if not any("front-matter" in e for e in errors):
            errors.append("docs front-matter invalid — run python build_tools/scripts/check_docs_freshness.py")


def check_schema_docs(errors):
    """docs/reference/schemas.md is in sync with merlin/schemas/ (see gen_schema_docs.py)."""
    import subprocess
    gen = os.path.join(ROOT, "build_tools", "scripts", "gen_schema_docs.py")
    r = subprocess.run([sys.executable, gen, "--check"], capture_output=True, text=True)
    if r.returncode != 0:
        errors.append("docs/reference/schemas.md stale — run python build_tools/scripts/gen_schema_docs.py")


def check_docs_index(errors):
    """docs/README.md hub is in sync with doc front-matter (see gen_docs_index.py)."""
    import subprocess
    gen = os.path.join(ROOT, "build_tools", "scripts", "gen_docs_index.py")
    r = subprocess.run([sys.executable, gen, "--check"], capture_output=True, text=True)
    if r.returncode != 0:
        errors.append("docs/README.md hub stale — run python build_tools/scripts/gen_docs_index.py")


# The library (merlin/python/merlin) reads INPUTS from benchmarks/ and contract/, never from
# experiments/ (experiments consume the library, one-way). The ONE sanctioned indirection to a corpus
# still under experiments/ is the corpus locator; everything else must not name experiments/ as a path.
_BOUNDARY_ALLOW = {os.path.join("merlin", "python", "merlin", "targetgen", "corpora.py")}


def check_schema_usage(errors):
    """Every merlin/schemas/*.schema.yaml must be referenced by name in merlin/python/ — either
    validated (validate/validate_or_raise/_SCHEMA/PLAN_SCHEMAS) or mirrored as a vocabulary spec
    (a code constant / generated view / docstring that names it). Zero references ⇒ dead schema."""
    import glob
    lib = os.path.join(ROOT, "merlin", "python")
    corpus = []
    for dp, _d, files in os.walk(lib):
        for fn in files:
            if fn.endswith(".py"):
                corpus.append(open(os.path.join(dp, fn), encoding="utf-8").read())
    blob = "\n".join(corpus)
    for path in sorted(glob.glob(os.path.join(ROOT, "merlin", "schemas", "*.schema.yaml"))):
        name = os.path.basename(path)[: -len(".schema.yaml")]
        if f'"{name}"' not in blob and f"'{name}'" not in blob and f"{name}.schema" not in blob:
            errors.append(f"dead schema (no reference in merlin/python): merlin/schemas/{name}.schema.yaml")


def check_library_boundary(errors):
    """No library module may reference ``experiments/`` as a path component (consumption-direction:
    benchmarks/ = library reads it; experiments/ = only consumes the library). See targetgen/corpora.py."""
    lib = os.path.join(ROOT, "merlin", "python", "merlin")
    for dirpath, _dirs, files in os.walk(lib):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            if rel in _BOUNDARY_ALLOW:
                continue
            for i, ln in enumerate(open(full, encoding="utf-8"), 1):
                code = ln.split("#", 1)[0]
                if '"experiments"' in code or "'experiments'" in code or "experiments/" in code:
                    errors.append(f"library reads experiments/ (use merlin.common.corpora): {rel}:{i}")


def main():
    errors: list[str] = []
    checks = [
        ("required directories", check_required_dirs),
        ("AGENT.md coverage", check_agent_md),
        ("schema usage", check_schema_usage),
        ("library boundary", check_library_boundary),
        ("root docs", check_root_docs),
        ("schemas", check_schemas),
        ("docs", check_docs),
        ("benchmarks", check_benchmarks),
        ("cli docs", check_cli_docs),
        ("package docs", check_package_docs),
        ("schema docs", check_schema_docs),
        ("docs index", check_docs_index),
        ("docs freshness", check_docs_freshness),
        ("doc paths", check_doc_paths),
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
