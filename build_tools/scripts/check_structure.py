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

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _source_layout  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _discovered_targets():
    """Curated reference target names — the dirs under ``merlin/targets/`` carrying a
    ``contracts/target_contract.yaml``. This covers only the retained native shape;
    example-owned inputs have no required in-tree generated directory. Discovery
    stays import-free and runs with no deps installed (as in the docs CI job). Targets are
    DISCOVERED, never named here, so registering a new target extends the gate with zero edits.
    """
    tdir = os.path.join(ROOT, "merlin", "targets")
    if not os.path.isdir(tdir):
        return []
    return sorted(
        name
        for name in os.listdir(tdir)
        if os.path.isfile(os.path.join(tdir, name, "contracts", "target_contract.yaml"))
    )


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
    "merlin/schemas",
    "merlin/benchmarks/semantic_memory",
    "merlin/experiments/kernel_policy",
    "merlin/tests/data",
]

# Canonical per-target shape: contracts/ (the target definition) + generated/ (its output dir) are
# REQUIRED for every DISCOVERED target; docs/ and examples/ are present only when there's content (no
# empty stubs — see the WS4 de-pin cleanup). The set is derived from what is registered, so no target
# name is hardcoded here.
for _t in _discovered_targets():
    REQUIRED_DIRS.append(f"merlin/targets/{_t}/contracts")
    REQUIRED_DIRS.append(f"merlin/targets/{_t}/generated")

REQUIRED_SCHEMAS = [
    "target_contract",
    "dialect_plan",
    "kernel_record",
    "abstraction_candidate",
    "policy_rule",
    "workload_region",
    "design_pressure",
    "interface_candidate",
    "dse_result",
    "exploitability_report",
    "compilation_strategy",
    "search_space",
    # Runtime + TargetGen plans (Merlin-owned runtime model; targets adapt it).
    "runtime_adapter_plan",
    "zephyr_plan",
    "llvm_extension_plan",
    "evidence_report",
    "target_source_manifest",
    "command_buffer",
    "metrics",
    "trace",
    # Kernel-mining L6/L8 outputs (feed TargetGen's dialect_plan / llvm_extension_plan).
    "runtime_candidate",
    "dialect_requirement",
    "llvm_requirement",
    # DSE-guidance + rvvgen subsystem schemas (were used but unlisted).
    "baseline_cost",
    "cpu_coupling",
    "dse_axis_triage",
    "temporal_workload_metadata",
    "rvv_package_manifest",
    "rvv_result",
    # Frozen-compiler paper methodology (study input + one matrix-cell result).
    "paper_study",
    "paper_run_result",
    "session_contract",
    "compiler_freeze",
    "cpu_host_experiment",
    "deployment_profile",
    # Quantization-format registry entry schema (merlin.common.quant_formats).
    "quant_format",
]

REQUIRED_DOCS = [
    # reference/ — durable, code-derived facts
    "reference/architecture",
    "reference/repo_structure",
    "reference/contracts",
    "reference/dialects",
    "reference/core_dialects",
    "reference/runtime",
    "reference/xdsl",
    "reference/generated_target_repos",
    # guides/ — task-oriented how-tos
    "guides/getting_started",
    "guides/targetgen",
    "guides/kernel_mining",
    "guides/design_pressure",
    "guides/dse",
    "guides/integrations",
    "guides/adding_a_target",
    "guides/compilation_strategies",
    "guides/search",
    "guides/zephyr",
    "guides/llvm_integration",
    # design/ — rationale
    "design/parallel_workstreams",
]

REQUIRED_BENCHMARKS = [
    "repeated_rhs_matmul",
    "matmul_bias_requant_relu",
    "no_reuse_matmul",
    "capacity_stress_reuse",
]

# Directories whose contents are gitignored / exempt from the AGENT.md walk.
# out/ is the single gitignored generated-output root (out/{runs,artifacts,build}; see CLAUDE.md
# "Generated-output convention"); only its top-level AGENT.md skeletons are tracked.
# generated_targets/ is retired (folded into out/artifacts/targets/, no symlink).
SKIP_DIRS = {
    ".git",
    "out",
    "build",
    "output",
    "runs",
    "artifacts",
    "results",
    "_qa_ws",
    "tmp",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}


def check_required_dirs(errors):
    for d in REQUIRED_DIRS:
        actual = d
        if d.startswith("merlin/python/merlin/"):
            tail = d.removeprefix("merlin/python/merlin/")
            locations = [
                package / tail for package in _source_layout.source_packages(Path(ROOT)) if package.name == "merlin"
            ]
            actual = str(
                next((path for path in locations if path.is_dir()), _source_layout.core_package(Path(ROOT)) / tail)
            )
        if not os.path.isdir(os.path.join(ROOT, actual)):
            errors.append(f"missing required directory: {actual}")


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


EXPERIMENT_STATUSES = ("active", "frozen", "reference")


def check_experiment_status(errors):
    """Every experiment says where it stands, so "dormant" is a stated fact rather than a guess.

    Measured 2026-09-14: an audit classed experiments as abandoned from commit age alone; two were cited
    by live guides and one was under development on another branch. A ``Status:`` line in the first 15
    lines of the experiment's AGENT.md says which: ``active``, ``frozen`` (finished; results in its
    FINDINGS.md, which must exist) or ``reference`` (a scaffold others copy, with no runs of its own).
    Retiring an experiment means deleting it -- git history is the archive -- so no status names that.
    """
    exp = os.path.join(ROOT, "merlin", "experiments")
    if not os.path.isdir(exp):
        return
    for sub in sorted(os.listdir(exp)):
        sp = os.path.join(exp, sub)
        md = os.path.join(sp, "AGENT.md")
        if not os.path.isdir(sp) or sub in SKIP_DIRS or not os.path.isfile(md):
            continue  # a missing AGENT.md is check_agent_md's finding, not this one
        with open(md, encoding="utf-8") as fh:
            head = [next(fh, "") for _ in range(15)]
        declared = [ln[len("Status:") :].strip() for ln in head if ln.startswith("Status:")]
        if not declared or not declared[0]:
            errors.append(f"experiment declares no `Status:` in its AGENT.md: merlin/experiments/{sub}")
            continue
        word = declared[0].split()[0]
        if word not in EXPERIMENT_STATUSES:
            errors.append(
                f"experiment status {word!r} is not one of {list(EXPERIMENT_STATUSES)}: merlin/experiments/{sub}"
            )
        elif word == "frozen" and not os.path.isfile(os.path.join(sp, "FINDINGS.md")):
            errors.append(f"frozen experiment has no FINDINGS.md: merlin/experiments/{sub}")


TARGET_MARKER_RATCHET = os.path.join(ROOT, "build_tools", "scripts", "test_target_marker_ratchet.txt")
MODULE_SIZE_RATCHET = os.path.join(ROOT, "build_tools", "scripts", "module_size_ratchet.txt")
#: Lines a library module may reach before it has to be split. Measured 2026-09-16 on formatted code:
#: 26 modules were past it, led by targetgen/capsule_runner.py at 5,145 -- a size at which nobody holds
#: the whole module in their head, which is how a second copy of a helper gets written beside the first.
MODULE_SIZE_LIMIT = 1500
TARGET_HEAVY_LITERALS = 5


def _quoted_target_literals(text, names):
    return sum(text.count(f'"{n}"') + text.count(f"'{n}'") for n in names)


def check_test_target_marker(errors):
    """A test in a SUBSYSTEM bucket whose subject is one target must say so with ``pytest.mark.target``.

    Buckets name subsystems, not targets; measured 2026-09-14, 120 test files in the generic buckets name
    a target five or more times and none said so, so "the infra suite passes" silently meant "passes on
    the targets these files hardcode". The marker makes that selectable (``-m 'not target'``). Existing
    files are recorded in test_target_marker_ratchet.txt, which may only shrink; a NEW target-heavy test
    must carry the marker. Target names come from the derived roster (build_tools/scripts/_target_roster).
    """
    sys.path.insert(0, os.path.join(ROOT, "build_tools", "scripts"))
    try:
        import _target_roster

        names = sorted(_target_roster.target_names(ROOT))
    except Exception as exc:  # noqa: BLE001 -- no roster, nothing to measure against: say so
        errors.append(f"test target marker: target roster unreadable ({exc})")
        return
    ratchet = set()
    if os.path.isfile(TARGET_MARKER_RATCHET):
        with open(TARGET_MARKER_RATCHET, encoding="utf-8") as fh:
            ratchet = {ln.split("#", 1)[0].strip() for ln in fh if ln.split("#", 1)[0].strip()}
    tests = os.path.join(ROOT, "merlin", "tests")
    if not os.path.isdir(tests):
        return
    for bucket in sorted(os.listdir(tests)):
        bp = os.path.join(tests, bucket)
        # A bucket NAMED after a target (by the same derived roster) is about that target by construction.
        if not os.path.isdir(bp) or bucket in names or bucket in SKIP_DIRS or bucket in ("fixtures", "data"):
            continue
        for fn in sorted(os.listdir(bp)):
            if not (fn.startswith("test_") and fn.endswith(".py")):
                continue
            rel = f"merlin/tests/{bucket}/{fn}"
            with open(os.path.join(bp, fn), encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if _quoted_target_literals(text, names) < TARGET_HEAVY_LITERALS or "mark.target(" in text or rel in ratchet:
                continue
            errors.append(f"target-heavy test without `pytestmark = pytest.mark.target(...)`: {rel}")


def check_module_size(errors):
    """No library module may grow past MODULE_SIZE_LIMIT lines unless it is recorded debt.

    Existing offenders are listed in module_size_ratchet.txt, which may only shrink; a module leaves it
    by being split. A NEW module over the limit fails here. Counts physical lines of the file as
    committed-shape source, i.e. after ruff format, so a formatter pass cannot push a module over.
    """
    ratchet = set()
    if os.path.isfile(MODULE_SIZE_RATCHET):
        with open(MODULE_SIZE_RATCHET, encoding="utf-8") as fh:
            ratchet = {ln.split("#", 1)[0].strip() for ln in fh if ln.split("#", 1)[0].strip()}
    for relative in _source_layout.python_files(Path(ROOT), _source_layout.SOURCE_SCAN_ROOTS):
        rel = relative.as_posix()
        with open(Path(ROOT) / relative, encoding="utf-8", errors="replace") as fh:
            n = sum(1 for _ in fh)
        if n > MODULE_SIZE_LIMIT and rel not in ratchet and _source_layout.policy_path(rel) not in ratchet:
            errors.append(f"module over {MODULE_SIZE_LIMIT} lines ({n}); split it: {rel}")


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


# Subsystem test buckets. The generic (cross-target) buckets are fixed; a target may additionally own
# a bucket named after itself (e.g. the gemmini/ RoCC tests), so those are DISCOVERED from the target
# registry rather than named here — registering a target that ships its own tests needs no edit.
def _example_test_buckets():
    """Authored target examples may own tests without an in-tree target implementation."""
    examples = Path(ROOT) / "examples"
    if not examples.is_dir():
        return set()
    return {
        path.name
        for path in examples.iterdir()
        if (path / "target/descriptor.yaml").is_file() or (path / "target/contracts/target_contract.yaml").is_file()
    }


_GENERIC_TEST_BUCKETS = {"kernels", "rvv", "dse", "targetgen", "ir", "runtime", "infra"}
TEST_BUCKETS = _GENERIC_TEST_BUCKETS | set(_discovered_targets()) | _example_test_buckets()


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

    # A basename may appear once across ALL buckets. The buckets carry no __init__.py, so pytest imports
    # each test file under its bare basename; two files sharing one cannot both be imported and pytest
    # aborts COLLECTION for the entire run. That failure is disproportionate and easy to misread -- it
    # reports as one broken file while actually preventing every test in the suite from running, and it
    # only appears when the two buckets are collected together, so a per-bucket run looks clean.
    seen: dict[str, str] = {}
    for b in sorted(os.listdir(tdir)):
        bp = os.path.join(tdir, b)
        if not os.path.isdir(bp) or b in {"fixtures", "data", "__pycache__"}:
            continue
        for f in sorted(os.listdir(bp)):
            if not (f.startswith("test_") and f.endswith(".py")):
                continue
            if f in seen:
                errors.append(
                    f"duplicate test module basename {f!r}: merlin/tests/{seen[f]}/{f} and "
                    f"merlin/tests/{b}/{f} — pytest imports test files by bare basename, so this "
                    f"aborts collection for the WHOLE suite. Rename one after what it actually covers."
                )
            else:
                seen[f] = b


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
ROOT_STALE_PHRASES = (
    "placeholder modules",
    "not working compiler",
    "do not implement major algorithms",
    "currently a scaffold",
    "status: **scaffold**",
)


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
BOUNDARY_RATCHET = os.path.join(ROOT, "build_tools", "scripts", "library_boundary_ratchet.txt")
# The module a violating caller is supposed to go through is NOT spelled here: `check_library_boundary`
# derives it from `_BOUNDARY_ALLOW` above, so the remedy the message names and the module the gate
# actually exempts cannot drift apart. The message used to name `merlin.common.corpora`, which has
# never existed, and a second literal is exactly how that happens again.

# References to experiments/ that ARE a path but cannot go through the corpus locator, keyed by
# ``(repo-relative file, the exact string literal)`` so an entry survives the line moving while a
# DIFFERENT literal in the same file is still a violation. Ratchet: this may shrink, never grow.
_BOUNDARY_LITERAL_ALLOW = {
    (
        "merlin/python/merlin/compare/host_experiment.py",
        "merlin/experiments/cpu_host_compiler_v0/grader.py",
    ): "loads that experiment's source-sealed grader to reuse its gate functions verbatim (not a corpus)",
    (
        "merlin/python/merlin/compare/host_experiment.py",
        "merlin/experiments/cpu_host_compiler_v0/optimization_space_v1.yaml",
    ): "binds the calibration to that experiment's frozen optimization space, by content",
    (
        "merlin/python/merlin/targetgen/generate_bundles.py",
        "experiments/",
    ): "guest-visible sandbox mount spec: a path inside the agent's bundle, never read by this process",
}


def check_schema_usage(errors):
    """Every merlin/schemas/*.schema.yaml must be referenced by name in merlin/python/ — either
    validated (validate/validate_or_raise/_SCHEMA/PLAN_SCHEMAS) or mirrored as a vocabulary spec
    (a code constant / generated view / docstring that names it). Zero references ⇒ dead schema."""
    import glob

    corpus = []
    for path in _source_layout.python_files(Path(ROOT), _source_layout.SOURCE_SCAN_ROOTS):
        corpus.append((Path(ROOT) / path).read_text(encoding="utf-8"))
    blob = "\n".join(corpus)
    for path in sorted(glob.glob(os.path.join(ROOT, "merlin", "schemas", "*.schema.yaml"))):
        name = os.path.basename(path)[: -len(".schema.yaml")]
        if f'"{name}"' not in blob and f"'{name}'" not in blob and f"{name}.schema" not in blob:
            errors.append(f"dead schema (no reference in merlin/python): merlin/schemas/{name}.schema.yaml")


def _prose_lines(src):
    """Line numbers occupied by DOCSTRINGS, which are prose and not a path reference.

    A string that is a bare expression statement is documentation; every other string literal is code
    (``Path("experiments")`` is a real violation and must stay one). Returns an empty set when the file
    does not parse, so an unparsable file is scanned exactly as before rather than silently exempted."""
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            lines.update(range(node.value.lineno, (node.value.end_lineno or node.value.lineno) + 1))
    return lines


# A string naming ``experiments`` as a path component is a read of that layout UNLESS the code uses it
# as something other than a path. These are the non-path uses, recognised by the SHAPE of the call, not
# by matching the word: a string predicate tests a declared-path prefix, argparse names a subcommand.
_NON_PATH_CALLS = {
    # str predicates: the literal is a pattern being tested, not a path being built
    "startswith",
    "endswith",
    "count",
    "find",
    "rfind",
    "index",
    "split",
    "rsplit",
    "partition",
    "rpartition",
    "removeprefix",
    "removesuffix",
    "strip",
    "lstrip",
    "rstrip",
    # argparse: the literal is a subcommand NAME on the CLI, not a directory
    "add_parser",
}


def _names_experiments(value):
    """True when ``value`` names ``experiments`` as a whole path COMPONENT.

    Two structural tests, no pattern matching. A path literal carries no whitespace, so anything with
    a space in it is prose (a refusal message, an error string, a sentence about the layout) and is
    not a path this rule is about -- the AST exemptions below catch docstrings, but a refusal message
    explaining WHY the library may not read a corpus is an ordinary assignment and must not be the
    violation it describes. What remains is split on the separator instead of searched for the word,
    so ``out/experiments_summary`` (a different directory) is not a read of ``experiments/``."""
    if value.split() != [value]:
        return False
    return "experiments" in value.replace("\\", "/").split("/")


def _experiments_refs(src):
    """``[(lineno, literal)]`` for every string in ``src`` that names ``experiments/`` AS A PATH.

    Structural, over the AST. Every string literal naming the directory counts; the exemptions are the
    uses that provably do not build a path: a bare string statement (a docstring), an operand of a
    comparison (``if anc.name == "experiments"``), an argument to a string predicate
    (``p.startswith(("merlin/", "experiments/"))``), and an argparse subcommand name
    (``sub.add_parser("experiments")``). Everything else is reported — ``root / "experiments"``,
    ``os.path.join(..., "experiments")``, a literal handed to ``open``, an f-string prefix, or one just
    stored in a list for later — so a violation spelled a new way is still caught.

    ``None`` when the file does not parse, so the caller falls back to the textual scan rather than
    silently exempting an unparsable file."""
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    refs = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if not _names_experiments(node.value):
            continue
        cur, up = node, parent.get(node)
        # An f-string chunk, or a member of a literal choice set, is used wherever its container is.
        while isinstance(up, (ast.JoinedStr, ast.Tuple, ast.List, ast.Set)):
            cur, up = up, parent.get(up)
        if isinstance(up, ast.Expr):
            continue  # a bare string statement is a docstring: prose about the layout, not a read
        if isinstance(up, ast.Compare):
            continue  # compared against a directory NAME, never joined onto a path
        if isinstance(up, ast.Call) and any(a is cur for a in up.args):
            fn = up.func
            if (fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")) in _NON_PATH_CALLS:
                continue
        refs.append((node.lineno, node.value))
    return refs


def check_library_boundary(errors):
    """No library module may reference ``experiments/`` as a path component (consumption-direction:
    benchmarks/ = library reads it; experiments/ = only consumes the library). See targetgen/corpora.py.

    Read STRUCTURALLY, from the module's own syntax tree, because the line scan this replaces flagged
    eleven docstrings that merely EXPLAIN the boundary -- including the docstring of the resolver that
    implements it -- and so told a reader the rule was broken in the one place it was being obeyed.
    Pre-existing real violations are recorded in library_boundary_ratchet.txt, which may only shrink.
    """
    ratchet = set()
    if os.path.isfile(BOUNDARY_RATCHET):
        with open(BOUNDARY_RATCHET, encoding="utf-8") as fh:
            ratchet = {ln.split("#", 1)[0].strip() for ln in fh if ln.split("#", 1)[0].strip()}
    # Name the sanctioned indirection from the allowlist itself. The message used to say
    # `merlin.common.corpora`, which has never existed -- an author following the instruction found no
    # such module, and the gate that reported four docstring false positives also told them the wrong
    # place to put the fix.
    _sanctioned = ".".join(os.path.splitext(sorted(_BOUNDARY_ALLOW)[0])[0].split(os.sep)[2:])
    for relative in _source_layout.python_files(Path(ROOT), _source_layout.SOURCE_SCAN_ROOTS):
        key = relative.as_posix()
        stable = _source_layout.policy_path(key)
        # Independently named experiment orchestration is not the compiler library. Shared merlin
        # namespace packages retain exactly the library rule they had before extraction.
        if not stable.startswith("merlin/python/merlin/"):
            continue
        if stable in _BOUNDARY_ALLOW or stable in ratchet:
            continue
        src = (Path(ROOT) / relative).read_text(encoding="utf-8")
        refs = _experiments_refs(src)
        if refs is None:
            # Fail closed: an unparseable module is UNSCANNED, not a clean boundary.
            try:
                ast.parse(src)
                why = "unknown"
            except SyntaxError as exc:
                why = str(exc)
            errors.append(f"library boundary unscannable (module does not parse): {key}: {why}")
            continue
        for lineno, literal in refs:
            # Optional research distributions may consume explicitly versioned DATA
            # from the study catalog. This is not permission to import engine code,
            # and the same dependency remains forbidden in the compiler core.
            resource = Path(literal)
            if (
                key.startswith("packages/")
                and resource.parts[:2] == ("experiments", "reference-data")
                and ".." not in resource.parts
            ):
                continue
            if (stable, literal) in _BOUNDARY_LITERAL_ALLOW:
                continue
            errors.append(f"library reads experiments/ (use {_sanctioned}): {key}:{lineno}: {literal!r}")


def check_core_dependencies(errors):
    from check_core_dependencies import audit

    errors.extend(audit(Path(ROOT)))


def main():
    errors: list[str] = []
    checks = [
        ("required directories", check_required_dirs),
        ("AGENT.md coverage", check_agent_md),
        ("experiment status", check_experiment_status),
        ("schema usage", check_schema_usage),
        ("library boundary", check_library_boundary),
        ("core dependency direction", check_core_dependencies),
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
        ("test target marker", check_test_target_marker),
        ("module size", check_module_size),
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
