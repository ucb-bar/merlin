#!/usr/bin/env python3
"""Validate the merlin repository scaffold.

Checks:
  1. Required directories exist.
  2. Maintained area and study boundaries carry scoped instructions.
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


def _reference_contract_dirs():
    """Existing authored example contracts; generated target packages stay outside source."""
    examples = Path(ROOT) / "examples"
    return sorted(
        path.parent.relative_to(ROOT).as_posix()
        for category in ("compiler-generation", "targets")
        for path in (examples / category).glob("*/target/contracts/target_contract.yaml")
        if path.is_file()
    )


REQUIRED_DIRS = [
    "build_tools/scripts",
    "docs",
    "third_party",
    "tests",
    "src/merlin/compiler",
    "src/merlin/target",
    "src/merlin/runtime",
    "src/merlin/verification",
    "src/merlin/experiments",
    "src/merlin/analysis/dse",
    "src/merlin/common",
    "src/merlin/runtime/native/c",
    "src/merlin/runtime/native/abi",
    "src/merlin/runtime/native/baremetal",
    "src/merlin/resources/schemas",
    "src/merlin/resources/benchmarks/semantic_memory",
    "src/merlin/development",
    "experiments/definitions",
    "tests/data",
]

# Authored example contracts share the same source-owned directory checks.
REQUIRED_DIRS.extend(_reference_contract_dirs())

REQUIRED_SCHEMAS = [
    "target_contract",
    "instruction_semantics",
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
    # Per-form exactness contract every whole-model gate enforces (merlin.perf.exactness).
    "exactness_contract",
    # Full-model check report (merlin.model_check): build, census, kernel exactness, model metrics.
    "model_check_report",
    # Custody of sealed Phase 0 evidence (merlin.experiments.corpus.evidence_custody).
    "evidence_seal",
    "evidence_repair",
]

REQUIRED_DOCS = [
    "reference/architecture",
    "reference/repo_structure",
    "guides/01-inputs",
    "guides/02-targets",
    "guides/03-preparation",
    "guides/04-planning",
    "guides/05-transformations",
    "guides/06-lowering",
    "guides/07-building",
    "guides/08-running",
    "guides/09-verification",
    "guides/10-compiler-generation",
    "guides/11-optimization",
    "guides/12-experiments",
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
    for directory in REQUIRED_DIRS:
        if not (Path(ROOT) / directory).is_dir():
            errors.append(f"missing required directory: {directory}")


def check_agent_md(errors):
    """Require scoped instructions at maintained area and study boundaries.

    Package instructions are checked by gen_package_docs. Deep fixture, resource
    and archival directories inherit the nearest meaningful owner's rules.
    """

    def require(relative):
        if not (Path(ROOT) / relative / "AGENT.md").is_file():
            errors.append(f"missing AGENT.md: {relative or '(root)'}")

    require("")
    require("experiments")
    for area in ("tests",):
        root = Path(ROOT) / area
        require(area)
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name not in SKIP_DIRS | {"fixtures", "data"}:
                require(child.relative_to(ROOT).as_posix())


TARGET_MARKER_RATCHET = os.path.join(ROOT, "build_tools", "scripts", "test_target_marker_ratchet.txt")
MODULE_SIZE_RATCHET = os.path.join(ROOT, "build_tools", "scripts", "module_size_ratchet.txt")
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
    tests = os.path.join(ROOT, "tests")
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
            rel = f"tests/{bucket}/{fn}"
            with open(os.path.join(bp, fn), encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if _quoted_target_literals(text, names) < TARGET_HEAVY_LITERALS or "mark.target(" in text or rel in ratchet:
                continue
            errors.append(f"target-heavy test without `pytestmark = pytest.mark.target(...)`: {rel}")


def _module_size_ceilings(errors) -> dict[str, int]:
    """``module_size_ratchet.txt`` as ``{debt identity: line ceiling}``.

    Each entry is ``<module>  # <ceiling>``. The number is enforced, not a note: the ledger used to
    excuse a listed module at ANY size, and listed modules grew past their recorded sizes with nothing
    able to notice (capsule_runner.py: recorded at 5,145 lines, 6,821 by 2026-10-08). An entry without
    a numeric ceiling is itself an error.
    """
    ceilings: dict[str, int] = {}
    if not os.path.isfile(MODULE_SIZE_RATCHET):
        return ceilings
    with open(MODULE_SIZE_RATCHET, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, _, ceiling = stripped.partition("#")
            key, ceiling = key.strip(), ceiling.strip()
            if not ceiling.isdigit():
                errors.append(f"module-size ratchet entry has no numeric line ceiling after '#': {key}")
                continue
            ceilings[key] = int(ceiling)
    return ceilings


def check_module_size(errors):
    """No library module may grow past MODULE_SIZE_LIMIT lines, nor a recorded one past its ceiling.

    Existing offenders are listed in module_size_ratchet.txt with the line count each may not exceed;
    the ledger may only shrink (check_ratchets_shrink.py holds both the entries and the ceilings). A
    listed module may shrink but not grow, and leaves the ledger by being split under the limit. A NEW
    module over the limit fails here. Counts physical lines of the file as committed-shape source, i.e.
    after ruff format, so a formatter pass cannot push a module over.
    """
    ceilings = _module_size_ceilings(errors)
    for relative in _source_layout.python_files(Path(ROOT), _source_layout.SOURCE_SCAN_ROOTS):
        rel = relative.as_posix()
        with open(Path(ROOT) / relative, encoding="utf-8", errors="replace") as fh:
            n = sum(1 for _ in fh)
        key = rel if rel in ceilings else _source_layout.policy_path(rel)
        ceiling = ceilings.get(key)
        if ceiling is None:
            if n > MODULE_SIZE_LIMIT:
                errors.append(f"module over {MODULE_SIZE_LIMIT} lines ({n}); split it: {rel}")
        elif n > ceiling:
            errors.append(f"module grew past its recorded ceiling ({n} > {ceiling} lines); shrink or split it: {rel}")
        elif n <= MODULE_SIZE_LIMIT:
            errors.append(f"module is now within {MODULE_SIZE_LIMIT} lines ({n}); delete its ratchet entry: {rel}")


def check_schemas(errors):
    for s in REQUIRED_SCHEMAS:
        p = os.path.join(ROOT, "src", "merlin", "resources", "schemas", f"{s}.schema.yaml")
        if not os.path.isfile(p):
            errors.append(f"missing schema: src/merlin/resources/schemas/{s}.schema.yaml")
        elif os.path.getsize(p) == 0:
            errors.append(f"empty schema: src/merlin/resources/schemas/{s}.schema.yaml")


def check_docs(errors):
    for d in REQUIRED_DOCS:
        p = os.path.join(ROOT, "docs", f"{d}.md")
        if not os.path.isfile(p):
            errors.append(f"missing doc: docs/{d}.md")


def check_benchmarks(errors):
    for b in REQUIRED_BENCHMARKS:
        p = os.path.join(ROOT, "src", "merlin", "resources", "benchmarks", "semantic_memory", f"{b}.yaml")
        if not os.path.isfile(p):
            errors.append(f"missing benchmark: src/merlin/resources/benchmarks/semantic_memory/{b}.yaml")


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
        for path in examples.glob("*/*")
        if (path / "target/descriptor.yaml").is_file() or (path / "target/contracts/target_contract.yaml").is_file()
    }


_GENERIC_TEST_BUCKETS = {"kernels", "rvv", "dse", "targetgen", "ir", "runtime", "infra"}
TEST_BUCKETS = (
    _GENERIC_TEST_BUCKETS
    | {Path(relative).parts[-3] for relative in _reference_contract_dirs()}
    | _example_test_buckets()
)


def check_test_layout(errors):
    """Every test lives in tests/<bucket>/test_*.py (bucket in the fixed set); none at root."""
    tdir = os.path.join(ROOT, "tests")
    if not os.path.isdir(tdir):
        return
    for name in os.listdir(tdir):
        if name.startswith("test_") and name.endswith(".py"):
            errors.append(f"test at tests/ root (must be in a subsystem bucket): {name}")
    for b in sorted(os.listdir(tdir)):
        bp = os.path.join(tdir, b)
        if not os.path.isdir(bp) or b in {"fixtures", "data", "__pycache__"}:
            continue
        if b not in TEST_BUCKETS:
            has_tests = any(f.startswith("test_") and f.endswith(".py") for f in os.listdir(bp))
            if has_tests:
                errors.append(f"unknown test bucket tests/{b} (allowed: {sorted(TEST_BUCKETS)})")

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
                    f"duplicate test module basename {f!r}: tests/{seen[f]}/{f} and "
                    f"tests/{b}/{f} — pytest imports test files by bare basename, so this "
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
    """Module index is current; existing scoped instructions contain no stale claims."""
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
    """out/artifacts/documentation/schemas.md is in sync with src/merlin/resources/schemas/ (see gen_schema_docs.py)."""
    import subprocess

    gen = os.path.join(ROOT, "build_tools", "scripts", "gen_schema_docs.py")
    r = subprocess.run([sys.executable, gen, "--check"], capture_output=True, text=True)
    if r.returncode != 0:
        errors.append(
            "out/artifacts/documentation/schemas.md stale — run python build_tools/scripts/gen_schema_docs.py"
        )


# The library (merlin/python/merlin) reads INPUTS from benchmarks/ and contract/, never from
# experiments/ (experiments consume the library, one-way). The ONE sanctioned indirection to a corpus
# still under experiments/ is the corpus locator; everything else must not name experiments/ as a path.
# Corpus discovery and the package/source-owner path resolver own physical layout.
# These are indirections, not permission for compiler algorithms to read study code.
_BOUNDARY_ALLOW = {
    "merlin/python/merlin/targetgen/corpora.py",
    "merlin/python/merlin/common/paths.py",
}
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
        "experiments/studies/cpu_host_compiler_v0/grader.py",
    ): "loads that experiment's source-sealed grader to reuse its gate functions verbatim (not a corpus)",
    (
        "merlin/python/merlin/compare/host_experiment.py",
        "experiments/studies/cpu_host_compiler_v0/optimization_space_v1.yaml",
    ): "binds the calibration to that experiment's frozen optimization space, by content",
}


def check_schema_usage(errors):
    """Every src/merlin/resources/schemas/*.schema.yaml must be referenced by name in merlin/python/ — either
    validated (validate/validate_or_raise/_SCHEMA/PLAN_SCHEMAS) or mirrored as a vocabulary spec
    (a code constant / generated view / docstring that names it). Zero references ⇒ dead schema."""
    import glob

    corpus = []
    for path in _source_layout.python_files(Path(ROOT), _source_layout.SOURCE_SCAN_ROOTS):
        corpus.append((Path(ROOT) / path).read_text(encoding="utf-8"))
    blob = "\n".join(corpus)
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "merlin", "resources", "schemas", "*.schema.yaml"))):
        name = os.path.basename(path)[: -len(".schema.yaml")]
        if f'"{name}"' not in blob and f"'{name}'" not in blob and f"{name}.schema" not in blob:
            errors.append(f"dead schema (no reference in src/merlin): src/merlin/resources/schemas/{name}.schema.yaml")


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
    components = (
        _source_layout.logical_components(Path(ROOT))
        if (Path(ROOT) / "build_tools/logical_components.json").is_file()
        else None
    )
    ratchet = set()
    if os.path.isfile(BOUNDARY_RATCHET):
        with open(BOUNDARY_RATCHET, encoding="utf-8") as fh:
            ratchet = {ln.split("#", 1)[0].strip() for ln in fh if ln.split("#", 1)[0].strip()}
    # Name the sanctioned indirection from the allowlist itself. The message used to say
    # `merlin.common.corpora`, which has never existed -- an author following the instruction found no
    # such module, and the gate that reported four docstring false positives also told them the wrong
    # place to put the fix.
    _sanctioned = "merlin.targetgen.corpora"
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
                (
                    key.startswith("packages/")
                    if components is None
                    else _source_layout.component_owner(
                        _source_layout.module_name(Path(ROOT) / relative, Path(ROOT)) or "", components
                    )
                    != "core"
                )
                and resource.parts[:2] == ("experiments", "reference-data")
                and ".." not in resource.parts
            ):
                continue
            # These read-only inputs moved intact from benchmarks/, targets/,
            # contract/capsules/ and tests/data/. The move does not make them
            # orchestration code; study engines remain behind the same boundary.
            if (
                resource.parts[:2]
                in {
                    ("experiments", "benchmarks"),
                    ("experiments", "reference-data"),
                    ("experiments", "corpora"),
                }
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
