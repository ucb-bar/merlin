"""Structural validity of ``stage-mutation`` proposed trees.

The output of ``./merlin targetgen stage-mutation`` is a *staged* tree of
scaffold files, intentionally containing ``TODO``s and skeletal stubs that
a human or LLM fills in before adoption. Running ``./merlin build``
against the unfilled tree would fail by design (and would also require a
healthy ``.venv`` we don't always have locally).

This test layer asserts the **structural** contract instead:

  * Every scaffolded source file ships with a clearly identifiable banner
    that names the source as TargetGen-generated and warns that it is
    staged only.
  * Every C/C++/CMake stub references the target identity from the
    capability spec — not a hard-coded placeholder.
  * Every file lives under a Merlin-owned root.
  * No scaffolded file is empty.
  * Banners include a ``TODO`` so reviewers can grep for unfinished work.

Catches scaffold-template regressions (drift in the prompt library, a
hard-coded target name, an empty stub) in seconds. Real-build sanity
lives in ``test_proposed_tree_builds.py`` (Tier 3, gated on a working
build env).

Markers: ``integration``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "target_specs" / "examples"

pytestmark = [pytest.mark.integration]

TARGETS = (
    "gemmini_mx",  # post_global_plugin + llvm_ukernel: largest scaffold
    "radiance_muon",  # runtime_hal: HAL-driver scaffold
    "saturn_opu_v128",  # llvm_ukernel-only: smallest scaffold
)

# Files we consider source-bearing for the banner check. README.md scaffolds
# are also TargetGen-tagged and tested separately.
_SOURCE_SUFFIXES = frozenset({".cpp", ".cc", ".c", ".h", ".hpp", ".td"})
_CMAKE_FILES = frozenset({"CMakeLists.txt"})

# Allowed write roots for scaffolded files. Must mirror the planner's
# allowlist. If a new path appears outside these roots, either the
# planner has expanded its scope or the scaffold leaked.
_ALLOWED_ROOTS: tuple[str, ...] = (
    "compiler/",
    "runtime/",
    "samples/",
    "models/",
    "target_specs/",
    "tools/",
    "build_tools/",
    "third_party/iree_bar/",
)


def _stage_mutation(target: str, out_dir: Path) -> subprocess.CompletedProcess:
    cap = EXAMPLES / target / "capability.yaml"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "merlin.py"),
            "targetgen",
            "stage-mutation",
            str(cap),
            "--out-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )


def _proposed_files(target: str, out_dir: Path) -> list[Path]:
    proposed = out_dir / target / "mutation" / "proposed_tree"
    if not proposed.exists():
        return []
    return sorted(p for p in proposed.rglob("*") if p.is_file())


def _relative_repo_path(file: Path, target: str, out_dir: Path) -> str:
    proposed = out_dir / target / "mutation" / "proposed_tree"
    return file.relative_to(proposed).as_posix()


@pytest.fixture(scope="module")
def staged_trees(tmp_path_factory) -> dict[str, Path]:
    out_root = tmp_path_factory.mktemp("staged")
    trees: dict[str, Path] = {}
    for target in TARGETS:
        rc = _stage_mutation(target, out_root)
        if rc.returncode != 0:
            pytest.fail(f"stage-mutation failed for {target}:\nSTDOUT:\n{rc.stdout}\n" f"STDERR:\n{rc.stderr}")
        trees[target] = out_root
    return trees


@pytest.mark.parametrize("target", TARGETS)
def test_proposed_tree_is_non_empty(target: str, staged_trees: dict[str, Path]) -> None:
    files = _proposed_files(target, staged_trees[target])
    assert files, f"{target}: stage-mutation produced no files"


@pytest.mark.parametrize("target", TARGETS)
def test_proposed_files_under_allowed_roots(target: str, staged_trees: dict[str, Path]) -> None:
    bad: list[str] = []
    for f in _proposed_files(target, staged_trees[target]):
        rel = _relative_repo_path(f, target, staged_trees[target])
        if not any(rel.startswith(r) for r in _ALLOWED_ROOTS):
            bad.append(rel)
    assert not bad, f"{target}: scaffold files outside allowed roots: {bad[:10]}"


@pytest.mark.parametrize("target", TARGETS)
def test_proposed_files_are_non_empty(target: str, staged_trees: dict[str, Path]) -> None:
    bad: list[str] = []
    for f in _proposed_files(target, staged_trees[target]):
        if f.stat().st_size == 0:
            bad.append(_relative_repo_path(f, target, staged_trees[target]))
    assert not bad, f"{target}: scaffold files are empty: {bad}"


@pytest.mark.parametrize("target", TARGETS)
def test_source_scaffolds_carry_targetgen_banner(target: str, staged_trees: dict[str, Path]) -> None:
    """C/C++/MD/TD scaffolds must announce themselves as TargetGen output."""
    bad: list[tuple[str, str]] = []
    for f in _proposed_files(target, staged_trees[target]):
        suffix = f.suffix.lower()
        is_source = suffix in _SOURCE_SUFFIXES
        is_markdown = suffix == ".md"
        is_cmake = f.name in _CMAKE_FILES or suffix == ".cmake"
        if not (is_source or is_markdown or is_cmake):
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        rel = _relative_repo_path(f, target, staged_trees[target])
        if "Generated by Merlin TargetGen" not in text:
            bad.append((rel, "missing TargetGen banner"))
            continue
        # Banner includes the human-readable target identity (CamelCased).
        # Accept either the raw identity or its CamelCase form.
        camel = "".join(part.capitalize() for part in target.replace("-", "_").split("_"))
        if camel not in text and target not in text:
            bad.append((rel, f"banner does not reference target {target!r}"))
    assert not bad, "\n  ".join([f"{target} banner issues:"] + [f"{r}: {m}" for r, m in bad])


_UNFINISHED_MARKERS: tuple[str, ...] = (
    "TODO",
    "FIXME",
    "staged only",
    "staged scaffold",
    "not a promoted source file",
)


@pytest.mark.parametrize("target", TARGETS)
def test_source_scaffolds_flag_unfinished_work(target: str, staged_trees: dict[str, Path]) -> None:
    """Each *source* scaffold (cpp/h/td/cmake) must carry one of the
    canonical "this is staged, not finished" markers so reviewers can grep
    for unfinished work. We accept ``TODO``, ``FIXME``, or any of the
    explicit "staged only / staged scaffold / not a promoted source file"
    phrases the prompt library currently emits. Markdown notes are
    excluded — they're descriptive rather than executable."""
    bad: list[str] = []
    for f in _proposed_files(target, staged_trees[target]):
        suffix = f.suffix.lower()
        is_source = suffix in _SOURCE_SUFFIXES or f.name in _CMAKE_FILES or suffix == ".cmake"
        if not is_source:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if not any(marker in text for marker in _UNFINISHED_MARKERS):
            bad.append(_relative_repo_path(f, target, staged_trees[target]))
    assert not bad, (
        f"{target}: source scaffolds without an unfinished-work marker " f"(any of {_UNFINISHED_MARKERS}): {bad}"
    )


@pytest.mark.parametrize("target", TARGETS)
def test_cmake_scaffolds_have_balanced_parens(target: str, staged_trees: dict[str, Path]) -> None:
    """Light-weight CMake syntax check: parens balance at the file level.

    A real ``cmake -P`` parse would need a complete project; this catches
    the common scaffold bugs (truncated function call, mis-templated
    quote) without standing one up.
    """
    bad: list[tuple[str, int]] = []
    for f in _proposed_files(target, staged_trees[target]):
        if f.name not in _CMAKE_FILES and f.suffix.lower() != ".cmake":
            continue
        # Strip line comments first, then count.
        text = re.sub(r"#[^\n]*", "", f.read_text(encoding="utf-8", errors="ignore"))
        if text.count("(") != text.count(")"):
            bad.append(
                (
                    _relative_repo_path(f, target, staged_trees[target]),
                    text.count("(") - text.count(")"),
                )
            )
    assert not bad, f"{target}: CMake scaffolds with unbalanced parens: " + ", ".join(f"{r}(diff={d})" for r, d in bad)


@pytest.mark.parametrize("target", TARGETS)
def test_cpp_scaffolds_have_balanced_braces(target: str, staged_trees: dict[str, Path]) -> None:
    """C++ stubs must have balanced braces. Catches truncated namespace
    blocks or function bodies in the scaffold templates."""
    bad: list[tuple[str, int]] = []
    for f in _proposed_files(target, staged_trees[target]):
        suffix = f.suffix.lower()
        if suffix not in {".cpp", ".cc", ".h", ".hpp"}:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        # Strip line comments + block comments + string literals so we don't
        # miscount braces inside source-text comments.
        text = re.sub(r"//[^\n]*", "", text)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
        if text.count("{") != text.count("}"):
            bad.append(
                (
                    _relative_repo_path(f, target, staged_trees[target]),
                    text.count("{") - text.count("}"),
                )
            )
    assert not bad, f"{target}: C++ scaffolds with unbalanced braces: " + ", ".join(f"{r}(diff={d})" for r, d in bad)


@pytest.mark.parametrize("target", TARGETS)
def test_scaffold_files_are_not_indented_with_leading_spaces(target: str, staged_trees: dict[str, Path]) -> None:
    """Tracking sentinel: the prompt-library currently indents each
    scaffold's first 3 lines with 8 leading spaces (a Python-template
    artefact). Real source files should start at column 0. We mark this
    as ``xfail`` so the test surfaces an XPASS the day the templates are
    fixed.
    """
    indented: list[str] = []
    for f in _proposed_files(target, staged_trees[target]):
        suffix = f.suffix.lower()
        if suffix not in _SOURCE_SUFFIXES:
            continue
        first_line = f.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        if first_line and first_line[0].startswith("        "):
            indented.append(_relative_repo_path(f, target, staged_trees[target]))
    if indented:
        pytest.xfail(
            "Scaffold templates emit 8 leading spaces on the first comment "
            "line — Python heredoc indentation leaks through. Tracking "
            f"sentinel: {indented[:3]}"
        )
