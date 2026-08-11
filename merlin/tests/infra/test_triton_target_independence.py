"""Architecture guards for the Triton kernel frontend (``merlin.triton``).

The whole claim of that package is that a Triton kernel enters Merlin ABOVE the
target-selection boundary, so adding a target costs zero Triton work. That claim is only
true while the package stays target-blind, which is a property of the source — so it is
asserted here rather than described in a doc. Design + the numbered invariants:
``docs/design/triton_frontend.md``.

The forbidden target names are DISCOVERED from the target registry (``all_targets()``), never
listed here: registering a new target must extend this gate with zero edits, exactly like
``build_tools/scripts/check_no_target_name.py``.

INV-3 (no TargetGen/schema edits) is checked by ATTRIBUTION rather than by a time window. The
invariant is "*the Triton work* does not modify TargetGen", so the test asks whether any single
commit touches both the Triton frontend and the protected surface — never "did anything change
since date X". A window would be wrong twice over on this repo: this long-lived branch carried
unrelated TargetGen work before the frontend existed, and the tree is developed by several agents
at once, so a concurrent agent's TargetGen commit would report as a Triton violation. Attribution
has no baseline to keep up to date and stays correct however the tree is shared.
"""
from __future__ import annotations

import ast
import subprocess

import pytest

from merlin.common.paths import repo_root

REPO = repo_root()
PKG = REPO / "merlin" / "python" / "merlin" / "triton"

# INV-3's protected surface: TargetGen plus the two schemas a Triton-specific carve-out would
# most plausibly want to widen.
PROTECTED = (
    "merlin/python/merlin/targetgen",
    "merlin/schemas/target_contract.schema.yaml",
    "merlin/schemas/dialect_plan.schema.yaml",
)

# What counts as "the Triton work" for attribution: the frontend package and its tests/examples.
TRITON_PATHS = (
    "merlin/python/merlin/triton",
    "merlin/tests/fixtures/triton_kernels.py",
    "examples/triton",
)

# Imports that would smuggle a target-specific decision into the frontend. `targetgen.target_registry`
# is deliberately NOT here: resolving "what does this target's contract say" is the generic seam the
# frontend is supposed to use.
FORBIDDEN_IMPORT_PREFIXES = ("merlin.targets", "merlin.rvvgen")


def _sources() -> list:
    return sorted(PKG.rglob("*.py")) if PKG.is_dir() else []


def _tokens(text: str) -> set[str]:
    """Identifier-ish tokens of ``text``, lowercased.

    Split structurally (no regex — this mirrors the repo's no-regex mandate for the code under
    test): any character that cannot appear in an identifier is a separator. Tokenizing rather
    than substring-matching keeps a target name from being "found" inside an unrelated word.
    """
    out, cur = set(), []
    for ch in text.lower():
        if ch.isalnum() or ch == "_":
            cur.append(ch)
        elif cur:
            out.add("".join(cur))
            cur = []
    if cur:
        out.add("".join(cur))
    return out


def _target_names() -> set[str]:
    from merlin.targetgen.target_registry import all_targets

    return {t.lower() for t in all_targets()}


def _git(*args: str) -> str:
    r = subprocess.run(("git", "-C", str(REPO)) + args, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def test_package_exists_with_a_docstring():
    """The package is the thing under guard; a missing docstring also breaks gen_package_docs."""
    init = PKG / "__init__.py"
    assert init.is_file(), "merlin/python/merlin/triton/__init__.py is missing"
    assert ast.get_docstring(ast.parse(init.read_text(encoding="utf-8"))), \
        "merlin.triton needs a module docstring (module_index.md is generated from it)"


def test_no_target_name_literals_anywhere_in_the_frontend():
    """INV-2: the target is a parameter, never a literal — for every registered target."""
    names = _target_names()
    assert names, "target registry reported no targets — the gate would be vacuous"
    offenders = {}
    for path in _sources():
        hits = _tokens(path.read_text(encoding="utf-8")) & names
        if hits:
            offenders[str(path.relative_to(REPO))] = sorted(hits)
    assert not offenders, (
        "target-name literals in merlin.triton (INV-2) — thread the target through as a "
        f"parameter from the contract instead: {offenders}")


def test_no_target_specific_imports():
    """INV-2/INV-5: no reaching into a target's own modules from the frontend."""
    offenders = {}
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bad = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            else:
                continue
            for mod in mods:
                if any(mod == p or mod.startswith(p + ".") for p in FORBIDDEN_IMPORT_PREFIXES):
                    bad.add(mod)
        if bad:
            offenders[str(path.relative_to(REPO))] = sorted(bad)
    assert not offenders, f"target-specific imports in merlin.triton (INV-2): {offenders}"


def test_no_regex_in_the_frontend():
    """The repo's structural-parsing mandate: a too-narrow pattern silently drops valid input.

    Enforced here as well as in check_no_regex.py so it fails in the suite a developer actually
    runs while writing the bridge, not only at commit time.
    """
    offenders = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(a.name == "re" for a in node.names):
                offenders.append(str(path.relative_to(REPO)))
            elif isinstance(node, ast.ImportFrom) and node.module == "re":
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        "`import re` in merlin.triton — parse TTIR structurally (tokenizer / xDSL IR), because a "
        f"regex line-matcher silently drops valid-but-differently-spelled input: {sorted(set(offenders))}")


def test_no_triton_change_also_touches_targetgen_or_the_schemas():
    """INV-3: no TargetGen or schema edit is needed to make a target Triton-programmable.

    Attribution, not a time window: a commit that changes the Triton frontend must not also change
    the protected surface, and the uncommitted working tree must not straddle both either. A
    concurrent agent's unrelated TargetGen commit is therefore correctly ignored.
    """
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    anchor = _git("log", "--diff-filter=A", "--format=%H", "--",
                  "merlin/python/merlin/triton/__init__.py").splitlines()
    if not anchor:
        pytest.skip("merlin.triton not committed yet — nothing to attribute")

    straddling = {}
    for sha in _git("log", "--format=%H", f"{anchor[-1]}^..HEAD", "--", *TRITON_PATHS).splitlines():
        if not sha:
            continue
        touched = [ln for ln in _git("show", "--format=", "--name-only", sha,
                                     "--", *PROTECTED).splitlines() if ln]
        if touched:
            straddling[sha[:8]] = sorted(touched)

    # Uncommitted work counts as one prospective commit: staged + unstaged, tracked files only.
    pending_triton = [ln for ln in _git("diff", "HEAD", "--name-only", "--", *TRITON_PATHS).splitlines() if ln]
    pending_protected = [ln for ln in _git("diff", "HEAD", "--name-only", "--", *PROTECTED).splitlines() if ln]
    if pending_triton and pending_protected:
        straddling["<working tree>"] = sorted(pending_protected)

    assert not straddling, (
        "a change to the Triton frontend also changes TargetGen/schemas (INV-3) — making a target "
        "Triton-programmable must cost zero TargetGen edits, so the fix belongs in Merlin's shared "
        f"lowering instead: {straddling}")
