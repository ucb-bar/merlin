"""The wheel bundle must not ship what the answer-key gate keeps unpublished.

``check_no_answer_keys.py`` is green because every golden, held-out subtree and holdout sidecar is
UNTRACKED. ``setup.py``'s ``build_py`` hook copies ``merlin/contract`` into ``merlin/python/merlin/_data``
from DISK, not from git, so untracked-ness bought the wheel nothing: measured before the fix, the
bundle filter for the ``contract`` tree had no exclusions at all and copied ``hidden/``, ``golden/``,
``golden.yaml`` and ``golden_w8a8.yaml`` straight into the package.

Neither gate was wrong on its own. ``check_no_answer_keys`` correctly reported 0 tracked answer keys;
``check_standalone_install.py`` -- the only gate that inspects the WHEEL -- is wired into no hook, no CI
job, and does not complete (killed at 900 s; the ``contract`` tree it copies is 6.8 GB). The defect
lived in the gap between two green checks.

This file is the anti-drift link. ``_is_answer_key`` stays the single authority on what an answer
surface IS, and the EXCLUSION DATA is read out of ``setup.py`` by AST so it cannot drift from what the
build actually uses. Both then run against the real ``merlin/contract`` tree and must agree.
"""
from __future__ import annotations

import ast
import importlib.util

from merlin.common.paths import repo_root

REPO = repo_root()
CONTRACT = REPO / "merlin" / "contract"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AK = _load(REPO / "build_tools" / "scripts" / "check_no_answer_keys.py", "_ak_gate")


def _setup_literal(name: str):
    """One module-level literal out of ``setup.py``, by AST -- no execution, no setuptools.

    setup.py imports setuptools at module scope (a BUILD dependency, not guaranteed in a runtime venv),
    and a test whose real assertion sits behind an always-firing skip is the failure this file is about.
    So the exclusion data is read from setup.py itself; only the five-line predicate is restated.
    """
    tree = ast.parse((REPO / "setup.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"setup.py no longer defines {name}; the bundle filter has been restructured "
                         f"and this test must be re-pointed at it")


def _bundle_filter(kind: str):
    """setup.py's ignore predicate, over setup.py's own exclusion data."""
    prefixes = _setup_literal("_EXCLUDE").get(kind, ())
    suffixes = _setup_literal("_EXCLUDE_SUFFIXES").get(kind, ())
    code = tuple(_setup_literal("_CODE_SUFFIXES"))

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        return {n for n in names
                if n == "__pycache__" or n.endswith(code)
                or any(n.startswith(p) for p in prefixes)
                or any(n.endswith(x) for x in suffixes)}

    return _ignore


def _answer_surfaces_on_disk() -> list:
    """Real paths under merlin/contract that the answer-key gate classifies as answers."""
    return [p for p in CONTRACT.rglob("*")
            if AK._is_answer_key(p.relative_to(REPO).as_posix())]


def test_the_contract_tree_declares_answer_key_exclusions():
    """The ``contract`` tree must carry exclusions at all. It carried none."""
    assert _setup_literal("_EXCLUDE").get("contract"), (
        "setup.py's bundle filter declares no exclusions for the `contract` tree, so `uv build` copies "
        "every golden/ and hidden/ answer surface into merlin/_data and `pip install merlin` ships the "
        "benchmark's answers")


def test_the_tree_actually_contains_answer_surfaces():
    """Guard the guard: if the corpus has none, the agreement test below is vacuously true."""
    assert _answer_surfaces_on_disk(), (
        "no answer surface found under merlin/contract, so the agreement test proves nothing; check "
        "_is_answer_key or the corpus")


def test_every_answer_surface_is_dropped_from_the_bundle():
    """The two mechanisms must agree on the REAL tree, not on a restated rule.

    A file deep inside ``hidden/`` is dropped when its ANCESTOR directory is filtered, so a surface
    counts as excluded if it or any ancestor under merlin/contract is.
    """
    ignore = _bundle_filter("contract")

    def _excluded(path) -> bool:
        cur = path
        while cur != CONTRACT:
            if cur.name in ignore(str(cur.parent), [cur.name]):
                return True
            cur = cur.parent
        return False

    leaked = sorted(str(p.relative_to(REPO)) for p in _answer_surfaces_on_disk() if not _excluded(p))
    assert not leaked, (
        f"{len(leaked)} answer surface(s) survive setup.py's bundle filter and would be packaged into "
        f"the wheel:\n  " + "\n  ".join(leaked[:20]))


def test_the_filter_still_keeps_the_public_contract():
    """Over-exclusion is a real failure too: the wheel must still carry the public contract."""
    ignore = _bundle_filter("contract")
    keep = ["schemas", "command_buffer.schema.json", "hardware_pins.yaml", "capsule.yaml",
            "compute_endpoints.yaml", "perf_rules"]
    assert not ignore(str(CONTRACT), keep), (
        f"the bundle filter drops public contract data the SDK needs: {ignore(str(CONTRACT), keep)}")
