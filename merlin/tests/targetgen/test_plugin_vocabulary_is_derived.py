"""The ``plugin`` vocabulary must cover every key the tree actually ASKS FOR.

``plugins.PLUGIN_KEYS`` is a closed vocabulary: :func:`plugins.validate` rejects anything it does not
list, and that validation is the gate every package passes through on its way into
``registry.load_target``. So a key that core *consumes* but the vocabulary does not *list* is worse
than a missing feature — it is a package rejected for declaring the thing core asked it to declare.
That was not hypothetical: ``dialect`` (loaded by ``target_lowering._ensure_dialects_discovered``) and
``sim_oracle`` (loaded by ``capsule_runner._ensure_sim_oracles_discovered``) were both live seams,
both declared by shipped reference contracts, and neither was in the list. Saturn's own contract failed
Merlin's own validator; it went unnoticed only because ``validate`` runs from ``load_target`` alone and
a reference target never reaches it.

The check therefore DERIVES the required vocabulary from the call sites instead of restating it. A
literal list here would be a second place to update and would drift exactly the way the first one did.
Three structural forms count as "core asks for this key", each found by AST rather than by text:

* ``_oot_plugin_modules("<key>")`` — the shared plugin-discovery helper (and its default argument);
* ``plugins.load_declared(<target>, "<key>")`` — the load-the-target's-own-tool seam;
* ``plugin.get("<key>")`` / ``plugin["<key>"]`` on a variable literally named ``plugin`` — a reader
  that has a plugin block in hand and pulls one key out of it.

A form that is not constant-folded (``_oot_plugin_modules(key)`` with a variable) is skipped rather
than guessed at; the test says so rather than pretending completeness it does not have.
"""

from __future__ import annotations

import ast
import warnings

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import plugins

#: Where a consumer of a plugin key may live. ``experiments`` is included because the
#: ``reference_programs`` seam is consumed exclusively from there — leaving it out would make the
#: derivation quietly incomplete in precisely the direction that matters.
SCAN_ROOTS = ("python/merlin", "experiments", "targets")

#: Functions whose FIRST string-literal argument names a plugin key.
_KEY_IS_ARG0 = {"_oot_plugin_modules"}
#: Functions whose SECOND string-literal argument names a plugin key.
_KEY_IS_ARG1 = {"load_declared"}
#: A variable holding a plugin block; ``<name>.get("k")`` / ``<name>["k"]`` names key ``k``.
_BLOCK_NAMES = {"plugin"}


def _callee(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


def _const_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _keys_in(tree: ast.AST) -> set[str]:
    """Every plugin key this module asks for, by the three structural forms above."""
    found: set[str] = set()
    for node in ast.walk(tree):
        # `def _oot_plugin_modules(key: str = "backend")` — the default IS a call site.
        if isinstance(node, ast.FunctionDef) and node.name in _KEY_IS_ARG0:
            for default in node.args.defaults:
                if (k := _const_str(default)) is not None:
                    found.add(k)
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        if name in _KEY_IS_ARG0 and node.args and (k := _const_str(node.args[0])) is not None:
            found.add(k)
        if name in _KEY_IS_ARG1 and len(node.args) > 1 and (k := _const_str(node.args[1])) is not None:
            found.add(k)
        # plugin.get("key")
        if (
            name == "get"
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in _BLOCK_NAMES
            and node.args
            and (k := _const_str(node.args[0])) is not None
        ):
            found.add(k)
    # plugin["key"]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in _BLOCK_NAMES
            and (k := _const_str(node.slice)) is not None
        ):
            found.add(k)
    return found


def _consumed_keys() -> dict[str, list[str]]:
    """``key -> [files that ask for it]``, derived from the tree."""
    out: dict[str, list[str]] = {}
    root = merlin_dir()
    for rel in SCAN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                # Some scanned sources contain literals the compiler warns about; the warning belongs
                # to that file, not to this scan, and must not be re-emitted once per test run.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            for key in _keys_in(tree):
                out.setdefault(key, []).append(str(path.relative_to(root)))
    return out


def test_every_key_core_asks_for_is_in_the_vocabulary():
    """A key core loads but the vocabulary omits REJECTS the package that declares it."""
    consumed = _consumed_keys()
    assert consumed, "the scanner found no plugin call sites at all — it has stopped measuring"
    missing = {k: v for k, v in consumed.items() if k not in plugins.PLUGIN_KEYS}
    assert not missing, (
        "these plugin keys are consumed by core but absent from plugins.PLUGIN_KEYS, so "
        "plugins.validate REJECTS any package that declares one: "
        + "; ".join(f"{k} (asked for in {', '.join(v)})" for k, v in sorted(missing.items()))
    )


def test_the_two_keys_this_check_was_written_for_are_present_and_marked_consumed():
    """The regression. Both were live seams with shipped declarations and neither was listed."""
    for key in ("dialect", "sim_oracle"):
        assert key in plugins.PLUGIN_KEYS, f"plugin.{key} is loaded by core"
        assert plugins.PLUGIN_KEYS[key].consumed is True
        assert plugins.PLUGIN_KEYS[key].summary, "a consumed key must say what it is for"


@pytest.mark.parametrize("target", ["saturn", "muon", "gemmini"])
def test_a_shipped_reference_contract_passes_its_own_validator(target):
    """Saturn declared ``plugin.dialect`` and muon ``plugin.sim_oracle``; both failed ``validate``."""
    import yaml

    from merlin.common.paths import targets_dir

    path = targets_dir() / target / "contracts" / "target_contract.yaml"
    if not path.is_file():
        pytest.skip(f"no reference contract for {target}")
    contract = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = path.parent.parent
    assert plugins.validate(contract.get("plugin"), root=root) == []


def test_the_scanner_would_catch_a_removal():
    """MUTATION: drop a key from the vocabulary and the derivation must notice.

    Without this, a scanner that silently found nothing (a moved root, a renamed helper) would read
    as a clean bill of health forever.
    """
    consumed = set(_consumed_keys())
    known = set(plugins.PLUGIN_KEYS)
    assert consumed & known, "the scanner found no key that is in the vocabulary — it is not scanning"
    victim = sorted(consumed & known)[0]
    shrunk = {k: v for k, v in plugins.PLUGIN_KEYS.items() if k != victim}
    assert victim not in shrunk
    assert [k for k in consumed if k not in shrunk] == [victim]
