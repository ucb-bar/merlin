"""Every registrable lever must be visible to the SEARCH, not merely to the lowering.

`wholemodel_proposer._composes` catches the KeyError that `impr_features.get` raises for an
unregistered name and returns False. So a feature the proposer's process has never imported is not
"declined" — it is INVISIBLE. It can never be proposed, it is never reported as rejected, and a
later improvement to it stays invisible too.

This has now happened four times: `named_int8_contraction`, `fold_weight_transpose`,
`cse_through_provenance` (which, being in the champion config, made EVERY parent stack carrying it
uncomposable — so no lever at all could be proposed on top of the winner), and then `concat_dps` and
`fuse_epilogue_loops` together. Each was found by hand, days or hours apart. This test finds the next
one automatically, by enumerating the registrars rather than listing the features.
"""
from __future__ import annotations

import subprocess
import sys

from merlin.common.paths import merlin_dir, repo_root

#: Modules that expose `ensure_registered` but are lowering INFRASTRUCTURE rather than a lever the
#: search selects. Keep this list short and justified; anything else must be reachable.
_NOT_LEVERS = {"pipeline.py", "lower.py"}


def _registrar_modules() -> set[str]:
    """Lever modules whose registration is an IMPORT, i.e. one fixed feature name.

    A registrar whose ``ensure_registered`` REQUIRES an argument is a name-derived FAMILY (the
    argument is the tuning point), so there is no single name for the proposer to import and
    register; those resolve through ``impr_features._try_lazy_register`` instead, and are covered by
    :func:`test_every_name_derived_lever_family_resolves_from_its_name_alone`."""
    import importlib
    import inspect

    llvm = merlin_dir() / "python" / "merlin" / "llvmlower"
    out = set()
    for path in sorted(llvm.glob("*.py")):
        if path.name in _NOT_LEVERS:
            continue
        if "def ensure_registered" not in path.read_text(encoding="utf-8"):
            continue
        fn = getattr(importlib.import_module(f"merlin.llvmlower.{path.stem}"),
                     "ensure_registered", None)
        if fn is not None and any(
                p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                for p in inspect.signature(fn).parameters.values()):
            continue                     # a family, not a name: reachable lazily, tested below
        out.add(path.stem)
    return out


#: One name from each name-derived lever family, and the module that owns it. These cannot be
#: registered by import (the point IS the name), so what has to hold is that a proposer-only process
#: resolves them from the string alone -- the same property the import gives the fixed levers.
_NAME_DERIVED_LEVERS = [("parallel_grain", "parallel_grain_10000")]


def test_every_name_derived_lever_family_resolves_from_its_name_alone():
    """`_composes` swallows the KeyError for an unregistered name and returns False, so a family the
    lazy registrar does not know is invisible to the search exactly like an unimported registrar."""
    import importlib
    import inspect

    llvm = merlin_dir() / "python" / "merlin" / "llvmlower"
    families = set()
    for path in sorted(llvm.glob("*.py")):
        if path.name in _NOT_LEVERS or "def ensure_registered" not in path.read_text():
            continue
        fn = getattr(importlib.import_module(f"merlin.llvmlower.{path.stem}"),
                     "ensure_registered", None)
        if fn is not None and any(
                p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                for p in inspect.signature(fn).parameters.values()):
            families.add(path.stem)
    covered = {m for m, _ in _NAME_DERIVED_LEVERS}
    assert families <= covered, (
        f"name-derived lever families with no reachability case here: {sorted(families - covered)}")
    code = ("import json\n"
            "from merlin.mining.wholemodel_proposer import _composes\n"
            f"names = {[n for _, n in _NAME_DERIVED_LEVERS]!r}\n"
            "print(json.dumps([n for n in names if not _composes([n])]))\n")
    env = {"PYTHONPATH": str(merlin_dir() / "python"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=repo_root(), env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "[]", r.stdout


def test_every_lever_registrar_is_imported_by_the_proposer():
    """A registrar the proposer never imports is a lever the beam can never propose."""
    src = (merlin_dir() / "python" / "merlin" / "mining" / "wholemodel_proposer.py").read_text()
    missing = sorted(m for m in _registrar_modules() if f"llvmlower.{m} import ensure_registered" not in src)
    assert not missing, (
        f"these lever modules define ensure_registered but the proposer does not import it: {missing}. "
        "_composes swallows the KeyError and returns False, so each is silently unproposable.")


def test_every_ranked_lever_resolves_in_a_proposer_only_process():
    """The decisive check: a FRESH interpreter that imports only the proposer must resolve every
    ranked name. Importing `llvmlower.lower` first would mask the bug, which is exactly how
    `cse_through_provenance` passed review while being unreachable."""
    code = (
        "import json\n"
        "from merlin.mining.wholemodel_proposer import RANKED_LEVERS, _composes\n"
        "bad = [n for n, _ in RANKED_LEVERS if not _composes([n])]\n"
        "print(json.dumps(bad))\n"
    )
    env = {"PYTHONPATH": str(merlin_dir() / "python"), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=repo_root(), env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    unreachable = r.stdout.strip().splitlines()[-1]
    assert unreachable == "[]", f"ranked but unresolvable from a proposer-only import: {unreachable}"


def test_the_champion_config_can_still_be_extended():
    """`cse_through_provenance` being unreachable made the WINNER a dead end: no lever could be
    proposed on top of it, so the search could not improve on its own best result."""
    from merlin.mining.wholemodel_proposer import _composes

    champion = ["prepack_weight_layout", "perop_register_block", "promote_buffers_to_stack",
                "expand_memref_copy", "cse_through_provenance"]
    assert _composes(champion), "the champion config itself does not compose"
    assert _composes(champion + ["fuse_elementwise_post_contraction"])
