"""Every single-name lever must resolve BY NAME, in a process that imports nothing else.

The failure this guards against is silent in the worst way: a lever module registers itself where it
is RANKED (``wholemodel_proposer``), so the beam can propose it and a beam run works -- while a
package naming it in ``compiler_features``, or a ``--features`` request, reaches ``normalize``
through ``k1.build_k1_binary``, which imports no proposer, and dies on ``unknown impr feature``.

That gap was open on seven of thirteen lever modules, because the resolver carried a HAND-MAINTAINED
list of two. The other eleven resolved only by ACCIDENT -- an unrelated module happening to import
theirs (``pipeline`` pulls in ``prov_cse`` and ``im2col_pack``), or a call-time
``ensure_registered()`` buried inside ``k1.build_k1_binary`` (what ``prepack_weight_layout`` relies
on). Board runs therefore worked, which is precisely why this stayed invisible: nothing failed until
someone reached a lever by a path that skipped whichever import happened to carry it. A lever the
search can PROPOSE but a fresh process cannot RESOLVE by name is not a reproducible result.

The test is written against the MODULES, not against a list of names, so a lever added tomorrow is
covered the day it lands.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


def _lever_modules() -> dict[str, str]:
    """FEATURE -> module stem, parsed from the sources (the same rule the resolver derives)."""
    out: dict[str, str] = {}
    for src in sorted((merlin_dir() / "python" / "merlin" / "llvmlower").glob("*.py")):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        feature, has_ensure = None, False
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str) \
                    and any(isinstance(t, ast.Name) and t.id == "FEATURE" for t in node.targets):
                feature = node.value.value
            elif isinstance(node, ast.FunctionDef) and node.name == "ensure_registered":
                has_ensure = True
        if feature and has_ensure:
            out[feature] = src.stem
    return out


def test_lever_modules_are_discovered() -> None:
    """The resolver's derived map covers every module that declares one."""
    from merlin.llvmlower import impr_features as F

    assert _lever_modules() == F._single_name_lever_modules()


@pytest.mark.parametrize("name", sorted(_lever_modules()))
def test_lever_resolves_by_name_in_a_fresh_process(name: str) -> None:
    """Resolvable in a SUBPROCESS that imports only ``impr_features``.

    The subprocess is the point, not decoration: the lowering runs in one, and a registration the
    parent performed at run time is invisible there. Asserting in-process would pass on a lever that
    only ever registers as a side effect of something else this test session already imported.
    """
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "from merlin.llvmlower import impr_features as F\n"
         f"f = F.get({name!r})\n"
         f"assert f.name == {name!r}, f.name\n"
         "print('resolved')\n"],
        capture_output=True, text=True, timeout=300,
        cwd=str(Path(merlin_dir()).parent),
    )
    assert proc.returncode == 0, f"{name} did not resolve by name: {proc.stderr[-1500:]}"
    assert "resolved" in proc.stdout


def test_normalize_accepts_every_lever_at_once() -> None:
    """``normalize`` is the door every package and ``--features`` request goes through."""
    from merlin.llvmlower import impr_features as F

    names = sorted(_lever_modules())
    assert set(F.normalize(names)) >= set(names)
