"""``rtl_backend`` must derive levers when staged FLAT, not only when imported as a package.

MEASURED, and invisible in-tree. The experiment sandbox does not import this module as
``merlin.targetgen.rtl_backend``; it stages it at the workspace root and imports it as plain
``rtl_backend``. Its ``from .rtl.facts import load_facts`` then raises "attempted relative import with
no known parent package", the caller caught that and returned None, the live bridge is masked in the
box too, and ``target_profile`` handed back a profile with every field None.

The consequence was not a crash but a silent wrong answer: the MANDATED authoring command
``derived_levers(target_profile('gemmini'))`` returned ``[]`` for a target with a 16x16 mesh and a
64 KiB accumulator, and the run was recorded NOT CONFORMANT on ``rtl_derived_levers_used`` for a
reason with nothing to do with the work being judged. Every in-tree test passed throughout, because
in-tree the relative import resolves.

So this test stages the module the way the sandbox does and asserts the levers survive it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

SRC = merlin_dir() / "python/merlin/targetgen/rtl_backend.py"

#: A target with a mesh and an accumulator — the shape whose levers the empty list was hiding.
_FACTS = {
    "facts": {
        "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
        "memories": [
            {"name": "scratchpad", "bytes": 262144, "depth": 4096},
            {"name": "accumulator", "bytes": 65536, "depth": 512},
        ],
        "interfaces": [{"legal_funct": [0, 1, 2, 3, 126]}],
    }
}


def _stage_flat(tmp_path: Path):
    """Reproduce the sandbox layout: rtl_backend.py at the root, `rtl` a package beside it."""
    (tmp_path / "rtl_backend.py").write_text(SRC.read_text())
    pkg = tmp_path / "rtl"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "facts.py").write_text(
        "import json, pathlib\n"
        "def load_facts(target, *, explicit=None):\n"
        "    return json.loads((pathlib.Path(__file__).parent / 'facts.json').read_text())\n"
    )
    (pkg / "facts.json").write_text(json.dumps(_FACTS))
    return tmp_path


def _import_flat(tmp_path: Path):
    """Import rtl_backend as a TOP-LEVEL module, exactly as the sandbox does."""
    sys.path.insert(0, str(tmp_path))
    for name in ("rtl_backend", "rtl", "rtl.facts"):
        sys.modules.pop(name, None)
    try:
        spec = importlib.util.spec_from_file_location("rtl_backend", tmp_path / "rtl_backend.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["rtl_backend"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        pass


@pytest.fixture
def flat(tmp_path, monkeypatch):
    _stage_flat(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = _import_flat(tmp_path)
    yield mod
    for name in ("rtl_backend", "rtl", "rtl.facts"):
        sys.modules.pop(name, None)


def test_profile_is_grounded_when_staged_flat(flat):
    """The failure mode was an all-None profile, which reads as 'this hardware has nothing'."""
    p = flat.target_profile("gemmini")
    assert not p.discovered_nothing, (
        "flat-staged rtl_backend produced an ungrounded profile: the relative import fell through and "
        "every RTL fact came back absent, which is the condition that made derived_levers return []"
    )
    assert p.has_mesh and p.has_accumulator
    assert p.dim == 16


def test_derived_levers_are_not_empty_when_staged_flat(flat):
    """The mandated authoring command must answer for a target that plainly has levers."""
    levers = flat.derived_levers(flat.target_profile("gemmini"))
    assert levers, (
        "derived_levers returned [] for a mesh + accumulator target staged the way the sandbox stages "
        "it — the exact silent answer that failed rtl_derived_levers_used on a real run"
    )
    assert "spatial.dataflow" in levers
    assert "spatial.accumulator_resident" in levers


def test_gaps_stay_silent_when_nothing_is_actually_missing(flat):
    """An empty gap tuple is only correct when the facts really were read."""
    assert flat.lever_derivation_gaps(flat.target_profile("gemmini")) == ()
