"""The ``generate`` package must import when staged FLAT, not only as ``merlin.targetgen.generate``.

Same defect class as ``test_rtl_backend_survives_flat_staging``, and it cost the same thing. The
experiment sandbox stages this package at the workspace root, so ``generate`` IS the top-level package
and ``from ...common.artifacts import Artifact`` raises "attempted relative import beyond top-level
package". Because ``generate/__init__.py`` imports its modules EAGERLY, one such import takes the
entire package down: the agent's mandated ``import generate.mlir_scaffold`` failed outright, it
hand-staged a stub package to get past it, and the run was recorded not conformant on
``scaffold_generators_used``.

In-tree every one of these imports resolves, which is why no existing test saw it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

GEN = merlin_dir() / "python/merlin/targetgen/generate"

#: Imported eagerly by ``generate/__init__.py``. ``runtime_adapter`` is excluded on purpose: it renders
#: a callable route to the reference simulator and is deny-masked in the authoring sandbox, so it is
#: NOT staged and must not be made importable there.
EAGER = ("llvm_plan", "mlir_scaffold", "target_repo", "xdsl", "zephyr_module")


def _stage(tmp_path: Path) -> Path:
    """Reproduce the sandbox layout: `generate` and `common` as sibling top-level packages."""
    gen = tmp_path / "generate"
    gen.mkdir()
    for f in GEN.glob("*.py"):
        (gen / f.name).write_text(f.read_text())
    common = tmp_path / "common"
    common.mkdir()
    (common / "__init__.py").write_text("")
    # Only the surface the staged modules import, so the test pins the IMPORT PATH rather than
    # dragging the real artifacts module (and its own dependencies) into a unit test.
    (common / "artifacts.py").write_text(
        textwrap.dedent(
            """
            class Artifact:
                def __init__(self, *a, **k):
                    self.args, self.kwargs = a, k

            def yaml_artifact(*a, **k):
                return Artifact(*a, **k)
            """
        )
    )
    return tmp_path


@pytest.mark.parametrize("mod", EAGER)
def test_eagerly_imported_module_imports_when_staged_flat(tmp_path, mod):
    """Each module the package imports eagerly must survive flat staging on its own."""
    root = _stage(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", f"import generate.{mod}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"generate.{mod} does not import when staged flat — the condition that made a mandated "
        f"authoring command fail and cost a run its conformance:\n{proc.stderr.strip()[-600:]}"
    )


def test_the_package_itself_imports_when_staged_flat(tmp_path):
    """__init__ imports its modules eagerly, so the package is all-or-nothing."""
    root = _stage(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", "import generate; print(generate.mlir_scaffold.__name__)"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"the flat-staged `generate` package does not import:\n{proc.stderr.strip()[-600:]}"
    assert "mlir_scaffold" in proc.stdout


def test_in_tree_package_import_is_unchanged():
    """The fallback must not have cost us the ordinary path."""
    from merlin.targetgen import generate

    for mod in EAGER:
        assert hasattr(generate, mod), f"merlin.targetgen.generate lost {mod}"
