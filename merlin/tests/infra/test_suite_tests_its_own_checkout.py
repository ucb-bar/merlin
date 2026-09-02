"""The suite must exercise the ``merlin`` that lives beside it, not whichever one is installed.

``merlin`` is installed editable: one ``.pth`` line naming ONE checkout's ``merlin/python``. The
``.venv`` is shared between several checkouts of this repo, so whichever last ran an editable install
owns that line for all of them, and every other checkout's ``pytest`` then imports the winner's
library while collecting its own tests.

Measured 2026-09-02: a sibling checkout held the line. Tests of code written in THIS tree passed
against the sibling's copy of that module, and tests of modules the sibling lacked errored at
collection. The first half is the dangerous one — a green test for code that never ran.

``merlin/tests/conftest.py`` fixes it by putting the checkout's own ``merlin/python`` on ``sys.path``
before anything imports ``merlin``. This test is here so that fix cannot be undone silently: without
it the symptom is a PASS, and nothing distinguishes "the code is right" from "the code was never
imported".
"""
from __future__ import annotations

from pathlib import Path

import merlin
import merlin.common.paths as paths


def _checkout_of(p: Path) -> Path:
    """The repo root containing ``p``: ``<root>/merlin/...`` -> ``<root>``."""
    return p.resolve().parents[2]


def test_the_imported_library_comes_from_this_checkout():
    """``import merlin`` must resolve inside the tree this test file lives in."""
    # <root>/merlin/tests/infra/<this file>  ->  <root>
    here = Path(__file__).resolve().parents[3]
    # <root>/merlin/python/merlin/__init__.py -> parents[1] == <root>/merlin/python
    got = Path(merlin.__file__).resolve()
    assert got.is_relative_to(here), (
        f"the suite is testing another checkout's library: this test file is in {here}, but "
        f"`import merlin` resolved to {got}. A shared editable venv points at one checkout; "
        f"merlin/tests/conftest.py must put this checkout's merlin/python on sys.path first.")


def test_repo_root_agrees_with_the_test_files_checkout():
    """``paths.repo_root()`` is what every fixture path is built from, so it must agree too.

    It resolves from the imported module's own ``__file__``, so a wrong library silently relocates
    every fixture, artifact and capsule lookup in the suite to a tree nobody is editing -- and those
    lookups then succeed, against the wrong content.
    """
    here = Path(__file__).resolve().parents[3]
    assert paths.repo_root().resolve() == here, (
        f"paths.repo_root() is {paths.repo_root()} but this test file lives under {here}; "
        f"fixture and capsule lookups would read a different checkout")
