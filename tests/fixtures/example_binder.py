"""The corpus binding a test states a group under: an example target's own Phase 0 recipe datapath.

A group put to a package is stated as the capsule the corpus writes for it
(:func:`merlin.llvmlower.whole_program.group_binder`), so a test needs the same explicit recipe
input a build does. Read from ``examples/<target>/phase0/recipe.yaml``; nothing is assumed here.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace


@functools.cache
def binder(target: str):
    import yaml

    from merlin.common.paths import repo_root
    from merlin.llvmlower import whole_program as WP

    recipe = repo_root() / "examples" / target / "phase0" / "recipe.yaml"
    datapath = yaml.safe_load(recipe.read_text(encoding="utf-8"))["datapath"]
    return WP.group_binder(SimpleNamespace(target=target, sim_via=""), datapath)
