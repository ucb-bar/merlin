"""Release-local corpus selection for a run with no historical corpus input."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..spec import SpecError


def bind_generated_corpus(te, generated: Path, *, generated_only: bool):
    """Choose a public category without discovering any historical source pool.

    The primary category is an anchor: native discovery also includes every
    public sibling category. Its spelling therefore does not select a subset
    of the generated public cohort. Private and feedback categories stay out.
    """
    if te.capsule_corpus is not None:
        return te
    if not generated_only:
        raise SpecError("a release without a source corpus requires generated-only preparation")
    categories = [
        directory
        for directory in sorted(generated.iterdir())
        if directory.is_dir()
        and not directory.is_symlink()
        and directory.name != "hidden"
        and not directory.name.startswith(("_", "."))
        and next(directory.glob("*/capsule.yaml"), None) is not None
    ]
    if not categories:
        raise SpecError("generated-only release has no public capsule category")
    return replace(te, capsule_corpus=categories[0])
