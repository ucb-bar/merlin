"""Write captured compute groups as corpus capsules through the one Phase 0 writer.

The binding and the interface are core (:mod:`merlin.targetgen.group_capsule_entries`), so a
whole-program build can ask a package for a group without importing this package. Materializing a
group as a capsule on disk -- golden, instruction classes, declared blocks, path scrubbing -- is the
Phase 0 writer's job and happens only here, for every derived group capsule: model forms, form-perf
cells and the ad-hoc ``corpus groups`` command alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.targetgen.group_capsule_entries import interface_entry

from . import provenance, writer


def write_group_capsule(
    entry: Mapping[str, Any], binding, out_root: str | Path, *, facts_sha256: str = ""
) -> Path | None:
    """Build one stated group under ``binding`` and write it below ``out_root/<cat>/<name>``.

    The same builder the corpus uses (``writer._write_capsule``), then the same scrub the
    generator applies to every emitted capsule, so a derived group capsule carries no local path and
    nothing distinguishes it from any other corpus member except its declared ``source_role``.
    """
    stated = interface_entry(entry)
    arguments = (stated, binding, Path(out_root), *((facts_sha256,) if facts_sha256 else ()))
    written = writer._write_capsule(*arguments)
    if written:
        if Path(written).is_dir():
            provenance._scrub_capsule_dir(written)
        return Path(written)
    return None
