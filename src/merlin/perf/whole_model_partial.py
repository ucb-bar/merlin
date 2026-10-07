"""A whole-model build of only some groups: how it is made, and why it can never be read as a whole model.

``only_groups=["g12"]`` (``--only-group g12``) builds the program with the package asked for g12 alone.
Every other group keeps the target's library call exactly as a caller-declined group does
(``decline=``): it is stated by the reference, never put to the package, so a one-group debug build
costs one question instead of one per group. Which groups the model HAS is read off the statement
with no package (the same reference statement the per-group timing programs use), so a misspelled
group is refused rather than silently building nothing.

THAT PROGRAM RUNS AND PRINTS EVERY GROUP LINE. Nothing in its log says the other groups are the
library's, so the build marks itself: ``partial_build`` on the build record, on its oracle, on the
expectations a measurement reads, and a ``PARTIAL_BUILD.json`` beside them. :func:`refuse` is what a
verdict, a grade, a gate and a measurement call on what they are handed; a partial build is refused
by each, never scored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: The key a partial build carries on every document a whole-model reader is handed.
MARKER = "partial_build"
#: The file a partial build writes beside its record, so a directory listing says so too.
MARKER_FILE = "PARTIAL_BUILD.json"


class PartialBuildRefused(ValueError):
    """A partial (``only_groups``) build was handed to something that reads a whole model."""


def parse_groups(items: Iterable[Any]) -> list[int]:
    """``["g12,g14", 3]`` -> ``[3, 12, 14]``: group indices as the build names them (``g12`` or ``12``)."""
    groups: set[int] = set()
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"{item!r} names no group")
        for token in str(item).split(",") if not isinstance(item, int) else [str(item)]:
            token = token.strip()
            if not token:
                continue
            digits = token[1:] if token[:1] == "g" else token
            if not digits.isdigit():
                raise ValueError(f"{token!r} names no group; spell a group as g<index> (e.g. g12)")
            groups.add(int(digits))
    return sorted(groups)


def unasked(capsule: Any, *, target: str, only: Sequence[Any]) -> list[int]:
    """Every group of ``capsule``'s model that is NOT in ``only`` -- the groups a partial build declines.

    Read off the reference statement (no package asked anything), so the indices are the model's own;
    a group ``only`` names that the model does not have is refused."""
    from . import whole_model_build as W

    keep = parse_groups(only)
    if not keep:
        raise W.WholeModelBuildError("only_groups names no group")
    stated = W.state(capsule, target=target)
    present = [int(row["group"]) for row in stated["whole_program"]["per_group"]]
    missing = sorted(set(keep) - set(present))
    if missing:
        raise W.WholeModelBuildError(
            f"only_groups names {[f'g{g}' for g in missing]}, which this model does not have "
            f"(its groups are g{min(present)}..g{max(present)})"
        )
    return [group for group in present if group not in keep]


def marker(only: Sequence[Any], attribution: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What a partial build was: the groups asked for, the ones the package answered, and why it is not a
    whole model."""
    keep = parse_groups(only)
    answered = sorted(int(r["group"]) for r in attribution if r.get("on") == "package")
    return {
        "only_groups": [f"g{g}" for g in keep],
        "package_answered": [f"g{g}" for g in answered],
        "why": (
            "built with only_groups: every other group is the target's library call, so this program is "
            "not the candidate's whole model and is never measured, graded or gated as one"
        ),
    }


def mark(record: dict[str, Any], out: str | Path, only: Sequence[Any]) -> dict[str, Any]:
    """Put the marker on ``record``, on the oracle the build wrote, and beside them as :data:`MARKER_FILE`."""
    note = marker(only, (record.get("attribution") or {}).get("per_group") or ())
    record[MARKER] = note
    out = Path(out)
    (out / MARKER_FILE).write_text(json.dumps(note, indent=1) + "\n", encoding="utf-8")
    oracle = out / "oracle.json"
    if oracle.is_file():
        document = json.loads(oracle.read_text(encoding="utf-8"))
        document[MARKER] = note
        oracle.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return note


def refuse(document: Mapping[str, Any] | None, *, reader: str) -> None:
    """Raise :class:`PartialBuildRefused` when ``document`` (a build record, an oracle, expectations)
    carries the partial-build marker; ``reader`` names who refused it."""
    note = (document or {}).get(MARKER) if isinstance(document, Mapping) else None
    if note:
        groups = ", ".join((note or {}).get("only_groups") or []) if isinstance(note, Mapping) else ""
        raise PartialBuildRefused(
            f"{reader} refuses a partial build (only {groups or 'some groups'}): the other groups are the "
            "target's library call, so it is not the whole model"
        )
