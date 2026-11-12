"""Which capsules answer for a whole-model group: its exact op-form first, its op otherwise.

Shared by the whole-model gate and the measured mode's capsule check, so a failing group is mapped to
the same capsules wherever it is reported.  The catalog is an explicit directory; the group's form
comes from the caller (derived once from the one grouping), never recomputed here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: How many capsules a group name expands to, at most, so a check stays minutes long.
GROUP_CAPSULE_LIMIT = 12


def _capsule_rows(catalog: Path | None) -> list[dict[str, Any]]:
    import yaml

    if catalog is None or not catalog.is_dir():
        return []  # the launch names its capsule catalog; there is no implicit checkout corpus
    root = catalog
    rows = []
    for path in sorted(root.rglob("capsule.yaml")):
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if document.get("label") not in (None, "public"):
            continue
        form = (
            ((document.get("model_form") or {}).get("form") or {})
            if isinstance(document.get("model_form"), Mapping)
            else {}
        )
        rows.append(
            {
                "name": str(document.get("name") or path.parent.name),
                "op": str(((document.get("operation") or {}).get("op")) or form.get("op") or ""),
                "model_form": bool(form),
                "form": dict(form) if form else None,
                "extends": document.get("extends"),
            }
        )
    return rows


def _form_text(form: Mapping[str, Any] | None) -> str | None:
    return json.dumps(form, sort_keys=True, default=str) if form else None


def resolve_capsules(
    names: str,
    routes: Mapping[str, Mapping[str, Any]],
    catalog: Path | None = None,
    forms: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """``names`` is a comma list of capsule names and/or groups (``g6``).

    A group becomes the capsules of ITS OWN op-form when ``forms`` names it (``{group: form}``, see
    the launch's declared ``group_forms``): the capsules whose declared model form is that form exactly, the quick one
    before its certificate sibling. When the corpus holds no capsule of that form the mapping says so
    (``exact_form: false``) and the group falls back to the capsules of its op -- model-form capsules
    first (they carry the model's own extents). Certificate-spill siblings are always left out (they
    are sized past a quick check); at most GROUP_CAPSULE_LIMIT per group."""
    rows = _capsule_rows(catalog)
    known = {row["name"] for row in rows}
    chosen: list[str] = []
    mapping: dict[str, Any] = {}
    for token in (part.strip() for part in str(names or "").split(",")):
        if not token:
            continue
        index = token[1:] if token[:1] in ("g", "G") and token[1:].isdigit() else (token if token.isdigit() else None)
        if index is None:
            mapping[token] = token if token in known else "unknown capsule (the check will say so)"
            if token not in chosen:
                chosen.append(token)
            continue
        op = str((routes.get(index) or {}).get("op") or "")
        if not op:
            mapping[token] = "no build of this model has named this group's op yet"
            continue
        wanted = _form_text((forms or {}).get(index))
        exact = [r for r in rows if wanted and _form_text(r.get("form")) == wanted and not r["name"].endswith("_spill")]
        if exact:
            candidates = sorted(exact, key=lambda r: (r["name"].endswith("_cert"), r["name"]))
        else:
            candidates = [r for r in rows if r["op"] == op and not r["name"].endswith("_spill")]
            candidates.sort(key=lambda r: (not r["model_form"], r["name"].endswith("_cert"), r["name"]))
        picked = [r["name"] for r in candidates[:GROUP_CAPSULE_LIMIT]]
        mapping[token] = {
            "op": op,
            "capsules": picked,
            "of_matching": len(candidates),
            "exact_form": bool(exact),
            **({"form": (forms or {}).get(index)} if wanted else {}),
            **(
                {}
                if exact or not wanted
                else {"note": "the corpus holds no capsule of this group's exact op-form; these share only its op"}
            ),
        }
        chosen.extend(name for name in picked if name not in chosen)
    return chosen, mapping


__all__ = ["GROUP_CAPSULE_LIMIT", "resolve_capsules"]
