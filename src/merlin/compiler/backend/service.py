"""Resolve explicitly selected installed, trusted compiler/runtime services.

This entry point is for ordinary compilation. Candidate grading retains its
independent source and execution admission; these plugins cannot qualify it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from importlib.metadata import entry_points
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _origin(distribution) -> dict[str, Any]:
    """Read declared installation origin, withholding URL credentials and queries."""
    if distribution is None:
        return {"status": "unknown"}
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return {"status": "unknown"}
    try:
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get("url"), str):
            raise ValueError("invalid origin")
        url = urlsplit(document["url"])
        public_url = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
        result: dict[str, Any] = {
            "status": "declared",
            "url": public_url,
            "url_redacted": public_url != document["url"],
        }
        directory = document.get("dir_info")
        vcs = document.get("vcs_info")
        archive = document.get("archive_info")
        if isinstance(directory, dict) and type(directory.get("editable", False)) is bool:
            result.update(kind="directory", editable=directory.get("editable", False))
        elif isinstance(vcs, dict):
            result.update(
                kind="vcs",
                vcs={
                    key: value
                    for key, value in vcs.items()
                    if key in {"vcs", "requested_revision", "commit_id"} and isinstance(value, str)
                },
            )
        elif isinstance(archive, dict):
            hashes = archive.get("hashes", {})
            result.update(
                kind="archive",
                hashes={key: value for key, value in hashes.items() if isinstance(key, str) and isinstance(value, str)}
                if isinstance(hashes, dict)
                else {},
            )
        else:
            result["kind"] = "unknown"
        return result
    except (ValueError, TypeError):
        return {"status": "invalid_metadata"}


def _description(entry) -> dict[str, Any]:
    distribution = getattr(entry, "dist", None)
    metadata = distribution.metadata if distribution is not None else {}
    return {
        "name": getattr(entry, "name", "unknown"),
        "entry_point": getattr(entry, "value", "unknown"),
        "distribution": {"name": metadata.get("Name", "unknown"), "version": getattr(distribution, "version", None)},
        "origin": _origin(distribution),
    }


def inventory(*, group: str = "merlin.backends") -> dict[str, Any]:
    """Describe installed services without importing, validating or executing them.

    Metadata and declared installation origins establish neither compatibility
    nor qualification. Checkout contracts and run-scoped candidates retain their
    separate explicit resolution and admission paths.
    """
    if group not in {"merlin.backends", "merlin.runners"}:
        raise ValueError("service inventory requires a compiler or runtime entry-point group")
    rows = [_description(entry) for entry in entry_points(group=group)]
    rows.sort(
        key=lambda row: (
            row["name"],
            row["distribution"]["name"],
            row["distribution"]["version"] or "",
            row["entry_point"],
        )
    )
    counts = Counter(row["name"] for row in rows)
    for row in rows:
        row.update(selection="ambiguous" if counts[row["name"]] > 1 else "unique", compatibility="not_checked")
    return {
        "schema": "merlin.service_inventory.v1",
        "group": group,
        "python": {"executable": sys.executable, "prefix": sys.prefix, "version": sys.version.split()[0]},
        "code_loaded_by_inspection": False,
        "scope": "installed_entry_point_metadata_only",
        "providers": rows,
    }


def resolve(selected: str | Any, *, group: str, methods: tuple[str, ...]) -> tuple[str, Any]:
    if isinstance(selected, str):
        matches = list(entry_points(group=group, name=selected))
        if len(matches) != 1:
            choices = sorted(
                f"{row['distribution']['name']}=={row['distribution']['version']} ({row['entry_point']})"
                for row in map(_description, matches)
            )
            detail = f"; candidates: {', '.join(choices)}" if choices else ""
            raise ValueError(
                f"expected one installed {group} provider named {selected!r}, found {len(matches)}{detail}"
            )
        provider = matches[0].load()
        name = selected
    else:
        provider = selected
        name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("an explicitly supplied service requires a nonempty name")
    for method in methods:
        if not callable(getattr(provider, method, None)):
            raise ValueError(f"service {name!r} lacks callable {method}")
    return name, provider
