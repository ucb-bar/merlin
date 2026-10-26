"""Exact identities from one fresh inventory; independent boundaries rescan."""

from pathlib import Path, PurePosixPath
from typing import Any

from .sealed_static import SealedCaptureError, _digest, _json


def _snapshot_rows(rows: dict[str, dict[str, Any]], relative: str = ".") -> dict[str, Any]:
    """Digest an exact subtree without reading its bytes again at this checkpoint."""
    member = PurePosixPath(relative)
    if member.is_absolute() or ".." in member.parts or member.as_posix() != relative:
        raise SealedCaptureError("inventory subtree path is not canonical")
    if rows.get(relative, {}).get("kind") != "directory":
        raise SealedCaptureError("inventory subtree is absent or not a directory")
    if relative != ".":
        prefix = relative + "/"
        rows = {
            ".": rows[relative],
            **{name[len(prefix) :]: info for name, info in rows.items() if name.startswith(prefix)},
        }
    compact = [(name, *sorted(info.items())) for name, info in sorted(rows.items())]
    return {
        "members": len(rows),
        "bytes": sum(row.get("bytes", 0) for row in rows.values()),
        "sha256": _digest(_json(compact)),
    }


def _verify_staged_selection(
    plan: dict[str, Any],
    source: Path,
    runtime: Path,
    *,
    source_rows: dict[str, dict[str, Any]] | None = None,
    runtime_rows: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reject changed selected bytes before executing any guest code."""
    from . import sealed_m2m as M

    def identity(path: Path, *, allow_dangling_symlinks: bool = False) -> dict[str, Any]:
        root, rows = (runtime, runtime_rows) if path.is_relative_to(runtime) else (source, source_rows)
        if rows is None:
            return M._snapshot_tree(path, allow_dangling_symlinks=allow_dangling_symlinks)
        return _snapshot_rows(rows, path.relative_to(root).as_posix())

    roots = {
        "venv": runtime / "opt/capture-venv",
        "base": runtime / Path(plan["base"]).relative_to("/"),
        "m2m": source / "m2m-src/m2m",
        "workload": source / "workload",
        "schemas": source / "merlin-src/merlin/_data/schemas",
    }
    selected_trees = plan["selected_trees"]
    if plan.get("frozen_origin") is not None:
        try:
            origin_commit = M.verify_frozen_receipt(
                plan["frozen_origin"],
                (source / "m2m-origin.json").read_bytes(),
                Path(plan["m2m_root"]),
                selected_trees["m2m"],
            )
        except (M.M2MOriginError, OSError) as exc:
            raise M.SealedM2MError("staged frozen M2M origin differs from selection") from exc
        if origin_commit != plan["m2m_commit"]:
            raise M.SealedM2MError("staged frozen M2M origin revision differs from selected plan")
    for name, path in roots.items():
        if identity(path, allow_dangling_symlinks=name == "base") != selected_trees[name]:
            raise M.SealedM2MError(f"staged {name} bytes differ from the pre-execution selection")
    if plan.get("schema") == M.SCHEMA_V3:
        for row in plan["selected_inputs"]:
            M._verify_selected_input(row, source / "inputs" / row["guest_member"])
    if plan["schemas_root"] == str(Path(plan["merlin_root"]) / "_data/schemas"):
        if identity(source / "merlin-src/merlin") != selected_trees["merlin"]:
            raise M.SealedM2MError("staged Merlin bytes differ from the pre-execution selection")
    return selected_trees["schemas"]


def _replay_output_difference(original: Path, replay: Path) -> str:
    """Failure-only detail; never relax exact byte, membership or mode equality."""
    from .sealed_static import _tree

    expected, actual = _tree(original), _tree(replay)
    for name in sorted(set(expected) | set(actual)):
        if name not in actual:
            return f"missing member {name!r}"
        if name not in expected:
            return f"unexpected member {name!r}"
    modes = []
    for name, row in sorted(expected.items()):
        other = actual[name]
        if row.get("kind") != other.get("kind"):
            return f"member kind differs at {name!r}"
        if row.get("sha256") != other.get("sha256") or row.get("bytes") != other.get("bytes"):
            return f"content hash or size differs at {name!r}"
        if {k: v for k, v in row.items() if k != "mode"} != {k: v for k, v in other.items() if k != "mode"}:
            return f"inventory fields differ at {name!r}"
        if row.get("mode") != other.get("mode"):
            modes.append((name, row["mode"], other["mode"]))
    if modes:
        name, old, new = modes[0]
        return f"mode-only difference at {name!r} (expected {old:#o}, actual {new:#o}); check caller umask"
    return "materialized receipt differs although file inventory matches"


def _verify_replayed_output(replay, source, output, doc, schema, plan) -> None:
    """The original complete materialization and inventory checks, with failure detail."""
    from . import sealed_m2m as M

    materialized = (
        M._materialized(replay, source, output)
        if schema == M.SCHEMA_V1
        else (
            M._materialized_v3(replay, source, output, plan)
            if schema == M.SCHEMA_V3
            else M._materialized_v2(replay, source, output, plan)
        )
    )
    if materialized != doc.get("materialized") or M._snapshot_tree(replay) != doc["output"]:
        detail = _replay_output_difference(output, replay)
        raise M.SealedM2MError("fresh M2M replay differs: " + detail)
