"""Closed operator selection of the original toolchain lookup snapshot.

This is optional storage/build plumbing. It supplies no compiler, device
strategy, runtime qualification, measured cycles or saved execution authority.
"""

from pathlib import Path

from .contracts import StageGateError

_FIELDS = {
    "tools",
    "readonly_roots",
    "bwrap",
    "snapshot_files",
    "snapshot_bytes",
    "cache_entries",
    "cache_bytes",
}


def validate(selection):
    document = selection.document.get("build_runtime")
    if document is None:
        return
    if type(document) is not dict or set(document) != _FIELDS:
        raise StageGateError("build lookup selection requires its complete closed original declaration")
    for key in ("snapshot_files", "snapshot_bytes", "cache_entries", "cache_bytes"):
        if type(document[key]) is not int or document[key] <= 0:
            raise StageGateError("build lookup and cache budgets must be explicit positive integers")
    from .evaluation_pool import AssignedResources

    values = selection.document["resources"]
    assigned = AssignedResources(**{**values, "cpu_ids": tuple(values["cpu_ids"])})
    reserved = int(assigned.disk_bytes * assigned.reserve_fraction)
    if document["snapshot_bytes"] + document["cache_bytes"] + selection.document["retained_source_bytes"] > reserved:
        raise StageGateError("shared build lookup, cache and source retention exceed the original disk reserve")
    tools, roots = document["tools"], document["readonly_roots"]
    if type(tools) is not list or not 1 <= len(tools) <= 32 or type(roots) is not list or not 1 <= len(roots) <= 32:
        raise StageGateError("build lookup requires bounded original tool and readonly root rosters")
    values = [*tools, document["bwrap"]]
    for row in roots:
        if type(row) is str:
            values.append(row)
        elif type(row) is dict and set(row) == {"source", "guest"}:
            values.extend((row["source"], row["guest"]))
        else:
            raise StageGateError("build lookup roots must be original paths or closed source/guest pairs")
    if any(type(value) is not str or not Path(value).is_absolute() or ".." in Path(value).parts for value in values):
        raise StageGateError("build lookup paths must be explicit absolute original selections")


def install(selection, policy, *, root, output_root):
    """Construct original live core owners before the normal queue starts."""
    validate(selection)
    document = selection.document.get("build_runtime")
    if document is None:
        return
    from merlin.runtime.build_runtime import prepare_build_runtime_view
    from merlin.targetgen.contract.build_products import prepare_build_product_index

    root.mkdir(mode=0o700)
    grant = selection.document["job_resources"]
    view = prepare_build_runtime_view(
        root / "lookup",
        tools=tuple(Path(value) for value in document["tools"]),
        readonly_roots=tuple(
            (Path(row["source"]), Path(row["guest"])) if type(row) is dict else Path(row)
            for row in document["readonly_roots"]
        ),
        output_root=output_root,
        bwrap=Path(document["bwrap"]),
        max_files=document["snapshot_files"],
        max_bytes=document["snapshot_bytes"],
        max_output_files=grant["files"],
        max_output_bytes=grant["disk_bytes"],
        max_console_bytes=min(selection.document["source_bytes"], grant["ram_bytes"]),
    )
    index = prepare_build_product_index(
        root / "products",
        context=policy.artifact_retest_context,
        build_owner=view,
        max_entries=document["cache_entries"],
        max_bytes=document["cache_bytes"],
        max_files=grant["files"],
    )
    policy.install_build_runtime(view, index)
