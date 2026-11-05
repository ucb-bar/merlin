"""Explicit additional public archive inputs for native diagnostic exploration.

These are data declarations, never saved qualification owners. Every archive,
binding and original report is reopened before unchanged joint held validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from merlin_timing import archive as A
from merlin_timing import archive_union as UNION
from merlin_timing import contracts as C

from merlin.runtime import _native_files as F

from . import archive_spike_runtime as RT
from .contracts import StageGateError


@dataclass(frozen=True)
class AdditionalArchiveEvidence:
    archive: Path
    archive_sha256: str
    binding_path: Path
    observations_path: Path


def declarations(additional):
    if (
        type(additional) is not tuple
        or len(additional) >= UNION.MAX_ARCHIVES
        or any(type(item) is not AdditionalArchiveEvidence for item in additional)
    ):
        raise StageGateError("additional native archives require exact original host declarations")
    result = []
    for item in additional:
        C.sha(item.archive_sha256, "additional archive")
        for path in (item.archive, item.binding_path, item.observations_path):
            if type(path) is not type(Path()) or path.is_symlink() or not path.is_absolute() or path.resolve() != path:
                raise StageGateError("additional archive evidence requires canonical ordinary file paths")
        result.append(
            {
                "archive": str(item.archive),
                "archive_sha256": item.archive_sha256,
                "binding_path": str(item.binding_path),
                "observations_path": str(item.observations_path),
            }
        )
    return result


def selected(selection):
    """Content to pin privately at issuance; V1 has no additional declaration."""
    return declarations(selection.additional_archives)


def paths(selection):
    declarations(selection.additional_archives)
    return tuple(
        path
        for item in selection.additional_archives
        for path in (item.archive, item.binding_path, item.observations_path)
    )


def component(runtime, *, archive, archive_sha256, binding_path, observations_path):
    inventory = A.inspect_archive(archive, archive_sha256)
    binding, binding_pin = RT.metadata(binding_path)
    observations, observations_pin = RT.metadata(observations_path)
    model = json.loads(runtime.model_json)
    domain = dict(model["domain"], target_model_sha256=F.digest(runtime.model_path))
    if (
        binding["domain"] != domain
        or binding["review"]["timing_selection"] != json.loads(runtime.timing_json)
        or inventory.get("export_scope") != "training_eligible_public"
    ):
        raise StageGateError("joint archive differs from original public runtime/model/window selection")
    plan = A.prepare(inventory, binding)
    return (
        {
            "archive": str(archive),
            "archive_sha256": archive_sha256,
            "binding_path": str(binding_path),
            "observations_path": str(observations_path),
            "dataset": A.attach(plan, observations),
            "input_files": [
                [str(archive), archive_sha256],
                [str(binding_path), binding_pin["sha256"]],
                [str(observations_path), observations_pin["sha256"]],
            ],
        },
        inventory,
        plan,
    )


def review(runtime, primary, additional):
    declarations(additional)
    items = [
        primary,
        *(
            {
                "archive": item.archive,
                "archive_sha256": item.archive_sha256,
                "binding_path": item.binding_path,
                "observations_path": item.observations_path,
            }
            for item in additional
        ),
    ]
    components, joins = [], {}
    for item in items:
        part, inventory, plan = component(runtime, **item)
        cases = {case["case_id"]: case for case in part["dataset"]["cases"]}
        observations = {row["record_id"]: row for row in plan["observations"]}
        for row in inventory["records"]:
            group = (row["row"].get("join") or {}).get("source_group_key")
            if group is not None:
                C.sha(group, "original joined source group")
                split = cases[observations[row["record_id"]]["case_id"]]["split"]
                if joins.setdefault(group, split) != split:
                    raise StageGateError("original joined compiler variants cross source splits")
        components.append(part)
    dataset = UNION.join(tuple(part["dataset"] for part in components))
    check_inputs(components)
    return dataset, components


def check_inputs(components):
    for part in components:
        for path, digest in part["input_files"]:
            if F.pin(Path(path), max_bytes=256 << 20)["sha256"] != digest:
                raise StageGateError("joint archive input changed during original review")


def catalog(reviewed, *, archive, archive_sha256, binding_path):
    """Use each original descriptive source consumer on its exact component."""
    from . import _archive_feedback as HISTORY

    if "archive_components" not in reviewed:
        return HISTORY.prepare(
            archive=archive, archive_sha256=archive_sha256, binding_path=binding_path, dataset=reviewed["dataset"]
        )
    components = reviewed["archive_components"]
    check_inputs(components)
    if UNION.join(tuple(part["dataset"] for part in components)) != reviewed["dataset"]:
        raise StageGateError("joint source catalog changed original component membership")
    catalogs = [
        HISTORY.prepare(
            archive=Path(part["archive"]),
            archive_sha256=part["archive_sha256"],
            binding_path=Path(part["binding_path"]),
            dataset=part["dataset"],
        )
        for part in components
    ]
    result = dict(catalogs[0], schema="merlin.phase2.retained_archive_feedback.v2")
    result["additional_archive_sha256s"] = [part["archive_sha256"] for part in catalogs[1:]]
    result["sources"], result["cases"] = {}, []
    for part in catalogs:
        for digest, source in part["sources"].items():
            if result["sources"].setdefault(digest, source) != source:
                raise StageGateError("same original source has different descriptive semantics")
        result["cases"].extend(dict(entry, archive_sha256=part["archive_sha256"]) for entry in part["cases"])
    check_inputs(components)
    return result
