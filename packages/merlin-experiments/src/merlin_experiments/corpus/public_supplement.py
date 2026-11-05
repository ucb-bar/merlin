"""Explicit public test inputs, separate from completed automatic generation."""

from pathlib import Path

import yaml

from merlin.common.digest import is_sha256, sha256_bytes, sha256_file
from merlin.common.strict_json import loads

from ..spec import SpecError

SCHEMA = "merlin.corpus.public_supplement.v1"


def read(path: Path, *, target: str) -> dict:
    from ..runner import fingerprint
    from .preparation import _members, ordinary_tree

    path = path.expanduser().absolute()
    ordinary_tree(path)
    if path != path.resolve() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise SpecError("public supplement requires a bounded ordinary selection file")
    raw = path.read_bytes()
    document = loads(raw, max_bytes=1024 * 1024)
    if (
        type(document) is not dict
        or set(document) != {"schema", "target", "origin", "members", "provenance"}
        or document["schema"] != SCHEMA
        or document["target"] != target
        or document["origin"] != "operator_requested"
        or type(document["members"]) is not list
        or not 1 <= len(document["members"]) <= 64
    ):
        raise SpecError("public supplement requires explicit same-target operator-requested members")
    provenance = document["provenance"]
    if (
        type(provenance) is not dict
        or set(provenance) != {"path", "sha256"}
        or type(provenance["path"]) is not str
        or not is_sha256(provenance["sha256"])
    ):
        raise SpecError("public supplement provenance pin is malformed")
    source = Path(provenance["path"])
    ordinary_tree(source)
    if (
        not source.is_absolute()
        or source != source.resolve()
        or not source.is_file()
        or sha256_file(source) != provenance["sha256"]
    ):
        raise SpecError("public supplement provenance differs from its selected bytes")
    actual = _members(path.parent)
    members = {}
    for row in document["members"]:
        if type(row) is not dict or set(row) != {"member", "sha256"} or type(row["member"]) is not str:
            raise SpecError("public supplement member pin is malformed")
        member = row["member"]
        relative = Path(member)
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.as_posix() != member
            or any(part in {".", "..", ""} or part.startswith(("_", ".")) for part in relative.parts)
            or relative.parts[0] == "hidden"
            or member in members
            or member not in actual
            or not is_sha256(row["sha256"])
        ):
            raise SpecError("public supplement must name unique ordinary public category/member paths")
        directory, capsule = actual[member]
        if capsule.get("label") != "public" or capsule.get("kind") == "model":
            raise SpecError("public supplement accepts public operator tests, never hidden or model sources")
        if fingerprint(directory) != row["sha256"]:
            raise SpecError("public supplement capsule differs from its selected bytes")
        members[member] = row["sha256"]
    if set(members) != set(actual) or path.read_bytes() != raw:
        raise SpecError("public supplement selection omits members or changed while reading")
    return {
        "schema": SCHEMA,
        "target": target,
        "origin": "operator_requested",
        "selection": {"path": str(path), "sha256": sha256_bytes(raw)},
        "provenance": provenance,
        "members": members,
    }


def stage(path: Path, destination: Path, *, te, private_roots=()) -> dict:
    from merlin.targetgen.sandbox.bwrap import _validate_host_sources

    from .preparation import _members, copy_input, read_yaml
    from .publication import require_accelerator_placement, require_public_capsules

    record = read(path, target=te.target)
    root = Path(record["selection"]["path"]).parent
    existing = _members(destination)
    names = {document["name"] for _, document in existing.values()}
    sources = [root / member for member in record["members"]]
    _validate_host_sources(
        [(str(source), source) for source in sources], [(str(source), source) for source in private_roots]
    )
    require_public_capsules(te, sources)
    require_accelerator_placement(sources)
    for member, digest in record["members"].items():
        if member in existing or (root / member).name in names:
            raise SpecError("public supplement cannot replace or alias an existing capsule identity")
        copy_input(root / member, destination / member, expected_sha256=digest)
    manifest = read_yaml(destination / "MANIFEST.yaml")
    manifest["operator_requested"] = sorted(set(manifest.get("operator_requested", [])) | set(record["members"]))
    manifest.setdefault("public_supplements", []).append(
        {
            "schema": SCHEMA,
            "origin": "operator_requested",
            "selection_sha256": record["selection"]["sha256"],
            "members": record["members"],
        }
    )
    (destination / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    verify(record, destination)
    return record


def verify(record: dict, corpus: Path) -> None:
    from ..runner import fingerprint

    if read(Path(record["selection"]["path"]), target=record["target"]) != record:
        raise SpecError("public supplement source selection changed after preparation")
    if any(fingerprint(corpus / member) != digest for member, digest in record["members"].items()):
        raise SpecError("staged public supplement differs from selected source bytes")
