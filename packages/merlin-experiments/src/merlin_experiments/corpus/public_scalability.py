"""Explicit required public functional inputs, separate from the core RTL cohort."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from merlin.common.digest import is_sha256, sha256_bytes, sha256_file
from merlin.common.strict_json import loads

from ..spec import SpecError

SCHEMA = "merlin.corpus.public_scalability.v1"
STAGED = "merlin.corpus.public_scalability.staged.v1"
KEY = "public_scalability_suite"
POLICY = {"required": True, "engine": "spike", "tiers": ["L0", "L1", "L2"]}


def read(path: Path, *, target: str) -> dict:
    from ..runner import fingerprint
    from .preparation import _members, ordinary_tree

    path = path.expanduser().absolute()
    ordinary_tree(path)
    if path != path.resolve() or not path.is_file():
        raise SpecError("public scalability selection must be an ordinary canonical file")
    raw = path.read_bytes()
    doc = loads(raw, max_bytes=1024 * 1024)
    if (
        type(doc) is not dict
        or set(doc) != {"schema", "target", "origin", "policy", "members", "provenance"}
        or doc["schema"] != SCHEMA
        or doc["target"] != target
        or doc["origin"] != "operator_requested"
        or doc["policy"] != POLICY
        or type(doc["members"]) is not list
        or not 1 <= len(doc["members"]) <= 512
    ):
        raise SpecError("select a bounded explicit required Spike-only public scalability suite")
    provenance = doc["provenance"]
    if (
        type(provenance) is not dict
        or set(provenance) != {"path", "sha256"}
        or type(provenance["path"]) is not str
        or not is_sha256(provenance["sha256"])
    ):
        raise SpecError("public scalability provenance pin is malformed")
    source = Path(provenance["path"])
    ordinary_tree(source)
    if not source.is_absolute() or source != source.resolve() or sha256_file(source) != provenance["sha256"]:
        raise SpecError("public scalability provenance changed")
    actual, members, names, groups, sources = _members(path.parent), {}, set(), {}, {}
    for row in doc["members"]:
        if (
            type(row) is not dict
            or set(row) != {"member", "sha256", "source_group", "split"}
            or type(row["member"]) is not str
        ):
            raise SpecError("public scalability member pin is malformed")
        member, digest = row["member"], row["sha256"]
        group, split = row["source_group"], row["split"]
        if not is_sha256(group) or split not in {"public", "train"} or groups.setdefault(group, split) != split:
            raise SpecError("public scalability source variants must share one public/train source split")
        relative = Path(member)
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.as_posix() != member
            or any(p in {"", ".", "..", "hidden"} or p.startswith(("_", ".")) for p in relative.parts)
            or member in members
            or member not in actual
            or not is_sha256(digest)
        ):
            raise SpecError("public scalability requires unique public category/member pins")
        directory, cap = actual[member]
        if (
            cap.get("label") not in {"public", "dev"}
            or cap.get("kind") == "model"
            or cap.get("name") != directory.name
            or cap["name"] in names
            or cap.get("required_oracle_tiers") != POLICY["tiers"]
            or cap.get("max_oracle_tier") is not None
            or cap.get("extends") is not None
            or cap.get("performance") is not None
            or (cap.get("expected") or {}).get("behavior", "compute") != "compute"
        ):
            raise SpecError(
                "scalability requires positive native public contracts at exactly L0/L1/L2, no timing policy"
            )
        if fingerprint(directory) != digest:
            raise SpecError("public scalability source/input/reference bytes changed")
        ir = cap.get("interface_mlir") or cap.get("linalg_mlir")
        if not isinstance(ir, str) or Path(ir).suffix != ".mlir" or Path(ir).is_absolute() or ".." in Path(ir).parts:
            raise SpecError("public scalability needs capsule-local mathematical IR")
        ordinary_tree(directory / ir)
        source_sha = sha256_file(directory / ir)
        if sources.setdefault(source_sha, group) != group:
            raise SpecError("identical public mathematical sources cannot cross source groups")
        members[member] = {"sha256": digest, "source_group": group, "split": split}
        names.add(cap["name"])
    if set(members) != set(actual) or path.read_bytes() != raw:
        raise SpecError("public scalability selection omits cases or changed while reading")
    return {
        "schema": SCHEMA,
        "target": target,
        "policy": copy.deepcopy(POLICY),
        "selection": {"path": str(path), "sha256": sha256_bytes(raw)},
        "provenance": provenance,
        "members": members,
    }


def stage(path: Path, destination: Path, *, te, private_roots=()) -> dict:
    from merlin.targetgen.capsule_common import load_capsule
    from merlin.targetgen.sandbox.bwrap import _validate_host_sources

    from ..runner import fingerprint
    from .preparation import _members, copy_input, ordinary_tree
    from .publication import require_accelerator_placement, require_public_capsules

    record = read(path, target=te.target)
    source = Path(record["selection"]["path"]).parent
    directories = [source / member for member in record["members"]]
    _validate_host_sources([(str(p), p) for p in directories], [(str(p), p) for p in private_roots])
    require_public_capsules(te, directories)
    require_accelerator_placement(directories)
    if destination.exists() or destination.is_symlink():
        raise SpecError("public scalability requires a fresh release directory")
    destination.mkdir(mode=0o700)
    cases, public_cases, public = {}, [], destination / "public_sources"
    public.mkdir()
    actual = _members(source)
    for member, lineage in record["members"].items():
        digest = lineage["sha256"]
        directory, cap = actual[member]
        load_capsule(directory)  # Original schema, capture-tool custody and complete declared policy.
        name = cap["name"]
        copy_input(directory, destination / "cases" / name, expected_sha256=digest, private=True)
        # Only the declared mathematical IR is public. Goldens, arrays and producer
        # metadata remain in the private cases, never a broad source-directory grant.
        ir = cap.get("interface_mlir") or cap.get("linalg_mlir")
        if not isinstance(ir, str) or Path(ir).suffix != ".mlir" or Path(ir).is_absolute() or ".." in Path(ir).parts:
            raise SpecError("public scalability needs capsule-local mathematical IR")
        ordinary_tree(directory / ir)
        copy_input(directory / ir, public / f"{name}.mlir")
        cases[name] = digest
        public_cases.append(
            {
                "name": name,
                "source": f"{name}.mlir",
                "source_group": lineage["source_group"],
                "split": lineage["split"],
                "operation": cap.get("operation"),
                "numeric_policy": cap.get("numeric_policy"),
            }
        )
    summary = {"schema": SCHEMA, "policy": POLICY, "cases": public_cases}
    (public / "suite.json").write_text(json.dumps(summary, indent=2) + "\n")
    staged = {"schema": STAGED, "source": record, "cases": cases, "public_sources_sha256": fingerprint(public)}
    (destination / "selection.json").write_text(json.dumps(staged, indent=2) + "\n")
    verify(destination, target=te.target)
    return {"path": str(destination), "sha256": fingerprint(destination)}


def verify(root: Path, *, target: str) -> dict:
    from ..runner import fingerprint
    from .preparation import ordinary_tree

    ordinary_tree(root)
    record = loads((root / "selection.json").read_bytes(), max_bytes=2 * 1024 * 1024)
    if (
        type(record) is not dict
        or set(record) != {"schema", "source", "cases", "public_sources_sha256"}
        or record["schema"] != STAGED
        or read(Path(record["source"]["selection"]["path"]), target=target) != record["source"]
    ):
        raise SpecError("public scalability staged selection changed")
    actual = {p.name: fingerprint(p) for p in (root / "cases").iterdir() if p.is_dir()}
    if (
        actual != record["cases"]
        or not actual
        or fingerprint(root / "public_sources") != record["public_sources_sha256"]
    ):
        raise SpecError("public scalability complete roster/input/reference/source view changed")
    return record


def selected(te) -> tuple[Path, dict] | None:
    from ..runner import fingerprint

    binding = (getattr(te, "workload_spec", None) or {}).get(KEY)
    if binding is None:
        return None
    if type(binding) is not dict or set(binding) != {"path", "sha256"} or not is_sha256(binding["sha256"]):
        raise SpecError("public scalability descriptor binding is malformed")
    root = Path(binding["path"])
    if not root.is_absolute() or root != root.resolve() or fingerprint(root) != binding["sha256"]:
        raise SpecError("public scalability selected release bytes changed")
    return root, verify(root, target=te.target)
