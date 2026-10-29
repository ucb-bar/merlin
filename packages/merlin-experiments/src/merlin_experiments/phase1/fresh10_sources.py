"""Reviewed fresh10 source/library projections, never ambient import discovery."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import fresh10_profiles as P

DATA_FILE = "fresh10_profile_data.json"
# Set only when the reviewed inventory is materialized; an unfinished selection
# must fail closed instead of accepting an arbitrary adjacent JSON document.
DATA_SHA256 = "be73e4e32be7d362eefd1065e23aeff242bed3520403eb83ca06b3aaaa59bb33"


def _data():
    path = Path(__file__).with_name(DATA_FILE)
    if path.is_symlink():
        raise ValueError("fresh10 projection inventory must be an original regular resource")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("fresh10 reviewed projection inventory changed or is incomplete")
    data = json.loads(raw)
    if data.get("schema") != "merlin.fresh10_tooling_projection.v1" or data.get("baseline_revision") != P.BASE_REVISION:
        raise ValueError("fresh10 projection belongs to another source baseline")
    return data


def _arm(name):
    P.require_preparable(name)
    return {P.NAMES[0]: "A", P.NAMES[1]: "B", P.NAMES[2]: "C"}[name]


def _rows(name, kind):
    return _data()["profiles"][_arm(name)][kind + "_members"]


def entry_modules(name):
    return tuple(_data()["profiles"][_arm(name)]["entry_modules"])


def _diagnostics(name):
    data = _data()["profiles"][_arm(name)]
    return tuple(module + ": " + reason for module, reason in sorted(data["unavailable_entry_modules"].items())) + tuple(data["unavailable_api_routes"])


def _exact_files(root, rows):
    from . import full_compiler_workspace as F

    root = Path(root)
    expected = {row["path"]: row["sha256"] for row in rows}
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("fresh10 selected source contains an indirect path")
        if path.is_file():
            actual[path.relative_to(root).as_posix()] = hashlib.sha256(F._read(root, path.relative_to(root).as_posix())).hexdigest()
    if actual != expected:
        raise ValueError("fresh10 selected source membership or bytes differ from the reviewed projection")


def _payload(repo, row, *, library_root=None):
    from . import full_compiler_workspace as F

    data = _data()
    relative = row["path"]
    candidate = None if library_root is None else Path(library_root) / relative
    if candidate is not None and candidate.is_file():
        payload = F._read(Path(library_root), relative)
    else:
        payload = F._read(Path(repo), row.get("source_relative", "src/" + relative))
    digest = hashlib.sha256(payload).hexdigest()
    permitted = {row["baseline_sha256"], row["sha256"], *row.get("donor_sha256", ())}
    if digest not in permitted:
        raise ValueError("fresh10 source revision drift before projection: " + relative)
    if row.get("projection") == "reviewed_f9_sdk_namespace":
        payload = data["namespace_facades"][relative].encode()
    elif row.get("projection") == "reviewed_common_public_schema_api":
        payload = F._read(Path(repo) / "src", relative)
    elif hashlib.sha256(payload).hexdigest() != row["sha256"]:
        # Original readonly SDK namespace facades may differ from the original
        # full compiler namespace. The reviewed union fixes donor priority;
        # reopening a different approved donor cannot change that selection.
        payload = F._read(Path(repo), row.get("source_relative", "src/" + relative))
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError("fresh10 source projection has unexpected bytes: " + relative)
    return payload


def prepare_sources(repo, destination, profile_name, review_id=""):
    from . import full_compiler_workspace as F

    rows = _rows(profile_name, "full")
    destination = Path(destination).absolute()
    if destination.exists() or not review_id:
        raise ValueError("fresh10 source preparation requires a new destination and explicit review")
    # Read and check every byte before creating any selected output.
    payloads = [(row, _payload(repo, row)) for row in rows]
    root = destination / "python"
    for row, payload in payloads:
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o444)
    data = dict(schema=F.SCHEMA, review_id=review_id,
                members=[{key: row[key] for key in ("path", "sha256", "module")} for row in rows],
                source_roots=["reviewed-fresh10-public-member-projection"],
                mutable_namespaces=list(F.MUTABLE_NAMESPACES),
                unavailable_dependencies=list(_diagnostics(profile_name)),
                entry_modules=list(entry_modules(profile_name)),
                fresh10_projection=dict(profile=profile_name, baseline_revision=P.BASE_REVISION,
                                        inventory_sha256=DATA_SHA256),
                scope=("reviewed fresh10 general compiler and transformation library; no CCA or reference-kernel material"
                       if P.resolve(profile_name).transform_library else
                       "reviewed retained fresh10 infrastructure; no supplied optimized transformation library or CCA"))
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    manifest.chmod(0o444)
    selected = F.select(root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())
    verify_sources(profile_name, selected)
    return selected


def verify_sources(profile_name, selection):
    from . import full_compiler_workspace as F

    if type(selection) is not F.FullCompilerSelection:
        raise ValueError("fresh10 profile requires its original reviewed source selection")
    rows = _rows(profile_name, "full")
    selection.verify()
    document = json.loads(selection.manifest_path.read_bytes())
    if document.get("fresh10_projection") != dict(profile=profile_name, baseline_revision=P.BASE_REVISION,
                                                   inventory_sha256=DATA_SHA256):
        raise ValueError("fresh10 source selection belongs to another capability projection")
    if document.get("entry_modules") != list(entry_modules(profile_name)):
        raise ValueError("fresh10 source entry inventory differs")
    expected = tuple((row["path"], row["sha256"], row["module"]) for row in rows)
    diagnostics = _diagnostics(profile_name)
    if selection.members != expected or selection.diagnostics != diagnostics:
        raise ValueError("fresh10 source inventory differs from its reviewed capability boundary")
    _exact_files(selection.root, rows)


def project_author_library(author_library_dir, profile_name, *, repo):
    """Replace the selected readonly SDK, retaining the original preparation host-only."""
    from merlin.targetgen.compiler_library import CompilerLibraryContract, CompilerLibrarySelection, LibraryMember

    arm = _arm(profile_name)
    directory = Path(author_library_dir).absolute()
    path = directory / "compiler-library-selection.json"
    original = json.loads(path.read_bytes())
    data = _data()
    original_rows = original["contract"]["members"]
    expected = {r["path"]: r for r in data["baseline_sdk_members"]}
    if {r["path"] for r in original_rows} != set(expected):
        raise ValueError("fresh10 prepared SDK differs from the original baseline membership")
    for row in original_rows:
        old = expected[row["path"]]
        repair = data["common_patches"].get(row["path"], {})
        if row["module"] != old["module"] or row["sha256"] not in {old["sha256"], repair.get("sha256")}:
            raise ValueError("fresh10 prepared SDK has unreviewed source changes")
    rows = _rows(profile_name, "sdk")
    payloads = [(row, _payload(repo, row, library_root=original["root"])) for row in rows]
    root = directory / "profile-library" / "python"
    if root.parent.exists():
        raise ValueError("fresh10 filtered SDK needs a new destination")
    for row, payload in payloads:
        target = root / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(0o444)
    contract = CompilerLibraryContract("fresh10-capability-projection:" + profile_name,
        tuple(data["profiles"][arm]["sdk_public_modules"]),
        tuple(LibraryMember(row["path"], row["sha256"], row["module"]) for row in rows))
    selection = CompilerLibrarySelection(contract, root)
    verify_library(profile_name, selection)
    contract_path = root.parent / "contract.json"
    contract_path.write_text(json.dumps(contract.record(), indent=2, sort_keys=True) + "\n")
    contract_path.chmod(0o444)
    # The ordinary planner and library binder already consume this exact owner.
    path.write_text(json.dumps(selection.record(), indent=2, sort_keys=True) + "\n")
    return dict(root=str(root), contract_sha256=contract.sha256, members=len(rows),
                public_modules=len(contract.public_modules), profile=profile_name,
                selection_path=str(path), contract_path=str(contract_path))


def verify_library(profile_name, selection):
    from merlin.targetgen.compiler_library import CompilerLibrarySelection

    if type(selection) is not CompilerLibrarySelection:
        raise ValueError("fresh10 profile requires an explicitly selected retained SDK")
    selected = _data()["profiles"][_arm(profile_name)]
    rows = selected["sdk_members"]
    selection.record()
    if tuple(selection.contract.public_modules) != tuple(selected["sdk_public_modules"]):
        raise ValueError("fresh10 additional SDK public API inventory differs")
    actual = [dict(path=m.path, sha256=m.sha256, module=m.module) for m in selection.contract.members]
    if actual != [{k: row[k] for k in ("path", "sha256", "module")} for row in rows]:
        raise ValueError("fresh10 additional SDK includes unreviewed capability members")
    _exact_files(selection.root, rows)


def common_allowed(bundle, profile_name):
    """Retain fresh10 public data while moving every original SDK source alias host-only."""
    from .author_profiles import _source_package

    profile = P.resolve(profile_name)
    rows = []
    for row in bundle.get("allowed", ()):
        path = row.get("path", "")
        framework_source = ("merlin/python/merlin/" in path or "merlin/python/merlin_experiments/" in path
                            or "/author-library/base-sdk/" in path or "/author-library/generic-library/" in path
                            or _source_package(path))
        # A run ID can itself contain 'no-reference-kernels'. Only actual
        # container/file components identify reference material; matching an
        # ancestor's substring would also withhold that run's public ABI data.
        components = set(Path(path).parts)
        reference = bool(components & {"reference-kernels", "public-references", "REFERENCE_MAP.json"})
        timing_target = bool(components & {"performance_baselines", "performance-baselines",
                                           "expert_targets", "expert-targets", "SEED_PERFORMANCE"})
        if not framework_source and not timing_target and (profile.raw_kernel_references or not reference):
            rows.append(copy.deepcopy(row))
    if not rows:
        raise ValueError("fresh10 common public-data selection is empty")
    return rows
