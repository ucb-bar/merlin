#!/usr/bin/env python3
"""Verify preserved kernel images and replay one through an explicitly selected wrapper.

Planning and validation never compile code, contact a service or launch hardware.
A successful wrapper exit is transport status, not a numerical or timing verdict.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.paper-kernel-replay.v1"
IGNORED_SOURCE_PARTS = frozenset({"build", "__pycache__", ".git"})


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def relative(root: Path, name: str) -> Path:
    parts = PurePosixPath(name)
    if not name or parts.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")) or "\\" in name:
        raise ValueError(f"unsafe artifact path: {name!r}")
    path = root.joinpath(*parts.parts)
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError(f"artifact path follows a symlink: {item}")
    return path


def pinned(root: Path, record: dict) -> Path:
    path = relative(root, record["path"])
    digest = record.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"missing exact SHA-256 for {path}")
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"missing or changed artifact: {path}")
    return path


def source_sha(root: Path) -> str:
    """The original compiler inventory order and exclusions, not a new hash scheme."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or IGNORED_SOURCE_PARTS.intersection(rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"compiler source contains an indirect file: {path}")
        digest.update(rel.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def configuration(path: Path) -> dict:
    config = read(path)
    if config.get("schema") != SCHEMA or not isinstance(config.get("images"), dict):
        raise ValueError("unsupported paper replay configuration")
    return config


def _objects(root: Path, config: dict, system: str, case: str, expected: str) -> Path:
    candidates = [relative(root, name) / system / case / "kernel.o" for name in config["object_roots"]]
    found = [path for path in candidates if path.is_file()]
    if not found or any(sha(path) != expected for path in found):
        raise ValueError(f"missing or conflicting consumed object: {system}/{case}")
    return found[0]


def _batch_inputs(root: Path, config: dict, image: dict, manifest: dict) -> dict:
    product = relative(root, config["artifact_root"])
    catalog_path = pinned(root, config["cases"])
    study_path = pinned(root, config["study"])
    catalog = {row["case_id"]: row for row in read(catalog_path)["cases"]}
    sources = {row["label"]: row for row in read(study_path)["sources"]}
    verified_sources, verified_cases, pairs = set(), set(), set()
    for slot in manifest["slots"]:
        system, case_id = slot["system"], slot["case_id"]
        if (system, case_id) in pairs:
            raise ValueError("image repeats a system/case slot")
        pairs.add((system, case_id))
        result, case = slot["compiler_result"], catalog[case_id]
        if result.get("status") != "compiled" or result.get("kernel_sha256") != slot["kernel_o_sha256"]:
            raise ValueError(f"consumed object differs from compiler result: {system}/{case_id}")
        if result.get("interface_sha256") != case["interface_sha256"]:
            raise ValueError(f"compiled interface differs from selected case: {system}/{case_id}")
        _objects(root, config, system, case_id, slot["kernel_o_sha256"])
        if system in sources and system not in verified_sources:
            source = sources[system]
            path = product / "sources" / system
            if source["expected"] != result["source_sha256"] or source_sha(path) != source["expected"]:
                raise ValueError(f"compiler source differs from consumed result: {system}")
            verified_sources.add(system)
        if case_id not in verified_cases:
            for tensor in case["tensors"].values():
                name = PurePosixPath(tensor["file"]).name
                path = product / "fixtures" / case_id / name
                if not path.is_file() or sha(path) != tensor["sha256"]:
                    raise ValueError(f"missing or changed tensor fixture: {case_id}/{name}")
            interface = product / "fixtures" / case_id / "input.mlir"
            if not interface.is_file() or sha(interface) != case["interface_sha256"]:
                raise ValueError(f"missing or changed interface: {case_id}")
            verified_cases.add(case_id)
    directory = relative(root, image["directory"])
    for name, expected in manifest["runtime_sources"].items():
        path = directory / ("image-" + name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"missing or changed linked runtime source: {path}")
    for name, field in (("image-harness.c", "harness_sha256"), ("image-link.ld", "linker_sha256")):
        if sha(directory / name) != manifest[field]:
            raise ValueError(f"changed image build input: {name}")
    return {
        "bound_objects": len(pairs),
        "bound_compiler_sources": sorted(verified_sources),
        "bound_cases": len(verified_cases),
    }


def _paired_inputs(root: Path, image: dict, manifest: dict) -> dict:
    directory = relative(root, image["directory"])
    for name, expected in manifest["kernel_object_sha256"].items():
        if sha(relative(directory, name)) != expected:
            raise ValueError(f"changed paired-control object: {name}")
    if sha(directory / "harness.c") != manifest["harness_sha256"]:
        raise ValueError("changed paired-control harness")
    for header in manifest["upstream_headers"]:
        if sha(relative(directory, header["path"])) != header["sha256"]:
            raise ValueError(f"changed control-library header: {header['path']}")
    return {
        "bound_objects": len(manifest["kernel_object_sha256"]),
        "bound_cases": manifest["cases"],
        "calling_protocol": "two opposite-order passes in one boot",
        "fsm_commit": manifest["fsm_upstream_commit"],
    }


def validate(root: Path, config: dict, key: str, *, _seen: frozenset[str] = frozenset()) -> dict:
    if key in _seen:
        raise ValueError("cyclic source-image reference")
    _seen = _seen | {key}
    image = config["images"][key]
    manifest_path = pinned(root, image["manifest"])
    elf = pinned(root, image["elf"])
    manifest = read(manifest_path)
    if manifest.get("elf_sha256") != image["elf"]["sha256"]:
        raise ValueError("ELF digest differs between replay selection and image manifest")
    if elf.read_bytes()[:4] != b"\x7fELF":
        raise ValueError("selected image is not an ELF")
    if manifest.get("schema") == "firesim-controlled-batch.v1":
        bindings = _batch_inputs(root, config, image, manifest)
    elif manifest.get("schema") == "fsm-paired-rerun.v1":
        bindings = _paired_inputs(root, image, manifest)
        parent = config["images"][image["source_image"]]
        if manifest["original_elf_sha256"] != parent["elf"]["sha256"]:
            raise ValueError("paired control names another original compiler image")
        validate(root, config, image["source_image"], _seen=_seen)
    else:
        raise ValueError("unsupported preserved image manifest")
    hardware = read(pinned(root, image["hardware"]))
    # A construction-stage manifest may still say built_not_executed. Preserve it.
    # Historical execution has its own separately pinned evidence.
    summary_path = pinned(root, image["hardware_summary"])
    summary = dict(line.split(": ", 1) for line in summary_path.read_text().splitlines() if ": " in line)
    if summary.get("AGFI") != hardware["hardware"].get("agfi") or summary.get("RuntimeHWConfig") != hardware[
        "hardware"
    ].get("runtime_hw_config"):
        raise ValueError("observed hardware summary differs from the selected configuration")
    historical = read(pinned(root, image["historical_execution"]))
    if historical.get("elf_sha256") != image["elf"]["sha256"]:
        raise ValueError("historical execution summary names another ELF")
    return {
        "image": key,
        "elf": str(elf),
        "elf_sha256": image["elf"]["sha256"],
        "manifest": str(manifest_path),
        "inputs_verified": True,
        **bindings,
        "required_hardware": hardware["hardware"],
        "historical_execution_preserved": True,
        "new_execution": False,
        "new_numerical_verdict": "not_executed",
    }


def wrapper_selection(path: Path, expected: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or not os.access(path, os.X_OK) or sha(path) != expected:
        raise ValueError("selected wrapper is missing, not executable, indirect or changed")
    return path


def replay(
    root: Path, config: dict, key: str, *, wrapper: Path, wrapper_sha256: str, platform: Path, output: Path
) -> dict:
    verified = validate(root, config, key)
    wrapper = wrapper_selection(wrapper, wrapper_sha256)
    selected = read(platform)
    required = verified["required_hardware"]
    for field in ("agfi", "runtime_hw_config"):
        if not required.get(field) or selected.get(field) != required[field]:
            raise ValueError(f"explicit platform selection differs from the measured {field}")
    # The wrapper owns its one-job lock and all manager/F2 operations.
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "image.elf"
    shutil.copyfile(verified["elf"], frozen)
    if sha(frozen) != verified["elf_sha256"]:
        raise ValueError("ELF changed while staging replay")
    frozen.chmod(0o444)
    argv = [str(wrapper), "-n", key, str(frozen)]
    receipt = {
        **verified,
        "argv": argv,
        "wrapper_sha256": wrapper_sha256,
        "platform_selection_sha256": sha(platform),
        "platform_selection_is_execution_evidence": False,
        "new_numerical_verdict": "awaiting_full_uart_and_actual_hardware_checks",
    }
    (output / "invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (output / "wrapper.stdout").open("wb") as stdout, (output / "wrapper.stderr").open("wb") as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
    receipt.update(
        wrapper_invoked=True,
        hardware_execution_verified=False,
        wrapper_exit_status=result.returncode,
        status="wrapper_completed" if result.returncode == 0 else "wrapper_failed",
    )
    if sha(frozen) != verified["elf_sha256"] or sha(wrapper) != wrapper_sha256:
        receipt["status"] = "input_changed"
    (output / "replay.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def dependency_inventory(root: Path, mount: dict, metadata_root: Path | None = None) -> dict:
    """Check exact restored membership, modes, file bytes and symlink spellings."""
    inventory_pin = mount["inventory"]
    if metadata_root is not None:
        inventory_pin = {**inventory_pin, "path": Path(inventory_pin["path"]).name}
    document = read(pinned(metadata_root or root, inventory_pin))
    if document.get("schema") != "merlin.paper_dependency_inventory.v1":
        raise ValueError("unsupported dependency inventory")
    members = document["members"]
    encoded = json.dumps(members, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if hashlib.sha256(encoded).hexdigest() != document["tree_sha256"]:
        raise ValueError("dependency inventory tree digest changed")
    tree = relative(root, mount["source"])
    expected = {row["path"]: row for row in members}
    if len(expected) != len(members):
        raise ValueError("duplicate dependency inventory member")
    actual = set()
    for base, directories, files in os.walk(tree, followlinks=False):
        for name in (*directories, *files):
            path = Path(base) / name
            name = path.relative_to(tree).as_posix()
            actual.add(name)
            row = expected.get(name)
            if row is None:
                raise ValueError(f"unrecorded dependency member: {mount['source']}/{name}")
            # Validate the lexical spelling without following the member itself.
            if PurePosixPath(name).is_absolute() or any(part in {"", ".", ".."} for part in name.split("/")):
                raise ValueError("unsafe dependency inventory member")
            observed = path.lstat()
            if observed.st_mode & 0o7777 != int(row["mode"], 8):
                raise ValueError(f"dependency mode changed: {name}")
            if path.is_symlink():
                if row["kind"] != "symlink" or os.readlink(path) != row["target"]:
                    raise ValueError(f"dependency symlink changed: {name}")
            elif path.is_dir():
                if row["kind"] != "directory":
                    raise ValueError(f"dependency directory changed: {name}")
            elif path.is_file():
                if row["kind"] != "file" or observed.st_size != row["bytes"] or sha(path) != row["sha256"]:
                    raise ValueError(f"dependency file changed: {name}")
            else:
                raise ValueError(f"unsupported dependency member kind: {name}")
    if actual != set(expected):
        raise ValueError(f"missing dependency members: {mount['source']}")
    return {"tree": str(tree), "entries": expected, "tree_sha256": document["tree_sha256"]}


def _normal_path(path: Path) -> Path:
    parts = []
    for part in path.parts:
        if part in {"/", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError("dependency symlink escapes absolute root")
            parts.pop()
        else:
            parts.append(part)
    return Path("/", *parts)


class MissingDependencyTarget(ValueError):
    """A preserved link names an absent member inside a selected mount."""


def resolve_dependency(path: Path, selected: list[dict]) -> tuple[dict, str]:
    """Resolve links only in the declared virtual mount inventory, never the host."""
    path, seen = _normal_path(path), set()
    for _ in range(128):
        key = str(path)
        if key in seen:
            raise ValueError("dependency symlink cycle")
        seen.add(key)
        choices = [row for row in selected if path.is_relative_to(Path(row["destination"]))]
        if not choices:
            raise ValueError(f"dependency points outside selected mounts: {path}")
        chosen = max(choices, key=lambda row: len(Path(row["destination"]).parts))
        destination = Path(chosen["destination"])
        relative_parts = path.relative_to(destination).parts
        if not relative_parts:
            return chosen, ""
        for index in range(1, len(relative_parts) + 1):
            prefix = PurePosixPath(*relative_parts[:index]).as_posix()
            entry = chosen["inventory_data"]["entries"].get(prefix)
            if entry is None:
                raise MissingDependencyTarget(f"missing dependency target in selected inventory: {path}")
            if entry["kind"] == "symlink":
                target = Path(entry["target"])
                target = target if target.is_absolute() else (destination / prefix).parent / target
                path = _normal_path(target.joinpath(*relative_parts[index:]))
                break
        else:
            return chosen, PurePosixPath(*relative_parts).as_posix()
    raise ValueError("dependency symlink resolution budget exceeded")


def check_dependency_links(selected: list[dict]) -> list[dict]:
    """Report preserved absent optional targets, while refusing escapes and cycles.

    Selected executables still require complete resolution in verify_recorded_tools.
    Installed but unused optional runtime links need not be synthesized or repaired.
    """
    dangling = []
    for mount in selected:
        for name, row in mount["inventory_data"]["entries"].items():
            if row["kind"] == "symlink":
                path = Path(mount["destination"]) / name
                try:
                    resolve_dependency(path, selected)
                except MissingDependencyTarget as error:
                    dangling.append({"path": str(path), "target": row["target"], "reason": str(error)})
    return dangling


def check_python_search_paths(selected: list[dict]) -> list[dict]:
    """Record preserved tool startup hooks and paths absent from the original namespace.

    Hooks are pinned toolchain code, not candidate code. Do not mount a current
    framework just to make an intentionally inaccessible editable path resolve.
    The separate startup probe checks the modules that these hooks actually load.
    """
    records = []
    for mount in selected:
        if mount.get("recorded_name") != "RUNTIME":
            continue
        for name, row in mount["inventory_data"]["entries"].items():
            if row["kind"] != "file" or not name.endswith((".pth", ".egg-link")):
                continue
            path = Path(mount["source"]) / name
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("import ", "import\t")):
                    records.append(
                        {"file": name, "sha256": row["sha256"], "kind": "preserved_tool_startup_hook", "line": line}
                    )
                    continue
                target = Path(line)
                parent = (Path(mount["destination"]) / name).parent
                logical = _normal_path(target if target.is_absolute() else parent / target)
                choices = [item for item in selected if logical.is_relative_to(Path(item["destination"]))]
                if choices:
                    resolve_dependency(logical, selected)
                records.append(
                    {
                        "file": name,
                        "sha256": row["sha256"],
                        "path": str(logical),
                        "kind": "selected_path" if choices else "inaccessible_in_original_namespace",
                    }
                )
    return records


def verify_recorded_tools(root: Path, config: dict, selected: list[dict]) -> list[dict]:
    tools = read(pinned(root, config["rebuild"]["toolchains"]))
    verified = []
    for name, record in tools.items():
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            continue
        mount, relative_name = resolve_dependency(Path(record["path"]), selected)
        entry = mount["inventory_data"]["entries"].get(relative_name)
        if entry is None or entry["kind"] != "file" or entry["sha256"] != record["sha256"]:
            raise ValueError(f"selected tool differs from historical executable: {name}")
        verified.append({"tool": name, "sha256": record["sha256"]})
    if not verified:
        raise ValueError("no historical tool executable hashes verified")
    headers = config["rebuild"].get("vendor_headers", [])
    vendors = [mount for mount in selected if mount.get("recorded_name") == "VENDOR"]
    if headers and len(vendors) != 1:
        raise ValueError("runtime header inventory has no unique vendor mount")
    for header in headers:
        mount, name = resolve_dependency(Path(vendors[0]["destination"]) / header["path"], selected)
        row = mount["inventory_data"]["entries"].get(name)
        if row is None or row.get("sha256") != header["sha256"]:
            raise ValueError(f"selected vendor header differs from preserved input: {header['path']}")
    return verified


def recorded_paths(script: Path) -> dict[str, str]:
    """Read literal Path declarations from pinned source without importing it."""
    tree = ast.parse(script.read_text())
    selected = {}
    all_paths = set()

    def literal(node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Path"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and Path(node.args[0].value).is_absolute()
        ):
            return node.args[0].value
        return None

    for node in ast.walk(tree):
        value = literal(node)
        if value is not None:
            all_paths.add(value)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = literal(node.value)
            if value is not None:
                selected[node.targets[0].id] = value
    unnamed = all_paths - set(selected.values())
    if len(unnamed) == 1:
        selected["UNNAMED_PYTHON_ROOT"] = unnamed.pop()
    return selected


def rebuild_plan(
    root: Path,
    config: dict,
    system: str,
    case: str,
    workspace: Path,
    dependency_root: Path | None = None,
    metadata_root: Path | None = None,
) -> dict:
    """Describe the original scripts' real namespace requirements, without changing them."""
    dependency_root = dependency_root or root
    recipe = config["rebuild"]
    script = pinned(root, recipe["compile_script"])
    batch = pinned(root, recipe["batch_script"])
    study_path = pinned(root, config["study"])
    script_study = pinned(root, recipe["script_study"])
    study = read(study_path)
    sources = {row["label"]: row for row in study["sources"]}
    source = sources[system]
    catalog = {row["case_id"]: row for row in read(pinned(root, config["cases"]))["cases"]}
    selected_case = catalog[case]
    compiler = relative(root, config["artifact_root"]) / "sources" / system
    if source_sha(compiler) != source["expected"]:
        raise ValueError("selected rebuild compiler source differs from the frozen snapshot")
    fixture = relative(root, config["artifact_root"]) / "fixtures" / case / "input.mlir"
    if sha(fixture) != selected_case["interface_sha256"]:
        raise ValueError("selected rebuild interface differs from the frozen case")
    recorded = {"compile_script": recorded_paths(script), "batch_script": recorded_paths(batch)}
    mounts, missing = [], []
    library = recipe.get("source_libraries", {}).get(system)
    library_mount = None
    if library is None:
        missing.append({"path": system, "reason": "compiler_library_not_selected"})
    else:
        library_root = relative(root, library["path"])
        if source_sha(library_root) != library["source_sha256"]:
            raise ValueError("selected compiler library differs from preserved bytes")
        library_destination = source.get("generic_library") or str(
            Path(source["framework"]) / library["framework_relative"]
        )
        library_mount = {"source": str(library_root), "destination": library_destination}
    for item in recipe["dependency_mounts"]:
        host = relative(dependency_root, item["source"])
        selector = item["destination_from"]
        destination = Path(recorded[selector["script"]][selector["name"]])
        if not destination.is_absolute() or destination == Path("/"):
            raise ValueError("dependency mount needs a bounded absolute destination")
        mounts.append(
            {
                "source": str(host),
                "destination": str(destination),
                "access": "read_only",
                "recorded_name": selector["name"],
            }
        )
        if not host.is_dir():
            missing.append({"path": item["source"], "reason": "dependency_tree_not_restored"})
        # Presence alone is not a complete dependency pin.
        if not item.get("inventory"):
            missing.append({"path": item["source"], "reason": "complete_verified_inventory_not_supplied"})
        elif host.is_dir():
            mounts[-1]["inventory_data"] = dependency_inventory(dependency_root, item, metadata_root)
    llvm, runtime = recorded["compile_script"]["LLVM"], recorded["compile_script"]["RUNTIME"]
    argv = ["bwrap", "--die-with-parent", "--new-session", "--unshare-net", "--unshare-pid", "--clearenv"]
    for name in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        argv += ["--ro-bind", name, name]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    product = relative(root, config["artifact_root"])
    argv += ["--ro-bind", str(product), study["artifact_dir"]]
    for mount in mounts:
        argv += ["--ro-bind", mount["source"], mount["destination"]]
    if library_mount is not None:
        argv += ["--ro-bind", library_mount["source"], library_mount["destination"]]
    argv += [
        "--ro-bind",
        str(study_path),
        study["artifact_dir"] + "/scripts/study.json",
        "--bind",
        str(workspace.absolute()),
        study["build_dir"],
        "--setenv",
        "PATH",
        runtime + "/bin:" + llvm + "/bin:/usr/bin:/bin",
        runtime + "/bin/python3.12",
        study["artifact_dir"] + "/scripts/" + script.name,
        system,
        case,
    ]
    verified_tools, startup_paths, dangling_links = [], [], []
    if not missing:
        dangling_links = check_dependency_links(mounts)
        startup_paths = check_python_search_paths(mounts)
        verified_tools = verify_recorded_tools(root, config, mounts)
    return {
        "system": system,
        "case": case,
        "source_sha256": source["expected"],
        "compile_script": str(script),
        "batch_script": str(batch),
        "argv": argv,
        "selected_study": str(study_path),
        "historical_script_study": str(script_study),
        "study_overlay_explicit": study_path.read_bytes() != script_study.read_bytes(),
        "dependency_mounts": [
            {key: value for key, value in mount.items() if key != "inventory_data"} for mount in mounts
        ],
        "verified_tools": verified_tools,
        "python_startup_paths": startup_paths,
        "preserved_dangling_dependency_links": dangling_links,
        "full_installed_tree_link_closure": not dangling_links if not missing else False,
        "dependency_scope": "selected_executables_and_pinned_installed_bytes; imports_require_separate_probe",
        "missing": missing,
        "ready_for_source_rebuild": not missing,
        "historical_python_runtime_identity_established": False,
        "reason": "new source build requires subsequent object and numerical comparison",
        "execution": "not_executed",
    }


def startup_probe(
    root: Path,
    config: dict,
    system: str,
    case: str,
    *,
    workspace: Path,
    bwrap: Path,
    bwrap_sha256: str,
    dependency_root: Path | None = None,
    metadata_root: Path | None = None,
) -> dict:
    """Import interpreter dependencies in the compiler namespace, never the compiler."""
    plan = rebuild_plan(root, config, system, case, workspace, dependency_root, metadata_root)
    if not plan["ready_for_source_rebuild"]:
        raise ValueError("startup probe requires verified dependency inventories")
    bwrap = wrapper_selection(bwrap, bwrap_sha256)
    product = relative(root, config["artifact_root"])
    source = next(row for row in read(pinned(root, config["study"]))["sources"] if row["label"] == system)
    pkg = source["build_source"]
    library = config["rebuild"]["source_libraries"][system]
    library_root = relative(root, library["path"])
    library_dest = source.get("generic_library") or str(Path(source["framework"]) / library["framework_relative"])
    runtime = next(m["destination"] for m in plan["dependency_mounts"] if m["recorded_name"] == "RUNTIME")
    llvm = next(m["destination"] for m in plan["dependency_mounts"] if m["recorded_name"] == "LLVM")
    conda = next(m["destination"] for m in plan["dependency_mounts"] if m["recorded_name"] == "CONDA")
    allowed = [
        "/usr",
        "/bin",
        "/lib",
        "/lib64",
        "/etc",
        pkg,
        library_dest,
        *(m["destination"] for m in plan["dependency_mounts"]),
    ]
    program = (
        "import sys,json,xdsl;from pathlib import Path;"
        f"allowed={allowed!r};"
        "origins={n:getattr(m,'__file__',None) for n,m in sorted(sys.modules.items())};"
        "outside={n:p for n,p in origins.items() if isinstance(p,str) and Path(p).is_absolute() "
        "and not any(Path(p).is_relative_to(Path(a)) for a in allowed)};"
        "print(json.dumps({'python':sys.version,'prefix':sys.prefix,'origins':origins,'outside':outside},sort_keys=True));"
        "sys.exit(bool(outside))"
    )
    argv = [str(bwrap), "--die-with-parent", "--new-session", "--unshare-net", "--unshare-pid", "--clearenv"]
    for path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        argv += ["--ro-bind", path, path]
    argv += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for mount in plan["dependency_mounts"]:
        argv += ["--ro-bind", mount["source"], mount["destination"]]
    argv += [
        "--ro-bind",
        str(product / "sources" / system),
        pkg,
        "--ro-bind",
        str(library_root),
        library_dest,
        "--setenv",
        "PATH",
        runtime + "/bin:" + llvm + "/bin:" + conda + "/bin:" + conda + "/riscv-tools/bin:/usr/bin:/bin",
        "--setenv",
        "LD_LIBRARY_PATH",
        llvm + "/lib:" + conda + "/lib:" + conda + "/riscv-tools/lib",
        "--setenv",
        "MERLIN_MLIR_INSTALL",
        llvm,
        "--setenv",
        "MERLIN_CLANG",
        llvm + "/bin/clang-23",
        "--setenv",
        "PYTHONPATH",
        pkg + "/compiler_library:" + library_dest,
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "HOME",
        "/tmp/home",
        "--setenv",
        "TMPDIR",
        "/tmp",
        runtime + "/bin/python3.12",
        "-c",
        program,
    ]
    workspace = workspace.absolute()
    workspace.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    (workspace / "startup.stdout").write_text(result.stdout)
    (workspace / "startup.stderr").write_text(result.stderr)
    receipt = {
        "schema": "merlin.paper-startup-probe.v1",
        "argv": argv,
        "bwrap_sha256": bwrap_sha256,
        "exit_status": result.returncode,
        "status": "imports_verified" if result.returncode == 0 else "probe_failed",
        "scope": "interpreter_startup_and_xdsl_imports_only",
        "compiler_invoked": False,
        "hardware_execution": False,
        "python_startup_paths": plan["python_startup_paths"],
    }
    if result.returncode == 0:
        observed = json.loads(result.stdout)
        if observed.get("outside"):
            raise ValueError("probe imported code outside selected or ordinary OS roots")
        receipt["observed"] = observed
    (workspace / "startup.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def execute_rebuild(
    root: Path,
    config: dict,
    system: str,
    case: str,
    *,
    image: str,
    workspace: Path,
    bwrap: Path,
    bwrap_sha256: str,
    dependency_root: Path | None = None,
    metadata_root: Path | None = None,
) -> dict:
    """Build one frozen compiler/case through the original script in a new namespace."""
    original = validate(root, config, image)
    manifest = read(Path(original["manifest"]))
    matches = [slot for slot in manifest.get("slots", ()) if slot["system"] == system and slot["case_id"] == case]
    if len(matches) != 1:
        raise ValueError("selected image has no unique compiler/case object")
    plan = rebuild_plan(root, config, system, case, workspace, dependency_root, metadata_root)
    if not plan["ready_for_source_rebuild"]:
        raise ValueError("source rebuild refused: dependency trees or inventories are incomplete")
    bwrap = wrapper_selection(bwrap, bwrap_sha256)
    workspace = workspace.absolute()
    workspace.mkdir(parents=True, exist_ok=False)
    argv = [str(bwrap), *plan["argv"][1:]]
    # The inner original script invokes bwrap by name. Bind that exact selected
    # executable over the conventional host path in this namespace only.
    insertion = argv.index("--proc")
    argv[insertion:insertion] = ["--ro-bind", str(bwrap), "/usr/bin/bwrap"]
    argv[-4:-4] = ["--setenv", "PYTHONDONTWRITEBYTECODE", "1"]
    receipt = {
        "schema": "merlin.paper-kernel-rebuild.v1",
        "system": system,
        "case": case,
        "selected_image": image,
        "original_kernel_sha256": matches[0]["kernel_o_sha256"],
        "source_sha256": plan["source_sha256"],
        "argv": argv,
        "bwrap_sha256": bwrap_sha256,
        "verified_tools": plan["verified_tools"],
        "source_and_libraries_read_only": True,
        "new_build": True,
        "hardware_execution": False,
        "numerical_verdict": "not_executed",
        "historical_python_runtime_identity_established": False,
    }
    (workspace / "rebuild-invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (workspace / "rebuild.stdout").open("wb") as stdout, (workspace / "rebuild.stderr").open("wb") as stderr:
        completed = subprocess.run(argv, stdout=stdout, stderr=stderr, check=False)
    result_path = workspace / "kernels" / system / case / "result.json"
    receipt.update(script_exit_status=completed.returncode, status="build_failed")
    if completed.returncode == 0 and result_path.is_file():
        result = read(result_path)
        kernel = result_path.parent / "kernel.o"
        if (
            result.get("status") == "compiled"
            and result.get("source_unchanged") is True
            and result.get("source_sha256") == plan["source_sha256"]
            and result.get("interface_sha256") == matches[0]["compiler_result"]["interface_sha256"]
            and kernel.is_file()
            and sha(kernel) == result.get("kernel_sha256")
        ):
            receipt.update(
                status="compiled",
                kernel_sha256=sha(kernel),
                matches_preserved_object=sha(kernel) == receipt["original_kernel_sha256"],
            )
    # Reopen every selected dependency, source and library after the build.
    after = rebuild_plan(root, config, system, case, workspace, dependency_root, metadata_root)
    if not after["ready_for_source_rebuild"]:
        raise ValueError("source rebuild closure changed during execution")
    if sha(bwrap) != bwrap_sha256:
        raise ValueError("selected bwrap executable changed during rebuild")
    (workspace / "rebuild.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="restored data/ directory")
    parser.add_argument(
        "--dependency-root", type=Path, help="separate restored runtime/ directory containing dependencies/"
    )
    parser.add_argument(
        "--dependency-metadata-root", type=Path, help="directory containing SHA-pinned dependency inventories"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "validate", "replay"):
        command = sub.add_parser(action)
        command.add_argument("--image", required=True)
        if action == "replay":
            command.add_argument("--wrapper", required=True, type=Path)
            command.add_argument("--wrapper-sha256", required=True)
            command.add_argument("--platform", required=True, type=Path)
            command.add_argument("--output", required=True, type=Path)
    for action in ("rebuild-plan", "rebuild", "probe"):
        command = sub.add_parser(action)
        command.add_argument("--system", required=True)
        command.add_argument("--case", required=True)
        command.add_argument("--workspace", required=True, type=Path)
        if action == "rebuild":
            command.add_argument("--image", required=True)
        if action in {"rebuild", "probe"}:
            command.add_argument("--bwrap", required=True, type=Path)
            command.add_argument("--bwrap-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        config = configuration(args.config)
        root = args.root.resolve(strict=True)
        dependency_root = args.dependency_root.resolve(strict=True) if args.dependency_root else root
        metadata_root = args.dependency_metadata_root.resolve(strict=True) if args.dependency_metadata_root else None
        if args.action == "rebuild-plan":
            result = rebuild_plan(root, config, args.system, args.case, args.workspace, dependency_root, metadata_root)
            status = 0 if result["ready_for_source_rebuild"] else 2
        elif args.action == "probe":
            result = startup_probe(
                root,
                config,
                args.system,
                args.case,
                workspace=args.workspace,
                bwrap=args.bwrap,
                bwrap_sha256=args.bwrap_sha256,
                dependency_root=dependency_root,
                metadata_root=metadata_root,
            )
            status = 0 if result["status"] == "imports_verified" else 1
        elif args.action == "rebuild":
            result = execute_rebuild(
                root,
                config,
                args.system,
                args.case,
                image=args.image,
                workspace=args.workspace,
                bwrap=args.bwrap,
                bwrap_sha256=args.bwrap_sha256,
                dependency_root=dependency_root,
                metadata_root=metadata_root,
            )
            status = 0 if result["status"] == "compiled" else 1
        elif args.action == "plan":
            image = config["images"][args.image]
            result = {
                "image": args.image,
                "selection": image,
                "wrapper_command": ["<explicit SHA-pinned fpga-run-elf>", "-n", args.image, image["elf"]["path"]],
                "execution": "not_executed",
                "inputs_verified": False,
            }
            status = 0
        elif args.action == "validate":
            result, status = validate(root, config, args.image), 0
        else:
            result = replay(
                root,
                config,
                args.image,
                wrapper=args.wrapper,
                wrapper_sha256=args.wrapper_sha256,
                platform=args.platform,
                output=args.output,
            )
            status = 0 if result["status"] == "wrapper_completed" else 1
        print(json.dumps(result, indent=2))
        return status
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
