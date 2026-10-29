"""Select the unchanged, previously bound pure graph context for declared calls.

Never imports the affected source-operational leaves or refreshes their bindings.
Every discovered standard import must belong to the earlier fixed Init closure.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.verify import lean_graph_regions as prior
from merlin.verify import lean_graph_regions_dependencies as dependencies


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def select(destination: Path):
    """Check fixed prior objects/sidecars/runtime BEFORE any Lean invocation."""
    destination.mkdir(parents=True, exist_ok=False)
    root = repo_root()
    old = root / "out/artifacts/verification/graph-regions/qualification-v4"
    receipt_path = old / "fusion-realized/EVIDENCE.json"
    receipt = json.loads(receipt_path.read_text())
    identity = receipt["identity"]
    prior.fresh_ref(identity["lean"])
    if identity["lean"]["sha256"] != prior.LEAN_SHA or identity["lean_arguments"] != ["-j1"]:
        raise ValueError("changed fixed pure Lean selection")
    fixed = dependencies.validate(identity["toolchain_dependencies"], prior.fresh_ref)
    if any("Std/Time/Format/Basic" in key for key in fixed["files"]):
        raise ValueError("affected dependency is inside proposed context")
    for field in ("sources", "objects", "host_sources"):
        for ref in identity[field].values():
            prior.fresh_ref(ref)
    # Reuse frozen syntax only, independently of the affected semantic profile.
    syntax = {}
    for module in ("AST", "Eq"):
        path = root / f"merlin/lean/MerlinVerify/Core/{module}.lean"
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:merlin/lean/MerlinVerify/Core/{module}.lean"], cwd=root
        )
        if committed != path.read_bytes():
            raise ValueError("frozen syntax differs from selected primary revision")
        syntax["MerlinVerify.Core." + module] = prior.file_ref(path)
    oracle_path = old / "valid-realized-source/EVIDENCE.json"
    oracle = json.loads(oracle_path.read_text())
    for ref in [
        oracle["worker"],
        oracle["oracle"]["python_executable"],
        oracle["oracle"]["ir_module"],
        *oracle["oracle"]["native_modules"].values(),
    ]:
        prior.fresh_ref(ref)
    invocation = json.loads((oracle_path.parent / "process.json").read_text())["command"][0]
    if prior.file_ref(Path(invocation)) != oracle["oracle"]["python_executable"]:
        raise ValueError("changed fixed compiler-oracle invocation")
    record = dict(
        schema="merlin.graph-regions.declared-context.v1",
        prior_receipt=prior.file_ref(receipt_path),
        fixed_identity=identity,
        fixed_syntax=syntax,
        primary_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        oracle_evidence=prior.file_ref(oracle_path),
        oracle_invocation=invocation,
        old_source_operational_profile_used=False,
        old_source_operational_bridge="UNQUALIFIED; separate Basic.olean obligation unchanged",
        selected_incident_object=False,
        fixed_core_modules=fixed["module_count"],
        fixed_core_references=len(fixed["files"]),
        new_host=prior.file_ref(Path(__file__)),
    )
    save(destination / "FIXED-CONTEXT.json", record)
    return record


def build(destination: Path, selected: dict, modules: list[str]):
    """Discover actual imports structurally; refuse anything outside fixed roots."""
    destination.mkdir(parents=True, exist_ok=False)
    root = repo_root()
    fixed = selected["fixed_identity"]
    lean = prior.fresh_ref(fixed["lean"])
    inventory = dependencies.validate(fixed["toolchain_dependencies"], prior.fresh_ref)
    allowed = {row["path"]: row for row in inventory["files"].values() if row["path"].endswith(".lean")}
    owned = root / "src/merlin/verify/lean"
    syntax = root / "merlin/lean"
    compiled = destination / "compiled"
    compiled.mkdir()
    env = os.environ.copy()
    env.update(LEAN_PATH=str(compiled.resolve()), LEAN_SRC_PATH=os.pathsep.join(map(str, (owned, syntax))))
    fixed_projects = {**{"Merlin." + name: ref for name, ref in fixed["sources"].items()}, **selected["fixed_syntax"]}
    closure, order, visiting = {}, [], set()

    def visit(path):
        path = path.resolve(strict=True)
        key = str(path)
        if key in closure:
            return
        if key in visiting:
            raise ValueError("cyclic import discovery")
        visiting.add(key)
        if key in allowed:
            prior.fresh_ref(allowed[key])
            is_project = False
            module = key
        else:
            source_root = syntax if path.is_relative_to(syntax) else owned
            if not path.is_relative_to(source_root):
                raise ValueError("import outside selected roots")
            module = ".".join(path.relative_to(source_root).with_suffix("").parts)
            is_project = True
            if module in fixed_projects:
                prior.fresh_ref(fixed_projects[module])
            elif not module.startswith("Merlin.GraphRegionsDeclared"):
                raise ValueError("unfixed or affected project import")
        before = prior.file_ref(path)
        command = [str(lean), "-j1", "--src-deps", key]
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
        log = destination / (f"imports-{len(closure):04d}-{len(visiting):04d}.json")
        save(
            log,
            dict(
                command=command,
                returncode=process.returncode,
                stdout=process.stdout,
                stderr=process.stderr,
                source=before,
            ),
        )
        process.check_returncode()
        children = list(dict.fromkeys(Path(line).resolve(strict=True) for line in process.stdout.splitlines() if line))
        for child in children:
            visit(child)
        prior.fresh_ref(before)
        closure[key] = dict(
            source=before,
            module=module,
            project=is_project,
            dependencies=list(map(str, children)),
            discovery=prior.file_ref(log),
        )
        visiting.remove(key)
        if is_project:
            order.append((module, source_root, path))

    for module in modules:
        visit(owned / (module.replace(".", "/") + ".lean"))
    objects = {}
    for module, source_root, path in order:
        prior.fresh_ref(closure[str(path)]["source"])
        target = compiled / (module.replace(".", "/") + ".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.run(
            [str(lean), "-j1", f"--root={source_root}", "-o", str(target), str(path)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        save(
            destination / (module + ".json"),
            dict(returncode=process.returncode, stdout=process.stdout, stderr=process.stderr),
        )
        process.check_returncode()
        prior.fresh_ref(closure[str(path)]["source"])
        objects[module] = prior.file_ref(target)
    dependencies.validate(fixed["toolchain_dependencies"], prior.fresh_ref)
    identity = dict(
        context=selected,
        actual_import_closure=closure,
        project_objects=objects,
        builder=prior.file_ref(Path(__file__)),
        old_source_profile_imported=False,
    )
    save(destination / "IDENTITY.json", identity)
    return lean, env, identity


def bind_provider_host(destination: Path, selected: dict, intake: Path):
    """Freeze actually loaded provider code/libraries, without adopting bad bindings.

    The intake is source/data generation, not a machine execution qualification.
    A later check must require every loaded identity to match this fixed packet;
    encountering an additional dependency is unavailable, never an implicit pass.
    """
    destination.mkdir(parents=True, exist_ok=False)
    dependencies.validate(selected["fixed_identity"]["toolchain_dependencies"], prior.fresh_ref)
    host = json.loads(intake.read_text())
    files = {**host["loaded_modules"], **host["mapped_host_files"]}
    if not files:
        raise ValueError("no actual provider host closure")
    for name, reference in files.items():
        if "lean_graph_regions_source" in name or "Std/Time/Format/Basic" in reference["path"]:
            raise ValueError("affected source profile entered proposed host context")
        prior.fresh_ref(reference)
    for key in ("python", "configuration"):
        prior.fresh_ref(host[key])
    # The actual executable is fixed in the earlier independent pure-graph
    # producer receipt, rather than inferred from its virtual environment name.
    capture = json.loads(
        (repo_root() / "out/artifacts/verification/graph-regions/qualification-v4/realized/CAPTURE.json").read_text()
    )
    if host["python"] != capture["python_executable"]:
        raise ValueError("changed provider interpreter from fixed earlier selection")
    record = dict(
        schema="merlin.graph-regions.declared-provider-host.v1",
        context=selected,
        source_only_intake=prior.file_ref(intake),
        loaded_host=host,
        old_source_profile_imported=False,
        incident_object_selected=False,
        native_execution_qualified=False,
    )
    save(destination / "FIXED-HOST.json", record)
    return record
