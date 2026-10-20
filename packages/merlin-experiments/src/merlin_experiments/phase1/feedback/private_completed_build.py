"""Reuse completed production, never a prior private-model verdict.

The private spec may explicitly select ``completed_build_reuse`` with this
module's schema, a cache root, an original producer run, an ordinary sealed
``source_snapshot`` pin, and complete ``sdk_roots`` in that run's V4 archive.
The producer release is independent of today's checker. Only an actual build
under that release may publish; historical receipts cannot seed this store.
All PFM post-build checks remain outside this helper and run on every hit.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from merlin.common import bundle_snapshot as B
from merlin.common import invocation_record as I
from merlin.common import read_lease_digest as leases
from merlin.common.digest import sha256_file
from merlin.common.paths import build_dir, module_source_path
from merlin.llvmlower.compilation_recipe import verify_completed_recipe
from merlin_experiments import source_snapshot as S

SCHEMA = "merlin.phase1.completed_build_reuse.v1"
ENTRY = "merlin.phase1.completed_build_production.v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _ordinary(value, *, exists=True):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or path.resolve() != path:
        raise ValueError("completed-build reuse requires canonical ordinary paths")
    if any(p.is_symlink() for p in (path, *path.parents)) or (exists and not path.exists()):
        raise ValueError("completed-build reuse input is absent or indirect")
    return path


def _sha(path):
    path = Path(path)
    digest = leases.digest(path) if path.stat().st_size >= leases.minimum_bytes() else None
    return digest if digest is not None else sha256_file(path)


def _pin(path):
    path = _ordinary(path)
    return {"path": str(path), "sha256": _sha(path)}


def _read(pin):
    if type(pin) is not dict or set(pin) != {"path", "sha256"} or _pin(pin["path"]) != pin:
        raise ValueError("completed-build input bytes changed")
    return json.loads(Path(pin["path"]).read_bytes())


def _sdk(root, frozen):
    """Compare a complete declared input with its ordinary original archive.

    Follow SDK aliases as the ordinary input freezer does, retaining each
    lexical link and target name. Alias cycles and special files refuse. Hash
    reuse requires the existing live read lease; membership is always fresh.
    """
    rows, names = [], set()
    pending = [(root, frozen, ".", frozenset())]
    while pending:
        live, saved, name, ancestors = pending.pop()
        info = live.stat()
        identity = info.st_dev, info.st_ino
        alias = os.readlink(live) if live.is_symlink() else None
        if live.is_dir():
            if identity in ancestors or not saved.is_dir():
                raise ValueError("SDK has a directory alias cycle or changed member type")
            rows.append([name, "directory", alias, str(live.resolve())])
            members = sorted(live.iterdir())
            if {p.name for p in members} != {p.name for p in saved.iterdir()}:
                raise ValueError("SDK membership differs from its original frozen input")
            pending.extend((p, saved / p.name, name + "/" + p.name, ancestors | {identity}) for p in members)
        elif live.is_file() and saved.is_file():
            digest = _sha(live.resolve())
            if digest != _sha(saved):
                raise ValueError("SDK differs from its original frozen input")
            names.add(name)
            rows.append([name, digest, info.st_mode & 0o777, alias, str(live.resolve())])
        else:
            raise ValueError("SDK differs from its original frozen input")
    if not names:
        raise ValueError("SDK closure has no ordinary file members")
    return _digest(sorted(rows))


def _runtime_paths(tools):
    """Conservative default distributions, Python paths and native loader inputs.

    A bin directory is never a distribution. This deliberately over-selects a
    system tool's prefix: operators can use contained LLVM/GCC/Python installs,
    but may not guess just the dependencies observed by a particular tiny build.
    """
    from merlin.common.paths import python_import_roots

    required = {tool.parent.parent for tool in tools}
    required.update((Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()))
    lexical = Path(sys.executable).absolute().parent.parent
    if (lexical / "pyvenv.cfg").is_file():
        required.add(lexical.resolve())
    # Include the system loader's default namespaces, including its exact cache.
    required.update(
        Path(p).resolve() for p in ("/lib", "/lib64", "/usr/lib", "/usr/lib64", "/etc/ld.so.cache") if Path(p).exists()
    )
    sources = python_import_roots()
    for value in sys.path:
        if not value:
            raise ValueError("completed-build reuse refuses implicit Python working-directory imports")
        path = Path(value).resolve()
        if path.exists() and not any(path == root.resolve() for root in sources):
            required.add(path)
    for value in os.environ.get("PATH", "").split(os.pathsep):
        if not value or not Path(value).is_absolute():
            raise ValueError("completed-build reuse refuses implicit executable search paths")
        if Path(value).exists():
            required.add(Path(value).resolve())
    # Loaded Python extensions can have their own RUNPATH outside the prefix.
    for row in Path("/proc/self/maps").read_text().splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            required.add(_ordinary(Path(fields[5]).resolve()))
    return required


def _tool_defaults(tools, archive):
    """Reopen actual built-in search namespaces, not guessed header/library lists."""
    required = set()

    def query(tool, arguments):
        (saved,) = B.snapshot_input_paths(
            Path(archive["workspace"]), _read(archive["manifest"]), [tool], repo=Path(archive["repo"])
        )
        if _sha(tool) != _sha(saved):
            raise ValueError("runtime default-query tool differs from selected archive")
        result = subprocess.run([str(tool), *arguments], capture_output=True, text=True, check=True, timeout=10)
        return result.stdout

    for role in ("clang", "gcc"):
        if role not in tools:
            raise ValueError("completed-build reuse requires selected Clang and GCC defaults")
        tool = Path(tools[role]["path"])
        output = query(tool, ["-print-search-dirs"])
        paths = []
        for category in ("programs:", "libraries:"):
            selected = [
                line.partition(":")[2].strip().removeprefix("=")
                for line in output.splitlines()
                if line.startswith(category)
            ]
            if len(selected) != 1 or not selected[0]:
                raise ValueError("unsupported compiler default search namespaces")
            paths.extend(selected[0].split(os.pathsep))
        for name in paths:
            if not Path(name).is_absolute():
                raise ValueError("implicit compiler library search path is unsupported")
            if Path(name).exists():
                required.add(Path(name).resolve())
        if role == "gcc":
            sysroot = query(tool, ["-print-sysroot"]).strip()
            if sysroot:
                required.add(_ordinary(Path(sysroot).resolve()))
    loader = shutil.which("ldconfig")
    if loader is None:
        raise ValueError("native loader default namespace cannot be inspected")
    output = query(Path(loader).resolve(), ["-p"])
    entries = [line.rpartition(" => ")[2].strip() for line in output.splitlines() if " => " in line]
    if not entries:
        raise ValueError("native loader cache has no understood library entries")
    for entry in entries:
        required.add(_ordinary(Path(entry).resolve()))
    dependencies = shutil.which("ldd")
    if dependencies is None:
        raise ValueError("native loader dependency resolution cannot be inspected")
    # Include actual transitive DT_NEEDED and RPATH/RUNPATH resolution.
    for row in tools.values():
        output = query(Path(dependencies).resolve(), [row["path"]])
        if not output.strip():
            raise ValueError("native loader returned no dependency information")
        for line in output.splitlines():
            line = line.strip()
            if not line or line == "statically linked" or line.startswith(("linux-vdso.so.", "linux-gate.so.")):
                continue
            path = line.rpartition(" => ")[2] if " => " in line else line
            path = path.partition(" (")[0]
            if not Path(path).is_absolute():
                raise ValueError("unresolved or unsupported native loader dependency")
            required.add(_ordinary(Path(path).resolve()))
    return required


def _producer(selection, *, populate=False):
    from . import compiler_profile_runtime as R

    if (
        type(selection) is not dict
        or set(selection) != {"schema", "cache_root", "producer_run", "source_snapshot", "sdk_roots", "runtime_archive"}
        or selection["schema"] != SCHEMA
    ):
        raise ValueError("completed-build reuse needs its explicit closed producer selection")
    run = _ordinary(selection["producer_run"])
    owner = R.functional_service(run)
    if owner is None:
        raise ValueError("completed-build reuse requires the original selected compiler profile")
    owner._verify()
    snapshot = _ordinary(selection["source_snapshot"]["path"]).parent
    release = S.verify(snapshot)
    seal, _ = S.load_seal(snapshot, "snapshot")
    if _pin(seal) != selection["source_snapshot"] or release["external_links"]:
        raise ValueError("producer source release is changed or has unbound external links")
    # The common-profile owner and the actual framework producer are distinct.
    # Only guarded execution of this release can populate (checked below).
    required = (
        "src/merlin",
        "packages/merlin-experiments/src/merlin_experiments",
        "packages/merlin-experiments/src/merlin",
        "merlin/runtime",
        "merlin/contract",
        "merlin/schemas",
    )
    if any(not any(Path(name).is_relative_to(root) for root in release["source_roots"]) for name in required):
        raise ValueError("producer release omits framework, runtime or contract/schema inputs")
    roots = selection["sdk_roots"]
    if not isinstance(roots, list) or not roots or len(set(roots)) != len(roots):
        raise ValueError("completed-build reuse needs complete selected SDK/runtime roots")
    roots = [_ordinary(p) for p in roots]
    archive = selection["runtime_archive"]
    if type(archive) is not dict or set(archive) != {"manifest", "workspace", "repo"}:
        raise ValueError("completed-build reuse requires its independent ordinary runtime archive")
    archive_bundle = _read(archive["manifest"])
    workspace, repository = _ordinary(archive["workspace"]), _ordinary(archive["repo"])
    frozen = B.snapshot_input_paths(workspace, archive_bundle, roots, repo=repository)
    sdk = {str(root): _sdk(root, saved) for root, saved in zip(roots, frozen, strict=True)}
    if populate:
        active = R.active_original_execution()
        if (
            active is None
            or active.owner.workspace != owner.workspace
            or active.owner.bundle_bytes != owner.bundle_bytes
        ):
            raise ValueError("only the live selected original executor may populate completed builds")
        _executing_release(snapshot, selection["source_snapshot"], release)
    return (
        {
            "run": str(run),
            "environment": _pin(run / "environment.yaml"),
            "source_snapshot": selection["source_snapshot"],
            "bundle": _pin(owner.bundle_manifest),
            "profile": owner.profile,
            "snapshot_marker": B.snapshot_marker_digest(owner.workspace),
            "sdk": sdk,
            "runtime_archive": archive,
            "runtime_marker": B.snapshot_marker_digest(workspace),
        },
        owner,
        release,
    )


def _executing_release(snapshot, seal, release):
    """Use the existing import guard, never the historical profile's repo name."""
    from merlin.common.paths import repo_root, runtime_dir
    from merlin_experiments.frozen_python import active_source_identity

    if active_source_identity() != seal or repo_root() != snapshot:
        raise ValueError("completed-build publication requires its actual guarded frozen producer")
    for module in (__name__, "merlin.compile.baremetal_model"):
        path = module_source_path(module)
        if not path.is_relative_to(snapshot) or release["files"].get(str(path.relative_to(snapshot))) != _sha(path):
            raise ValueError("executed builder is outside the selected producer release")
    if runtime_dir() != snapshot / "merlin/runtime":
        raise ValueError("runtime sources are outside the selected producer release")


def _route(device):
    if device is None:
        return None
    from merlin.compile.complete_group_routing import select_complete_group
    from merlin.llvmlower.device_build import DeviceRouting

    if type(device) is not DeviceRouting or device.select is not select_complete_group:
        raise ValueError("completed-build reuse requires canonical complete-group routing")
    opaque = (
        "exact_selection",
        "catalog_manifest",
        "catalog_object",
        "prepared_transform",
        "catalog_builder",
        "final_elf_audit",
        "post_offload_transform",
        "target_configuration",
        "offload_selection",
    )
    if any(getattr(device, name) is not None for name in opaque):
        raise ValueError("completed-build reuse refuses unknown routing hooks")
    result = {}
    for field in dataclasses.fields(device):
        if field.name in (*opaque, "select", "linked_elf_admission"):
            continue
        value = getattr(device, field.name)
        result[field.name] = (
            value.record()
            if field.name == "device_ir_budget" and value is not None
            else str(value)
            if isinstance(value, Path)
            else value
        )
    return result


def _material(kwargs, context, selection, producer, owner):
    from merlin.common.paths import contract_dir, schemas_dir
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.llvmlower import toolchain
    from merlin.model_check.full_build import BuildSettings
    from merlin.model_check.full_build_identity import build_material

    allowed = {
        "capture",
        "package",
        "board_catalog",
        "board",
        "dts",
        "output",
        "target",
        "run",
        "arena_mb",
        "device",
        "respect_captured_quantization_scope",
        "readback",
        "group_profile",
        "host_llvm_bitcode",
        "blob_weights",
        "host_math_policy",
        "math_archive_symbols",
        "host_math_symbols",
        "op_profile",
    }
    if set(kwargs) - allowed or kwargs.get("run") != "none":
        raise ValueError("completed-build reuse refuses unsupported build options or execution")
    if type(context) is not dict or set(context) != {"capability", "facts"}:
        raise ValueError("completed-build reuse needs exact selected capability and facts")
    settings = BuildSettings(
        board=kwargs["board"],
        board_catalog=Path(kwargs["board_catalog"]),
        host_dts=Path(kwargs["dts"]),
        datapath="",
        arena_mb=kwargs["arena_mb"],
        host_package=Path(kwargs["package"]),
        dedup_weights=kwargs.get("blob_weights", True),
        op_profile=kwargs.get("op_profile", False),
        host_math_policy=kwargs.get("host_math_policy", "native"),
    )
    device = kwargs.get("device")
    candidate = Path(device.package_dir) if device is not None else Path(kwargs["package"])
    material = build_material(
        Path(kwargs["capture"]),
        kwargs["target"],
        "native-production",
        candidate,
        strict_tree_sha256(candidate)["sha256"],
        settings,
        device is not None,
    )
    # Explicit old producer selection, never inferred equality with today's checker.
    material["source"] = producer["source_snapshot"]
    material["host_package_tree"] = strict_tree_sha256(Path(kwargs["package"]))
    material["process_environment"] = I.environment_identity(os.environ)
    material["python_import_paths"] = list(sys.path)
    compiler_python = toolchain.compiler_python().absolute()
    material["compiler_python_command"] = str(compiler_python)
    if material["compiler_driver"] != "python" or any(
        os.environ.get(name)
        for name in (
            "LD_PRELOAD",
            "LD_AUDIT",
            "GCC_EXEC_PREFIX",
            "COMPILER_PATH",
            "CFLAGS",
            "CXXFLAGS",
            "LDFLAGS",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "PYTHONSTARTUP",
            "BASH_ENV",
            "ENV",
        )
    ):
        raise ValueError("completed-build reuse refuses external compiler hooks")
    tools = [Path(row["path"]) for row in material["tools"].values() if not row.get("missing")]
    if len(tools) != len(material["tools"]):
        raise ValueError("completed-build reuse has an unavailable selected tool")
    tools.append(Path(sys.executable).resolve())
    archive = selection["runtime_archive"]
    mapped = B.snapshot_input_paths(
        Path(archive["workspace"]), _read(archive["manifest"]), tools, repo=Path(archive["repo"])
    )
    if any(_sha(tool) != _sha(saved) for tool, saved in zip(tools, mapped, strict=True)):
        raise ValueError("selected build tools differ from original producer inputs")

    def covered(paths):
        return all(any(path.is_relative_to(Path(root)) for root in producer["sdk"]) for path in paths)

    required = _runtime_paths(tools)
    producer_root = Path(producer["source_snapshot"]["path"]).parent
    required.update(
        path.resolve() for path in (contract_dir(), schemas_dir()) if not path.is_relative_to(producer_root)
    )
    if (compiler_python.parent.parent / "pyvenv.cfg").is_file():
        required.add(compiler_python.parent.parent.resolve())
    if not covered(required) or not covered(_tool_defaults(material["tools"], archive)):
        raise ValueError("complete compiler distribution, Python or loader runtime is outside the selected archive")
    for name in ("CPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"):
        for value in os.environ.get(name, "").split(os.pathsep) if os.environ.get(name) else ():
            path = _ordinary(value)
            if not any(path.is_relative_to(Path(root)) for root in producer["sdk"]):
                raise ValueError("selected search path is outside complete SDK/runtime closure")
    options = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in kwargs.items()
        if name not in {"output", "device"}
    }
    return json.loads(
        json.dumps(
            {
                "producer": producer,
                "build": material,
                "options": options,
                "routing": _route(device),
                "declarations": context,
            },
            allow_nan=False,
        )
    )


def _products(receipt, expected=None):
    if receipt.get("status") != "compiled" or receipt.get("schema") != "merlin.baremetal-saved-model.v1":
        raise ValueError("only a successfully completed compiler product can be retained")
    output = receipt["output"]
    recipe = output["compilation_recipe"]
    verify_completed_recipe(
        Path(recipe["path"]), executable=Path(output["elf"]), expected_recipe_sha256=recipe["sha256"]
    )
    if _sha(_ordinary(output["elf"])) != output["elf_sha256"]:
        raise ValueError("completed-build ELF changed")
    # Retain every consumed object, source, sidecar, typed custody and relink input.
    root = Path(output["elf"]).parent
    if expected is not None:
        if any(not Path(name).is_relative_to(root) or pin != _pin(name) for name, pin in expected.items()):
            raise ValueError("retained completed-build product changed")
        return expected
    rows = {str(p): _pin(p) for p in root.rglob("*") if p.is_file()}
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("completed-build products must be ordinary files")
    return rows


def _produce(*, kwargs, result):
    from merlin.compile.baremetal_model import compile_saved_model as compile_model

    receipt = compile_model(**kwargs)
    products = _products(receipt)
    result.write_text(json.dumps({"receipt": receipt, "products": products}, sort_keys=True) + "\n")
    return receipt


def _admit(receipt, kwargs, directory):
    from merlin.llvmlower.device_build import admit_linked_device_elf, freeze_device_elf_admission

    device = kwargs.get("device")
    if device is not None:
        selection = freeze_device_elf_admission(device)
        if selection is None:
            raise ValueError("completed device image requires today's independently selected ISA policy")
        return admit_linked_device_elf(
            device,
            selection,
            elf=Path(receipt["output"]["elf"]),
            linked_sha256=receipt["output"]["elf_sha256"],
            directory=directory,
        )
    return None


def compile_saved_model(*, completed_build_reuse=None, build_context=None, **kwargs):
    """PFM's compilation seam; absence preserves the ordinary compiler verbatim."""
    if completed_build_reuse is None:
        from merlin.compile.baremetal_model import compile_saved_model as compile_model

        return compile_model(**kwargs)
    started = time.monotonic()
    production_seconds = 0.0
    selection = completed_build_reuse
    with leases.verification():
        producer, owner, release = _producer(selection)
        material = _material(kwargs, build_context, selection, producer, owner)
        root = _ordinary(selection["cache_root"], exists=False)
        if not root.is_relative_to(build_dir().resolve()) or any(
            root.is_relative_to(Path(kwargs[k])) for k in ("capture", "package")
        ):
            raise ValueError("completed-build cache needs an independent selected build root")
        if any(root.is_relative_to(Path(sdk)) for sdk in producer["sdk"]):
            raise ValueError("completed-build cache cannot mutate its selected runtime inputs")
        root.mkdir(parents=True, mode=0o700, exist_ok=True)
        directory = root / _digest(material)
        directory.mkdir(mode=0o700, exist_ok=True)
        with (directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entry = directory / "production.json"
            hit = entry.exists()
            if hit:
                record = json.loads(entry.read_bytes())
                if record.get("schema") != ENTRY or record.get("material") != material:
                    raise ValueError("completed-build producer selection changed")
                actual = I.verify(Path(record["invocation"]["path"]))
                producer_root = Path(selection["source_snapshot"]["path"]).parent
                producer_relative = (
                    "packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_completed_build.py"
                )
                producer_file = producer_root / producer_relative
                if (
                    _pin(record["invocation"]["path"]) != record["invocation"]
                    or actual.get("kind") != "python_call"
                    or actual.get("stage") != "completed_model_production"
                    or actual.get("callable") != __name__ + "._produce"
                    or actual.get("outcome") != "returned"
                    or actual.get("arguments") != {"material_sha256": _digest(material)}
                    or actual.get("executable") != _pin(producer_file)
                    or release["files"].get(producer_relative) != _sha(producer_file)
                    or actual.get("inputs") != [selection["source_snapshot"]]
                    or actual.get("dependencies")
                    != sorted(
                        [_pin(producer_root / "src/merlin/compile/baremetal_model.py"), _pin(producer_file)],
                        key=lambda pin: pin["path"],
                    )
                    or actual.get("outputs") != [record["result"]]
                    or Path(record["result"]["path"]) != directory / "compiler-result.json"
                ):
                    raise ValueError("completed-build cache has no original returned producer invocation")
                original = _read(record["result"])
                receipt = original["receipt"]
                if _products(receipt, original["products"]) != original["products"]:
                    raise ValueError("completed-build artifacts changed")
                from merlin.compile.baremetal_model import _output_root

                output = _output_root(_ordinary(kwargs["output"], exists=False))
                output.mkdir(parents=True, mode=0o700)
                admission = _admit(receipt, kwargs, output / "reused-image-admission")
            else:
                _producer(selection, populate=True)
                result = directory / "compiler-result.json"
                with I.observe_call(
                    directory,
                    stage="completed_model_production",
                    function=_produce,
                    arguments={"material_sha256": _digest(material)},
                    inputs=(Path(selection["source_snapshot"]["path"]),),
                    outputs=(result,),
                    dependencies=(module_source_path("merlin.compile.baremetal_model"),),
                ) as observed:
                    producing = time.monotonic()
                    receipt = _produce(kwargs=kwargs, result=result)
                    production_seconds = time.monotonic() - producing
                    observed.returned()
                admission = None  # Ordinary compilation already performed its selected policy.
                record = {
                    "schema": ENTRY,
                    "material": material,
                    "result": _pin(result),
                    "invocation": _pin(observed.path),
                }
                if _producer(selection, populate=True)[0] != producer:
                    raise ValueError("producer release changed while compiling")
                if _material(kwargs, build_context, selection, producer, owner) != material:
                    raise ValueError("codegen inputs changed while compiling")
                pending = entry.with_suffix(".pending")
                pending.write_text(json.dumps(record, sort_keys=True) + "\n")
                pending.replace(entry)
            if (
                _producer(selection)[0] != producer
                or _material(kwargs, build_context, selection, producer, owner) != material
            ):
                raise ValueError("completed-build inputs changed during reuse")
            reuse = {
                "hit": hit,
                "production": _pin(entry),
                "producer": producer,
                "checker": _pin(module_source_path("merlin_experiments.phase1.feedback.private_full_models")),
                "lookup_and_validation_seconds": time.monotonic() - started - production_seconds,
                "production_transaction_seconds": production_seconds,
                "fresh_instruction_admission": admission,
                "scope": "compiler products only; all current private source/host/offload/numerical gates required",
            }
            # Reopen products after policy evaluation, before consumer publication.
            original = _read(record["result"])
            if _products(receipt, original["products"]) != original["products"]:
                raise ValueError("completed-build products changed before publication")
            receipt = {**receipt, "completed_build_reuse": reuse}
            if hit:
                (Path(kwargs["output"]) / "baremetal_model.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
            return receipt
