"""Rebind inputs to an admitted whole-model build without changing its computation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from merlin.common.process import invocation_records as I
from merlin.common.process import invocation_records as proc
from merlin.evaluation.model.kernels.replay import compute_step, content_digest, package_digest, target_inputs
from merlin.evaluation.model.whole_model import build as FB
from merlin.runtime.execution.model_inputs import strict_tree_sha256

# ---- relink ------------------------------------------------------------------------------------


@I.timed_event("phase", "relink artifact verification")
def _relink_artifacts(build_dir: Path, elf: Path, recipe_sha256: str) -> tuple[dict, dict]:
    """Reopen the existing complete recipe and the driver's shared preparation inputs."""
    from merlin.compiler.backend import providers as provided
    from merlin.compiler.lower.c import weights as blob_weights
    from merlin.compiler.lower.model.compilation_recipe import FILENAME, verify_completed_recipe
    from merlin.compiler.lower.options import quant_hoist
    from merlin.evaluation.model.whole_model.build import FullBuildError, sha256

    recipe = verify_completed_recipe(build_dir / FILENAME, executable=elf, expected_recipe_sha256=recipe_sha256)
    # The weight-panel plan is the selected vector support provider's preparation input.
    weight_panel = provided.optional_component("weight_panel")
    files = [
        *sorted((build_dir / "cgen").glob("*.h")),
        *(
            build_dir / name
            for name in (
                blob_weights.PLAN_FILE,
                blob_weights.VALUES_FILE,
                quant_hoist.PLAN_FILE,
                quant_hoist.VALUES_FILE,
                *((weight_panel.PLAN_FILE,) if weight_panel is not None else ()),
            )
        ),
    ]
    metadata = {}
    for path in files:
        if any(member.is_symlink() for member in (path, *path.parents)):
            raise FullBuildError(f"relink metadata contains a symlink: {path}")
        metadata[str(path)] = sha256(path) if path.exists() else None
    return {"schema": "merlin.full_model_relink.v1", "recipe_sha256": recipe_sha256, "metadata": metadata}, recipe


def _relink_artifacts_match(record) -> bool:
    from merlin.evaluation.model.whole_model.build import FullBuildError

    if "relink_binding" not in record:
        return False
    binding = record.get("relink_binding")
    if binding is None:
        recipe = Path(record["build_dir"]) / "compilation_recipe.json"
        return not recipe.exists() and not recipe.is_symlink()  # Executable-only builds cannot relink.
    try:
        current, _ = _relink_artifacts(Path(record["build_dir"]), Path(record["elf"]), binding["recipe_sha256"])
        return current == binding
    except (KeyError, TypeError, OSError, ValueError, FullBuildError):
        return False


def _verify_relink(record: dict, stage: Path, *, inputs_current: bool = True) -> dict:
    """Require current origin inputs and actual reusable products before launching a tool."""
    from merlin.evaluation.model.whole_model.build import (
        BuildSettings,
        FullBuildError,
        _build_inputs,
        _lowering_identity,
        _profile_table_matches,
    )

    try:
        binding, key = record["relink_binding"], record["key"]
        if not binding:
            raise FullBuildError("build has no checked relink recipe; rebuild through the ordinary compiler")
        cache = Path(record["cache_directory"])
        stored = json.loads((cache / "build.json").read_text())
        for name in (
            "key",
            "cache_directory",
            "build_dir",
            "elf",
            "elf_sha256",
            "build_hash",
            "relink_binding",
            "weights_bin_bytes",
            "weights_bin_sha256",
            "device_routed",
        ):
            if record[name] != stored[name]:
                raise FullBuildError("relink record differs from its stored build origin: " + name)
        if Path(stage).resolve() != Path(key["stage"]):
            raise FullBuildError("relink capture differs from the selected build")
        settings = BuildSettings(**key["settings"])
        inputs = key["inputs"]
        submission = inputs.get("device_package", {}).get("path")
        weights = inputs.get("weights_source", {}).get("path")
        if (
            inputs_current
            and _build_inputs(stage, submission, settings, cache, key["offload"], weights, key["target"]) != inputs
        ):
            raise FullBuildError("relink build inputs changed; rebuild the image")
        current, recipe = _relink_artifacts(Path(record["build_dir"]), Path(record["elf"]), binding["recipe_sha256"])
        if current != binding or _lowering_identity() != key["lowering"]:
            raise FullBuildError("relink preparation or compiler selection changed; rebuild the image")
        if settings.op_profile and not _profile_table_matches(record):
            raise FullBuildError("relink operation profile table changed; rebuild the image")
        return recipe
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise FullBuildError(f"relink origin or recipe changed; rebuild the image: {error}") from error


# --------------------------------------------------------------------------------------------- relink


def _recipe_command(recipe: dict, output: Path) -> dict:
    from merlin.evaluation.model.whole_model.build import FullBuildError

    rows = [row for row in recipe["commands"] if Path(row["output"]["path"]) == output]
    if len(rows) != 1:
        raise FullBuildError(f"the compilation recipe records {len(rows)} commands producing {output}")
    return rows[0]


def _run(argv: list[str], cwd: str):
    from merlin.evaluation.model.whole_model.build import FullBuildError

    with compute_step():
        done = proc.run_checked(argv, cwd=cwd, error=FullBuildError)
    return done


def _shadow(stage: Path, inputs: dict[str, np.ndarray], directory: Path) -> Path:
    from merlin.evaluation.model.whole_model.build import SHADOWED, FullBuildError

    shadow = directory / "capture"
    shadow.mkdir(parents=True, exist_ok=True)
    for item in Path(stage).iterdir():
        link = shadow / item.name
        if item.name in SHADOWED or link.exists() or link.is_symlink():
            continue
        link.symlink_to(item.resolve())
    expected = np.load(Path(stage) / "inputs.npz")
    arrays = {}
    for key in expected.files:
        if key not in inputs:
            raise FullBuildError(f"{stage.name}: no bound value for model input {key}")
        value = np.asarray(inputs[key])
        if tuple(value.shape) != tuple(expected[key].shape):
            raise FullBuildError(f"{stage.name}: input {key} shape {value.shape} != capture {expected[key].shape}")
        arrays[key] = value.astype(expected[key].dtype, copy=False)
    np.savez(shadow / "inputs.npz", **arrays)
    return shadow


def relink(
    record: dict[str, Any],
    stage: Path,
    inputs: dict[str, np.ndarray],
    directory: Path,
    *,
    admission=None,
) -> Path:
    """The build's image with ``inputs`` (``{"in0": ...}``) embedded instead of the capture's; its ELF path."""
    import fcntl

    from merlin.evaluation.model.whole_model.build import FullBuildError, _write_timings

    if type(record) is NativeBuild:
        return _relink_native(record, stage, inputs, directory, admission=admission)
    cache = record.get("cache_directory")
    if not isinstance(cache, str):
        raise FullBuildError("build has no checked relink origin; rebuild through the ordinary compiler")
    lock = Path(cache) / ".lock"
    if any(path.is_symlink() for path in (lock, *lock.parents)) or not lock.is_file():
        raise FullBuildError("relink build lock is absent or indirect; rebuild the image")
    with lock.open("r") as stream:
        with I.observe_timings() as events:
            with I.timed_event("phase", "model image relink lock acquisition"):
                fcntl.flock(stream, fcntl.LOCK_EX)
            with I.timed_event("phase", "model image relink"):
                result = _relink(record, stage, inputs, directory, admission=admission)
        _write_timings(directory, events)
        return result


def _relink(record, stage, inputs, directory, *, admission):
    from merlin.common.paths import module_source_path
    from merlin.compiler.lower.c import runtime as c_runtime
    from merlin.compiler.lower.model.compilation_recipe import CompilationRecipe, verify_completed_recipe
    from merlin.evaluation.model.whole_model.build import FullBuildError, _cached_admission, sha256

    build_dir = Path(record["build_dir"])
    directory = Path(directory).absolute()
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise FullBuildError("relink output contains a symlink")
    for root in (Path(stage).resolve(), build_dir.resolve(), Path(record.get("cache_directory", build_dir)).resolve()):
        if directory.resolve().is_relative_to(root) or root.is_relative_to(directory.resolve()):
            raise FullBuildError("relink output overlaps a capture or original build")
    recipe = _verify_relink(record, Path(stage))
    admission_selection = None
    host_required = record["key"].get("host_admission_selected", False)
    if type(host_required) is not bool:
        raise FullBuildError("relink host admission selection must be an exact boolean")
    if record["device_routed"] or host_required or admission is not None:
        from merlin.target.contract.build.object_admission import LinkedElfAdmissionService

        if type(admission) is not LinkedElfAdmissionService:
            raise FullBuildError("relinking requires the currently selected linked ELF admission service")
        admission_selection = admission.verify(record["key"]["target"])
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    shadow = _shadow(Path(stage), inputs, directory)
    cgen = directory / "cgen"
    cgen.mkdir()
    for item in (build_dir / "cgen").iterdir():  # what the driver includes beside the generated files
        if item.suffix == ".h" and item.name not in ("model_io.h", "model_gen.h"):
            shutil.copy2(item, cgen / item.name)
    with I.timed_event("phase", "relink driver and weight generation"):
        info = c_runtime.generate(
            shadow, cgen, shadow / "inputs.npz", prepared_dir=build_dir, max_session_steps=1, materialize_weights=False
        )
    weights = cgen / "weights.bin"
    if (
        weights.exists()
        or info["weights_bytes"] != record["weights_bin_bytes"]
        or info["weights_sha256"] != record["weights_bin_sha256"]
    ):
        raise FullBuildError("the regenerated weights blob differs from the build's; refusing to relink")
    if (cgen / "model_call.c").read_bytes() != (build_dir / "cgen" / "model_call.c").read_bytes():
        raise FullBuildError("the regenerated argument-table call differs from the build's")
    _verify_relink(record, Path(stage), inputs_current=False)
    main = _recipe_command(recipe, build_dir / "model_main.o")
    argv = [directory / "model_main.o" if a == str(build_dir / "model_main.o") else a for a in main["argv"]]
    argv = [str(cgen) if a == str(build_dir / "cgen") else a for a in argv]
    observed = CompilationRecipe(directory, producer=module_source_path(__name__))
    main_inputs = [
        cgen / Path(item["path"]).name
        if Path(item["path"]).parent == build_dir / "cgen" and Path(item["path"]).suffix == ".h"
        else Path(item["path"])
        for item in main["inputs"]
    ]
    observed.run(
        argv,
        runner=_run,
        inputs=list(dict.fromkeys([*main_inputs, *sorted(cgen.glob("*.h"))])),
        output=directory / "model_main.o",
        cwd=Path(main["cwd"]),
    )
    _verify_relink(record, Path(stage), inputs_current=False)
    link = _recipe_command(recipe, Path(record["elf"]))
    elf = directory / "model.elf"
    swap = {str(build_dir / "model_main.o"): str(directory / "model_main.o"), record["elf"]: str(elf)}
    observed.run(
        [swap.get(a, a) for a in link["argv"]],
        runner=_run,
        inputs=[Path(swap.get(item["path"], item["path"])) for item in link["inputs"]],
        output=elf,
        cwd=Path(link["cwd"]),
    )
    if not elf.is_file() or elf.is_symlink():
        raise FullBuildError("relink command produced no ordinary executable")
    _verify_relink(record, Path(stage), inputs_current=not record["device_routed"])
    if admission_selection is not None:
        if admission.verify(record["key"]["target"]) != admission_selection:
            raise FullBuildError("linked ELF admission selection changed during relinking")
        selected, result = _cached_admission({**record, "elf": str(elf)}, admission, record["key"]["target"], directory)
        _verify_relink(record, Path(stage))
        selected.revalidate(elf=elf, result=result, target=record["key"]["target"])
    observed.completed(elf)
    verify_completed_recipe(observed.path, executable=elf, expected_recipe_sha256=sha256(observed.path))
    return elf


def _relink_native(
    record: dict[str, Any], stage: Path, inputs: dict[str, np.ndarray], directory: Path, *, admission=None
) -> Path:
    """The build's image with ``inputs`` (``{"in0": ...}``) embedded instead of the capture's; its ELF path."""
    from merlin.compiler.lower.model.compilation_recipe import CompilationRecipe
    from merlin.evaluation.model.whole_model.build import FullBuildError, sha256
    from merlin.target.contract.build.object_admission import LinkedElfAdmissionService

    selected_admission = record.admission
    if (
        type(selected_admission) is not LinkedElfAdmissionService
        or admission is not None
        and admission is not selected_admission
    ):
        raise FullBuildError("native relink lost its original live admission owner")
    selected_admission.verify(record["key"]["target"])
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise FullBuildError("native relink output contains a symlink")
    for root in (Path(stage).resolve(), Path(record["build_dir"]).resolve()):
        if directory.resolve().is_relative_to(root) or root.is_relative_to(directory.resolve()):
            raise FullBuildError("native relink output overlaps capture or original build")
    if Path(record["compilation_recipe"]["path"]) != Path(record["build_dir"]) / "compilation_recipe.json":
        raise FullBuildError("native relink has a foreign original recipe")
    require_preparation_selection(verified_recipe(record))

    build_dir = Path(record["build_dir"])
    if str(Path(stage).resolve()) != record["key"]["stage"]:
        raise FullBuildError("bound inputs name another model capture")
    for name, digest in record["runtime_abi_sha256"].items():
        if sha256(Path(stage) / name) != digest:
            raise FullBuildError("the original runtime input ABI changed after the model build")
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    shadow = _shadow(Path(stage), inputs, directory)
    cgen = directory / "cgen"
    cgen.mkdir()
    for item in (build_dir / "cgen").iterdir():  # what the driver includes beside the generated files
        if item.suffix == ".h" and item.name not in ("model_io.h", "model_gen.h"):
            shutil.copy2(item, cgen / item.name)
    generate(record, Path(stage), shadow, cgen)
    recipe = json.loads((build_dir / "compilation_recipe.json").read_text())
    if sha256(build_dir / "compilation_recipe.json") != record["compilation_recipe"]["sha256"]:
        raise FullBuildError("the original compilation recipe changed before input rebinding")
    observed = CompilationRecipe(directory, producer=Path(__file__))
    main = _recipe_command(recipe, build_dir / "model_main.o")
    argv = [directory / "model_main.o" if a == str(build_dir / "model_main.o") else a for a in main["argv"]]
    argv = [str(cgen) if a == str(build_dir / "cgen") else a for a in argv]

    def rebind(path):
        path = Path(path)
        return cgen / path.name if path.parent == build_dir / "cgen" else path

    def invoke(original, argv, inputs, output, changed):
        def checked(argv, cwd):
            current = observed.record["commands"][-1]
            if current["executable"] != original["executable"]:
                raise FullBuildError("an original model build tool changed before input rebinding")
            expected = {row["path"]: row for row in original["inputs"] if row["path"] not in changed}
            actual = {row["path"]: row for row in current["inputs"]}
            if any(actual.get(path) != identity for path, identity in expected.items()):
                raise FullBuildError("an immutable original build input changed before input rebinding")
            return _run(argv, cwd)

        return observed.run(argv, runner=checked, cwd=Path(original["cwd"]), inputs=inputs, output=output)

    main_inputs = {rebind(row["path"]) for row in main["inputs"]} | set(cgen.glob("*.h"))
    invoke(
        main,
        argv,
        sorted(main_inputs),
        directory / "model_main.o",
        {row["path"] for row in main["inputs"] if Path(row["path"]).parent == build_dir / "cgen"},
    )
    link = _recipe_command(recipe, Path(record["elf"]))
    elf = directory / "model.elf"
    swap = {str(build_dir / "model_main.o"): str(directory / "model_main.o"), record["elf"]: str(elf)}
    linked = invoke(
        link,
        [swap.get(a, a) for a in link["argv"]],
        [Path(swap.get(row["path"], row["path"])) for row in link["inputs"]],
        elf,
        {str(build_dir / "model_main.o")},
    )
    if record.get("native_qualification") and recipe.get("link_suppliers"):
        observed.record_link_suppliers(
            {
                symbol: Path(item["selected_input"]["path"])
                for symbol, item in recipe["link_suppliers"]["symbols"].items()
            },
            linked,
        )
    verified_recipe(record)
    observed.completed(elf)

    if type(record) is NativeBuild:
        record.admit_relink(elf, directory)
    return elf


# ---- full_build_identity -----------------------------------------------------------------------
# Content correspondence for reusable whole-model builds, separate from qualification.
#
# The identity binds direct selected tools, project source/runtime, capture bytes and
# build settings. Observed compilation recipes additionally reopen actual explicit
# producer inputs and outputs. Neither claims a hermetic transitive toolchain closure.


def file_identity(path: Path) -> dict:
    path = Path(shutil.which(str(path)) or path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": content_digest(path)}


def host_math_identity(policy: str) -> dict:
    """Bind the selected recipe's actual source bytes and flags, never an unselected C file."""
    import hashlib

    from merlin.runtime.host_math import host_math_recipe

    recipe = host_math_recipe(policy)
    return {
        "policy": policy,
        "source_sha256": hashlib.sha256(recipe.source.encode("utf-8")).hexdigest()
        if recipe.source is not None
        else None,
        "compile_flags": list(recipe.compile_flags),
        "link_flags": list(recipe.link_flags),
    }


def build_material(stage, target, label, submission, submission_sha256, settings, offload) -> dict:
    """Read build inputs once per build request, never once per recurrent input."""
    from merlin_rvv.src.rvv.runtime import spike

    from merlin.common.paths import repo_root, runtime_dir
    from merlin.compiler.lower.model import toolchain
    from merlin.runtime.backends.baremetal import spike_launcher as _merlin_spike_launcher_1
    from merlin.runtime.execution.model_inputs import strict_tree_sha256

    gcc = _merlin_spike_launcher_1.gcc_path()
    tools = {
        "clang": toolchain.clang(),
        "mlir_optimizer": toolchain.llvm_install() / "bin" / "mlir-opt",
        "translate": toolchain.mlir_translate(),
        "optimizer": toolchain.llvm_opt(),
        "compiler_python": toolchain.compiler_python(),
        "gcc": gcc,
        "linker": gcc.with_name("riscv64-unknown-elf-ld"),
        "objcopy": toolchain.objcopy(),
        "nm": toolchain.nm(),
        "readelf": toolchain.readelf(),
    }
    selected = {}
    for name, path in tools.items():
        resolved = Path(shutil.which(str(path)) or path)
        selected[name] = file_identity(resolved) if resolved.is_file() else {"path": str(resolved), "missing": True}
    source = repo_root() / "src" / "merlin"
    # package_digest includes bytecode; select source files explicitly so importing
    # a module cannot change the identity of the same compiler implementation.
    project = {str(path.relative_to(source)): content_digest(path) for path in sorted(source.rglob("*.py"))}
    return {
        "schema": "merlin.evaluation.model.full_build_material.v2",
        "scope": "project source and direct selected build inputs; transitive toolchain closure unknown",
        "stage": str(Path(stage).resolve()),
        "capture": strict_tree_sha256(Path(stage)),
        "target": target,
        "label": label,
        "submission_sha256": submission_sha256 if offload else None,
        "submission_content_sha256": package_digest(Path(submission)) if offload else None,
        "host_package_sha256": package_digest(settings.host_package),
        "settings": settings.record(),
        "host_math": host_math_identity(settings.host_math_policy),
        "board_catalog": file_identity(settings.board_catalog),
        "host_dts": file_identity(settings.host_dts),
        "source": project,
        "runtime_sha256": package_digest(runtime_dir()),
        "tools": selected,
        "compiler_driver": toolchain._env("MERLIN_COMPILER_DRIVER", "python"),
        "dotenv_sha256": content_digest(repo_root() / ".env") if (repo_root() / ".env").is_file() else None,
        "target_inputs": target_inputs(),
        "environment_sha256": content_digest_bytes(
            {
                name: value
                for name, value in os.environ.items()
                if name.startswith("MERLIN_")
                or name in {"PATH", "PYTHONPATH", "CFLAGS", "CXXFLAGS", "LDFLAGS", "CPATH", "LIBRARY_PATH"}
            }
        ),
        "offload": offload,
    }


def content_digest_bytes(value) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verified_recipe(record: dict) -> dict:
    """Reject legacy/changed evidence instead of retrospectively minting custody."""
    from merlin.compiler.lower.model.compilation_recipe import verify_completed_recipe

    evidence = record["compilation_recipe"]
    return verify_completed_recipe(
        Path(evidence["path"]), executable=Path(record["elf"]), expected_recipe_sha256=evidence["sha256"]
    )


# ---- full_build_inputs -------------------------------------------------------------------------
# Reuse immutable single-forward C artifacts while rebinding typed runtime inputs.
#
# The first call uses the ordinary generator and proves weights/call agreement with
# the observed model build. Later calls replace only declarations independently
# derived from the original argument manifest, signature and input-order mapping.
# No candidate metadata grants authority to change an argument or a weight.


def _bindings(stage: Path, inputs: Path) -> list[dict]:
    from merlin.compiler.lower.c import runtime as c_runtime

    # Older captures without an explicit order retain ordinary regeneration.
    if not (stage / "input_order.json").is_file():
        raise ValueError("an explicit input_order.json is required for runtime template reuse")
    order = {name: int(index) for name, index in json.loads((stage / "input_order.json").read_text()).items()}
    manifest = json.loads((stage / "weights.safetensors.manifest.json").read_text())
    signature = c_runtime.parse_forward_signature(stage / "model.mlir")
    with np.load(inputs) as arrays:
        bindings, positional = [], 0
        for index, (shape, dtype) in enumerate(signature):
            entry = manifest[str(index)]
            name = entry.get("name", "")
            if entry["kind"] in {"param", "buffer"} or name.startswith("c_lifted_tensor_"):
                continue
            if name in order and f"in{order[name]}" in arrays.files:
                key = f"in{order[name]}"
            else:
                key = f"in{positional}"
                positional += 1
            value = np.asarray(arrays[key])
            if list(value.shape) != list(shape):
                raise ValueError(f"runtime template input {key} disagrees with the original signature")
            bindings.append({"argument": index, "key": key, "dtype": dtype, "shape": list(shape)})
    return bindings


def _declaration(binding: dict, arrays) -> str:
    from merlin.compiler.lower.c import runtime as c_runtime

    value = np.asarray(arrays[binding["key"]])
    if list(value.shape) != binding["shape"]:
        raise ValueError("bound runtime input shape changed")
    return (
        f"static {c_runtime.C_OF[binding['dtype']]} merlin_in_{binding['argument']}[] = "
        + "{"
        + c_runtime._embed_array(np.ascontiguousarray(value), binding["dtype"])
        + "};\n"
    )


def _render(template: str, bindings: list[dict], arrays, *, initial: bool) -> str:
    from merlin.compiler.lower.c import runtime as c_runtime

    prefixes = {f"static {c_runtime.C_OF[row['dtype']]} merlin_in_{row['argument']}[] = {{": row for row in bindings}
    seen, lines = set(), []
    for line in template.splitlines(keepends=True):
        matches = [prefix for prefix in prefixes if line.startswith(prefix)]
        if not matches:
            lines.append(line)
            continue
        (prefix,) = matches
        if prefix in seen or not line.endswith("};\n"):
            raise ValueError("runtime template declaration is duplicate or malformed")
        seen.add(prefix)
        replacement = _declaration(prefixes[prefix], arrays)
        if initial and replacement != line:
            raise ValueError("runtime template input does not match the ordinary C generator")
        lines.append(replacement)
    if seen != set(prefixes):
        raise ValueError("runtime template has no unique declaration for every bound input")
    return "".join(lines)


def generate(record: dict, stage: Path, shadow: Path, cgen: Path) -> None:
    """Generate current I/O; render large immutable weights only once per model build."""
    import fcntl

    from merlin.compiler.lower.c import runtime as c_runtime
    from merlin.evaluation.model.whole_model.build import FullBuildError, sha256

    build_dir = Path(record["build_dir"])
    stamp = {"key": content_digest_bytes(record["key"]), "recipe": record["compilation_recipe"]}
    # Fresh protected adoptions can share the original product directory while
    # binding different current owners and environments. Keep their templates
    # separate without changing or accepting another adoption's exact stamp.
    template = build_dir / ("model_check_single_forward_" + content_digest_bytes(stamp))
    with (build_dir / ".model_check_inputs.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = template / "template.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        if manifest is not None and manifest.get("stamp") != stamp:
            raise FullBuildError("single-forward runtime template belongs to another model build")
        if manifest is not None and manifest.get("bindings") != _bindings(stage, shadow / "inputs.npz"):
            raise FullBuildError("single-forward runtime template changed original input ownership")
        if manifest is None:
            shutil.rmtree(template, ignore_errors=True)
            template.mkdir()
            c_runtime.generate(shadow, template, shadow / "inputs.npz", prepared_dir=build_dir, max_session_steps=1)
            weights = template / "weights.bin"
            if weights.stat().st_size != record["weights_bin_bytes"] or sha256(weights) != record["weights_bin_sha256"]:
                raise FullBuildError("the regenerated weights blob differs from the build's; refusing to relink")
            weights.unlink()
            if (template / "model_call.c").read_bytes() != (build_dir / "cgen/model_call.c").read_bytes():
                raise FullBuildError("the regenerated argument-table call differs from the build's")
            try:
                bindings = _bindings(stage, shadow / "inputs.npz")
            except ValueError:
                # An old or unusual capture gets the complete ordinary path on
                # every step; no guessed argument ownership enters the fast path.
                for item in template.iterdir():
                    shutil.copy2(item, cgen / item.name)
                shutil.rmtree(template)
                return
            io = (template / "model_io.h").read_text()
            with np.load(shadow / "inputs.npz") as arrays:
                _render(io, bindings, arrays, initial=True)
            manifest = {
                "stamp": stamp,
                "bindings": bindings,
                "files": {item.name: sha256(item) for item in sorted(template.iterdir()) if item.is_file()},
            }
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        for name, digest in manifest["files"].items():
            item = template / name
            if Path(name).name != name or not item.is_file() or sha256(item) != digest:
                raise FullBuildError("single-forward runtime template bytes changed")
            if name != "model_io.h":
                shutil.copy2(item, cgen / name)
        with np.load(shadow / "inputs.npz") as arrays:
            io = _render((template / "model_io.h").read_text(), manifest["bindings"], arrays, initial=False)
        (cgen / "model_io.h").write_text(io)


# ---- full_build_native -------------------------------------------------------------------------
# Import an already observed whole-model build without recompiling its objects.
#
# This is artifact transport, not native qualification. The protected caller must
# first verify the native source, request, instruction and supplier gate and bind
# its producer record in ``qualification``. Structural builds cannot manufacture
# that qualification through this module.


def require_preparation_selection(recipe):
    """Match the current repeat choice to a freshly verified original recipe.

    A current environment cannot label an opposite-arm retained image. The
    ordinary compiler binds this preparation report only when the pass was
    selected, including selected invocations with no eligible pointwise work.
    This check supplies no source, numerical or admission qualification.
    """
    from merlin.common.digest import sha256_bytes
    from merlin.common.record_formats import loads
    from merlin.compiler.lower.source.repeated_pointwise import selected

    preparation = recipe.get("preparation", {})
    observed = preparation.get("repeated_pointwise")
    if (observed is not None) is not selected():
        raise FB.FullBuildError("native image repeated-pointwise selection differs from its original preparation")
    if observed is not None:
        payload = Path(observed["path"]).read_bytes()
        if sha256_bytes(payload) != observed["sha256"]:
            raise FB.FullBuildError("native image repeated-pointwise preparation changed")
        document = loads(payload)
        if not isinstance(document, dict) or document.get("schema") != "merlin.repeated_pointwise_preparation.v1":
            raise FB.FullBuildError("native image has no original repeated-pointwise preparation report")
    from merlin.compiler.lower.quant.host_bounded_rne import PREPARATION, require_report
    from merlin.compiler.lower.quant.host_bounded_rne import selected as bounded_rne_selected

    bounded = preparation.get(PREPARATION)
    if (bounded is not None) is not bounded_rne_selected():
        raise FB.FullBuildError("native image host bounded RNE selection differs from its original preparation")
    if bounded is not None:
        payload = Path(bounded["path"]).read_bytes()
        if sha256_bytes(payload) != bounded["sha256"]:
            raise FB.FullBuildError("native image host bounded RNE preparation changed")
        try:
            require_report(loads(payload), require_combined=True)
        except ValueError as error:
            raise FB.FullBuildError(str(error)) from error


class NativeBuild(dict):
    """JSON-safe build fields plus the actual live, independently selected service."""

    def admit_relink(self, elf, directory):
        result = self.admission.evaluate(
            elf=elf, target=self["key"]["target"], evidence_root=directory / "native-instruction-policy"
        )
        if result["status"] != "accepted":
            raise FB.FullBuildError("input-specific whole-model ELF failed selected instruction policy")
        import json

        (directory / "native_linked_elf.json").write_text(json.dumps(result, indent=2) + "\n")


def adopt(receipt, *, stage, target, label, submission, submission_sha256, settings, qualification, admission=None):
    """Reopen the original build's exact inputs, executable and compilation recipe."""
    stage, submission = Path(stage), Path(submission)
    inputs, output = receipt.get("inputs", {}), receipt.get("output", {})
    expected = {
        "capture": str(stage.resolve()),
        "target": target,
        "board": settings.board,
        "board_catalog_sha256": FB.sha256(settings.board_catalog),
        "dts_sha256": FB.sha256(settings.host_dts),
        "arena_mb": settings.arena_mb,
        "readback": "full",
        "host_math_policy": settings.host_math_policy,
        "respect_captured_quantization_scope": True,
        "host_llvm_bitcode": True,
        "capture_tree": strict_tree_sha256(stage),
        "package_tree": strict_tree_sha256(settings.host_package),
    }
    device = inputs.get("device")
    route = "device_requested_dispatch_unverified" if device is not None else "host_baseline"
    if (
        receipt.get("schema") != "merlin.baremetal-saved-model.v1"
        or receipt.get("status") != "compiled"
        or receipt.get("execution_route") != route
        or any(inputs.get(key) != value for key, value in expected.items())
        or inputs.get("host_elementwise_fusion", False) is not settings.host_elementwise_fusion
        or (device is not None and device.get("package_tree") != strict_tree_sha256(submission))
    ):
        raise FB.FullBuildError("native image does not correspond to the selected full-model build inputs")
    if not isinstance(qualification, dict) or not qualification:
        raise FB.FullBuildError("native image requires a protected qualification binding")
    elf = Path(output.get("elf", ""))
    if not elf.is_file() or elf.is_symlink() or FB.sha256(elf) != output.get("elf_sha256"):
        raise FB.FullBuildError("original native whole-model ELF is absent or changed")
    key = build_material(stage, target, label, submission, submission_sha256, settings, True)
    key["native_qualification"] = qualification
    build_dir = elf.parent
    weights = build_dir / "cgen/weights.bin"
    record = {
        "schema": FB.BUILD_SCHEMA,
        "key": key,
        "status": "linked",
        "elf": str(elf),
        "elf_sha256": output["elf_sha256"],
        "elf_bytes": elf.stat().st_size,
        "sections": FB.elf_sections(elf),
        "build_hash": output["build_hash"],
        "build_dir": str(build_dir),
        "compilation_recipe": output["compilation_recipe"],
        "runtime_abi_sha256": {
            name: FB.sha256(stage / name)
            for name in (
                "model.mlir",
                "weights.safetensors.manifest.json",
                "input_order.json",
            )
            if (stage / name).is_file()
        },
        "weights_bin_sha256": FB.sha256(weights),
        "weights_bin_bytes": weights.stat().st_size,
        "device_routed": device is not None,
        "dedup_weights": bool(device and inputs.get("blob_weights")),
        "seconds": 0.0,
        "reused": True,
        "native_qualification": qualification,
    }
    recipe = verified_recipe(record)
    require_preparation_selection(recipe)
    marked = "host_bounded_rne" in recipe.get("preparation", {})
    declaration = {"host_isa": "rv64gc", "combine_clamp": True} if marked else None
    if inputs.get("host_bounded_rne") != declaration:
        raise FB.FullBuildError("native image host bounded RNE declaration differs from its original preparation")
    if admission is not None:
        from merlin.target.contract.build.object_admission import LinkedElfAdmissionService

        if type(admission) is not LinkedElfAdmissionService:
            raise FB.FullBuildError("native image lost its original instruction admission service")
        admission.verify(target)
        record = NativeBuild(record)
        record.admission = admission
    return record
