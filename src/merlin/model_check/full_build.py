"""One program's linked whole-model image: built once per submission, relinked per input, run on Spike.

The build is Merlin's ordinary whole-model build (``compile_saved_model``, ``run="none"``, full readback,
group profile) with the submission as the device package, routed exactly as a full-model qualification
routes it (every source-admitted group, :func:`merlin.compile.route_before_build.plan_before_build`).

A new input needs no re-lowering. The build records every compile and link command it ran
(``build/compilation_recipe.json``); the inputs live only in the generated ``cgen/model_io.h``, which
only the driver ``model_main.c`` includes. Relinking for an input therefore regenerates ``cgen`` from a
shadow of the capture whose ``inputs.npz`` is the bound input, checks that everything else it generated
(weights blob, argument table call) is byte-identical to the build's, recompiles ``model_main.o`` with
the recipe's own command and repeats the recipe's link. The shadow carries no session contract, so the
image runs exactly one forward on exactly the bound inputs (a captured session's carried state and
stream are inputs like any other here; the chain supplies them).
"""

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

from merlin.common import invocation_record as I
from merlin.common import proc
from merlin.model_check.compute_execution import compute_step

BUILD_SCHEMA = "merlin.model_check.full_build.v1"
#: Shadowed capture files: the bound input replaces the first; the session files are left out.
SHADOWED = ("inputs.npz", "session_contract.yaml", "session_inputs.npz")


class FullBuildError(RuntimeError):
    """A program image could not be built, relinked, run or read back."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def elf_sections(path: Path) -> dict[str, int]:
    """``{section: bytes}`` of every allocated section of a little-endian ELF64 image."""
    with Path(path).open("rb") as stream:
        header = stream.read(64)
        if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
            raise FullBuildError(f"{path} is not a little-endian ELF64 image")
        (shoff,) = struct.unpack_from("<Q", header, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from("<HHH", header, 0x3A)
        stream.seek(shoff)
        table = stream.read(shentsize * shnum)
        rows = [struct.unpack_from("<IIQQQQIIQQ", table, i * shentsize) for i in range(shnum)]
        names_offset, names_size = rows[shstrndx][4], rows[shstrndx][5]
        stream.seek(names_offset)
        names = stream.read(names_size)
    out: dict[str, int] = {}
    for name_at, _kind, flags, _addr, _offset, size, *_rest in rows:
        if flags & 0x2 and size:  # SHF_ALLOC
            name = names[name_at : names.index(b"\0", name_at)].decode()
            out[name] = out.get(name, 0) + int(size)
    return out


# ------------------------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class BuildSettings:
    """How a model's programs are built (the models file's ``full`` blocks; contract §2)."""

    board: str
    board_catalog: Path
    host_dts: Path
    datapath: str
    arena_mb: int
    host_package: Path
    dedup_weights: bool = True
    op_profile: bool = False

    def __post_init__(self):
        if type(self.op_profile) is not bool:
            raise FullBuildError("operation profiling must be an explicit Boolean")

    @classmethod
    def of(
        cls,
        models,
        model,
        host_package: Path,
        *,
        dedup_weights: bool = True,
        op_profile: bool = False,
    ) -> BuildSettings:
        from merlin.common.paths import repo_root

        common = dict(models.full or {})
        own = dict(model.extra.get("full") or {})
        missing = [k for k in ("board", "board_catalog", "host_dts", "datapath") if k not in common]
        if missing or "arena_mb" not in own:
            raise FullBuildError(f"models file declares no full-mode build for {model.name}: {missing or ['arena_mb']}")

        def path(value: str) -> Path:
            expanded = Path(os.path.expandvars(str(value)))
            return expanded if expanded.is_absolute() else repo_root() / expanded

        return cls(
            board=str(common["board"]),
            board_catalog=path(common["board_catalog"]),
            host_dts=path(common["host_dts"]),
            datapath=str(common["datapath"]),
            arena_mb=int(own["arena_mb"]),
            host_package=Path(host_package),
            dedup_weights=dedup_weights,
            op_profile=op_profile,
        )

    def record(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------------------------- build


def _profile_table_matches(record: dict) -> bool:
    """A profiled image can be reused only with its original adjacent table."""
    table = record.get("op_profile_table")
    if (
        not isinstance(table, dict)
        or not isinstance(table.get("path"), str)
        or not isinstance(record.get("build_dir"), str)
    ):
        return False
    path = Path(table["path"])
    return (
        path == Path(record["build_dir"]) / "op_profile_table.json"
        and path.is_file()
        and not path.is_symlink()
        and sha256(path) == table.get("sha256")
    )


def _lowering_sources() -> dict[str, Path]:
    """Direct host pass owners, without importing their implementations.

    Bind both source families so adding or editing a pass cannot reuse a prior
    default. This is not a transitive dependency or toolchain closure.
    """
    from merlin.common.paths import module_source_path

    sources = {}
    for namespace, leaf in (("merlin.llvmlower", "lower"), ("merlin.transforms", "pipeline")):
        root = module_source_path(namespace + "." + leaf).parent
        for path in sorted(root.rglob("*.py")):
            sources[namespace + "/" + path.relative_to(root).as_posix()] = path
    for name in (
        __name__,
        "merlin.model_check.full",
        "merlin.model_check.compute_execution",
        "merlin.compile.baremetal_model",
        "merlin.compile.model_execution_inputs",
        "merlin.frontends.linalg_mlir",
        "merlin.targetgen.rtl.facts",
        "merlin.common.invocation_record",
        "merlin.common.proc",
        "merlin.common.cpu_pool",
        "merlin.common.digest",
        "merlin.common.jsonio",
        "merlin.common.mlir_query",
        "merlin.targetgen.target_registry",
        "merlin.runtime.backends.zephyr_model",
        "merlin.runtime.backends.spike_model",
        "merlin.runtime.operation_profile",
        "merlin.runtime.whole_model_readback",
    ):
        sources[name] = module_source_path(name)
    return sources


@I.timed_event("phase", "compiler source identity")
def _lowering_identity() -> dict:
    from merlin.common.paths import runtime_dir
    from merlin.llvmlower.codegen_env import cache_digest
    from merlin.model_check.cache import target_inputs

    return {
        "schema": "merlin.full_model_lowering_selection.v1",
        "project_controls_sha256": cache_digest(),
        "direct_sources_sha256": {name: sha256(path) for name, path in _lowering_sources().items()},
        "selected_target_files": target_inputs(),
        "native_abi_source_sha256": sha256(runtime_dir() / "abi" / "mlir_runtime.c"),
    }


@I.timed_event("phase", "build input identity")
def _build_inputs(stage, submission, settings, directory, offload, weights_source, target) -> dict:
    """Stream current explicit inputs; paths and stat metadata cannot authorize reuse.

    Reuse the saved-model compiler's strict tree reader, including policy, state
    and unknown files. Read existing selected/scoped facts without extraction.
    This is not a transitive provider, native resource or toolchain closure.
    """
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.model_check.cache import TARGET_INPUTS
    from merlin.targetgen.rtl.facts import find_facts, load_facts
    from merlin.targetgen.target_registry import resolve

    def path(value):
        lexical = Path(value).absolute()
        if any(member.is_symlink() for member in (lexical, *lexical.parents)):
            raise FullBuildError(f"build input contains a symlink: {value}")
        resolved = lexical.resolve(strict=True)
        if resolved == directory.resolve() or directory.resolve().is_relative_to(resolved):
            raise FullBuildError(f"build input contains the image cache: {resolved}")
        if resolved.is_relative_to(directory.resolve()):
            raise FullBuildError(f"build input is inside the image cache: {resolved}")
        return resolved

    try:
        trees = {"capture": stage, "host_package": settings.host_package}
        if offload:
            trees["device_package"] = submission
        inputs = {name: {"path": str(root := path(value)), **strict_tree_sha256(root)} for name, value in trees.items()}
        files = {"board_catalog": settings.board_catalog, "host_dts": settings.host_dts}
        files.update({name: value for name in TARGET_INPUTS if (value := os.environ.get(name))})
        if weights_source is not None:
            files["weights_source"] = weights_source
        inputs.update(
            {name: {"path": str(root := path(value)), "sha256": sha256(root)} for name, value in files.items()}
        )
        try:
            facts = load_facts(target, regenerate=False)
        except FileNotFoundError:
            if os.environ.get("MERLIN_RTL_FACTS"):
                raise
            facts = None  # Unavailable remains unknown; cache lookup never extracts RTL.
        inputs["selected_facts_sha256"] = (
            hashlib.sha256(json.dumps(facts, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if facts is not None
            else None
        )
        artifact = find_facts(target) if facts is not None else None
        inputs["selected_facts_file"] = (
            {"path": str(root := path(artifact)), "sha256": sha256(root)} if artifact is not None else None
        )
        selected = resolve(target)  # Metadata only: never import an executable provider.
        try:
            software = selected.load_contract()
        except FileNotFoundError:
            if os.environ.get("MERLIN_TARGET_CONTRACT"):
                raise
            software = None
        inputs["selected_capability_sha256"] = (
            hashlib.sha256(json.dumps(software, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if software is not None
            else None
        )
        capability = selected.capability_contract_path
        inputs["selected_capability_file"] = (
            {"path": str(root := path(capability)), "sha256": sha256(root)}
            if capability != selected.contract_path
            else None
        )
        inputs["provider_metadata"] = {
            name: (
                {"path": str(root := path(value)), "sha256": sha256(root)}
                if value.exists() or value.is_symlink()
                else None
            )
            for name, value in (("contract", selected.contract_path), ("dialect_plan", selected.dialect_plan_path))
        }
        return {"schema": "merlin.full_model_build_inputs.v1", **inputs}
    except (OSError, ValueError) as error:
        raise FullBuildError(f"cannot establish current full-model build inputs: {error}") from error


def _cached_admission(record, admission, target, directory):
    """Evaluate today's selected policy on actual cached bytes, never an old verdict."""
    if not record["device_routed"] and admission is None:
        return None, None
    if admission is None:
        from merlin.model_check.admission import service

        admission = service(target)
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    if type(admission) is not LinkedElfAdmissionService:
        raise FullBuildError("linked image requires an explicit linked ELF admission service")
    result = admission.evaluate(elf=Path(record["elf"]), target=target, evidence_root=directory / "cache-admission")
    if result["status"] != "accepted":
        raise FullBuildError("current linked ELF admission refused the linked image")
    return admission, result


@I.timed_event("phase", "relink artifact verification")
def _relink_artifacts(build_dir: Path, elf: Path, recipe_sha256: str) -> tuple[dict, dict]:
    """Reopen the existing complete recipe and the driver's shared preparation inputs."""
    from merlin.llvmlower import blob_weights, quant_hoist, weight_panel
    from merlin.llvmlower.compilation_recipe import FILENAME, verify_completed_recipe

    recipe = verify_completed_recipe(build_dir / FILENAME, executable=elf, expected_recipe_sha256=recipe_sha256)
    files = [
        *sorted((build_dir / "cgen").glob("*.h")),
        *(
            build_dir / name
            for name in (
                blob_weights.PLAN_FILE,
                blob_weights.VALUES_FILE,
                quant_hoist.PLAN_FILE,
                quant_hoist.VALUES_FILE,
                weight_panel.PLAN_FILE,
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


@I.timed_event("phase", "device routing")
def _routing(target: str, stage: Path, submission: Path, model: str, settings: BuildSettings, admission):
    """The submission's device routing over every source-admitted group, or ``None`` (host only)."""
    from merlin.compile.route_before_build import plan_before_build

    try:
        plan = plan_before_build(
            target,
            (stage / "model.mlir").read_text(encoding="utf-8"),
            datapath=settings.datapath,
            device_package=str(submission),
            model=model,
            capture=stage / "model.mlir",
            granularity="group",
            linked_elf_admission=admission,
            require_all_accelerator_groups=True,
        )
    except ValueError as error:
        if "no admitted accelerator groups" in str(error):
            return None
        raise
    routing = plan.get("device_routing")
    if routing is None:
        raise FullBuildError(f"{model}: no device routing: {plan.get('device_routing_why')}")
    return routing


def build(
    stage: Path,
    *,
    target: str,
    label: str,
    submission: Path,
    submission_sha256: str,
    settings: BuildSettings,
    directory: Path,
    admission=None,
    offload: bool = True,
    weights_source: Path | None = None,
) -> dict[str, Any]:
    """Build ``stage``'s image under ``directory`` (reused when its record matches) and return its record.

    ``offload=False`` builds the host twin: the same recipe and host code with every group on the host
    and no device object of the submission (the differential's reference image).

    Two checks of one submission share the directory: the first builds while holding its lock, the
    second waits and reuses the record."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("w") as lock:
        with I.observe_timings() as events:
            with I.timed_event("phase", "model image build lock acquisition"):
                fcntl.flock(lock, fcntl.LOCK_EX)
            with I.timed_event("phase", "model image build or lookup"):
                result = _build(
                    stage,
                    target,
                    label,
                    submission,
                    submission_sha256,
                    settings,
                    directory,
                    admission,
                    offload,
                    weights_source,
                )
        result["timing"] = _write_timings(directory, events)
        return result


def _write_timings(directory, events):
    root = Path(directory) / "timings"
    if root.is_symlink():
        raise FullBuildError("timing output must not be a symlink")
    root.mkdir(exist_ok=True)
    path = root / (uuid.uuid4().hex + ".json")
    path.write_text(json.dumps(I.timing_document(events), indent=1) + "\n")
    return {"path": str(path), "sha256": sha256(path)}


def _build(
    stage,
    target,
    label,
    submission,
    submission_sha256,
    settings,
    directory: Path,
    admission,
    offload,
    weights_source=None,
):
    from merlin.compile.baremetal_model import compile_saved_model

    lookup_started = time.monotonic()
    if admission is not None:
        from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

        if type(admission) is not LinkedElfAdmissionService:
            raise FullBuildError("build requires an exact selected linked ELF admission service")
        admission_selection = admission.verify(target)
    else:
        admission_selection = None
    key = {
        "stage": str(Path(stage).resolve()),
        "target": target,
        # The host twin depends on no submission byte.
        "submission_sha256": submission_sha256 if offload else None,
        "settings": settings.record(),
        "offload": offload,
        "host_admission_selected": not offload and admission is not None,
        "lowering": _lowering_identity(),
        "inputs": _build_inputs(stage, submission, settings, directory, offload, weights_source, target),
    }
    record_path = directory / "build.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        elf = Path(record.get("elf") or "")
        if (
            record.get("key") == key
            and elf.is_file()
            and not elf.is_symlink()
            and elf.stat().st_size == record.get("elf_bytes")
            and sha256(elf) == record.get("elf_sha256")
            and (not settings.op_profile or _profile_table_matches(record))
            and _relink_artifacts_match(record)
        ):
            if admission is not None and admission.verify(target) != admission_selection:
                raise FullBuildError("linked ELF admission selection changed during cache lookup")
            current_admission, admission_result = _cached_admission(record, admission, target, directory)
            if _build_inputs(stage, submission, settings, directory, offload, weights_source, target) != key["inputs"]:
                raise FullBuildError("full-model build inputs changed during cache lookup")
            if _lowering_identity() != key["lowering"]:
                raise FullBuildError("selected lowering controls or direct sources changed during cache lookup")
            if elf.is_symlink() or sha256(elf) != record["elf_sha256"]:
                raise FullBuildError("linked ELF changed during cache lookup")
            if settings.op_profile and not _profile_table_matches(record):
                raise FullBuildError("operation profile table changed during cache lookup")
            if current_admission is not None:
                current_admission.revalidate(elf=elf, result=admission_result, target=target)
            return {
                **record,
                "reused": True,
                "cache_admission": admission_result,
                "cache_lookup_seconds": time.monotonic() - lookup_started,
            }
    image = directory / "image"
    if image.exists():
        shutil.rmtree(image)
    if admission is None and offload:
        from merlin.model_check.admission import service

        admission = service(target)
    device = _routing(target, Path(stage), Path(submission), label, settings, admission) if offload else None
    receipt = compile_saved_model(
        capture=stage,
        package=settings.host_package,
        board_catalog=settings.board_catalog,
        board=settings.board,
        dts=settings.host_dts,
        output=image,
        target=target,
        run="none",
        arena_mb=settings.arena_mb,
        device=device,
        respect_captured_quantization_scope=True,
        readback="full",
        group_profile=device is not None,
        **({"op_profile": True} if settings.op_profile else {}),
        host_llvm_bitcode=True,
        blob_weights=bool(device is not None and settings.dedup_weights),
        **({"weights_source": weights_source} if weights_source is not None else {}),
    )
    elf = Path(receipt["output"]["elf"])
    build_dir = elf.parent
    record = {
        "schema": BUILD_SCHEMA,
        "key": key,
        "status": "linked",
        "elf": str(elf),
        "elf_sha256": receipt["output"]["elf_sha256"],
        "elf_bytes": elf.stat().st_size,
        "sections": elf_sections(elf),
        "build_hash": receipt["output"]["build_hash"],
        "build_dir": str(build_dir),
        "cache_directory": str(directory.resolve()),
        "weights_bin_sha256": sha256(build_dir / "cgen" / "weights.bin"),
        "weights_bin_bytes": (build_dir / "cgen" / "weights.bin").stat().st_size,
        "device_routed": device is not None,
        "dedup_weights": bool(device is not None and settings.dedup_weights),
        "reused": False,
    }
    recipe = receipt["output"].get("compilation_recipe")
    record["relink_binding"] = None
    if recipe is not None:
        if Path(recipe["path"]) != build_dir / "compilation_recipe.json":
            raise FullBuildError("build returned a foreign relink recipe")
        record["relink_binding"], _ = _relink_artifacts(build_dir, elf, recipe["sha256"])
    host_admission = None
    if not offload and admission is not None:
        if admission.verify(target) != admission_selection:
            raise FullBuildError("linked ELF admission selection changed during compilation")
        _, host_admission = _cached_admission(record, admission, target, directory)
        record["host_admission"] = host_admission
    if _build_inputs(stage, submission, settings, directory, offload, weights_source, target) != key["inputs"]:
        raise FullBuildError("full-model build inputs changed during compilation")
    if _lowering_identity() != key["lowering"]:
        raise FullBuildError("selected lowering controls or direct sources changed during compilation")
    if settings.op_profile:
        record["op_profile_table"] = receipt["output"].get("op_profile_table")
        if not _profile_table_matches(record):
            raise FullBuildError("profiled build returned no current operation table")
    if elf.is_symlink() or sha256(elf) != record["elf_sha256"]:
        raise FullBuildError("linked ELF changed during compilation")
    if host_admission is not None:
        admission.revalidate(elf=elf, result=host_admission, target=target)
    record["seconds"] = round(time.monotonic() - lookup_started, 1)
    pending = record_path.with_suffix(".pending")
    pending.write_text(json.dumps(record, indent=1) + "\n")
    pending.replace(record_path)
    return record


# --------------------------------------------------------------------------------------------- relink


def _recipe_command(recipe: dict, output: Path) -> dict:
    rows = [row for row in recipe["commands"] if Path(row["output"]["path"]) == output]
    if len(rows) != 1:
        raise FullBuildError(f"the compilation recipe records {len(rows)} commands producing {output}")
    return rows[0]


def _run(argv: list[str], cwd: str):
    with compute_step():
        done = proc.run_checked(argv, cwd=cwd, error=FullBuildError)
    return done


def _shadow(stage: Path, inputs: dict[str, np.ndarray], directory: Path) -> Path:
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
    from merlin.llvmlower import c_runtime
    from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe

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
        from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

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


def discard(directory: Path) -> None:
    """Remove a relinked step's image and generated sources, keeping its bound inputs (``capture/inputs.npz``)."""
    directory = Path(directory)
    for name in ("model.elf", "model_main.o"):
        (directory / name).unlink(missing_ok=True)
    shutil.rmtree(directory / "cgen", ignore_errors=True)


# ------------------------------------------------------------------------------------------ execution


def run(target: str, elf: Path, *, timeout_s: float) -> tuple[bytes, float]:
    """``(console bytes, wall seconds)`` of ``elf`` on the target's functional simulator, on the machine
    the image declares (never the simulator's defaults), within ``timeout_s`` and the backend's own
    ceiling for one run."""
    from merlin.runtime.backends.base import get_backend

    backend = get_backend(target)
    runner = getattr(backend, "run_declared_model_elf", None)
    if not callable(runner):
        raise FullBuildError(f"the {target} backend cannot run a linked model on its declared machine")
    ceiling = getattr(backend, "execution_timeout_limit", None)
    if callable(ceiling):
        timeout_s = min(float(timeout_s), float(ceiling("spike")))
    with compute_step(timeout_s) as remaining:
        started = time.monotonic()
        console = runner(Path(elf).resolve(), timeout=remaining, capture_bytes=True)
    return console, time.monotonic() - started


def run_host(elf: Path, *, timeout_s: float) -> tuple[bytes, float]:
    """``(console bytes, wall seconds)`` of a host-only image (a host twin) on the plain functional
    simulator with the ISA and memory span the image declares.

    A twin issues no accelerator instruction, so it needs no extension -- and it computes every group in
    scalar code, which can take longer than the selected engine's ceiling for one submission run
    (SmolVLA's prefix: ~1 G MAC/min). Its bound is the check's own time budget."""
    from merlin.runtime.backends.spike_model import declared_isa, run_raw

    isa = declared_isa(Path(elf))
    if isa is None:
        raise FullBuildError(f"{elf} declares no ISA to run it with")
    with compute_step(timeout_s) as remaining:
        started = time.monotonic()
        console = run_raw(Path(elf).resolve(), isa=isa, timeout=remaining)
    return console, time.monotonic() - started


def read(console: bytes, stage: Path, build_hash: str) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Every result of the forward (i1 as bool), its text metrics and its group profile."""
    from merlin.compile.baremetal_model import read_full_outputs
    from merlin.llvmlower.c_runtime import _out_specs
    from merlin.runtime.out_bin import binary_console_diagnostics
    from merlin.runtime.whole_model_readback import parse_group_profile

    values, metrics = read_full_outputs(console, Path(stage), build_hash)
    specs = _out_specs(Path(stage) / "model.mlir")
    values = [
        np.asarray(v).astype(bool) if dt == "i1" else np.asarray(v) for v, (_s, dt) in zip(values, specs, strict=True)
    ]
    profile = parse_group_profile(binary_console_diagnostics(console).decode("utf-8", errors="replace"))
    out = {"metrics": metrics}
    if profile is not None:
        out["group_profile"] = {
            "calls": profile["calls"],
            "dropped": profile["dropped"],
            "group_cycles": sum(row["cycles"] for row in profile["groups"]),
            "gap_cycles": sum(row["gap_before"] for row in profile["groups"]) + profile["tail_gap"],
        }
    return values, out
