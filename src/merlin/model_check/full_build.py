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
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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

    @classmethod
    def of(cls, models, model, host_package: Path, *, dedup_weights: bool = True) -> BuildSettings:
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
        )

    def record(self) -> dict[str, Any]:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------------------------- build


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
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _build(
            stage, target, label, submission, submission_sha256, settings, directory, admission, offload, weights_source
        )


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

    key = {
        "stage": str(Path(stage).resolve()),
        # The host twin depends on no submission byte.
        "submission_sha256": submission_sha256 if offload else None,
        "settings": settings.record(),
        "offload": offload,
    }
    if weights_source is not None:
        key["weights_source"] = {
            "path": str(Path(weights_source).resolve(strict=True)),
            "sha256": sha256(weights_source),
        }
    record_path = directory / "build.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        elf = Path(record.get("elf") or "")
        if record.get("key") == key and elf.is_file() and elf.stat().st_size == record.get("elf_bytes"):
            return {**record, "reused": True}
    image = directory / "image"
    if image.exists():
        shutil.rmtree(image)
    if admission is None:
        from merlin.model_check.admission import service

        admission = service(target)
    started = time.monotonic()
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
        host_llvm_bitcode=True,
        blob_weights=bool(device is not None and settings.dedup_weights),
        **({"weights_source": weights_source} if weights_source is not None else {}),
    )
    seconds = time.monotonic() - started
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
        "weights_bin_sha256": sha256(build_dir / "cgen" / "weights.bin"),
        "weights_bin_bytes": (build_dir / "cgen" / "weights.bin").stat().st_size,
        "device_routed": device is not None,
        "dedup_weights": bool(device is not None and settings.dedup_weights),
        "seconds": round(seconds, 1),
        "reused": False,
    }
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


def _run(argv: list[str], cwd: str) -> None:
    with compute_step():
        done = subprocess.run([str(a) for a in argv], cwd=cwd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise FullBuildError(f"{Path(argv[0]).name} failed ({done.returncode}): {done.stderr[-2000:]}")


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


def relink(record: dict[str, Any], stage: Path, inputs: dict[str, np.ndarray], directory: Path) -> Path:
    """The build's image with ``inputs`` (``{"in0": ...}``) embedded instead of the capture's; its ELF path."""
    from merlin.llvmlower import c_runtime

    build_dir = Path(record["build_dir"])
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
    recipe = json.loads((build_dir / "compilation_recipe.json").read_text())
    main = _recipe_command(recipe, build_dir / "model_main.o")
    argv = [directory / "model_main.o" if a == str(build_dir / "model_main.o") else a for a in main["argv"]]
    argv = [str(cgen) if a == str(build_dir / "cgen") else a for a in argv]
    _run(argv, main["cwd"])
    link = _recipe_command(recipe, Path(record["elf"]))
    elf = directory / "model.elf"
    swap = {str(build_dir / "model_main.o"): str(directory / "model_main.o"), record["elf"]: str(elf)}
    _run([swap.get(a, a) for a in link["argv"]], link["cwd"])
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
