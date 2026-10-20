"""Kernel objects and replay results reused across checks, keyed by every input that produced them.

A kernel object is a function of the submission package's bytes, the interface module the package is
asked to lower, and the compile flags (with Merlin's own build code and the tools it runs). Each built
object is stored under ``build_dir()/model_check/cache/kernels/<key>/`` with its kernel symbol renamed
to one derived from the compiled CODE (:func:`code_identity`: the built object's bytes, its symbol and
the interface it implements), so objects built in different checks link side by side. A package
refusal is stored too (a timed-out lowering is not: it says nothing about the package).

A replay image is a function of its generated sources, its operand bytes, the objects it links, the
link toolchain and the simulator it runs on, so its console is stored under ``runs/<key>.json`` and
an image whose key is already there is neither linked nor run again. Any byte of an operand or of a
linked object changes the key; a package that changes while it is being built is not stored.
``--no-cache`` bypasses both stores.

Any package byte still invalidates a stored OBJECT, because which bytes a lowering reads cannot be
known without running it. The rename is what lets the expensive half survive an edit: an edit that
leaves a kernel's compiled code unchanged rebuilds it into a byte-identical object under the same
name, so ``main.c`` and every replay image built from it are unchanged and answered from the store.
Objects link in content order for the same reason. Named by the key, every edit renamed every
kernel: measured over one 5 h authoring run, 1,200 kernel builds produced only 164 distinct code
bodies, yet every rebuilt program re-ran its replay images (one 300 s replay per edit for one model).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CACHE_SCHEMA = "merlin.model_check.cache.v1"

#: Merlin modules whose code shapes a kernel object or a replay image; their bytes are part of every key.
BUILDER_MODULES = (
    "merlin.llvmlower.device_build",
    "merlin.llvmlower.device_build_jobs",
    "merlin.llvmlower.device_object_optimization",
    "merlin.llvmlower.device_ir_scalability",
    "merlin.llvmlower.device_shim",
    "merlin.targetgen.capsule_common",
    "merlin.targetgen.package_runtime",
    "merlin.model_check.replay",
    "merlin.model_check.cache",
)


def cache_root() -> Path:
    from merlin.common.paths import build_dir

    return build_dir() / "model_check" / "cache"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, default=str).encode())


_MEMO: dict[Any, Any] = {}


def content_digest(path: Path) -> str:
    """sha256 of a file's bytes, read every time (a timestamp cannot be trusted to show a rewrite)."""
    return _sha_file(Path(path))


def package_digest(root: Path) -> str:
    """sha256 of every file reachable under ``root`` (symlinks followed): relative path and bytes."""
    root = Path(root)
    digest = hashlib.sha256()
    seen: set[str] = set()
    for directory, dirs, files in os.walk(root, followlinks=True):
        real = os.path.realpath(directory)
        if real in seen:  # a symlink cycle
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            if path.is_file():
                digest.update(b"F" + bytes.fromhex(_sha_file(path)))
            else:  # a dangling link: its target is still part of what the package says
                digest.update(b"L" + os.readlink(path).encode())
    return digest.hexdigest()


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    """A tool's identity: resolved path, size and modification time (a reinstall changes it)."""
    if path is None:
        return None
    resolved = Path(shutil.which(str(path)) or path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {"path": str(resolved), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


#: Environment variables that select the target's facts; the bytes of the files they name are keyed.
TARGET_INPUTS = ("MERLIN_TARGET_CONTRACT", "MERLIN_RTL_FACTS")


def compiler_environment() -> dict[str, str | None]:
    """Command lookup/import context, preserving lexical virtualenv paths.

    A manifest may invoke bare ``python3`` rather than ``sys.executable``. The
    resolved Python binary alone loses its virtualenv context, and a PATH or
    import-path correction must invalidate cached refusals as well as objects.
    """
    return {
        name: os.environ.get(name)
        for name in (
            "PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "PYTHONNOUSERSITE",
            "VIRTUAL_ENV",
        )
    }


def target_inputs() -> dict[str, str | None]:
    """sha256 of the selected target contract and RTL facts (they bind the interface and the ABI)."""
    found: dict[str, str | None] = {}
    for name in TARGET_INPUTS:
        value = os.environ.get(name)
        found[name] = _sha_file(Path(value)) if value and Path(value).is_file() else value
    return found


def builder_digest() -> str:
    """sha256 of the Merlin sources that build kernel objects and replay images."""
    if "builder" not in _MEMO:
        from importlib import util

        digest = hashlib.sha256()
        for name in BUILDER_MODULES:
            spec = util.find_spec(name)
            origin = Path(spec.origin) if spec is not None and spec.origin else None
            digest.update(name.encode() + b"\0")
            digest.update(bytes.fromhex(_sha_file(origin)) if origin is not None and origin.is_file() else b"-")
        _MEMO["builder"] = digest.hexdigest()
    return _MEMO["builder"]


# ---------------------------------------------------------------------------------------- kernels


@dataclass
class KernelSet:
    """The linkable kernels of one program's replay, from the store or freshly built."""

    kernels: dict[str, str] = field(default_factory=dict)  # symbol -> kernel symbol to call
    objects: list[Path] = field(default_factory=list)  # distinct objects to link
    refused: dict[str, str] = field(default_factory=dict)  # symbol -> package/build refusal
    stats: dict[str, Any] = field(default_factory=dict)
    unattempted: dict[str, str] = field(default_factory=dict)


def kernel_material(catalog, symbol: str, *, package_sha256: str, toolchain, tools: Mapping[str, Any]) -> dict:
    """What one kernel object is a function of: package, interface, flags, tools and Merlin's builder."""
    return {
        "schema": CACHE_SCHEMA,
        "package_sha256": package_sha256,
        "interface_sha256": _sha_bytes(catalog.interfaces[symbol].encode()),
        "flags": {
            "target": catalog.target,
            "codegen_target": "riscv",
            "operand_dtype": catalog.routing["operand_dtype"],
            "accum_dtype": catalog.routing["accum_dtype"],
            "numeric_policy": catalog.routing["numeric_policy"],
            "device_cflags": list(toolchain.device_cflags),
        },
        "tools": dict(tools),
        "compiler_environment": compiler_environment(),
        "builder_sha256": builder_digest(),
        "target_inputs": target_inputs(),
    }


def code_identity(raw: Path, built_symbol: str, material: Mapping[str, Any]) -> str:
    """What a stored kernel is named by: its built object's bytes, its symbol and the interface it implements.

    Not the package: two packages that compile one interface to the same bytes yield the same object
    under the same name, so the images that link it are the same images. The interface keeps two
    kernels of one program from ever sharing a name, whatever their code.
    """
    return _sha_json(
        {
            "schema": CACHE_SCHEMA,
            "object_sha256": _sha_file(raw),
            "built_symbol": built_symbol,
            "interface_sha256": material["interface_sha256"],
        }
    )


def _linked_symbol(code: str) -> str:
    return f"mc_kernel_{code[:24]}"


def _read_entry(directory: Path) -> dict[str, Any] | None:
    try:
        entry = json.loads((directory / "entry.json").read_text())
    except (OSError, ValueError):
        return None
    if entry.get("schema") != CACHE_SCHEMA:
        return None
    if entry.get("status") == "built":
        obj = directory / "kernel.o"
        if not obj.is_file() or content_digest(obj) != entry.get("object_sha256"):
            return None
    return entry


def _publish(directory: Path, write: Callable[[Path], None]) -> None:
    """Write an entry into a private directory, then rename it into place (the first writer wins)."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.{os.getpid()}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        write(staging)
        os.rename(staging, directory)
    except (OSError, subprocess.SubprocessError):
        shutil.rmtree(staging, ignore_errors=True)


def _store_object(directory: Path, raw: Path, built_symbol: str, objcopy: str, material) -> None:
    code = code_identity(raw, built_symbol, material)
    linked = _linked_symbol(code)

    def write(staging: Path) -> None:
        # LLVM and GNU object tools apply rename and visibility options in
        # different orders. Materialize the rename before isolating exports.
        renamed = staging / "renamed.o"
        for argv in (
            [objcopy, f"--redefine-sym={built_symbol}={linked}", str(raw), str(renamed)],
            [objcopy, f"--keep-global-symbol={linked}", str(renamed), str(staging / "kernel.o")],
        ):
            done = subprocess.run(argv, capture_output=True, text=True, timeout=300)
            if done.returncode != 0:
                raise OSError(done.stderr)
        renamed.unlink()
        record = {
            "schema": CACHE_SCHEMA,
            "status": "built",
            "kernel_symbol": linked,
            "code_sha256": code,
            "object_sha256": _sha_file(staging / "kernel.o"),
            "key": material,
        }
        (staging / "entry.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    _publish(directory, write)


def _store_refusal(directory: Path, why: str, material) -> None:
    def write(staging: Path) -> None:
        record = {"schema": CACHE_SCHEMA, "status": "refused", "refusal": why, "key": material}
        (staging / "entry.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")

    _publish(directory, write)


def _build(build, catalog, symbols, package_dir, workdir, toolchain, stop_on_first_failure=False):
    return build(
        catalog.target,
        catalog.sidecar,
        symbols,
        package_dir=package_dir,
        workdir=workdir,
        operand_dtype=catalog.routing["operand_dtype"],
        accum_dtype=catalog.routing["accum_dtype"],
        numeric_policy=catalog.routing["numeric_policy"],
        toolchain=toolchain,
        **({"stop_on_first_failure": True} if stop_on_first_failure else {}),
    )


def build_kernels(
    catalog,
    symbols: Sequence[str],
    *,
    package_dir: Path,
    workdir: Path,
    toolchain,
    build: Callable[..., Any],
    enabled: bool = True,
    root: Path | None = None,
    stop_on_first_failure: bool = False,
) -> KernelSet:
    """The package's kernels for ``symbols``: from the store where their key is, else ``build``.

    ``build`` has the signature of :func:`merlin.model_check.replay.build_kernels`. A freshly built
    object is stored only when the interface it was lowered from is the catalog's (the key's) and the
    package did not change during the build; otherwise it is used as built and not stored. Without the
    cache the build's own objects and kernel symbols are returned unchanged.
    """
    wanted = sorted(set(symbols))
    if not enabled:
        built = _build(build, catalog, wanted, package_dir, workdir, toolchain, stop_on_first_failure)
        objects = [Path(o) for o in built.objects if Path(o).name != "device_shim.o"]
        refused = {str(sym): str(why) for sym, why in built.skipped}
        unattempted = {}
        if stop_on_first_failure and refused:
            if "all" in refused:
                refused = {s: refused["all"] for s in wanted}
            unattempted = {
                s: "not attempted after a required kernel build failure"
                for s in wanted
                if s not in built.kernels and s not in refused
            }
        return KernelSet(
            dict(built.kernels), objects, refused, {"enabled": False, "hits": 0, "misses": len(wanted)}, unattempted
        )
    from merlin.llvmlower.device_build import _objcopy
    from merlin.llvmlower.toolchain import clang, mlir_translate

    store = Path(root or cache_root()) / "kernels"
    package_sha256 = package_digest(Path(package_dir))
    objcopy = _objcopy()
    tools = {
        "clang": file_identity(clang()),
        "mlir_translate": file_identity(mlir_translate()),
        "objcopy": file_identity(objcopy),
        "python": file_identity(sys.executable),
    }
    result = KernelSet(stats={"enabled": True, "hits": 0, "misses": 0, "stored": 0})
    materials = {
        s: kernel_material(catalog, s, package_sha256=package_sha256, toolchain=toolchain, tools=tools) for s in wanted
    }
    keys = {s: _sha_json(m) for s, m in materials.items()}
    missing: list[str] = []
    for symbol in wanted:
        entry = _read_entry(store / keys[symbol])
        if entry is None:
            missing.append(symbol)
        elif entry["status"] == "refused":
            result.refused[symbol] = entry["refusal"]
        else:
            result.kernels[symbol] = entry["kernel_symbol"]
            result.objects.append(store / keys[symbol] / "kernel.o")
    result.stats.update(hits=len(wanted) - len(missing), misses=len(missing))
    if missing and stop_on_first_failure and result.refused:
        result.unattempted = {s: "not attempted after a cached required kernel refusal" for s in missing}
    elif missing:
        built = _build(build, catalog, missing, package_dir, workdir, toolchain, stop_on_first_failure)
        environment_after = compiler_environment()
        storable = (
            objcopy is not None
            and package_digest(Path(package_dir)) == package_sha256
            and all(material["compiler_environment"] == environment_after for material in materials.values())
        )
        skipped = {str(sym): str(why) for sym, why in built.skipped}
        for symbol in missing:
            directory = store / keys[symbol]
            if symbol not in built.kernels:
                if stop_on_first_failure and skipped and symbol not in skipped and "all" not in skipped:
                    result.unattempted[symbol] = "not attempted after a required kernel build failure"
                    continue
                why = skipped.get(symbol) or skipped.get("all") or "the device build produced no kernel for this symbol"
                result.refused[symbol] = why
                if storable and symbol in skipped and "timed out" not in why:
                    _store_refusal(directory, why, materials[symbol])
                    result.stats["stored"] += 1
                continue
            raw = Path(workdir) / f"{symbol}.o"
            lowered = Path(workdir) / f"{symbol}.iface.mlir"
            same = lowered.is_file() and lowered.read_text(encoding="utf-8") == catalog.interfaces[symbol]
            if storable and same and raw.is_file():
                _store_object(directory, raw, built.kernels[symbol], objcopy, materials[symbol])
            entry = _read_entry(directory) if storable and same else None
            if entry is not None and entry.get("status") == "built":
                result.stats["stored"] += 1
                result.kernels[symbol] = entry["kernel_symbol"]
                result.objects.append(directory / "kernel.o")
            else:
                result.kernels[symbol] = built.kernels[symbol]
                result.objects.append(raw)
    # One object per distinct entry, in CONTENT order: the same code links in the same order on every
    # check, whichever package (and so whichever store directory) built it. Path order followed the
    # package-keyed directory names, so an unchanged kernel set could still link a different image.
    result.objects = sorted(set(result.objects), key=lambda path: (content_digest(path), str(path)))
    return result


# ----------------------------------------------------------------------------------------- images


def engine_identity(target: str, *, dram_base: int, dram_bytes: int) -> dict[str, Any]:
    """The simulator a replay image runs on, without the image: binary, extension flags and libraries."""
    memo = ("engine", target, dram_base, dram_bytes)
    if memo not in _MEMO:
        from merlin.runtime.backends import base as backends

        backend = backends.get_backend(target)
        engine = backend._engine("spike")  # noqa: SLF001
        identity = backend._engine_identity("spike")  # noqa: SLF001
        flags = [f for f in identity["extension"]["flags"] if not f.startswith(("--isa=", "-m"))]
        referenced = {}
        for flag in flags:
            _name, sep, value = flag.partition("=")
            if sep and Path(value).is_file():
                referenced[flag] = content_digest(Path(value))
        binary = Path(engine["binary"]["path"])
        _MEMO[memo] = {
            "binary": str(binary),
            "binary_sha256": content_digest(binary) if binary.is_file() else None,
            "flags": flags,
            "referenced_files": referenced,
            "environment": dict(sorted((engine.get("environment") or {}).items())),
            "dram": [dram_base, dram_bytes],
        }
    return _MEMO[memo]


def image_key(main_c: Path, data_s: Path, objects: Sequence[Path], toolchain, engine: Mapping[str, Any]) -> str:
    """Everything a replay image's console depends on, as one digest (operands by content, not path)."""
    data_lines = []
    for line in Path(data_s).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(".incbin"):
            data_lines.append(f".incbin sha256:{content_digest(Path(stripped.split(chr(34))[1]))}")
        else:
            data_lines.append(line)
    harness = Path(toolchain.harness_dir)
    material = {
        "schema": CACHE_SCHEMA,
        "main_c": _sha_bytes(Path(main_c).read_bytes()),
        "data_s": _sha_bytes("\n".join(data_lines).encode()),
        "objects": [content_digest(Path(o)) for o in objects],
        "gcc": file_identity(toolchain.gcc),
        "gcc_cflags": list(toolchain.gcc_cflags),
        "harness": {
            p.name: content_digest(p)
            for p in sorted(harness.iterdir())
            if p.is_file() and p.suffix in {".S", ".c", ".h", ".ld"}
        },
        "engine": dict(engine),
        "builder_sha256": builder_digest(),
        "target_inputs": target_inputs(),
    }
    return _sha_json(material)


def _run_path(key: str, root: Path | None) -> Path:
    return Path(root or cache_root()) / "runs" / key[:2] / f"{key}.json"


def cached_run(key: str, *, root: Path | None = None) -> dict[str, Any] | None:
    try:
        doc = json.loads(_run_path(key, root).read_text())
    except (OSError, ValueError):
        return None
    return doc if doc.get("schema") == CACHE_SCHEMA and isinstance(doc.get("console"), str) else None


def store_run(key: str, console: str, seconds: float, *, root: Path | None = None) -> None:
    path = _run_path(key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps({"schema": CACHE_SCHEMA, "console": console, "seconds": seconds}) + "\n")
    os.replace(staging, path)


def replay_image(
    work: Path,
    main_c: Path,
    data_s: Path,
    objects: Sequence[Path],
    toolchain,
    *,
    target: str,
    dram_base: int,
    dram_bytes: int,
    timeout: float,
    enabled: bool = True,
    root: Path | None = None,
) -> tuple[str, float, dict[str, Any]]:
    """Link and run one replay image, or answer it from the store: ``(console, seconds, record)``.

    ``seconds`` is the simulator wall time of the run that produced the console; ``record`` says
    whether it was stored (``cached``), under which key, and the ELF when one was linked.
    """
    from merlin.model_check import replay as RP

    key = None
    if enabled:
        engine = engine_identity(target, dram_base=dram_base, dram_bytes=dram_bytes)
        key = image_key(main_c, data_s, objects, toolchain, engine)
        hit = cached_run(key, root=root)
        if hit is not None:
            return hit["console"], float(hit.get("seconds") or 0.0), {"cached": True, "key": key}
    elf = RP.link_image(work, main_c, data_s, objects, toolchain)
    console, wall = RP.run_image(target, elf, dram_base=dram_base, dram_bytes=dram_bytes, timeout=timeout)
    if key is not None:
        store_run(key, console, round(wall, 3), root=root)
    return console, wall, {"cached": False, "key": key, "elf": str(elf)}
