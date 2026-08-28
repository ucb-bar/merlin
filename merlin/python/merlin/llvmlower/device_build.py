"""Build the device side of an offloaded model into objects the host archive can link.

The pieces either side of this already exist. The rewrite turns a contraction into a call to a
private symbol; the shim adapts the MLIR calling convention to the device's kernel ABI; the target's
own package emits a device kernel from an interface capsule. What was missing is the step that runs
the package once per distinct extent and turns each result into an object with a distinct symbol.

**Why once per extent.** A package emits a kernel for the capsule it is given, with the extents baked
in -- it is not a general GEMM. A model with several distinct contraction shapes therefore needs
several kernels, which is the same reason the rewrite mints one symbol per signature.

**Why the rename matters.** Every one of those objects defines the entry under the single name the
backend contract declares. Linking them together without renaming is not a link error: the linker
binds every call to whichever object it resolved first, so a model quietly runs one layer's kernel for
every layer. Each object is renamed to the symbol the shim declares for that signature.

Nothing here knows which target it is building for. The package, the kernel name and the extents are
all arguments.
"""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

__all__ = ["DeviceBuild", "DeviceRouting", "build_device_objects", "kernel_symbol"]


@dataclass(frozen=True)
class DeviceRouting:
    """Everything a whole-model build needs to offload onto one device.

    The mirror of the matrix-unit path's routing descriptor, and required for the same reason: a build
    that enabled offload without saying WHICH device and WHICH backend package would have to guess
    both, and a guessed package emits kernels for the wrong hardware that link and run.

    ``select`` is the placement decision, passed in rather than made here -- see
    :mod:`merlin.system.place`. ``None`` moves nothing, so the whole path is inert unless a decision
    has been made.
    """

    device: str
    package_dir: str | Path
    operand_dtype: str
    accum_dtype: str
    select: "Callable[[Any], bool] | None" = None


@dataclass(frozen=True)
class DeviceBuild:
    """What was built, and what could not be."""

    device: str
    #: Objects to add to the archive: one kernel per signature, plus the shim.
    objects: tuple[Path, ...] = ()
    shim_object: Path | None = None
    #: shim entry symbol -> the kernel symbol it calls.
    kernels: dict[str, str] = field(default_factory=dict)
    skipped: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.objects) and self.shim_object is not None

    def archive(self, path: str | Path) -> Path | None:
        """Bundle the objects into a static archive, or None when there is nothing to bundle.

        An archive rather than a link: the final link belongs to the board build, which owns the
        target's toolchain and links this beside the model object exactly as the existing matrix-unit
        shim is linked. Bundling here keeps the device side one artifact to hand over.
        """
        if not self.objects:
            return None
        ar = _ar()
        if ar is None:
            return None
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()                     # ar appends; a stale member would shadow a rebuilt one
        r = _run([ar, "rcs", str(out), *[str(o) for o in self.objects]], timeout=300)
        return out if r.returncode == 0 and out.exists() else None


def kernel_symbol(base: str, index: int) -> str:
    """The per-signature kernel symbol. Distinct by construction; see the module docstring."""
    return f"{base}_{int(index)}"


def _ar() -> str | None:
    from .toolchain import DEFAULT_LLVM_INSTALL

    local = Path(DEFAULT_LLVM_INSTALL) / "bin" / "llvm-ar"
    if local.exists():
        return str(local)
    return shutil.which("llvm-ar") or shutil.which("ar")


def _objcopy() -> str | None:
    from .toolchain import DEFAULT_LLVM_INSTALL

    local = Path(DEFAULT_LLVM_INSTALL) / "bin" / "llvm-objcopy"
    if local.exists():
        return str(local)
    return shutil.which("llvm-objcopy") or shutil.which("objcopy")


def _run(argv: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout)


def build_device_objects(device: str,
                         signatures: Mapping[str, Sequence[int]],
                         dtypes: Mapping[str, Sequence[str]],
                         *,
                         package_dir: str | Path,
                         workdir: str | Path,
                         operand_dtype: str,
                         accum_dtype: str,
                         codegen_target: str = "riscv",
                         timeout: int = 900) -> DeviceBuild:
    """One kernel object per signature plus the shim object, ready to archive.

    ``signatures`` / ``dtypes`` come from the offload rewrite. ``operand_dtype`` / ``accum_dtype`` are
    the device's own datapath tokens, which the caller derived from the device rather than assumed.

    Every failure is recorded and skipped rather than raised: a model whose third extent the package
    declines should still build its other two and say what it lost, because the alternative is an
    all-or-nothing build whose failure names none of the shapes involved.
    """
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.oot_runner import load_package, run_entrypoint

    from .device_shim import emit_translation_unit, kernel_abi_for
    from .toolchain import clang, mlir_translate

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    skipped: list[tuple[str, str]] = []

    abi = kernel_abi_for(device)
    if abi is None:
        return DeviceBuild(device=device, skipped=(("all", "no readable kernel_abi"),))
    try:
        pkg = load_package(str(package_dir))
    except Exception as exc:                     # noqa: BLE001
        return DeviceBuild(device=device, skipped=(("all", f"package unusable: {exc}"),))

    from merlin.compile_cli import _mesh_tile_binding
    binding = _mesh_tile_binding(device, operand_dtype, accum_dtype)

    objs: list[Path] = []
    kernels: dict[str, str] = {}
    oc = _objcopy()

    for index, sym in enumerate(sorted(signatures)):
        key = tuple(int(v) for v in signatures[sym])
        if len(key) not in (3, 4):
            skipped.append((sym, f"signature {key} has neither 3 nor 4 extents; no kernel shape for it"))
            continue
        # A batched signature needs the SAME kernel as its unbatched form: the batch is a loop in the
        # shim over disjoint slices, not a third axis the device sees. Building a separate kernel per
        # batch size would mint one per B for identical work.
        m, n, k = key[-3:]
        want = kernel_symbol(abi.symbol, index)
        stem = work / f"{sym}"

        entry = {"name": sym, "op": "matmul", "kind": "op",
                 "source_role": "mesh_tile_synthesized",
                 "source_reference": f"offloaded layer {m}x{k}x{n} for {device}",
                 "M": m, "K": k, "N": n}
        try:
            _capsule, iface = CS.build(entry, binding)
        except Exception as exc:                 # noqa: BLE001
            skipped.append((sym, f"interface capsule: {exc}"))
            continue
        ifc = stem.with_suffix(".iface.mlir")
        ifc.write_text(iface, encoding="utf-8")

        r = run_entrypoint(pkg, "emit_target_artifact", ifc, timeout=timeout)
        if r.returncode != 0:
            skipped.append((sym, f"package declined {m}x{k}x{n}: {(r.stderr or '').strip()[:200]}"))
            continue
        art = stem.with_suffix(".device.mlir")
        art.write_text(r.stdout, encoding="utf-8")

        ll = stem.with_suffix(".ll")
        t = _run([mlir_translate(), "--mlir-to-llvmir", str(art), "-o", str(ll)], timeout=timeout)
        if t.returncode != 0:
            skipped.append((sym, f"mlir-translate: {(t.stderr or '').strip()[:200]}"))
            continue

        raw = stem.with_suffix(".raw.o")
        c = _run([clang(), *_flags(codegen_target), "-c", str(ll), "-o", str(raw)], timeout=timeout)
        if c.returncode != 0:
            skipped.append((sym, f"clang: {(c.stderr or '').strip()[:200]}"))
            continue

        obj = stem.with_suffix(".o")
        if oc is None:
            skipped.append((sym, "no objcopy available to give this kernel a distinct symbol"))
            continue
        rn = _run([oc, f"--redefine-sym={abi.symbol}={want}", str(raw), str(obj)], timeout=timeout)
        if rn.returncode != 0:
            skipped.append((sym, f"symbol rename: {(rn.stderr or '').strip()[:200]}"))
            continue

        objs.append(obj)
        kernels[sym] = want

    if not kernels:
        return DeviceBuild(device=device, skipped=tuple(skipped))

    unit = emit_translation_unit(device, {s: signatures[s] for s in kernels},
                                 {s: dtypes.get(s, ()) for s in kernels},
                                 kernel_symbol_for=kernels.get)
    if not unit.symbols:
        return DeviceBuild(device=device, objects=tuple(objs), kernels=kernels,
                           skipped=tuple([*skipped, *unit.skipped]))
    shim_c = work / "device_shim.c"
    shim_c.write_text(unit.text, encoding="utf-8")
    shim_o = work / "device_shim.o"
    s = _run([clang(), *_flags(codegen_target), "-c", str(shim_c), "-o", str(shim_o)], timeout=timeout)
    if s.returncode != 0:
        skipped.append(("shim", f"clang: {(s.stderr or '').strip()[:300]}"))
        return DeviceBuild(device=device, objects=tuple(objs), kernels=kernels,
                           skipped=tuple(skipped))

    return DeviceBuild(device=device, objects=(*objs, shim_o), shim_object=shim_o,
                       kernels=kernels, skipped=tuple(skipped))


def _flags(codegen_target: str) -> list[str]:
    from .codegen import RISCV_FLAGS, X86_FLAGS

    return list(RISCV_FLAGS if codegen_target == "riscv" else X86_FLAGS)
