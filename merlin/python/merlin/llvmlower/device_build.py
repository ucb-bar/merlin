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

__all__ = ["DeviceBuild", "DeviceRouting", "build_device_objects", "kernel_symbol",
           "routing_for_placement"]


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


def routing_for_placement(placement, device: str, package_dir: str | Path) -> "DeviceRouting":
    """The ``DeviceRouting`` a whole-model build needs, derived from a placement rather than declared.

    This is the step that made the fused single-ELF path unreachable in production. Every piece of it
    exists -- the rewrite, the shim, the per-extent kernel build, the link -- and all of it is inert
    until a caller supplies ``select``, which nothing did. So a "compiled" whole model ran its
    contractions through the Python interpreter while the artifact that would have run them on the
    device was never asked for.

    The operand and accumulate formats are read off the placement, NOT defaulted: they are what the
    router matched the contraction against, and a build that assumed them would emit kernels in a
    precision the placement never chose. Two consequences, both deliberate:

    * a placement that put nothing on this device raises, because a routing with no work is a caller
      error rather than an empty build;
    * device placements that disagree about the datapath raise too. One ELF carries one device
      datapath; picking the first and dropping the rest is how half a model gets computed in a
      precision nobody selected.

    The accumulate format has a SECOND derivation because a unit legitimately has no accumulate rule to
    match: a contract that declares its formats without an accumulate matrix routes fine and reports
    ``acc=None`` (measured: the reference systolic mesh does exactly this). That is a gap in the
    contract, not in the hardware, so the format is then read off the device's own RTL datapath facts
    for this operand pair -- the same source :mod:`merlin.system.offload` uses -- and it is still an
    error when those facts name none, or name more than one.
    """
    from merlin.system.place import device_selector

    on_dev = [p for p in placement.placed if p.on_device and p.device == device]
    if not on_dev:
        raise ValueError(f"this placement puts no work on {device!r}; there is nothing to build")
    operands = {getattr(p.demand, "in_fmt", None) for p in on_dev}
    weights = {getattr(p.demand, "weight_fmt", None) or getattr(p.demand, "in_fmt", None)
               for p in on_dev}
    accums = {p.acc for p in on_dev}
    if len(operands) != 1 or len(weights) != 1 or len(accums) != 1:
        raise ValueError(
            f"{device!r} placements disagree about the datapath (operands={sorted(map(str, operands))}, "
            f"accumulate={sorted(map(str, accums))}); one image carries one device datapath, so the "
            f"placement has to be split before it can be built")
    operand, weight, accum = operands.pop(), weights.pop(), accums.pop()
    if not operand:
        raise ValueError(f"{device!r} placements carry no operand format; the kernel precision is "
                         f"underivable and assuming one emits the wrong datapath")
    accum = accum or _accum_from_facts(device, operand, weight)
    return DeviceRouting(device=device, package_dir=package_dir, operand_dtype=str(operand),
                         accum_dtype=str(accum), select=device_selector(placement))


def _accum_from_facts(device: str, operand: str, weight: str) -> str:
    """The accumulate format this device's RTL datapath declares for ``operand`` x ``weight``.

    Fails closed in both directions. No matching triple means the device does not declare what it
    accumulates this pair into, and more than one means it declares several -- and a build that picked
    among them would be choosing the model's arithmetic on the strength of dictionary order.
    """
    from merlin.system.offload import device_dtype_triples
    from merlin.targetgen.routing import _fmt_ok  # noqa: PLC2701 -- one format-equality predicate

    found = {a for i, w, a in device_dtype_triples(device)
             if _fmt_ok(operand, (i,)) and _fmt_ok(weight, (w,))}
    if len(found) != 1:
        raise ValueError(
            f"{device!r} declares {len(found)} accumulate format(s) for {operand} x {weight} "
            f"({sorted(found) or 'none'}); the unit matched no accumulate rule either, so the kernel "
            f"precision is underivable and assuming one emits the wrong datapath")
    return found.pop()


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


#: Transports whose package artifact this pipeline can turn into an object. `host_instruction` is a
#: device the host drives with its own instructions (the artifact is LLVM-dialect MLIR); None is a
#: unit that lowers through a stock LLVM target and has no separate device artifact at all.
BUILDABLE_TRANSPORTS: frozenset = frozenset({"host_instruction", None})


def boundary_buildable(device: str) -> str | None:
    """Why no path in this repo can emit ``device``'s host/device boundary, or None when one can.

    PUBLIC because the composition axis needs it. ``targetgen.boundary`` classifies a capsule's
    accelerator/host seam from ELIGIBILITY alone, which silently claims a crossing on a target whose
    seam nothing can compile -- so it has to be able to ask, and a private cross-package import is how
    that drifts. Same predicate, one name.
    """
    try:
        from merlin.system.derive import link_for
        from merlin.targetgen.target_experiment import load_capability_manifest
        endpoint = getattr(load_capability_manifest(device), "endpoint_kind", None)
        link = link_for(device, endpoint)
    except Exception:            # noqa: BLE001 -- an unresolvable device is caught by the package load
        return None
    if link.command_transport not in BUILDABLE_TRANSPORTS:
        return (f"{device!r} is reached by {link.command_transport!r}; this path compiles a device "
                f"whose artifact is LLVM-dialect MLIR, and that transport's package emits "
                f"{link.emitted_artifact or 'another artifact'} instead")
    return None


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
                         cflags: "Sequence[str] | None" = None,
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

    # WHICH DEVICES THIS PATH CAN BUILD, asked of the device's derived link rather than assumed.
    #
    # The pipeline below runs the package's artifact through mlir-translate and clang, so it works
    # exactly for a device whose artifact IS LLVM-dialect MLIR. A device driven by a command buffer
    # emits a JSON command buffer, and a self-hosted one emits its own source; handing either to
    # mlir-translate fails obscurely, and worse, the shim would declare an extern kernel symbol that
    # nothing in the archive defines. Declining with the transport named is the honest answer, and it
    # is the first consumer of the transport axis the Link derives.
    unbuildable = boundary_buildable(device)
    if unbuildable:
        return DeviceBuild(device=device, skipped=(("all", unbuildable),))

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
        c = _run([clang(), *_flags(codegen_target, cflags), "-c", str(ll), "-o", str(raw)], timeout=timeout)
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
    s = _run([clang(), *_flags(codegen_target, cflags), "-c", str(shim_c), "-o", str(shim_o)], timeout=timeout)
    if s.returncode != 0:
        skipped.append(("shim", f"clang: {(s.stderr or '').strip()[:300]}"))
        return DeviceBuild(device=device, objects=tuple(objs), kernels=kernels,
                           skipped=tuple(skipped))

    return DeviceBuild(device=device, objects=(*objs, shim_o), shim_object=shim_o,
                       kernels=kernels, skipped=tuple(skipped))


def _flags(codegen_target: str, cflags: "Sequence[str] | None" = None) -> list[str]:
    """Compile flags for the device objects: the CALLER's when it supplied them.

    The defaults name an ISA (`-march=rv64gcv`), and a default ISA is an assumption about the
    hardware. Baking it in here meant the device shim was compiled with the vector extension for a
    core that does not have one: the whole-model image trapped mid-run on its first `vsetvli`
    (`mcause=2`, mtval opcode 0x57) on a Rocket whose own DTS reads
    `rv64imafdcbzicsr_..._xrocket` -- no `v`. The kernels themselves came out clean because they are
    translated from the target's own lowering; only this C shim was compiled against the default.
    The matrix/OPU shim path already took its flags from the caller; this one now does too."""
    if cflags:
        return list(cflags)
    from .codegen import RISCV_FLAGS, X86_FLAGS

    return list(RISCV_FLAGS if codegen_target == "riscv" else X86_FLAGS)
