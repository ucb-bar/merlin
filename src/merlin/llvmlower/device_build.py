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

import hashlib
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

__all__ = [
    "FROM_EXTENTS",
    "FROM_GROUP",
    "DeviceBuild",
    "DeviceRouting",
    "build_device_objects",
    "kernel_entry",
    "kernel_symbol",
    "routing_for_placement",
]

#: The kernel was built from the layer's own stated program -- its epilogue, its requantize
#: multiplier, its convolution geometry, the type it commits.
FROM_GROUP = "stated_group"
#: The kernel was built from the extents ALONE, as a bare contraction. Everything a readout absorbs
#: is absent from it, so the caller has to run those stages somewhere else. Honest for the
#: contraction-granular rewrite (:mod:`.device_offload`), which routes exactly the contraction and
#: leaves the epilogue in the driver -- and a silent loss for anything that meant to route a group.
FROM_EXTENTS = "bare_extents"


@dataclass(frozen=True)
class DeviceRouting:
    """Everything a whole-model build needs to offload onto one device.

    The mirror of the matrix-unit path's routing descriptor, and required for the same reason: a build
    that enabled offload without saying WHICH device and WHICH backend package would have to guess
    both, and a guessed package emits kernels for the wrong hardware that link and run.

    ``select`` is one placement decision, passed in rather than made here -- see
    :mod:`merlin.system.place`. A source-bound catalog can instead select the exact
    operations in its manifest. With neither, the path moves nothing.
    """

    device: str
    package_dir: str | Path
    operand_dtype: str
    accum_dtype: str
    select: Callable[[Any], bool] | None = None
    numeric_policy: dict | None = None
    #: Exact Phase 0 operation/interface identities; mutually exclusive with a shape selector.
    exact_selection: Any | None = None
    #: The capture bundle this model came from, when there is one. It is what carries the weights
    #: manifest, and the manifest is what says which argument of a first layer is the stored tensor --
    #: a group route with no manifest refuses that layer by name rather than guessing a side.
    capture: str | Path | None = None
    #: The model's name, carried so a routed group's statement names the layer it came from.
    model: str = ""
    #: WHAT IS MOVED: one contraction per call, or one closed compute group per call. The two build
    #: different programs -- a contraction leaves the layer's bias, requantize, activation and pooling
    #: on the host -- so the choice travels with the routing rather than being a property of whichever
    #: rewrite the build happened to call. See :mod:`merlin.llvmlower.device_offload`.
    granularity: str = "contraction"

    #: Optional source-identified, already compiled device catalog using a declared pointer ABI.
    catalog_manifest: str | Path | None = None
    catalog_object: str | Path | None = None
    #: Target-owned exact graph preparation, before catalog compilation and source binding.
    prepared_transform: Callable[[Path, Path], Path] | None = None
    #: Build a source-bound catalog from Merlin's final prepared file, before offload rewrite.
    catalog_builder: Callable[[Path, Path], tuple[Path, Path]] | None = None
    #: Target package's binary policy check, run on the final linked image.
    final_elf_audit: Callable[[Path], None] | None = None
    #: Provider-owned host ABI preparation after source-bound offload declarations.
    post_offload_transform: Callable[[Path, Path, Path], Path] | None = None
    #: Explicit independently selected final linked-image policy. Every active
    #: ordinary device build requires it; the legacy callback cannot replace it.
    linked_elf_admission: LinkedElfAdmissionService | None = None


def freeze_device_elf_admission(device):
    """Freeze supplied policy selection; an empty host route needs no policy."""
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    service = getattr(device, "linked_elf_admission", None)
    if service is None:
        return None
    if type(service) is not LinkedElfAdmissionService:
        raise ValueError("device linked ELF admission requires an explicitly selected service")
    return device.device, service, service.verify(device.device)


def require_device_elf_admission(device, selection):
    """An active routed program cannot downgrade or replace its selected policy."""
    if selection is None:
        raise ValueError("active device route requires independent linked ELF admission")
    target, service, identity = selection
    if (
        device is None
        or device.device != target
        or getattr(device, "linked_elf_admission", None) is not service
        or service.verify(target) != identity
    ):
        raise ValueError("device linked ELF admission selection changed during build")
    return service


def admit_linked_device_elf(device, selection, *, elf, linked_sha256, directory):
    """Enforce the selected policy on unchanged final link bytes, not objects.

    The selected evaluator owns ISA facts and the original protected policy.
    This caller retains actual evaluation and refuses absent or changed evidence;
    it supplies no default decoder, instruction/effect or runtime authority.
    """
    import json

    from merlin.common import invocation_record
    from merlin.common.digest import sha256_file

    service = require_device_elf_admission(device, selection)
    if sha256_file(elf) != linked_sha256:
        raise ValueError("device linked ELF changed after its actual link")
    with invocation_record.observe_call(
        directory,
        stage="device_final_linked_elf_policy",
        function=service.evaluate,
        arguments={"target": selection[0], "scope": "selected final linked-artifact policy"},
        inputs=(elf,),
        dependencies=tuple(Path(path) for path, _digest in service.source_pins),
    ) as observation:
        result = service.evaluate(elf=elf, target=selection[0], evidence_root=observation.directory / "policy")
        observation.outputs = (Path(result["report_path"]),)
        observation.returned(stdout=json.dumps(result, sort_keys=True))
    require_device_elf_admission(device, selection)
    if service.revalidate(elf=elf, result=result, target=selection[0]) != "accepted":
        raise ValueError("device linked ELF was refused by its selected policy")
    if sha256_file(elf) != linked_sha256:
        raise ValueError("device linked ELF changed during its selected policy")
    return result


def routing_for_placement(
    placement,
    device: str,
    package_dir: str | Path,
    *,
    numeric_policy=None,
    granularity: str = "contraction",
    capture: str | Path | None = None,
    model: str = "",
    linked_elf_admission: LinkedElfAdmissionService | None = None,
) -> DeviceRouting:
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
    weights = {getattr(p.demand, "weight_fmt", None) or getattr(p.demand, "in_fmt", None) for p in on_dev}
    accums = {p.acc for p in on_dev}
    if len(operands) != 1 or len(weights) != 1 or len(accums) != 1:
        raise ValueError(
            f"{device!r} placements disagree about the datapath (operands={sorted(map(str, operands))}, "
            f"accumulate={sorted(map(str, accums))}); one image carries one device datapath, so the "
            f"placement has to be split before it can be built"
        )
    operand, weight, accum = operands.pop(), weights.pop(), accums.pop()
    if not operand:
        raise ValueError(
            f"{device!r} placements carry no operand format; the kernel precision is "
            f"underivable and assuming one emits the wrong datapath"
        )
    accum = accum or _accum_from_facts(device, operand, weight)
    return DeviceRouting(
        device=device,
        package_dir=package_dir,
        operand_dtype=str(operand),
        accum_dtype=str(accum),
        select=device_selector(placement),
        numeric_policy=numeric_policy,
        granularity=str(granularity),
        capture=capture,
        model=str(model),
        linked_elf_admission=linked_elf_admission,
    )


def _accum_from_facts(device: str, operand: str, weight: str) -> str:
    """The accumulate format this device's RTL datapath declares for ``operand`` x ``weight``.

    Fails closed in both directions. No matching triple means the device does not declare what it
    accumulates this pair into, and more than one means it declares several -- and a build that picked
    among them would be choosing the model's arithmetic on the strength of dictionary order.
    """
    from merlin.system.offload import device_dtype_triples
    from merlin.targetgen.routing import _fmt_ok  # noqa: PLC2701 -- one format-equality predicate

    found = {a for i, w, a in device_dtype_triples(device) if _fmt_ok(operand, (i,)) and _fmt_ok(weight, (w,))}
    if len(found) != 1:
        raise ValueError(
            f"{device!r} declares {len(found)} accumulate format(s) for {operand} x {weight} "
            f"({sorted(found) or 'none'}); the unit matched no accumulate rule either, so the kernel "
            f"precision is underivable and assuming one emits the wrong datapath"
        )
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
    #: symbol -> which entry the kernel was built from: :data:`FROM_GROUP` (the layer's own stated
    #: program, epilogue and all) or :data:`FROM_EXTENTS` (a bare contraction of those extents).
    #: A MIXED BUILD MUST NOT READ AS ONE MECHANISM: the two compute different functions, and an
    #: archive that held some of each while reporting only "N kernels" would say nothing about
    #: which of a model's layers kept their readout.
    built_from: dict[str, str] = field(default_factory=dict)
    #: Coarse build receipt: distinct compiled artifact texts versus exported kernel symbols.
    object_dedup: dict[str, int] = field(default_factory=dict)

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
            out.unlink()  # ar appends; a stale member would shadow a rebuilt one
        r = _run([ar, "rcs", str(out), *[str(o) for o in self.objects]], timeout=300)
        return out if r.returncode == 0 and out.exists() else None


def kernel_symbol(base: str, index: int) -> str:
    """The per-signature kernel symbol. Distinct by construction; see the module docstring."""
    return f"{base}_{int(index)}"


def _ar() -> str | None:
    from .toolchain import llvm_install

    local = llvm_install() / "bin" / "llvm-ar"
    if local.exists():
        return str(local)
    return shutil.which("llvm-ar") or shutil.which("ar")


#: Transports whose package artifact THIS pipeline can turn into an object. `host_instruction` is a
#: device the host drives with its own instructions (the artifact is LLVM-dialect MLIR); None is a
#: unit that lowers through a stock LLVM target and has no separate device artifact at all.
#:
#: NOT the set of transports whose boundary this repo can emit -- see :func:`boundary_buildable`. A
#: `device_native` boundary is not an object at all (it is a DRAM address contract), so it is emitted
#: by :mod:`merlin.llvmlower.device_native` and would be wrong to list here: this constant gates
#: `build_device_objects`, which really can only build the ones named.
BUILDABLE_TRANSPORTS: frozenset = frozenset({"host_instruction", None})

#: Transports with an emitter somewhere in this repo, and where it lives. Keyed on the derived
#: transport, never on a target: two devices sharing a transport share an emitter by construction.
_SEAM_EMITTERS: dict = {
    "device_native": "merlin.llvmlower.device_native",
}


def boundary_buildable(device: str) -> str | None:
    """Why no path in this repo can emit ``device``'s host/device boundary, or None when one can.

    PUBLIC because the composition axis needs it. ``targetgen.boundary`` classifies a capsule's
    accelerator/host seam from ELIGIBILITY alone, which silently claims a crossing on a target whose
    seam nothing can compile -- so it has to be able to ask, and a private cross-package import is how
    that drifts. Same predicate, one name.

    A BOUNDARY IS NOT ALWAYS AN OBJECT, which is what this used to assume. Asking only whether
    ``build_device_objects`` could compile the artifact answered "no" for every ``device_native``
    device -- correctly at the time, because nothing emitted that boundary, and wrongly once something
    did. A transport with its own emitter is delegated to it, and that emitter answers with its own
    fail-closed reasons (an underivable DRAM window, an operand placement that is not an address
    contract). So a target still reports UNKNOWN when its seam genuinely cannot be emitted; it stops
    reporting UNKNOWN for the reason "this is not the transport I know how to build".
    """
    try:
        from merlin.system.derive import link_for
        from merlin.targetgen.target_experiment import load_capability_manifest

        endpoint = getattr(load_capability_manifest(device), "endpoint_kind", None)
        link = link_for(device, endpoint)
    except Exception:  # noqa: BLE001 -- an unresolvable device is caught by the package load
        return None
    # DELEGATED TO `_SEAM_EMITTERS` ONLY BECAUSE A SEAM HAS NOW BEEN EMITTED. This line was held back on
    # purpose while nothing had built one: turning it on flips the composition axis from UNDETERMINABLE
    # to a shape for every `device_native` capsule, and doing that on the strength of a module that
    # merely exists is exactly the move `boundary.profile_linalg` exists to prevent -- "the composition
    # numbers would improve by argument alone". What changed is an artifact, not an argument:
    # `merlin/tests/infra/test_device_native_seam_emits.py` builds a complete seam for a device_native
    # target from its own backend package -- the package's emitted directives ASSEMBLED to the bytes the
    # device fetches, the address contract read off the command buffer the same package emitted for the
    # same capsule and converted to offsets in the device's own derived DRAM window, and the host stager
    # compiled to an object -- then LINKS that object against a harness generated from the contract and
    # runs it, checking each operand lands at its declared offset and each result is collected from
    # its own. The emitter keeps its own fail-closed refusals (an underivable window, an operand
    # placement that is not an address contract), so a target whose seam genuinely cannot be emitted
    # still reports UNKNOWN; it stops reporting UNKNOWN for the reason "this is not the transport I
    # know how to build".
    emitter = _SEAM_EMITTERS.get(link.command_transport)
    if emitter is not None:
        from importlib import import_module

        return import_module(emitter).seam_emittable(device)
    if link.command_transport not in BUILDABLE_TRANSPORTS:
        return (
            f"{device!r} is reached by {link.command_transport!r}; this path compiles a device "
            f"whose artifact is LLVM-dialect MLIR, and that transport's package emits "
            f"{link.emitted_artifact or 'another artifact'} instead"
        )
    return None


def objects_buildable(device: str) -> str | None:
    """Why THIS pipeline cannot turn ``device``'s package artifact into linkable objects, or None.

    Split from :func:`boundary_buildable` the moment a second transport gained an emitter. The two
    questions had one answer while there was one emitter, and collapsing them again is a real hazard
    in the direction that matters: a device whose boundary IS emittable (as an address contract) would
    otherwise be let into the loop below, hand a stream of ``.word`` directives to ``mlir-translate``,
    and fail obscurely -- or worse, produce a shim declaring extern kernel symbols nothing defines.
    """
    try:
        from merlin.system.derive import link_for
        from merlin.targetgen.target_experiment import load_capability_manifest

        endpoint = getattr(load_capability_manifest(device), "endpoint_kind", None)
        link = link_for(device, endpoint)
    except Exception:  # noqa: BLE001 -- an unresolvable device is caught by the package load
        return None
    if link.command_transport not in BUILDABLE_TRANSPORTS:
        return (
            f"{device!r} is reached by {link.command_transport!r}; this path compiles a device "
            f"whose artifact is LLVM-dialect MLIR, and that transport's package emits "
            f"{link.emitted_artifact or 'another artifact'} instead"
            + (
                f". Its boundary IS emittable -- as a DRAM address contract, by "
                f"{_SEAM_EMITTERS[link.command_transport]} -- just not as an object"
                if link.command_transport in _SEAM_EMITTERS
                else ""
            )
        )
    return None


def _objcopy() -> str | None:
    from .toolchain import llvm_install

    local = llvm_install() / "bin" / "llvm-objcopy"
    if local.exists():
        return str(local)
    return shutil.which("llvm-objcopy") or shutil.which("objcopy")


def _nm() -> str | None:
    from .toolchain import llvm_install

    local = llvm_install() / "bin" / "llvm-nm"
    if local.exists():
        return str(local)
    return shutil.which("llvm-nm") or shutil.which("nm")


def verify_object_symbol_binding(
    kernel_object: Path,
    shim_object: Path,
    *,
    entry_symbol: str,
    kernel_symbol: str,
    original_kernel_symbol: str,
    timeout: int,
) -> dict[str, str]:
    """Check the exact staged objects' exported call edge before claiming build evidence.

    This is an object-level check, not a link or execution verdict. In particular,
    every external reference must resolve between these two objects; otherwise a
    later link could silently pick up an unrelated runtime definition.
    """
    nm = _nm()
    if nm is None:
        raise ValueError("staged object symbol binding needs a readable nm tool")

    def symbols(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
        if not path.is_file() or path.is_symlink():
            raise ValueError("staged object symbol binding needs regular object files")
        result = _run([nm, "--format=posix", "--extern-only", str(path)], timeout=timeout)
        if result.returncode != 0:
            raise ValueError(f"could not inspect staged object symbols: {(result.stderr or '')[-200:]}")
        defined, undefined = [], []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2 or len(parts[1]) != 1:
                raise ValueError("nm returned an unrecognized staged object symbol record")
            if parts[1].upper() == "U":
                undefined.append(parts[0])
            else:
                defined.append((parts[0], parts[1]))
        return defined, undefined

    kernel_defined, kernel_undefined = symbols(kernel_object)
    shim_defined, shim_undefined = symbols(shim_object)
    if (
        kernel_defined.count((kernel_symbol, "T")) != 1
        or shim_defined.count((entry_symbol, "T")) != 1
        or any(name == kernel_symbol for name, _kind in shim_defined)
        or any(name == entry_symbol for name, _kind in kernel_defined)
        or shim_undefined.count(kernel_symbol) != 1
        or kernel_undefined
        or sorted(shim_undefined) != [kernel_symbol]
        or original_kernel_symbol
        in [
            *(name for name, _kind in kernel_defined),
            *kernel_undefined,
            *(name for name, _kind in shim_defined),
            *shim_undefined,
        ]
    ):
        raise ValueError("staged kernel and shim object symbols disagree or have unresolved references")
    return {"status": "object_symbol_binding_verified", "entry_symbol": entry_symbol, "kernel_symbol": kernel_symbol}


def _run(argv: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout)


def _run_build_tool(argv: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess:
    """Keep tool timeouts in the per-symbol failure roster, never admit partial outputs."""
    try:
        return _run(argv, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            argv,
            124,
            stdout="",
            stderr=f"timed out after {timeout} seconds; incomplete outputs are not admitted",
        )


def kernel_entry(
    symbol: str, extents: Sequence[int], stated: Mapping[str, Any] | None, device: str
) -> tuple[dict[str, Any] | None, str, str]:
    """``(entry, provenance, refusal)`` -- what a backend is asked to emit for one routed symbol.

    Two sources, named apart because they compute different functions. A STATED group program is
    carried verbatim: it is the layer's own program -- epilogue, requantize multiplier, convolution
    geometry, committed type -- and rebuilding any of it from the extents would emit a kernel that
    computes something else. Without one the entry is synthesized as a bare contraction of those
    extents, which is honest for a caller that routed exactly the contraction.

    THE EXTENT TRIPLE IS REQUIRED ONLY OF THE SYNTHESIZED FORM. A statement carries its own shape --
    a convolution states taps and strides, an integer sum reduces over nothing -- so demanding
    ``M x K x N`` of one would decline exactly the layers a whole-program route exists to carry
    (measured on a captured ResNet-50: 54 of its 69 routed groups are convolutions). A separate
    function so that rule is reachable without a package and a toolchain: inside the build loop it
    sits behind a package load, and a test of it there passes whether or not the rule is present.
    """
    key = tuple(int(v) for v in extents)
    if stated is not None:
        entry = dict(stated)
        entry.setdefault("name", symbol)
        return entry, FROM_GROUP, ""
    if len(key) not in (3, 4):
        return None, FROM_EXTENTS, f"signature {key} has neither 3 nor 4 extents; no kernel shape for it"
    m, n, k = key[-3:]
    return (
        {
            "name": symbol,
            "op": "matmul",
            "kind": "op",
            "source_role": "mesh_tile_synthesized",
            "source_reference": f"offloaded layer {m}x{k}x{n} for {device}",
            "M": m,
            "K": k,
            "N": n,
        },
        FROM_EXTENTS,
        "",
    )


def build_device_objects(
    device: str,
    signatures: Mapping[str, Sequence[int]],
    dtypes: Mapping[str, Sequence[str]],
    *,
    package_dir: str | Path,
    workdir: str | Path,
    operand_dtype: str,
    accum_dtype: str,
    numeric_policy: dict | None = None,
    codegen_target: str = "riscv",
    cflags: Sequence[str] | None = None,
    entries: Mapping[str, Mapping[str, Any]] | None = None,
    timeout: int = 900,
    stop_on_first_failure: bool = False,
    expected_interfaces: Mapping[str, Mapping[str, str]] | None = None,
    package_sha256: str | None = None,
    tile_edge: int | None = None,
) -> DeviceBuild:
    """One kernel object per signature plus the shim object, ready to archive.

    ``signatures`` / ``dtypes`` come from the offload rewrite. ``operand_dtype`` / ``accum_dtype`` are
    the device's own datapath tokens, which the caller derived from the device rather than assumed.

    ``entries`` is ``symbol -> the group's own stated program`` (see
    :mod:`merlin.llvmlower.group_offload`), and supplying it is what makes this a WHOLE-MODEL build
    rather than a per-contraction one. Without it a kernel is synthesized from the extents alone --
    which is correct for a caller that routed exactly the contraction and kept the epilogue on the
    host, and quietly wrong for one that routed a whole group: the bias, the requantize, the
    activation and the pooling the layer carries would simply not be in the kernel, and nothing in
    the artifact would say so. Which of the two each kernel came from is recorded in
    :attr:`DeviceBuild.built_from`, per symbol.

    FAIL CLOSED WHEN AN ENTRY IS MISSING. If ``entries`` is supplied at all, a symbol absent from it
    is DECLINED by name rather than falling back to the synthesized form: the caller has said it is
    routing stated programs, so an unstated symbol is a gap in the statement, and substituting a bare
    contraction for it is exactly the silent loss this parameter exists to prevent.

    By default every failure is recorded and skipped, so a diagnostic build reports all declined
    symbols. A caller that requires every object may request ``stop_on_first_failure``; its first
    actual failure is retained, no subsequent symbol is attempted, and no shim is emitted.
    """
    from merlin.common.digest import sha256_text
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.package_runtime import load_package, run_entrypoint

    from .device_shim import emit_translation_unit, kernel_abi_for
    from .toolchain import clang, mlir_translate

    work = Path(workdir)
    if type(stop_on_first_failure) is not bool:
        raise ValueError("stop_on_first_failure must be a bool")
    work.mkdir(parents=True, exist_ok=True)
    skipped: list[tuple[str, str]] = []

    if expected_interfaces is not None:
        from .exact_offload import _package_sha256

        if set(expected_interfaces) != set(signatures) or not package_sha256:
            raise ValueError("exact offload must bind every emitted symbol and an OOT package digest")
        if _package_sha256(Path(package_dir)) != package_sha256:
            raise ValueError("OOT compiler package changed after exact model selection")
    if tile_edge is not None and (type(tile_edge) is not int or tile_edge <= 0):
        raise ValueError("explicit shim tile edge must be a positive integer")

    # WHICH DEVICES THIS PATH CAN BUILD, asked of the device's derived link rather than assumed.
    #
    # The pipeline below runs the package's artifact through mlir-translate and clang, so it works
    # exactly for a device whose artifact IS LLVM-dialect MLIR. A device driven by a command buffer
    # emits a JSON command buffer, and a self-hosted one emits its own source; handing either to
    # mlir-translate fails obscurely, and worse, the shim would declare an extern kernel symbol that
    # nothing in the archive defines. Declining with the transport named is the honest answer, and it
    # is the first consumer of the transport axis the Link derives.
    #
    # `objects_buildable`, NOT `boundary_buildable`. The two had one answer while one transport had an
    # emitter, and they diverged the moment a second one did: a `device_native` device's BOUNDARY is
    # emittable (as a DRAM address contract) while its ARTIFACT is an instruction stream this loop
    # cannot compile. Asking the boundary question here would let exactly that device through.
    unbuildable = objects_buildable(device)
    if unbuildable:
        return DeviceBuild(device=device, skipped=(("all", unbuildable),))

    # FAIL CLOSED ON THE CALLER'S CONTRACT FIRST. A symbol the caller routed as a stated program but
    # supplied no statement for is a gap in the CALLER, not in the package or the toolchain -- and
    # asking those first would report it as whichever of them happened to be unavailable, which is
    # how a dropped readout gets attributed to a missing manifest.
    if entries is not None:
        unstated = sorted(sym for sym in signatures if sym not in entries)
        if unstated:
            return DeviceBuild(
                device=device,
                skipped=tuple(
                    (
                        sym,
                        "no stated group program for this symbol; the caller routed stated programs, so "
                        "building it from the extents alone would drop whatever readout the layer carries",
                    )
                    for sym in unstated
                ),
            )

    abi = kernel_abi_for(device)
    if abi is None:
        return DeviceBuild(device=device, skipped=(("all", "no readable kernel_abi"),))
    try:
        pkg = load_package(str(package_dir))
    except Exception as exc:  # noqa: BLE001
        return DeviceBuild(device=device, skipped=(("all", f"package unusable: {exc}"),))
    if expected_interfaces is not None and pkg.target != device:
        raise ValueError("exact interface package target differs from selected device")

    if expected_interfaces is None:
        from merlin.compile.mesh import _mesh_tile_binding

        binding = _mesh_tile_binding(device, operand_dtype, accum_dtype, numeric_policy=numeric_policy)

    objs: list[Path] = []
    kernels: dict[str, str] = {}
    built_from: dict[str, str] = {}
    oc = _objcopy()
    # The selected interface is still checked and emitted for every signature. Two
    # interfaces may nevertheless produce the same LLVM artifact; compile its raw
    # object once, then rename a separate copy for each shim entry.
    compiled_by_artifact: dict[str, list[tuple[Path, Path]]] = {}

    def dedup_receipt() -> dict[str, int]:
        return {
            "unique_artifacts": sum(map(len, compiled_by_artifact.values())),
            "emitted_symbols": len(kernels),
        }

    for index, sym in enumerate(sorted(signatures)):
        if stop_on_first_failure and skipped:
            break
        key = tuple(int(v) for v in signatures[sym])
        entry, provenance, refusal = kernel_entry(sym, key, None if entries is None else entries.get(sym), device)
        if entry is None:
            skipped.append((sym, refusal))
            continue
        # A batched signature needs the SAME kernel as its unbatched form: the batch is a loop in the
        # shim over disjoint slices, not a third axis the device sees. Building a separate kernel per
        # batch size would mint one per B for identical work.
        m, n, k = key[-3:] if len(key) in (3, 4) else (None, None, None)
        want = kernel_symbol(abi.symbol, index)
        stem = work / f"{sym}"
        if expected_interfaces is None:
            try:
                _capsule, iface = CS.build(entry, binding)
            except Exception as exc:  # noqa: BLE001
                skipped.append((sym, f"interface capsule: {exc}"))
                continue
        else:
            from merlin.targetgen.contract.resident_interface_abi import bind_single_resident_matmul

            chosen = expected_interfaces[sym]
            iface = chosen.get("mlir")
            if not isinstance(iface, str) or sha256_text(iface) != chosen.get("sha256"):
                raise ValueError(f"{sym} no longer matches its selected interface bytes")
            resident = bind_single_resident_matmul(iface, target=device)
            if (resident.m, resident.n, resident.k) != (m, n, k) or resident.dtypes != tuple(dtypes[sym]):
                raise ValueError(f"{sym} selected interface disagrees with pointer ABI, device, shape, or precision")
        ifc = stem.with_suffix(".iface.mlir")
        ifc.write_text(iface, encoding="utf-8")

        try:
            r = run_entrypoint(pkg, "emit_target_artifact", ifc, timeout=timeout)
        except subprocess.TimeoutExpired:
            skipped.append((sym, f"emit_target_artifact: timed out after {timeout} seconds"))
            continue
        if r.returncode != 0:
            shape = f"{m}x{k}x{n}" if m is not None else str(entry.get("op") or "this program")
            skipped.append((sym, f"package declined {shape}: {(r.stderr or '').strip()[:200]}"))
            continue
        if expected_interfaces is not None and f"llvm.func @{abi.symbol}(" not in r.stdout:
            skipped.append((sym, "exact package artifact has no contract-named LLVM kernel entry"))
            continue
        art = stem.with_suffix(".device.mlir")
        art.write_text(r.stdout, encoding="utf-8")
        digest = hashlib.sha256(r.stdout.encode("utf-8")).hexdigest()
        # Compare actual bytes as well, so a digest collision cannot merge kernels.
        raw = next(
            (base for source, base in compiled_by_artifact.get(digest, ()) if source.read_bytes() == art.read_bytes()),
            None,
        )
        if raw is None:
            ll = stem.with_suffix(".ll")
            t = _run_build_tool([mlir_translate(), "--mlir-to-llvmir", str(art), "-o", str(ll)], timeout=timeout)
            if t.returncode != 0:
                skipped.append((sym, f"mlir-translate: {(t.stderr or '').strip()[:200]}"))
                continue

            raw = stem.with_suffix(".raw.o")
            c = _run_build_tool(
                [clang(), *_flags(codegen_target, cflags), "-c", str(ll), "-o", str(raw)], timeout=timeout
            )
            if c.returncode != 0:
                skipped.append((sym, f"clang: {(c.stderr or '').strip()[:200]}"))
                continue
            compiled_by_artifact.setdefault(digest, []).append((art, raw))

        obj = stem.with_suffix(".o")
        if oc is None:
            skipped.append((sym, "no objcopy available to give this kernel a distinct symbol"))
            continue
        rn = _run_build_tool([oc, f"--redefine-sym={abi.symbol}={want}", str(raw), str(obj)], timeout=timeout)
        if rn.returncode != 0:
            skipped.append((sym, f"symbol rename: {(rn.stderr or '').strip()[:200]}"))
            continue

        objs.append(obj)
        kernels[sym] = want
        built_from[sym] = provenance

    if stop_on_first_failure and skipped:
        return DeviceBuild(
            device=device,
            objects=tuple(objs),
            kernels=kernels,
            built_from=built_from,
            skipped=tuple(skipped),
            object_dedup=dedup_receipt(),
        )

    if not kernels:
        return DeviceBuild(device=device, skipped=tuple(skipped), object_dedup=dedup_receipt())

    unit = emit_translation_unit(
        device,
        {s: signatures[s] for s in kernels},
        {s: dtypes.get(s, ()) for s in kernels},
        kernel_symbol_for=kernels.get,
        tile_edge=tile_edge,
    )
    if not unit.symbols:
        return DeviceBuild(
            device=device,
            objects=tuple(objs),
            kernels=kernels,
            built_from=built_from,
            skipped=tuple([*skipped, *unit.skipped]),
            object_dedup=dedup_receipt(),
        )
    # A KERNEL THE SHIM DECLINED IS NOT A KERNEL THIS ARCHIVE CAN OFFER. The shim is what defines the
    # symbol the model's own call binds to, so a kernel object with no entry beside it ships a private
    # definition nothing reaches and leaves the call undefined at link -- a diagnostic that names the
    # symbol and not the reason. Drop it here, with the shim's own reason carried forward, so the
    # caller can compare what it routed against what was built.
    shimmed = set(unit.symbols)
    for sym in [s for s in kernels if s not in shimmed]:
        skipped.append(
            (sym, next((why for name, why in unit.skipped if name == sym), "the kernel ABI shim emitted no entry"))
        )
        kernels.pop(sym)
        built_from.pop(sym, None)
    shim_c = work / "device_shim.c"
    shim_c.write_text(unit.text, encoding="utf-8")
    shim_o = work / "device_shim.o"
    s = _run_build_tool(
        [clang(), *_flags(codegen_target, cflags), "-c", str(shim_c), "-o", str(shim_o)], timeout=timeout
    )
    if s.returncode != 0:
        skipped.append(("shim", f"clang: {(s.stderr or '').strip()[:300]}"))
        return DeviceBuild(
            device=device,
            objects=tuple(objs),
            kernels=kernels,
            built_from=built_from,
            skipped=tuple(skipped),
            object_dedup=dedup_receipt(),
        )

    return DeviceBuild(
        device=device,
        objects=(*objs, shim_o),
        shim_object=shim_o,
        kernels=kernels,
        built_from=built_from,
        skipped=tuple(skipped),
        object_dedup=dedup_receipt(),
    )


def _flags(codegen_target: str, cflags: Sequence[str] | None = None) -> list[str]:
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


def apply_post_offload_transform(routing, prepared: Path, sidecar: Path, workdir: Path) -> Path:
    """Apply an explicit host ABI rewrite while preserving the routing sidecar.

    The callback owns any bridge implementation. It receives the exact routed IR,
    immutable offload sidecar, and private output directory; no target ABI is
    inferred here. Absent callbacks preserve existing behavior and bytes.
    """
    callback = getattr(routing, "post_offload_transform", None)
    if callback is None:
        return Path(prepared)
    import hashlib
    import json

    prepared, sidecar, workdir = Path(prepared), Path(sidecar), Path(workdir)
    original_sidecar = sidecar.read_bytes()
    original_source = prepared.read_bytes()
    workdir.mkdir(parents=True, exist_ok=True)
    selected = Path(callback(prepared, sidecar, workdir))
    if sidecar.read_bytes() != original_sidecar:
        raise ValueError("post-offload transform changed source routing identity")
    if not selected.is_file():
        raise ValueError("post-offload transform returned no model file")
    digest = lambda data: hashlib.sha256(data).hexdigest()
    (workdir / "post_offload_transform.json").write_text(
        json.dumps(
            {
                "source_path": str(prepared.resolve()),
                "source_sha256": digest(original_source),
                "routing_sidecar_sha256": digest(original_sidecar),
                "selected_path": str(selected.resolve()),
                "selected_sha256": digest(selected.read_bytes()),
            },
            indent=2,
        )
        + "\n"
    )
    return selected
