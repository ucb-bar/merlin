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
import json
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Callable as Callable
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from merlin.compiler.lower.device.device_build_contracts import (
    FROM_EXTENTS,
    FROM_GROUP,
    _flags,
    apply_post_offload_transform,
    executable_kernel_entry,
    kernel_entry,
    kernel_symbol,
    rename_object_command,
)
from merlin.compiler.lower.device.native import (
    _SEAM_EMITTERS,
    BUILDABLE_TRANSPORTS,
    boundary_buildable,
    objects_buildable,
)
from merlin.compiler.lower.device.region import _accum_from_facts, routing_for_placement

# ---- device_build ------------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.compiler.lower.device.selection import DeviceIRBudget, DeviceOffloadSelection
    from merlin.compiler.lower.device.shim import ShimUnit
    from merlin.compiler.lower.device.sibling_source_route import SiblingSourceRoute
    from merlin.compiler.lower.device.spatial import ConnectedSpatialRoute
    from merlin.compiler.lower.device.target import TargetScheduleSelection
    from merlin.compiler.lower.host.borrowed_buffer_calls import BorrowedBufferCalls
    from merlin.compiler.lower.host.transforms import HostLLVMTransformChain
    from merlin.compiler.schedule.loops.configuration_ir import ConfigurationIRSelection
    from merlin.target.contract.build.object_admission import LinkedElfAdmissionService


@contextmanager
def _sibling_emission_batch(selection):
    """Freeze source-bound interface bytes between two complete owner checks.

    Package callbacks receive only these exact interfaces. No source/weight
    validation cache crosses a callback: the complete owner is reopened after
    all admitted callbacks and tool jobs have drained, including error paths,
    before any DeviceBuild result can be returned. Individual emitted ABI
    checks continue to consume the frozen interface's original arity.
    """
    from types import MappingProxyType

    if selection is None:
        yield MappingProxyType({})
        return
    from merlin.compiler.lower.device.sibling_source_route import _sha, component_hosts, require_host

    selected, host = selection
    require_host(host, selected)
    interfaces = {}
    for symbol, item in host.rewrite.expected_interfaces.items():
        source = item.get("mlir")
        if not isinstance(source, str) or _sha(source) != item.get("sha256"):
            raise ValueError("original sibling batch interface changed")
        interfaces[symbol] = source, len(host.rewrite.call_buffers[symbol])
    for component in component_hosts(host):
        if interfaces[component.device_symbol][0] != component.interface:
            raise ValueError("original sibling component and batch interfaces disagree")
    try:
        yield MappingProxyType(interfaces)
    finally:
        # This is deliberately a fresh full check, not a same-stat/mtime cache.
        # A failing callback cannot publish earlier objects under stale owners.
        require_host(host, selected)


@dataclass(frozen=True)
class DeviceRouting:
    """Everything a whole-model build needs to offload onto one device.

    The mirror of the matrix-unit path's routing descriptor, and required for the same reason: a build
    that enabled offload without saying WHICH device and WHICH backend package would have to guess
    both, and a guessed package emits kernels for the wrong hardware that link and run.

    ``select`` is one placement decision, passed in rather than made here -- see
    :mod:`merlin.target.system.place`. A source-bound catalog can instead select the exact
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
    #: rewrite the build happened to call. See :mod:`merlin.compiler.lower.device.offload`.
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
    #: Original-caller static compilation resource policy, never read from emitted programs.
    device_ir_budget: DeviceIRBudget | None = None
    #: Original full-model qualification requests every source-admitted group.
    require_complete_groups: bool = False
    #: Original selected target-IR cleanup; requires a declared consuming emitter.
    target_configuration: ConfigurationIRSelection | None = None
    #: Original current-source narrowing; never device eligibility or qualification.
    offload_selection: DeviceOffloadSelection | None = None
    #: Expose only checked generated-shim descriptor identity, retaining kernel effects.
    descriptor_results: bool = False
    #: Original live owner qualifying the actual kernel/runtime borrowed ABI.
    descriptor_effects: Callable[[Path, DeviceBuild], BorrowedBufferCalls] | None = None
    #: Original selected LLVM emission stages; no device-effect qualification is inferred.
    llvm_transform_chain: HostLLVMTransformChain | None = None

    #: Original selected finite target scheduling; no target effect qualification is inferred.
    target_schedule: TargetScheduleSelection | None = None

    #: Original source pair, staged plan and explicit host scratch ownership.
    connected_spatial: ConnectedSpatialRoute | None = None
    #: Original source identity and selected synchronous borrowed sibling effects.
    sibling_source: SiblingSourceRoute | None = None
    #: Explicit current-source producer Q1 and ordered residual grouping.
    quantized_sum: object | None = None
    #: Original software-owned logical layout policy, consumed before complete group binding.
    pre_group_layout: object | None = None
    #: Original software-owned final host allocation lifetime policy.
    intermediate_storage: object | None = None

    def __post_init__(self):
        if self.quantized_sum is not None:
            from merlin.compiler.lower.device.outline import freeze_route

            freeze_route(self)
        if self.sibling_source is not None:
            from merlin.compiler.lower.device.sibling_source_route import freeze_route

            freeze_route(self)
        if self.connected_spatial is not None:
            from merlin.compiler.lower.device.spatial import freeze_route

            freeze_route(self)
        if self.target_schedule is not None:
            from merlin.compiler.lower.device.target import TargetScheduleSelection

            if type(self.target_schedule) is not TargetScheduleSelection:
                raise TypeError("target schedule requires an original TargetScheduleSelection")
            if self.target_configuration is not None:
                raise ValueError("target schedule and configuration selections are mutually exclusive")
        if self.llvm_transform_chain is not None:
            from merlin.compiler.lower.host.transforms import HostLLVMTransformChain

            if type(self.llvm_transform_chain) is not HostLLVMTransformChain:
                raise TypeError("device LLVM requires an original typed transform chain")
        if type(self.descriptor_results) is not bool:
            raise TypeError("descriptor results require an original Boolean")
        if type(self.require_complete_groups) is not bool:
            raise TypeError("complete source group routing requires an original Boolean")
        if self.require_complete_groups and self.granularity != "group":
            raise ValueError("complete source group routing requires group granularity")


def freeze_device_elf_admission(device):
    """Freeze supplied policy selection; an empty host route needs no policy."""
    from merlin.target.contract.build.object_admission import LinkedElfAdmissionService

    service = getattr(device, "linked_elf_admission", None)
    if service is None:
        return None
    if type(service) is not LinkedElfAdmissionService:
        raise ValueError("device linked ELF admission requires an explicitly selected service")
    return device.device, service, service.verify(device.device)


def freeze_device_ir_budget(device):
    """Copy original caller policy before any package or provider callback runs."""
    from merlin.compiler.lower.device.selection import DeviceIRBudget

    selected = getattr(device, "device_ir_budget", None)
    if selected is not None and type(selected) is not DeviceIRBudget:
        raise ValueError("device_ir_budget must be an original-caller DeviceIRBudget")
    return DeviceIRBudget(**(selected.record() if selected is not None else {}))


def freeze_device_target_configuration(device):
    """Bind original target pass selection before any provider callback runs."""
    selected = getattr(device, "target_configuration", None)
    return _freeze_target_configuration(selected, getattr(device, "device", None))


def _freeze_target_configuration(selected, target):
    if selected is None:
        return None
    from merlin.compiler.lower.device.target import freeze_configuration

    return freeze_configuration(selected, target)


def require_device_target_configuration(device, frozen):
    """A later callback cannot replace the original configuration selection."""
    if frozen is None and getattr(device, "target_configuration", None) is None:
        return None
    from merlin.compiler.lower.device.target import require_configuration

    return require_configuration(device, frozen)


def revalidate_device_target_artifacts(device, frozen, artifacts):
    """Reopen original configuration receipts after later provider callbacks."""
    selected = require_device_target_configuration(device, frozen)
    if selected is None:
        return
    from merlin.compiler.lower.device.target import configuration_artifacts

    configuration_artifacts(device, selected, artifacts)


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

    from merlin.common.digest import sha256_file
    from merlin.common.process import invocation_records as invocation_record

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


@dataclass(frozen=True)
class _DeviceElfCustody:
    elf: Path
    linked_sha256: str
    selection: tuple = field(repr=False)
    result: dict = field(repr=False)
    result_record: bytes = field(repr=False)
    selection_record: bytes = field(repr=False)
    reports: tuple = field(repr=False)
    original: tuple = field(repr=False)


def _device_elf_custody_binding(custody):
    from merlin.common.digest import canonical_json

    if (
        type(custody) is not _DeviceElfCustody
        or (
            custody.elf,
            custody.linked_sha256,
            id(custody.selection),
            id(custody.result),
            custody.result_record,
            custody.selection_record,
            custody.reports,
        )
        != custody.original
    ):
        raise ValueError("device linked ELF custody binding changed")
    if (
        canonical_json(custody.result) != custody.result_record
        or canonical_json(custody.selection[2]) != custody.selection_record
    ):
        raise ValueError("device linked ELF original result or policy receipt changed")


def _device_elf_report_bytes(reports):
    from merlin.common.digest import sha256_file

    for path, digest in reports:
        if (
            not path.is_absolute()
            or any(p.is_symlink() for p in (path, *path.parents))
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise ValueError("device linked ELF frozen report bytes changed")


def freeze_device_elf_custody(device, selection, *, elf, linked_sha256, result):
    """Freeze the original accepted result before another caller callback.

    This retains actual receipt/report bytes, not policy or execution authority.
    The caller still owns stable filesystem inputs during the checked boundary.
    """
    from merlin.common.digest import canonical_json, is_sha256

    # Capture original values before service validation can invoke a validator.
    if type(result) is not dict or selection is None or not is_sha256(linked_sha256):
        raise ValueError("device linked ELF custody requires an original result and link digest")
    result_record, selection_record = canonical_json(result), canonical_json(selection[2])
    snapshot = json.loads(result_record)
    if snapshot.get("status") != "accepted" or snapshot.get("elf_sha256") != linked_sha256:
        raise ValueError("device linked ELF custody requires the original accepted linked image")
    reports, row = [], snapshot
    while True:
        if len(reports) >= 32 or type(row) is not dict:
            raise ValueError("device linked ELF custody report chain is invalid or exceeds 32 services")
        reports.append((Path(row["report_path"]), row["report_sha256"]))
        if "prerequisite" not in row:
            break
        row = row["prerequisite"]
    elf, reports = Path(elf), tuple(reports)
    original = (elf, linked_sha256, id(selection), id(result), result_record, selection_record, reports)
    custody = _DeviceElfCustody(
        elf, linked_sha256, selection, result, result_record, selection_record, reports, original
    )
    revalidate_device_elf_custody(device, selection, custody)
    return custody


def revalidate_device_elf_custody(device, selection, custody):
    """Reopen frozen original reports after validators; read linked ELF last.

    Extra selection/provider callbacks may run between freeze and this check.
    They cannot replace the original service/result or endorse changed reports.
    This check adds no policy and is not an atomic filesystem observation.
    """
    from merlin.common.digest import canonical_json, sha256_file

    _device_elf_custody_binding(custody)
    if selection is not custody.selection:
        raise ValueError("device linked ELF original policy selection changed")
    service = require_device_elf_admission(device, selection)
    snapshot = json.loads(custody.result_record)
    _device_elf_report_bytes(custody.reports)
    if service.revalidate(elf=custody.elf, result=snapshot, target=selection[0]) != "accepted":
        raise ValueError("device linked ELF original accepted policy was refused")
    # The validator is an original callback: reopen all its possible outputs
    # without invoking it again, and check its original implementation selection.
    require_device_elf_admission(device, selection)
    _device_elf_custody_binding(custody)
    if canonical_json(snapshot) != custody.result_record:
        raise ValueError("device linked ELF isolated result changed during validation")
    _device_elf_report_bytes(custody.reports)
    if sha256_file(custody.elf) != custody.linked_sha256:
        raise ValueError("device linked ELF changed after original callback validation")


@dataclass(frozen=True)
class DeviceBuild:
    """What was built, and what could not be."""

    device: str
    #: Objects to add to the archive: one kernel per signature, plus the shim.
    objects: tuple[Path, ...] = ()
    shim_object: Path | None = None
    #: shim entry symbol -> the kernel symbol it calls.
    kernels: dict[str, str] = field(default_factory=dict)
    #: Actual translation/compile/rename invocations for each linked kernel.
    custody: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    skipped: tuple[tuple[str, str], ...] = ()
    #: symbol -> which entry the kernel was built from: :data:`FROM_GROUP` (the layer's own stated
    #: program, epilogue and all) or :data:`FROM_EXTENTS` (a bare contraction of those extents).
    #: A MIXED BUILD MUST NOT READ AS ONE MECHANISM: the two compute different functions, and an
    #: archive that held some of each while reporting only "N kernels" would say nothing about
    #: which of a model's layers kept their readout.
    built_from: dict[str, str] = field(default_factory=dict)
    #: Coarse build receipt: distinct compiled artifact texts versus exported kernel symbols.
    object_dedup: dict[str, int] = field(default_factory=dict)
    build_metrics: dict[str, dict] = field(default_factory=dict)
    shim_unit: ShimUnit | None = None
    shim_invocation: dict | None = None

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


def _ar() -> str | None:
    from merlin.compiler.lower.model.toolchain import llvm_install

    local = llvm_install() / "bin" / "llvm-ar"
    if local.exists():
        return str(local)
    return shutil.which("llvm-ar") or shutil.which("ar")


def _objcopy() -> str | None:
    from merlin.compiler.lower.model.toolchain import llvm_install

    local = llvm_install() / "bin" / "llvm-objcopy"
    if local.exists():
        return str(local)
    return shutil.which("llvm-objcopy") or shutil.which("objcopy")


def _nm() -> str | None:
    from merlin.compiler.lower.model.toolchain import llvm_install

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
    """Verify the staged objects with this build owner's selected tools."""
    from merlin.compiler.lower.device.device_object_symbols import verify_object_symbol_binding as verify

    return verify(
        kernel_object,
        shim_object,
        entry_symbol=entry_symbol,
        kernel_symbol=kernel_symbol,
        original_kernel_symbol=original_kernel_symbol,
        timeout=timeout,
        nm=_nm(),
        run=_run,
    )


def _run(argv: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess:
    from merlin.compiler.lower.options.optional_passes import child_environment

    return subprocess.run(
        [str(a) for a in argv], capture_output=True, text=True, timeout=timeout, env=child_environment()
    )


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


def _observed_build_tool(
    argv,
    *,
    timeout,
    directory,
    stage,
    inputs,
    output,
    metrics=None,
    expected_input_sha256=None,
    metrics_stage="compilation_s",
    dependencies=(),
):
    """Return a process, closed file-identity pin, and separate wall duration."""
    from merlin.common.process import invocation_records as invocation_record
    from merlin.common.process.cpu_pool import shared_compute_pool

    with (
        shared_compute_pool(timeout_s=timeout),
        invocation_record.observe(
            directory,
            stage=stage,
            argv=argv,
            inputs=inputs,
            outputs=(output,),
            dependencies=tuple(
                dict.fromkeys(
                    (
                        Path(__file__),
                        Path(__file__).with_name("device_build_jobs.py"),
                        Path(__file__).with_name("device_build_contracts.py"),
                        Path(__file__).with_name("device_object_optimization.py"),
                        *dependencies,
                    )
                )
            ),
        ) as observed,
    ):
        if metrics is not None:
            observed.document["device_build_metrics"] = metrics
            observed._write()
        if expected_input_sha256 is not None and (
            len(observed.document["inputs"]) != 1 or observed.document["inputs"][0]["sha256"] != expected_input_sha256
        ):
            raise ValueError("device LLVM changed after its static build admission")
        started = time.monotonic()
        if metrics is not None:
            metrics["compiler_started"] = True
        result = _run_build_tool(argv, timeout=timeout)
        observed.document["tool_elapsed_s"] = time.monotonic() - started
        if metrics is not None:
            metrics["stages"][metrics_stage] = observed.document["tool_elapsed_s"]
        observed.complete(result)
    path = observed.path
    return (
        result,
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        },
        observed.document["tool_elapsed_s"],
    )


__all__ = [
    "DeviceBuild",
    "DeviceRouting",
    "FROM_EXTENTS",
    "FROM_GROUP",
    "build_device_objects",
    "kernel_entry",
    "kernel_symbol",
    "routing_for_placement",
]
