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
import os
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Callable as Callable
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import copy_context
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
    call_buffers: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    device_ir_budget: DeviceIRBudget | None = None,
    target_configuration: ConfigurationIRSelection | None = None,
    llvm_transform_chain: HostLLVMTransformChain | None = None,
    descriptor_results: bool = False,
    target_schedule: TargetScheduleSelection | None = None,
    connected_spatial: tuple | None = None,
    sibling_source: tuple | None = None,
) -> DeviceBuild:
    """One kernel object per signature plus the shim object, ready to archive.

    ``call_buffers`` is ``symbol -> the host call's operands`` (``{"role", "shape", "dtype"}`` in call
    order), recorded by a group rewrite under the logical kernel ABI. With it every pointer of each
    kernel's logical interface is passed from the host call (:func:`device_shim.logical_buffers`,
    :func:`device_shim.emit_logical_translation_unit`); an interface value the call does not supply is
    declined by name.

    ``signatures`` / ``dtypes`` come from the offload rewrite. ``operand_dtype`` / ``accum_dtype`` are
    the device's own datapath tokens, which the caller derived from the device rather than assumed.

    ``entries`` is ``symbol -> the group's own stated program`` (see
    :mod:`merlin.compiler.lower.device.group_route`), and supplying it is what makes this a WHOLE-MODEL build
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

    ``llvm_transform_chain`` explicitly selects original-owned checked LLVM stages
    before object optimization; selected Python callbacks run serially on the
    caller thread. It grants no instruction effects or target qualification.

    By default every failure is recorded and skipped, so a diagnostic build reports all declined
    symbols. A caller that requires every object may request ``stop_on_first_failure``; its first
    actual failure is retained, no subsequent symbol is admitted, and no shim is emitted.
    Already admitted external tool jobs drain and retain their genuine outputs. Package emission
    stays on its owner thread unless that owner explicitly supports concurrent entrypoints.
    Admitted compiler emissions and independent LLVM/tool jobs use the declared shared
    compute capacity (or explicit ``MERLIN_DEVICE_BUILD_JOBS``). Legacy callers default to one job.
    """
    from merlin.compiler.lower.device.artifacts import (
        DeviceBuild,
        _freeze_target_configuration,
        _objcopy,
        _observed_build_tool,
        _sibling_emission_batch,
        verify_object_symbol_binding,
    )

    if type(descriptor_results) is not bool:
        raise ValueError("descriptor results require an explicit original boolean")
    private_symbol = None
    if connected_spatial is not None:
        from merlin.compiler.lower.device.spatial import ConnectedSpatialHost, ConnectedSpatialRoute

        selected_spatial, spatial_host = connected_spatial
        if type(selected_spatial) is not ConnectedSpatialRoute or type(spatial_host) is not ConnectedSpatialHost:
            raise ValueError("private spatial object requires the original host ownership preparation")
        ConnectedSpatialRoute.record(selected_spatial)
        private_symbol = selected_spatial.writer.symbol
        if (
            entries is None
            or private_symbol not in signatures
            or entries.get(private_symbol) != spatial_host.source_entry
            or call_buffers is None
        ):
            raise ValueError("private spatial object lost its original routed source and logical call")
    sibling_symbol = None
    sibling_hosts = {}
    if sibling_source is not None:
        from merlin.compiler.lower.device.sibling_source_route import component_hosts, require_host

        selected_sibling, sibling_host = sibling_source
        require_host(sibling_host, selected_sibling)
        sibling_hosts = {item.device_symbol: item for item in component_hosts(sibling_host)}
        if (
            connected_spatial is not None
            or signatures != sibling_host.rewrite.signatures
            or entries != sibling_host.rewrite.entries
            or call_buffers != sibling_host.rewrite.call_buffers
            or expected_interfaces != sibling_host.rewrite.expected_interfaces
            or package_sha256 != sibling_host.rewrite.package_sha256
        ):
            raise ValueError("sibling object lost its original source, interface or logical ABI")
    from merlin.common.digest import sha256_text
    from merlin.compiler.lower.device import device_object_optimization as object_input
    from merlin.compiler.lower.device.device_object_optimization import observe as optimize_object_input
    from merlin.compiler.lower.device.device_shim import emit_logical_translation_unit, logical_buffers
    from merlin.compiler.lower.device.selection import DeviceIRBudget, inspect_device_ir
    from merlin.compiler.lower.device.shim import (
        descriptor_result_dependencies,
        emit_translation_unit,
        kernel_abi_for,
        logical_argument_roles,
    )
    from merlin.compiler.lower.model.toolchain import clang, mlir_translate
    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import capsule_io as CC
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1
    from merlin.development.targetgen.packaging.loading import BackendDeclined, CertFailure, load_package

    frozen_transform = object_input.freeze_transform_chain(llvm_transform_chain)
    # Opted-in Python stages stay on their original owner thread. Legacy tools
    # retain their existing parallel execution; no callback context is copied.
    workers = build_workers()
    if frozen_transform is not None:
        workers = 1
        if not signatures:
            raise ValueError("selected device LLVM transform has no routed object consumer")
    if target_schedule is not None and target_configuration is not None:
        raise ValueError("target schedule and configuration selections are mutually exclusive")
    selected_target = target_configuration if target_schedule is None else target_schedule
    target_kind = "target_configuration" if target_schedule is None else "target_schedule"
    freeze_target = _freeze_target_configuration
    if target_schedule is not None:
        from merlin.compiler.lower.device.preparation import _freeze_target_schedule

        freeze_target = _freeze_target_schedule
    revalidate_target = CC.revalidate_target_configuration if target_schedule is None else CC.revalidate_target_schedule
    target_selection = freeze_target(selected_target, device)
    if target_schedule is not None:
        workers = 1
        if not signatures:
            raise ValueError("selected target schedule has no routed object consumer")
    if device_ir_budget is not None and type(device_ir_budget) is not DeviceIRBudget:
        raise ValueError("device_ir_budget must be an original-caller DeviceIRBudget")
    # Freeze before the package runs. Its returned LLVM/options cannot change this policy.
    budget = DeviceIRBudget(**(device_ir_budget.record() if device_ir_budget is not None else {}))
    work = Path(workdir)
    if type(stop_on_first_failure) is not bool:
        raise ValueError("stop_on_first_failure must be a bool")
    work.mkdir(parents=True, exist_ok=True)
    skipped: list[tuple[str, str]] = []

    if expected_interfaces is not None:
        from merlin.compiler.lower.device.head import _package_sha256

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
        if target_schedule is not None:
            raise ValueError("selected target schedule has no LLVM object consumer: " + unbuildable)
        if frozen_transform is not None:
            raise ValueError("selected device LLVM transform has no LLVM object consumer: " + unbuildable)
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
    if (expected_interfaces is not None or target_selection is not None) and pkg.target != device:
        raise ValueError("exact interface package target differs from selected device")
    entry_symbol = executable_kernel_entry(device)

    if expected_interfaces is None:
        from merlin.compiler.selection.kernels import _mesh_tile_binding

        binding = _mesh_tile_binding(device, operand_dtype, accum_dtype, numeric_policy=numeric_policy)

    objs: list[Path] = []
    kernels: dict[str, str] = {}
    built_from: dict[str, str] = {}
    # Under the logical kernel ABI each kernel's pointer order is its own interface's.
    logical_abi = getattr(abi, "version", 1) == 2
    argument_roles: dict[str, tuple[str, ...]] = {}
    grouped = logical_abi and call_buffers is not None
    group_buffers: dict[str, tuple] = {}
    oc = _objcopy()
    # The selected interface is still checked and emitted for every signature. Two
    # interfaces may nevertheless produce the same LLVM artifact; compile its raw
    # object once, then rename a separate copy for each shim entry.
    custody = {}
    build_metrics = {}
    emission_times = {}
    target_receipts = {}
    private_receipts = {}
    objects_by_symbol = {}
    pending = deque()

    def dedup_receipt() -> dict[str, int]:
        return {"unique_artifacts": compiler.completed_count(), "emitted_symbols": len(kernels)}

    def compile_artifact(art, ll, raw, translate_command, compile_command, artifact_sha256):
        t, translated_invocation, translation_s = _observed_build_tool(
            translate_command,
            timeout=timeout,
            directory=work,
            stage="device_llvm_translation",
            inputs=(art,),
            output=ll,
            expected_input_sha256=artifact_sha256,
        )
        if t.returncode != 0:
            return ArtifactBuild(refusal=f"mlir-translate: {(t.stderr or '').strip()[:200]}")
        emission = None
        try:
            if frozen_transform is not None:
                ll, emission = object_input.transform_input(
                    ll, ll.with_suffix(".transform"), llvm_transform_chain, frozen_transform
                )
                compile_command = [*compile_command[:-3], str(ll), *compile_command[-2:]]
            report = inspect_device_ir(ll, budget)
        except (OSError, ValueError) as error:
            return ArtifactBuild(refusal=f"device_ir_analysis: {type(error).__name__}: {error}")
        report["stages"] = {
            "translation_s": translation_s,
            "analysis_s": report["analysis"]["elapsed_s"],
            "compilation_s": None,
            "optimization_s": None,
            "optimized_import_s": None,
        }
        report["compiler_started"] = False
        report_path = ll.with_suffix(".build-cost.json")
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        if report["decision"]["status"] != "accepted":
            return ArtifactBuild(
                custody={"build_metrics": report, "build_metrics_path": str(report_path)},
                refusal=f"{report['decision']['reason']}: {report['decision']['exceeded']}; "
                f"retain counted loops; report={report_path.name}",
            )
        try:

            def observe(*args, **kwargs):
                if emission is not None:
                    object_input.recheck_transform_chain(llvm_transform_chain, frozen_transform, emission)
                    kwargs["dependencies"] = object_input.transform_dependencies(frozen_transform, emission)
                result = _observed_build_tool(*args, **kwargs)
                object_input.recheck_transform_chain(llvm_transform_chain, frozen_transform, emission)
                return result

            optimized, refusal = optimize_object_input(
                compile_command,
                translate_command[0],
                observer=observe,
                timeout=timeout,
                directory=work,
                report=report,
            )
            if refusal:
                report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
                return ArtifactBuild(
                    custody={"build_metrics": report, "build_metrics_path": str(report_path)}, refusal=refusal
                )
            c, compiled_invocation, compilation_s = observe(
                optimized["argv"],
                timeout=timeout,
                directory=work,
                stage="device_object_compilation",
                inputs=(Path(optimized["argv"][-3]),),
                output=raw,
                metrics=report,
                expected_input_sha256=optimized["sha256"],
            )
        except ValueError as error:
            report["decision"].update(status="refused", reason="device_ir_changed_after_analysis", action=str(error))
            report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
            return ArtifactBuild(
                custody={"build_metrics": report, "build_metrics_path": str(report_path)}, refusal=str(error)
            )
        report["compiler_started"] = True
        report["stages"]["compilation_s"] = compilation_s
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        if c.returncode != 0:
            return ArtifactBuild(
                custody={"build_metrics": report, "build_metrics_path": str(report_path)},
                refusal=f"clang: {(c.stderr or '').strip()[:200]}",
            )
        return ArtifactBuild(
            raw,
            {
                "translation": translated_invocation,
                "compilation": compiled_invocation,
                "optimization": optimized["optimization"],
                "optimized_import": optimized["optimized_import"],
                "build_metrics": report,
                "build_metrics_path": str(report_path),
                **({"llvm_transform": emission} if emission is not None else {}),
            },
        )

    def finish_one():
        # Refill after any completed job: one slow group must not idle the other
        # compute slots. Final collection below restores original symbol order.
        wait([record[-1] for record in pending], return_when=FIRST_COMPLETED)
        record = next(record for record in pending if record[-1].done())
        pending.remove(record)
        sym, provenance, want, stem, future = record
        built = future.result()
        object_input.recheck_transform_chain(
            llvm_transform_chain, frozen_transform, built.custody.get("llvm_transform")
        )
        if "build_metrics" in built.custody:
            # A fresh symbol/shape record wraps any shared byte-identical analysis.
            build_metrics[sym] = {
                **built.custody["build_metrics"],
                "stages": {**built.custody["build_metrics"]["stages"], "package_emission_s": emission_times[sym]},
                "shape": list(signatures[sym]),
                "dtypes": list(dtypes[sym]),
                "compiler_package_sha256": package_sha256,
                "report_path": built.custody["build_metrics_path"],
            }
            context = {
                "shape": list(signatures[sym]),
                "dtypes": list(dtypes[sym]),
                "compiler_package_sha256": package_sha256,
                "policy": budget.record(),
                "artifact_sha256": built.custody["build_metrics"]["analysis"]["sha256"],
                "compilation_invocation": built.custody.get("compilation"),
            }
            build_metrics[sym]["context_sha256"] = hashlib.sha256(
                json.dumps(context, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
        if built.refusal:
            skipped.append((sym, built.refusal))
            return
        target_receipt = None
        if sym in target_receipts:
            lowering, receipt_sha256 = target_receipts[sym]
            if freeze_target(selected_target, device) != target_selection:
                raise ValueError("target configuration context changed during native compilation")
            revalidate_target(
                pkg,
                lowering,
                selected_target,
                expected_receipt_sha256=receipt_sha256,
            )
            target_receipt = {
                "path": str(lowering / (target_kind + ".json")),
                "sha256": receipt_sha256,
            }
        raw = built.raw
        translated = raw.with_name(raw.name.removesuffix(".raw.o") + ".ll")
        ll = stem.with_suffix(".ll")
        if translated != ll:
            # Keep each selected interface beside its exact shared translation.
            ll.write_bytes(translated.read_bytes())
        obj = stem.with_suffix(".o")
        if oc is None:
            skipped.append((sym, "no objcopy available to give this kernel a distinct symbol"))
            return
        rn, renamed_invocation, _rename_s = _observed_build_tool(
            rename_object_command(oc, entry_symbol, want, raw, obj),
            timeout=timeout,
            directory=work,
            stage="device_object_symbol_rename",
            inputs=(raw,),
            output=obj,
        )
        if rn.returncode != 0:
            skipped.append((sym, f"symbol rename: {(rn.stderr or '').strip()[:200]}"))
            return
        objs.append(obj)
        objects_by_symbol[sym] = obj
        kernels[sym] = want
        custody[sym] = {
            key: built.custody[key] for key in ("translation", "optimization", "optimized_import", "compilation")
        }
        custody[sym]["rename"] = renamed_invocation
        if sym in private_receipts:
            custody[sym]["connected_spatial"] = private_receipts[sym]
        if "llvm_transform" in built.custody:
            custody[sym]["llvm_transform"] = built.custody["llvm_transform"]
        if target_receipt is not None:
            custody[sym][target_kind] = target_receipt
        built_from[sym] = provenance

    def prepare_emissions():
        for index, sym in enumerate(sorted(signatures)):
            if stop_on_first_failure and (skipped or compiler.has_failed()):
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
            if sibling_source is not None:
                iface = sibling_interfaces[sym][0]
            elif expected_interfaces is None:
                try:
                    _capsule, iface = _merlin_model_builders_1.build(entry, binding)
                except Exception as exc:  # noqa: BLE001
                    skipped.append((sym, f"interface capsule: {exc}"))
                    continue
            else:
                from merlin.target.contract.interfaces.matmul import bind_single_resident_matmul

                chosen = expected_interfaces[sym]
                iface = chosen.get("mlir")
                if not isinstance(iface, str) or sha256_text(iface) != chosen.get("sha256"):
                    raise ValueError(f"{sym} no longer matches its selected interface bytes")
                resident = bind_single_resident_matmul(iface, target=device)
                if (resident.m, resident.n, resident.k) != (m, n, k) or resident.dtypes != tuple(dtypes[sym]):
                    raise ValueError(
                        f"{sym} selected interface disagrees with pointer ABI, device, shape, or precision"
                    )
            if grouped:
                observed_outputs = None
                if entry.get("schema") in ("connected_spatial_entry_v1", "connected_quantized_sum_entry_v1"):
                    operation = _capsule.get("operation") or {}
                    if operation.get("op") not in ("spatial_producer_pool", "quantized_producer_sum"):
                        raise ValueError("connected source entry lost its original composed program")
                    observed_outputs = (operation["attributes"]["out"],)
                buffers, refusal = (
                    logical_buffers(iface)
                    if observed_outputs is None
                    else logical_buffers(iface, observed_outputs=observed_outputs)
                )
                call = (call_buffers or {}).get(sym)
                if buffers is None or call is None:
                    skipped.append((sym, refusal or "the group rewrite recorded no host call for this symbol"))
                    continue
                group_buffers[sym] = (buffers, tuple(call))
            elif logical_abi:
                roles, refusal = logical_argument_roles(iface)
                if roles is None:
                    skipped.append((sym, refusal))
                    continue
                argument_roles[sym] = roles
            ifc = stem.with_suffix(".iface.mlir")
            ifc.write_text(iface, encoding="utf-8")

            yield sym, provenance, want, stem, ifc

    def emit_interface(job):
        from merlin.compiler.lower.device.region_lowering import lower_region

        sym, _provenance, _want, _stem, ifc = job
        # Model builds must honor the same checked command-buffer
        # acceptance as capsule builds. A successful artifact process
        # can still accompany an explicit refusal or a fence-only stub.
        lowering = work / f"{sym}.lowering"

        def check_inputs():
            object_input.recheck_transform_chain(llvm_transform_chain, frozen_transform)

        return lower_region(
            pkg,
            ifc,
            lowering,
            timeout=timeout,
            check_inputs=check_inputs,
            target_kind=target_kind if target_selection is not None else None,
            target=selected_target,
            target_is_current=lambda: freeze_target(selected_target, device) == target_selection,
            revalidate_target=revalidate_target,
        )

    # A selected target transformation has its own callback/lifetime contract;
    # package-entrypoint concurrency alone cannot authorize sharing that owner.
    with (
        _sibling_emission_batch(sibling_source) as sibling_interfaces,
        ArtifactCompiler(workers) as compiler,
        EmissionJobs(workers if target_selection is None else 1) as emissions,
    ):
        for job, emission in emissions.map(prepare_emissions(), emit_interface):
            sym, provenance, want, stem, _ifc = job
            lowering = work / f"{sym}.lowering"
            try:
                artifact, receipt_sha256, emission_times[sym] = emission.result()
                if receipt_sha256 is not None:
                    target_receipts[sym] = lowering, receipt_sha256
            except subprocess.TimeoutExpired:
                skipped.append((sym, f"package lowering: timed out after {timeout} seconds"))
                continue
            except (BackendDeclined, CertFailure, ValueError) as error:
                skipped.append((sym, f"package lowering: {type(error).__name__}: {str(error)[:300]}"))
                continue
            if sibling_source is not None:
                from merlin.compiler.lower.device.sibling_source_route import require_logical_emission

                require_logical_emission(artifact, entry_symbol=entry_symbol, arity=sibling_interfaces[sym][1])
            elif expected_interfaces is not None:
                from merlin.compiler.lower.device.sibling_source_route import require_logical_emission

                buffers, refusal = logical_buffers(expected_interfaces[sym]["mlir"])
                if buffers is None:
                    skipped.append((sym, refusal or "exact package interface has no complete pointer ABI"))
                    continue
                try:
                    require_logical_emission(artifact, entry_symbol=entry_symbol, arity=len(buffers))
                except ValueError as error:
                    skipped.append((sym, f"exact package artifact: {error}"))
                    continue
            art = stem.with_suffix(".device.mlir")
            # Keep the checked producer path for proof custody without a second
            # full artifact copy. Both names are in this build's filesystem.
            art.unlink(missing_ok=True)
            if sym == private_symbol:
                from merlin.compiler.lower.device.spatial import select_emission

                artifact, private_receipts[sym] = select_emission(
                    artifact, selected_spatial, spatial_host, entry_symbol=entry_symbol
                )
                # The independently checked producer remains in lowering/. A
                # literal closed CFG derivation is a distinct compilation input.
                art.write_text(artifact)
                stem.with_suffix(".private-helper.json").write_text(
                    json.dumps(private_receipts[sym], sort_keys=True, indent=2) + "\n"
                )
            else:
                art.hardlink_to(lowering / "lowered.llvm.mlir")
            digest = hashlib.sha256(artifact.encode("utf-8")).hexdigest()
            ll = stem.with_suffix(".ll")
            raw = stem.with_suffix(".raw.o")
            # Resolve original tool selection on the caller thread. Workers receive only
            # frozen external commands, not the package/executor's contextual services.
            translate_command = [mlir_translate(), "--mlir-to-llvmir", str(art), "-o", str(ll)]
            compile_command = [clang(), *_flags(codegen_target, cflags), "-c", str(ll), "-o", str(raw)]
            future = compiler.submit(
                art,
                digest,
                partial(compile_artifact, art, ll, raw, translate_command, compile_command, digest),
            )
            pending.append((sym, provenance, want, stem, future))
            if len(pending) >= workers:
                finish_one()
        while pending:
            finish_one()

    objs[:] = [objects_by_symbol[sym] for sym in sorted(kernels)]
    for records in (kernels, custody, built_from):
        ordered = sorted(records.items())
        records.clear()
        records.update(ordered)
    skipped.sort(key=lambda row: row[0])

    if stop_on_first_failure and skipped:
        return DeviceBuild(
            device=device,
            objects=tuple(objs),
            kernels=kernels,
            custody=custody,
            built_from=built_from,
            skipped=tuple(skipped),
            object_dedup=dedup_receipt(),
            build_metrics=build_metrics,
        )

    if not kernels:
        return DeviceBuild(
            device=device, skipped=tuple(skipped), object_dedup=dedup_receipt(), build_metrics=build_metrics
        )

    unit = (
        emit_logical_translation_unit(
            device,
            {s: group_buffers[s] for s in kernels if s != private_symbol},
            kernel_symbol_for=kernels.get,
            descriptor_results=descriptor_results,
        )
        if grouped
        else emit_translation_unit(
            device,
            {s: signatures[s] for s in kernels},
            {s: dtypes.get(s, ()) for s in kernels},
            kernel_symbol_for=kernels.get,
            tile_edge=tile_edge,
            argument_roles=argument_roles if logical_abi else None,
            descriptor_results=descriptor_results,
        )
    )
    built_siblings = tuple(symbol for symbol in sibling_hosts if symbol in kernels)
    if built_siblings:
        from merlin.compiler.lower.device.sibling_source_route import with_borrowed_bridges

        # All external package/target callbacks have finished. Reopen the whole
        # source owner once for this pure final bridge assembly, without sharing
        # that validation with any earlier or later build operation.
        unit = with_borrowed_bridges(unit, selected_sibling, sibling_host, symbols=built_siblings)
    for sibling_symbol, component_host in sibling_hosts.items():
        if sibling_symbol not in kernels:
            continue
        custody[sibling_symbol]["sibling_source"] = {
            "source_ir_sha256": selected_sibling.source_ir_sha256,
            "interface_sha256": hashlib.sha256(component_host.interface.encode()).hexdigest(),
            "borrowed_effects": selected_sibling.record()["effects"],
            "original_epilogues_and_all_live_outputs_preserved": True,
        }
    if private_symbol in kernels:
        from merlin.compiler.lower.device.spatial import with_private_bridge

        unit = with_private_bridge(
            unit, selected_spatial, spatial_host, symbol=private_symbol, kernel=kernels[private_symbol]
        )
        # Keep the unchanged strict two-object guard applicable even when the
        # ordinary model shim also forwards many unrelated routed kernels.
        private_c = work / "spatial_descriptor_bridge.c"
        private_o = work / "spatial_descriptor_bridge.audit.o"
        private_c.write_text(spatial_host.scratch.bridge_source)
        audit, bridge_invocation, _ = _observed_build_tool(
            [clang(), *_flags(codegen_target, cflags), "-c", str(private_c), "-o", str(private_o)],
            timeout=timeout,
            directory=work,
            stage="spatial_descriptor_bridge_audit",
            inputs=(private_c,),
            output=private_o,
        )
        if audit.returncode != 0:
            raise ValueError("private spatial descriptor bridge compilation failed: " + (audit.stderr or "")[:300])
        binding = verify_object_symbol_binding(
            objects_by_symbol[private_symbol],
            private_o,
            entry_symbol="_mlir_ciface_" + selected_spatial.writer.borrowed_symbol,
            kernel_symbol=kernels[private_symbol],
            original_kernel_symbol=selected_spatial.helper_symbol,
            timeout=timeout,
        )
        custody[private_symbol]["connected_spatial"].update(
            descriptor_bridge_audit=bridge_invocation, original_object_binding=binding
        )
    if not unit.symbols:
        return DeviceBuild(
            device=device,
            objects=tuple(objs),
            kernels=kernels,
            built_from=built_from,
            skipped=tuple([*skipped, *unit.skipped]),
            object_dedup=dedup_receipt(),
            build_metrics=build_metrics,
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
    s, _shim_invocation, _shim_s = _observed_build_tool(
        [clang(), *_flags(codegen_target, cflags), "-c", str(shim_c), "-o", str(shim_o)],
        timeout=timeout,
        directory=work,
        stage="device_shim_compilation",
        inputs=(shim_c,),
        output=shim_o,
        dependencies=descriptor_result_dependencies(unit) if descriptor_results else (),
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
            build_metrics=build_metrics,
        )

    for sibling_symbol in sibling_hosts:
        if sibling_symbol not in kernels:
            continue

        def bridge_pin(path):
            return {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }

        custody[sibling_symbol]["sibling_source"]["descriptor_bridge"] = {
            "source": bridge_pin(shim_c),
            "object": bridge_pin(shim_o),
            "compilation": _shim_invocation,
        }

    if sibling_source is not None:
        from merlin.compiler.lower.device.head import HeadBatchSourceRoute
        from merlin.compiler.lower.device.head import require_host as require_head_host

        if type(selected_sibling) is HeadBatchSourceRoute:
            require_head_host(sibling_host, selected_sibling)
            for family in sibling_host.receipt["head_families"]:
                symbol = family["physical_symbol"]
                if symbol not in kernels:
                    raise ValueError("head source lost its actual per-matrix artifact")

                def pin_head(path):
                    return {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "size_bytes": path.stat().st_size,
                    }

                custody[symbol]["head_source"] = {
                    "source_ir_sha256": selected_sibling.source_ir_sha256,
                    "interface_sha256": expected_interfaces[symbol]["sha256"],
                    "matrix_invocations": family["batch_extent"],
                    "descriptor_bridge": {
                        "source": pin_head(shim_c),
                        "object": pin_head(shim_o),
                        "compilation": _shim_invocation,
                    },
                    "bridge_kind": (
                        "whole_logical_batch" if selected_sibling.whole_batch else "stock_logical_batch_slices"
                    ),
                }
                if selected_sibling.whole_batch:
                    custody[symbol]["head_source"].update(
                        kernel_invocations=1, mathematical_products=len(family["members"])
                    )

    for lowering, receipt_sha256 in target_receipts.values():
        revalidate_target(
            pkg,
            lowering,
            selected_target,
            expected_receipt_sha256=receipt_sha256,
        )
    for record in custody.values():
        object_input.recheck_transform_chain(llvm_transform_chain, frozen_transform, record.get("llvm_transform"))
    return DeviceBuild(
        device=device,
        objects=(*objs, shim_o),
        shim_object=shim_o,
        shim_unit=unit,
        shim_invocation=_shim_invocation,
        kernels=kernels,
        custody=custody,
        built_from=built_from,
        skipped=tuple(skipped),
        object_dedup=dedup_receipt(),
        build_metrics=build_metrics,
    )


# ---- device_build_jobs -------------------------------------------------------------------------
# Bounded device jobs under explicit caller-owned execution capabilities.


def build_workers() -> int:
    """Use declared shared capacity, retaining one job for legacy callers by default."""
    from merlin.common.process.cpu_pool import grading_cpus, shared_compute_enabled

    shared = shared_compute_enabled()
    capacity = len(grading_cpus()) if shared else len(os.sched_getaffinity(0))
    raw = os.environ.get("MERLIN_DEVICE_BUILD_JOBS")
    if raw is None:
        return capacity if shared else 1
    if not raw.isdecimal() or int(raw) <= 0:
        raise ValueError("MERLIN_DEVICE_BUILD_JOBS must be a positive integer")
    return min(int(raw), capacity)


class EmissionJobs:
    """Overlap only an explicitly concurrency-capable package execution owner.

    Each job receives its own copied context and private output directory from
    the caller. The owner remains alive until every admitted job has drained.
    Legacy and unscoped calls retain caller-thread execution and order.
    """

    def __init__(self, workers):
        from merlin.development.targetgen.packaging.loading import active_package_executor

        owner = active_package_executor()
        capability = getattr(owner, "concurrent_entrypoints_v1", None)
        self._scope = getattr(owner, "concurrent_entrypoint_scope_v1", None)
        self.workers = workers if callable(capability) and capability() is True else 1
        if self.workers > 1 and not callable(self._scope):
            raise ValueError("concurrent package execution requires its original worker scope")
        self._executor = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def map(self, items, function):
        if self._executor is None:
            for item in items:
                future = Future()
                try:
                    future.set_result(function(item))
                except Exception as error:  # Returned through the ordinary caller's refusal boundary.
                    future.set_exception(error)
                yield item, future
            return
        pending, iterator, exhausted = {}, iter(items), False
        while pending or not exhausted:
            while not exhausted and len(pending) < self.workers:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = self._executor.submit(copy_context().run, self._run, function, item)
                pending[future] = item
            if pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Stable admission order when several jobs complete together.
                for future in tuple(pending):
                    if future in completed:
                        item = pending.pop(future)
                        yield item, future

    def _run(self, function, item):
        with self._scope():
            return function(item)


@dataclass(frozen=True)
class ArtifactBuild:
    raw: Path | None = None
    custody: dict = field(default_factory=dict)
    refusal: str = ""


class ArtifactCompiler:
    """Share only byte-identical raw compilations, including jobs already in flight.

    The caller bounds submissions and drains admitted jobs on failure. No compiler
    package service or context is moved into a worker; callbacks run external tools.
    """

    def __init__(self, workers: int):
        self.workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self._artifacts: dict[str, list[tuple[Path, Future]]] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def submit(self, artifact: Path, digest: str, compile_artifact) -> Future:
        for source, future in self._artifacts.get(digest, ()):
            if future.done() and future.result().refusal:
                continue  # A diagnostic build may retry failed bytes for another symbol.
            if source.read_bytes() == artifact.read_bytes():
                return future
        if self._executor is None:
            future = Future()
            future.set_result(compile_artifact())
        else:
            future = self._executor.submit(copy_context().run, compile_artifact)
        self._artifacts.setdefault(digest, []).append((artifact, future))
        return future

    def has_failed(self) -> bool:
        return any(future.done() and future.result().refusal for rows in self._artifacts.values() for _, future in rows)

    def completed_count(self) -> int:
        return sum(
            future.done() and not future.result().refusal for rows in self._artifacts.values() for _, future in rows
        )
