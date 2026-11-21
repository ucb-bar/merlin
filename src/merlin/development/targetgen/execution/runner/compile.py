"""Runner-owned compile + execute of a *package-produced* lowered LLVM/RoCC MLIR.

The contract splits responsibility: the package emits ``lowered.llvm.mlir`` (a module defining a
kernel function under the entry symbol its contract declares, with the kernel ABI in
``mlir_oot_backend_contract.yaml``); the runner owns the harness (which embeds the deterministic leaf
tensors by name + output buffers and prints ``OUT/METRIC/DONE``), the link, and the oracle
invocation. This path is uniform for Python and C++ packages — the only difference is who produced
the MLIR.
**This module names no target and imports none.** ``target`` is a required argument throughout, and
everything target-specific is resolved through it: the harness ABI from the target's contract
(:mod:`.harness_abi`), and the harness renderer, build recipe and oracle from its backend via
:mod:`merlin.runtime.backends.registry`. What remains here is orchestration — lower, render, link, run.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable as Callable
from collections.abc import Mapping as Mapping
from pathlib import Path
from typing import Any

from merlin.common.process import ownership as _deadline_owner
from merlin.common.process.ownership import ExecutionDeadline, selected_deadline
from merlin.development.targetgen.execution.host.admission import active_selection, host_choices_build
from merlin.development.targetgen.execution.runner.artifact_retesting import (
    active_artifact_retests,
    compare_selected_artifacts,
    retain_selected_artifacts,
)
from merlin.development.targetgen.execution.runner.compile_inputs import (
    _explicit_prepack_inputs,
    _recorded_operands,
    _strict_warm_profile,
)
from merlin.development.targetgen.execution.runner.counter_observations import (
    counter_observations as _counter_observations,
)
from merlin.development.targetgen.execution.runner.memory_decode import (
    observed_memory_decode as _observed_memory_decode,
)
from merlin.development.targetgen.execution.runner.oracle_identity import (
    _selected_run_timeout,
    simulator_provenance,
)
from merlin.development.targetgen.execution.runner.ordinary_inputs import (
    ordinary_integer_harness_inputs as _ordinary_integer_harness_inputs,
)
from merlin.target.contract.build.object_admission import (
    _abi_receipt,
    _module_target_abi,
    _repair_oversized_frame,
)
from merlin.target.contract.build.services import named_object_paths
from merlin.target.contract.harness.readback_policy import accepts_keyword as _accepts_keyword
from merlin.target.contract.harness.readback_policy import compile_sources, renderer_scratch_options
from merlin.target.contract.harness.readout import stage_harness_blobs
from merlin.verification.lean.emission_custody import observe_object, requires_original_object

# ---- compile -----------------------------------------------------------------------------------


def _observed_run(argv, *, workdir, stage, inputs=(), outputs=(), **kwargs):
    from merlin.common.process import cpu_pool as cpu_owner
    from merlin.common.process import invocation_records as invocation_record
    from merlin.common.process.cpu_pool import parse_roster, selected_pool
    from merlin.runtime.views.build_lookup import run_in_build_view

    started = time.monotonic()
    timeout = kwargs.get("timeout")
    if cpu_owner.shared_compute_enabled():
        pool = cpu_owner.shared_compute_pool(timeout_s=timeout)
    else:
        cpus = parse_roster(os.environ.get("MERLIN_BUILD_CPUS", ""))
        slots = os.environ.get("MERLIN_BUILD_SLOT_ROOT", "")
        if cpus and not slots:
            raise ValueError("selected build CPU pool requires an explicit slot root")
        pool = selected_pool(cpus, slot_root=Path(slots) if cpus else None, timeout_s=timeout)
    try:
        with pool:
            if timeout is not None:
                kwargs["timeout"] = max(0, timeout - (time.monotonic() - started))
            selected = run_in_build_view(
                argv,
                directory=Path(workdir).resolve(),
                stage=stage,
                inputs=inputs,
                outputs=outputs,
                **kwargs,
            )
            if selected is not None:
                return selected
            return invocation_record.run(
                argv,
                directory=workdir,
                stage=stage,
                inputs=inputs,
                outputs=outputs,
                dependencies=(
                    *compile_sources(),
                    Path(_deadline_owner.__file__),
                    Path(cpu_owner.__file__),
                ),
                **kwargs,
            )
    except TimeoutError as exc:
        raise subprocess.TimeoutExpired(argv, timeout) from exc


@observe_object
def llvm_mlir_to_object(
    lowered_mlir_text: str,
    workdir: Path,
    *,
    target: str | None = None,
    _build_service=None,
    _host_optimization=None,
    build_timeout_s: int | None = None,
    execution_deadline: ExecutionDeadline | None = None,
) -> Path:
    """Build emitted IR with the original toolchain, ISA and shared harness ABI.
    ``target=None`` preserves the legacy unbound recipe. Optional host choices
    require original semantic/runtime admission and an actual compatible IR stage.
    Finished LLVM cannot select tensor transformations. Original and selected
    products retain distinct identities. Build-only deadlines and measured stack
    preflight are structural gates, never numerical or execution verdicts.
    """
    from merlin.compiler.lower.model import codegen
    from merlin.development.targetgen.packaging.loading import active_package_executor

    deadline = selected_deadline(seconds=build_timeout_s, parent=execution_deadline, build_service=_build_service)

    def remaining() -> float | None:
        if deadline is None:
            return None
        return deadline.remaining("public object-build budget expired")

    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "kernel.host_optimization.json").unlink(missing_ok=True)
    source = workdir / "kernel.llvm.mlir"
    source.write_text(lowered_mlir_text, encoding="utf-8")
    executor = active_package_executor()
    from merlin.compiler.lower.host.graph import HostCompilationOptimization

    host = HostCompilationOptimization.select(
        lowered_mlir_text, workdir, explicit=_host_optimization, finished_llvm_only=_build_service is not None
    )
    host_report, host_lowering_observation = None, None
    if host is not None:
        lowered_mlir_text = host.prepared.selected_mlir

    extra: tuple[str, ...] = ()
    recipe = None
    if _build_service is not None:
        from merlin.target.contract.build.services import BuildOnlyService

        if type(_build_service) is not BuildOnlyService:
            raise ValueError("build-only override requires an exact host service")
        _build_service.verify(target)
        recipe = _build_service.recipe.with_effective_abi()
        extra = (recipe.march(), recipe.mabi())
        # This opt-in service consumes finished target LLVM, not tensors or
        # partially lowered programs. Do not invoke an unrelated model importer
        # and its Python environment merely to translate an LLVM module.
        from xdsl.context import Context
        from xdsl.dialects import builtin, llvm
        from xdsl.parser import Parser

        from merlin.compiler.lower.model import toolchain

        # xDSL's LLVM schema need not model every metadata attribute or
        # property emitted by the selected stock LLVM installation. Retain
        # those bytes and leave semantic verification to mlir-translate below;
        # xDSL is used here only to inspect the complete operation inventory.
        context = Context(allow_unregistered=True)
        context.load_dialect(builtin.Builtin)
        context.load_dialect(llvm.LLVM)
        module = Parser(context, lowered_mlir_text).parse_module()

        def operation_name(op):
            if op.name == "builtin.unregistered":
                return op.op_name.data
            return op.name

        if any(
            operation_name(op) != "builtin.module" and not operation_name(op).startswith("llvm.")
            for op in module.walk()
        ):
            raise ValueError("build-only translation requires a complete LLVM/Builtin module")
        try:
            translated = _observed_run(
                [str(toolchain.mlir_translate()), "--mlir-to-llvmir", str(source), "-o", str(workdir / "kernel.ll")],
                workdir=workdir,
                stage="llvm_translation",
                inputs=(source,),
                outputs=(workdir / "kernel.ll",),
                capture_output=True,
                text=True,
                timeout=remaining(),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("public LLVM translation budget expired") from exc
        if translated.returncode:
            raise _build_service.recipe.error_cls("LLVM translation failed:\n" + translated.stderr[-2000:])
        _build_service.verify(target)
    else:
        from merlin.common.process import invocation_records as invocation_record
        from merlin.compiler.ir.lowering_audit import IrAudit
        from merlin.compiler.lower.native.pipeline import lower_to_llvm_ir

        requested = getattr(executor, "record_stage_inspection", False) is True
        audit = IrAudit(
            workdir,
            enabled=requested,
            producer="merlin.development.targetgen.execution.runner.compile",
            source=__file__,
            sidecars=compile_sources()[1:],
        )

        with (
            audit,
            invocation_record.observe_call(
                workdir,
                stage="llvm_translation",
                function=lower_to_llvm_ir,
                arguments=host.translation_arguments() if host is not None else {"workdir": str(workdir.resolve())},
                inputs=(source, host.selected_source_path) if host is not None else (source,),
                outputs=(workdir / "kernel.ll",),
                dependencies=compile_sources(),
            ) as observed,
        ):
            options = {"audit": audit} if requested else {}
            if host is not None:
                host_options, host_lowering_observation = host.lowering_options()
                options.update(host_options)
            ll = lower_to_llvm_ir(lowered_mlir_text, workdir=workdir, **options)
            (workdir / "kernel.ll").write_text(ll, encoding="utf-8")
            observed.returned()
        if target is not None:
            from merlin.runtime.backends import registry as _backends

            recipe = _backends.harness_build_recipe(target).with_effective_abi()
            extra = (recipe.march(), recipe.mabi())
    llvm_path, object_path = workdir / "kernel.ll", workdir / "kernel.o"
    if host is not None:
        llvm_path, host_report = host.optimize_translated(llvm_path, observation=host_lowering_observation)
    if recipe is None:
        compiled = Path(codegen.compile_ll(llvm_path, object_path, "riscv", extra_flags=extra))
        return host.complete(compiled, llvm_path, report=host_report) if host is not None else compiled

    report_path = object_path.with_suffix(".su")
    receipt_path = workdir / "kernel.stack_frame.json"
    object_path.with_suffix(".measured.su").unlink(missing_ok=True)
    for stale_output in (object_path, report_path, receipt_path, workdir / "kernel.abi.json"):
        stale_output.unlink(missing_ok=True)
    abi = recipe.mabi().partition("=")[2]
    declared = _module_target_abi(llvm_path.read_text(encoding="utf-8"))
    if declared is not None and declared != abi:
        raise recipe.error_cls(
            f"LLVM module target-abi {declared!r} conflicts with selected build recipe ABI {abi!r}; "
            "the candidate IR cannot be rewritten to match the harness"
        )

    # ``clang -fstack-usage`` emits a deterministic sibling of the named object.  Remove a previous
    # report first so a compiler invocation that unexpectedly stops producing the sidecar cannot be
    # admitted using stale evidence from an earlier object in a reused work directory.
    policy = recipe.require_kernel_stack_frame()
    compile_kwargs = {"extra_flags": (*extra, "-fstack-usage")}
    if deadline is not None:
        compile_kwargs["timeout_s"] = remaining()
    compiled = Path(codegen.compile_ll(llvm_path, object_path, "riscv", **compile_kwargs))
    from merlin.target.contract.build.object_admission import (
        StackFramePreflightError,
        measure_entrypoint,
        write_receipt,
    )
    from merlin.target.contract.build.object_admission import _sha256 as _stack_sha

    try:
        if compiled != object_path or compiled.is_symlink() or not compiled.is_file():
            raise StackFramePreflightError(f"compiler produced no regular object at the requested path {object_path}")
        measurement = measure_entrypoint(
            report_path, llvm_path=llvm_path, entry_symbol=policy.entry_symbol, max_static_bytes=policy.max_static_bytes
        )
    except StackFramePreflightError as exc:
        # Select generic stack repair only after an observed frame failure.
        # Rebuild and remeasure its actual object; unavailable repairs retain
        # the original refusal. Already fitting objects remain unchanged.
        repaired = _repair_oversized_frame(
            llvm_path,
            object_path,
            recipe=recipe,
            policy=policy,
            extra=extra,
            remaining=remaining if deadline is not None else None,
        )
        if repaired is None:
            write_receipt(
                receipt_path,
                status="rejected",
                llvm_path=llvm_path,
                object_path=compiled,
                report_path=report_path,
                entry_symbol=policy.entry_symbol,
                max_static_bytes=policy.max_static_bytes,
                measurement=exc.measurement,
                diagnostic=str(exc),
            )
            raise recipe.error_cls("kernel stack-frame preflight failed: " + str(exc)) from exc
        compiled, measurement, arena_llvm, arena_su, arena_report = repaired
        write_receipt(
            receipt_path,
            status="passed",
            llvm_path=arena_llvm,
            object_path=compiled,
            report_path=arena_su,
            entry_symbol=policy.entry_symbol,
            max_static_bytes=policy.max_static_bytes,
            measurement=measurement,
            repair={
                "transform": "stack_arena_bind",
                "frame_bytes_before": (exc.measurement.frame_bytes if exc.measurement is not None else None),
                "diagnostic_before": str(exc),
                "emitted_llvm_ir_sha256": _stack_sha(llvm_path),
                **arena_report,
            },
        )
        _abi_receipt(workdir, compiled, abi)
        return host.complete(compiled, arena_llvm, report=host_report) if host is not None else compiled
    write_receipt(
        receipt_path,
        status="passed",
        llvm_path=llvm_path,
        object_path=compiled,
        report_path=report_path,
        entry_symbol=policy.entry_symbol,
        max_static_bytes=policy.max_static_bytes,
        measurement=measurement,
    )
    _abi_receipt(workdir, compiled, abi)
    return host.complete(compiled, llvm_path, report=host_report) if host is not None else compiled


def link_elf(
    cb: dict[str, Any],
    obj: Path,
    workdir: Path,
    *,
    target: str,
    inputs: dict | None = None,
    prepack_authorizations=None,
    _compact_caller=None,
    _build_service=None,
    _compile_only_linkage=None,
    build_timeout_s=None,
    execution_deadline: ExecutionDeadline | None = None,
    warm_profile=None,
    readback_policy=None,
) -> Path:
    """Build the runner-owned harness from ``cb`` and link it with the package object -> ELF.

    Orchestration only: the harness TEXT comes from ``target``'s declared harness ABI and the BUILD
    from its declared recipe, both resolved through the backend registry. This module names no target
    and imports no target's module — ``target`` is a required argument precisely so no default can
    reintroduce one.

    ``prepack_authorizations`` is a trusted-host-only capability. It is not read
    from the command buffer, and requires explicitly supplied immutable operands.
    A typed compile-only reference with an explicit pure build service retains
    the ordinary kernel without tensor data, readback or execution authority.
    """
    warm_profile = _strict_warm_profile(warm_profile, cb)
    deadline = selected_deadline(seconds=build_timeout_s, parent=execution_deadline, build_service=_build_service)

    def remaining():
        if deadline is None:
            return None
        return deadline.remaining("public link-build budget expired")

    def deadline_options():
        return {} if deadline is None else {"timeout": remaining()}

    if _compile_only_linkage is not None:
        from merlin.target.contract.build.services import CompileOnlyLinkage

        if (
            type(_compile_only_linkage) is not CompileOnlyLinkage
            or _build_service is None
            or any(
                value is not None
                for value in (inputs, prepack_authorizations, _compact_caller, warm_profile, readback_policy)
            )
        ):
            raise ValueError("compile-only linking needs explicit pure build support and no value/execution authority")
        _compile_only_linkage.verify(cb)
    from merlin.target.contract.harness.readback_policy import BUILD_RECEIPT, selected

    readback_policy = selected(readback_policy)
    if _build_service is None:
        from merlin.runtime.backends import registry as _backends

        if readback_policy is None:
            readback_policy = selected(None, backend=_backends.get_backend(target))
    receipt_path = workdir / BUILD_RECEIPT
    if readback_policy is not None:
        # A failed rebuild may never leave a previous complete policy receipt.
        receipt_path.write_text('{"status":"incomplete"}\n', encoding="utf-8")
    elif receipt_path.exists():
        # A legacy rebuild must not inherit an opt-in receipt from a reused workdir.
        receipt_path.unlink()
    if _build_service is not None:
        from merlin.target.contract.build.services import BuildOnlyService

        if (
            type(_build_service) is not BuildOnlyService
            or _compact_caller is not None
            or prepack_authorizations is not None
        ):
            raise ValueError("build-only service cannot mix caller authority paths")
        _build_service.verify(target)
        recipe = _build_service.recipe.with_effective_abi()
        _render = _build_service.render
    else:
        recipe = _backends.harness_build_recipe(target).with_effective_abi()
        _render = _backends.harness_renderer(target)
    readback_inputs = None
    if readback_policy is not None:
        from merlin.target.contract.harness.readback_policy import selected_build_inputs

        if _compact_caller is not None:
            raise ValueError("full-value readback does not support the prepared compact caller")
        import inspect

        if "readback_policy" not in inspect.signature(_render).parameters:
            raise NotImplementedError("selected backend cannot render the explicit full-value policy")
        readback_inputs = selected_build_inputs(target, recipe, _build_service, policy=readback_policy)
    abi_receipt = workdir / "kernel.abi.json"
    if abi_receipt.exists():
        try:
            record = json.loads(abi_receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise recipe.error_cls("kernel ABI receipt is unreadable") from exc
        if (
            record.get("schema") != "merlin_kernel_abi_v1"
            or record.get("abi") != recipe.mabi().partition("=")[2]
            or record.get("object_sha256") != hashlib.sha256(Path(obj).read_bytes()).hexdigest()
        ):
            raise recipe.error_cls("kernel ABI receipt does not match selected harness ABI and object")
    # An opt-in renderer may return large constant operands as exact bytes.
    # The target still decides the tensor layout; the runner owns sidecar
    # filenames, assembly and linking for every target in the same way.
    blob_payloads: dict = {}
    blob_kwargs = {"blobs": blob_payloads} if _accepts_keyword(_render, "blobs") else {}
    policy_kwargs = {"readback_policy": readback_policy} if readback_policy is not None else {}
    policy_kwargs.update(
        renderer_scratch_options(
            cb,
            _render,
            alternate_caller=(
                _build_service is not None or _compact_caller is not None or prepack_authorizations is not None
            ),
        )
    )
    # ``inputs`` INJECTS the caller's real operands into the device harness. A renderer written before
    # this parameter existed still works and still materializes from names -- but silently doing that
    # while the reference and simulator use injected data produces a guaranteed three-way mismatch that
    # reads as a functional failure of the TARGET, so an injecting caller is told instead.
    compact_object_sha = None
    if _compile_only_linkage is not None:
        harness = _compile_only_linkage.render(cb)
    elif _compact_caller is not None:
        # Only compile_lowered_to_elf's trusted preparation path supplies this
        # object. No serialized candidate ABI facts or fallback inputs enter it.
        if inputs is not None or prepack_authorizations is not None:
            raise ValueError("prepared compact caller cannot be combined with other input sources")
        compact_object_sha = hashlib.sha256(Path(obj).read_bytes()).hexdigest()
        kwargs = {"target": target, "compact_caller": _compact_caller}
        if warm_profile is not None:
            if not _accepts_keyword(_render, "warm_profile"):
                raise NotImplementedError("backend compact harness cannot consume a strict warm profile")
            kwargs["warm_profile"] = warm_profile
        harness = _render(cb, **kwargs, **blob_kwargs, **policy_kwargs)
    else:
        _explicit_prepack_inputs(inputs, prepack_authorizations)
    # Preserve an explicitly empty source-input roster; only absent input data
    # may use the historical recorded-operand fallback.
    if _compact_caller is None and prepack_authorizations is None and _build_service is None and inputs is None:
        inputs = _recorded_operands(cb)
        if inputs is None and getattr(_backends.get_backend(target), "REQUIRES_EXPLICIT_LOGICAL_INPUTS", False):
            budget_provider = getattr(_backends.get_backend(target), "logical_storage_budget_bytes", None)
            inputs = _ordinary_integer_harness_inputs(
                cb,
                max_storage_bytes=budget_provider() if callable(budget_provider) else 16 << 20,
                stream_inputs=bool(blob_kwargs),
            )
    if _compile_only_linkage is not None:
        pass
    elif _compact_caller is not None:
        pass
    elif inputs is not None or prepack_authorizations is not None:
        if not _accepts_keyword(_render, "inputs"):
            raise NotImplementedError(
                f"backend for target {target!r} declares a render_harness that cannot take `inputs`, so "
                f"the device would compute on name-materialized operands while the reference and the "
                f"simulator use the injected ones. Add an `inputs` parameter to its render_harness."
            )
        if prepack_authorizations is not None:
            if not _accepts_keyword(_render, "prepack_authorizations"):
                raise NotImplementedError("backend harness cannot consume host prepack authorization")
            kwargs = {"target": target, "inputs": inputs, "prepack_authorizations": prepack_authorizations}
        else:
            kwargs = {"target": target, "inputs": inputs}
        if warm_profile is not None:
            if not _accepts_keyword(_render, "warm_profile"):
                raise NotImplementedError("backend harness cannot consume a strict warm profile")
            kwargs["warm_profile"] = warm_profile
        harness = _render(cb, **kwargs, **blob_kwargs, **policy_kwargs)
    else:
        kwargs = {"target": target}
        if warm_profile is not None:
            if not _accepts_keyword(_render, "warm_profile"):
                raise NotImplementedError("backend harness cannot consume a strict warm profile")
            kwargs["warm_profile"] = warm_profile
        harness = _render(cb, **kwargs, **blob_kwargs, **policy_kwargs)
    (workdir / "harness.c").write_text(harness, encoding="utf-8")
    codec_inputs = ()
    if readback_policy is not None:
        from merlin.target.contract.harness.readback_policy import _codec_names, stage_codec_header

        stage_codec_header(workdir, policy=readback_policy)
        codec_inputs = tuple(workdir / name for name in _codec_names(readback_policy))
    blob_sources = stage_harness_blobs(workdir, blob_payloads)
    # Linker load address DERIVED from the RTL memory map (platform DRAM base), reusing the curated
    # script's proven section layout but replacing its BAKED origin — so the base is a HW fact, not a
    # hardcoded literal in a vendored file.
    from merlin.target.simulators.spike_runtime import derived_link_script

    link_ld = derived_link_script(recipe.load_address, recipe.link_script, Path(workdir))
    from merlin.development.targetgen.grading.elf_lanes import PACKAGE_ELF_NAME

    elf = workdir / PACKAGE_ELF_NAME
    # REPRODUCIBLE BUILD, in two phases. A single compile+link invocation lets the driver name its
    # intermediate objects `ccXXXXXX.o`, and those random names are recorded in the ELF as STT_FILE
    # symbols -- so two builds of byte-identical sources differ (measured: 6 bytes) while producing
    # identical cycles. That defeats content-addressed reuse of a measurement for no reason. Naming
    # each object explicitly makes the artifact a function of its inputs again.
    # ORDER IS PRESERVED EXACTLY. The single-step command linked
    # `harness.c, <kernel obj>, *support_sources`; object order decides placement within a section,
    # so reordering could move code and change cycles. This build changes how each object is NAMED,
    # nothing about which objects are linked or in what order.
    objects: list[Path] = []
    # ... except a recipe's ``link_first`` support sources, which lead (see HarnessBuildRecipe).
    linked = [workdir / "harness.c", obj, *blob_sources]
    ordered = getattr(recipe, "ordered_link_sources", None)  # a duck-typed recipe keeps declared order
    sources = ordered(linked) if callable(ordered) else [*linked, *recipe.support_sources]
    for source, unit in zip(sources, named_object_paths(sources, workdir), strict=True):
        source = Path(source)
        # Assembly counts: the driver assembles a .S through the same temp-named intermediate that
        # a .c goes through, so leaving crt.S to the link step reintroduced the very STT_FILE symbol
        # this two-phase build exists to remove.
        if source.suffix not in (".c", ".S", ".s"):
            objects.append(source)
            continue
        # `.incbin` names the staged payload by basename; assemble only these
        # generated stubs from their own directory, leaving existing build
        # command lines and support-source compilation unchanged.
        if source in blob_sources:
            command = recipe.compile_command(source=source.resolve(), output=unit.resolve())
            compile_cwd = workdir.resolve()
        else:
            command = recipe.compile_command(source=source, output=unit)
            compile_cwd = None
        fresh_inputs = (source,)
        if source in blob_sources:
            fresh_inputs = (source, source.with_suffix(".bin"))
        elif source == workdir / "harness.c":
            fresh_inputs = (source, *codec_inputs)
        step = _observed_run(
            command,
            workdir=workdir,
            stage="harness_object",
            inputs=fresh_inputs,
            outputs=(unit,),
            cwd=compile_cwd,
            capture_output=True,
            text=True,
            **deadline_options(),
        )
        if step.returncode != 0:
            raise recipe.error_cls(f"compile of {source.name} failed:\n{step.stderr[-2000:]}")
        objects.append(unit)
    cmd = recipe.link_command(objects=objects, output=elf, link_script=link_ld)
    proc = _observed_run(
        cmd,
        workdir=workdir,
        stage="elf",
        inputs=(*objects, link_ld),
        outputs=(elf,),
        capture_output=True,
        text=True,
        **deadline_options(),
    )
    if proc.returncode != 0:
        # A FREESTANDING LINK IS ALLOWED ONE DECLARED RETRY. newlib's libm references a handful of
        # hosted-libc symbols that a bare-metal image does not provide: a whole-model harness that
        # calls `pow` pulls in `__errno`, and the link fails with the model already compiled.
        # `bundle_harness` supplies exactly those shims, each carrying the argument for why it is
        # faithful -- `__errno` is WRITTEN by libm on a domain error and never read by generated
        # kernel code, so storage is the whole requirement and no computed value changes.
        #
        # Only symbols the LINKER named are supplied, parsed from its own message rather than
        # predicted from the object: `pow`, `sin` and `memcpy` are all referenced and all resolve,
        # so the question is what the environment failed to supply, not what the code mentions. A
        # symbol with no declared shim re-raises with the original error, so an unexplained missing
        # symbol is still a build failure and not a silently stubbed one.
        from merlin.development.targetgen.packaging.bundle_harness import (
            BundleHarnessError,
            render_freestanding_support,
            unresolved_symbols,
        )

        missing = unresolved_symbols(proc.stderr)
        support = ""
        if missing:
            try:
                support = render_freestanding_support(missing)
            except BundleHarnessError:
                support = ""
        if not support:
            raise recipe.error_cls(f"link failed:\n{proc.stderr[-2000:]}")
        shim_c = workdir / "freestanding_support.c"
        shim_c.write_text(support, encoding="utf-8")
        shim_o = workdir / "freestanding_support.o"
        step = _observed_run(
            recipe.compile_command(source=shim_c, output=shim_o),
            workdir=workdir,
            stage="freestanding_support_object",
            inputs=(shim_c,),
            outputs=(shim_o,),
            capture_output=True,
            text=True,
            **deadline_options(),
        )
        if step.returncode != 0:
            raise recipe.error_cls(f"freestanding support for {list(missing)} did not compile:\n{step.stderr[-2000:]}")
        # Appended, so the order of every pre-existing object -- which decides placement within a
        # section and therefore cycles -- is unchanged.
        retry = recipe.link_command(objects=[*objects, shim_o], output=elf, link_script=link_ld)
        proc = _observed_run(
            retry,
            workdir=workdir,
            stage="elf",
            inputs=(*objects, shim_o, link_ld),
            outputs=(elf,),
            capture_output=True,
            text=True,
            **deadline_options(),
        )
        if proc.returncode != 0:
            raise recipe.error_cls(f"link failed after supplying freestanding {list(missing)}:\n{proc.stderr[-2000:]}")
    if _compact_caller is not None:
        verify = getattr(_backends.get_backend(target), "verify_compact_caller_link", None)
        if verify is None:
            raise NotImplementedError("target cannot verify linked compact caller allocations")
        verify(
            cb,
            _compact_caller,
            object_path=obj,
            elf_path=elf,
            workdir=workdir,
            expected_object_sha256=compact_object_sha,
        )
    if _build_service is not None:
        _build_service.verify(target)
    if readback_policy is not None:
        from merlin.target.contract.harness.readback_policy import build_receipt, selected_build_inputs

        if readback_inputs != selected_build_inputs(target, recipe, _build_service, policy=readback_policy):
            raise ValueError("selected full-value renderer, codec, or recipe changed during link")
        recipe_record, source_pins = readback_inputs
        receipt = build_receipt(
            policy=readback_policy,
            cb=cb,
            target=target,
            recipe_record=recipe_record,
            source_pins=source_pins,
            object_path=obj,
            harness_path=workdir / "harness.c",
            elf_path=elf,
        )
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return elf
