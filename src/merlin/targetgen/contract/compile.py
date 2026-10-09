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
:mod:`merlin.runtime.backends.base`. What remains here is orchestration — lower, render, link, run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from merlin.common import execution_deadline as _deadline_owner
from merlin.common.execution_deadline import ExecutionDeadline, selected_deadline

from .build_recipe import named_object_paths
from .harness_blobs import stage_harness_blobs


def _observed_run(argv, *, workdir, stage, inputs=(), outputs=(), **kwargs):
    from merlin.common import invocation_record

    return invocation_record.run(
        argv,
        directory=workdir,
        stage=stage,
        inputs=inputs,
        outputs=outputs,
        dependencies=(Path(__file__), Path(_deadline_owner.__file__)),
        **kwargs,
    )


def _observed_memory_decode(reader, console, *, cb, elf, workdir, policy, dependencies=()):
    """Retain the actual selected decoder and its declared private products.

    A callback return or payload pin supplies attribution only. Original packet
    membership, observer integrity and execution semantics remain independent.
    """
    from merlin.common import invocation_record

    from .readback_policy import BUILD_RECEIPT, require_memory_value_roster

    work = Path(workdir).resolve()
    product = work / "readback_decode.json"
    if product.exists() or product.is_symlink():
        raise ValueError("memory decoder product already exists in its private execution owner")
    with invocation_record.observe_call(
        work,
        stage="coherent_memory_decode",
        function=reader.decode,
        arguments={"readback_policy": policy.record()},
        inputs=(Path(elf), work / "oracle_console.log", work / BUILD_RECEIPT),
        outputs=(product,),
        dependencies=(Path(__file__), *dependencies),
    ) as observed:
        outputs, evidence = reader.decode(console)
        if type(outputs) is not dict or type(evidence) is not dict or evidence.get("status") != "complete":
            raise ValueError("memory output reader returned no completed full-value admission")
        require_memory_value_roster(cb, outputs)
        declared_payload = evidence.get("payload")
        if declared_payload is not None:
            if type(declared_payload) is not dict or set(declared_payload) != {"path", "sha256"}:
                raise ValueError("memory decoder declared an unsupported payload product")
            payload = Path(declared_payload["path"])
            if (
                not payload.is_absolute()
                or payload.resolve() != payload
                or any(path.is_symlink() for path in (payload, *payload.parents))
                or not payload.is_relative_to(work)
                or not payload.is_file()
            ):
                raise ValueError("memory decoder payload escapes its private execution owner")
            if hashlib.sha256(payload.read_bytes()).hexdigest() != declared_payload["sha256"]:
                raise ValueError("memory decoder payload differs from its actual declared bytes")
            observed.outputs = (*observed.outputs, payload)
        encoded = (
            json.dumps(
                {
                    "outputs": outputs,
                    "memory_evidence": evidence,
                    "scope": "actual decoder return and declared products only; observer/runtime/effects UNKNOWN",
                },
                sort_keys=True,
            )
            + "\n"
        )
        # The selected observer runs before publication. Exclusive creation also
        # refuses a file or symlink it creates after the earlier metadata check.
        with product.open("x", encoding="utf-8") as output:
            output.write(encoded)
        observed.returned(stdout=encoded)
    invocation_record.verify(observed.path)
    return (
        outputs,
        evidence,
        {
            "record": {"path": str(observed.path), "sha256": hashlib.sha256(observed.path.read_bytes()).hexdigest()},
            "product": {"path": str(product), "sha256": hashlib.sha256(product.read_bytes()).hexdigest()},
            "scope": "source-bound decoder invocation only; no observer, stage, hardware or timer authority",
        },
    )


def _module_target_abi(llvm_text: str) -> str | None:
    """Read the LLVM module's target-abi flag, without treating IR as ABI authority."""
    references: list[str] = []
    definitions: dict[str, str] = {}
    for line in llvm_text.splitlines():
        line = line.strip()
        if line.startswith("!llvm.module.flags = !{"):
            if references:
                raise ValueError("LLVM module has duplicate module-flag tables")
            payload = line.partition("!{")[2].removesuffix("}")
            references = [item.strip() for item in payload.split(",")]
        elif line.startswith("!") and " = !{" in line:
            name, _, payload = line.partition(" = !{")
            definitions[name] = payload.removesuffix("}")
    found: list[str] = []
    for reference in references:
        operands = [part.strip() for part in definitions.get(reference, "").split(",")]
        if len(operands) > 1 and operands[1] == '!"target-abi"':
            if len(operands) != 3 or not operands[2].startswith('!"') or not operands[2].endswith('"'):
                raise ValueError("LLVM module has malformed target-abi flag")
            found.append(operands[2][2:-1])
    if len(found) > 1:
        raise ValueError("LLVM module has ambiguous target-abi flags")
    return found[0] if found else None


def _abi_receipt(workdir: Path, obj: Path, abi: str) -> None:
    """Bind the object to the ABI selected for its harness in a paired build."""
    (workdir / "kernel.abi.json").write_text(
        json.dumps(
            {
                "schema": "merlin_kernel_abi_v1",
                "abi": abi,
                "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def llvm_mlir_to_object(
    lowered_mlir_text: str,
    workdir: Path,
    *,
    target: str | None = None,
    _build_service=None,
    build_timeout_s: int | None = None,
    execution_deadline: ExecutionDeadline | None = None,
) -> Path:
    """Lower package-emitted llvm-dialect MLIR to an rv64 object (.o) for ``target``'s own ISA.

    THE MARCH IS THE TARGET'S, NOT A DEFAULT. This object and the runner-owned harness are linked into
    one ELF and executed on one core, so they have to be built for the same instruction set. The
    default here is the vector-capable ``rv64gcv``, while a systolic target's harness recipe declares
    ``rv64gc`` -- and for years that disagreement was invisible, because every kernel on such a target
    was inline-asm accelerator instructions with nothing for the auto-vectorizer to take. The first
    kernel that gives it something (a scalar float program placed on the host lane) was compiled with
    ``vsetivli``/``vle32.v``/``vfadd.vv`` for a core with no vector unit, trapped on the first one, and
    was reported as the submission's kernel faulting at runtime.

    ``target=None`` keeps the previous default, for callers with no target in hand.
    A pure build-only caller may additionally bound translation and every object
    compiler subprocess with one declining diagnostic wall budget. This does
    not turn parsing or static stack inspection into a numerical verdict.
    """
    from merlin.llvmlower import codegen

    deadline = selected_deadline(seconds=build_timeout_s, parent=execution_deadline, build_service=_build_service)

    def remaining() -> float | None:
        if deadline is None:
            return None
        return deadline.remaining("public object-build budget expired")

    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "kernel.llvm.mlir"
    source.write_text(lowered_mlir_text, encoding="utf-8")
    extra: tuple[str, ...] = ()
    recipe = None
    if _build_service is not None:
        from .build_service import BuildOnlyService

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

        from merlin.llvmlower import toolchain

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
        from merlin.common import invocation_record
        from merlin.common.ir_audit import IrAudit
        from merlin.llvmlower.pipeline import lower_to_llvm_ir
        from merlin.targetgen.package_runtime import active_package_executor

        executor = active_package_executor()
        requested = getattr(executor, "record_stage_inspection", False) is True
        audit = IrAudit(workdir, enabled=requested, producer="merlin.targetgen.contract.compile", source=__file__)

        with (
            audit,
            invocation_record.observe_call(
                workdir,
                stage="llvm_translation",
                function=lower_to_llvm_ir,
                arguments={"workdir": str(workdir.resolve())},
                inputs=(source,),
                outputs=(workdir / "kernel.ll",),
                dependencies=(Path(__file__),),
            ) as observed,
        ):
            options = {"audit": audit} if requested else {}
            ll = lower_to_llvm_ir(lowered_mlir_text, workdir=workdir, **options)
            (workdir / "kernel.ll").write_text(ll, encoding="utf-8")
            observed.returned()
        if target is not None:
            from merlin.runtime.backends import base as _backends

            recipe = _backends.harness_build_recipe(target).with_effective_abi()
            extra = (recipe.march(), recipe.mabi())
    llvm_path, object_path = workdir / "kernel.ll", workdir / "kernel.o"
    if recipe is None:
        return Path(codegen.compile_ll(llvm_path, object_path, "riscv", extra_flags=extra))

    report_path = object_path.with_suffix(".su")
    receipt_path = workdir / "kernel.stack_frame.json"
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
    from .stack_usage import StackFramePreflightError, measure_entrypoint, write_receipt
    from .stack_usage import _sha256 as _stack_sha

    try:
        if compiled != object_path or compiled.is_symlink() or not compiled.is_file():
            raise StackFramePreflightError(f"compiler produced no regular object at the requested path {object_path}")
        measurement = measure_entrypoint(
            report_path, llvm_path=llvm_path, entry_symbol=policy.entry_symbol, max_static_bytes=policy.max_static_bytes
        )
    except StackFramePreflightError as exc:
        # REPAIR ON A PROVEN FAILURE, never pre-emptively. The host lane hoists one `alloca` per
        # intermediate into the entry frame with no reuse, so the frame scales with the model:
        # 816 bytes on ResNet-50 and 99,897,984 bytes on SmolVLA's flow_denoise against a
        # 65,536-byte budget. Seating that storage in one `.bss` arena fits it (measured: 496).
        #
        # It runs ONLY after the emitted frame has been measured over budget, so a build that
        # already fits is byte-identical to before -- which is what keeps the one bundle known to
        # have run correctly on hardware a valid acceptance test for this path.
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
        return compiled
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
    return compiled


def _repair_oversized_frame(
    llvm_path, object_path, *, recipe, policy, extra, remaining: Callable[[], float | None] | None = None
):
    """Seat the entry frame's static temporaries in one arena and RE-MEASURE.

    Returns ``(object, measurement, llvm_path, report_path, report)``, or ``None`` when the repair
    is unavailable or did not actually fit. The measurement is taken on the REBUILT object; a repair
    that is believed rather than measured is how an over-budget frame reaches a device.

    On ``None`` the caller's original refusal stands, and it is deliberately the diagnostic the
    caller sees: the emitted program is what was asked about, and a second diagnostic describing a
    repaired variant would name a program nobody requested.
    """
    from merlin.llvmlower import codegen as _codegen
    from merlin.llvmlower.stack_arena import StackArenaError, bind_stack_arena

    from .stack_usage import StackFramePreflightError, measure_entrypoint

    llvm_path, object_path = Path(llvm_path), Path(object_path)
    try:
        rewritten, report = bind_stack_arena(llvm_path.read_text(encoding="utf-8"), entry_symbol=policy.entry_symbol)
    except StackArenaError:
        return None
    if not report.n_bound:
        return None
    arena_llvm = llvm_path.with_suffix(".arena.ll")
    arena_object = object_path.with_suffix(".arena.o")
    arena_su = arena_object.with_suffix(".su")
    arena_llvm.write_text(rewritten, encoding="utf-8")
    for stale in (arena_object, arena_su):
        stale.unlink(missing_ok=True)
    compile_kwargs = {"extra_flags": (*extra, "-fstack-usage")}
    if remaining is not None:
        compile_kwargs["timeout_s"] = remaining()
    rebuilt = Path(_codegen.compile_ll(arena_llvm, arena_object, "riscv", **compile_kwargs))
    try:
        measurement = measure_entrypoint(
            arena_su, llvm_path=arena_llvm, entry_symbol=policy.entry_symbol, max_static_bytes=policy.max_static_bytes
        )
    except StackFramePreflightError:
        return None
    # The arena's bytes are NOT free: they are `.bss` in the same image, and a large one narrows the
    # PC-relative window a medany build has to work in. Reported through the receipt, never absorbed.
    return rebuilt, measurement, arena_llvm, arena_su, report.to_dict()


def _recorded_operands(cb: dict[str, Any]) -> dict[str, list] | None:
    """The operands an independent-float or WHOLE-PROGRAM buffer must run on, or ``None``.

    A capsule graded under a float policy cannot have had its answer recomputed on the integer engine,
    so its golden is the INDEPENDENT one — computed off-device, on the operands the runner attached to
    the buffer as ``canonical_inputs``. The device is only answerable against that golden if it runs
    on those same operands; nothing consumed them on this path, so the harness materialized each leaf
    from its NAME and the device computed the right function of the wrong inputs — a guaranteed
    mismatch, reported as a functional failure of the submission.

    THE FLOAT CONDITION remains load-bearing for ordinary buffers. A per-op integer buffer is graded
    against a golden recomputed from deterministic name-materialized fill, and may carry unrelated
    historical operands (``GS0_matmul_spec`` is the guard). A WHOLE-PROGRAM buffer is different: the
    runner has already replaced/re-keyed ``canonical_inputs`` with the exact semantic stimulus because
    its compiler may rename leaves positionally. Discarding that table rematerializes from ``arg0``
    instead of ``A0`` and guarantees a false mismatch. The explicit ABI discriminator separates that
    exception without changing legacy per-op integer behavior.
    """
    from merlin.runtime.backends import base as _backends
    from merlin.runtime.commandbuffer import declared_output_dtypes

    recorded = cb.get("canonical_inputs") or {}
    tensors = cb.get("tensors") or {}
    if not recorded:
        return None
    dtypes = declared_output_dtypes(cb)
    outputs = [n for n, s in tensors.items() if (s or {}).get("role") == "output"]
    whole_program = (cb.get("kernel_abi") or {}).get("kind") == "whole_program"
    if not whole_program and (not outputs or not all(_backends.float_format_of(dtypes.get(n, "")) for n in outputs)):
        return None
    return {
        name: spec["values"]
        for name, spec in recorded.items()
        if isinstance(spec, dict) and spec.get("values") is not None and name in tensors
    } or None


def _explicit_prepack_inputs(inputs, authorizations) -> None:
    """Never satisfy a host immutable-payload grant from candidate recorded operands."""
    if authorizations is None:
        return
    if not isinstance(authorizations, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("host prepack authorization requires explicit logical inputs")
    from merlin.runtime.prepack_authority import HostPrepackAuthorization

    if any(type(grant) is not HostPrepackAuthorization for grant in authorizations.values()):
        raise ValueError("prepack requires an exact host authorization object")
    if set(authorizations) - set(inputs):
        raise ValueError("authorized prepack input is missing from explicit logical inputs")


def _strict_warm_profile(profile, cb=None):
    """Validate the explicit final-profile capability without naming a target."""
    if profile is None:
        return None
    from merlin.perf.warm_profile_harness import require_strict_final_warm_profile

    validated = require_strict_final_warm_profile(profile)
    if cb is not None and (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
        raise ValueError("strict final warm profiling requires an explicit whole-program kernel ABI")
    return validated


def _accepts_keyword(callable_object, name: str) -> bool:
    """Whether a renderer explicitly accepts ``name`` or a generic keyword set."""
    import inspect

    parameters = inspect.signature(callable_object).parameters
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


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
        from .compile_only import CompileOnlyLinkage

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
    from .readback_policy import BUILD_RECEIPT, selected

    readback_policy = selected(readback_policy)
    receipt_path = workdir / BUILD_RECEIPT
    if readback_policy is not None:
        # A failed rebuild may never leave a previous complete policy receipt.
        receipt_path.write_text('{"status":"incomplete"}\n', encoding="utf-8")
    elif receipt_path.exists():
        # A legacy rebuild must not inherit an opt-in receipt from a reused workdir.
        receipt_path.unlink()
    if _build_service is not None:
        from .build_service import BuildOnlyService

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
        from merlin.runtime.backends import base as _backends

        recipe = _backends.harness_build_recipe(target).with_effective_abi()
        _render = _backends.harness_renderer(target)
    readback_inputs = None
    if readback_policy is not None:
        from .readback_policy import selected_build_inputs

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
    if readback_policy is not None:
        from .readback_policy import stage_codec_header

        stage_codec_header(workdir, policy=readback_policy)
    blob_sources = stage_harness_blobs(workdir, blob_payloads)
    # Linker load address DERIVED from the RTL memory map (platform DRAM base), reusing the curated
    # script's proven section layout but replacing its BAKED origin — so the base is a HW fact, not a
    # hardcoded literal in a vendored file.
    from ..runtime_build import derived_link_script

    link_ld = derived_link_script(recipe.load_address, recipe.link_script, Path(workdir))
    from ..elf_lanes import PACKAGE_ELF_NAME

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
    sources = [workdir / "harness.c", obj, *blob_sources, *recipe.support_sources]
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
        step = _observed_run(
            command,
            workdir=workdir,
            stage="harness_object",
            inputs=(source,),
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
        from merlin.targetgen.bundle_harness import BundleHarnessError, render_freestanding_support, unresolved_symbols

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
        from .readback_policy import build_receipt, selected_build_inputs

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


def compile_lowered_to_elf(
    cb: dict[str, Any],
    lowered_mlir_text: str,
    workdir: str | Path | None = None,
    *,
    target: str,
    inputs: dict | None = None,
    prepack_authorizations=None,
    compact_contract=None,
    logical_payloads=None,
    compact_storage_limit_bytes: int = 64 * 1024,
    _build_service=None,
    warm_profile=None,
    readback_policy=None,
    execution_deadline: ExecutionDeadline | None = None,
) -> Path:
    """Full package-lowered-MLIR -> rv64 ELF (object + runner harness + link).

    The result is a pure function of its inputs, so an unchanged capsule is not recompiled: see
    :mod:`merlin.targetgen.build_cache` for the key, and for why reusing a BUILD carries none of the
    risk of reusing a verdict. A restored build reproduces the whole generated directory, not only the
    executable, because the agent reads what is in it. Every failure to establish a key -- an
    unresolvable recipe, a toolchain that will not answer, an unreadable source -- falls through to an
    ordinary build, which is also what ``MERLIN_ELF_BUILD_CACHE=0`` does.

    Measured before this existed: the screen tier spent 4.06 s building per capsule against 0.153 s
    simulating, and re-paid it on every capsule of every grade.

    The optional compact route requires both ``compact_contract`` and exact
    ``logical_payloads`` bytes, forbids legacy/fallback ``inputs``, and bypasses
    this cache entirely. Its target-owned preparation derives ABI facts and
    validates storage/prepack authority; the linker checks actual arena symbols.
    ``compact_storage_limit_bytes`` bounds host format setup, not hardware
    capacity. A successful build is not a numerical or execution verdict.

    ``warm_profile`` is an explicit, cache-free final measurement build.  It
    must be the strict shared contract (one warm invocation, one measured
    invocation, cycles only); the target renderer supplies launch/completion
    hooks and keeps result readback outside the cycle window.
    """
    deadline = selected_deadline(seconds=None, parent=execution_deadline, build_service=_build_service)
    warm_profile = _strict_warm_profile(warm_profile, cb)
    from .readback_policy import selected

    readback_policy = selected(readback_policy)
    if _build_service is not None:
        from .build_service import BuildOnlyService

        if (
            type(_build_service) is not BuildOnlyService
            or inputs is None
            or prepack_authorizations is not None
            or compact_contract is not None
            or logical_payloads is not None
        ):
            raise ValueError("build-only service requires explicit inputs and no alternate caller authority")
        _build_service.verify(target)
        work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_build_only_"))
        budget_options = {"execution_deadline": deadline} if deadline is not None else {}
        obj = llvm_mlir_to_object(
            lowered_mlir_text, work, target=target, _build_service=_build_service, **budget_options
        )
        if deadline is not None:
            deadline.remaining()
        kwargs = {"target": target, "inputs": inputs, "_build_service": _build_service}
        kwargs.update(budget_options)
        if warm_profile is not None:
            kwargs["warm_profile"] = warm_profile
        if readback_policy is not None:
            kwargs["readback_policy"] = readback_policy
        elf = link_elf(cb, obj, work, **kwargs)
        if deadline is not None:
            deadline.remaining()
        return elf
    from merlin.runtime.backends import base as _backends

    from .. import build_cache as _bc
    from ..elf_lanes import PACKAGE_ELF_NAME

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_compile_"))
    if compact_contract is not None or logical_payloads is not None:
        if readback_policy is not None:
            raise ValueError("full-value readback does not support the prepared compact caller")
        if compact_contract is None or logical_payloads is None or inputs is not None:
            raise ValueError("compact build requires explicit contract + logical bytes, with no other inputs")
        prepare = getattr(_backends.get_backend(target), "prepare_compact_caller", None)
        if prepare is None:
            raise NotImplementedError("target has no verified compact caller preparation")
        prepared = prepare(
            cb,
            compact_contract,
            logical_payloads,
            lowered_mlir_text=lowered_mlir_text,
            workdir=work,
            prepack_authorizations=prepack_authorizations,
            max_storage_bytes=compact_storage_limit_bytes,
        )
        # Never reuse/publish a cached build: the ABI authority and exact logical
        # byte/prepack grants are not part of the legacy ELF cache key.
        obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
        kwargs = {"target": target, "_compact_caller": prepared}
        if warm_profile is not None:
            kwargs["warm_profile"] = warm_profile
        return link_elf(cb, obj, work, **kwargs)
    _explicit_prepack_inputs(inputs, prepack_authorizations)
    if prepack_authorizations is not None:
        # Cached builds skip the renderer's exact payload/binding checks. Until the
        # authorization policy is itself part of cache admission, never reuse or
        # publish such a build. The absent-authorization path remains unchanged.
        obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
        kwargs = {"target": target, "inputs": inputs, "prepack_authorizations": prepack_authorizations}
        if warm_profile is not None:
            kwargs["warm_profile"] = warm_profile
        if readback_policy is not None:
            kwargs["readback_policy"] = readback_policy
        return link_elf(cb, obj, work, **kwargs)
    if readback_policy is not None:
        # This invocation-only harness choice is absent from the historical
        # shared cache key. Never reuse or publish a legacy/digest-only ELF.
        obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
        return link_elf(
            cb,
            obj,
            work,
            target=target,
            inputs=inputs,
            warm_profile=warm_profile,
            readback_policy=readback_policy,
        )
    if warm_profile is not None:
        # The profile changes the runner-owned harness but is deliberately not
        # serialized into the command buffer.  Never let the legacy build key
        # reuse/publish a cold or differently instrumented ELF under this opt-in.
        obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
        return link_elf(cb, obj, work, target=target, inputs=inputs, warm_profile=warm_profile)
    # Coalesced ONCE. The harness embeds these operands, so a key computed from the caller's argument
    # while the build used the recorded ones would key two different executables the same way.
    if inputs is None:
        inputs = _recorded_operands(cb)
    try:
        key = _bc.build_identity(
            target=target,
            lowered_mlir_text=lowered_mlir_text,
            cb=cb,
            inputs=inputs,
            recipe=_backends.harness_build_recipe(target).with_effective_abi(),
        )
        # The paired ABI check is new build semantics. Do not restore an older cache entry that
        # predates it, even when an explicitly declared -mabi made the old recipe token identical.
        if key is not None:
            key = hashlib.sha256(("paired-kernel-abi-v1:" + key).encode("ascii")).hexdigest()
    except Exception:  # noqa: BLE001 -- an unkeyable build is an ordinary build
        key = None
    cached = _bc.reuse(work, key, PACKAGE_ELF_NAME)
    if cached is not None:
        return cached
    obj = llvm_mlir_to_object(lowered_mlir_text, work, target=target)
    elf = link_elf(cb, obj, work, target=target, inputs=inputs)
    _bc.store(key, work, PACKAGE_ELF_NAME)
    return elf


def simulator_provenance(backend, simulator: str) -> dict[str, Any] | None:
    """Identify the simulator BINARY an engine ran on: its path, its digest, and its recorded lineage.

    A result that claims a hardware verdict must record which hardware revision it came from. A cert
    record already pins the ELF, the RTL commits and the toolchain — and said nothing at all about the
    prebuilt simulator that produced the numbers, which for GSIM is an out-of-tree build whose bytes are
    the only thing tying the verdict to an elaboration. A stale or mis-provenanced emulator was therefore
    undetectable after the fact, which is the same failure mode the binary provenance stamp was added to
    catch on the ELF side.

    Derived from the backend, not from a table: it is asked for ``<engine>_status()`` (a sentence about
    how the engine resolved) and ``<engine>_path()`` (where its binary is), both DERIVED attribute names,
    so an engine or a backend that does not publish them contributes nothing rather than a fabricated
    entry. Never raises — provenance that cannot be established is recorded as absent.
    """
    from pathlib import Path as _Path

    rec: dict[str, Any] = {"engine": simulator}
    try:
        getter = getattr(backend, f"{simulator}_path", None)
        if callable(getter):
            binary = _Path(str(getter()))
            rec["binary"] = str(binary)
            if binary.is_file():
                from merlin.common import provenance as _prov

                rec["sha256"] = _prov.file_digest(binary)
        status = getattr(backend, f"{simulator}_status", None)
        if callable(status):
            _ok, _why = status()
            rec["resolution"] = str(_why)
    except Exception as exc:  # noqa: BLE001 — unestablished provenance is recorded, never invented
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec if len(rec) > 1 else None


def _counter_observations(
    console: str, *, target: str, simulator: str, cycles: int | None, oracle: Any
) -> tuple[list[dict] | None, dict | None]:
    """``(timing_observations, timing_capability)`` a bracketed run earned, or ``(None, None)``.

    THE HOP THAT WAS MISSING. The bracket emitter, the console parser, the wire contract and every
    consumer of a per-unit activity vector already existed; nothing turned the readings into the block
    a tier record carries, so a target that counts overlap in HARDWARE still reported
    ``missing: ['at least one activity source']`` and no composition operator, headroom or eta could
    resolve from it.

    ONLY AN RTL ORACLE MAY CARRY ONE. A functional model runs the program correctly without modelling
    the engines, so its counter CSRs describe nothing: measured on one, a 52-cycle window returned
    per-engine busy totals in the THOUSANDS. Those are not imprecise numbers, they are numbers about a
    different machine, and a composition operator derived from them would be a fabrication carrying a
    measurement's provenance. An oracle that states nothing fails closed for the same reason.

    NOTHING IS SWALLOWED. The first version of this guard put an unimported name inside a broad
    ``except``, so EVERY call returned "no capability": the negative cases passed for the wrong reason
    and the positive case silently never fired. Each refusal below is a specific, reachable condition.
    """
    if not isinstance(oracle, Mapping) or oracle.get("derived_from_rtl") is not True:
        return None, None  # a model's counters describe a different machine
    from merlin.perf import hw_counters
    from merlin.perf import observations as _observations

    readings = hw_counters.parse_counter_output(console)
    if not readings:
        return None, None  # unbracketed: byte-identical to before
    discovery = hw_counters.counters_for_target(target)
    if discovery.get("status") != "derived":
        return None, None  # no counter set derived from this target's own header
    measured_schema = hw_counters.parse_counter_schema(console)
    if measured_schema is not None and measured_schema != discovery.get("header_sha256"):
        return None, None  # the ELF was bracketed against a DIFFERENT schema
    # An ABSENT schema line is UNKNOWN, not a mismatch -- a real bracketed run need not emit one, and
    # refusing on its absence would refuse every such run. What actually binds the readings to this
    # header is the coverage check below: the reading set must contain every combination the header
    # derives, which a run bracketed against a different counter set cannot satisfy.
    header = Path(discovery["header"]).read_text(encoding="utf-8", errors="replace")
    occupancy = hw_counters.derive_occupancy_counters(header)
    required = set(occupancy.by_combination.values())
    if not required or not required <= set(readings):
        return None, None  # a partial combination set is a lower bound, not a total
    # The KIND of each engine is the TARGET's declaration: a kind cannot be read off a counter name,
    # and a consumer refuses a unit that lacks one. Absent when the backend declares none.
    from merlin.runtime.backends import base as _backends

    _kinds_reader = getattr(_backends.get_backend(target), "counter_engine_kinds", None)
    kinds = _kinds_reader() if callable(_kinds_reader) else None
    block = hw_counters.observations_from_counters(
        readings,
        occupancy,
        total_cycles=cycles,
        source=f"hardware combination counters ({discovery['header']})",
        kind_of=kinds,
    )
    validated = _observations.validate_block(block)
    if validated is None:
        return None, None
    # A refused block still travels as a capability record: "the producer emitted a block we could not
    # believe" is a fact about the instrument, and dropping it hides the instrument rather than the bug.
    return ([dict(o) for o in validated.observations] or None), validated.to_dict()


def run_on_oracle(
    cb: dict[str, Any],
    lowered_mlir_text: str,
    *,
    simulator: str,
    target: str,
    workdir: str | Path | None = None,
    timeout: int = 600,
    inputs: dict | None = None,
    readback_policy=None,
    memory_readback=None,
    oracle_revalidate=None,
    _build_service=None,
    execution_revalidate=None,
    _execution_service=None,
    _elf_admission=None,
    execution_deadline: ExecutionDeadline | None = None,
) -> dict[str, Any]:
    """Compile the package's lowered MLIR + run on ``simulator``; return outputs/metrics/console.

    ``timing`` splits the work: ``build_s`` (ELF compile/link) and ``sim_active_s`` (the simulator
    subprocess) are *active* time; ``oracle_wait_s`` is queue/FPGA-slot wait (0 for local sims like
    spike/verilator — only VCS/FireSim adapters that route through a queue set it).
    """
    from subprocess import CalledProcessError, TimeoutExpired

    deadline = selected_deadline(seconds=None, parent=execution_deadline, build_service=_build_service)
    if deadline is not None and _execution_service is None:
        raise ValueError("shared oracle deadline requires explicit independent execution support")

    def check_budget():
        return None if deadline is None else deadline.remaining()

    from merlin.runtime.backends import base as _backends

    if _execution_service is None:
        backend = _backends.get_backend(target)
    else:
        from .execution_service import FunctionalExecutionService

        if type(_execution_service) is not FunctionalExecutionService or _build_service is None:
            raise ValueError("explicit execution transport requires typed functional and build services")
        _execution_service.verify(target, simulator)
        backend = _execution_service
    from .readback_policy import FULL_VALUES_BIN, MEMORY_TRANSPORTS, selected

    readback_policy = selected(readback_policy)
    execution_identity = None
    if execution_revalidate is not None:
        if not callable(execution_revalidate):
            raise ValueError("execution revalidation requires an explicit selected-engine verifier")
        from copy import deepcopy

        execution_identity = deepcopy(execution_revalidate())
        if type(execution_identity) is not dict or not execution_identity:
            raise ValueError("execution revalidation requires a nonempty selected-engine citation")

    def verify_execution():
        if _execution_service is not None:
            _execution_service.verify(target, simulator)
        if execution_identity is not None and execution_revalidate() != execution_identity:
            raise ValueError("selected execution engine changed during ordinary oracle execution")

    memory = readback_policy is not None and readback_policy.transport in MEMORY_TRANSPORTS
    # A trusted evaluator supplies the admitted memory reader. Core never
    # discovers an optional grader, reads a candidate-selected transport, or
    # substitutes missing serial frames with an empty output roster.
    if memory:
        if not callable(getattr(memory_readback, "prepare", None)) or not callable(
            getattr(memory_readback, "decode", None)
        ):
            raise ValueError("memory output requires an explicit trusted admission reader")
        if not callable(oracle_revalidate):
            raise ValueError("memory output requires an explicit selected-engine revalidator")
    elif memory_readback is not None or oracle_revalidate is not None:
        raise ValueError("memory output reader requires the explicit memory readback policy")
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="oot_run_"))
    binary = readback_policy is not None and readback_policy.transport == FULL_VALUES_BIN
    console_path = work / ("oracle_console.bin" if binary else "oracle_console.log")
    stderr_path = work / "oracle_stderr.log"
    # A refused build or failed launch must not leave an earlier attempt's
    # transcript at this invocation's diagnostic path.
    console_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    _t0 = time.perf_counter()
    policy_kwargs = {"readback_policy": readback_policy} if readback_policy is not None else {}
    service_kwargs = {"_build_service": _build_service} if _build_service is not None else {}
    if deadline is not None:
        service_kwargs["execution_deadline"] = deadline
    elf = compile_lowered_to_elf(
        cb,
        lowered_mlir_text,
        work,
        target=target,
        inputs=inputs,
        **policy_kwargs,
        **service_kwargs,
    )
    check_budget()
    readback_build = None
    if readback_policy is not None:
        from .readback_policy import require_current_build_receipt

        readback_build = require_current_build_receipt(
            cb=cb,
            target=target,
            workdir=work,
            elf_path=elf,
            policy=readback_policy,
            **({"build_service": _build_service} if _build_service is not None else {}),
        )
    _t1 = time.perf_counter()
    check_budget()
    verify_execution()
    check_budget()
    admission = None
    if _elf_admission is not None:
        from .elf_admission import LinkedElfAdmissionService

        if type(_elf_admission) is not LinkedElfAdmissionService or _execution_service is None:
            raise ValueError("linked ELF admission requires the explicit independent execution route")
        admission = _elf_admission.evaluate(elf=elf, target=target, evidence_root=work / "elf_admission")
        check_budget()
        if admission["status"] == "refused":
            # This is a completed build and evaluated rejection, never an
            # executed binary, numerical verdict or simulator measurement.
            return {
                "status": "refused_before_execution",
                "elf": str(elf),
                "elf_admission": admission,
                "readback_build": readback_build,
                "console": "",
                "execution": "not_attempted",
            }
    memory_kwargs = {}
    memory_engine = None
    if memory:
        memory_kwargs = memory_readback.prepare(
            cb=cb,
            target=target,
            elf_path=Path(elf),
            workdir=work,
            simulator=simulator,
            backend=backend,
        )
        if (
            type(memory_kwargs) is not dict
            or set(memory_kwargs) != {"memory_readback"}
            or type(memory_kwargs["memory_readback"]) is not dict
        ):
            raise ValueError("memory admission must provide only the closed backend readback request")
        if memory_kwargs["memory_readback"].get("elf_sha256") != readback_build.get("elf_sha256"):
            raise ValueError("memory admission differs from the completed build ELF identity")
        from copy import deepcopy

        memory_engine = deepcopy(oracle_revalidate())
        if type(memory_engine) is not dict or not memory_engine:
            raise ValueError("memory output requires a selected-engine citation")
    try:
        run_kwargs = {"capture_bytes": True} if binary else {}
        from merlin.common import invocation_record

        # Record the actual Python dispatch. The selected backend owns the
        # engine subprocess argv and its own lower-level invocation records.
        provenance = (simulator_provenance(backend, simulator) or {}) if _execution_service is None else {}
        engine_path = provenance.get("binary")
        dependencies = (Path(__file__), Path(_deadline_owner.__file__))
        if isinstance(engine_path, str) and Path(engine_path).is_file():
            dependencies += (Path(engine_path),)
        if execution_identity is not None:
            dependencies += tuple(
                Path(row["path"])
                for row in execution_identity.values()
                if type(row) is dict and isinstance(row.get("path"), str)
            )
        if _execution_service is not None:
            dependencies += tuple(Path(path) for path, _ in _execution_service.source_pins)
        if admission is not None:
            _elf_admission.revalidate(elf=elf, result=admission, target=target)
        run_timeout = timeout if deadline is None else min(timeout, deadline.remaining())
        with invocation_record.observe_call(
            work,
            stage="execution",
            function=backend.run_elf,
            arguments={
                "target": target,
                "simulator": simulator,
                "timeout_s": run_timeout,
                "capture_bytes": binary,
                "memory_transport": memory,
            },
            inputs=(elf,),
            dependencies=dependencies,
        ) as observed:
            console = backend.run_elf(elf, simulator=simulator, timeout=run_timeout, **run_kwargs, **memory_kwargs)
            observed.returned(stdout=console)
    except (TimeoutExpired, CalledProcessError) as exc:
        # Standard process failures can carry partial output even with
        # text=True. Preserve bytes verbatim; they are diagnostic evidence,
        # never a completed frame or a numerical verdict. Re-raise unchanged.
        for path, data in ((console_path, exc.stdout), (stderr_path, exc.stderr)):
            if isinstance(data, (bytes, str)):
                path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        raise
    _t2 = time.perf_counter()
    # Parsing can refuse a truncated frame before this function returns a
    # result. Retain the complete, unfiltered transcript at the execution
    # boundary, not only on the successful grading path. This is diagnostic
    # evidence, never a completion or numerical verdict.
    console_path.write_bytes(console if type(console) is bytes else console.encode("utf-8"))
    check_budget()
    verify_execution()
    process_consumption = (
        None if _execution_service is None else _execution_service.consumption(elf=elf, console=console)
    )
    if admission is not None:
        _elf_admission.revalidate(elf=elf, result=admission, target=target)
    if memory and oracle_revalidate() != memory_engine:
        raise ValueError("memory oracle engine changed before output decoding")
    if memory and readback_build != require_current_build_receipt(
        cb=cb,
        target=target,
        workdir=work,
        elf_path=elf,
        policy=readback_policy,
        **({"build_service": _build_service} if _build_service is not None else {}),
    ):
        raise ValueError("memory readback build changed before output decoding")
    check_budget()
    outputs, raw = backend.parse_output(console)
    check_budget()
    memory_evidence = memory_observation = None
    if memory:
        from .readback_policy import require_memory_completion, require_memory_value_roster

        require_memory_completion(console, outputs)
        if _execution_service is not None:
            outputs, memory_evidence, memory_observation = _observed_memory_decode(
                memory_readback,
                console,
                cb=cb,
                elf=elf,
                workdir=work,
                policy=readback_policy,
                dependencies=tuple(Path(path) for path, _ in _execution_service.source_pins),
            )
        else:
            outputs, memory_evidence = memory_readback.decode(console)
            if (
                type(outputs) is not dict
                or type(memory_evidence) is not dict
                or memory_evidence.get("status") != "complete"
            ):
                raise ValueError("memory output reader returned no completed full-value admission")
            require_memory_value_roster(cb, outputs)
    check_budget()
    if readback_policy is not None:
        from .readback_policy import (
            require_current_build_receipt,
            require_full_value_roster,
        )

        if not memory:
            require_full_value_roster(cb, console, outputs, policy=readback_policy)
        if readback_build != require_current_build_receipt(
            cb=cb,
            target=target,
            workdir=work,
            elf_path=elf,
            policy=readback_policy,
            **({"build_service": _build_service} if _build_service is not None else {}),
        ):
            raise ValueError("full-value build identity changed during oracle execution")
    if memory and oracle_revalidate() != memory_engine:
        raise ValueError("memory oracle engine changed during output decoding")
    check_budget()
    # DECODE A FLOAT RESULT THAT CAME BACK AS ITS CONTAINER WORD. `parse_output` yields whatever the
    # console carried; a target whose harness has integer-only formatting prints a float destination
    # buffer's stored PATTERN, so an f32 result arrives as its 32-bit word and a bf16 result as its
    # 16-bit one. That is a lossless hand-back, not a broken one -- but only once the pattern is read
    # back as the value, which needs the dtype the buffer was DECLARED in. Read here, from the command
    # buffer itself (the same declaration the harness sized the buffer from), so the writer and the
    # reader cannot disagree; keyed on that dtype and on nothing target-specific. A no-op for an
    # integer-declared output and for a backend that already prints decimals, so every existing
    # readback is byte-identical.
    from merlin.runtime.commandbuffer import declared_output_dtypes

    outputs = _backends.decode_float_readback(outputs, declared_output_dtypes(cb))
    verify_execution()
    if (
        _execution_service is not None
        and _execution_service.consumption(elf=elf, console=console) != process_consumption
    ):
        raise ValueError("functional native consumption changed during output decoding")
    # WHICH BUILD of the simulator answered — recorded beside the oracle's declared kind, not inferred
    # afterwards. The tier record identifies the ELF, the RTL pins and the tools, and identified the one
    # remaining input to the verdict not at all: the prebuilt simulator binary. Derived, never assumed:
    # the backend is asked where its ``<engine>_path()`` is and the bytes there are digested. A backend
    # that does not expose one contributes nothing rather than a guess.
    _oracle = dict(backend.ORACLE[simulator]) if _execution_service is None else _execution_service.oracle
    _prov = simulator_provenance(backend, simulator) if _execution_service is None else None
    if _prov:
        _oracle["provenance"] = _prov
    if memory_engine is not None:
        _oracle["memory_engine"] = memory_engine
    result = {
        "outputs": outputs,
        "raw_metrics": raw,
        "cycles": raw.get("cycles", 0),
        "oracle": _oracle,
        "elf": str(elf),
        "console": console,
        "timing": {"build_s": round(_t1 - _t0, 3), "sim_active_s": round(_t2 - _t1, 3), "oracle_wait_s": 0.0},
    }
    if readback_build is not None:
        result["readback_build"] = readback_build
    if admission is not None:
        result["elf_admission"] = admission
    if execution_identity is not None:
        result["execution_identity"] = execution_identity
    if process_consumption is not None:
        result["process_consumption"] = process_consumption
    if memory_evidence is not None:
        result["readback_memory"] = memory_evidence
    if memory_observation is not None:
        result["readback_observation"] = memory_observation
    # Counter markers are a target-independent wire protocol.  The event names/codes remain the
    # target's own: this boundary merely preserves readings the runner already paid to collect.  If
    # they exactly cover a structurally derived joint-occupancy block, compute eta; otherwise retain
    # the raw named readings without guessing what they mean.
    from merlin.perf import counter_trust, hw_counters

    # Binary readback is an arbitrary byte stream. Only the text outside
    # structurally validated payload spans may carry counter/schema markers;
    # decoding the raw stream would also allow output values to forge them.
    counter_console = console
    if binary:
        from merlin.runtime.out_bin import binary_console_diagnostics

        counter_console = binary_console_diagnostics(console).decode("utf-8")
    readings = hw_counters.parse_counter_output(counter_console)
    # An engine that SYNTHESISES its accelerator counters must not have its readings stamped
    # "measured". The eta path above already refuses a non-RTL oracle; this raw-readings path did
    # not, so a functional model's numbers reached the report with a measurement's provenance --
    # which is the defect that block's own docstring describes. Refuse by ENGINE and say why, so an
    # absent field is distinguishable from one nobody collected.
    _trust = counter_trust.verdict_for(simulator)
    if readings and (_execution_service is not None or not _trust.trusted):
        result["counters"] = {
            "status": "unknown",
            "readings": None,
            "why": (
                "explicit functional transport has no qualified RTL counter authority"
                if _execution_service is not None
                else _trust.refusal()
            ),
            "engine": _trust.to_dict(),
        }
    elif readings:
        discovery = hw_counters.counters_for_target(target)
        measured_schema = hw_counters.parse_counter_schema(counter_console)
        report: dict[str, Any] = {
            "status": "measured",
            "readings": readings,
            "discovery": discovery,
            "measured_header_sha256": measured_schema,
        }
        if discovery.get("status") == "derived" and measured_schema == discovery.get("header_sha256"):
            header = Path(discovery["header"]).read_text(encoding="utf-8", errors="replace")
            occupancy = hw_counters.derive_occupancy_counters(header)
            required = set(occupancy.by_combination.values())
            if required and required <= set(readings):
                report["occupancy"] = occupancy.to_dict()
                partition_reader = getattr(backend, "counter_partition_inputs", None)
                partition = (
                    partition_reader()
                    if callable(partition_reader)
                    else {
                        "status": "unknown",
                        "why": "the target backend exposes no CIRCT counter-partition artifact",
                    }
                )
                if partition.get("status") == "available":
                    report["overlap"] = hw_counters.eta_from_counters(
                        readings,
                        occupancy,
                        hw_text=partition["hw_text"],
                        codes=hw_counters.event_codes(header),
                        module=partition["module"],
                        counter_module=partition["counter_module"],
                        measurement_cycles=raw.get("cycles"),
                        source=partition["source"],
                    )
                else:
                    report["overlap"] = {
                        "state": "unknown",
                        "eta": None,
                        "why": partition.get("why", "CIRCT counter-partition proof is unavailable"),
                    }
        elif discovery.get("status") == "derived":
            report["status"] = "unknown"
            report["overlap"] = {
                "state": "unknown",
                "eta": None,
                "why": "the measured ELF counter-schema digest does not match current discovery",
            }
        result["counters"] = report
    _obs, _cap = _counter_observations(
        counter_console, target=target, simulator=simulator, cycles=raw.get("cycles"), oracle=_oracle
    )
    if _cap is not None:
        result["timing_observations"] = _obs
        result["timing_capability"] = _cap
    check_budget()
    if (
        _execution_service is not None
        and _execution_service.consumption(elf=elf, console=console) != process_consumption
    ):
        raise ValueError("functional native consumption changed during output publication")
    return result
