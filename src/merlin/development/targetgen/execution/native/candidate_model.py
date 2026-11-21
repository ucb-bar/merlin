"""Bind original positional source results to actual compiled output names.

Only the checked current mixed-program source/return plan can justify a name
change. The independent golden and its numerical policy remain source-owned;
actual packet decoding and output hashing retain the compiler's real ABI names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.development.targetgen.execution.native.native_model_values import (
    logical_values as _decode_logical_values,
)

# ---- native_model_output_binding ---------------------------------------------------------------


def _refuse(detail):

    raise NativeModelExecutionError(detail)


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _refuse("source output binding requires ordinary frozen source files")
    return path.read_bytes()


def bind_source_outputs(command_buffer, capsule_dir, source_path, golden):
    """Return a fresh checked binding only when original output names differ.

    Same-name historical evidence keeps its original interpretation and checks.
    Mismatched names require the source-owned ordered declaration/provenance,
    byte-identical capsule/runtime source, full typed source plan and exact ABI.
    No candidate-supplied name-alias list is accepted.
    """
    import numpy as np
    import yaml

    compiled = (command_buffer.get("kernel_abi") or {}).get("outputs")
    outputs = golden.get("outputs") if isinstance(golden, Mapping) else None
    if isinstance(compiled, list) and isinstance(outputs, Mapping) and set(compiled) == set(outputs):
        return None
    if (
        not isinstance(compiled, list)
        or not compiled
        or any(type(name) is not str for name in compiled)
        or len(set(compiled)) != len(compiled)
        or not isinstance(outputs, Mapping)
    ):
        _refuse("independent golden outputs do not match a complete compiled output ABI")
    root = Path(capsule_dir)
    declaration_bytes = _read(root / "capsule.yaml")
    declaration = yaml.safe_load(declaration_bytes)
    if not isinstance(declaration, Mapping):
        _refuse("source output binding has no original capsule declaration")
    attributes = (declaration.get("operation") or {}).get("attributes") or {}
    names = attributes.get("outs")
    if names is None:
        names = [attributes.get("out")]
    if (
        not isinstance(names, list)
        or any(type(name) is not str or not name for name in names)
        or len(names) != len(compiled)
        or len(set(names)) != len(names)
        or set(names) != set(outputs)
    ):
        _refuse("source output binding requires the complete original ordered output declaration")
    inputs = declaration.get("inputs")
    if not isinstance(inputs, list) or any(
        not isinstance(row, Mapping) or type(row.get("name")) is not str for row in inputs
    ):
        _refuse("source output binding has no complete original input order")
    order = [row["name"] for row in inputs] + names
    provenance = golden.get("oracle_provenance") or {}
    if (
        attributes.get("arg_order") != order
        or provenance.get("arg_order") != order
        or provenance.get("interface") != "linalg_positional"
    ):
        _refuse("source output binding original declaration/provenance order differs")
    interface = declaration.get("interface_mlir") or declaration.get("linalg_mlir")
    if (
        type(interface) is not str
        or not interface
        or Path(interface).is_absolute()
        or ".." in Path(interface).parts
        or provenance.get("linalg_mlir") != interface
    ):
        _refuse("source output binding has no exact local positional interface")
    if declaration.get("linalg_mlir", interface) != interface or source_path is None:
        _refuse("source output binding interface declarations are inconsistent")
    source = _read(source_path)
    if source != _read(root / interface):
        _refuse("source output binding runtime source differs from the frozen interface")
    from merlin.development.targetgen.oot_starterkit.plan import source_operation_inventory, validate_mixed_program_plan

    checked = validate_mixed_program_plan(source, command_buffer)
    if checked.get("ok") is not True:
        _refuse(f"source output binding has no exact source/return plan: {checked.get('findings', [])[:2]}")
    inventory = source_operation_inventory(source)
    returned = inventory["returns"]
    if len(returned) != len(names):
        _refuse("source output binding count differs from complete source returns")
    rows = []
    for position, (name, tensor, value) in enumerate(zip(names, compiled, returned, strict=True)):
        policy = declaration.get("numeric_policy") or {}
        declared_dtype = (policy.get("outputs", {}).get(name, policy)).get("dtype")
        if (
            "shape" not in value
            or "dtype" not in value
            or value.get("source") is None
            or list(np.asarray(outputs[name]).shape) != value["shape"]
            or declared_dtype is not None
            and declared_dtype != value["dtype"]
        ):
            _refuse("source output binding original reference shape or typed return differs")
        rows.append(
            dict(
                position=position,
                source_name=name,
                compiled_name=tensor,
                shape=value["shape"],
                dtype=value["dtype"],
                source_value=value["source"],
            )
        )
    return dict(
        schema="merlin.source_output_binding.v1",
        source_sha256=hashlib.sha256(source).hexdigest(),
        capsule_declaration_sha256=hashlib.sha256(declaration_bytes).hexdigest(),
        returns=rows,
        authority="checked source return order and typed compiled ABI; names only",
    )


def source_output_values(observed, binding):
    """Rename only decoded complete actual outputs for original-policy comparison."""
    if binding is None:
        return observed
    rows = binding["returns"]
    if (
        not isinstance(observed, Mapping)
        or set(observed) != {row["compiled_name"] for row in rows}
        or len(observed) != len(rows)
    ):
        _refuse("actual compiled output roster differs from the checked source output binding")
    return {row["source_name"]: observed[row["compiled_name"]] for row in rows}


def checked_source_output_values(command_buffer, native, golden, observed):
    """Reconstruct correspondence from original pins before any saved verdict use."""
    from merlin.common.digest import strict_json_equal

    physical = (command_buffer.get("kernel_abi") or {}).get("outputs")
    if physical is None:
        # Historical tier replay used the unchanged readback tensor roster.
        # Fresh whole-program execution still requires its explicit ABI above.
        from merlin.runtime.engine.command_buffer import declared_output_dtypes

        physical = list(declared_output_dtypes(command_buffer))
    names = golden.get("outputs") or {}
    expected = None
    if set(physical) != set(names):
        source = native.get("source") or {}
        capsule = _pinned_native_file(source, "capsule_declaration").parent
        interface = _pinned_native_file(source, "interface")
        expected = bind_source_outputs(command_buffer, capsule, interface, golden)
    if not strict_json_equal(native.get("source_output_binding"), expected):
        _refuse("native source output binding differs from freshly checked original evidence")
    return source_output_values(observed, expected)


# ---- native_model_execution --------------------------------------------------------------------
# Execute an emitted whole-model kernel on a frozen capsule's exact operands.
#
# This is an evidence adapter, not a model compiler or a certification verdict.  The
# caller supplies the candidate's already-emitted command buffer and LLVM-dialect
# artifact; no reference compiler can substitute its own model implementation here.


def audit_candidate_tiers(
    emission: Mapping[str, Any] | None,
    certificate: Mapping[str, Any] | None,
    native: Mapping[str, Any] | None,
    *,
    target: str,
    entry_symbol: str,
    completed_dispatch: Mapping[str, Any] | None,
    readback_policy=None,
    oracle_tier_ceiling=None,
    source_intake_protocol=None,
) -> dict[str, Any]:
    """Audit each frozen mandatory tier against this candidate's own artifact.

    L0/L1 are recomputed; L2/L3 reopen console bytes and independent golden.
    Merely listing a tier in the receipt never counts as exercising it.
    Transport comes only from the trusted invocation's explicit policy, never
    a candidate receipt, console filename or heuristic inspection of bytes.
    """
    from merlin.development.targetgen.execution.native.model_execution import (
        NativeModelExecutionError,
        _frozen_model_policy,
        _pinned_native_file,
        audit_candidate_static_tiers,
    )
    from merlin.target.contract.harness.readback_policy import selected

    readback_policy = selected(readback_policy)
    if oracle_tier_ceiling not in (None, "L2"):
        raise NativeModelExecutionError("unsupported public numerical ceiling")
    tiers = audit_candidate_static_tiers(emission, certificate, native, target=target, entry_symbol=entry_symbol)
    try:
        if not isinstance(native, Mapping):
            raise NativeModelExecutionError("native receipt is absent")
        policy = _frozen_model_policy(
            _pinned_native_file(native["source"], "capsule_declaration").parent, target=target
        )
        if native.get("frozen_policy") != policy:
            raise NativeModelExecutionError("native receipt names different frozen tier policy")
        required = policy["required_tiers"]
    except Exception as exc:  # noqa: BLE001 -- no policy means no tier may be admitted
        return {
            "status": "unverified",
            "detail": f"{type(exc).__name__}: {exc}",
            "tiers": tiers,
            "required_tiers": None,
        }
    from merlin.development.targetgen.execution.native.model_policy import audit_native_numerical_tier

    tiers["L2"] = audit_native_numerical_tier(
        emission,
        native,
        target=target,
        readback_policy=readback_policy,
        **({"source_intake_protocol": source_intake_protocol} if source_intake_protocol is not None else {}),
    )
    from merlin.development.targetgen.execution.native.model_policy import audit_rtl_tier

    tiers["L3"] = (
        {"status": "pending", "scope": "outside explicit L2 numerical ceiling"}
        if oracle_tier_ceiling == "L2"
        else audit_rtl_tier(native, completed_dispatch)
    )
    missing = [tier for tier in required if (tiers.get(tier) or {}).get("status") != "pass"]
    failed = [tier for tier in required if (tiers.get(tier) or {}).get("status") == "fail"]
    return {
        "status": "fail" if failed else "pass" if not missing else "unverified",
        "required_tiers": required,
        "failed_tiers": failed,
        "missing_tiers": missing,
        "tiers": tiers,
        "scope": "candidate-only mandatory tier evidence; L0/L1 structural/ISA, L2 functional, L3 RTL",
    }


def audit_candidate_completed_dispatch(
    emission: Mapping[str, Any] | None,
    certificate: Mapping[str, Any] | None,
    native: Mapping[str, Any] | None,
    *,
    target: str,
    entry_symbol: str,
    readback_policy=None,
) -> dict[str, Any]:
    """Infer mandatory candidate dispatch from exact native completion and codegen.

    This is NOT a hardware commit trace. The trusted translator/compiler/linker
    form the bridge from a side-effecting LLVM command to object and linked ELF;
    each independently eligible source task must have a family-matched work command
    that no returning CFG path can bypass. The complete custom-command multiset
    must survive in the *kernel symbol* of both compiled artifacts. Unknown
    transforms or missing attribution stay unverified, not clean.
    """
    from merlin.development.targetgen.execution.native.dispatch_accounting import (
        _audit_candidate_completed_dispatch,
    )
    from merlin.development.targetgen.execution.native.model_execution import audit_candidate_source_placement

    return _audit_candidate_completed_dispatch(
        emission,
        certificate,
        native,
        target=target,
        entry_symbol=entry_symbol,
        source_placement=audit_candidate_source_placement,
        **({"readback_policy": readback_policy} if readback_policy is not None else {}),
    )


def execute_candidate_model(
    *,
    command_buffer: Mapping[str, Any],
    lowered_mlir_text: str,
    capsule_dir: str | Path,
    capture_bundle: str | Path,
    target: str,
    out_dir: str | Path,
    simulator: str | None = None,
    timeout: int = 600,
    numeric_policy: Mapping[str, Any] | None = None,
    rtl_facts: str | Path | None = None,
    board_config: str | None = None,
    readback_policy=None,
    complete_call_timing=False,
    oracle_tier_ceiling=None,
    input_storage: str = "inline",
    source_intake_protocol=None,
) -> dict[str, Any]:
    """Build the source-bound candidate ELF and compare every declared output.

    The frozen runtime bundle and independent capsule golden retain their
    original guards. A numerical screen leaves deeper required tiers pending;
    no comparison alone establishes source transformation or physical timing.
    """
    from merlin.development.targetgen.execution.native.model_execution import (
        NativeModelExecutionError,
        _bind_inputs,
        _build_artifacts,
        _build_service_for,
        _digest,
        _frozen_model_policy,
        _functional_engine,
        _host_compute_report,
    )
    from merlin.development.targetgen.grading.golden_store import load_golden
    from merlin.development.targetgen.packaging.bundle_harness import (
        emitted_entry_arity,
        is_executable_emission,
    )
    from merlin.target.contract.harness.readback_policy import FULL_VALUES_BIN, MEMORY_TRANSPORTS, selected

    readback_policy = selected(readback_policy)
    binary_console = readback_policy is not None and readback_policy.transport == FULL_VALUES_BIN
    memory_console = readback_policy is not None and readback_policy.transport in MEMORY_TRANSPORTS

    source = Path(capsule_dir)
    capture = Path(capture_bundle)
    output = Path(out_dir)
    if output.is_symlink():
        raise NativeModelExecutionError(f"artifact directory must not be a symlink: {output}")
    if output.exists() and any(output.iterdir()):
        raise NativeModelExecutionError(f"fresh artifact directory required: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "merlin_candidate_native_model_execution_v1",
        "status": "incomplete",
        "target": target,
        "simulator": simulator,
        "scope": "candidate whole-program ELF output diagnostic; not a Phase-1 certification",
    }
    receipt = output / "result.json"
    from merlin.runtime.harness.call_timing import _NUMERICS_ONLY

    numerical_only = simulator == "spike" and oracle_tier_ceiling == "L2" and complete_call_timing is False
    timing_scope = _NUMERICS_ONLY.set(numerical_only)
    try:
        if source_intake_protocol is not None:
            from merlin.development.targetgen.execution.native import (
                native_pointer_source_binding as source_intake,
            )

            source_intake.intake_limits(source_intake_protocol)
            if not numerical_only:
                raise NativeModelExecutionError("prospective source intake requires explicit Spike/L2 numerical scope")
            record["source_intake"] = source_intake.preflight_large_intake(
                command_buffer, source, capture, intake_protocol=source_intake_protocol
            )
            input_storage = "streamed"
        if (simulator == "spike" and not numerical_only) or (oracle_tier_ceiling is not None and not numerical_only):
            raise NativeModelExecutionError("Spike model execution requires its explicit L2 numerical ceiling")
        if numerical_only:
            record["oracle_tier_ceiling"] = "L2"
        if not source.is_dir() or source.is_symlink() or not capture.is_dir() or capture.is_symlink():
            raise NativeModelExecutionError("model capsule and frozen runtime bundle must be ordinary directories")
        cb = json.loads(json.dumps(command_buffer))
        if (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
            raise NativeModelExecutionError("candidate did not declare a whole_program pointer ABI")
        from merlin.runtime.backends import registry as backends

        if type(complete_call_timing) is not bool:
            raise NativeModelExecutionError("complete-call selection must be an explicit boolean")
        selected_backend = backends.get_backend(target)
        resolver = getattr(selected_backend, "complete_call_timing", None)
        timing_plan = resolver() if callable(resolver) else None
        if complete_call_timing and timing_plan is None:
            raise NativeModelExecutionError(
                "selected normal caller has no complete-call counter/clock source selection"
            )
        from merlin.common.process.ownership import ExecutionDeadline

        deadline = ExecutionDeadline.start(timeout) if timing_plan is not None or numerical_only else None

        def stage_timeout():
            return timeout if deadline is None else deadline.remaining("complete public program budget expired")

        if timing_plan is not None:
            timing_plan.verify()
            record["complete_call_timing"] = timing_plan.record()

        entry_symbol = backends.harness_build_recipe(target).require_kernel_stack_frame().entry_symbol
        ok, reason = is_executable_emission(cb, artifact_text=lowered_mlir_text, entry_symbol=entry_symbol)
        if not ok:
            raise NativeModelExecutionError(reason)
        abi = cb["kernel_abi"]
        outputs = abi.get("outputs")
        if (
            not isinstance(outputs, list)
            or not outputs
            or any(type(name) is not str for name in outputs)
            or len(set(outputs)) != len(outputs)
        ):
            raise NativeModelExecutionError("complete unique whole-model outputs are required")
        if any((cb.get("tensors") or {}).get(name, {}).get("role") != "output" for name in outputs):
            raise NativeModelExecutionError("declared model output has no output tensor role")
        golden = load_golden(source, **({"compact_arrays": True} if source_intake_protocol is not None else {}))
        if not isinstance(golden, dict):
            raise NativeModelExecutionError("independent golden outputs do not match the candidate's ABI")
        output_binding = bind_source_outputs(cb, source, capture / "model.mlir", golden)
        if output_binding is not None:
            record["source_output_binding"] = output_binding
        if numeric_policy is None:
            import yaml

            declaration = yaml.safe_load((source / "capsule.yaml").read_text(encoding="utf-8"))
            if not isinstance(declaration, dict):
                raise NativeModelExecutionError("frozen capsule declaration is not a mapping")
            policy = dict(declaration.get("numeric_policy") or {})
        else:
            policy = dict(numeric_policy)
        if policy.get("compare") not in ("exact_int", "tolerance_float", "per_output"):
            raise NativeModelExecutionError("no explicit supported numeric comparison policy")
        original_names = (
            {row["compiled_name"]: row["source_name"] for row in output_binding["returns"]} if output_binding else {}
        )
        if any(
            (policy.get("outputs", {}).get(original_names.get(name, name), policy)).get("compare") == "exact_int"
            and str(cb["tensors"][name]["dtype"]).startswith("f")
            for name in outputs
        ):
            raise NativeModelExecutionError("exact_int policy cannot certify a floating output")
        input_options = {"input_storage": input_storage} if input_storage != "inline" else {}
        inputs, binding = _bind_inputs(
            cb, capture, target=target, rtl_facts=rtl_facts, board_config=board_config, **input_options
        )
        record["binding"] = binding
        record["source"] = {
            "capsule_declaration": _digest(source / "capsule.yaml"),
            "interface": _digest(capture / "model.mlir"),
            "weights": _digest(capture / "weights.safetensors"),
            "weight_manifest": _digest(capture / "weights.safetensors.manifest.json"),
            "inputs": _digest(capture / "inputs.npz"),
            "golden": _digest(source / "golden.yaml"),
        }
        arrays = source / "golden.npz"
        if arrays.is_file():
            record["source"]["golden_arrays"] = _digest(arrays)
        record["frozen_policy"] = _frozen_model_policy(source, target=target)

        def revalidate_source() -> None:
            for role, pinned in record["source"].items():
                current = _digest(Path(pinned["path"]))
                if current["sha256"] != pinned["sha256"] or current["size_bytes"] != pinned["size_bytes"]:
                    raise NativeModelExecutionError(f"frozen {role} bytes changed during candidate execution")

        record["candidate"] = {
            "command_buffer_sha256": hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest(),
            "lowered_mlir_sha256": hashlib.sha256(lowered_mlir_text.encode()).hexdigest(),
            "entry_arity": emitted_entry_arity(lowered_mlir_text, entry_symbol=entry_symbol),
            "abi_args": len(abi["args"]),
        }
        from merlin.evaluation.host_compute_accounting import (
            HostComputeUnverified,
            HostComputeViolation,
            require_clean_host_compute,
        )

        scratch = tuple(binding["source_entry_binding"]["source_owned_mutables"])
        service = _build_service_for(
            target,
            source_owned_mutables=scratch or None,
            **({"source_intake_protocol": source_intake_protocol} if source_intake_protocol is not None else {}),
        )
        if service.recipe.require_kernel_stack_frame().entry_symbol != entry_symbol:
            raise NativeModelExecutionError("selected entry symbol changed during candidate source binding")
        host_report = _host_compute_report(cb, lowered_mlir_text, entry_symbol=entry_symbol)
        record["host_compute"] = host_report.to_dict()
        try:
            require_clean_host_compute(host_report)
        except (HostComputeUnverified, HostComputeViolation) as exc:
            record["host_compute_qualification"] = {
                "status": "unverified" if isinstance(exc, HostComputeUnverified) else "violation",
                "detail": str(exc),
            }
        else:
            record["host_compute_qualification"] = {"status": "clean"}
        from merlin.development.targetgen.execution.runner.oracle_run_compile import compile_lowered_to_elf

        record["host_build_sources"] = [{"path": p, "sha256": digest} for p, digest in service.source_pins]
        policy_kwargs = {"readback_policy": readback_policy} if readback_policy is not None else {}
        elf = compile_lowered_to_elf(
            cb,
            lowered_mlir_text,
            output / "build",
            target=target,
            inputs=inputs,
            _build_service=service,
            execution_deadline=deadline,
            **policy_kwargs,
        )
        if readback_policy is not None:
            from merlin.development.targetgen.execution.runner.native_readback import verify_build

            recipe_record, source_pins, record["readback_build"] = verify_build(
                target=target,
                policy=readback_policy,
                service=service,
                output=output,
                cb=cb,
                elf=elf,
            )
        record["elf"] = _digest(Path(elf))
        if timing_plan is not None:
            harness_path = output / "build" / "harness.c"
            expected_caller = service.render(cb, target=target, inputs=inputs, blobs={}, **policy_kwargs)
            declarations = (
                "\n".join(timing_plan.declarations(console="stdio")),
                "\n".join(timing_plan.declarations(console="htif")),
            )
            if (
                not harness_path.is_file()
                or harness_path.read_text() != expected_caller
                or not any(code in expected_caller for code in declarations)
                or "merlin_complete_call_start" not in expected_caller
                or "merlin_complete_call_cycles" not in expected_caller
            ):
                raise NativeModelExecutionError(
                    "ordinary build differs from its original selected complete-call caller"
                )
            record["complete_call_timing"]["caller"] = _digest(harness_path)
            record["complete_call_timing"]["elf"] = record["elf"]
            timing_plan.verify()
        revalidate_source()
        if rtl_facts is not None and board_config:
            from merlin.runtime.execution.model_inputs import selected_firrtl

            record["rtl_facts"] = selected_firrtl(rtl_facts, target=target, config=board_config)
        if simulator is None:
            record["status"] = "compiled_not_run"
            return record
        if rtl_facts is None or not board_config:
            raise NativeModelExecutionError("native RTL execution requires selected facts and board config")
        from merlin.development.targetgen.grading.golden import compare
        from merlin.runtime.backends import registry as backends
        from merlin.runtime.engine.command_buffer import declared_output_dtypes
        from merlin.runtime.execution.model_inputs import native_engine, selected_firrtl
        from merlin.target.simulators.oracle_policy import selected_l3_engine_report

        selected = None if numerical_only else selected_l3_engine_report(target)
        if not numerical_only and (not selected.get("available") or selected.get("engine") != simulator):
            raise NativeModelExecutionError("requested simulator is not the selected RTL oracle")
        facts = selected_firrtl(rtl_facts, target=target, config=board_config)
        if memory_console:
            from merlin.development.backend.feedback.candidates.memory_readback import execute_memory_elf

            def memory_run(selected_backend, engine_name, revalidate_engine):
                def after_run():
                    revalidate_engine()
                    revalidate_source()
                    verify_build(
                        target=target,
                        policy=readback_policy,
                        service=service,
                        output=output,
                        cb=cb,
                        elf=elf,
                        expected=(recipe_record, source_pins),
                        expected_receipt=record["readback_build"],
                        stage="memory pre-decode",
                    )

                return execute_memory_elf(
                    cb=cb,
                    target=target,
                    elf_path=Path(elf),
                    workdir=output / f"memory-{engine_name}",
                    facts_path=Path(facts["path"]),
                    simulator=engine_name,
                    backend=selected_backend,
                    timeout=stage_timeout(),
                    expected_elf_sha256=record["elf"]["sha256"],
                    post_run_revalidate=after_run,
                    readback_policy=readback_policy,
                )

        # The original byte-bound functional tier consumes this same candidate ELF.
        try:
            functional_backend, functional_citation, revalidate_functional = _functional_engine(target)
            revalidate_functional()
            capture_kwargs = {"capture_bytes": True} if binary_console else {}
            if memory_console:
                functional_console, functional_observed, functional_metrics, memory_record = memory_run(
                    functional_backend, "spike", revalidate_functional
                )
                record["readback_memory_l2"] = memory_record
            else:
                from merlin.development.targetgen.execution.native.model_policy import observed_execution

                functional_console, execution_pin = observed_execution(
                    functional_backend,
                    elf,
                    simulator="spike",
                    citation=functional_citation,
                    directory=output,
                    timeout=stage_timeout(),
                    capture_kwargs=capture_kwargs,
                )
                record.setdefault("execution_invocations", {})["L2"] = execution_pin
            revalidate_functional()
            revalidate_source()
            if _digest(Path(elf)) != record["elf"]:
                raise NativeModelExecutionError("candidate ELF changed during L2 functional execution")
            if readback_policy is not None:
                verify_build(
                    target=target,
                    policy=readback_policy,
                    service=service,
                    output=output,
                    cb=cb,
                    elf=elf,
                    expected=(recipe_record, source_pins),
                    expected_receipt=record["readback_build"],
                    stage="L2",
                )
            functional_path = output / ("console_l2.bin" if binary_console else "console_l2.txt")
            functional_path.write_bytes(
                functional_console if type(functional_console) is bytes else functional_console.encode("utf-8")
            )
            if not memory_console:
                functional_observed, functional_metrics = functional_backend.parse_output(functional_console)
            if readback_policy is not None and not memory_console:
                from merlin.target.contract.harness.readback_policy import require_full_value_roster

                require_full_value_roster(cb, functional_console, functional_observed, policy=readback_policy)
            functional_observed = backends.decode_float_readback(functional_observed, declared_output_dtypes(cb))
            compare_outputs = (
                source_intake.compare_complete_integer_outputs if source_intake_protocol is not None else compare
            )
            functional_numeric = compare_outputs(
                golden["outputs"],
                checked_source_output_values(cb, record, golden, functional_observed),
                policy,
                golden_source=str(golden.get("golden_source") or "independent_capsule"),
            )
            record.setdefault("tiers", {})["L2"] = {
                "status": "pass" if functional_numeric.get("status") == "pass" else "fail",
                "engine": "spike",
                "engine_citation": functional_citation,
                "elf": record["elf"],
                "console": _digest(functional_path),
                "numeric": functional_numeric,
                "metrics": functional_metrics,
                "scope": "same candidate ELF on selected functional engine; independent full golden",
            }
        except Exception as exc:  # noqa: BLE001 -- no substitution for unavailable L2
            record.setdefault("tiers", {})["L2"] = {
                "status": "unavailable",
                "detail": f"{type(exc).__name__}: {exc}",
                "elf": record["elf"],
            }
        if numerical_only:
            from merlin.development.targetgen.execution.native.model_policy import finish_numerical_execution

            return finish_numerical_execution(record)
        backend, engine, revalidate, prepare = native_engine(target, simulator, facts)
        record["rtl_facts"] = facts
        record["simulator_provenance"] = {"selection": selected, "citation": engine}
        command = (
            prepare(elf, expected_elf_sha256=record["elf"]["sha256"], expected_engine_provenance=engine)
            if prepare is not None
            else None
        )
        if command is not None:
            record["native_base_command" if memory_console else "native_command"] = command.to_evidence()
        revalidate()
        capture_kwargs = {"capture_bytes": True} if binary_console else {}
        if memory_console:
            console, observed, metrics, memory_record = memory_run(backend, simulator, revalidate)
            record["readback_memory_l3"] = memory_record
        elif simulator == "gsim":
            from merlin.target.simulators.engine_policy import gsim_runtime_slot

            with gsim_runtime_slot(wait_timeout_s=stage_timeout()):
                from merlin.development.targetgen.execution.native.model_policy import observed_execution

                console, execution_pin = observed_execution(
                    backend,
                    elf,
                    simulator=simulator,
                    citation=engine,
                    directory=output,
                    timeout=stage_timeout(),
                    capture_kwargs=capture_kwargs,
                )
                record.setdefault("execution_invocations", {})["L3"] = execution_pin
        else:
            from merlin.development.targetgen.execution.native.model_policy import observed_execution

            console, execution_pin = observed_execution(
                backend,
                elf,
                simulator=simulator,
                citation=engine,
                directory=output,
                timeout=stage_timeout(),
                capture_kwargs=capture_kwargs,
            )
            record.setdefault("execution_invocations", {})["L3"] = execution_pin
        if command is not None:
            command.revalidate()
        revalidate()
        revalidate_source()
        if _digest(Path(elf))["sha256"] != record["elf"]["sha256"]:
            raise NativeModelExecutionError("candidate ELF bytes changed during execution")
        if readback_policy is not None:
            verify_build(
                target=target,
                policy=readback_policy,
                service=service,
                output=output,
                cb=cb,
                elf=elf,
                expected=(recipe_record, source_pins),
                expected_receipt=record["readback_build"],
                stage="L3",
            )
        console_path = output / ("console.bin" if binary_console else "console.txt")
        console_path.write_bytes(console if type(console) is bytes else console.encode("utf-8"))
        if not memory_console:
            observed, metrics = backend.parse_output(console)
        if readback_policy is not None and not memory_console:
            from merlin.target.contract.harness.readback_policy import require_full_value_roster

            require_full_value_roster(cb, console, observed, policy=readback_policy)
        observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
        compare_outputs = (
            source_intake.compare_complete_integer_outputs if source_intake_protocol is not None else compare
        )
        comparison = compare_outputs(
            golden["outputs"],
            checked_source_output_values(cb, record, golden, observed),
            policy,
            golden_source=str(golden.get("golden_source") or "independent_capsule"),
        )
        record.update(
            console=_digest(console_path),
            metrics=metrics,
            numeric=comparison,
            status="numeric_match_diagnostic" if comparison["status"] == "pass" else "numeric_mismatch_diagnostic",
        )
        from merlin.development.backend.optimization.measurement.gsim_workload import (
            OUTPUT_ENCODING,
            encode_declared_outputs,
        )

        output_sha, output_tensors = encode_declared_outputs(observed, cb)
        record["output_observation"] = {"sha256": output_sha, "encoding": OUTPUT_ENCODING, "tensors": output_tensors}
        if timing_plan is not None:
            service.verify(target)
            timing_plan.verify()
            if (
                _digest(Path(record["complete_call_timing"]["caller"]["path"]))
                != record["complete_call_timing"]["caller"]
            ):
                raise NativeModelExecutionError("original complete-call caller changed during execution")
        record.setdefault("tiers", {})["L3"] = {
            "status": "pass" if comparison.get("status") == "pass" else "fail",
            "engine": simulator,
            "engine_citation": engine,
            "selection": selected,
            "elf": record["elf"],
            "console": record["console"],
            "numeric": comparison,
            "metrics": metrics,
            "scope": "same candidate ELF on selected RTL engine; independent full golden",
        }
        return record
    except Exception as exc:  # noqa: BLE001 -- always retain a build/receipt on failure
        if record.get("status") == "numeric_match_diagnostic":
            record["status"] = "incomplete"
            record.pop("numeric", None)
        if record.get("tiers", {}).get("L3", {}).get("status") == "pass":
            record["tiers"]["L3"] = {"status": "unavailable", "detail": "post-run source or build check failed"}
        record["failure"] = {"type": type(exc).__name__, "detail": str(exc)[:2000]}
        if isinstance(exc, NativeModelExecutionError) and exc.code is not None:
            record["failure"]["code"] = exc.code
        return record
    finally:
        try:
            record["build_artifacts"] = _build_artifacts(output)
            receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        finally:
            _NUMERICS_ONLY.reset(timing_scope)


def __getattr__(name: str) -> Any:
    """Keep historical private helper imports without eagerly cycling the grader modules."""
    if name in {
        "_mandatory_command_blocks",
        "_kernel_command_inventory",
        "_verified_work_functs_by_family",
        "_completed_eligible_tasks",
    }:
        from merlin.development.targetgen.execution.native import (
            dispatch_accounting as native_dispatch_accounting,
        )

        return getattr(native_dispatch_accounting, name)
    raise AttributeError(name)
