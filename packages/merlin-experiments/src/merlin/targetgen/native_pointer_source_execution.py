"""Explicit standalone pointer-source execution through the original harness.

The candidate's command buffer remains unchanged. Typed storage binding and a
complete numerical screen do not replace the independent live source/effect
owner, original writer, required deeper tiers or hardware/timing qualification.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.execution_deadline import ExecutionDeadline
from merlin.runtime.complete_call_timing import _NUMERICS_ONLY
from merlin.targetgen.contract.compile import compile_lowered_to_elf
from merlin.targetgen.contract.compile_only import require_pointer_entry
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy, require_full_value_roster

from . import native_model_execution as original
from . import native_pointer_source_binding as pointer_binding
from .native_model_execution import NativeModelExecutionError
from .native_model_policy import finish_numerical_execution, observed_execution
from .native_pointer_source_binding import PROTOCOL, _plain, bind_pointer_source, source_tree

MAX_TIMEOUT_SECONDS = 600


def execute_candidate_pointer_source(
    *,
    command_buffer,
    lowered_mlir_text,
    capsule_dir,
    capture_bundle,
    target,
    out_dir,
    simulator=None,
    timeout=MAX_TIMEOUT_SECONDS,
    numeric_policy=None,
    rtl_facts=None,
    board_config=None,
    readback_policy=None,
    complete_call_timing=False,
    oracle_tier_ceiling=None,
    input_storage="inline",
):
    """Run only the explicitly selected complete signed-i8/raw-i32 pointer mode.

    The signature matches the whole-model adapter, whose behavior is unchanged.
    Constants, nontrivial layouts and mutable entry storage remain unsupported.
    No serialized result issues the required original live realization owner.
    """
    from merlin.compile.model_execution_inputs import selected_firrtl
    from merlin.targetgen.capsule_golden import compare
    from merlin.targetgen.golden_store import load_golden

    source, capture, output = map(_plain, (capsule_dir, capture_bundle, out_dir))
    if output.exists() or any(
        output == root or output.is_relative_to(root) or root.is_relative_to(output) for root in (source, capture)
    ):
        raise NativeModelExecutionError("standalone pointer source needs a fresh disjoint evidence directory")
    output.mkdir(parents=True)
    record = dict(
        schema="merlin_candidate_native_pointer_source_v1",
        status="incomplete",
        target=target,
        simulator=simulator,
        source_execution_protocol=PROTOCOL,
        scope="standalone logical-pointer complete-output diagnostic; source/effect and deeper tiers pending",
        hardware_execution_qualified=False,
        timing_authority="NONE",
    )
    receipt = output / "result.json"
    scope = _NUMERICS_ONLY.set(True)
    try:
        if (
            simulator != "spike"
            or oracle_tier_ceiling != "L2"
            or type(complete_call_timing) is not bool
            or complete_call_timing is not False
            or input_storage != "inline"
        ):
            raise NativeModelExecutionError(
                "standalone pointer source v1 requires explicit Spike/L2, inline inputs and no timing"
            )
        if type(timeout) is not int or not 0 < timeout <= MAX_TIMEOUT_SECONDS or rtl_facts is None or not board_config:
            raise NativeModelExecutionError(
                "standalone pointer source needs bounded timeout and exact hardware selection"
            )
        if readback_policy is None:
            readback_policy = ReadbackPolicy(FULL_VALUES_B64)
        if type(readback_policy) is not ReadbackPolicy or readback_policy.transport != FULL_VALUES_B64:
            raise NativeModelExecutionError("standalone pointer source v1 requires complete OUT_B64 readback")
        deadline = ExecutionDeadline.start(timeout)
        cb = json.loads(json.dumps(command_buffer))
        cb_hash = hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()
        inputs, binding = bind_pointer_source(cb, source, capture)
        deadline.remaining()
        record["binding"] = binding
        record["oracle_tier_ceiling"] = "L2"
        record["readback_policy"] = readback_policy.record()
        policy = yaml.safe_load((source / "capsule.yaml").read_bytes())["numeric_policy"]
        if numeric_policy is not None and numeric_policy != policy:
            raise NativeModelExecutionError("standalone pointer source cannot change its original numerical policy")
        golden = load_golden(source)
        record["source"] = {
            "capsule_declaration": original._digest(source / "capsule.yaml"),
            "interface": original._digest(capture / "model.mlir"),
            "weights": original._digest(capture / "weights.safetensors"),
            "weight_manifest": original._digest(capture / "weights.safetensors.manifest.json"),
            "inputs": original._digest(capture / "inputs.npz"),
            "golden": original._digest(source / "golden.yaml"),
            **({"golden_arrays": original._digest(source / "golden.npz")} if (source / "golden.npz").is_file() else {}),
        }
        record["frozen_policy"] = original._frozen_model_policy(source, target=target)

        def unchanged():
            deadline.remaining()
            if source_tree(source) != binding["capsule_tree"] or source_tree(capture) != binding["capture_tree"]:
                raise NativeModelExecutionError(
                    "standalone pointer source/reference namespace changed during execution"
                )
            if original._digest(selected_contract) != contract_pin:
                raise NativeModelExecutionError("standalone pointer selected harness contract changed")
            if original._frozen_model_policy(source, target=target) != record["frozen_policy"]:
                raise NativeModelExecutionError("standalone pointer source original instruction/tier policy changed")
            if (
                json.loads(json.dumps(command_buffer)) != cb
                or hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest() != cb_hash
            ):
                raise NativeModelExecutionError("standalone pointer source emitted command buffer changed")

        service = original._build_service_for(target)
        entry = service.recipe.require_kernel_stack_frame().entry_symbol
        require_pointer_entry(
            lowered_mlir_text, entry_symbol=entry, pointer_arity=len(cb["compiler_pointer_abi"]["arguments"])
        )
        record["candidate"] = dict(
            command_buffer_sha256=cb_hash,
            lowered_mlir_sha256=hashlib.sha256(lowered_mlir_text.encode()).hexdigest(),
            entry_arity=len(cb["compiler_pointer_abi"]["arguments"]),
            abi_args=len(cb["compiler_pointer_abi"]["arguments"]),
            entry_symbol=entry,
        )
        from merlin.targetgen.contract.harness_abi import from_contract
        from merlin.targetgen.target_registry import resolve

        selected_contract = resolve(target).capability_contract_path
        contract_pin = original._digest(_plain(selected_contract))
        harness_abi = from_contract(yaml.safe_load(selected_contract.read_bytes()), target=target)
        if harness_abi.entry_symbol != entry:
            raise NativeModelExecutionError("standalone pointer recipe and harness entry differ")
        record["pointer_execution"] = dict(
            protocol=PROTOCOL,
            entry_symbol=entry,
            completion_symbol=harness_abi.fence_symbol,
            selected_contract=contract_pin,
            kernel_stack_frame=service.recipe.require_kernel_stack_frame().record(),
        )
        record["host_build_sources"] = [{"path": p, "sha256": digest} for p, digest in service.source_pins]
        record["transport_implementation"] = [
            original._digest(Path(__file__).resolve()),
            original._digest(Path(pointer_binding.__file__).resolve()),
        ]
        record["rtl_facts"] = selected_firrtl(rtl_facts, target=target, config=board_config)
        unchanged()
        elf = compile_lowered_to_elf(
            cb,
            lowered_mlir_text,
            output / "build",
            target=target,
            inputs=inputs,
            _build_service=service,
            execution_deadline=deadline,
            readback_policy=readback_policy,
        )
        record["elf"] = original._digest(Path(elf))
        record["pointer_execution"]["harness_source"] = original._digest(output / "build/harness.c")
        from .native_readback import verify_build

        recipe_record, build_sources, build_receipt = verify_build(
            target=target, policy=readback_policy, service=service, output=output, cb=cb, elf=elf
        )
        record["readback_build"] = build_receipt
        unchanged()
        service.verify(target)
        backend, citation, revalidate = original._functional_engine(target)
        revalidate()
        console, invocation = observed_execution(
            backend,
            elf,
            simulator="spike",
            citation=citation,
            directory=output,
            timeout=deadline.remaining(),
            capture_kwargs={},
        )
        revalidate()
        service.verify(target)
        unchanged()
        if original._digest(Path(elf)) != record["elf"]:
            raise NativeModelExecutionError("standalone pointer source ELF changed during functional execution")
        verify_build(
            target=target,
            policy=readback_policy,
            service=service,
            output=output,
            cb=cb,
            elf=elf,
            expected=(recipe_record, build_sources),
            expected_receipt=build_receipt,
            stage="L2",
        )
        if original._digest(selected_contract) != contract_pin:
            raise NativeModelExecutionError("standalone pointer selected harness contract changed")
        actual, metrics = backend.parse_output(console)
        require_full_value_roster(cb, console, actual, policy=readback_policy)
        expected_names = {row["compiled_name"] for row in binding["returns"]}
        if set(actual) != expected_names:
            raise NativeModelExecutionError("standalone pointer source did not return its complete exact output roster")
        renamed = {row["source_name"]: actual[row["compiled_name"]] for row in binding["returns"]}
        comparison = compare(
            golden["outputs"], renamed, policy, golden_source=str(golden.get("golden_source") or "independent_capsule")
        )
        console_path = output / "console_l2.txt"
        console_path.write_text(console, encoding="utf-8")
        record["execution_invocations"] = {"L2": invocation}
        record["tiers"] = {
            "L2": dict(
                status="pass" if comparison.get("status") == "pass" else "fail",
                engine="spike",
                engine_citation=citation,
                elf=record["elf"],
                console=original._digest(console_path),
                numeric=comparison,
                metrics=metrics,
                scope="same standalone candidate ELF; all independent original outputs",
            )
        }
        from merlin_experiments.phase2.gsim_workload import OUTPUT_ENCODING, encode_declared_outputs

        digest, tensors = encode_declared_outputs(actual, cb)
        record["output_observation"] = dict(sha256=digest, encoding=OUTPUT_ENCODING, tensors=tensors)
        unchanged()
        return finish_numerical_execution(record)
    except Exception as error:  # always retain a precise refused build/transport attempt
        record["status"] = "incomplete"
        record.pop("numeric", None)
        if record.get("tiers", {}).get("L2", {}).get("status") == "pass":
            record["tiers"]["L2"] = {"status": "unavailable", "detail": "post-run source/build identity check failed"}
        record["failure"] = {"type": type(error).__name__, "detail": str(error)[:2000]}
        return record
    finally:
        try:
            record["build_artifacts"] = original._build_artifacts(output)
            receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        finally:
            _NUMERICS_ONLY.reset(scope)
