"""Reopen current standalone pointer artifacts without inventing a global plan.

L0 binds typed storage and the actual build/functional invocation. Independent
source arithmetic and every accelerator product/effect remain obligations of
its original live owner; deeper required tiers are never inferred here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml
from xdsl.parser import Parser

from merlin.common import invocation_record
from merlin.common.paths import module_source_path
from merlin.runtime.complete_call_timing import _NUMERICS_ONLY
from merlin.targetgen.contract.compile_only import require_pointer_entry
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy
from merlin.targetgen.golden_store import load_golden
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

from . import native_model_execution as original
from .native_model_execution import NativeModelExecutionError
from .native_pointer_source_binding import PROTOCOL, _plain, bind_pointer_source


def _pin(path):
    return original._digest(_plain(path))


def _verify_current(emission, native, *, target, entry_symbol, readback_policy, source_intake_protocol=None):
    if not isinstance(emission, Mapping) or not isinstance(native, Mapping):
        raise NativeModelExecutionError("standalone source audit needs actual emission/native records")
    if (
        native.get("schema") != "merlin_candidate_native_pointer_source_v1"
        or native.get("source_execution_protocol") != PROTOCOL
        or native.get("simulator") != "spike"
        or native.get("oracle_tier_ceiling") != "L2"
        or native.get("failure")
        or native.get("readback_policy") != readback_policy.record()
    ):
        raise NativeModelExecutionError("standalone source audit cannot select a different execution protocol")
    cb = json.loads(original._read_pinned_payload(emission, "command_buffer"))
    lowered = original._read_pinned_payload(emission, "lowered_mlir").decode()
    source = original._read_pinned_payload(emission, "source_interface")
    expected = dict(
        command_buffer_sha256=hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest(),
        lowered_mlir_sha256=hashlib.sha256(lowered.encode()).hexdigest(),
    )
    if any(native.get("candidate", {}).get(key) != value for key, value in expected.items()):
        raise NativeModelExecutionError("standalone source actual build differs from current emitted bytes")
    sources = native.get("source")
    if not isinstance(sources, Mapping):
        raise NativeModelExecutionError("standalone source original source pins are absent")
    for role in sources:
        original._pinned_native_file(sources, role)
    capsule = original._pinned_native_file(sources, "capsule_declaration").parent
    capture_interface = original._pinned_native_file(sources, "interface")
    if capture_interface.read_bytes() != source:
        raise NativeModelExecutionError("standalone native execution used another complete source")
    capture = capture_interface.parent
    intake_options = {"intake_protocol": source_intake_protocol} if source_intake_protocol is not None else {}
    inputs, binding = bind_pointer_source(cb, capsule, capture, **intake_options)
    if native.get("source_intake") != binding.get("intake"):
        raise NativeModelExecutionError("standalone native intake differs from its original host selection")
    if binding != native.get("binding"):
        raise NativeModelExecutionError("standalone native execution storage binding changed")
    current_cap = Path(emission["source_interface"]["path"]).parent
    _, current = bind_pointer_source(cb, current_cap, capture, **intake_options)
    if any(current[key] != binding[key] for key in ("source_sha256", "inputs", "returns")):
        raise NativeModelExecutionError("current standalone source has different original logical storage")
    old, now = (yaml.safe_load((root / "capsule.yaml").read_bytes()) for root in (capsule, current_cap))
    compact = {"compact_arrays": True} if source_intake_protocol is not None else {}
    golden, current_golden = load_golden(capsule, **compact), load_golden(current_cap, **compact)
    if (
        old["numeric_policy"] != now["numeric_policy"]
        or set(golden["outputs"]) != set(current_golden["outputs"])
        or any(not np.array_equal(value, current_golden["outputs"][name]) for name, value in golden["outputs"].items())
    ):
        raise NativeModelExecutionError("current standalone source changed independent full references or policy")
    policy = original._frozen_model_policy(capsule, target=target)
    if native.get("frozen_policy") != policy:
        raise NativeModelExecutionError("standalone source original instruction/tier policy changed")
    for pin in native.get("host_build_sources", ()):
        if type(pin) is not dict or set(pin) != {"path", "sha256"} or _pin(pin["path"])["sha256"] != pin["sha256"]:
            raise NativeModelExecutionError("standalone source original build producer changed")
    if not native.get("host_build_sources"):
        raise NativeModelExecutionError("standalone source build producer closure is absent")
    expected_transport = [
        _pin(module_source_path(name))
        for name in (
            "merlin.targetgen.native_pointer_source_execution",
            "merlin.targetgen.native_pointer_source_binding",
        )
    ]
    if native.get("transport_implementation") != expected_transport:
        raise NativeModelExecutionError("standalone source actual execution implementation changed")
    artifacts = native.get("build_artifacts")
    if not isinstance(artifacts, Mapping) or "harness.c" not in artifacts or not artifacts:
        raise NativeModelExecutionError("standalone source complete build namespace is absent")
    build = original._pinned_native_file(artifacts, "harness.c").parent
    count, size, directories = 0, 0, 0
    for path in build.rglob("*"):
        _plain(path)
        if path.is_file():
            count += 1
            size += path.stat().st_size
            if count > 256 or size > 64 << 20:
                raise NativeModelExecutionError("standalone source complete build audit bound exceeded")
        elif path.is_dir():
            directories += 1
            if directories > 256 or len(path.relative_to(build).parts) > 16:
                raise NativeModelExecutionError("standalone source build directory bound exceeded")
        else:
            raise NativeModelExecutionError("standalone source build namespace has special storage")
    actual_artifacts = {
        path.relative_to(build).as_posix(): _pin(path) for path in sorted(build.rglob("*")) if path.is_file()
    }
    if actual_artifacts != artifacts:
        raise NativeModelExecutionError("standalone source complete build namespace changed")
    elf = original._pinned_native_file(native, "elf")
    if artifacts.get(elf.relative_to(build).as_posix()) != native["elf"]:
        raise NativeModelExecutionError("standalone source original ELF is outside its complete build")
    from merlin_experiments.phase1.feedback.candidate_observations import reopen_native_products

    emission_root = _plain(Path(emission["command_buffer"]["path"]).parent)
    if Path(emission["lowered_mlir"]["path"]).parent != emission_root:
        raise NativeModelExecutionError("standalone emitted leaves have no exact common source directory")
    reopen_native_products(
        build,
        emission=emission,
        emission_root=emission_root,
        elf_pin=native["elf"],
        source=sources,
        build_artifacts=artifacts,
        candidate=native["candidate"],
    )
    service = original._build_service_for(target, **(
        {"source_intake_protocol": source_intake_protocol} if source_intake_protocol is not None else {}
    ))
    selected_entry = service.recipe.require_kernel_stack_frame().entry_symbol
    execution = native.get("pointer_execution") or {}
    if (
        selected_entry != entry_symbol
        or execution.get("entry_symbol") != entry_symbol
        or execution.get("protocol") != PROTOCOL
    ):
        raise NativeModelExecutionError("standalone source selected pointer entry differs")
    from merlin.targetgen.contract.harness_abi import from_contract
    from merlin.targetgen.target_registry import resolve

    contract = _plain(resolve(target).capability_contract_path)
    harness_abi = from_contract(yaml.safe_load(contract.read_bytes()), target=target)
    if (
        execution.get("selected_contract") != _pin(contract)
        or execution.get("completion_symbol") != harness_abi.fence_symbol
        or harness_abi.entry_symbol != entry_symbol
    ):
        raise NativeModelExecutionError("standalone source selected completion/contract differs")
    if execution.get("kernel_stack_frame") != service.recipe.require_kernel_stack_frame().record():
        raise NativeModelExecutionError("standalone source original stack declaration changed")
    require_pointer_entry(
        lowered, entry_symbol=entry_symbol, pointer_arity=len(cb["compiler_pointer_abi"]["arguments"])
    )
    module = Parser(make_llvm_context(), lowered).parse_module()
    module.verify()
    from .native_readback import verify_build

    verify_build(
        target=target,
        policy=readback_policy,
        service=service,
        output=build.parent,
        cb=cb,
        elf=elf,
        expected_receipt=native.get("readback_build"),
        stage="standalone re-audit",
    )
    harness = original._pinned_native_file(artifacts, "harness.c")
    if execution.get("harness_source") != _pin(harness) or harness.read_text() != service.render(
        cb, target=target, inputs=inputs, readback_policy=readback_policy, blobs={}
    ):
        raise NativeModelExecutionError("standalone source actual harness differs from original selected renderer")
    backend, citation, revalidate = original._functional_engine(target)
    revalidate()
    tier = (native.get("tiers") or {}).get("L2") or {}
    if tier.get("engine_citation") != citation or tier.get("elf") != native["elf"]:
        raise NativeModelExecutionError("standalone source functional engine or exact ELF changed")
    invocation_pin = (native.get("execution_invocations") or {}).get("L2")
    if (
        type(invocation_pin) is not dict
        or set(invocation_pin) != {"path", "sha256"}
        or _pin(invocation_pin["path"])["sha256"] != invocation_pin["sha256"]
    ):
        raise NativeModelExecutionError("standalone source original functional invocation is absent or changed")
    invocation = invocation_record.verify(_plain(invocation_pin["path"]))
    callback = backend.run_elf
    args = invocation.get("arguments") or {}
    seconds = args.get("timeout")
    console = original._pinned_native_file(tier, "console")
    if (
        invocation.get("kind") != "python_call"
        or invocation.get("stage") != "execution"
        or invocation.get("outcome") != "returned"
        or invocation.get("callable") != callback.__module__ + "." + callback.__qualname__
        or set(args) != {"simulator", "timeout"}
        or args["simulator"] != "spike"
        or type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or not 0 < seconds <= 600
        or invocation.get("inputs") != [{key: native["elf"][key] for key in ("path", "sha256")}]
        or invocation["stdout"]["sha256"] != _pin(console)["sha256"]
        or Path(invocation["stdout"]["path"]).read_bytes() != console.read_bytes()
    ):
        raise NativeModelExecutionError("standalone source console/producer is not its original same-ELF dispatch")
    revalidate()
    return cb, module, policy, binding


def audit_pointer_source_tiers(emission, native, *, target, entry_symbol, readback_policy=None,
                               source_intake_protocol=None):
    """Recompute original required tiers under the explicit L2 pointer protocol."""
    from .native_instruction_audit import audit_native_instruction_tier
    from .native_numerical_audit import audit_native_numerical_tier

    tiers, required = {}, None
    scope = _NUMERICS_ONLY.set(True)
    try:
        if readback_policy is None:
            readback_policy = ReadbackPolicy(FULL_VALUES_B64)
        if type(readback_policy) is not ReadbackPolicy or readback_policy.transport != FULL_VALUES_B64:
            raise NativeModelExecutionError("standalone source audit needs its explicit complete B64 transport")
        cb, module, policy, binding = _verify_current(
            emission, native, target=target, entry_symbol=entry_symbol, readback_policy=readback_policy,
            source_intake_protocol=source_intake_protocol,
        )
        required = policy["required_tiers"]
        tiers["L0"] = dict(
            status="pass",
            scope=(
                "exact standalone typed storage/pointer ABI, actual IR/build/function-call custody; "
                "source/product effects require original live owner"
            ),
            elf=native["elf"],
        )
        tiers["L1"] = audit_native_instruction_tier(native, target=target, module=module, policy=policy)

        def output_values(current_cb, current_native, golden, observed):
            if current_cb != cb or current_native is not native:
                raise NativeModelExecutionError("standalone numerical re-audit changed current artifacts")
            cap = original._pinned_native_file(native["source"], "capsule_declaration").parent
            capture = original._pinned_native_file(native["source"], "interface").parent
            _, fresh = bind_pointer_source(cb, cap, capture, **(
                {"intake_protocol": source_intake_protocol} if source_intake_protocol is not None else {}
            ))
            if fresh != binding or set(observed) != {r["compiled_name"] for r in fresh["returns"]}:
                raise NativeModelExecutionError("standalone numerical re-audit changed original complete storage")
            return {r["source_name"]: observed[r["compiled_name"]] for r in fresh["returns"]}

        tiers["L2"] = audit_native_numerical_tier(
            emission, native, target=target, readback_policy=readback_policy, output_values=output_values,
            **({"source_intake_protocol": source_intake_protocol} if source_intake_protocol is not None else {}),
        )
        _verify_current(emission, native, target=target, entry_symbol=entry_symbol, readback_policy=readback_policy,
                        source_intake_protocol=source_intake_protocol)
    except Exception as error:
        tiers = {
            tier: dict(status="unverified", detail=f"{type(error).__name__}: {error}") for tier in ("L0", "L1", "L2")
        }
    finally:
        _NUMERICS_ONLY.reset(scope)
    tiers["L3"] = dict(status="pending", scope="outside explicit L2 pointer diagnostic ceiling")
    missing = None if required is None else [tier for tier in required if tiers.get(tier, {}).get("status") != "pass"]
    failed = [] if required is None else [tier for tier in required if tiers.get(tier, {}).get("status") == "fail"]
    return dict(
        status="fail" if failed else "pass" if missing == [] else "unverified",
        required_tiers=required,
        missing_tiers=missing,
        failed_tiers=failed,
        tiers=tiers,
        source_effect_correspondence="requires original live realization owner",
        hardware_execution_qualified=False,
        timing_authority="NONE",
        scope="explicit standalone pointer diagnostic; deeper original required tiers pending",
    )
