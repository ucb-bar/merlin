"""Public source-bound model build using the coordinator's fixed host provider.

This adapter never consumes a candidate whole-program LLVM artifact. Existing
model gates must independently inspect its ordinary device/source/native proof;
a completed build or numeric match does not authorize a tier by itself.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from merlin.compiler.build.owned_sources import OwnedHostModelCompiler
from merlin.compiler.frontend.inputs import MaterializedModelInputs
from merlin.compiler.lower.device.source_compute_groups import form_exact_groups
from merlin.development.targetgen.execution.host.dispatch_checks import (
    CandidateDispatchRefusal,
    _composition_preferences,
    _functional_engine_for_facts,
    _host_call_proof,
    _independently_eligible_group_ids,
    _llvm_function,
    _mandatory_source_commands,
    _native_engine_for_facts,
    _optimized_command_functs,
    _pinned,
    _require_instruction_classes,
    _resolve_weight_preparation,
    _same_owned_frozen_policy,
    _selected_facts,
    _selected_file,
    _stopped_before_receipt,
    _untagged_assert_comparisons,
    _verify_device_custody,
)
from merlin.development.targetgen.execution.host.selections import _numeric_full_output
from merlin.development.targetgen.execution.host.selections import group_profile as _group_profile
from merlin.runtime.execution.model_inputs import file_sha256

# ---- owned_host_model_execution ----------------------------------------------------------------


def _dispatch_proof(record, *, target, facts, evidence_root=None):
    """Original-source calls -> actual candidate kernel -> object -> same ELF."""
    import yaml
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.parser import Parser

    from merlin.common.target_data import interface as facts_interface
    from merlin.compiler.ir.llvm_context import make_llvm_context
    from merlin.compiler.lower.device.build import executable_kernel_entry
    from merlin.development.targetgen.execution.native.dispatch_accounting import (
        _kernel_command_inventory,
        _verified_work_functs_by_family,
    )
    from merlin.development.targetgen.execution.native.native_model_execution import _digest
    from merlin.development.targetgen.grading.trace_check import instruction_class_coverage
    from merlin.target.contract.interfaces.abi import contract_dir
    from merlin.target.isa.rocc.decode import decode_module
    from merlin.target.rtl.facts import observed_facts

    built = record["build"]["output"]
    elf = _pinned(
        {"elf": {"path": built["elf"], "sha256": built["elf_sha256"], "size_bytes": Path(built["elf"]).stat().st_size}},
        "elf",
    )
    artifact_record = built.get("device_artifacts")
    if not isinstance(artifact_record, dict) or artifact_record.get("target") != target:
        raise ValueError("actual device build symbol/artifact binding is absent")
    source = record["source_device_join"]
    assigned = source["assigned"]
    composition = source.get("composition")
    if composition is not None:
        from merlin.development.targetgen.execution.native.private_dispatch import composition_members

        composition_members(composition, assigned=assigned)
    kernels = artifact_record.get("kernels")
    if not isinstance(kernels, dict) or set(kernels) != {row["symbol"] for row in assigned}:
        raise ValueError("actual device kernel roster differs from source-bound calls")
    table = facts_interface(json.loads(Path(facts["path"]).read_text()).get("facts"), "funct_decode_table") or {}
    endpoint_spec = yaml.safe_load((contract_dir() / "compute_endpoints.yaml").read_text())
    endpoints = [
        row
        for row in (endpoint_spec.get("endpoints") or {}).values()
        if isinstance(row, dict) and row.get("target") == target and row.get("exposure") == "rocc"
    ]
    work = _verified_work_functs_by_family(endpoints, table, work_roles_only=True)
    names = {int(key) for key in (table.get("names") or {})}
    entry_symbol = executable_kernel_entry(target)
    completed, decoded_classes, artifact_pins = [], set(), {}
    class_observations = {}
    for row in assigned:
        symbol = row["symbol"]
        entry = kernels[symbol]
        pins = entry.get("artifacts")
        kernel_symbol = entry.get("kernel_symbol")
        if not isinstance(kernel_symbol, str) or not kernel_symbol or not isinstance(pins, dict):
            raise ValueError("actual device kernel has no exact symbol binding")
        artifact = _pinned(pins, "llvm_mlir")
        _pinned(pins, "llvm_ir")
        obj = _pinned(pins, "object")
        from merlin.development.targetgen.execution.native.private_dispatch import component_for_symbol

        component = component_for_symbol(composition, symbol, group=row["group"])
        optimized_artifact = _verify_device_custody(
            entry,
            artifact=artifact,
            llvm_ir=_pinned(pins, "llvm_ir"),
            obj=obj,
            original_symbol=entry_symbol,
            kernel_symbol=kernel_symbol,
            source_composition=component,
        )
        module = Parser(make_llvm_context(), artifact.read_text()).parse_module()
        function = _llvm_function(module, entry_symbol)
        with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
            trace = decode_module(module, target=target)
            coverage = instruction_class_coverage(trace, target=target)
        class_observations[symbol] = coverage
        decoded_classes.update(coverage["observed_classes"])
        opcode = (trace.get("abi") or {}).get("custom_opcode")
        if not isinstance(opcode, str) or int(opcode, 16) != table.get("custom_opcode"):
            raise ValueError("device LLVM decoder differs from selected RTL opcode")
        asm = [op for op in module.walk() if op.name == "llvm.inline_asm"]
        instructions = trace.get("instructions")
        if not isinstance(instructions, list) or len(asm) != len(instructions):
            raise ValueError("actual device LLVM has incomplete structural decode")
        commands, all_functs = {}, []
        for op, decoded in zip(asm, instructions, strict=True):
            if decoded.get("class") == "UNKNOWN":
                raise ValueError("actual device LLVM contains an unknown command")
            funct = decoded.get("funct")
            if funct is None:
                continue
            if type(funct) is not int or funct not in names:
                raise ValueError("actual device command has no selected ISA identity")
            all_functs.append(funct)
            for family, functs in work.items():
                if funct not in functs:
                    continue
                if not isinstance(getattr(op, "has_side_effects", None), UnitAttr) or "~{memory}" not in getattr(
                    getattr(op, "constraints", None), "data", ""
                ).split(","):
                    raise ValueError("actual device work lacks side effect or memory clobber")
                commands.setdefault(family, set()).add(op)
        if component is None:
            joined = source["joined_groups"] if composition is None else composition["ordinary_joined_groups"]
            source_group = joined[row["group"]]
            obligations = [
                item for item in source["placement"]["eligible_source_regions"] if item["source_group"] == source_group
            ]
            completed.extend(_mandatory_source_commands(function, commands, obligations))
        elif component.get("schema") == "merlin.original_head_family.v1":
            from merlin.development.targetgen.execution.native.private_dispatch import (
                mandatory_heads,
                verify_bridge,
            )

            obligations = [
                r
                for r in source["placement"]["eligible_source_regions"]
                if r["source_group"] in component["covered_source_groups"]
            ]
            bridge = verify_bridge(entry, component, built=built, target=target, collection=composition)
            completed.extend(
                mandatory_heads(function, commands, obligations, component, mandatory=_mandatory_source_commands)
            )
        else:
            from merlin.development.targetgen.execution.native.private_dispatch import (
                mandatory_member_commands,
                verify_descriptor_bridge,
            )

            obligations = [
                r
                for r in source["placement"]["eligible_source_regions"]
                if r["source_group"] in component["covered_source_groups"]
            ]
            completed.extend(
                mandatory_member_commands(
                    function, commands, obligations, component["members"], mandatory=_mandatory_source_commands
                )
            )
            bridge = verify_descriptor_bridge(entry, component, built=built, target=target, collection=composition)
        inventory_functs = all_functs
        if optimized_artifact is not None:
            optimized = Parser(make_llvm_context(), optimized_artifact.read_text()).parse_module()
            _llvm_function(optimized, entry_symbol)
            with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
                optimized_trace = decode_module(optimized, target=target)
            inventory_functs = _optimized_command_functs(optimized, optimized_trace, table, source_functs=all_functs)
        object_words, object_functs = _kernel_command_inventory(obj, symbol=kernel_symbol, table=table)
        elf_words, elf_functs = _kernel_command_inventory(elf, symbol=kernel_symbol, table=table)
        if (
            Counter(inventory_functs) != Counter(object_functs)
            or Counter(object_words) != Counter(elf_words)
            or Counter(object_functs) != Counter(elf_functs)
        ):
            inventory = {
                "schema": "merlin.device_command_inventory_refusal.v1",
                "group": row["group"],
                "symbol": symbol,
                "kernel_symbol": kernel_symbol,
                "llvm_funct_counts": dict(Counter(all_functs)),
                "object_input_llvm_funct_counts": dict(Counter(inventory_functs)),
                "object_funct_counts": dict(Counter(object_functs)),
                "elf_funct_counts": dict(Counter(elf_functs)),
                "object_and_elf_words_equal": Counter(object_words) == Counter(elf_words),
                "scope": "static inventories; optimizer transformations are not a dispatch proof",
            }
            raw = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode()
            from hashlib import sha256

            diagnostic = (elf.parent if evidence_root is None else evidence_root) / (
                f"{symbol}.command-inventory-refusal-{sha256(raw).hexdigest()}.json"
            )
            try:
                with diagnostic.open("xb") as stream:
                    stream.write(raw)
            except FileExistsError:
                if diagnostic.is_symlink() or diagnostic.read_bytes() != raw:
                    raise ValueError("device command inventory diagnostic changed")
            stage = (
                "device LLVM to object"
                if Counter(inventory_functs) != Counter(object_functs)
                else "device object to linked ELF"
            )
            raise ValueError(
                "complete custom-command inventory changed across device LLVM/object/ELF; "
                f"group {row['group']} ({symbol}), first mismatch: {stage}; "
                "static counts cannot yet certify this compiler transformation"
            )
        artifact_pins[symbol] = {
            "kernel_symbol": kernel_symbol,
            **{role: _digest(_pinned(pins, role)) for role in pins},
        }
        if component is not None:
            artifact_pins[symbol]["descriptor_bridge"] = bridge
    eligible = source["placement"]["eligible_source_op_indices"]
    if sorted(row["source_op_index"] for row in completed) != sorted(eligible):
        raise ValueError("mandatory owned-device work does not cover the complete eligible source roster")
    host = _pinned(built.get("host_artifacts") or {}, "llvm_ir")
    _pinned(built.get("host_artifacts") or {}, "object")
    host_calls = _host_call_proof(
        host,
        assigned=assigned,
        output=elf.parent if evidence_root is None else evidence_root,
        source_composition=composition,
    )
    return {
        "status": "verified",
        "source_sha256": source["placement"]["source_sha256"],
        "linked_elf": _digest(elf),
        "eligible_tasks": completed,
        "device_artifacts": artifact_pins,
        "host_calls": host_calls,
        "observed_classes": sorted(decoded_classes),
        "instruction_class_observations": class_observations,
        "scope": "completion-implied unavoidable source-bound work; "
        "not hardware commit trace or general transform theorem",
    }


def _operation_profile(record, console):
    from merlin.development.targetgen.execution.host.selections import operation_profile

    return operation_profile(record, console, selected_file=_selected_file)


def _static_tiers(capsule, record, *, target, facts, dispatch, policy):
    """Original-source closure and whole-ELF ISA; neither grants numerics."""
    from merlin.analysis.whole_model.grading.isa_prohibition import scan_elf
    from merlin.common.target_data import interface as facts_interface
    from merlin.compiler.lower.model import toolchain
    from merlin.compiler.lower.model.compilation_recipe import verify_completed_recipe
    from merlin.development.targetgen.execution.native.native_model_execution import _digest
    from merlin.development.targetgen.grading import elf_lanes
    from merlin.runtime.backends.baremetal.spike_launcher import gcc_path
    from merlin.runtime.boards.catalog import load_boards
    from merlin.target.rtl.facts import observed_facts

    output = record["build"]["output"]
    inputs = record["build"]["inputs"]
    board = load_boards(Path(inputs["board_catalog"])).get(inputs["board"])
    if board is None or board.target != target:
        raise ValueError("owned model selected board and RTL facts identify different configurations")
    from merlin.runtime.execution.model_inputs import selected_firrtl

    selected_firrtl(facts["path"], target=target, config=board.rtl_sim_config)
    elf = Path(output["elf"])
    recipe = output["compilation_recipe"]
    compilation = verify_completed_recipe(Path(recipe["path"]), executable=elf, expected_recipe_sha256=recipe["sha256"])
    quantization = (compilation.get("preparation") or {}).get("quantization_policy")
    if quantization is not None:
        document = json.loads(Path(quantization["path"]).read_text())
        if document.get("selection") != "captured_scope" or document.get("passes") != ["contraction_int8"]:
            raise ValueError("owned host lowering did not preserve captured quantization scope")
    host = output["host_artifacts"]
    model_ir, model_obj = _pinned(host, "llvm_ir"), _pinned(host, "object")
    model_commands = [row for row in compilation["commands"] if row["output"]["path"] == str(model_obj)]
    if (
        len(model_commands) != 1
        or "-c" not in model_commands[0]["argv"]
        or Path(model_commands[0]["executable"]["path"]) != toolchain.clang().resolve()
        or not any(
            row["path"] == str(model_ir) and row["sha256"] == file_sha256(model_ir)
            for row in model_commands[0]["inputs"]
        )
    ):
        raise ValueError("owned host CFG is not the actual model object compiler input")
    linked_inputs = {(row["path"], row["sha256"]) for row in compilation["commands"][-1]["inputs"]}
    if (str(model_obj), file_sha256(model_obj)) not in linked_inputs:
        raise ValueError("owned host object was not an actual final link input")
    if Path(compilation["commands"][-1]["executable"]["path"]) != gcc_path().resolve():
        raise ValueError("owned model final link did not use the selected compiler")
    for entry in output["device_artifacts"]["kernels"].values():
        obj = _pinned(entry["artifacts"], "object")
        if (str(obj), file_sha256(obj)) not in linked_inputs:
            raise ValueError("source-bound device object was not an actual final link input")
    if dispatch.get("status") != "verified" or dispatch["linked_elf"] != _digest(elf):
        raise ValueError("owned model has no source-bound actual ELF structure")
    facts_document = json.loads(Path(facts["path"]).read_text())
    table = facts_interface(facts_document.get("facts"), "funct_decode_table") or {}
    names = {int(key) for key in (table.get("names") or {})}
    opcode = table.get("custom_opcode")
    if type(opcode) is not int or not names:
        raise ValueError("selected RTL has no complete custom-command ISA table")
    blob = elf.read_bytes()
    sections = elf_lanes.executable_sections(blob)
    if not sections:
        raise ValueError("owned model ELF has no executable sections")
    count = 0
    for _name, offset, size, address in sections:
        for _at, word in elf_lanes.instruction_words(blob[offset : offset + size], address):
            if word & 0x7F == opcode:
                count += 1
                if word >> 25 not in names:
                    raise ValueError("owned model ELF contains an unknown selected custom selector")
    _require_instruction_classes(policy["expected_instruction_classes"], dispatch["observed_classes"])
    with observed_facts(target, facts_document, path=Path(facts["path"])):
        if policy["prohibited_instruction_roles"]:
            prohibition = scan_elf(elf, target=target, roles=policy["prohibited_instruction_roles"])
            if prohibition.get("status") != "measured" or prohibition.get("clean") is not True:
                raise ValueError("owned model ELF has prohibited or unmeasured instruction roles")
        else:
            prohibition = {
                "status": "not_applicable",
                "clean": True,
                "reason": "frozen reviewed policy explicitly prohibits no roles",
            }
    return {
        "L0": {
            "status": "pass",
            "elf": _digest(elf),
            "source_sha256": dispatch["source_sha256"],
            "scope": "original source closure, fixed host lowering, built device host veto "
            "and mandatory call/command CFG",
        },
        "L1": {
            "status": "pass",
            "elf": _digest(elf),
            "rtl_facts": facts,
            "whole_elf_known_custom_commands": count,
            "prohibition": prohibition,
            "observed_classes": sorted(set(dispatch["observed_classes"])),
            "required_classes": policy["expected_instruction_classes"],
        },
    }


def _run_selected_functional(backend, elf, *, timeout):
    """Apply the selected process ceiling to the owned functional transport."""
    from merlin.common.process.ownership import ExecutionDeadline
    from merlin.development.targetgen.execution.runner.compile import _selected_run_timeout

    bounded = _selected_run_timeout(backend, "spike", timeout, ExecutionDeadline.start(timeout))
    declared_runner = getattr(backend, "run_declared_model_elf", None)
    if callable(declared_runner):
        return declared_runner(elf, timeout=bounded, capture_bytes=True)
    return backend.run_elf(elf, simulator="spike", timeout=bounded, capture_bytes=True)


def _run_selected_rtl(elf, *, target, engine, facts, timeout, elf_sha256):
    """Execute under explicitly protected facts without changing global env.

    Ambient release facts can have another materialization identity. The
    backend instead binds the actual selected file; exact byte, FIRRTL and
    engine checks remain intact rather than equating different receipts.
    """
    import time
    from contextlib import nullcontext

    from merlin.common.process.ownership import ExecutionDeadline
    from merlin.development.targetgen.execution.runner.compile import _selected_run_timeout
    from merlin.runtime.execution.model_inputs import selected_firrtl
    from merlin.target.simulators.engine_policy import gsim_runtime_slot

    backend, citation, revalidate, prepare = _native_engine_for_facts(target, engine, facts)
    command = (
        prepare(elf, expected_elf_sha256=elf_sha256, expected_engine_provenance=citation) if callable(prepare) else None
    )
    evidence = {"engine": engine, "engine_citation": citation, "rtl_facts_identity": facts}
    if command is not None:
        evidence["native_command_sha256"] = command.revalidate()["command_sha256"]
        evidence["max_cycles"] = command.to_evidence().get("max_cycles")
    revalidate()
    started = time.monotonic()
    bounded = _selected_run_timeout(backend, engine, timeout, ExecutionDeadline.start(timeout))
    deadline = ExecutionDeadline.start(bounded)
    with gsim_runtime_slot(wait_timeout_s=bounded) if engine == "gsim" else nullcontext():
        console = backend.run_elf(elf, simulator=engine, timeout=deadline.remaining(), capture_bytes=True)
    if command is not None:
        command.revalidate()
    revalidate()
    if selected_firrtl(facts["path"], target=target, config=facts["config"]) != facts:
        raise ValueError("protected owned-model RTL facts changed during execution")
    return {**evidence, "console": console, "wall_s": round(time.monotonic() - started, 1)}


def _execute_required_tiers(capsule, record, *, target, facts, timeout, reachable_oracle_tiers=None):
    """Build once, execute the exact same ELF on every frozen required engine."""
    import os

    from merlin.development.targetgen.execution.native.native_model_execution import (
        _digest,
        _frozen_model_policy,
    )
    from merlin.target.simulators.oracle_policy import selected_l3_engine_report

    policy = _frozen_model_policy(Path(capsule["__dir__"]), target=target, source_bound_groups=True)
    selected_facts = _selected_facts(facts, target=target)
    dispatch = _dispatch_proof(record, target=target, facts=selected_facts)
    tiers = _static_tiers(capsule, record, target=target, facts=selected_facts, dispatch=dispatch, policy=policy)
    output = record["build"]["output"]
    elf = Path(output["elf"])
    identity = _digest(elf)
    for tier in policy["required_tiers"]:
        if tier in {"L0", "L1"}:
            continue
        if reachable_oracle_tiers is not None and tier not in reachable_oracle_tiers:
            tiers[tier] = {
                "status": "unavailable",
                "elf": identity,
                "detail": "required owned-model tier is not selected in this grading request",
            }
            continue
        directory = elf.parent.parent / f"owned_{tier.lower()}"
        directory.mkdir()
        try:
            if tier == "L2":
                backend, citation, revalidate = _functional_engine_for_facts(target, selected_facts)
                revalidate()
                console = _run_selected_functional(backend, elf, timeout=timeout)
                revalidate()
                engine_evidence = {"engine": "spike", "engine_citation": citation}
            elif tier == "L3":
                selection = selected_l3_engine_report(target)
                engine = selection.get("engine")
                required_engine = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
                if (
                    not selection.get("available")
                    or not isinstance(engine, str)
                    or (required_engine and engine != required_engine)
                ):
                    raise ValueError(
                        "owned model selected mandatory RTL engine is unavailable or differs from invocation"
                    )
                run = _run_selected_rtl(
                    elf,
                    engine=engine,
                    target=target,
                    timeout=timeout,
                    facts=selected_facts,
                    elf_sha256=identity["sha256"],
                )
                console = run.pop("console")
                engine_evidence = {**run, "selection": selection}
                fidelity = selection.get("fidelity")
                if isinstance(fidelity, str) and fidelity:
                    from merlin.target.simulators.engine_policy import ELABORATED_RTL

                    engine_evidence.update(
                        fidelity=fidelity, derived_from_rtl=fidelity == ELABORATED_RTL, cycle_accurate=False
                    )
                # Reopen exactly the selected backend, not just an L3 label.
                _, citation, revalidate, _ = _native_engine_for_facts(target, engine, selected_facts)
                revalidate()
                if engine_evidence.get("engine_citation") != citation:
                    raise ValueError("owned model RTL run cited another engine")
            else:
                raise ValueError("owned model has an unsupported frozen mandatory tier")
            if not isinstance(console, bytes):
                raise ValueError("owned full binary readback did not preserve console bytes")
            path = directory / "console.bin"
            path.write_bytes(console)
            numeric, metrics = _numeric_full_output(capsule, record, console)
            profile = _operation_profile(record, console)
            group_profile = _group_profile(record, console)
            if _digest(elf) != identity:
                raise ValueError("owned model ELF changed during required-tier execution")
            tiers[tier] = {
                "status": numeric["status"],
                **engine_evidence,
                "elf": identity,
                "console": _digest(path),
                "numeric": numeric,
                "metrics": metrics,
                **({"operation_profile": profile} if profile is not None else {}),
                **({"group_profile": group_profile} if group_profile is not None else {}),
            }
        except Exception as exc:  # noqa: BLE001 -- no alternate-engine or host numeric substitution
            tiers[tier] = {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}", "elf": identity}
    missing = [tier for tier in policy["required_tiers"] if tiers[tier]["status"] != "pass"]
    failed = [tier for tier in policy["required_tiers"] if tiers[tier]["status"] == "fail"]
    preferred = tiers.get("L3") or tiers.get("L2") or {}
    return {
        "status": "fail" if failed else "verified" if not missing else "unverified",
        "dispatch": dispatch,
        "tiers": tiers,
        "rtl_facts": selected_facts,
        "frozen_policy": policy,
        "required_tiers": policy["required_tiers"],
        "numeric": preferred.get("numeric"),
        "missing_tiers": missing,
        "failed_tiers": failed,
    }


def audit_owned_host_model(result, capsule, *, target, readback_policy=None, descriptor_effects=None):
    """Independently reopen the provided-host result before promoting any tier."""
    import os

    import yaml

    from merlin.development.targetgen.execution.host.model_execution import (
        _audit_owned_host_device_build,
        _frozen_split,
    )
    from merlin.development.targetgen.execution.native.native_model_execution import _frozen_model_policy
    from merlin.evaluation.capsule.models.bundle import _model_runtime_bundle
    from merlin.evaluation.capsule.models.runtime import _resolve_model_host_lane
    from merlin.runtime.execution.model_inputs import strict_tree_sha256
    from merlin.target.contract.harness.readback_policy import selected
    from merlin.target.simulators.oracle_policy import selected_l3_engine_report

    try:
        record = result.get("owned_host_native_execution")
        stopped = result.get("candidate_emission_failure")
        if not isinstance(record, dict) and isinstance(stopped, dict):
            return _stopped_before_receipt(stopped)
        if not isinstance(record, dict) or not isinstance(capsule, dict):
            raise ValueError("owned host model receipt or original frozen capsule is absent")
        root = Path(capsule["__dir__"])
        declaration = yaml.safe_load((root / "capsule.yaml").read_text())
        capsule = {**declaration, "__dir__": str(root)}
        policy = selected(readback_policy)
        if record.get("readback_policy") != (policy.record() if policy is not None else None):
            raise ValueError("owned model readback differs from trusted invocation selection")
        selection_path = _selected_file({"selection": record["selection_record"]}, "selection")
        selection = json.loads(selection_path.read_text())
        paths = {
            role: _selected_file(selection, role)
            for role in ("board_catalog", "dts", "instruction_declaration", "rtl_facts")
        }
        attrs = (capsule.get("operation") or {}).get("attributes") or {}
        _, host, identity = _resolve_model_host_lane(target, attrs.get("compile_dtype", "fp32"))
        if identity != record.get("host_provider"):
            raise ValueError("owned host provider differs from run-selected provider")
        inputs = record["build"]["inputs"]
        if (
            inputs.get("package") != str(Path(host).resolve())
            or inputs.get("package_tree") != strict_tree_sha256(host)
            or inputs.get("target") != target
            or inputs.get("board") != selection["board"]
            or inputs.get("arena_mb") != selection["arena_mb"]
            or inputs.get("run") != "none"
            or inputs.get("readback") != "full"
            or inputs.get("respect_captured_quantization_scope") is not True
            or inputs.get("board_catalog_sha256") != file_sha256(paths["board_catalog"])
            or inputs.get("dts_sha256") != file_sha256(paths["dts"])
        ):
            raise ValueError("owned model build used different selected host/board inputs")
        frozen_policy = _frozen_model_policy(Path(capsule["__dir__"]), target=target, source_bound_groups=True)
        if (
            not _same_owned_frozen_policy(record.get("frozen_policy"), frozen_policy)
            or record.get("required_tiers") != frozen_policy["required_tiers"]
        ):
            raise ValueError("owned model changed the frozen mandatory tier or ISA policy")
        # A saved result cannot select its own inputs or goldens by editing a
        # materialization record. Replay the existing frozen-source validator.
        root = Path(record["runtime_inputs"])
        with _model_runtime_bundle(capsule, timeout=inputs["timeout_s"], target=target) as (fresh, provenance, verify):
            bound = MaterializedModelInputs.bind(
                root,
                artifacts=provenance["bundle"]["artifacts"],
                validator_identity=provenance["tool"],
                source_identity=provenance["source"],
            )
            if bound.record() != record.get("materialization"):
                raise ValueError("owned model materialization differs from frozen source validator")
            verify()
        source = _audit_owned_host_device_build(
            root,
            record["build"],
            target=target,
            capsule=capsule,
            certificate=result.get("candidate_source_eligibility"),
            composition_selection=record.get("source_composition_selection"),
            descriptor_effects=descriptor_effects,
        )
        if record.get("frozen_placement") != _frozen_split(capsule, root / "model.mlir", target=target):
            raise ValueError("owned model split differs from the sealed capsule's frozen placement")
        fresh_record = {**record, "source_device_join": source}
        facts = _selected_facts(paths["rtl_facts"], target=target)
        if record.get("rtl_facts") != facts:
            raise ValueError("owned model RTL facts differ from selected input bytes")
        dispatch = _dispatch_proof(fresh_record, target=target, facts=facts)
        tiers = _static_tiers(
            capsule, fresh_record, target=target, facts=facts, dispatch=dispatch, policy=frozen_policy
        )
        elf = dispatch["linked_elf"]
        for tier in frozen_policy["required_tiers"]:
            if tier in {"L0", "L1"}:
                continue
            evidence = (record.get("tiers") or {}).get(tier)
            if tier == "L3" and (
                evidence is None
                or (isinstance(evidence, dict) and evidence.get("status") in {"unavailable", "unverified", "not_run"})
            ):
                tiers[tier] = {"status": "unverified", "detail": "selected native tier has not completed"}
                continue
            if (
                not isinstance(evidence, dict)
                or evidence.get("status") not in {"pass", "fail"}
                or evidence.get("elf") != elf
            ):
                raise ValueError(f"owned model {tier} has no completed exact-ELF run")
            console = _pinned(evidence, "console")
            if tier == "L2":
                _, citation, revalidate = _functional_engine_for_facts(target, facts)
                if evidence.get("engine") != "spike" or evidence.get("engine_citation") != citation:
                    raise ValueError("owned model L2 engine/extension bytes changed")
                revalidate()
            elif tier == "L3":
                selected_engine = selected_l3_engine_report(target)
                required_engine = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
                engine = selected_engine.get("engine")
                if (
                    not selected_engine.get("available")
                    or evidence.get("engine") != engine
                    or evidence.get("selection") != selected_engine
                    or (required_engine and required_engine != engine)
                ):
                    raise ValueError("owned model L3 used another selected RTL engine")
                _, citation, revalidate, _ = _native_engine_for_facts(target, engine, facts)
                if evidence.get("engine_citation") != citation or evidence.get("rtl_facts_identity") != facts:
                    raise ValueError("owned model L3 engine/FIRRTL lineage changed")
                revalidate()
            numeric, metrics = _numeric_full_output(capsule, fresh_record, console.read_bytes())
            profile = _operation_profile(fresh_record, console.read_bytes())
            group_profile = _group_profile(fresh_record, console.read_bytes())
            if (
                numeric != evidence.get("numeric")
                or evidence["status"] != numeric["status"]
                or metrics != evidence.get("metrics")
                or profile != evidence.get("operation_profile")
                or group_profile != evidence.get("group_profile")
            ):
                raise ValueError(f"owned model {tier} receipt differs from parsed full values")
            tiers[tier] = {**evidence, "numeric": numeric, "dispatch": dispatch}
            if tier == "L2":
                # Assigned only after exact functional engine and full-output re-audit.
                tiers[tier].update(fidelity="functional_model", derived_from_rtl=False, cycle_accurate=False)
            if tier == "L3":
                fidelity = selected_engine.get("fidelity")
                if isinstance(fidelity, str) and fidelity:
                    from merlin.target.simulators.engine_policy import ELABORATED_RTL

                    tiers[tier].update(
                        fidelity=fidelity, derived_from_rtl=fidelity == ELABORATED_RTL, cycle_accurate=False
                    )
        required = frozen_policy["required_tiers"]
        failed = [tier for tier in required if tiers[tier]["status"] == "fail"]
        pending = [tier for tier in required if tiers[tier]["status"] == "unverified"]
        numeric = next((tiers[tier]["numeric"] for tier in ("L3", "L2") if "numeric" in tiers.get(tier, {})), None)
        coverage = {
            "status": "verified",
            "source_sha256": dispatch["source_sha256"],
            "n_source_operations": source["placement"]["n_source_operations"],
            "eligible_source_op_indices": source["placement"]["eligible_source_op_indices"],
            "completed_eligible_source_op_indices": sorted(
                row["source_op_index"] for row in dispatch["eligible_tasks"]
            ),
            "n_eligible": len(dispatch["eligible_tasks"]),
            "n_completed_eligible": len(dispatch["eligible_tasks"]),
        }
        from merlin.development.backend.feedback.guidance.composition_brief import (
            build as composition_feedback,
        )

        composition_summary = composition_feedback(fresh_record, tiers=tiers)
        return {
            "status": "fail" if failed else "incomplete" if pending else "pass",
            "kind": "owned_host_model_execution",
            "violations": ["candidate_verified_numeric_mismatch"] if failed else [],
            "owned_host_model_check": {
                "source_placement": source["placement"],
                "completed_dispatch": dispatch,
                "candidate_source_coverage": coverage,
                "source_composition": composition_summary,
            },
            "candidate_required_tiers": {
                "status": "fail" if failed else "incomplete" if pending else "pass",
                "required_tiers": required,
                "failed_tiers": failed,
                "pending_tiers": pending,
                "tiers": tiers,
            },
            "candidate_completed_dispatch": dispatch,
            "candidate_source_placement": source["placement"],
            "candidate_source_coverage": coverage,
            "numeric": numeric,
            "violations_scope": "provided Merlin host; candidate closed device kernels only",
        }
    except Exception as exc:  # noqa: BLE001 -- no partial structure or diagnostic native credit
        return {
            "status": "incomplete",
            "kind": "owned_host_model_execution",
            "violations": ["candidate_completed_dispatch_unverified", "candidate_required_tiers_unverified"],
            "detail": f"{type(exc).__name__}: {exc}",
        }
