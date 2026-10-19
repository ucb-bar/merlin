"""Read-only completed native-dispatch accounting for exact candidate artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .native_model_execution import NativeModelExecutionError, _digest, _pinned_native_file, _read_pinned_payload


def _host_compute_report(command_buffer: Mapping[str, Any], lowered_mlir_text: str, *, entry_symbol: str):
    """Audit only task-scoped host IR of the emitted entry, or report why it is unverified.

    The independent route-quality analyzer owns the dataflow decision. Parsing
    may be narrower than upstream MLIR's LLVM metadata vocabulary; that never
    becomes a clean report merely because the executable build accepts it.
    """
    from merlin.runtime.route_quality import host_compute

    try:
        from xdsl.parser import Parser

        from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

        # LLVM metadata that xDSL does not model (for example loop-unroll
        # hints) must not hide a task-scoped function that upstream MLIR can
        # compile. Unregistered operations are still handled conservatively
        # by route_quality, which marks unknown dataflow incomplete.
        context = make_llvm_context()
        module = Parser(context, lowered_mlir_text).parse_module()
        matches = [
            op
            for op in module.body.block.ops
            if op.name == "llvm.func" and getattr(op.properties.get("sym_name"), "data", None) == entry_symbol
        ]
        if len(matches) != 1:
            raise NativeModelExecutionError(
                f"candidate LLVM has {len(matches)} parsed entry function(s) named {entry_symbol!r}"
            )
        return host_compute(command_buffer, function=matches[0])
    except Exception as exc:  # noqa: BLE001 -- no parse/attribution is not a clean audit
        return host_compute(
            command_buffer,
            function=None,
            absent_cause=f"candidate host IR is not task-scoped/auditable: {type(exc).__name__}: {str(exc)[:350]}",
        )


def _mandatory_command_blocks(function: Any, commands: set[Any]) -> bool:
    """A return cannot be reached after removing blocks carrying real work commands.

    This proves one command issues on every *returning* path, including a loop's
    zero-trip edge. Nontermination and equivalence to source computation are
    separate obligations; neither is inferred from this CFG property.
    """
    from merlin.perf.host_cfg_index import prepare_host_cfg

    cfg = prepare_host_cfg(function)
    carrying = {block for block in cfg.blocks if any(op in commands for op in block.ops)}
    if not carrying or not cfg.blocks:
        return False
    pending = [cfg.blocks[0]]
    reached: set[Any] = set()
    while pending:
        block = pending.pop()
        if block in reached or block in carrying:
            continue
        reached.add(block)
        if block.last_op is not None and block.last_op.name == "llvm.return":
            return False
        pending.extend(cfg.successors[block])
    return True


def _kernel_command_inventory(path: Path, *, symbol: str, table: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Read only the named kernel symbol, never decoy opcodes in the harness."""
    import subprocess

    from merlin.kernels.decode.objdump import tokenize_text, word_of
    from merlin.kernels.decode.rocc import decode_stream, fields_of
    from merlin.llvmlower import toolchain
    from merlin.perf.compiled_static_features import linked_entry_extent

    image = path.read_bytes()
    try:
        extent = linked_entry_extent(image, symbol)
        start, size = extent["address"], extent["size_bytes"]
        if type(start) is not int or start < 0 or type(size) is not int or size <= 0:
            raise ValueError("function has no positive declared extent")
    except (ValueError, KeyError, TypeError) as exc:
        raise NativeModelExecutionError(f"no verified {symbol!r} function extent: {exc}") from exc

    disassembler = toolchain.objdump()
    if not disassembler.is_file() or disassembler.is_symlink():
        raise NativeModelExecutionError("selected compiler has no matching LLVM objdump")
    # --disassemble-symbols stops at interior local labels in relocatable
    # objects (e.g. PC-relative relocation labels), truncating genuine work.
    # Decode the independently verified owner section and complete extent.
    run = subprocess.run(
        [
            str(disassembler), "-d", "--disassemble-zeroes", "--triple=riscv64", "-M", "no-aliases",
            f"--section={extent['section']}", f"--start-address={start}", f"--stop-address={start + size}",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if run.returncode:
        raise NativeModelExecutionError(f"symbol-scoped disassembly failed: {run.stderr[-350:]}")
    if path.read_bytes() != image:
        raise NativeModelExecutionError("device image changed during symbol inventory")
    rows = [row for row in tokenize_text(run.stdout) if start <= row.addr < start + size]
    # The disassembler may print a same-address body/local alias instead of the
    # selected public symbol. Ownership comes from the verified ELF extent,
    # never that display label. Every byte in that extent must still be present
    # exactly once; decoys, omitted rows and truncated instructions cannot pass.
    cursor = start
    for row in rows:
        encoded = row.hexcode.replace(" ", "")
        width = len(encoded) // 2
        if not width or len(encoded) % 2 or row.addr != cursor:
            raise NativeModelExecutionError("incomplete or overlapping selected function instruction coverage")
        if row.addr + width > start + size:
            raise NativeModelExecutionError("decoded instruction crosses the selected function extent")
        cursor += width
    if cursor != start + size:
        raise NativeModelExecutionError("incomplete selected function instruction coverage")
    decoded = decode_stream(rows, table)
    opcode = table.get("custom_opcode")
    if type(opcode) is not int:
        raise NativeModelExecutionError("selected target has no derived accelerator opcode")
    all_custom: list[int] = []
    known: list[int] = []
    for row, instruction in zip(rows, decoded, strict=True):
        word = word_of(row.hexcode, width_bits=32)
        if word is None:
            continue
        fields = fields_of(word)
        if fields["opcode"] == opcode:
            all_custom.append(word)
            if instruction.from_endpoint:
                known.append(fields["funct"])
    if len(all_custom) != len(known):
        raise NativeModelExecutionError(f"{path.name} has an unrecognized custom command in {symbol}")
    return all_custom, known


_SOURCE_FAMILY_DEVICE_ROLES = {
    "contraction": ("accumulate",),
    "convolution": ("accumulate",),
    "elementwise_map": ("elementwise",),
    "movement": ("move", "operand_load", "readout", "dma"),
}


def _verified_work_functs_by_family(
    endpoints: list[Mapping[str, Any]], table: Mapping[str, Any], *, work_roles_only: bool = False
) -> dict[str, set[int]]:
    """Bind trusted source families to selected RTL-confirmed work roles.

    Config, wait, flush and generic loop-descriptor setup are never credited.
    Fused epilogues need an independent candidate-transform witness, not a
    broad role mapping that mistakes configuration for arithmetic.
    """
    if len(endpoints) != 1 or endpoints[0].get("engine") != "spatial" or endpoints[0].get("exposure") != "rocc":
        raise NativeModelExecutionError("selected target has no unique verified spatial endpoint")
    names = {int(key): value for key, value in (table.get("names") or {}).items()}
    roles = endpoints[0].get("roles") or {}
    checked_roles = roles
    if work_roles_only and isinstance(roles, Mapping):
        accepted_roles = {
            role for roles_for_family in _SOURCE_FAMILY_DEVICE_ROLES.values() for role in roles_for_family
        }
        checked_roles = {role: declared for role, declared in roles.items() if role in accepted_roles}
    if not isinstance(checked_roles, Mapping) or any(
        not isinstance(declared, list) or not declared or any(name not in names.values() for name in declared)
        for declared in checked_roles.values()
    ):
        raise NativeModelExecutionError("endpoint roles are not confirmed by selected RTL decode table")
    return {
        family: {funct for funct, name in names.items() if any(name in roles.get(role, ()) for role in accepted)}
        for family, accepted in _SOURCE_FAMILY_DEVICE_ROLES.items()
    }


def _completed_eligible_tasks(
    function: Any,
    placement: Mapping[str, Any],
    task_commands: Mapping[int, Mapping[str, set[Any]]],
    task_region_commands: Mapping[int, Mapping[str, set[Any]]],
    task_op_commands: Mapping[int, Mapping[int, set[Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Require a mandatory family-matched work command per eligible region."""
    eligible = set(placement["eligible_source_op_indices"])
    eligible_regions = {row["source_op_index"]: row["region_id"] for row in placement["eligible_source_regions"]}
    eligible_families = {row["source_op_index"]: row["semantic_family"] for row in placement["eligible_source_regions"]}
    op_commands = task_op_commands or {}
    completed: list[dict[str, Any]] = []
    for task in placement["task_source_regions"]:
        indices = task["source_op_indices"]
        task_eligible = sorted(eligible.intersection(indices))
        if not task_eligible:
            continue  # genuinely unsupported host glue is not an accelerator obligation
        ident = task["task_index"]
        by_family = task_commands.get(ident, {})
        commands = set().union(*(by_family.get(eligible_families[index], set()) for index in task_eligible))
        for family in {eligible_families[index] for index in task_eligible}:
            if not _mandatory_command_blocks(function, by_family.get(family, set())):
                raise NativeModelExecutionError(
                    f"eligible source task {ident} has no unavoidable {family} device command"
                )
        if len(task_eligible) > 1:
            # A repeated region ID can contain multiple linalg roots. One
            # command may not credit all of them: use exact source-op identity
            # when present, or region identity only when unique in this group.
            group_regions = [eligible_regions[index] for index in task_eligible]
            for index in task_eligible:
                rid = eligible_regions[index]
                attributed = op_commands.get(ident, {}).get(index, set())
                if not attributed and group_regions.count(rid) == 1:
                    attributed = task_region_commands[ident].get(rid, set())
                matched = attributed & by_family.get(eligible_families[index], set())
                if not _mandatory_command_blocks(function, matched):
                    raise NativeModelExecutionError(
                        f"eligible source operation {index} in region {rid} lacks an unavoidable "
                        "family-matched attributed command; fused groups need independent "
                        "candidate-transform evidence"
                    )
        else:
            named = task_region_commands[ident]
            if named and set(named) != {eligible_regions[task_eligible[0]]}:
                raise NativeModelExecutionError(f"eligible source task {ident} commands name a different source region")
        completed.append(
            {
                "task_index": ident,
                "source_op_indices": sorted(indices),
                "eligible_source_region_ids": [eligible_regions[index] for index in task_eligible],
                "mandatory_device_work_commands": len(commands),
            }
        )
    if not completed:
        raise NativeModelExecutionError("no independently eligible source task was joined")
    return completed


def _audit_candidate_completed_dispatch(
    emission: Mapping[str, Any] | None,
    certificate: Mapping[str, Any] | None,
    native: Mapping[str, Any] | None,
    *,
    target: str,
    entry_symbol: str,
    source_placement: Callable[..., dict[str, Any]],
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
    try:
        from merlin.targetgen.contract.readback_policy import read_console, selected

        readback_policy = selected(readback_policy)
        if not isinstance(emission, Mapping) or not isinstance(native, Mapping):
            raise NativeModelExecutionError("candidate emission or native execution is absent")
        placement = source_placement(emission, certificate, target=target)
        if placement.get("status") != "clean":
            raise NativeModelExecutionError("independent source placement is not clean")
        if (
            native.get("status") not in {"numeric_match_diagnostic", "numeric_mismatch_diagnostic"}
            or native.get("simulator") is None
        ):
            raise NativeModelExecutionError("exact candidate ELF has no completed native oracle run")
        source_text = _read_pinned_payload(emission, "source_interface").decode("utf-8")
        lowered_text = _read_pinned_payload(emission, "lowered_mlir").decode("utf-8")
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        candidate = native.get("candidate") or {}
        if (
            candidate.get("lowered_mlir_sha256") != hashlib.sha256(lowered_text.encode()).hexdigest()
            or candidate.get("command_buffer_sha256")
            != hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()
        ):
            raise NativeModelExecutionError("native build is not bound to exact candidate bytes")

        import yaml
        from xdsl.parser import Parser

        from merlin.common.facts_view import interface as facts_interface
        from merlin.compile.model_execution_inputs import native_engine, selected_firrtl
        from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan
        from merlin.targetgen.contract.schemas import contract_dir
        from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
        from merlin.targetgen.oracle_policy import selected_l3_engine_report
        from merlin.targetgen.rocc.decode import decode_module

        context = make_llvm_context()
        module = Parser(context, lowered_text).parse_module()
        functions = [
            op
            for op in module.body.block.ops
            if op.name == "llvm.func" and getattr(op.properties.get("sym_name"), "data", None) == entry_symbol
        ]
        if len(functions) != 1:
            raise NativeModelExecutionError("emitted LLVM has no unique selected entry function")
        plan = verify_compiler_global_plan(
            source_text=source_text,
            lowered_text=lowered_text,
            command_buffer=cb,
            candidate_sha256=hashlib.sha256(lowered_text.encode()).hexdigest(),
            parsed_lowered_module=module,
        )
        if plan.get("status") != "verified" or (plan.get("control_flow") or {}).get("status") != "verified":
            raise NativeModelExecutionError(
                "independent source/ABI/task CFG proof is not verified: "
                + str(plan.get("problems") or plan.get("reason"))
            )
        selected = native.get("rtl_facts") or {}
        if (
            not isinstance(selected, Mapping)
            or selected.get("target") != target
            or not isinstance(selected.get("config"), str)
            or not isinstance(selected.get("path"), str)
            or selected_firrtl(selected["path"], target=target, config=selected["config"]) != selected
        ):
            raise NativeModelExecutionError("native RTL fact lineage changed or is absent")
        if native.get("simulator") == "spike" and native.get("oracle_tier_ceiling") == "L2":
            from .native_model_execution import _functional_engine

            _, citation, revalidate_engine = _functional_engine(target)
        else:
            selection = selected_l3_engine_report(target)
            if (
                not selection.get("available")
                or selection.get("engine") != native["simulator"]
                or (native.get("simulator_provenance") or {}).get("selection") != selection
            ):
                raise NativeModelExecutionError("native run is not on the selected mandatory RTL engine")
            _, citation, revalidate_engine, _ = native_engine(target, native["simulator"], dict(selected))
        if (native.get("simulator_provenance") or {}).get("citation") != citation:
            raise NativeModelExecutionError("native run cites different selected engine bytes")
        revalidate_engine()
        selected_facts_path = Path(selected["path"])
        facts = json.loads(selected_facts_path.read_text(encoding="utf-8"))
        table = facts_interface(facts.get("facts"), "funct_decode_table") or {}
        spec = yaml.safe_load((contract_dir() / "compute_endpoints.yaml").read_text())
        endpoints = [
            row
            for row in (spec.get("endpoints") or {}).values()
            if isinstance(row, Mapping) and row.get("target") == target and row.get("exposure") == "rocc"
        ]
        names = {int(key): value for key, value in (table.get("names") or {}).items()}
        from .native_model_execution import _frozen_model_policy

        public_zero_device_demand = (
            _frozen_model_policy(
                _pinned_native_file(native["source"], "capsule_declaration").parent, target=target
            ).get("public_zero_device_demand")
            is True
            and placement.get("eligible_source_op_indices") == []
        )
        work_functs = {} if public_zero_device_demand else _verified_work_functs_by_family(endpoints, table)
        trace = decode_module(module, target=target)
        trace_opcode = (trace.get("abi") or {}).get("custom_opcode")
        if (
            type(table.get("custom_opcode")) is not int
            or not isinstance(trace_opcode, str)
            or int(trace_opcode, 16) != table["custom_opcode"]
        ):
            raise NativeModelExecutionError("selected RTL opcode differs from host decoder")
        asm = [op for op in module.walk() if op.name == "llvm.inline_asm"]
        instructions = trace.get("instructions")
        if not isinstance(instructions, list) or len(instructions) != len(asm):
            raise NativeModelExecutionError("emitted inline assembly has incomplete structural decode")
        tasks = ((cb.get("params") or {}).get("global_program_plan") or {}).get("tasks") or []
        task_commands: dict[int, dict[str, set[Any]]] = {row["task_index"]: {} for row in tasks}
        task_region_commands: dict[int, dict[str, set[Any]]] = {row["task_index"]: {} for row in tasks}
        task_op_commands: dict[int, dict[int, set[Any]]] = {row["task_index"]: {} for row in tasks}
        allowed_regions = {row["region_id"] for row in placement["eligible_source_regions"]}
        region_by_index = {row["source_op_index"]: row["region_id"] for row in placement["eligible_source_regions"]}
        all_functs: list[int] = []
        for operation, decoded in zip(asm, instructions, strict=True):
            funct = decoded.get("funct")
            if funct is None:
                if decoded.get("class") == "UNKNOWN":
                    raise NativeModelExecutionError("unrecognized inline assembly in candidate kernel")
                continue
            if type(funct) is not int or funct not in names:
                raise NativeModelExecutionError("candidate command lacks a derived ISA identity")
            all_functs.append(funct)
            families = [family for family, functs in work_functs.items() if funct in functs]
            if not families:
                continue
            if getattr(operation, "has_side_effects", None) is None:
                raise NativeModelExecutionError("device-work inline asm is not side-effecting")
            constraints = getattr(getattr(operation, "constraints", None), "data", "")
            if "~{memory}" not in constraints.split(","):
                raise NativeModelExecutionError("device-work inline asm lacks memory-ordering clobber")
            attr = operation.attributes.get("merlin.global_task")
            owner = getattr(getattr(attr, "value", None), "data", None)
            if type(owner) is not int or owner not in task_commands:
                raise NativeModelExecutionError("device-work command has no verified task owner")
            for family in families:
                task_commands[owner].setdefault(family, set()).add(operation)
            region = getattr(operation.attributes.get("prov.region_id"), "data", None)
            source_index_attr = operation.attributes.get("merlin.source_op_index")
            source_index = getattr(getattr(source_index_attr, "value", None), "data", None)
            if source_index_attr is not None:
                if (
                    type(source_index) is not int
                    or source_index not in region_by_index
                    or source_index not in tasks[owner]["source_op_indices"]
                ):
                    raise NativeModelExecutionError("device-work command names an absent source operation")
                if region is not None and region != region_by_index[source_index]:
                    raise NativeModelExecutionError("device-work command source index/region disagree")
                task_op_commands[owner].setdefault(source_index, set()).add(operation)
            if region is not None:
                if region not in allowed_regions:
                    raise NativeModelExecutionError("device-work command names an absent source region")
                task_region_commands[owner].setdefault(region, set()).add(operation)
        if public_zero_device_demand:
            completed = []
        else:
            completed = _completed_eligible_tasks(
                functions[0], placement, task_commands, task_region_commands, task_op_commands
            )

        artifacts = native.get("build_artifacts")
        if not isinstance(artifacts, Mapping):
            raise NativeModelExecutionError("native build artifact pins are absent")
        for name in ("kernel.llvm.mlir", "kernel.ll", "kernel.o", "kernel.stack_frame.json", "kernel.abi.json"):
            _pinned_native_file(artifacts, name)
        if _pinned_native_file(artifacts, "kernel.llvm.mlir").read_bytes() != lowered_text.encode():
            raise NativeModelExecutionError("translated object used different candidate LLVM bytes")
        stack = json.loads(_pinned_native_file(artifacts, "kernel.stack_frame.json").read_text())
        if stack.get("status") != "passed" or stack.get("repair") is not None:
            raise NativeModelExecutionError("object transform needs separate command-preservation proof")
        obj = _pinned_native_file(artifacts, "kernel.o")
        if (
            stack.get("entry_symbol") != entry_symbol
            or stack.get("llvm_ir_sha256") != _digest(_pinned_native_file(artifacts, "kernel.ll"))["sha256"]
            or stack.get("object_sha256") != _digest(obj)["sha256"]
        ):
            raise NativeModelExecutionError("kernel object is not bound to translated LLVM and entry")
        abi = json.loads(_pinned_native_file(artifacts, "kernel.abi.json").read_text())
        if abi.get("object_sha256") != _digest(obj)["sha256"]:
            raise NativeModelExecutionError("ABI receipt names a different kernel object")
        host_sources = native.get("host_build_sources")
        if not isinstance(host_sources, list) or not host_sources:
            raise NativeModelExecutionError("trusted harness renderer source closure is absent")
        for row in host_sources:
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("path"), str)
                or not isinstance(row.get("sha256"), str)
            ):
                raise NativeModelExecutionError("trusted harness renderer source pin is malformed")
            host_path = Path(row["path"])
            if (
                not host_path.is_absolute()
                or host_path.is_symlink()
                or not host_path.is_file()
                or _digest(host_path)["sha256"] != row["sha256"]
            ):
                raise NativeModelExecutionError("trusted harness renderer source changed")
        elf = _pinned_native_file(native, "elf")
        object_words, object_functs = _kernel_command_inventory(obj, symbol=entry_symbol, table=table)
        elf_words, elf_functs = _kernel_command_inventory(elf, symbol=entry_symbol, table=table)
        if (
            Counter(all_functs) != Counter(object_functs)
            or Counter(object_words) != Counter(elf_words)
            or Counter(object_functs) != Counter(elf_functs)
        ):
            raise NativeModelExecutionError("task command inventory changed across LLVM/object/linked ELF")
        console = read_console(_pinned_native_file(native, "console"), policy=readback_policy)
        from merlin.runtime.backends import base as backends
        from merlin.runtime.backends.base import get_backend
        from merlin.runtime.commandbuffer import declared_output_dtypes
        from merlin.targetgen.capsule_golden import compare
        from merlin.targetgen.golden_store import load_golden

        sources = native.get("source") or {}
        # The model bundle is deliberately temporary and is removed when the
        # runner leaves its context. Reopen *stable frozen capsule assets* at
        # grade time; the build-time source pins for generated inputs/manifest
        # remain in the trusted execution receipt, but their deleted paths
        # cannot be misread as durable proof.
        for name in ("capsule_declaration", "golden"):
            _pinned_native_file(sources, name)
        capsule_dir = _pinned_native_file(sources, "capsule_declaration").parent
        if (capsule_dir / "golden.npz").is_file() != ("golden_arrays" in sources):
            raise NativeModelExecutionError("native golden archive presence differs from source pins")
        if "golden_arrays" in sources:
            _pinned_native_file(sources, "golden_arrays")
        declaration = yaml.safe_load((capsule_dir / "capsule.yaml").read_text(encoding="utf-8"))
        attrs = (declaration.get("operation") or {}).get("attributes") or {}
        for role, relative in (
            ("interface", declaration.get("interface_mlir")),
            ("weights", attrs.get("weights")),
            ("weight_manifest", attrs.get("weights_manifest")),
        ):
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise NativeModelExecutionError(f"frozen {role} has no capsule-local path")
            frozen = capsule_dir / relative
            if (
                frozen.is_symlink()
                or not frozen.is_file()
                or _digest(frozen)["sha256"] != (sources.get(role) or {}).get("sha256")
            ):
                raise NativeModelExecutionError(f"frozen {role} differs from bound runtime bundle")
        if (
            _digest(capsule_dir / declaration["interface_mlir"])["sha256"]
            != hashlib.sha256(source_text.encode()).hexdigest()
        ):
            raise NativeModelExecutionError("candidate source differs from frozen model interface")
        golden = load_golden(capsule_dir)
        observed, _ = get_backend(target).parse_output(console)
        if readback_policy is not None:
            from merlin.targetgen.contract.readback_policy import require_full_value_roster

            require_full_value_roster(cb, console, observed, policy=readback_policy)
        observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
        if set(observed) != set(golden["outputs"]):
            raise NativeModelExecutionError("completed candidate console has no complete declared full output")
        numeric = compare(
            golden["outputs"],
            observed,
            declaration["numeric_policy"],
            golden_source=str(golden.get("golden_source") or "independent_capsule"),
        )
        if (
            numeric.get("status") not in {"pass", "fail"}
            or numeric != native.get("numeric")
            or native.get("status")
            != ("numeric_match_diagnostic" if numeric["status"] == "pass" else "numeric_mismatch_diagnostic")
        ):
            raise NativeModelExecutionError("native numeric receipt differs from independently parsed full output")
        from merlin.llvmlower import toolchain

        return {
            "status": "verified",
            "scope": "completion-implied mandatory family-matched "
            "device-work command issue; "
            "not hardware commit trace or source-transform semantic equivalence",
            "source_sha256": placement["source_sha256"],
            "lowered_mlir_sha256": candidate["lowered_mlir_sha256"],
            "kernel_object": _digest(obj),
            "linked_elf": _digest(elf),
            "console": _digest(_pinned_native_file(native, "console")),
            "numeric_status": numeric["status"],
            "eligible_tasks": completed,
            **(
                {
                    "public_zero_device_demand": True,
                    "scope": "complete public source/task CFG, actual object/ELF and full numerical output; "
                    "zero mandatory accelerator obligations; optional legal device lowering "
                    "retains ISA and output gates",
                }
                if public_zero_device_demand
                else {}
            ),
            "kernel_command_count": len(all_functs),
            "plan_digest": plan.get("plan_digest"),
            "disassembler": _digest(toolchain.objdump()),
        }
    except Exception as exc:  # noqa: BLE001 -- no partial dispatch credit
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
