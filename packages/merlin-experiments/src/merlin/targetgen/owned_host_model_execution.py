"""Public source-bound model build using the coordinator's fixed host provider.

This adapter never consumes a candidate whole-program LLVM artifact. Existing
model gates must independently inspect its ordinary device/source/native proof;
a completed build or numeric match does not authorize a tier by itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from merlin.compile.materialized_model_inputs import MaterializedModelInputs
from merlin.compile.model_execution_inputs import file_sha256
from merlin.compile.owned_host_model import OwnedHostModelCompiler
from merlin.llvmlower.source_compute_groups import form_exact_groups


class CandidateDispatchRefusal(ValueError):
    """The candidate's own emitted program fails a source-attribution requirement of its dispatch.

    Raised only from checks of the candidate's emitted device code, never for a harness, input or
    tool fault, so the grade can report the compiler's failure as a failure rather than leaving it
    behind the absent receipt it causes (which reads as unmeasured tooling).
    """


def _selected_file(selection, role):
    record = selection.get(role)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"owned host build needs exact {role} selection")
    path = Path(record["path"])
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"owned host build {role} is indirect")
    if not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError(f"owned host build {role} changed")
    return path


def execute_owned_host_model(
    *,
    capsule,
    candidate_package,
    target,
    bundle,
    provenance,
    verify,
    output,
    selection_record,
    simulator,
    timeout,
    source_certificate=None,
    readback_policy=None,
    reachable_oracle_tiers=None,
):
    from merlin_experiments.phase1.feedback import private_instruction_coordinator as coordinator
    from merlin_experiments.phase1.feedback import private_instruction_declaration as declaration

    from merlin.targetgen.capsule_runner import _resolve_model_host_lane

    if not isinstance(selection_record, dict):
        raise ValueError("public model needs coordinator-selected owned host build inputs")
    selection_path = _selected_file({"selection": selection_record}, "selection")
    selection = json.loads(selection_path.read_text())
    if set(selection) != {"board_catalog", "dts", "board", "arena_mb", "instruction_declaration", "rtl_facts"}:
        raise ValueError("owned host build selection fields are incomplete or unexpected")
    catalog = _selected_file(selection, "board_catalog")
    dts = _selected_file(selection, "dts")
    instruction = _selected_file(selection, "instruction_declaration")
    facts = _selected_file(selection, "rtl_facts")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("owned host model needs a fresh output directory")
    from merlin.common.artifacts import new_product

    product = new_product(
        "owned-host-model",
        version=1,
        target=target,
        update_latest=False,
        sources=[str(output)],
        notes="Owned-host model build and source-bound native evidence custody",
    )
    output = product.path
    from merlin.targetgen.contract.readback_policy import FULL_VALUES_BIN, selected

    readback_policy = selected(readback_policy)
    if readback_policy is not None and readback_policy.transport != FULL_VALUES_BIN:
        raise ValueError("owned saved-model compiler needs its selected full binary value transport")
    prepared = coordinator.prepare(
        declaration.read(instruction, target=target),
        target=target,
        output=(output / "instruction_selection").absolute(),
    )
    attrs = (capsule.get("operation") or {}).get("attributes") or {}
    experiment, host, identity = _resolve_model_host_lane(target, attrs.get("compile_dtype", "fp32"))
    service = OwnedHostModelCompiler.select(
        host_package=host, target=target, board_catalog=catalog, board=selection["board"], dts=dts
    )
    verify()
    # Keep the validated bytes for independent durable-result auditing. This
    # copies the trusted materialization, never a candidate-supplied capture.
    retained = output / "runtime_inputs"
    shutil.copytree(bundle, retained)
    bundle = retained
    materialized = MaterializedModelInputs.bind(
        bundle,
        artifacts=provenance["bundle"]["artifacts"],
        validator_identity=provenance["tool"],
        source_identity=provenance["source"],
    )
    # Validate each required source operation before invoking the candidate, including its actual
    # group assignment. Supporting fusion is selected by Merlin's ordinary capability checks.
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as groups_api

    source_path = Path(bundle) / "model.mlir"
    module = mq.parse(source_path.read_text())
    groups, _ = form_exact_groups(module, target, bundle, groups_api.form_groups)
    frozen_split = _source_placement(
        capsule, source_certificate, module, groups, target=target, source_path=source_path
    )["frozen_placement"]
    receipt = service.compile(
        capture=bundle,
        candidate_package=candidate_package,
        datapath=attrs.get("dtype"),
        linked_elf_admission=prepared.service,
        output=output / "model",
        arena_mb=selection["arena_mb"],
        timeout_s=timeout,
        run="none",
        rtl_facts=facts,
        materialized_inputs=materialized,
    )
    verify()
    audit = _audit_owned_host_device_build(
        bundle, receipt, target=target, capsule=capsule, certificate=source_certificate
    )
    prepared.verify()
    for role in ("board_catalog", "dts", "instruction_declaration", "rtl_facts"):
        _selected_file(selection, role)
    _selected_file({"selection": selection_record}, "selection")
    record = {
        "status": "compiled_not_run",
        "scope": "Merlin-owned host model with candidate closed device groups; "
        "mandatory source/device/tier gates remain independent",
        "host_provider": identity,
        "build": receipt,
        "source_device_join": audit,
        "frozen_placement": frozen_split,
        "instruction_selection": prepared.record(),
        "materialization": materialized.record(),
        "selection_record": selection_record,
        "runtime_inputs": str(retained.absolute()),
        "readback_policy": readback_policy.record() if readback_policy is not None else None,
    }
    record.update(
        _execute_required_tiers(
            capsule,
            record,
            target=target,
            facts=facts,
            timeout=timeout,
            reachable_oracle_tiers=reachable_oracle_tiers,
        )
    )
    verify()
    materialized.verify(bundle)
    prepared.verify()
    _selected_file({"selection": selection_record}, "selection")
    (output / "qualification.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    for artifact in ("qualification.json", "model", "runtime_inputs", "instruction_selection"):
        product.add_artifact(artifact)
    product.write_manifest()
    return record


def _frozen_declaration(capsule):
    """The pinned capsule's own declaration, never an in-memory copy a caller could have edited."""
    import yaml

    root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((root / "capsule.yaml").read_text())
    if not isinstance(declaration, dict):
        raise ValueError("frozen model capsule declaration is malformed")
    return declaration, root


def _frozen_split(capsule, source_path, *, target, groups=None, eligible=None, module=None):
    """Bind Merlin's device groups (and, when given, fresh eligibility) to the frozen placement."""
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as groups_api

    from . import frozen_placement

    declaration, root = _frozen_declaration(capsule)
    frozen = frozen_placement.read(declaration, root)
    if module is None:
        module = mq.parse(Path(source_path).read_text())
    if groups is None:
        groups, _ = form_exact_groups(module, target, source_path, groups_api.form_groups)
    return frozen_placement.require_split(
        frozen,
        source_sha256=file_sha256(source_path),
        device=frozen_placement.device_operations(groups, groups_api.HOST),
        eligible=eligible,
        source_operations=Counter(frozen_placement.operation_name(op) for op in mq.walk(module)),
    )


def _independently_eligible_group_ids(placement, groups):
    """Keep every group containing proven eligible work, regardless of its root."""
    rows = placement.get("eligible_source_regions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("owned source lacks independently eligible group roster")
    selected = set()
    for row in rows:
        index = row.get("source_group") if isinstance(row, dict) else None
        if type(index) is not int or index < 0:
            raise ValueError("owned source eligible group identity is invalid")
        selected.add(index)
    available = {group.index for group in groups}
    if not selected <= available:
        raise ValueError("owned source eligible group is absent from original groups")
    return selected


def _audit_owned_host_device_build(bundle, receipt, *, target, capsule=None, certificate=None):
    """Recompute original group identities and audit the actual linked objects."""
    from merlin_experiments.phase1.feedback.private_device_audit import audit_built_device_host_compute
    from merlin_experiments.phase1.feedback.private_full_models import _linked_symbols
    from merlin_experiments.phase1.feedback.private_group_provenance import (
        convolution_input_preparations,
        join_routed_source_groups,
        require_intended_host_selection,
        routed_kernel_symbols,
        source_group_roster,
    )

    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as groups_api

    # Match the exact source passed to OwnedHostModelCompiler.plan_before_build.
    text = (Path(bundle) / "model.mlir").read_text()
    module = mq.parse(text)
    groups, _ = form_exact_groups(module, target, bundle, groups_api.form_groups)
    groups_api.require_explained(groups_api.plan(module, target, groups=groups))
    placement = _source_placement(
        capsule, certificate, module, groups, target=target, source_path=Path(bundle) / "model.mlir"
    )
    eligible_groups = _independently_eligible_group_ids(placement, groups)
    source = source_group_roster(groups, eligible_groups, host_placement=groups_api.HOST, module=module)
    output = receipt["output"]
    sidecar = output["device_sidecar"]
    path = Path(sidecar["path"])
    if file_sha256(path) != sidecar["sha256"]:
        raise ValueError("owned host device sidecar changed")
    emitted = json.loads(path.read_text())
    if emitted.get("device") != target:
        raise ValueError("owned host model omitted eligible device groups")
    assigned = emitted.get("routed")
    joined = join_routed_source_groups(assigned, source)
    require_intended_host_selection(emitted.get("skipped", []))
    elf = Path(output["elf"])
    if file_sha256(elf) != output["elf_sha256"]:
        raise ValueError("owned host linked ELF changed")
    _linked_symbols(elf, routed_kernel_symbols(emitted))
    source["conv_input_preparation"] = convolution_input_preparations(emitted, source)
    audits = audit_built_device_host_compute(
        elf.parent, assigned, source, target, diagnostics=elf.parent / "audit-diagnostics",
        host_artifacts=output.get("host_artifacts"),
    )
    return {
        "source": source,
        "placement": placement,
        "joined_groups": joined,
        "static_device_host_compute": audits,
        "assigned": assigned,
        "optional_source_groups_routed": sorted(set(joined.values()) & set(source["optional_groups"])),
        "scope": "source identities joined to actual linked device objects; "
        "dynamic execution and required tiers remain separate",
    }


def _source_placement(capsule, certificate, module, groups, *, target, source_path):
    """Join original source operations to Merlin's independently selected groups."""
    import yaml

    from merlin.common.quant_formats import get as quant_format
    from merlin.xdsl_dialects.lowering import compute_groups as groups_api

    from .native_model_execution import independent_frozen_source_eligibility

    root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((root / "capsule.yaml").read_text())
    fresh = independent_frozen_source_eligibility({**declaration, "__dir__": str(root)}, target=target)
    if certificate is not None and certificate != fresh:
        raise ValueError("owned source certificate differs from frozen source")
    if fresh["source_mlir_sha256"] != file_sha256(source_path):
        raise ValueError("owned host routed another source")
    if (
        fresh.get("n_unknown_capture_formats") != 0
        or fresh.get("n_precision_transform_obligations") != 0
        or (fresh.get("precision_transform_verification") or {}).get("status") != "not_required"
    ):
        raise ValueError("owned source has unverified captured precision obligations")
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("owned source has no unique single-block model entry")
    ops = [op for op in functions[0].body.block.ops if op.name != "func.return"]

    def name(op):
        return op.op_name.data if op.name == "builtin.unregistered" else op.name

    indexed = [(i, op) for i, op in enumerate(ops) if name(op) != "linalg.fill"]
    rows = fresh.get("regions")
    if not isinstance(rows, list) or len(rows) != len(indexed) or fresh.get("n_regions") != len(rows):
        raise ValueError("owned source lacks complete operation eligibility")
    owner = {id(member): group for group in groups for member in group.members}
    eligible = []
    for (index, op), row in zip(indexed, rows, strict=True):
        rid = getattr(op.attributes.get("prov.region_id"), "data", None)
        if (
            type(row.get("target_eligible")) is not bool
            or row.get("region_id") != rid
            or row.get("carrier_op") != name(op)
        ):
            raise ValueError("owned source eligibility names another operation")
        if not row["target_eligible"]:
            continue
        for observed, admitted in ((row.get("captured_input_format"), row.get("eligibility_input_format")),):
            if (
                not isinstance(observed, str)
                or not isinstance(admitted, str)
                or quant_format(observed).name != quant_format(admitted).name
            ):
                raise ValueError("eligible owned source lacks observed input-format admission")
        if (
            row.get("semantic_family") in {"contraction", "convolution"}
            or row.get("eligibility_weight_format") is not None
        ):
            observed, admitted = row.get("captured_weight_format"), row.get("eligibility_weight_format")
            if (
                not isinstance(observed, str)
                or not isinstance(admitted, str)
                or quant_format(observed).name != quant_format(admitted).name
            ):
                raise ValueError("eligible owned source lacks observed weight-format admission")
        group = owner.get(id(op))
        if group is None or group.placement == groups_api.HOST or not isinstance(rid, str) or not rid:
            raise ValueError("eligible owned source was omitted or assigned to host")
        if not isinstance(row.get("semantic_family"), str) or not row["semantic_family"]:
            raise ValueError("eligible owned source has no independent semantic family")
        eligible.append(
            {
                "source_op_index": index,
                "region_id": rid,
                "semantic_family": row["semantic_family"],
                "source_group": group.index,
            }
        )
    if not eligible:
        raise ValueError("accelerated model has no independently eligible source operation")
    eligible_operations = Counter(name(ops[row["source_op_index"]]) for row in eligible)
    frozen = _frozen_split(
        capsule, source_path, target=target, groups=groups, eligible=dict(eligible_operations), module=module
    )
    return {
        "status": "clean",
        "frozen_placement": frozen,
        "source_sha256": fresh["source_mlir_sha256"],
        "n_source_operations": len(ops),
        "eligible_source_regions": eligible,
        "eligible_source_op_indices": [row["source_op_index"] for row in eligible],
        "certificate": fresh,
    }


def _pinned(record, role):
    from .native_model_execution import _pinned_native_file, _read_pinned_payload

    _read_pinned_payload(record, role)
    return _pinned_native_file(record, role)


def _llvm_function(module, symbol):
    matches = [
        op
        for op in module.body.block.ops
        if op.name == "llvm.func" and getattr(op.properties.get("sym_name"), "data", None) == symbol
    ]
    if len(matches) != 1 or not matches[0].body.blocks:
        raise ValueError(f"actual LLVM has no unique defined function {symbol!r}")
    return matches[0]


def _mandatory_source_commands(function, commands, obligations):
    """Retain the old per-operation attribution rule for fused source groups."""
    from .native_dispatch_accounting import _mandatory_command_blocks

    completed = []
    for obligation in obligations:
        family = obligation["semantic_family"]
        matched = commands.get(family, set())
        if len(obligations) > 1:
            rid = obligation["region_id"]
            index = obligation["source_op_index"]
            unique_region = sum(row["region_id"] == rid for row in obligations) == 1
            matched = {
                op
                for op in matched
                if (
                    getattr(getattr(op.attributes.get("merlin.source_op_index"), "value", None), "data", None) == index
                    or (unique_region and getattr(op.attributes.get("prov.region_id"), "data", None) == rid)
                )
            }
        if not _mandatory_command_blocks(function, matched):
            # The candidate's emitted device code, not the harness, is what lacks the work.
            raise CandidateDispatchRefusal(
                f"eligible source operation {obligation['source_op_index']} lacks unavoidable "
                f"attributed {family} work; fused transforms need independent evidence"
            )
        completed.append({**obligation, "mandatory_device_work_commands": len(matched)})
    return completed


def _host_call_proof(host_ir, *, assigned, output):
    """Read the actual final host IR, including any selected late legalization.

    The compiler's LLVM importer supplies typed CFGs; every original source
    call needs its own unavoidable call site. Repeated kernels cannot collapse
    the source call roster into one opcode-presence check.
    """
    from xdsl.parser import Parser

    from merlin.llvmlower import toolchain
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    from .native_dispatch_accounting import _mandatory_command_blocks
    from .native_model_execution import _digest

    translator = toolchain.mlir_translate().resolve(strict=True)
    importer_before = _digest(translator)
    before = _digest(host_ir)
    translated = Path(output) / "owned_host_cfg.mlir"
    result = subprocess.run(
        [str(translator), "--import-llvm", str(host_ir), "-o", str(translated)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise ValueError("actual final host LLVM cannot be CFG-audited: " + result.stderr[-500:])
    # The stock importer emits module metadata with upstream custom assembly
    # absent from xDSL. Generic printing preserves those operations/attributes
    # and the actual CFG; do not drop metadata or rewrite LLVM instructions.
    printer = translator.with_name("mlir-opt").resolve(strict=True)
    printer_before = _digest(printer)
    generic = Path(output) / "owned_host_cfg.generic.mlir"
    printed = subprocess.run(
        [str(printer), str(translated), "--mlir-print-op-generic", "-o", str(generic)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if printed.returncode != 0 or _digest(printer) != printer_before:
        raise ValueError("actual host CFG cannot be printed generically with selected MLIR")
    module = Parser(make_llvm_context(), generic.read_text()).parse_module()
    wrapper = _llvm_function(module, "_mlir_ciface_forward")

    def callee(op):
        attr = op.properties.get("callee")
        return getattr(getattr(attr, "root_reference", None), "data", None)

    wrapper_calls = {op for op in wrapper.walk() if op.name == "llvm.call" and callee(op) == "forward"}
    if wrapper_calls:
        if len(wrapper_calls) != 1 or not _mandatory_command_blocks(wrapper, wrapper_calls):
            raise ValueError("owned host C entry can bypass or repeat the actual model body")
        function = _llvm_function(module, "forward")
    else:
        # The fixed host's selected wide-C-interface legalization can inline
        # the implementation into the C entry. Audit that actual entry body.
        function = wrapper
    required = Counter(row["symbol"] for row in assigned)
    sites = {}
    for op in function.walk():
        if op.name != "llvm.call":
            continue
        name = callee(op)
        if isinstance(name, str):
            name = name.removeprefix("_mlir_ciface_")
            if name in required:
                sites.setdefault(name, []).append(op)
    if Counter({name: len(calls) for name, calls in sites.items()}) != required:
        raise ValueError("actual host calls omit, duplicate or collapse the source device call roster")
    for name, calls in sites.items():
        for call in calls:
            if not _mandatory_command_blocks(function, {call}):
                raise ValueError(f"owned source call {name} can be bypassed on a returning host CFG path")
    if _digest(host_ir) != before or _digest(translator) != importer_before:
        raise ValueError("owned host LLVM or selected importer changed during CFG import")
    return {
        "host_llvm": before,
        "importer": importer_before,
        "imported_cfg": _digest(translated),
        "generic_printer": printer_before,
        "generic_cfg": _digest(generic),
        "mandatory_call_roster": dict(required),
        "scope": "actual final LLVM source-call dominance; not hardware commit trace",
    }


def _dispatch_proof(record, *, target, facts):
    """Original-source calls -> actual candidate kernel -> object -> same ELF."""
    import yaml
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.parser import Parser

    from merlin.common.facts_view import interface as facts_interface
    from merlin.llvmlower.device_shim import kernel_abi_for
    from merlin.targetgen.contract.schemas import contract_dir
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
    from merlin.targetgen.rocc.decode import decode_module
    from merlin.targetgen.rtl.facts import observed_facts

    from .native_dispatch_accounting import _kernel_command_inventory, _verified_work_functs_by_family
    from .native_model_execution import _digest

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
    abi = kernel_abi_for(target)
    if abi is None:
        raise ValueError("actual device kernel has no independently selected ABI")
    completed, decoded_classes, artifact_pins = [], set(), {}
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
        _verify_device_custody(
            entry,
            artifact=artifact,
            llvm_ir=_pinned(pins, "llvm_ir"),
            obj=obj,
            original_symbol=abi.symbol,
            kernel_symbol=kernel_symbol,
        )
        module = Parser(make_llvm_context(), artifact.read_text()).parse_module()
        function = _llvm_function(module, abi.symbol)
        with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
            trace = decode_module(module, target=target)
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
            decoded_classes.add(str(decoded.get("class")))
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
        source_group = source["joined_groups"][row["group"]]
        obligations = [
            item for item in source["placement"]["eligible_source_regions"] if item["source_group"] == source_group
        ]
        completed.extend(_mandatory_source_commands(function, commands, obligations))
        object_words, object_functs = _kernel_command_inventory(obj, symbol=kernel_symbol, table=table)
        elf_words, elf_functs = _kernel_command_inventory(elf, symbol=kernel_symbol, table=table)
        if (
            Counter(all_functs) != Counter(object_functs)
            or Counter(object_words) != Counter(elf_words)
            or Counter(object_functs) != Counter(elf_functs)
        ):
            inventory = {"schema": "merlin.device_command_inventory_refusal.v1", "group": row["group"],
                         "symbol": symbol, "kernel_symbol": kernel_symbol,
                         "llvm_funct_counts": dict(Counter(all_functs)),
                         "object_funct_counts": dict(Counter(object_functs)),
                         "elf_funct_counts": dict(Counter(elf_functs)),
                         "object_and_elf_words_equal": Counter(object_words) == Counter(elf_words),
                         "scope": "static inventories; optimizer transformations are not a dispatch proof"}
            raw = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode()
            from hashlib import sha256

            diagnostic = elf.parent / f"{symbol}.command-inventory-refusal-{sha256(raw).hexdigest()}.json"
            try:
                with diagnostic.open("xb") as stream:
                    stream.write(raw)
            except FileExistsError:
                if diagnostic.is_symlink() or diagnostic.read_bytes() != raw:
                    raise ValueError("device command inventory diagnostic changed")
            stage = ("device LLVM to object" if Counter(all_functs) != Counter(object_functs)
                     else "device object to linked ELF")
            raise ValueError("complete custom-command inventory changed across device LLVM/object/ELF; "
                             f"group {row['group']} ({symbol}), first mismatch: {stage}; "
                             "static counts cannot yet certify this compiler transformation")
        artifact_pins[symbol] = {
            "kernel_symbol": kernel_symbol,
            **{role: _digest(_pinned(pins, role)) for role in pins},
        }
    eligible = source["placement"]["eligible_source_op_indices"]
    if sorted(row["source_op_index"] for row in completed) != sorted(eligible):
        raise ValueError("mandatory owned-device work does not cover the complete eligible source roster")
    host = _pinned(built.get("host_artifacts") or {}, "llvm_ir")
    _pinned(built.get("host_artifacts") or {}, "object")
    host_calls = _host_call_proof(host, assigned=assigned, output=elf.parent)
    return {
        "status": "verified",
        "source_sha256": source["placement"]["source_sha256"],
        "linked_elf": _digest(elf),
        "eligible_tasks": completed,
        "device_artifacts": artifact_pins,
        "host_calls": host_calls,
        "observed_classes": sorted(decoded_classes),
        "scope": "completion-implied unavoidable source-bound work; "
        "not hardware commit trace or general transform theorem",
    }


def _verify_device_custody(entry, *, artifact, llvm_ir, obj, original_symbol, kernel_symbol):
    """Reopen actual stock translation/compilation and a symbol-only rename."""
    from merlin.common import invocation_record
    from merlin.llvmlower import toolchain

    from .native_model_execution import _digest

    custody = entry.get("custody")
    if not isinstance(custody, dict) or set(custody) != {"translation", "compilation", "rename"}:
        raise ValueError("actual device object has no complete codegen custody")
    stages = {}
    for role, stage in (
        ("translation", "device_llvm_translation"),
        ("compilation", "device_object_compilation"),
        ("rename", "device_object_symbol_rename"),
    ):
        observation = invocation_record.verify(_pinned(custody, role))
        if (
            observation.get("stage") != stage
            or len(observation.get("inputs") or []) != 1
            or len(observation.get("outputs") or []) != 1
        ):
            raise ValueError("device codegen custody selected another invocation")
        stages[role] = observation
    translation, compilation, rename = (stages[role] for role in ("translation", "compilation", "rename"))
    if (
        translation["inputs"][0]["sha256"] != _digest(artifact)["sha256"]
        or translation["outputs"][0]["sha256"] != _digest(llvm_ir)["sha256"]
        or compilation["inputs"] != translation["outputs"]
        or rename["inputs"] != compilation["outputs"]
        or rename["outputs"][0] != {key: _digest(obj)[key] for key in ("path", "sha256")}
    ):
        raise ValueError("device LLVM/object custody does not join the exact emitted and linked bytes")
    if (
        Path(translation["executable"]["path"]) != toolchain.mlir_translate().resolve()
        or Path(compilation["executable"]["path"]) != toolchain.clang().resolve()
        or translation["argv"][1:]
        != ["--mlir-to-llvmir", translation["inputs"][0]["path"], "-o", translation["outputs"][0]["path"]]
        or "-c" not in compilation["argv"]
        or rename["argv"][1:]
        != [
            f"--redefine-sym={original_symbol}={kernel_symbol}",
            rename["inputs"][0]["path"],
            rename["outputs"][0]["path"],
        ]
    ):
        raise ValueError("device object underwent an unverified codegen or rename transform")


def _numeric_full_output(capsule, record, console):
    """Use the unchanged public numeric policy on every typed forward result."""
    import numpy as np
    import yaml

    from merlin.compile.baremetal_model import read_full_outputs
    from merlin.runtime.out_bin import binary_console_diagnostics
    from merlin.targetgen.capsule_golden import compare
    from merlin.targetgen.golden_store import load_golden

    text = binary_console_diagnostics(console).decode("utf-8", errors="replace")
    if sum(line.strip() == "DONE" for line in text.splitlines()) != 1:
        raise ValueError("owned model full console has no unique successful completion")
    values, metrics = read_full_outputs(
        console, Path(record["runtime_inputs"]), record["build"]["output"]["build_hash"]
    )
    order = json.loads((Path(record["runtime_inputs"]) / "output_order.json").read_text())
    capsule_root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((capsule_root / "capsule.yaml").read_text())
    golden = load_golden(capsule_root)
    expected = golden["outputs"]
    if len(order) != len(values) or set(order) != set(expected) or len(order) != len(set(order)):
        raise ValueError("actual owned model output roster differs from frozen full golden")
    observed = {}
    for name, value in zip(order, values, strict=True):
        shape = np.asarray(expected[name]).shape
        if value.size != np.asarray(expected[name]).size:
            raise ValueError("owned full result extent differs from frozen golden")
        observed[name] = value.reshape(shape).tolist()
    return compare(
        expected,
        observed,
        declaration["numeric_policy"],
        golden_source=str(golden.get("golden_source") or "independent_capsule"),
    ), metrics


def _static_tiers(capsule, record, *, target, facts, dispatch, policy):
    """Original-source closure and whole-ELF ISA; neither grants numerics."""
    from merlin.common.facts_view import interface as facts_interface
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.perf.isa_prohibition import scan_elf
    from merlin.runtime.backends.spike import gcc_path
    from merlin.runtime.boards import load_boards
    from merlin.targetgen import elf_lanes
    from merlin.targetgen.rtl.facts import observed_facts

    from .native_model_execution import _digest

    output = record["build"]["output"]
    inputs = record["build"]["inputs"]
    board = load_boards(Path(inputs["board_catalog"])).get(inputs["board"])
    if board is None or board.target != target:
        raise ValueError("owned model selected board and RTL facts identify different configurations")
    from merlin.compile.model_execution_inputs import selected_firrtl

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
    classes = set(dispatch["observed_classes"])
    if not set(policy["expected_instruction_classes"]) <= classes:
        raise ValueError("owned device kernels omit frozen required instruction classes")
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
            "observed_classes": sorted(classes),
            "required_classes": policy["expected_instruction_classes"],
        },
    }


def _selected_facts(path, *, target):
    from merlin.compile.model_execution_inputs import selected_execution_firrtl

    return selected_execution_firrtl(path, target=target)


def _native_engine_for_facts(target, engine, facts):
    from merlin.compile.model_execution_inputs import native_engine, selected_firrtl
    from merlin.targetgen.rtl.facts import observed_facts

    if selected_firrtl(facts["path"], target=target, config=facts["config"]) != facts:
        raise ValueError("protected owned-model RTL facts changed")
    with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
        return native_engine(target, engine, facts)


def _functional_engine_for_facts(target, facts):
    from merlin.targetgen.rtl.facts import observed_facts

    from .native_model_execution import _functional_engine

    with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
        return _functional_engine(target)


def _run_selected_functional(backend, elf, *, timeout):
    """Apply the selected process ceiling to the owned functional transport."""
    from merlin.common.execution_deadline import ExecutionDeadline
    from merlin.targetgen.contract.compile import _selected_run_timeout

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

    from merlin.common.execution_deadline import ExecutionDeadline
    from merlin.compile.model_execution_inputs import selected_firrtl
    from merlin.targetgen.contract.compile import _selected_run_timeout
    from merlin.targetgen.rtl_engine_policy import gsim_runtime_slot

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

    from merlin.targetgen.oracle_policy import selected_l3_engine_report

    from .native_model_execution import _digest, _frozen_model_policy

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
                    from merlin.targetgen.rtl_engine_policy import ELABORATED_RTL

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
            if _digest(elf) != identity:
                raise ValueError("owned model ELF changed during required-tier execution")
            tiers[tier] = {
                "status": numeric["status"],
                **engine_evidence,
                "elf": identity,
                "console": _digest(path),
                "numeric": numeric,
                "metrics": metrics,
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


def _same_owned_frozen_policy(recorded, fresh) -> bool:
    """Join a retained capsule copy by verified bytes, preserving policy owners."""
    from .native_model_execution import _read_pinned_payload

    local_roles = {"capsule_declaration", "instruction_coverage"}
    if not isinstance(recorded, dict) or not isinstance(fresh, dict) or set(recorded) != set(fresh):
        return False
    if any(recorded[key] != fresh[key] for key in fresh if key not in local_roles):
        return False
    for role in local_roles:
        # Reopen both ordinary files under their exact stored identities;
        # dropping path strings alone would let a missing or mutated copy pass.
        original = _read_pinned_payload(recorded, role)
        selected = _read_pinned_payload(fresh, role)
        old_pin, new_pin = recorded[role], fresh[role]
        if (
            set(old_pin) != {"path", "sha256", "size_bytes"}
            or set(new_pin) != {"path", "sha256", "size_bytes"}
            or old_pin["sha256"] != new_pin["sha256"]
            or old_pin["size_bytes"] != new_pin["size_bytes"]
            or original != selected
        ):
            return False
    return True


def _stopped_before_receipt(stopped):
    """The audit verdict for a run that stopped before producing its receipt: its root cause first.

    The absent receipt is only a consequence. Reporting it ("receipt ... is absent") hid the actual
    reason, and when that reason is the candidate's own dispatch refusal, it turned a compiler failure
    into what reads as unmeasured tooling. A proven emitted-device host-compute veto is also a
    candidate failure. Any other stop stays incomplete (NOT_RUN_IS_NOT_PASS) and
    names its stage and exception; host paths are scrubbed from the quoted message.
    """
    from merlin.common.path_scrub import scrub_host_paths

    stage, kind = str(stopped.get("stage") or "unknown stage"), str(stopped.get("type") or "error")
    message = scrub_host_paths(str(stopped.get("detail") or ""))[:2000]
    if kind == "HostComputeVeto" and stage == "owned_host_execution":
        return {
            "status": "fail",
            "kind": "owned_host_model_execution",
            "violations": ["candidate_host_tensor_compute_violation"],
            "detail": message,
            "failure": {"plane": "host_compute", "category": "host_compute_in_accelerator_group",
                        "detail": message, "stage": stage},
        }
    if kind == CandidateDispatchRefusal.__name__ and stage == "owned_host_execution":
        return {
            "status": "fail",
            "kind": "owned_host_model_execution",
            "violations": ["candidate_dispatch_attribution_refused"],
            "detail": message,
            "failure": {
                "plane": "candidate_compilation",
                "category": "CANDIDATE_EMISSION_FAILED",
                "detail": message,
                "stage": stage,
            },
        }
    detail = f"the model run stopped at {stage} with {kind} before producing its receipt: {message}"
    return {
        "status": "incomplete",
        "kind": "owned_host_model_execution",
        "violations": ["candidate_completed_dispatch_unverified", "candidate_required_tiers_unverified"],
        "detail": detail,
        "failure": {"plane": "candidate_model_execution", "category": "NOT_RUN_IS_NOT_PASS", "detail": detail},
    }


def audit_owned_host_model(result, capsule, *, target, readback_policy=None):
    """Independently reopen the provided-host result before promoting any tier."""
    import os

    import yaml

    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.targetgen.contract.readback_policy import selected
    from merlin.targetgen.oracle_policy import selected_l3_engine_report

    from .capsule_runner import _model_runtime_bundle, _resolve_model_host_lane
    from .native_model_execution import _frozen_model_policy

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
        with _model_runtime_bundle(capsule, timeout=inputs["timeout_s"]) as (fresh, provenance, verify):
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
            if (
                numeric != evidence.get("numeric")
                or evidence["status"] != numeric["status"]
                or metrics != evidence.get("metrics")
            ):
                raise ValueError(f"owned model {tier} receipt differs from parsed full values")
            tiers[tier] = {**evidence, "numeric": numeric, "dispatch": dispatch}
            if tier == "L3":
                fidelity = selected_engine.get("fidelity")
                if isinstance(fidelity, str) and fidelity:
                    from merlin.targetgen.rtl_engine_policy import ELABORATED_RTL

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
        return {
            "status": "fail" if failed else "incomplete" if pending else "pass",
            "kind": "owned_host_model_execution",
            "violations": ["candidate_verified_numeric_mismatch"] if failed else [],
            "owned_host_model_check": {
                "source_placement": source["placement"],
                "completed_dispatch": dispatch,
                "candidate_source_coverage": coverage,
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
