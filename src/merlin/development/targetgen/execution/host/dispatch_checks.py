"""Owned-host dispatch checks on the candidate's actual emitted device and host code.

These read the final host CFG's source call roster, each device kernel's mandatory
attributed work, its optimized command inventory and the stock codegen custody that
joins its LLVM to the linked object. Each reopens actual bytes; none grants a tier.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

from merlin.runtime.execution.model_inputs import file_sha256

# ---- owned_host_dispatch_checks ----------------------------------------------------------------


class CandidateDispatchRefusal(ValueError):
    """The candidate's own emitted program fails a source-attribution requirement of its dispatch.

    Raised only from checks of the candidate's emitted device code, never for a harness, input or
    tool fault, so the grade can report the compiler's failure as a failure rather than leaving it
    behind the absent receipt it causes (which reads as unmeasured tooling).
    """


def _llvm_function(module, symbol):
    from merlin.verification.host.llvm_symbols import defined_function

    try:
        return defined_function(module, symbol)
    except ValueError as error:
        raise ValueError(f"actual LLVM has no unique defined function {symbol!r}") from error


def _mandatory_source_commands(function, commands, obligations):
    """Retain the old per-operation attribution rule for fused source groups."""
    from merlin.development.targetgen.execution.native.dispatch_accounting import _mandatory_command_blocks

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


def _host_call_proof(host_ir, *, assigned, output, source_composition=None):
    """Read the actual final host IR, including any selected late legalization.

    The compiler's LLVM importer supplies typed CFGs; every original source
    call needs its own unavoidable call site. Repeated kernels cannot collapse
    the source call roster into one opcode-presence check.
    """
    from xdsl.parser import Parser

    from merlin.compiler.ir.llvm_context import make_llvm_context
    from merlin.compiler.lower.model import toolchain
    from merlin.development.targetgen.execution.native.dispatch_accounting import _mandatory_command_blocks
    from merlin.development.targetgen.execution.native.native_model_execution import _digest

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
    component_chains = []
    if source_composition is not None and source_composition.get("schema") in {
        "merlin.original_sibling_source_components.v1",
        "merlin.original_head_source_composition.v1",
    }:
        from merlin.development.targetgen.execution.native.private_dispatch import (
            borrowed_host_chain,
            composition_members,
        )

        composition_members(source_composition, assigned=assigned)
        siblings = [
            c
            for c in source_composition["components"]
            if c.get("schema") == "merlin.original_sibling_source_composition.v1"
        ]
        component_chains = [
            borrowed_host_chain(module, function, c, mandatory=_mandatory_command_blocks) for c in siblings
        ]
        composed = {c["physical_symbol"] for c in siblings}
        # Head-family calls remain direct and unavoidable. Their stock shim's
        # full slice multiplicity is independently checked at the bridge.
        assigned = [row for row in assigned if row["symbol"] not in composed]
    elif source_composition is not None:
        from merlin.development.targetgen.execution.native.private_dispatch import (
            borrowed_host_chain,
            composition_members,
        )

        composition_members(source_composition, assigned=assigned)
        chain = borrowed_host_chain(module, function, source_composition, mandatory=_mandatory_command_blocks)
        if _digest(host_ir) != before or _digest(translator) != importer_before:
            raise ValueError("owned host LLVM or selected importer changed during CFG import")
        return {
            "host_llvm": before,
            "importer": importer_before,
            "imported_cfg": _digest(translated),
            "generic_printer": printer_before,
            "generic_cfg": _digest(generic),
            **chain,
            "scope": "actual source-owned wrapper CFG; descriptor bridge and final link audited separately",
        }
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
        "mandatory_call_roster": dict(
            required
            + Counter(
                {
                    symbol: count
                    for chain in component_chains
                    for symbol, count in chain["mandatory_call_roster"].items()
                }
            )
        ),
        **({"component_chains": component_chains} if component_chains else {}),
        "scope": "actual final LLVM source-call dominance; not hardware commit trace",
    }


def _optimized_command_functs(module, trace, table, *, source_functs):
    """Count the complete structurally decoded actual object-input inventory."""
    opcode = (trace.get("abi") or {}).get("custom_opcode")
    instructions = trace.get("instructions")
    assembly = [op for op in module.walk() if op.name == "llvm.inline_asm"]
    if (
        not isinstance(opcode, str)
        or int(opcode, 16) != table.get("custom_opcode")
        or not isinstance(instructions, list)
        or len(assembly) != len(instructions)
    ):
        raise ValueError("optimized device LLVM has incomplete selected structural decode")
    names = {int(key) for key in (table.get("names") or {})}
    functs = []
    for decoded in instructions:
        if decoded.get("class") == "UNKNOWN":
            raise ValueError("optimized device LLVM contains an unknown command")
        funct = decoded.get("funct")
        if funct is not None:
            if type(funct) is not int or funct not in names:
                raise ValueError("optimized device command has no selected ISA identity")
            functs.append(funct)
    if set(functs) != set(source_functs):
        raise ValueError("device optimization introduced or discarded a custom-command function")
    return functs


def _verify_device_custody(entry, *, artifact, llvm_ir, obj, original_symbol, kernel_symbol, source_composition=None):
    """Reopen actual stock translation/compilation and a symbol-only rename."""
    from merlin.common.process import invocation_records as invocation_record
    from merlin.compiler.lower.model import toolchain
    from merlin.development.targetgen.execution.host.model_execution import _pinned
    from merlin.development.targetgen.execution.native.native_model_execution import _digest

    custody = entry.get("custody")
    if source_composition is not None:
        role = (
            "head_source" if source_composition.get("schema") == "merlin.original_head_family.v1" else "sibling_source"
        )
        if not isinstance(custody, dict) or role not in custody:
            raise ValueError("actual sibling device object lacks source-owned descriptor custody")
        # Its exact source/ABI/bridge is independently reopened by dispatch.
        custody = {key: value for key, value in custody.items() if key != role}
    original_roles = {"translation", "compilation", "rename"}
    optimized_roles = original_roles | {"optimization", "optimized_import"}
    if not isinstance(custody, dict) or set(custody) not in (original_roles, optimized_roles):
        raise ValueError("actual device object has no complete codegen custody")
    optimized = set(custody) == optimized_roles
    pins = entry.get("artifacts") or {}
    if not optimized and {"optimized_llvm_ir", "optimized_llvm_mlir"} & pins.keys():
        raise ValueError("optimized device LLVM has no complete optimization custody")
    stages = {}
    selected_stages = [
        ("translation", "device_llvm_translation"),
        ("compilation", "device_object_compilation"),
        ("rename", "device_object_symbol_rename"),
    ]
    if optimized:
        selected_stages += [
            ("optimization", "device_llvm_optimization"),
            ("optimized_import", "device_optimized_llvm_import"),
        ]
    for role, stage in selected_stages:
        observation = invocation_record.verify(_pinned(custody, role))
        if (
            observation.get("stage") != stage
            or len(observation.get("inputs") or []) != 1
            or len(observation.get("outputs") or []) != 1
        ):
            raise ValueError("device codegen custody selected another invocation")
        stages[role] = observation
    translation, compilation, rename = (stages[role] for role in ("translation", "compilation", "rename"))
    object_input = translation["outputs"]
    optimized_module = None
    if optimized:
        optimization, imported = stages["optimization"], stages["optimized_import"]
        optimized_ir = _pinned(pins, "optimized_llvm_ir")
        optimized_module = _pinned(pins, "optimized_llvm_mlir")
        if (
            optimization["inputs"] != translation["outputs"]
            or optimization["outputs"][0] != {key: _digest(optimized_ir)[key] for key in ("path", "sha256")}
            or imported["inputs"] != optimization["outputs"]
            or imported["outputs"][0] != {key: _digest(optimized_module)[key] for key in ("path", "sha256")}
            or Path(optimization["executable"]["path"]) != toolchain.clang().resolve()
            or Path(imported["executable"]["path"]) != toolchain.mlir_translate().resolve()
            or compilation["argv"][-6:]
            != ["-Xclang", "-disable-llvm-passes", "-c", str(optimized_ir), "-o", compilation["outputs"][0]["path"]]
            or optimization["argv"]
            != [
                compilation["argv"][0],
                *compilation["argv"][1:-6],
                "-S",
                "-emit-llvm",
                translation["outputs"][0]["path"],
                "-o",
                str(optimized_ir),
            ]
            or imported["argv"][1:]
            != ["--import-llvm", "--mlir-print-op-generic", str(optimized_ir), "-o", str(optimized_module)]
        ):
            raise ValueError("device optimization custody does not join the exact selected LLVM transform")
        object_input = optimization["outputs"]
    if (
        translation["inputs"][0]["sha256"] != _digest(artifact)["sha256"]
        or translation["outputs"][0]["sha256"] != _digest(llvm_ir)["sha256"]
        or compilation["inputs"] != object_input
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
        not in (
            [
                f"--redefine-sym={original_symbol}={kernel_symbol}",
                rename["inputs"][0]["path"],
                rename["outputs"][0]["path"],
            ],
            [
                f"--redefine-sym={original_symbol}={kernel_symbol}",
                f"--keep-global-symbol={original_symbol}",
                f"--keep-global-symbol={kernel_symbol}",
                rename["inputs"][0]["path"],
                rename["outputs"][0]["path"],
            ],
        )
    ):
        raise ValueError("device object underwent an unverified codegen or rename transform")
    return optimized_module


# ---- owned_host_selections ---------------------------------------------------------------------
# Original owned-model input selections, frozen policy and independent engine bindings.


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


def _composition_preferences(candidate_package):
    """Read bounded optimization requests, never source or effect permissions."""
    from merlin.development.targetgen.execution.host.source_composition_preferences import read

    return read(candidate_package)


def _resolve_weight_preparation(bundle, *, requested):
    from merlin.development.targetgen.execution.host.source_composition_preferences import weight_preparation

    return weight_preparation(bundle, requested=requested)


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


def _untagged_assert_comparisons(ops, frozen, groups):
    """Scalar support for tagged assertions is not another provenance demand.

    Keep these operations in the complete frozen placement/build inventory. Only
    exact source-bound host admission permits this distinction; arbitrary untagged
    arithmetic still fails the demand join.
    """
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition.groups import compute_groups as groups_api

    host = frozen["host_operations"]
    owner = {id(member): group for group in groups for member in group.members}
    support = set()
    assertions = set()
    for op in ops:
        if (
            mq.op_name(op) != "arith.cmpi"
            or "prov.op" in op.attributes
            or op.regions
            or op.successors
            or len(op.operands) != 2
            or [str(value.type) for value in op.operands] != ["index", "index"]
            or len(op.results) != 1
            or str(op.results[0].type) != "i1"
        ):
            continue
        uses = tuple(op.results[0].uses)
        if not uses or any(
            use.operation not in ops
            or mq.op_name(use.operation) != "cf.assert"
            or mq.attr_str(use.operation, "prov.op") != "shape_guard"
            or len(use.operation.operands) != 1
            or use.operation.results
            or use.operation.regions
            or use.operation.successors
            for use in uses
        ):
            continue
        members = [op, *(use.operation for use in uses)]
        if any(id(member) in owner and owner[id(member)].placement != groups_api.HOST for member in members):
            continue
        support.add(id(op))
        assertions.update(id(use.operation) for use in uses)
    if len(support) > host.get("arith.cmpi", 0) or len(assertions) > host.get("cf.assert", 0):
        raise ValueError("scalar assertion support lacks complete frozen host admission")
    return support


def _pinned(record, role):
    from merlin.development.targetgen.execution.native.native_model_execution import (
        _pinned_native_file,
        _read_pinned_payload,
    )

    _read_pinned_payload(record, role)
    return _pinned_native_file(record, role)


def _require_instruction_classes(required, observed):
    """Report declared structural obligations, never private reference values."""
    missing = set(required) - set(observed)
    if missing:
        raise ValueError(
            "owned device kernels omit frozen required instruction classes: "
            f"missing={sorted(missing)!r}; observed={sorted(set(observed))!r}; "
            f"required={sorted(set(required))!r}"
        )


def _selected_facts(path, *, target):
    from merlin.runtime.execution.model_inputs import selected_execution_firrtl

    return selected_execution_firrtl(path, target=target)


def _native_engine_for_facts(target, engine, facts):
    from merlin.runtime.execution.model_inputs import native_engine, selected_firrtl
    from merlin.target.rtl.facts import observed_facts

    if selected_firrtl(facts["path"], target=target, config=facts["config"]) != facts:
        raise ValueError("protected owned-model RTL facts changed")
    with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
        return native_engine(target, engine, facts)


def _functional_engine_for_facts(target, facts):
    from merlin.development.targetgen.execution.native.native_model_execution import _functional_engine
    from merlin.target.rtl.facts import observed_facts

    with observed_facts(target, json.loads(Path(facts["path"]).read_text()), path=Path(facts["path"])):
        return _functional_engine(target)


def _same_owned_frozen_policy(recorded, fresh) -> bool:
    """Join a retained capsule copy by verified bytes, preserving policy owners."""
    from merlin.development.targetgen.execution.native.native_model_execution import _read_pinned_payload

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
    from merlin.common.access import scrub_host_paths

    stage, kind = str(stopped.get("stage") or "unknown stage"), str(stopped.get("type") or "error")
    message = scrub_host_paths(str(stopped.get("detail") or ""))[:2000]
    if kind == "HostComputeVeto" and stage == "owned_host_execution":
        return {
            "status": "fail",
            "kind": "owned_host_model_execution",
            "violations": ["candidate_host_tensor_compute_violation"],
            "detail": message,
            "failure": {
                "plane": "host_compute",
                "category": "host_compute_in_accelerator_group",
                "detail": message,
                "stage": stage,
            },
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


# Preserve historical callable and record pickle lookup identities.
_selected_file.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_composition_preferences.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_resolve_weight_preparation.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_independently_eligible_group_ids.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_untagged_assert_comparisons.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_pinned.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_require_instruction_classes.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_selected_facts.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_native_engine_for_facts.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_functional_engine_for_facts.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_same_owned_frozen_policy.__module__ = "merlin.development.targetgen.execution.host.model_execution"
_stopped_before_receipt.__module__ = "merlin.development.targetgen.execution.host.model_execution"
