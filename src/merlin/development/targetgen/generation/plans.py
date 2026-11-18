"""Derive reviewable target, dialect, runtime, Zephyr and LLVM plans.

Selected provider contracts and typed operation declarations remain authoritative.
Absent authored inputs retain conservative synthesis and their existing review flags;
plan generation establishes no hardware, numerical or execution qualification.
The LLVM scaffold emitter has a separate, narrower compiler-authoring grant.
"""

from __future__ import annotations

import copy
from typing import Any

from merlin.development.targetgen.generation.evidence import Evidence
from merlin.target.capabilities import families as _families


def _conservative_contract(evidence: Evidence) -> dict[str, Any]:
    """Skeleton contract for a real target, seeded from detected concepts.

    Every field is intentionally coarse and flagged for human review. The detected concepts
    are surfaced under ``capabilities.detected_concepts`` rather than asserted as real ops.
    """
    concepts = sorted(evidence.concept_names())
    return {
        "name": evidence.target,
        "version": 0.0,
        # Empty pending human review.
        "features": [],
        "ops": [],
        "types": [],
        "runtime": {"backends": []},
        "capabilities": {
            "ops": [],
            "layouts": [],
            "detected_concepts": concepts,
        },
        "memory_model": {
            "resident": any(c in concepts for c in ("scratchpad", "resident_packed_tensor")),
            "accumulators": "accumulator" in concepts,
        },
        "compiler_obligations": ["TODO: derive from human review of evidence"],
        "hardware_promises": ["TODO: confirm against RTL/docs"],
        "runtime_promises": ["TODO: confirm runtime adapter capabilities"],
        "legality": ["TODO: derive legality predicates from human review"],
        "confidence": "low",
        "requires_human_review": True,
    }


def _curated_contract(target_name: str) -> dict[str, Any] | None:
    """Read the selected support contract, never mix it with another provider.

    Unknown targets may have no authored contract and retain conservative synthesis.
    Invalid explicit provider declarations propagate the registry's refusal.
    An explicit external selection is authoritative over a same-name checkout example.
    """
    from merlin.target.registry.identity import resolve

    selected = resolve(target_name)
    if not selected.contract_path.exists() and not selected.contract_path.is_symlink():
        return None
    contract = selected.load_contract()
    if not isinstance(contract, dict):
        raise ValueError(f"{selected.contract_path}: target contract must be a mapping")
    return contract


def synthesize_target_contract(evidence: Evidence, target_name: str) -> dict[str, Any]:
    """Use selected support data or conservative synthesis.

    Provider absence allows synthesis; invalid explicit provider data does not.
    """
    curated = _curated_contract(target_name)
    if curated is not None:
        return curated
    return _conservative_contract(evidence)


def _conservative(evidence: Evidence) -> dict[str, Any]:
    concepts = sorted(evidence.concept_names())
    return {
        "target": evidence.target,
        "dialect_name": evidence.target,
        "ops": [],
        "types": [],
        "lowering": [],
        "tests": [],
        "detected_concepts": concepts,
        "notes": "Ops/types/lowerings are a human-review decision; do not auto-generate "
        "dialect ops directly from instruction names.",
        "confidence": "low",
        "requires_human_review": True,
    }


def _curated(target_name: str) -> dict[str, Any] | None:
    """Read only the selected support provider's optional authored plan.

    No plan means derive one from its contract below, not borrow an in-tree plan.
    A present but malformed or escaping resource is an error, never absence.
    """
    import yaml

    from merlin.compiler.backend.providers import contained_resource
    from merlin.target.registry.identity import resolve

    selected = resolve(target_name)
    path = selected.dialect_plan_path
    if not path.exists() and not path.is_symlink():
        return None
    if selected.kind == "external":
        path = contained_resource(selected.base, str(path.relative_to(selected.base)))
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError(f"{path}: dialect plan must be a mapping")
    if plan.get("target") != selected.name:
        raise ValueError(f"{path}: dialect plan target differs from selected provider {selected.name!r}")
    return plan


def _generate_from_operation_capabilities(target_contract: dict[str, Any]) -> dict[str, Any] | None:
    """Project reviewed typed signatures from the existing operation contract.

    The source registry remains authoritative. This projection does not infer
    semantics, Pure effects, or a lowering from a mnemonic. An incomplete
    explicitly typed declaration refuses rather than falling back to the
    legacy name-only dialect generator.
    """
    declaration = target_contract.get("operation_capabilities") or {}
    if not isinstance(declaration, dict):
        raise ValueError("operation_capabilities must be a mapping")
    rows = declaration.get("operations") or []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("operation_capabilities.operations must be a list")
    typed_rows = [
        row
        for row in rows
        if row.get("domain") == "dialect"
        and isinstance(row.get("semantics"), dict)
        and "mlir_signature" in row["semantics"]
    ]
    if not typed_rows:
        return None
    dialect = target_contract.get("dialect_name")
    if not isinstance(dialect, str) or not dialect:
        raise ValueError("typed operation contract requires dialect_name")
    selected = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("domain") == "dialect" and row.get("dialect") == dialect
    ]
    if not selected or any(row not in selected for row in typed_rows):
        raise ValueError("typed operation dialect differs from selected dialect_name")
    if any(not isinstance(row.get("semantics"), dict) or "mlir_signature" not in row["semantics"] for row in selected):
        raise ValueError("every selected dialect operation needs an explicit MLIR signature")
    types = target_contract.get("types", [])
    if not isinstance(types, list) or any(not isinstance(row, dict) for row in types):
        raise ValueError("typed operation contracts require explicit custom type declarations")
    ops = []
    for row in sorted(selected, key=lambda item: str(item.get("operation"))):
        if not isinstance(row.get("operation"), str) or not row["operation"]:
            raise ValueError("typed operation contract has no operation name")
        ops.append(
            {
                "name": row["operation"],
                "summary": str((row.get("semantics") or {}).get("summary") or row["operation"]),
                "signature": copy.deepcopy(row["semantics"]["mlir_signature"]),
            }
        )
    plan = {
        "target": target_contract["name"],
        "dialect_name": target_contract["dialect_name"],
        "types": copy.deepcopy(types),
        "ops": ops,
        "lowering": [],
        "tests": [],
        "generated_from_contract": True,
        "requires_human_review": True,
        "source_operation_ids": [
            {"domain": "dialect", "dialect": target_contract["dialect_name"], "operation": op["name"]} for op in ops
        ],
    }
    from merlin.development.targetgen.generation.typed_mlir import validate

    validate(plan)
    return plan


# The Merlin tensor-resident interface a target dialect lowers. role -> (canonical op name, matcher,
# summary). A contract that advertises this shape gets a GENERATED, usable dialect_plan (not a stub).
_ROLES: list[tuple[str, str, Any, str]] = [
    ("resident_pack", "pack", lambda o: "pack" in o, "pack + make RHS resident"),
    ("matmul", "matmul", lambda o: "matmul" in o, "matmul vs resident tensor -> accumulator"),
    ("commit", "commit", lambda o: "commit" in o, "apply epilogue + commit accumulator"),
    ("resident_evict", "evict", lambda o: "evict" in o or "release" in o, "free resident storage"),
]

# Optional vector-lane roles — emitted only when the contract advertises non-matmul pointwise
# compute (relu/bias_add/elementwise), i.e. the target has vector/scalar lanes beyond the systolic
# matmul. A pure-matmul target omits them (its dialect stays the 4-op resident core).
_VECTOR_ROLES: list[tuple[str, Any, str]] = [
    (
        "vector_map",
        lambda o: "vector_map" in o,
        "elementwise combine (add/mul/identity) + activation on the vector lanes",
    ),
    ("vector_reduce", lambda o: "vector_reduce" in o, "reduce a tensor (sum) on the vector lanes"),
]


def _has_vector_lanes(tc: dict[str, Any]) -> bool:
    """True when the contract advertises non-matmul pointwise compute — a vector/scalar lane beyond
    the systolic matmul. Derived from its capability ops (never a target name): any capability op
    other than matmul (bias_add, relu, an elementwise add/mul, …) implies a vector datapath."""
    cap_ops = set((tc.get("capabilities") or {}).get("ops") or [])
    return bool(cap_ops - {"matmul"})


def _is_tensor_resident(tc: dict[str, Any]) -> bool:
    """A target that implements the Merlin tensor-resident interface (packs a resident weight,
    accumulates, commits via a command buffer). Detected from the contract's own declarations."""
    feats = set(tc.get("features") or [])
    ops = set(tc.get("ops") or []) | set((tc.get("capabilities") or {}).get("ops") or [])
    return (
        "command_buffer" in feats
        and ("resident_packed_tensor" in feats or "accumulator_commit" in feats)
        and "matmul" in ops
    )


def _generate(target_contract: dict[str, Any]) -> dict[str, Any]:
    """Generate a usable dialect_plan from a tensor-resident contract (its op/type names drive the
    dialect; the interface->target lowering is the canonical mapping). Feeds the dialect factory."""
    name = target_contract["name"]
    dname = name.replace("_", "")
    decl_ops = list(target_contract.get("ops") or [])
    role_op = {role: (next((o for o in decl_ops if match(o)), canon)) for role, canon, match, _ in _ROLES}
    # Vector-lane roles are conditional: a pure-matmul target keeps the 4-op resident core.
    vector_roles = []
    if _has_vector_lanes(target_contract):
        vector_roles = [
            (role, next((o for o in decl_ops if match(o)), role), summ) for role, match, summ in _VECTOR_ROLES
        ]
    types = target_contract.get("types") or ["resident_tensor", "accumulator"]
    plan = {
        "target": name,
        "dialect_name": dname,
        "ops": [
            {"name": role_op[r], "summary": summ, "source_interface": f"interface.{r}"} for r, _c, _m, summ in _ROLES
        ]
        + [{"name": op, "summary": summ, "source_interface": f"interface.{r}"} for r, op, summ in vector_roles],
        "types": [{"name": t} for t in types],
        "lowering": [{"from": f"interface.{r}", "to": f"{dname}.{role_op[r]}"} for r, _c, _m, _s in _ROLES]
        + [{"from": f"interface.{r}", "to": f"{dname}.{op}"} for r, op, _s in vector_roles],
        "tests": [{"lit": "pack_roundtrip"}, {"lit": "matmul_commit_epilogue"}, {"lit": "evict_after_use"}],
        "confidence": "medium",
        "requires_human_review": False,
        "generated_from_contract": True,
    }
    from merlin.common.record_formats import validate_or_raise

    validate_or_raise(plan, "dialect_plan")
    return plan


def synthesize_dialect_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Use the selected support plan or derive from this contract when it has no plan.

    Malformed present plans refuse; absence permits a generated tensor-resident plan
    or a review-flagged conservative skeleton, never another provider's authored plan.
    """
    name = target_contract.get("name")
    typed = _generate_from_operation_capabilities(target_contract)
    curated = _curated(name)
    if typed is not None and curated is not None:
        raise ValueError("selected typed operation contract and separate dialect plan duplicate dialect authority")
    if curated is not None:
        return curated
    if typed is not None:
        return typed
    if _is_tensor_resident(target_contract):
        return _generate(target_contract)  # usable generated plan (was: empty _conservative stub)
    return _conservative(evidence)


RUNTIME_ABI_VERSION = "0.1"


def _command_stream_adapter(target: str, *, reviewed: bool) -> dict[str, Any]:
    """The concrete command-stream adapter for a command-buffer tensor-resident target — the family
    default parameterized by target name."""
    dialect = target.replace("_", "")
    return {
        "target": target,
        "implements": {"runtime_abi_version": RUNTIME_ABI_VERSION},
        "command_encoding": {
            "format": f"{dialect}_command_stream",
            "supports_batching": True,
            "supports_async": False,
        },
        "queues": [{"name": "compute"}, {"name": "dma"}],
        "events": {"completion": "polling", "interrupts": "optional"},
        "handles": {
            "resident_tensor": "resident_store_slot",
            "accumulator": "accumulator_slot",
        },
        "metrics": {
            "maps_to_common": {
                "bytes_moved": ["pack_bytes", "commit_bytes"],
                "command_count": ["command_count"],
                "pack_count": ["pack_count"],
                "resident_hits": ["resident_hits"],
                "evictions": ["evictions"],
                "accumulator_commits": ["accumulator_commits"],
            },
            "target_specific": [],
        },
        "confidence": "high" if reviewed else "medium",
        "requires_human_review": not reviewed,
    }


def _is_command_buffer_resident(tc: dict[str, Any]) -> bool:
    """A contract that implements the Merlin command-buffer tensor-resident runtime (packs a resident
    weight, commits an accumulator, driven by a command buffer). Detected from the contract, not a name."""
    feats = set(tc.get("features") or [])
    return "command_buffer" in feats and bool(feats & {"resident_packed_tensor", "accumulator_commit"})


def _runtime_unknown(evidence: Evidence) -> dict[str, Any]:
    return {
        "target": evidence.target,
        "implements": {"runtime_abi_version": RUNTIME_ABI_VERSION},
        "command_encoding": {
            "format": "TODO_human_review",
            "supports_batching": False,
            "supports_async": False,
        },
        "queues": [],
        "events": {"completion": "TODO", "interrupts": "TODO"},
        "handles": {},
        "metrics": {"maps_to_common": {}, "target_specific": []},
        "detected_concepts": sorted(evidence.concept_names()),
        "confidence": "low",
        "requires_human_review": True,
    }


def synthesize_runtime_adapter_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return a runtime_adapter_plan dict for the contract's target: a command-buffer tensor-resident
    contract (the neutral toy_npu example is the family default) realizes the concrete command-stream
    adapter; everything else gets the review-flagged skeleton. Keyed on the contract family, not a name."""
    name = target_contract.get("name")
    if _is_command_buffer_resident(target_contract):
        reviewed = target_contract.get("confidence") == "high" and target_contract.get("requires_human_review") is False
        return _command_stream_adapter(name, reviewed=reviewed)
    return _runtime_unknown(evidence)


def _kconfig_symbol(dialect: str) -> str:
    return "MERLIN_" + dialect.upper()


def _command_buffer_module(target: str, *, reviewed: bool) -> dict[str, Any]:
    """The concrete Zephyr module for a command-buffer tensor-resident target — the family default
    parameterized by target name."""
    dialect = target.replace("_", "")
    sym = _kconfig_symbol(dialect)
    return {
        "target": target,
        "module": {"name": "merlin_" + dialect},
        "devicetree": {
            "compatible": f"ucb,{dialect}",
            "properties": [
                "reg",
                "interrupts",
                "resident-store-bytes",
                "accumulator-entries",
                "command-queue-depth",
            ],
        },
        "kconfig": {"symbols": [sym, sym + "_PROFILING"]},
        "driver_api": {
            "mode": "blocking",
            "functions": ["merlin_submit", "merlin_wait", "merlin_get_metrics"],
        },
        "samples": [f"{dialect}_repeated_rhs_matmul"],
        "confidence": "high" if reviewed else "medium",
        "requires_human_review": not reviewed,
    }


def _zephyr_unknown(evidence: Evidence) -> dict[str, Any]:
    target = evidence.target
    sym = _kconfig_symbol(target)
    return {
        "target": target,
        "module": {"name": "merlin_" + target},
        "devicetree": {
            "compatible": f"ucb,{target}",
            "properties": ["reg", "interrupts"],
        },
        "kconfig": {"symbols": [sym]},
        "driver_api": {
            "mode": "blocking",
            "functions": ["merlin_submit", "merlin_wait", "merlin_get_metrics"],
        },
        "samples": [],
        "confidence": "low",
        "requires_human_review": True,
    }


def synthesize_zephyr_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return a zephyr_plan dict for the contract's target (concrete module for a command-buffer
    tensor-resident contract — toy_npu is the family default; else a skeleton). Not keyed on a name."""
    name = target_contract.get("name")
    if _is_command_buffer_resident(target_contract):
        reviewed = target_contract.get("confidence") == "high" and target_contract.get("requires_human_review") is False
        return _command_buffer_module(name, reviewed=reviewed)
    return _zephyr_unknown(evidence)


_COMMON_OOT = ["td_fragments", "intrinsic_headers", "lit_tests", "patch_series"]


_COMMON_TRIGGERS = [
    "new_target_backend_registration",
    "new_register_classes",
    "new_instruction_selection_patterns",
    "mc_encoding_decoding",
    "assembler_disassembler_support",
]


_BY_ENDPOINT: dict[str, tuple[Any, str, str]] = {
    "command_buffer": (
        False,
        "runtime_calls_or_command_buffer",
        "Executes via the target command buffer / simulator; no LLVM changes.",
    ),
    "inline_asm_insn": (
        False,
        "inline_asm_insn_on_stock_llvm",
        "Emit the target dialect as llvm.inline_asm/.insn on STOCK LLVM; patch LLVM "
        "only if custom-instruction emission is later required.",
    ),
    "external_backend": (
        False,
        "external_device_assembler",
        "Emit a device kernel.S the target's own assembler builds; no host LLVM fork.",
    ),
    "upstream_target": (
        "maybe",
        "rvv_intrinsics_or_existing_riscv_vector_path",
        "Vector/scalar lowers through the upstream LLVM RISC-V/RVV path; custom "
        "extensions may eventually require LLVM TableGen/backend changes.",
    ),
}


_DEFAULT = (
    False,
    "runtime_calls_or_command_buffer",
    "Default out-of-tree posture (family-default seed); confirm against target source.",
)


def synthesize_llvm_extension_plan(evidence: Evidence, target_contract: dict[str, Any]) -> dict[str, Any]:
    """Return an llvm_extension_plan dict for the contract's target, fork posture DERIVED from its
    compute-unit family endpoint (family-default seed when the contract resolves no family)."""
    target = target_contract.get("name", evidence.target)
    endpoint = _families.contract_endpoint_kind(target_contract)
    grounded = endpoint in _BY_ENDPOINT
    fork, strategy, reason = _BY_ENDPOINT[endpoint] if grounded else _DEFAULT
    return {
        "target": target,
        "requires_llvm_fork": fork,
        "initial_strategy": strategy,
        "reason": reason,
        "out_of_tree_candidates": list(_COMMON_OOT),
        "fork_triggers": list(_COMMON_TRIGGERS),
        # The family-default seed is the well-understood reference (high confidence, no review); a real
        # family-resolved hardware target is flagged for human review.
        "confidence": "medium" if grounded else "high",
        "requires_human_review": grounded,
    }
