"""Instruction-level capabilities stay data-driven across self-hosted targets."""
from __future__ import annotations

import pytest

from merlin.targetgen.operation_capabilities import (
    derive_dialect_operation_contract,
    derive_isa_operation_contract,
    memory_operation_prompt_block,
    merge_operation_contracts,
    merge_operation_observations,
    scalar_memory_prompt_block,
)


def _taxonomy() -> dict:
    return {
        "by_mnemonic": {
            "LD8": {
                "class": "SyntheticLoad",
                "scalar_memory": {
                    "direction": "load",
                    "address_space": "local_mem",
                    "width_bytes": 1,
                    "address_unit_bytes": 1,
                    "addressing": {
                        "mode": "base_plus_immediate",
                        "base_operand": "rs1",
                        "offset_operand": "imm",
                        "offset_bits": 12,
                        "offset_scale": 1,
                    },
                    "effect_method": "read_local_mem",
                },
            },
            "ST8": {
                "class": "SyntheticStore",
                "scalar_memory": {
                    "direction": "store",
                    "address_space": "local_mem",
                    "width_bytes": 1,
                    "address_unit_bytes": 1,
                    "addressing": {
                        "mode": "base_plus_immediate",
                        "base_operand": "rs1",
                        "offset_operand": "imm",
                        "offset_bits": 12,
                        "offset_scale": 1,
                    },
                    "effect_method": "write_local_mem",
                },
            },
            "ADD": {"class": "SyntheticAlu", "role": "scalar"},
        }
    }


def test_contract_records_only_discovered_scalar_memory_operations():
    contract = derive_isa_operation_contract(_taxonomy(), dialect="synthetic.isa")
    by_name = {op["operation"]: op for op in contract["operations"]}

    # The outer record is intentionally generic: RVV instructions and target-dialect ops can use the
    # same domain/dialect/operation/status/effects envelope. Scalar memory is one semantic facet within it.
    assert {k: by_name["ADD"][k] for k in ("domain", "dialect", "operation")} == {
        "domain": "instruction", "dialect": "synthetic.isa", "operation": "ADD"}
    assert by_name["LD8"]["effects"] == ["movement"]
    assert by_name["LD8"]["semantics"] == {
        "kind": "memory",
        "scope": "scalar",
        "direction": "load",
        "address_space": "local_mem",
        "width_bytes": 1,
        "address_unit_bytes": 1,
        "addressing": {
            "mode": "base_plus_immediate",
            "base_operand": "rs1",
            "offset_operand": "imm",
            "offset_bits": 12,
            "offset_scale": 1,
        },
    }
    assert by_name["LD8"]["status"] == "unknown"
    assert by_name["LD8"]["evidence"] == [
        {"kind": "isa_definition", "detail": "read_local_mem"},
    ]


def test_target_dialect_and_machine_instructions_share_one_operation_envelope():
    dialect = derive_dialect_operation_contract({
        "dialect_name": "synthetic",
        "ops": [{"name": "matmul"}, {"name": "copy"}],
    })
    machine = derive_isa_operation_contract(_taxonomy(), dialect="synthetic")

    merged = merge_operation_contracts(dialect, machine)
    identities = {(op["domain"], op["dialect"], op["operation"])
                  for op in merged["operations"]}

    assert ("dialect", "synthetic", "matmul") in identities
    assert ("instruction", "synthetic", "LD8") in identities
    assert all(set(("domain", "dialect", "operation", "status", "evidence")) <= set(op)
               for op in merged["operations"])


def test_memory_address_units_can_differ_across_dialects_and_render_explicitly():
    dialect = derive_dialect_operation_contract({
        "dialect_name": "synthetic",
        "ops": [{
            "name": "vector_load",
            "effects": ["movement"],
            "semantics": {
                "kind": "memory", "scope": "vector", "direction": "load",
                "address_space": "local_mem", "address_unit_bytes": 4,
                "addressing": {
                    "mode": "base_plus_immediate", "base_operand": "rs1",
                    "offset_operand": "imm", "offset_scale": 32,
                },
            },
        }],
    })
    machine = derive_isa_operation_contract(_taxonomy(), dialect="synthetic")

    merged = merge_operation_contracts(dialect, machine)
    by_name = {op["operation"]: op for op in merged["operations"]}

    assert by_name["LD8"]["semantics"]["address_unit_bytes"] == 1
    assert by_name["vector_load"]["semantics"]["address_unit_bytes"] == 4
    assert by_name["vector_load"]["semantics"]["addressing"]["offset_scale"] == 32
    text = memory_operation_prompt_block({"operation_capabilities": merged})
    assert "address unit: 1 byte" in text
    assert "address unit: 4 bytes" in text
    assert "32 address units (128 bytes)" in text


def test_behavioral_observation_can_mark_one_declared_operation_unsupported():
    declared = derive_isa_operation_contract(_taxonomy(), dialect="synthetic.isa")
    observed = merge_operation_observations(declared, [{
        "domain": "instruction", "dialect": "synthetic.isa", "operation": "ST8",
        "status": "unsupported",
        "evidence": {
            "kind": "rtl_preflight",
            "detail": "store retired but addressed byte did not change",
        },
    }])

    by_name = {op["operation"]: op for op in observed["operations"]}
    assert by_name["LD8"]["status"] == "unknown"
    assert by_name["ST8"]["status"] == "unsupported"
    assert by_name["ST8"]["evidence"][-1]["kind"] == "rtl_preflight"


def test_observation_cannot_invent_an_operation_or_status():
    declared = derive_isa_operation_contract(_taxonomy(), dialect="synthetic.isa")
    with pytest.raises(ValueError, match="not declared"):
        merge_operation_observations(declared, [{
            "domain": "instruction", "dialect": "synthetic.isa", "operation": "NOPE",
            "status": "supported",
        }])
    with pytest.raises(ValueError, match="status"):
        merge_operation_observations(declared, [{
            "domain": "instruction", "dialect": "synthetic.isa", "operation": "LD8",
            "status": "flaky",
        }])


def test_prompt_surfaces_unsupported_ops_as_a_codegen_prohibition():
    declared = derive_isa_operation_contract(_taxonomy(), dialect="synthetic.isa")
    observed = merge_operation_observations(declared, [{
        "domain": "instruction", "dialect": "synthetic.isa", "operation": "ST8",
        "status": "unsupported",
        "evidence": {"kind": "rtl_preflight", "detail": "no memory effect"},
    }])

    text = scalar_memory_prompt_block({"operation_capabilities": observed})

    assert "`synthetic.isa.LD8`" in text and "unverified" in text
    assert "`synthetic.isa.ST8`" in text and "UNSUPPORTED" in text
    assert "must not emit" in text
    assert "local_mem" in text and "base + signed 12-bit immediate" in text


def test_manifest_joins_compute_unit_dialect_and_discovered_isa(monkeypatch):
    from merlin.targetgen import capability_manifests as cm
    from merlin.targetgen import isa_taxonomy

    monkeypatch.setattr(isa_taxonomy, "taxonomy_for_target", lambda target: _taxonomy())
    residual = {
        "version": "0.1",
        "compute_units": [{
            "name": "unit", "kind": "vector", "ops": ["matmul"], "dtypes": ["fp32"],
        }],
        "operation_capabilities": {
            "version": 1,
            "operations": [],
            "observations": [{
                "domain": "instruction", "dialect": "syntheticcore", "operation": "ST8",
                "status": "unsupported",
                "evidence": {"kind": "rtl_preflight", "detail": "no memory effect"},
            }],
        },
    }

    manifest = cm.derive_manifest({"target": "synthetic_core", "kind": "vector"}, {}, residual=residual)
    by_id = {(op["domain"], op["dialect"], op["operation"]): op
             for op in manifest["operation_capabilities"]["operations"]}

    assert ("dialect", "syntheticcore", "matmul") in by_id
    assert ("instruction", "syntheticcore", "LD8") in by_id
    assert by_id[("instruction", "syntheticcore", "ST8")]["status"] == "unsupported"
