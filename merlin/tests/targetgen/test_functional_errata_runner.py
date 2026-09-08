"""The functional target model consumes reviewed ISA corrections without mutating its source."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.targetgen.oracle_helpers.functional_errata_runner import (
    apply_reviewed_errata,
    apply_reviewed_model_errata,
)


def _instruction(**fields):
    return type("Instruction", (), fields)


def test_reviewed_fields_are_derived_from_complete_words() -> None:
    config = _instruction(opcode=0x7F, funct3=3, funct7=1)
    wait = _instruction(opcode=0x7F, funct3=3, funct7=1)
    module = SimpleNamespace(DMA_CONFIG_CH3=config, DMA_WAIT_CH3=wait)
    applied = apply_reviewed_errata(module, {
        "DMA_CONFIG_CH3": {
            "authoritative": "rtl",
            "declared": "0x0200307f",
            "hardware": "0x0000307f",
            "sources_against_spec": ["rtl_bitpat"],
        }
    })
    assert config.funct7 == 0
    assert wait.funct7 == 1
    assert applied == [{
        "class": "DMA_CONFIG_CH3",
        "declared": "0x0200307f",
        "hardware": "0x0000307f",
        "fields": {"funct7": {"declared": 1, "hardware": 0}},
        "sources_against_spec": ["rtl_bitpat"],
    }]


def test_funct3_and_vector_funct7_corrections_are_not_target_special_cases() -> None:
    csr = _instruction(opcode=0x73, funct3=4)
    vector = _instruction(opcode=0x57, funct7=0x4F)
    module = SimpleNamespace(CSRRCI=csr, VCUBE_BF16=vector)
    applied = apply_reviewed_errata(module, {
        "CSRRCI": {"authoritative": "rtl", "declared": "0x00004073",
                   "hardware": "0x00007073"},
        "VCUBE_BF16": {"authoritative": "rtl", "declared": "0x9e000057",
                       "hardware": "0x8e000057"},
    })
    assert csr.funct3 == 7
    assert vector.funct7 == 0x47
    assert [row["class"] for row in applied] == ["CSRRCI", "VCUBE_BF16"]


def test_stale_review_fails_closed_instead_of_patching_a_different_model() -> None:
    module = SimpleNamespace(OP=_instruction(opcode=0x7F, funct3=0, funct7=6))
    with pytest.raises(ValueError, match="does not match reviewed declared value"):
        apply_reviewed_errata(module, {
            "OP": {"authoritative": "rtl", "declared": "0x0200007f",
                   "hardware": "0x0000007f"}
        })


def test_non_rtl_or_missing_class_corrections_do_not_silently_apply() -> None:
    module = SimpleNamespace()
    assert apply_reviewed_errata(module, {
        "UNRESOLVED": {"authoritative": "unresolved", "declared": "0x1",
                       "hardware": "0x2"}
    }) == []
    with pytest.raises(ValueError, match="names no model class"):
        apply_reviewed_errata(module, {
            "MISSING": {"authoritative": "rtl", "declared": "0x1",
                        "hardware": "0x2"}
        })


def test_vmem_word_address_overlay_changes_only_named_engine_operations() -> None:
    observations = []

    class State:
        def read_vmem(self, base, offset, length):
            observations.append(("read", base, offset, length))
            return "data"

        def write_vmem(self, base, offset, data):
            observations.append(("write", base, offset, data))

    class EngineOp:
        def exec(self, state):
            assert state.read_vmem(7, 2, 9) == "data"
            state.write_vmem(11, 3, "value")

    module = SimpleNamespace(EngineOp=EngineOp)
    applied = apply_reviewed_model_errata(module, {
        "addressing": {"authoritative": "rtl", "correction": "vmem_base_unit_bytes",
                       "model_classes": ["EngineOp"], "declared_unit_bytes": 1,
                       "hardware_unit_bytes": 4}
    })
    EngineOp().exec(State())
    assert observations == [("read", 28, 2, 9), ("write", 44, 3, "value")]
    assert applied[0]["parameters"]["hardware_unit_bytes"] == 4


def test_e8m0_overlay_supplies_reciprocal_power_of_two_scale() -> None:
    observed = []

    class State:
        def read_erf(self, register):
            return {0: 127, 1: 130}[register]

    class ScaleOp:
        register = 0

        def exec(self, state):
            observed.append(state.read_erf(self.register))

    module = SimpleNamespace(ScaleOp=ScaleOp)
    apply_reviewed_model_errata(module, {
        "scale": {"authoritative": "rtl", "correction": "e8m0_biased_exponent",
                  "model_classes": ["ScaleOp"], "exponent_bias": 127,
                  "exponent_min": -128, "exponent_max": 127}
    })
    ScaleOp().exec(State())
    ScaleOp.register = 1
    ScaleOp().exec(State())
    assert observed == [1.0, 0.125]
