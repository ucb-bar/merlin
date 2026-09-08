"""Cyclotron validates full outputs without UART-streaming every element."""
from __future__ import annotations

import struct

import pytest

from merlin.runtime.backends.base import get_backend
from merlin.targetgen.isa_model import IsaModel


BACKEND = get_backend("muon")
H = BACKEND.muon_harness
MO = BACKEND.muon_oracles
MU = BACKEND.muon
MODEL = IsaModel(target="synthetic", runtime_abi={
    "special_csrs": {"mhartid": 0xF14},
    "apertures": {"console_mmio": 0xFF080000},
})


def _u32(blob: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(blob) // 4}I", blob))


def _cb() -> dict:
    return {
        "target": "radiance",
        "tensors": {
            "X": {"shape": [1], "dtype": "f32", "role": "input"},
            "Y": {"shape": [1], "dtype": "f32", "role": "output"},
        },
        "commands": [{"opcode": "VECTOR_MAP", "operands": {"arg0": "X", "dst": "Y"}}],
        "canonical_inputs": {"X": {"shape": [1], "values": [1.0]}},
        "_oracle_expected_outputs": {"Y": [1.0]},
        "_oracle_numeric_policy": {"compare": "tolerance_float", "atol": 0.0, "rtol": 0.0},
    }


def test_compact_harness_checks_every_word_and_does_not_print_output_values() -> None:
    harness = H.build_external_kernel_main(
        [H.TensorArg("X", 1, 2, [0.0, 0.0], "f32")],
        [H.TensorArg("Y", 1, 2, [0.0, 0.0], "f32")],
        kernel_symbol="kernel", model=MODEL,
        compact_expected={"Y": [1.0, -2.0]},
        compact_policy={"compare": "tolerance_float", "atol": 0.0, "rtol": 0.0},
    )

    assert '_ps("OUT Y' not in harness.source
    assert '_ps("MERLIN_NUMERIC ")' in harness.source
    assert "for(uint32_t _i=0;_i<2u;++_i)" in harness.source
    assert "_merlin_bad+=_nan||" in harness.source
    assert _u32(harness.blobs["_merlin_expected_lo_0"]) == [0x3F800000, 0xC0000000]
    assert _u32(harness.blobs["_merlin_expected_hi_0"]) == [0x3F800000, 0xC0000000]
    assert harness.results == [{"name": "Y", "elements": 2, "dtype": "f32"}]


def test_compact_expected_is_private_and_changes_the_linked_bounds_only() -> None:
    common = dict(
        in_args=[H.TensorArg("X", 1, 1, [0.0], "f32")],
        out_args=[H.TensorArg("Y", 1, 1, [0.0], "f32")],
        kernel_symbol="kernel", model=MODEL,
        compact_policy={"compare": "exact"},
    )
    positive = H.build_external_kernel_main(**common, compact_expected={"Y": [1.0]})
    perturbed = H.build_external_kernel_main(**common, compact_expected={"Y": [2.0]})

    assert positive.source == perturbed.source
    assert positive.blobs["_merlin_expected_lo_0"] != perturbed.blobs["_merlin_expected_lo_0"]
    assert b"1.0" not in positive.blobs["_merlin_expected_lo_0"]


@pytest.mark.parametrize(("marker", "status", "mismatches"), [
    ("MERLIN_NUMERIC PASS 113 0\nDONE\n", "pass", 0),
    ("MERLIN_NUMERIC FAIL 113 1\nDONE\n", "fail", 1),
])
def test_compact_parser_requires_explicit_consistent_verdict(marker, status, mismatches) -> None:
    verdict = MO._compact_numeric_from_console(marker, expected_elements=113)
    assert verdict["status"] == status
    assert verdict["elements_checked"] == 113
    assert verdict["mismatch_count"] == mismatches


@pytest.mark.parametrize("console", [
    "DONE\n",
    "MERLIN_NUMERIC PASS 1 0\n",
    "MERLIN_NUMERIC PASS 1 0\nMERLIN_NUMERIC PASS 1 0\nDONE\n",
    "MERLIN_NUMERIC PASS 1 1\nDONE\n",
    "MERLIN_NUMERIC FAIL 1 0\nDONE\n",
    "MERLIN_NUMERIC PASS 0 0\nDONE\n",
    "MERLIN_NUMERIC PASS 1 0\nDONE\nDONE\n",
])
def test_compact_parser_fails_closed_on_missing_or_inconsistent_evidence(console) -> None:
    with pytest.raises(MU.MuonError):
        MO._compact_numeric_from_console(console, expected_elements=1)


def test_compact_parser_rejects_partial_output_check() -> None:
    with pytest.raises(MU.MuonError):
        MO._compact_numeric_from_console(
            "MERLIN_NUMERIC PASS 1 0\nDONE\n", expected_elements=256)


def test_cyclotron_adapter_reports_perturbed_result_as_numeric_fail(monkeypatch, tmp_path) -> None:
    compiled = {}

    monkeypatch.setattr(MU, "available", lambda simulator: simulator == "cyclotron")
    monkeypatch.setattr(MU, "is_mlir_artifact", lambda source: True)

    def compile_stub(source, cb, workdir, **kwargs):
        compiled.update(kwargs)
        return tmp_path / "kernel.elf"

    monkeypatch.setattr(MU, "compile_mlir_forkfree", compile_stub)
    monkeypatch.setattr(
        MU, "run_elf",
        lambda *args, **kwargs: ("MERLIN_NUMERIC FAIL 1 1\nDONE\n"
                                 "simulation finished after 100 cycles\n", 100, {}),
    )

    result = MO.cyclotron_adapter()(_cb(), "builtin.module { llvm.func @k() }", tmp_path, 60)

    assert compiled["compact_expected"] == {"Y": [1.0]}
    assert compiled["compact_policy"]["compare"] == "tolerance_float"
    assert result["outputs"] == {}
    assert result["numeric_verdict"]["status"] == "fail"
    assert result["numeric_verdict"]["mismatch_count"] == 1
    assert result["numeric_verdict"]["witness"] == "trusted_muon_post_kernel_comparator"
