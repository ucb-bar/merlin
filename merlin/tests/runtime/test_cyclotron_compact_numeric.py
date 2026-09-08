"""Cyclotron validates full outputs without UART-streaming every element."""
from __future__ import annotations

import json
import struct
import subprocess

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
        compact_symbol_tag="n_test",
    )

    assert '_ps("OUT Y' not in harness.source
    assert '_ps("MERLIN_NUMERIC ")' in harness.source
    assert "for(uint32_t _i=0;_i<2u;++_i)" in harness.source
    assert "_merlin_bad+=_nan||" in harness.source
    assert _u32(harness.blobs["_merlin_private_n_test_lo_0"]) == [0x3F800000, 0xC0000000]
    assert _u32(harness.blobs["_merlin_private_n_test_hi_0"]) == [0x3F800000, 0xC0000000]
    assert harness.results == [{"name": "Y", "elements": 2, "dtype": "f32"}]


def test_compact_expected_is_private_and_changes_the_linked_bounds_only() -> None:
    common = dict(
        in_args=[H.TensorArg("X", 1, 1, [0.0], "f32")],
        out_args=[H.TensorArg("Y", 1, 1, [0.0], "f32")],
        kernel_symbol="kernel", model=MODEL,
        compact_policy={"compare": "exact"},
        compact_symbol_tag="n_test",
    )
    positive = H.build_external_kernel_main(**common, compact_expected={"Y": [1.0]})
    perturbed = H.build_external_kernel_main(**common, compact_expected={"Y": [2.0]})

    assert positive.source == perturbed.source
    assert positive.blobs["_merlin_private_n_test_lo_0"] != \
        perturbed.blobs["_merlin_private_n_test_lo_0"]
    assert b"1.0" not in positive.blobs["_merlin_private_n_test_lo_0"]


def test_legacy_predictable_answer_symbol_cannot_name_compact_key() -> None:
    harness = H.build_external_kernel_main(
        [H.TensorArg("X", 1, 1, [0.0], "f32")],
        [H.TensorArg("Y", 1, 1, [0.0], "f32")],
        kernel_symbol="kernel", model=MODEL,
        compact_expected={"Y": [1.0]}, compact_policy={"compare": "exact"},
        compact_symbol_tag="n_4c02f13a2b8d4e61",
    )

    malicious_legacy_reference = "extern float _merlin_expected_lo_0[];"
    assert "_merlin_expected_lo_0" in malicious_legacy_reference
    assert all("_merlin_expected_lo_0" not in symbol for symbol in harness.blobs)
    assert "_merlin_expected_lo_0" not in harness.source
    assert all("n_4c02f13a2b8d4e61" in symbol for symbol in harness.blobs
               if symbol.startswith("_merlin_private_"))


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

    monkeypatch.setenv("MERLIN_MUON_TRUSTED_COMPACT_NUMERIC", "1")
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
    assert result["numeric_verdict"]["trust_scope"] == \
        "frozen_non_adversarial_derived_evaluation_only"


def test_cyclotron_compact_path_requires_explicit_non_adversarial_opt_in(
        monkeypatch, tmp_path) -> None:
    compiled = {}
    monkeypatch.delenv("MERLIN_MUON_TRUSTED_COMPACT_NUMERIC", raising=False)
    monkeypatch.setattr(MU, "available", lambda simulator: simulator == "cyclotron")
    monkeypatch.setattr(MU, "is_mlir_artifact", lambda source: True)

    def compile_stub(source, cb, workdir, **kwargs):
        compiled.update(kwargs)
        return tmp_path / "kernel.elf"

    monkeypatch.setattr(MU, "compile_mlir_forkfree", compile_stub)
    monkeypatch.setattr(
        MU, "run_elf",
        lambda *args, **kwargs: ("OUT Y 1 1 1.0\nDONE\n"
                                 "simulation finished after 100 cycles\n", 100, {}),
    )

    result = MO.cyclotron_adapter()(_cb(), "builtin.module { llvm.func @k() }", tmp_path, 60)

    assert compiled["compact_expected"] is None
    assert result["outputs"] == {"Y": [[1.0]]}
    assert "numeric_verdict" not in result


def test_cyclotron_run_config_corrects_geometry_and_preserves_timing_includes() -> None:
    source = """[muon]
num_lanes = 16
num_warps = 8
num_cores = 2 # stale model default

[sim]
timeout = 1000000

[timing]
include = [
  "config/timing/gmem.toml",
  "config/timing/execute.toml",
]
"""
    capacity = {
        "cores": 1, "warps_per_core": 8, "lanes_per_warp": 16,
        "source": "reviewed RTL facts",
    }

    rendered, record = MU._cyclotron_run_config(
        source, capacity, timeout_cycles=20_000_000)

    assert "num_cores = 1 # stale model default" in rendered
    assert "timeout = 20000000" in rendered
    assert ('include = [\n  "config/timing/gmem.toml",\n'
            '  "config/timing/execute.toml",\n]') in rendered
    assert record["rtl"] == {"num_cores": 1, "num_warps": 8, "num_lanes": 16}
    assert record["corrected"] == {"num_cores": {"from": 2, "to": 1}}


@pytest.mark.parametrize("source", [
    "[muon]\nnum_lanes=16\nnum_warps=8\n[sim]\ntimeout=1\n",
    ("[muon]\nnum_lanes=16\nnum_warps=8\nnum_cores=2\nnum_cores=1\n"
     "[sim]\ntimeout=1\n"),
])
def test_cyclotron_run_config_refuses_missing_or_duplicate_geometry(source) -> None:
    with pytest.raises(MU.MuonUnavailable):
        MU._cyclotron_run_config(
            source,
            {"cores": 1, "warps_per_core": 8, "lanes_per_warp": 16},
            timeout_cycles=20_000_000,
        )


def test_cyclotron_exact_cycle_cap_is_unknown_not_tool_crash(monkeypatch, tmp_path) -> None:
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"elf")
    config = tmp_path / "config.toml"
    config.write_text("""[muon]
num_lanes = 16
num_warps = 8
num_cores = 1
[sim]
timeout = 1
""", encoding="utf-8")
    run = tmp_path / "performance_logs" / "run_cap"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "total": {"scheduler": {"cycles": 20_000_000}},
    }), encoding="utf-8")

    monkeypatch.setattr(MU, "config_path", lambda: config)
    monkeypatch.setattr(MU, "cyclotron_path", lambda: tmp_path / "cyclotron")
    monkeypatch.setattr(MU, "cyclotron_root", lambda: tmp_path)
    monkeypatch.setattr(MU, "_rtl_machine_capacity", lambda target: {
        "cores": 1, "warps_per_core": 8, "lanes_per_warp": 16,
        "source": "test RTL facts",
    })
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], 1, stdout="", stderr="Error: 0\n"))

    with pytest.raises(MU.MuonUnavailable, match=(
            "configured 20000000-cycle cap.*correctness is unknown")):
        MU._run_cyclotron(elf, 60)


@pytest.mark.parametrize(("returncode", "stderr", "summary_cycles"), [
    (2, "Error: 0\n", 20_000_000),
    (1, "a genuine simulator error\n", 20_000_000),
    (1, "Error: 0\n", 19_999_999),
])
def test_cyclotron_does_not_mask_non_cap_failures(
        monkeypatch, tmp_path, returncode, stderr, summary_cycles) -> None:
    elf = tmp_path / "kernel.elf"
    elf.write_bytes(b"elf")
    config = tmp_path / "config.toml"
    config.write_text("""[muon]
num_lanes = 16
num_warps = 8
num_cores = 1
[sim]
timeout = 1
""", encoding="utf-8")
    run = tmp_path / "performance_logs" / "run_failure"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "total": {"scheduler": {"cycles": summary_cycles}},
    }), encoding="utf-8")

    monkeypatch.setattr(MU, "config_path", lambda: config)
    monkeypatch.setattr(MU, "cyclotron_path", lambda: tmp_path / "cyclotron")
    monkeypatch.setattr(MU, "cyclotron_root", lambda: tmp_path)
    monkeypatch.setattr(MU, "_rtl_machine_capacity", lambda target: {
        "cores": 1, "warps_per_core": 8, "lanes_per_warp": 16,
        "source": "test RTL facts",
    })
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(
        args[0], returncode, stdout="", stderr=stderr))

    with pytest.raises(MU.MuonError, match=f"cyclotron exited {returncode}"):
        MU._run_cyclotron(elf, 60)


def test_fp_rates_use_rtl_derived_single_core_capacity(monkeypatch) -> None:
    monkeypatch.setattr(MU, "_rtl_machine_capacity", lambda target: {
        "clock_hz": 500_000_000,
        "peak_flops_per_cycle": 32,
    })

    assert MU.gflops(32, 1, target="radiance") == 16.0
    assert MU.pct_fp_peak(32, 1, target="radiance") == 100.0
