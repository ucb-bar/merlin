from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[3] / "examples/gemmini/verification/extract_rtl_timing_facts.py"
_SPEC = importlib.util.spec_from_file_location("rtl_timing_facts_example", _SOURCE)
F = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(F)


def _fixture(tmp_path):
    source, generated = tmp_path / "chipyard", tmp_path / "generated"
    (source / F.SOURCE).parent.mkdir(parents=True)
    (source / F.CHAIN).parent.mkdir(parents=True)
    (source / F.GEMMINI_CHAIN).parent.mkdir(parents=True)
    generated.mkdir()
    values = {
        "tileRows": 1,
        "tileColumns": 1,
        "meshRows": 16,
        "meshColumns": 16,
        "sp_banks": 4,
        "acc_banks": 2,
        "reservation_station_entries_ld": 8,
        "reservation_station_entries_ex": 16,
        "reservation_station_entries_st": 4,
        "ld_queue_length": 8,
        "ex_queue_length": 8,
        "st_queue_length": 2,
        "max_in_flight_mem_reqs": 16,
        "dma_maxbytes": 64,
        "dma_buswidth": 128,
        "num_counter": 8,
    }
    config = "object GemminiConfigs {\nval defaultConfig = GemminiArrayConfig[SInt, Float, Float](\n"
    config += "".join(f"  {key} = {value},\n" for key, value in values.items())
    config += "  sp_capacity = CapacityInKilobytes(256),\n  acc_capacity = CapacityInKilobytes(64),\n"
    config += (
        "  inputType = SInt(8.W),\n  weightType = SInt(8.W),\n  accType = SInt(32.W),\n)\nval dummyConfig = 0\n}\n"
    )
    config += (
        "class DefaultGemminiConfig[T,U,V](\n"
        " gemminiConfig: GemminiArrayConfig[T,U,V] = GemminiConfigs.defaultConfig\n"
        ") extends Config(())\n"
    )
    (source / F.SOURCE).write_text(config)
    (source / F.CHAIN).write_text(
        "class GemminiGsimFireSimAlignedConfig extends Config(\n"
        "new WithFireSimConfigTweaks ++\nnew chipyard.GemminiRocketConfig)\n"
    )
    (source / F.GEMMINI_CHAIN).write_text(
        "class GemminiRocketConfig extends Config(\nnew gemmini.DefaultGemminiConfig ++ other)\n"
    )
    fir = generated / f"{F.STEM}.fir"
    fir.write_text(
        "  module ReservationStation :\n"
        "    output io : { counter : { event_signal : UInt<1>[45], external_values : UInt<32>[8] }}\n"
        "    reg entries_ld : { bits : UInt<1>}[8], clock\n"
        "    reg entries_ex : { bits : UInt<1>}[16], clock\n"
        "    reg entries_st : { bits : UInt<1>}[4], clock\n"
        "  module LoadController :\n    inst cmd_q of Queue8_GemminiCmd\n"
        "  module StoreController :\n    inst cmd_q of Queue2_GemminiCmd_4\n"
        "  module ExecuteController :\n    inst cmd_q of MultiHeadedQueue_1\n"
        "  module MultiHeadedQueue_1 :\n    reg regs : { bits : UInt<1>}[8], clock\n"
    )
    dts = generated / f"{F.STEM}.dts"
    dts.write_text(
        "/ {\n cpu@0 {\n i-cache-size = <32768>;\n d-cache-size = <32768>;\n"
        "timebase-frequency = <1000000>;\n nested { reg = <1>; };\n };\n"
        "memory@80000000 {\n reg = <0x0 0x80000000 0x4 0x0>;\n };\n"
        "cache-controller@2010000 {\n cache-size = <524288>;\n"
        "sifive,mshr-count = <12>;\n };\n};\n"
    )
    (generated / f"{F.STEM}.l2.json").write_text(
        json.dumps({"banks": [{"ways": 8, "blockBytes": 64, "setBits": list(range(10))}]})
    )
    return source, generated, fir, dts


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_extract_reconciles_source_elaboration_and_dts(tmp_path):
    source, generated, fir, dts = _fixture(tmp_path)
    facts = F.extract(source, generated, firrtl_sha256=_sha(fir), dts_sha256=_sha(dts))
    assert facts["gemmini"]["ex_queue_length"] == 8
    assert facts["gemmini"]["reservation_station_entries_ex"] == 16
    assert facts["soc"]["memory_bytes"] == 16 * 1024**3
    assert facts["interpretation"]["timing_prices"] is None


def test_extract_refuses_source_rtl_mismatch_even_with_current_hash(tmp_path):
    source, generated, fir, dts = _fixture(tmp_path)
    fir.write_text(fir.read_text().replace("entries_ld : { bits : UInt<1>}[8]", "entries_ld : { bits : UInt<1>}[7]"))
    with pytest.raises(F.FactError, match="disagree on reservation_station_entries_ld"):
        F.extract(source, generated, firrtl_sha256=_sha(fir), dts_sha256=_sha(dts))


def test_extract_refuses_stale_elaboration_hash(tmp_path):
    source, generated, fir, dts = _fixture(tmp_path)
    with pytest.raises(F.FactError, match="firrtl SHA256"):
        F.extract(source, generated, firrtl_sha256="0" * 64, dts_sha256=_sha(dts))


def test_spike_profile_alignment_refuses_wrong_memory_or_geometry(tmp_path):
    source, generated, fir, dts = _fixture(tmp_path)
    facts = F.extract(source, generated, firrtl_sha256=_sha(fir), dts_sha256=_sha(dts))
    profile = {
        "schema": "gemmini.annotated_spike.target.v1",
        "target_config": "FireSimGemminiRocketConfig",
        "provenance": {
            "dram_base": facts["soc"]["memory_base"],
            "dram_bytes": facts["soc"]["memory_bytes"],
            "host_dts_sha256": facts["evidence_sha256"]["dts"],
        },
        "target": {
            "hardware": {
                "dim": 16,
                "element_bytes": 1,
                "accumulator_bytes": 4,
                "scratchpad_rows": 16384,
                "accumulator_rows": 1024,
            }
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert F.verify_spike_profile(facts, path)["status"] == "structural_match"
    profile["provenance"]["dram_bytes"] //= 2
    path.write_text(json.dumps(profile))
    with pytest.raises(F.FactError, match="provenance.dram_bytes"):
        F.verify_spike_profile(facts, path)
