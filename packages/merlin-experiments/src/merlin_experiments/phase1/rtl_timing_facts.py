"""Extract structural Gemmini facts from one pinned source and gSIM elaboration.

This is an operator-side evidence tool. Structural limits are not timing prices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


class FactError(ValueError):
    pass


SOURCE = Path("generators/gemmini/src/main/scala/gemmini/Configs.scala")
CHAIN = Path("generators/firechip/chip/src/main/scala/GemminiGsimFireSimAlignedConfig.scala")
GEMMINI_CHAIN = Path("generators/gemmini/chipyard/GemminiConfigs.scala")
STEM = "chipyard.harness.TestHarness.GemminiGsimFireSimAlignedConfig"
FIELDS = (
    "tileRows",
    "tileColumns",
    "meshRows",
    "meshColumns",
    "sp_banks",
    "acc_banks",
    "reservation_station_entries_ld",
    "reservation_station_entries_ex",
    "reservation_station_entries_st",
    "ld_queue_length",
    "ex_queue_length",
    "st_queue_length",
    "max_in_flight_mem_reqs",
    "dma_maxbytes",
    "dma_buswidth",
    "num_counter",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_values(text: str) -> dict[str, int]:
    match = re.search(r"\bval defaultConfig\s*=\s*GemminiArrayConfig\[", text)
    end = re.search(r"\bval dummyConfig\s*=", text)
    if not match or not end or end.start() <= match.start():
        raise FactError("cannot isolate Gemmini defaultConfig")
    body = text[match.start() : end.start()]
    values = {}
    for field in FIELDS:
        hits = re.findall(rf"(?m)^\s*{re.escape(field)}\s*=\s*(\d+)\s*,", body)
        if len(hits) != 1 or int(hits[0]) <= 0:
            raise FactError(f"defaultConfig has no unique positive {field}")
        values[field] = int(hits[0])
    for field in ("sp_capacity", "acc_capacity"):
        hits = re.findall(rf"(?m)^\s*{field}\s*=\s*CapacityInKilobytes\((\d+)\)\s*,", body)
        if len(hits) != 1 or int(hits[0]) <= 0:
            raise FactError(f"defaultConfig has no unique positive {field}")
        values[field + "_bytes"] = int(hits[0]) * 1024
    for field, width in (("inputType", 8), ("weightType", 8), ("accType", 32)):
        if len(re.findall(rf"(?m)^\s*{field}\s*=\s*SInt\({width}\.W\)\s*,", body)) != 1:
            raise FactError(f"defaultConfig {field} is not the expected signed {width}-bit type")
    return values


def _firrtl_values(path: Path) -> dict[str, int]:
    modules = {"ReservationStation", "LoadController", "ExecuteController", "StoreController", "MultiHeadedQueue_1"}
    found: dict[str, dict[str, int]] = {name: {} for name in modules}
    current = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            header = re.match(r"^  module ([A-Za-z0-9_]+)\s*:", line)
            if header:
                current = header.group(1) if header.group(1) in modules else None
                continue
            if current == "ReservationStation":
                counters = re.search(
                    r"counter\s*:\s*\{\s*event_signal\s*:\s*UInt<1>\[(\d+)\],\s*"
                    r"external_values\s*:\s*UInt<32>\[(\d+)\]",
                    line,
                )
                if counters:
                    if "counter_event_signals" in found[current]:
                        raise FactError("duplicate FIRRTL counter interface")
                    found[current]["counter_event_signals"] = int(counters.group(1))
                    found[current]["counter_external_values"] = int(counters.group(2))
                entry = re.match(r"^    reg entries_(ld|ex|st)\s*:.*\}\[(\d+)\], clock\b", line)
                if entry:
                    key = "reservation_station_entries_" + entry.group(1)
                    if key in found[current]:
                        raise FactError(f"duplicate FIRRTL {key}")
                    found[current][key] = int(entry.group(2))
            elif current == "MultiHeadedQueue_1":
                regs = re.match(r"^    reg regs\s*:.*\}\[(\d+)\], clock\b", line)
                if regs:
                    found[current]["queue_length"] = int(regs.group(1))
            elif current == "ExecuteController":
                if re.match(r"^    inst cmd_q of MultiHeadedQueue_1\b", line):
                    found[current]["uses_multiheaded_queue"] = 1
            elif current in modules:
                queue = re.match(r"^    inst cmd_q of Queue(\d+)_GemminiCmd(?:_\d+)?\b", line)
                if queue:
                    if found[current]:
                        raise FactError(f"duplicate FIRRTL cmd_q in {current}")
                    found[current]["queue_length"] = int(queue.group(1))
    expected = {
        "ReservationStation": {
            "reservation_station_entries_ld",
            "reservation_station_entries_ex",
            "reservation_station_entries_st",
            "counter_event_signals",
            "counter_external_values",
        },
        "LoadController": {"queue_length"},
        "ExecuteController": {"uses_multiheaded_queue"},
        "StoreController": {"queue_length"},
        "MultiHeadedQueue_1": {"queue_length"},
    }
    for module, keys in expected.items():
        if set(found[module]) != keys:
            raise FactError(f"FIRRTL {module} is missing expected entry or command queue evidence")
    return {
        **found["ReservationStation"],
        "ld_queue_length": found["LoadController"]["queue_length"],
        "ex_queue_length": found["MultiHeadedQueue_1"]["queue_length"],
        "st_queue_length": found["StoreController"]["queue_length"],
    }


def _node(text: str, name: str) -> str:
    hits = list(re.finditer(rf"(?m)^\s*(?:\w+:\s*)?{re.escape(name)}\s*\{{", text))
    if len(hits) != 1:
        raise FactError(f"DTS has no unique {name} node")
    start = hits[0].end()
    depth = 1
    for offset in range(start, len(text)):
        if text[offset] == "{":
            depth += 1
        elif text[offset] == "}":
            depth -= 1
            if depth == 0:
                return text[start:offset]
    raise FactError(f"DTS {name} node is not closed")


def _property(node: str, name: str) -> int:
    hits = re.findall(rf"(?m)^\s*{re.escape(name)}\s*=\s*<(\d+)>;", node)
    if len(hits) != 1:
        raise FactError(f"DTS has no unique integer {name}")
    return int(hits[0])


def _dts_values(text: str) -> dict:
    memory = _node(text, "memory@80000000")
    regs = re.findall(r"(?m)^\s*reg\s*=\s*<([^>]+)>;", memory)
    if len(regs) != 1:
        raise FactError("DTS target memory lacks unique reg")
    cells = [int(cell, 0) for cell in regs[0].split()]
    if len(cells) != 4:
        raise FactError("DTS target memory must use 2 address and 2 size cells")
    base, size = (cells[0] << 32) | cells[1], (cells[2] << 32) | cells[3]
    cpu = _node(text, "cpu@0")
    l2 = _node(text, "cache-controller@2010000")
    if base != 0x80000000 or size <= 0:
        raise FactError("DTS target memory base/size is unexpected")
    return {
        "memory_base": base,
        "memory_bytes": size,
        "l1i_bytes": _property(cpu, "i-cache-size"),
        "l1d_bytes": _property(cpu, "d-cache-size"),
        "l2_bytes": _property(l2, "cache-size"),
        "l2_mshr_count": _property(l2, "sifive,mshr-count"),
        "timebase_hz": _property(cpu, "timebase-frequency"),
    }


def extract(chipyard_source: Path, generated: Path, *, firrtl_sha256: str, dts_sha256: str) -> dict:
    paths = {
        "gemmini_config_source": chipyard_source / SOURCE,
        "gsim_config_source": chipyard_source / CHAIN,
        "chipyard_gemmini_config_source": chipyard_source / GEMMINI_CHAIN,
        "firrtl": generated / f"{STEM}.fir",
        "dts": generated / f"{STEM}.dts",
        "l2_json": generated / f"{STEM}.l2.json",
    }
    sha = {key: _sha(path) for key, path in paths.items()}
    for key, expected in (("firrtl", firrtl_sha256), ("dts", dts_sha256)):
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha[key] != expected:
            raise FactError(f"{key} SHA256 does not match the selected elaboration")
    chain = paths["gsim_config_source"].read_text()
    gemmini_chain = paths["chipyard_gemmini_config_source"].read_text()
    config_source = paths["gemmini_config_source"].read_text()
    if not re.search(
        r"class GemminiGsimFireSimAlignedConfig extends Config\([\s\S]*?"
        r"new WithFireSimConfigTweaks\s*\+\+\s*new chipyard.GemminiRocketConfig\)",
        chain,
    ):
        raise FactError("gSIM config does not select the FireSim tweak and GemminiRocketConfig chain")
    if not re.search(
        r"class GemminiRocketConfig extends Config\([\s\S]*?new gemmini.DefaultGemminiConfig", gemmini_chain
    ):
        raise FactError("GemminiRocketConfig does not select DefaultGemminiConfig")
    if not re.search(r"class DefaultGemminiConfig[\s\S]*?\) extends Config", config_source) or not re.search(
        r"gemminiConfig: GemminiArrayConfig\[T,U,V\] = GemminiConfigs\.defaultConfig", config_source
    ):
        raise FactError("DefaultGemminiConfig does not select defaultConfig")
    source = _source_values(config_source)
    rtl = _firrtl_values(paths["firrtl"])
    for key, value in rtl.items():
        if key in source and source[key] != value:
            raise FactError(f"source and FIRRTL disagree on {key}: {source[key]} != {value}")
    if rtl["counter_external_values"] != source["num_counter"]:
        raise FactError("source num_counter and FIRRTL external counter count disagree")
    dts = _dts_values(paths["dts"].read_text())
    l2 = json.loads(paths["l2_json"].read_text())
    banks = l2.get("banks")
    if not isinstance(banks, list) or len(banks) != 1:
        raise FactError("L2 JSON does not describe one bank")
    bank = banks[0]
    if (
        type(bank.get("ways")) is not int
        or type(bank.get("blockBytes")) is not int
        or not isinstance(bank.get("setBits"), list)
    ):
        raise FactError("L2 JSON lacks geometry")
    sets = 1 << len(bank["setBits"])
    if bank["ways"] * bank["blockBytes"] * sets != dts["l2_bytes"]:
        raise FactError("L2 JSON and DTS disagree on cache capacity")
    if dts["memory_bytes"] != 16 * 1024**3:
        raise FactError("selected elaboration is not the required 16 GiB target")
    return {
        "schema": "merlin.phase1.rtl_timing_facts.v1",
        "configuration": "GemminiGsimFireSimAlignedConfig",
        "evidence_sha256": sha,
        "gemmini": {
            "input_bits": 8,
            "weight_bits": 8,
            "accumulator_bits": 32,
            **source,
            "counter_event_signals": rtl["counter_event_signals"],
        },
        "soc": {**dts, "l2_ways": bank["ways"], "l2_block_bytes": bank["blockBytes"], "l2_sets": sets},
        "interpretation": {
            "structural_limits_only": True,
            "queue_lengths_are_not_timing_engine_resource_capacities": True,
            "target_cycle_frequency_hz": None,
            "timing_prices": None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chipyard-source", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--firrtl-sha256", required=True)
    parser.add_argument("--dts-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    facts = extract(args.chipyard_source, args.generated, firrtl_sha256=args.firrtl_sha256, dts_sha256=args.dts_sha256)
    if args.output.exists():
        parser.error("output already exists")
    args.output.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
