"""Connect selected RTL/cache geometry to a separate, unqualified Spike model."""
# ruff: noqa: E402 -- Select the task's package roots and existing target evidence reader.

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import common as C

C.configure()
import yaml
from merlin_timing.capacities import model_resources

from merlin.targetgen.rtl.timing_structure import apply_to_profile

sys.path.insert(0, str(C.MERLIN_ROOT / "examples/gemmini/verification"))
from extract_rtl_timing_facts import _node, _property

FACTS = C.P1 / "merlin/out/artifacts/probes/gemmini/fire-aligned-30mhz/input/v6/facts.json"
PROFILE = C.P1 / ".tmp/annotated-spike-evidence/provider-bundle/fire_aligned_30mhz.json"
RUN = C.P1 / "merlin/out/runs/gemmini/phase1/el4-fire-aligned-phase1-20261017-v3"
STEM = "chipyard.harness.TestHarness.GemminiGsimFireSimAlignedConfig"
GENERATED = C.P1 / ".tmp/chipyard-fire-aligned/sims/verilator/generated-src" / STEM


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=C.OUT / "spike-structural-config-v1")
    root = parser.parse_args(argv).output_dir
    if root.exists():
        parser.error("retain the previous output and select a fresh --output-dir")
    if C.file_hash(FACTS) != "b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af":
        raise ValueError("selected Phase 1 facts changed")
    if C.file_hash(PROFILE) != "12a1a0b45fe7767f03f1b277349b7600c7c68243f7161aba52242df1921721e0":
        raise ValueError("original timing-provider device profile changed")
    facts, original = json.loads(FACTS.read_text()), json.loads(PROFILE.read_text())
    declaration = C.MERLIN_ROOT / "examples/gemmini/target/contracts/target_contract.yaml"
    mapping = yaml.safe_load(declaration.read_text())["rtl_extraction"]["timing_structure"]["profile_resources"]
    dts_path, l2_path = GENERATED / (STEM + ".dts"), GENERATED / (STEM + ".l2.json")
    if C.file_hash(dts_path) != original["provenance"]["host_dts_sha256"]:
        raise ValueError("selected gSIM host/cache geometry differs from the timing provider")
    dts = dts_path.read_text()
    cpu, l2_node = _node(dts, "cpu@0"), _node(dts, "cache-controller@2010000")
    caches = {}
    for name, node, prefix in [("l1i", cpu, "i-cache"), ("l1d", cpu, "d-cache"), ("l2", l2_node, "cache")]:
        size, sets, line = (_property(node, prefix + suffix) for suffix in ("-size", "-sets", "-block-size"))
        ways, remainder = divmod(size, sets * line)
        if remainder or ways <= 0:
            raise ValueError("cache geometry is not a whole number of ways")
        caches[name] = {"sets": sets, "ways": ways, "line_bytes": line, "policy": "write_back_write_allocate"}
    banks = json.loads(l2_path.read_text())["banks"]
    if (
        len(banks) != 1
        or banks[0]["ways"] != caches["l2"]["ways"]
        or banks[0]["blockBytes"] != caches["l2"]["line_bytes"]
    ):
        raise ValueError("DTS and elaborated L2 geometry disagree")
    if 2 ** len(banks[0]["setBits"]) != caches["l2"]["sets"]:
        raise ValueError("DTS and elaborated L2 set counts disagree")
    receipt_path = C.P1 / ".tmp/gsim-fire-aligned-provisional-v3/native/build_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    selected_sources = {}
    for row in receipt["inputs"]:
        if row["role"] in {"harness:blackboxes.cpp", "vendor:mm.cc", "vendor:mm.h"}:
            if C.file_hash(row["path"]) != row["sha256"]:
                raise ValueError("gSIM external-memory implementation differs from its build receipt")
            selected_sources[row["role"]] = {"path": row["path"], "sha256": row["sha256"]}
    if set(selected_sources) != {"harness:blackboxes.cpp", "vendor:mm.cc", "vendor:mm.h"}:
        raise ValueError("gSIM receipt omits the selected memory implementation")
    mapped = apply_to_profile(original, facts, mapping)
    model_path = RUN / "count_only_model.json"
    model = copy.deepcopy(json.loads(model_path.read_text()))
    model["resources"] = [dict(row, id=index) for index, row in enumerate(model_resources(mapped))]
    model["memory"] = {
        "coherence_granule": caches["l2"]["line_bytes"],
        "coherence_policy": "writeback_invalidate",
        "caches": caches,
        "paths": {"cpu_fetch": ["l1i", "l2"], "cpu_data": ["l1d", "l2"], "device": ["l2"]},
    }
    structure = facts["facts"]["timing_structure"]
    report = {
        "schema": "phase2.spike_structural_config.v1",
        "launches": [],
        "prediction_authority": "NONE",
        "facts_sha256": C.file_hash(FACTS),
        "device_profile_sha256": C.file_hash(PROFILE),
        "original_model_sha256": C.file_hash(model_path),
        "mapping_source_sha256": C.file_hash(declaration),
        "dts_sha256": C.file_hash(dts_path),
        "l2_json_sha256": C.file_hash(l2_path),
        "gsim_binary_sha256": receipt["binary_sha256"],
        "memory_sources": selected_sources,
        "external_memory": "mm_magic_t; response queuing and AXI backpressure, no DRAMSim or FireSim DRAM bridge",
        "cache_geometry": caches,
        "resources": mapped["structural_mapping"]["resources"],
        "dma": structure["memory_clients"]["clients"],
        "dispatch_entries": {k: v["entries"]["value"] for k, v in structure["command_classes"]["classes"].items()},
        "local_memory": facts["facts"]["memory_address_spaces"],
        "fixed_latencies": mapped["structural_mapping"]["fixed_latencies"],
        "remaining": mapped["structural_mapping"]["fit_from_measurement"],
        "model_assumptions": [
            "Existing Spike LRU tags and writeback_invalidate coherence approximate the selected hierarchy.",
            "Cache geometry is derived; cache replacement/coherence protocol equivalence is not established.",
            "Only completely derived resource mappings enter the native model; no invented lanes or queues.",
            "CPU execution/issue binding, execute/config resource semantics and all unpriced costs remain unresolved.",
            "Reset activation and supported ISA/counter-taint semantics still need provider/observer closure.",
            "Original count reports bind the original model; they are not predictions from this new model.",
            "The new model must be frozen, replayed on saved public ELFs and held-validated before ranking authority.",
        ],
    }
    C.save(root / "device-profile.json", mapped)
    C.save(root / "model.json", model)
    report["model_sha256"] = C.file_hash(root / "model.json")
    C.save(root / "status.json", report)
    print(json.dumps({"resources": model["resources"], "caches": caches, "prediction_authority": "NONE"}, indent=2))


if __name__ == "__main__":
    main()
