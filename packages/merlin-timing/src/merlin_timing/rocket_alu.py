"""Prepare an explicit selected Rocket bitmanip ALU law from reopened sources.

Operator preparation only. This does not fit prices, run a simulator or qualify
whole-SoC accuracy. The recognized source pipeline is deliberately narrow.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def _source(pin):
    if set(pin) != {"path", "sha256"} or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]):
        raise ValueError("source requires an exact path and SHA256")
    path = Path(pin["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError(f"source digest mismatch: {path.name}")
    return raw.decode()


def select_bitmanip_alu(model, *, selected_isa, decode, alu, pipeline, rtl):
    """Return a prospective model and receipt; never mutate an active model.

    The ISA must come from the actual selected elaboration JSON, not compiler
    flags or Spike's allowed ISA. Reopen all original source pins before
    recognizing the ALU/decode/bypass equations. Any change requires review.
    """
    texts = {name: _source(pin) for name, pin in {
        "selected_isa": selected_isa, "decode": decode, "alu": alu,
        "pipeline": pipeline, "rtl": rtl}.items()}
    cpu = json.loads(texts["selected_isa"])["cpus"]["cpu@0"]
    isa, = cpu["riscv,isa"]
    if not isa.startswith("rv64") or any("_" + extension not in isa for extension in ("zba", "zbb", "zbs")):
        raise ValueError("selected elaboration does not enable all reviewed bitmanip extensions")
    required = {
        "decode": ("class ZbaDecode", "class ZbbDecode", "class ZbsDecode", "SH1ADD", "A1_RS1SHL", "FN_ADD", "FN_UNARY", "FN_BEXT"),
        "alu": ("io.out := out", "val out = MuxLookup", "if (coreParams.useZbb)", "FN_UNARY -> unary", "FN_ADD -> io.adder_out"),
        "pipeline": ("if (coreParams.useZba)", "if (coreParams.useZbb)", "if (coreParams.useZbs)", "mem_reg_valid := !ctrl_killx", "mem_reg_wdata := Mux(ex_reg_set_vconfig", "alu.io.out)", "ex_waddr, mem_reg_wdata"),
    }
    for name, anchors in required.items():
        if any(anchor not in texts[name] for anchor in anchors):
            raise ValueError(f"unreviewed selected {name} equations")
    if "module Rocket" not in texts["rtl"] or "module RocketALU" not in texts["rtl"]:
        raise ValueError("selected RTL lacks reviewed Rocket/ALU hierarchy")
    if any(model.get("parameters", {}).get("cpu.bitmanip." + key) is not None
           for key in ("latency_cycles", "occupancy_cycles")):
        raise ValueError("explicit bitmanip prices conflict with source service")
    mappings = model["cpu_execution_resources"]
    if mappings["cpu.bitmanip"] != mappings["cpu.integer"]:
        raise ValueError("selected bitmanip must use the same integer ALU")
    result = deepcopy(model)
    if any("_" + x not in result["cpu"]["isa"] for x in ("zba", "zbb", "zbs")):
        raise ValueError("native ISA does not match selected enabled extensions")
    service = dict(schema="merlin.rocket_bitmanip_alu_service.v1", xlen=64,
                   zba=True, zbb=True, zbs=True, execute_to_memory_stages=1,
                   initiation_interval=1, selected_isa_sha256=selected_isa["sha256"],
                   decode_source_sha256=decode["sha256"], alu_source_sha256=alu["sha256"],
                   pipeline_source_sha256=pipeline["sha256"], rtl_sha256=rtl["sha256"])
    result["cpu"]["bitmanip_alu_service"] = service
    receipt = dict(schema="merlin.selected_rocket_bitmanip_alu_preparation.v1",
                   sources=dict(selected_isa=selected_isa, decode=decode, alu=alu,
                                pipeline=pipeline, rtl=rtl), selected_isa=isa,
                   source_service=service, coefficients_fitted=False,
                   activation=False, qualification="not_established",
                   scope="enabled integer-ALU execute result/bypass and initiation; cache/branch/whole-SoC costs separate")
    return result, receipt
