#!/usr/bin/env python3
"""Relink a host-only live cycle observer around the pinned Verilator model.

No RTL is regenerated. Original model objects and latency-pipe objects are
reused. The observer reads the core's architectural CSR state and never writes
it. Source/header/archive identities and exact commands are retained.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

PREFIX = (
    "TestDriver__DOT__testHarness__DOT__chiptop0__DOT__system__DOT__tile_prci_domain__DOT__"
    "element_reset_domain_rockettile__DOT__core__DOT__csr__DOT__"
)
HEADER = r"""
#include "VTestDriver___024root.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
struct DSECycleMonitor {
  const char* file = std::getenv("MERLIN_DSE_CYCLE_LIMIT_FILE");
  uint64_t evaluations=0,limit=0,begin=0;
  unsigned stage=0;
  DSECycleMonitor(){
    if(file)std::fprintf(stderr,"[dse-cycle-monitor] enabled poll_evaluations=64 refresh_evaluations=256\n");
  }
  void step(VTestDriver___024root* root){
    if(!file || (++evaluations & 63))return;
    if((evaluations & 255)==0){
      FILE* stream=std::fopen(file,"rb");
      if(!stream || std::fread(&limit,sizeof(limit),1,stream)!=1){
        std::fprintf(stderr,"DSE_MONITOR_ERROR cannot read cutoff\n");std::fflush(nullptr);std::_Exit(126);
      }
      std::fclose(stream);
    }
    uint64_t marker=root->CSR_PREFIXreg_mscratch;
    if(stage==0){if(marker==UINT64_C(0xd5eabc0011223344))stage=1;return;}
    if(marker==UINT64_C(0xd5eabc0044332211)){stage=3;return;}
    if(stage==1 && marker!=UINT64_C(0xd5eabc0011223344)){
      begin=marker;stage=2;
      std::fprintf(stderr,"DSE_MEASURE_BEGIN cycle=%llu\n",(unsigned long long)begin);
    }
    uint64_t now=(root->CSR_PREFIXlarge_1<<6)|root->CSR_PREFIXsmall_1;
    if(stage==2 && limit && now>=begin && now-begin>limit){
      std::fprintf(stderr,"DSE_PRUNED limit=%llu elapsed=%llu\n",
        (unsigned long long)limit,(unsigned long long)(now-begin));
      std::fflush(nullptr);std::_Exit(124);
    }
  }
};
"""


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def build(receipt_path, output):
    receipt = json.loads(receipt_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    link = receipt["commands"][3]["argv"]
    archive = next(Path(arg) for arg in link if arg.endswith("VTestDriver__ALL.a"))
    model = archive.parent
    source = model / "VTestDriver__main.cpp"
    header = model / "VTestDriver___024root.h"
    text = header.read_text()
    for field in ["reg_mscratch", "large_1", "small_1"]:
        if PREFIX + field not in text:
            raise ValueError("Qualified CSR fields missing; refuse a different generated model")
    original = source.read_text()
    if original.count("topp->eval();") != 1 or original.count("while (!contextp->gotFinish())") != 1:
        raise ValueError("Unexpected generated Verilator main; observer refused")
    monitor = HEADER.replace("CSR_PREFIX", PREFIX)
    (output / "dse_cycle_monitor.h").write_text(monitor)
    modified = '#include "dse_cycle_monitor.h"\n' + original
    modified = modified.replace(
        "while (!contextp->gotFinish())", "DSECycleMonitor dse_monitor;\n    while (!contextp->gotFinish())"
    )
    modified = modified.replace("topp->eval();", "topp->eval();\n        dse_monitor.step(topp->rootp);")
    native = output / "monitor_main.cpp"
    native.write_text(modified)
    commands = []

    def run(argv):
        started = time.monotonic()
        p = subprocess.run(argv, cwd=output, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        commands.append(dict(argv=argv, cwd=str(output), rc=p.returncode, wall_s=time.monotonic() - started))
        (output / f"command-{len(commands)}.log").write_text(p.stdout)
        if p.returncode:
            raise RuntimeError(p.stdout[-4000:])

    flags = list(receipt["commands"][0]["argv"])
    i = flags.index("-o")
    flags[i + 1] = str(output / "monitor_main.o")
    flags[i + 2] = str(native)
    run(flags)
    renamed = output / "VTestDriver__monitor_model.a"
    run(["objcopy", "--redefine-sym", "main=merlin_original_verilator_main", str(archive), str(renamed)])
    linked = [str(renamed) if arg == str(archive) else arg for arg in link]
    linked.insert(1, str(output / "monitor_main.o"))
    linked[-1] = str(output / "emulator")
    run(linked)
    result = dict(receipt)
    result["binary"] = identity(output / "emulator")
    result["commands"] = receipt["commands"] + commands
    result["dse_cycle_monitor"] = dict(
        kind="verilator_live_csr_cutoff_v1",
        poll_evaluations=64,
        refresh_evaluations=256,
        enabled_by_environment="MERLIN_DSE_CYCLE_LIMIT_FILE",
        model_objects_regenerated=False,
        counter_source="reg_cycle / io_time_0 = {large_1, small_1}; CSR.scala:595,1162",
        parent_receipt=identity(receipt_path),
        source=identity(native),
        monitor=identity(output / "dse_cycle_monitor.h"),
        generated_header=identity(header),
        generated_main=identity(source),
        original_archive=identity(archive),
        qualification="pending positive/negative and live-cutoff controls",
    )
    (output / "build_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return output / "build_receipt.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.memory_receipt.resolve(), args.output.resolve()))
