"""The host observer must see in-flight cutoff changes and ignore untimed work."""

import importlib.util
import os
import struct
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def builder():
    path = repo_root() / "examples/gemmini/comparisons/tvm/build_dse_cycle_monitor.py"
    spec = importlib.util.spec_from_file_location("dse_cycle_monitor_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode,expected", [("live", 124), ("idle", 0), ("disabled", 0)])
def test_live_cutoff_and_phase_boundaries(tmp_path, mode, expected):
    module = builder()
    prefix = module.PREFIX
    (tmp_path / "VTestDriver___024root.h").write_text(
        "#pragma once\n#include <cstdint>\nstruct VTestDriver___024root {\n"
        + "\n".join(f"uint64_t {prefix}{name}=0;" for name in ["reg_mscratch", "large_1", "small_1"])
        + "\n};\n"
    )
    (tmp_path / "monitor.h").write_text(module.HEADER.replace("CSR_PREFIX", prefix))
    code = r"""
#include "monitor.h"
#include <cstring>
int main(int argc,char** argv){
  VTestDriver___024root root;DSECycleMonitor monitor;
  bool live=std::strcmp(argv[2],"live")==0;
  for(uint64_t i=0;i<2048;++i){
    root.PREFIXlarge_1=i>>6;root.PREFIXsmall_1=i&63;
    root.PREFIXreg_mscratch=(live && i>=100)?100:UINT64_C(0xd5eabc0011223344);
    if(i==400){
      FILE* stream=std::fopen(argv[1],"wb");uint64_t limit=100;
      std::fwrite(&limit,sizeof(limit),1,stream);std::fclose(stream);
    }
    monitor.step(&root);
  }
  return 0;
}
""".replace("PREFIX", prefix)
    source = tmp_path / "control.cpp"
    source.write_text(code)
    binary = tmp_path / "control"
    subprocess.run(["c++", "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True, capture_output=True)
    path = tmp_path / "cutoff.bin"
    path.write_bytes(struct.pack("<Q", 100000))
    environment = dict(os.environ)
    environment.pop("MERLIN_DSE_CYCLE_LIMIT_FILE", None)
    if mode != "disabled":
        environment["MERLIN_DSE_CYCLE_LIMIT_FILE"] = str(path)
    p = subprocess.run([str(binary), str(path), mode], env=environment, capture_output=True, text=True, timeout=5)
    assert p.returncode == expected, p.stderr
    if mode == "live":
        assert "DSE_PRUNED limit=100 elapsed=411" in p.stderr
    else:
        assert "DSE_PRUNED" not in p.stderr
