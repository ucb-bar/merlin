#!/usr/bin/env python3
"""Build a readback-instrumented copy of the existing RP10 fork-free capsule."""
from pathlib import Path
import os
import shutil
import subprocess

from merlin.runtime.backends.base import get_backend
from merlin.targetgen.contract.toolchain import mlir_bin
from merlin.targetgen.isa_model import isa_model_from_encoding
from merlin.targetgen.isa_transcode import derive_march
from merlin.targetgen.rtl import mlc_bridge

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SRC = ROOT / ".claude/worktrees/codex-radiance-smoke/out/artifacts/capsule-bench/m0_gsim_probe2/runs/radiance-capsule-bench/RP10_gemv_batched_fp16_pt/generated"
BUILD = HERE / "build"

muon = get_backend("muon").muon
muon_bsp = get_backend("muon").muon_bsp
muon_link = get_backend("muon").muon_link

BUILD.mkdir(parents=True, exist_ok=True)
source = (SRC / "main.c").read_text()
old = """  volatile uint32_t _out_Y0[32];
  radiance_kernel((const void*)_in_A0, (const void*)_in_V, (void*)_out_Y0);
  _ps(\"OUT Y0 32 1\");
  for(int i=0;i<32;i++){_pc(' ');_pf(_u2f(_out_Y0[i]));}
  _pc('\\n');
  _ps(\"DONE\\n\");
  return 0;
}"""
new = """  volatile uint32_t *_out_Y0=(volatile uint32_t*)0x10010000u;
  volatile uint32_t *_status=(volatile uint32_t*)0x10011000u;
  radiance_kernel((const void*)_in_A0, (const void*)_in_V, (void*)_out_Y0);
  _status[1]=32u;
  __asm__ volatile(\"fence rw,rw\" ::: \"memory\");
  _status[0]=0x52503130u;
  __asm__ volatile(\"fence rw,rw\" ::: \"memory\");
  while(_status[2]!=0x41434b31u){}
  return 0;
}"""
if source.count(old) != 1:
    raise SystemExit("refusing: RP10 harness tail no longer matches the recorded source")
(BUILD / "readback_main.c").write_text(source.replace(old, new))

model = isa_model_from_encoding("radiance", mlc_bridge.isa_encoding_for("radiance"))
clang = mlir_bin("clang")
triple, abi = muon.forkfree_compile_triple(model)
cflags = [f"--target={triple}", f"-march={derive_march(model)}", f"-mabi={abi}",
          "-mno-relax", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-pic",
          "-fno-jump-tables"]
subprocess.run([str(clang), *cflags, "-c", str(BUILD / "readback_main.c"),
                "-o", str(BUILD / "readback_main.o")], check=True)
muon_bsp.transcode_boot_object(BUILD / "readback_main.o", BUILD / "readback_main_muon.o",
                               isa_model=model)

bsp = muon.build_forkfree_bsp(BUILD, target="radiance", num_warps=1)
muon_link.link_fork_free([*[str(x) for x in bsp], str(BUILD / "readback_main_muon.o"),
                          str(SRC / "kernel_muon.o")],
                         str(muon.lib_dir() / "linker/mu_link.ld"),
                         str(BUILD / "rp10.readback.radiance.elf"), target="radiance")

cross = muon.rv64_cross_prefix()
fuse_dir = muon.soc_fuse_dir()
env = dict(os.environ, CROSS64=str(cross), RV32_ELF=str(BUILD / "rp10.readback.radiance.elf"),
           OUT=str(BUILD / "rp10.readback.soc.elf"), RV64_START=str(fuse_dir / "start.S"),
           RV64_MAIN=str(HERE / "readback_carrier.c"),
           RV64_CFLAGS="-march=rv64gc -mabi=lp64 -ffreestanding -nostdlib -mcmodel=medany")
subprocess.run(["bash", str(fuse_dir / "fuse_rv32_into_rv64.sh")], cwd=BUILD,
               env=env, check=True)

# The upstream fuse helper names raw-segment symbols after a mktemp directory,
# which makes the non-loadable symbol table nondeterministic.  Retain the three
# semantic addresses, then strip that table so the runnable ELF is byte-stable.
soc = BUILD / "rp10.readback.soc.elf"
symbols = subprocess.run(["readelf", "-Ws", str(soc)], check=True,
                         capture_output=True, text=True).stdout
wanted = ("rp10_numeric_pass", "rp10_numeric_fail", "main")
addresses = {}
for line in symbols.splitlines():
    fields = line.split()
    if len(fields) >= 8 and fields[-1] in wanted:
        addresses[fields[-1]] = f"0x{fields[1]}"
if set(addresses) != set(wanted):
    raise SystemExit(f"missing semantic symbols before strip: {addresses}")
(BUILD / "readback_symbols.txt").write_text(
    "".join(f"{name} {addresses[name]}\n" for name in wanted))
subprocess.run([f"{cross}-strip", "--strip-all", str(soc)], check=True)
