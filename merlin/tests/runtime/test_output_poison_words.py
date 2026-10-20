"""Execute exact poison/bounds controls and inspect selected RISC-V store width."""

import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.runtime.direct_kernel_harness import _OUTPUT_POISON_C, DirectKernelAbi, render_direct_kernel
from merlin.targetgen.contract.readback_policy import COHERENT_DUMP_V1, ReadbackPolicy


def test_word_poison_preserves_each_byte_and_adjacent_sentinels_at_every_alignment(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    source = tmp_path / "poison.c"
    source.write_text('#include <stdint.h>\n#include <string.h>\n' + _OUTPUT_POISON_C + """
int main(void){
  unsigned char data[160] __attribute__((aligned(64)));
  for(unsigned offset=0;offset<16;offset++){
    for(unsigned count=0;count<=95;count++){
      memset(data,0x3c,sizeof(data));
      merlin_poison_bytes(data+16+offset,count,0xa5);
      for(unsigned i=0;i<sizeof(data);i++){
        unsigned char expected=(i>=16+offset && i<16+offset+count)?0xa5:0x3c;
        if(data[i]!=expected) return 1;
      }
    }
  }
  merlin_poison_bytes((void*)0,0,0xa5);
  return 0;
}
""")
    binary = tmp_path / "poison"
    subprocess.run([compiler, "-std=gnu11", "-O2", "-fstrict-aliasing", "-Wall", "-Werror",
                    str(source), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True, capture_output=True, timeout=5)


def test_every_fresh_output_is_poisoned_before_entry_without_changing_readwrite_inputs():
    cb = {"tensors": {"I": {"dtype": "i8", "shape": [3]}, "Y": {"dtype": "i32", "shape": [5]},
                      "Z": {"dtype": "i8", "shape": [9]}},
          "kernel_abi": {"args": [{"tensor": "I", "access": "readwrite"},
                                    {"tensor": "Y", "access": "write"},
                                    {"tensor": "Z", "access": "write"}], "outputs": ["I", "Y", "Z"]}}
    options = dict(inputs={"I": {"shape": [3], "values": [1, 2, 3]}},
                   readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
                   abi=DirectKernelAbi("entry", "complete", 8, "little", "void"))
    source = render_direct_kernel(cb, output_poison=165, **options)
    assert "merlin_poison_bytes(tensor_0" not in source
    assert source.index("merlin_poison_bytes(tensor_1,UINT64_C(20),165)") < source.index("  entry(")
    assert source.index("merlin_poison_bytes(tensor_2,UINT64_C(9),165)") < source.index("  entry(")
    assert "merlin_poison_bytes" not in render_direct_kernel(cb, **options)


def test_selected_riscv_compiler_uses_word_stores_for_capacity_output(tmp_path):
    compiler = shutil.which("riscv64-unknown-elf-gcc")
    if compiler is None:
        compiler = "/work/merlin/hw/chipyard-1.14.0/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"
    if not Path(compiler).is_file():
        pytest.skip("selected RISC-V C compiler unavailable")
    cb = {"tensors": {"Y": {"dtype": "i32", "shape": [5476, 16]}},
          "kernel_abi": {"args": [{"tensor": "Y", "access": "write"}], "outputs": ["Y"]}}
    options = dict(inputs={}, readback_policy=ReadbackPolicy(COHERENT_DUMP_V1),
                   abi=DirectKernelAbi("entry", None, 64, "little", "void"))
    actual = render_direct_kernel(cb, output_poison=165, **options)
    legacy = render_direct_kernel(cb, **options).replace("  entry(",
        "  for(uint64_t index=0;index<UINT64_C(350464);index++)\n"
        "    ((volatile unsigned char*)tensor_0)[index]=165;\n  entry(", 1)
    (tmp_path / "htif.h").write_text("void console_init(void);void htif_puts(const char*);void htif_exit(int);\n")
    observed = {}
    for name, code in (("actual", actual), ("legacy", legacy)):
        source, assembly = tmp_path / (name + ".c"), tmp_path / (name + ".s")
        source.write_text(code)
        subprocess.run([compiler, "-std=gnu11", "-march=rv64gc", "-mabi=lp64d", "-O2",
                        "-fstrict-aliasing", "-I", str(tmp_path), "-S", str(source), "-o", str(assembly)],
                       check=True, capture_output=True)
        observed[name] = [line.strip().split()[0] for line in assembly.read_text().splitlines()
                         if line.strip() and not line.strip().startswith((".", "#")) and ":" not in line]
    assert "sd" in observed["actual"]
    assert "sb" not in observed["actual"]  # Aligned/divisible extent specializes to word stores.
    assert "sb" in observed["legacy"]
