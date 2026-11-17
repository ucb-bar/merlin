"""Actual bare-metal execution of returned results through the serial context.

This exercises the reusable ABI/runtime seam. The observation fixture below
is not the production staged model driver or dynamic-result wire protocol.
"""

import hashlib
import json
import subprocess

import numpy as np
import pytest
from owned_result_fixture import COMPACT

from merlin.common.digest import sha256_file
from merlin.common.paths import runtime_dir
from merlin.llvmlower import toolchain
from merlin.llvmlower.owned_result_lowering import lower_owned_results
from merlin.runtime.backends import spike, spike_model
from merlin.runtime.backends.host_compiler_policy import selected_model_compiler_plan

DRIVER = r'''
#include "htif.h"
#include "owned_result_views.h"
#include <stdlib.h>
extern int merlin_owned_results_begin(size_t,size_t);
extern void merlin_owned_results_end(void);
extern int merlin_owned_result_invoke_checked(void **,MerlinOwnedResultView *,size_t);
typedef struct {void *allocated,*aligned;intptr_t offset,sizes[1],strides[1];} D1;
_Noreturn void merlin_owned_results_failure(const char *reason) {
  htif_puts("OWNED_FAILURE "); htif_puts(reason); htif_putc('\n'); htif_exit(2);
}
int main(int hart) {
  if(hart) for(;;) {}
  console_init();
  int32_t words[3][6]={{0,-1,4,0,9,-3},{0},{INT32_MIN,INT32_MAX}};
  intptr_t sizes[]={6,0,2};
  for(int k=0;k<3;k++) {
    D1 input={words[k],words[k],0,{sizes[k]},{1}};
    void *arguments[]={&input};
#ifdef RESOURCE_FAILURE
    if(!merlin_owned_results_begin(1,1)) htif_exit(3);
#else
    if(!merlin_owned_results_begin(8192,64)) htif_exit(3);
#endif
    MerlinOwnedResultView views[3];
    int status=merlin_owned_result_invoke_checked(arguments,views,8192);
    if(status) merlin_owned_results_failure(merlin_owned_result_error(status));
    int32_t *copies[3];
    for(int i=0;i<3;i++) {
      copies[i]=malloc(views[i].bytes ? views[i].bytes : 1);
      if(!copies[i] || merlin_owned_result_copy(&views[i],copies[i],views[i].bytes)) htif_exit(4);
    }
    merlin_owned_results_end();
    for(int i=0;i<3;i++) {
      htif_puts("OWNED_RESULT ");htif_putd(k);htif_putc(' ');htif_putd(i);
      htif_putc(' ');htif_putd((long)views[i].count);
      for(size_t j=0;j<views[i].count;j++) {htif_putc(' ');htif_putd(copies[i][j]);}
      htif_putc('\n');free(copies[i]);
    }
  }
  htif_puts("DONE\n");htif_exit(0);
}
'''


@pytest.mark.parametrize("resource_failure", [False, True], ids=["results", "allocation-failure"])
def test_serial_owned_result_runtime_executes_on_actual_image(tmp_path, resource_failure):
    if not toolchain.available() or not spike.available():
        pytest.skip("existing native/cross/Spike toolchains required")
    plan = selected_model_compiler_plan(backend="scalar", cflags_override=None, vlen=None, features=frozenset())
    observed, gcc = plan["observation"], spike.gcc_path()
    lowered = lower_owned_results(
        COMPACT, workdir=tmp_path / "lower", data_layout=observed["data_layout"],
        index_bits=observed["index_bits"], timeout=30,
    )
    harness, abi = spike.harness_dir(), runtime_dir() / "abi"
    layout = spike_model._layout(4 << 20, 0, dram_bytes=64 << 20, code_reserve=2 << 20)
    driver = tmp_path / "driver.c"
    driver.write_text(DRIVER)
    flags = plan["gcc_cflags"]
    objects = []

    def compile_unit(source, name, compiler, options):
        output = tmp_path / name
        done = subprocess.run(
            [*map(str, compiler), *map(str, options), "-c", str(source), "-o", str(output)],
            capture_output=True, text=True, timeout=30,
        )
        assert done.returncode == 0, done.stderr
        objects.append(output)

    compile_unit(lowered.llvm, "model.o", [observed["compiler_resolved"]], observed["cross_flags"])
    includes = ["-I", abi, "-I", harness, "-std=c11"]
    for source, name, definitions in (
        (driver, "driver.o", ["-DRESOURCE_FAILURE=1"] if resource_failure else []),
        (lowered.bridge, "bridge.o", []),
        (abi / "owned_result_views.c", "views.o", []),
        (abi / "owned_results.c", "owned.o", ["-DMERLIN_OWNED_RESULTS_SERIAL_CONTEXT=1",
                                              "-DMERLIN_OWNED_RESULTS_EXTERNAL_FAILURE=1"]),
        (harness / "crt.S", "crt.o", []), (harness / "htif.c", "console.o", []),
        (harness / "libc_min.c", "libc.o", []),
        (harness / "merlin_malloc.c", "malloc.o", [
            f'-DMERLIN_ARENA_BASE_ADDR={layout["arena_base"]}ULL',
            f'-DMERLIN_ARENA_SIZE_BYTES={4 << 20}ULL']),
    ):
        compile_unit(source, name, [gcc], [*flags, *includes, *definitions])
    runtime_compiler = spike_model._mlir_runtime_compiler(
        type(gcc)(observed["compiler_resolved"]), gcc, flags,
    )
    compile_unit(abi / "mlir_runtime.c", "mlir.o", runtime_compiler, [])
    image = tmp_path / "model.elf"
    done = subprocess.run(
        [str(gcc), *flags, "-nostdlib", "-nostartfiles",
         f'-Wl,--defsym,MERLIN_WEIGHTS_BASE={layout["weights_base"]}',
         "-Wl,--defsym,MERLIN_STACK_BYTES=65536",
         f'-Wl,--defsym,{spike_model.DRAM_BASE_SYMBOL}={spike_model.DRAM_BASE}',
         f'-Wl,--defsym,{spike_model.DRAM_SPAN_SYMBOL}={layout["mem_bytes"]}',
         "-T", str(harness / "model_link.ld"), *map(str, objects), "-lm", "-lgcc", "-o", str(image)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    process = spike_model.run_process(image, harts=1, isa=spike_model.declared_isa(image), timeout=30)
    (tmp_path / "stdout.bin").write_bytes(process.stdout)
    (tmp_path / "stderr.bin").write_bytes(process.stderr)
    identities = {str(path): sha256_file(path) for path in (
        image, driver, lowered.llvm, lowered.bridge, gcc,
        type(gcc)(observed["compiler_resolved"]), spike.spike_path(),
        abi / "owned_results.c", abi / "owned_result_views.c", abi / "owned_result_views.h",
        abi / "mlir_runtime.c", abi / "half_extend.h", harness / "crt.S", harness / "htif.c",
        harness / "libc_min.c", harness / "merlin_malloc.c", harness / "model_link.ld",
    )}
    (tmp_path / "image-observation.json").write_text(json.dumps({
        "schema": "merlin.owned-result-image-test.v1", "identities": identities,
        "index_observation": observed, "gcc_flags": flags,
        "declared_isa": spike_model.declared_isa(image), "memory": spike_model.declared_memory(image),
        "source_fixture_sha256": hashlib.sha256(COMPACT.encode()).hexdigest(),
        "stdout_sha256": sha256_file(tmp_path / "stdout.bin"),
        "stderr_sha256": sha256_file(tmp_path / "stderr.bin"),
        "returncode": process.returncode, "resource_failure": resource_failure,
        "scope": "runtime seam fixture; no staged TTIR package, production readback or hardware performance verdict",
    }, sort_keys=True, indent=2) + "\n")
    if resource_failure:
        assert process.returncode != 0
        assert b"OWNED_FAILURE invocation allocation budget exceeded" in process.stdout
        assert b"OWNED_RESULT " not in process.stdout and b"DONE" not in process.stdout
        return
    assert process.returncode == 0, (process.returncode, process.stdout, process.stderr)
    console = process.stdout.decode()
    assert console.endswith("DONE\n")
    expected = []
    for k, values in enumerate(([0, -1, 4, 0, 9, -3], [], [-(1 << 31), (1 << 31) - 1])):
        x = np.asarray(values, np.int32)
        for i, result in enumerate((x[x > 0], x[x > 0], x)):
            expected.append([k, i, len(result), *result.tolist()])
    actual = [list(map(int, line.split()[1:])) for line in console.splitlines() if line.startswith("OWNED_RESULT ")]
    assert actual == expected
    lowered.verify()
