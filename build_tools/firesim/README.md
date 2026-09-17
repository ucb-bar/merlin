# build_tools/firesim

Link-time artifacts and a read-only preflight for running merlin on **FireSim** (the Saturn SoC
on an Alveo U250). The operating guide — building a bitstream, the mandatory job queue, reading
`uartlog`/`heartbeat.csv`, troubleshooting — is [`docs/guides/firesim.md`](../../docs/guides/firesim.md).

## Two target paths, and which files apply to which

FireSim runs whatever ELF you stage. Merlin has two ways to produce one, and they have
**opposite** linker requirements:

| | Zephyr application | Bare-metal |
|---|---|---|
| Built by | `zephyr_model.build_app(board="chipyard_riscv64")` | `merlin/runtime/baremetal/spike/` harness |
| Linker script | **Zephyr's own `zephyr.lds`** | ours (`-T`) |
| Large weights | devicetree `zephyr,memory-region` + a `zephyr_linker_sources(SECTIONS)` snippet, emitted by `zephyr_model.py` | absolute-address section in the linker script |
| libc | picolibc (from the Zephyr SDK) | none — `libc_min.c` |
| Files here | **none.** Do not pass `firesim_baremetal.ld` or `htif-nano.specs` to a Zephyr build | `firesim_baremetal.ld`, optionally `htif-nano.specs` |
| Status on FireSim | **validated** — models have run end-to-end on the FPGA | not run on the FPGA yet |

The single most important line in this table is that the Zephyr path takes **no linker script
from this directory**. Supplying one conflicts with `zephyr.lds`; the IREE-era toolchain file
this directory derives from carried a whole second profile that existed only to *skip* its own
linker script for exactly this reason.

## Files

### `preflight.py` — read-only "will a run work here?"

```bash
.venv/bin/python build_tools/firesim/preflight.py           # human
.venv/bin/python build_tools/firesim/preflight.py --json    # machine
```

Checks the chipyard checkout, the ModelBlaster runner `run_on_firesim()` imports, the queue
daemon's liveness (not merely its pid file), the XDMA device nodes, the `default_hw_config` →
hwdb → bitstream-tar chain, and the effective clock of the most recent run. Touches nothing:
no job is submitted, the FPGA is not opened, the daemon is not started. Exit 1 on any FAIL.

Each check corresponds to a failure whose error message does not name its own cause — see the
troubleshooting section of the guide.

### `firesim_baremetal.ld` — bare-metal layout for the FPGA

Code/rodata/data/bss/stack compact at the DRAM base; the weights blob at a fixed absolute high
address as its own loadable section, reached by literal constant rather than a relocation. That
split is what keeps a multi-GB model from silently pushing ordinary symbols outside
`-mcmodel=medany`'s ±2GB PC-relative window.

Parameterized at link time (`-Wl,--defsym=MERLIN_WEIGHTS_BASE=…`, `MERLIN_DRAM_BASE`,
`MERLIN_STACK_BYTES`). At its defaults it lays out the same image as
`merlin/runtime/baremetal/spike/model_link.ld`, so a spike build and a FireSim build are
directly comparable.

**Not yet exercised on the FPGA.** It is the bare-metal sibling of a spike script that is.

### `htif-nano.specs` — optional newlib link profile

Only for linking against chipyard's newlib sysroot when you want `printf`/`exit` over HTIF
without writing them. Merlin's own bare-metal harness is freestanding and needs no specs file,
so nothing in the repo uses this today. Adapted, not exercised.

## Provenance, and what was deliberately not ported

These files derive from `build_tools/firesim/` in the **deprecated IREE-based merlin** at
a separate IREE-era checkout (a different project — see the two-merlins distinction). That tree
targets IREE/VMFB bare-metal; we target a Zephyr application plus merlin's own data-driven C
runtime. What changed, and what was left behind:

| Prior-art file | Here | Why |
|---|---|---|
| `htif.ld`, `htif_large_model.ld` | merged into `firesim_baremetal.ld` | `htif_large_model.ld` solved the medany problem by placing the blob *after* heap+stack, with a comment conceding the layout was "approximate". Merlin's spike script already solved it more directly — a fixed **absolute** base the code reaches by literal — so the port keeps our mechanism and takes the prior art's contribution: that the problem exists above ~100 MB and that the blob must not be symbol-addressed. |
| `htif-nano.spec` | `htif-nano.specs`, minus the linker-script clause | the original ended with `-dT htif.ld%s`, pulling the script from GCC's startfile search path — which only works once `htif.ld` is installed into the sysroot. Our scripts live in the repo, so the caller passes `-T`. |
| `riscv_firesim.toolchain.cmake` | **not ported** | a CMake toolchain file for an IREE CMake build. Neither merlin path is built that way: the Zephyr path is driven by Zephyr's own CMake/west, the bare-metal path by direct compiler invocations from Python. Porting it would mean inventing a build system we do not use. Its durable facts — `-march=rv64imafdc -mabi=lp64d -mcmodel=medany -mstrict-align`, the sysroot cascade, and the bare-metal-vs-Zephyr profile split — are recorded here and in the guide instead. (Merlin uses `rv64gc`/`rv64gcv` rather than `rv64imafdc`, since the Saturn tiles have V.) |
| `zephyr_stubs/` (`pthread.h`, `alloca.h`, `memory.h`, `sys/socket.h`) | **not ported** | header stubs that existed to make IREE's runtime compile under `IREE_PLATFORM_GENERIC`. We do not build IREE. |
| `clear_dl_libs.cmake` | **not ported** | a workaround for IREE's unconditional `${CMAKE_DL_LIBS}`. Same reason. |
| `setup_toolchain.sh` | **not ported** | downloads IREE's prebuilt clang/QEMU tarballs from a SHARK blob store. Our toolchains are chipyard's riscv-tools and the Zephyr SDK, configured through `.env`. |
| `spike.cfg` | **not ported** | an OpenOCD `remote_bitbang` JTAG attach config for spike — unrelated to FireSim, and no merlin flow debugs that way. |
