/* firesim_baremetal.ld — Merlin bare-metal layout for the FireSim FPGA target.
 *
 * WHICH PATH IS THIS FOR?  The BARE-METAL path only (crt.S + htif.c from
 * merlin/runtime/baremetal/spike/, no libc, no OS).  The Zephyr application path — which is
 * the FireSim path that is actually validated today — must NOT use this file: Zephyr's own
 * `zephyr.lds` owns the link, and merlin's Zephyr emitter places a large weights blob with a
 * `zephyr_linker_sources(SECTIONS)` snippet into a devicetree `zephyr,memory-region` instead.
 * See build_tools/firesim/README.md.
 *
 * LAYOUT (three regions, all inside FireSim's ExtMem window):
 *
 *   MERLIN_DRAM_BASE      0x0_8000_0000   .text/.htif/.rodata/.data/.bss/stack, compact
 *   MERLIN_ARENA_BASE     0x0_C000_0000   activation arena (merlin_malloc.c literal; NOT a
 *                                          section — not reserved or loaded here)
 *   MERLIN_WEIGHTS_BASE   0x2_0000_0000   the weights blob, its own loadable section
 *
 * WHY THE SPLIT: under `-mcmodel=medany` every internal reference is PC-relative with a
 * +/-2GB reach.  A multi-GB weights blob linked next to the code pushes ordinary symbols out
 * of that window, and the failure is silent — correct arithmetic on the wrong bytes.  So code
 * and data stay compact at the DRAM base, and the blob goes to a FIXED ABSOLUTE address that
 * model_main.c reaches by literal constant (MERLIN_WEIGHTS_BASE_ADDR, compiled to `li`), never
 * by a symbol relocation.  The arena is absolute for the same reason (merlin_malloc.c).
 *
 * FIRESIM SPECIFICS.  The deployed FireSim configs set WithExtMemSize(16 GiB), so target DRAM
 * spans [0x8000_0000, 0x4_8000_0000).  All three regions above fit; the gaps between them are
 * simply unused DRAM (FireSim runs with `zero_out_dram: false`, so never assume they are zero
 * — crt.S clears .bss and the arena is write-before-read).  Lower MERLIN_WEIGHTS_BASE if you
 * ever target a design with a smaller ExtMem.
 *
 * OVERRIDES (link time, no edit required):
 *   -Wl,--defsym=MERLIN_DRAM_BASE=0x80000000
 *   -Wl,--defsym=MERLIN_WEIGHTS_BASE=0x100000000
 *   -Wl,--defsym=MERLIN_STACK_BYTES=0x40000
 * At their defaults this produces the same layout as merlin/runtime/baremetal/spike/
 * model_link.ld, so an image built for spike and one built here are directly comparable.
 *
 * STATUS: derived, NOT yet run on the FPGA.  The validated FireSim path is the Zephyr
 * application; this is the bare-metal sibling of the spike script that IS validated.
 */
OUTPUT_ARCH("riscv")
ENTRY(_start)

MERLIN_DRAM_BASE    = DEFINED(MERLIN_DRAM_BASE)    ? MERLIN_DRAM_BASE    : 0x80000000;
MERLIN_WEIGHTS_BASE = DEFINED(MERLIN_WEIGHTS_BASE) ? MERLIN_WEIGHTS_BASE : 0x200000000;
/* Must be >= MERLIN_STACK_PER_HART (16 KiB in crt.S) * number of harts the image starts. */
MERLIN_STACK_BYTES  = DEFINED(MERLIN_STACK_BYTES)  ? MERLIN_STACK_BYTES  : 0x40000;

SECTIONS {
  . = MERLIN_DRAM_BASE;
  .text : { *(.text.init) *(.text*) }

  /* tohost/fromhost are found BY SYMBOL NAME by fesvr — on FireSim through the TSI bridge,
   * the same protocol spike's fesvr speaks.  Keep the section page-aligned and never let
   * --gc-sections near it. */
  . = ALIGN(0x1000);
  .htif : { KEEP(*(.htif)) }

  . = ALIGN(8);
  .rodata : { *(EXCLUDE_FILE(*weights_blob.o) .rodata*)
              *(EXCLUDE_FILE(*weights_blob.o) .srodata*) }
  . = ALIGN(8);
  .data : { *(EXCLUDE_FILE(*weights_blob.o) .data*) *(.sdata*) }
  . = ALIGN(8);
  _bss_start = .;
  .bss : { *(.bss*) *(.sbss*) *(COMMON) }
  _bss_end = .;

  . = ALIGN(16);
  . += MERLIN_STACK_BYTES;      /* per-hart stacks, kept near the code */
  _stack_top = .;
  PROVIDE(_end = .);

  /* Weights at their absolute high address: a second PT_LOAD the loader writes directly. */
  . = MERLIN_WEIGHTS_BASE;
  .weights : { *weights_blob.o(.data .data.* .rodata .rodata.*) }
}
