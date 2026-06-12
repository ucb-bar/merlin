/* Merlin bare-metal layout for whole-model runs on spike.
 *
 * Code/rodata/data/bss stay compact at 0x80000000 (so all internal PC-relative
 * addressing stays within `-mcmodel=medany`'s +/-2GB reach). The weight blob — which can
 * be multiple GB — is placed at a FIXED ABSOLUTE high address (MERLIN_WEIGHTS_BASE,
 * default 8GB) as its own loadable section; the runtime addresses it by literal constant
 * (model_main.c), not a symbol relocation. The activation arena is likewise a fixed
 * absolute region addressed by literal (merlin_malloc.c) and is not a loaded section.
 *
 * MERLIN_WEIGHTS_BASE is set at link time (-Wl,--defsym). spike -m must cover
 * 0x80000000 .. MERLIN_WEIGHTS_BASE + sizeof(weights).
 */
OUTPUT_ARCH("riscv")
ENTRY(_start)
MERLIN_WEIGHTS_BASE = DEFINED(MERLIN_WEIGHTS_BASE) ? MERLIN_WEIGHTS_BASE : 0x200000000;
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text*) }
  . = ALIGN(0x1000);
  .htif : { *(.htif) }
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
  . += 0x40000;            /* stack (per-hart), kept near the code */
  _stack_top = .;

  /* weight blob at its absolute high address (loaded by spike from the ELF) */
  . = MERLIN_WEIGHTS_BASE;
  .weights : { *weights_blob.o(.data .data.* .rodata .rodata.*) }
}
