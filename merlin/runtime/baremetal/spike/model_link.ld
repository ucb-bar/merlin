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
/* Every loadable section is padded to an 8-byte multiple BELOW, not merely aligned above.
 * A segment's p_filesz is the extent of its contents, so a section that happens to end on an
 * odd 32-bit word leaves the whole segment ending there -- and a loader that moves the image in
 * 8-byte beats must then issue a half-masked final beat for it. Over TileLink that beat is a
 * PutPartial, which a manager is free not to support (testchipip's TSIToTileLink builds its mask
 * from which 32-bit words of the beat are valid, so an odd tail is exactly what produces one);
 * the chip-side manager reached over serial TL does not, and the load aborts on a monitor
 * assertion. `. = ALIGN(8);` INSIDE a section grows the section, and so the segment; the same
 * statement between sections only moves the location counter and does not pad anything. */
OUTPUT_ARCH("riscv")
ENTRY(_start)
MERLIN_WEIGHTS_BASE = DEFINED(MERLIN_WEIGHTS_BASE) ? MERLIN_WEIGHTS_BASE : 0x200000000;
/* Stack reserve, overridable at link time for the same reason the weights base is: it is a SIZING
 * decision that depends on the model, not a property of the layout. The lowering promotes static
 * intermediates to stack `alloca`s for this target, so the demand scales with the model rather than
 * with the harness -- and the region immediately below this reserve is .bss/.data/.rodata/.htif, so
 * running out of it corrupts the allocator's bookkeeping and the HTIF mailbox instead of faulting. */
MERLIN_STACK_BYTES = DEFINED(MERLIN_STACK_BYTES) ? MERLIN_STACK_BYTES : 0x40000;
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text*) . = ALIGN(8); }
  . = ALIGN(0x1000);
  .htif : { *(.htif) . = ALIGN(8); }
  . = ALIGN(8);
  .rodata : { *(EXCLUDE_FILE(*weights_blob.o) .rodata*)
              *(EXCLUDE_FILE(*weights_blob.o) .srodata*)   . = ALIGN(8); }
  . = ALIGN(8);
  .data : { *(EXCLUDE_FILE(*weights_blob.o) .data*) *(.sdata*) . = ALIGN(8); }
  . = ALIGN(8);
  _bss_start = .;
  .bss : { *(.bss*) *(.sbss*) *(COMMON) }
  _bss_end = .;
  . = ALIGN(16);
  . += MERLIN_STACK_BYTES; /* stack (per-hart), kept near the code */
  _stack_top = .;

  /* weight blob at its absolute high address (loaded by spike from the ELF) */
  . = MERLIN_WEIGHTS_BASE;
  .weights : { *weights_blob.o(.data .data.* .rodata .rodata.*) . = ALIGN(8); }
}
