/* Merlin bare-metal layout for spike (default DRAM base 0x80000000). */
/* Loadable sections are padded to an 8-byte multiple, not merely aligned above: a segment whose
 * file size is not a whole number of beats forces a beat-oriented loader to issue a partially masked
 * final beat, which over TileLink is a PutPartialData that the chip-side manager may reject -- the
 * load then aborts having printed nothing. `. = ALIGN(8);` inside a section grows the section (and so
 * the segment); between sections it only moves the location counter. See model_link.ld. */
OUTPUT_ARCH("riscv")
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text*) . = ALIGN(8); }
  . = ALIGN(0x1000);
  .htif : { *(.htif) . = ALIGN(8); }
  . = ALIGN(8);
  .data : { *(.data*) *(.sdata*) *(.rodata*) *(.srodata*) . = ALIGN(8); }
  . = ALIGN(8);
  _bss_start = .;
  .bss : { *(.bss*) *(.sbss*) *(COMMON) }
  _bss_end = .;
  . = ALIGN(16);
  . += 0x40000;               /* MERLIN_STACK_PER_HART * max harts */
  _stack_top = .;
}
