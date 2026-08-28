/* Merlin bare-metal layout for spike (default DRAM base 0x80000000). */
OUTPUT_ARCH("riscv")
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text*) }
  . = ALIGN(0x1000);
  .htif : { *(.htif) }
  . = ALIGN(8);
  .data : { *(.data*) *(.sdata*) *(.rodata*) *(.srodata*) }
  . = ALIGN(8);
  _bss_start = .;
  .bss : { *(.bss*) *(.sbss*) *(COMMON) }
  _bss_end = .;
  . = ALIGN(16);
  . += 0x40000;               /* MERLIN_STACK_PER_HART * max harts */
  _stack_top = .;
}
