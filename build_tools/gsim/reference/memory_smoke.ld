OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text*) }
  . = ALIGN(64);
  .tohost : { *(.tohost) }
  . = ALIGN(64);
  .data : { *(.data*) *(.rodata*) }
  . = ALIGN(64);
  .bss : { *(.bss*) }
}
