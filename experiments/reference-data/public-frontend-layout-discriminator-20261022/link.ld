OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text*) }
  . = ALIGN(64);
  .tohost : { *(.tohost) }
  .data : { *(.data*) *(.rodata*) *(.sdata*) }
  .bss : { *(.bss*) *(.sbss*) *(COMMON) }
}
