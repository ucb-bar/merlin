OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { DRAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x10000000 }
SECTIONS {
  . = ORIGIN(DRAM);
  .text : { *(.text.init) *(.text .text.*) } > DRAM
  .rodata : { *(.rodata .rodata.*) } > DRAM
  .data : { *(.data .data.*) *(.sdata .sdata.*) } > DRAM
  __global_pointer$ = . + 0x800;
  .tohost ALIGN(64) : { *(.tohost) } > DRAM
  .bss ALIGN(16) (NOLOAD) : {
    _bss_start = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(16);
    _bss_end = .;
  } > DRAM
  /* Operator outputs and workspace: written by the guest before use, so the
   * startup code does not spend simulated cycles zeroing them. */
  .noinit ALIGN(64) (NOLOAD) : { *(.noinit .noinit.*) } > DRAM
  .stack ALIGN(16) (NOLOAD) : {
    _stack_bottom = .;
    . += 16384;
    _stack_top = .;
  } > DRAM
  _image_end = .;
  ASSERT(_image_end <= ORIGIN(DRAM) + LENGTH(DRAM), "image exceeds guest DRAM")
}
