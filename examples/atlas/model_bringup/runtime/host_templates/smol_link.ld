
OUTPUT_ARCH(riscv)
ENTRY(_start)
PHDRS { text PT_LOAD FLAGS(5); rodata PT_LOAD FLAGS(4); data PT_LOAD FLAGS(6); dram PT_LOAD FLAGS(6); }
SECTIONS {
  . = 0x80000000;
  .text : { KEEP(*(.text.init)) *(.text .text.*) } :text
  . = ALIGN(64);
  .rodata : { *(.rodata .rodata.*) } :rodata
  . = ALIGN(64);
  .data : { *(.data .data.* .sdata .sdata.*) } :data
  . = ALIGN(64);
  .handoff : { KEEP(*(.handoff)) } :data
  . = ALIGN(64);
  .tohost : { KEEP(*(.tohost)) } :data
  . = ALIGN(64);
  __global_pointer$ = . + 0x800;
  .bss (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(64);
    __bss_end = .;
  } :data
  .stack (NOLOAD) : { __stack_bottom = .; . += 65536; __stack_top = .; } :data
  ASSERT(__stack_top < 0x90000000, "host exceeds bounded low DRAM region")
  .atlas_dram 0x100000000 (NOLOAD) : {
    __atlas_dram_start = .;
    . += 3625472;
    __atlas_dram_end = .;
  } :dram
  ASSERT(__atlas_dram_end <= 0x200000000, "DMA low32 window overflow")
}
