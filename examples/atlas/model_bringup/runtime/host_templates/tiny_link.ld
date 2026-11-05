OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M }
SECTIONS {
 . = ORIGIN(RAM);
 .text : { *(.text.init) *(.text .text.*) } > RAM
 .rodata : { *(.rodata .rodata.*) } > RAM
 . = ALIGN(64);
 .data : { __global_pointer$ = . + 0x800; *(.sdata .sdata.*) *(.data .data.*) } > RAM
 .tohost : ALIGN(64) { *(.tohost) } > RAM
 .bss (NOLOAD) : ALIGN(64) { __bss_start = .; *(.sbss .sbss.*) *(.bss .bss.*) *(COMMON) . = ALIGN(8); __bss_end = .; } > RAM
 .stack (NOLOAD) : ALIGN(64) { __stack_bottom = .; . += 65536; __stack_top = .; } > RAM
 ASSERT(__stack_top <= ORIGIN(RAM)+LENGTH(RAM), "RAM overflow")
 ASSERT((atlas_test_mailbox % 64) == 0, "mailbox alignment")
 /DISCARD/ : { *(.comment) *(.eh_frame*) }
}
