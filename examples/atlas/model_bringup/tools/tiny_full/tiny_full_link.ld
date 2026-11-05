OUTPUT_ARCH(riscv)
ENTRY(_start)
TINY_FULL_RAM_BASE = DEFINED(TINY_FULL_RAM_BASE) ? TINY_FULL_RAM_BASE : 0x80000000;
TINY_FULL_RAM_BYTES = DEFINED(TINY_FULL_RAM_BYTES) ? TINY_FULL_RAM_BYTES : 0x80000000;
TINY_FULL_DEVICE_BASE = DEFINED(TINY_FULL_DEVICE_BASE) ? TINY_FULL_DEVICE_BASE : 0x100000000;
TINY_FULL_DEVICE_BYTES = DEFINED(TINY_FULL_DEVICE_BYTES) ? TINY_FULL_DEVICE_BYTES : 0;
TINY_FULL_RODATA_BASE = DEFINED(TINY_FULL_RODATA_BASE) ? TINY_FULL_RODATA_BASE : 0;
MEMORY { RAM (rwx) : ORIGIN = TINY_FULL_RAM_BASE, LENGTH = TINY_FULL_RAM_BYTES }
SECTIONS {
 . = ORIGIN(RAM);
 .text : { *(.text.init) *(.text .text.*) } > RAM
 .rodata MAX(ALIGN(., 8), TINY_FULL_RODATA_BASE) : { *(.rodata .rodata.*) } > RAM
 . = ALIGN(64);
 .data : { __global_pointer$ = . + 0x800; *(.sdata .sdata.*) *(.data .data.*) } > RAM
 .tohost : ALIGN(64) { *(.tohost) } > RAM
 .bss (NOLOAD) : ALIGN(64) { __bss_start = .; *(.sbss .sbss.*) *(.bss .bss.*) *(COMMON) . = ALIGN(8); __bss_end = .; } > RAM
 .stack (NOLOAD) : ALIGN(64) { __stack_bottom = .; . += 65536; __stack_top = .; } > RAM
 ASSERT(tohost != fromhost, "distinct HTIF words")
 ASSERT((tohost % 64) == 0 && (fromhost % 64) == 0, "HTIF alignment")
 ASSERT(ADDR(.text)+SIZEOF(.text) <= TINY_FULL_DEVICE_BASE ||
        ADDR(.text) >= TINY_FULL_DEVICE_BASE+TINY_FULL_DEVICE_BYTES,
        "text overlaps device DRAM")
 ASSERT(ADDR(.rodata)+SIZEOF(.rodata) <= TINY_FULL_DEVICE_BASE ||
        ADDR(.rodata) >= TINY_FULL_DEVICE_BASE+TINY_FULL_DEVICE_BYTES,
        "rodata overlaps device DRAM")
 ASSERT(ADDR(.data)+SIZEOF(.data) <= TINY_FULL_DEVICE_BASE ||
        ADDR(.data) >= TINY_FULL_DEVICE_BASE+TINY_FULL_DEVICE_BYTES,
        "data overlaps device DRAM")
 ASSERT(ADDR(.tohost)+SIZEOF(.tohost) <= TINY_FULL_DEVICE_BASE ||
        ADDR(.tohost) >= TINY_FULL_DEVICE_BASE+TINY_FULL_DEVICE_BYTES,
        "HTIF overlaps device DRAM")
 ASSERT(ADDR(.bss)+SIZEOF(.bss) <= TINY_FULL_DEVICE_BASE ||
        ADDR(.bss) >= TINY_FULL_DEVICE_BASE+TINY_FULL_DEVICE_BYTES,
        "BSS overlaps device DRAM")
 ASSERT(ADDR(.stack)+SIZEOF(.stack) <= TINY_FULL_DEVICE_BASE ||
        ADDR(.stack) >= TINY_FULL_DEVICE_BASE+TINY_FULL_DEVICE_BYTES,
        "stack overlaps device DRAM")
 ASSERT(__stack_top <= ORIGIN(RAM)+LENGTH(RAM), "RAM overflow")
 ASSERT((tiny_full_mailbox % 64) == 0, "mailbox alignment")
 /DISCARD/ : { *(.comment) *(.eh_frame*) }
}
