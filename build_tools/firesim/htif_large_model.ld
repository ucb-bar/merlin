/* htif_large_model.ld — Linker script for large embedded models (>100MB).
 * Places .rodata AFTER heap/stack so that large model weights don't push
 * stack/heap symbols beyond the medany 2GB PC-relative addressing range.
 *
 * Memory layout (approximate for 1.1GB TinyLlama):
 *   0x80000000: .text (~5MB)
 *   0x80500000: .data/.bss (small)
 *   0x80600000: heap (1GB)
 *   0xC0600000: stack (128MB)
 *   0xC8600000: .rodata.vmfb (1.1GB) — accessed via absolute addressing
 */

OUTPUT_ARCH ("riscv")
ENTRY (_start)

SECTIONS
{
    . = 0x80000000;

    PROVIDE (__stack_size_min = 128M);

    .text : {
        *(.text.init)
        *(.text.unlikely .text.unlikely.*)
        *(.text.startup .text.startup.*)
        *(.text .text.*)
        *(.gnu.linkonce.t.*)
    }
    PROVIDE (_etext = .);

    /* Static Thread Local Storage template */
    .tdata : {
        PROVIDE_HIDDEN (__tdata_start = .);
        *(.tdata .tdata.*)
        *(.gnu.linkonce.td.*)
        PROVIDE_HIDDEN (__tdata_end = .);
    }
    PROVIDE (__tdata_size = SIZEOF (.tdata));

    .tbss (NOLOAD) : {
        PROVIDE_HIDDEN (__tbss_start = .);
        PROVIDE_HIDDEN (__tbss_offset = ABSOLUTE (__tbss_start - __tdata_start));
        *(.tbss .tbss.*)
        *(.gnu.linkonce.tb.*)
        *(.tcommon)
        PROVIDE_HIDDEN (__tbss_end = .);
    }
    PROVIDE (__tbss_size = SIZEOF (.tbss));

    /* Regular rodata: switch tables, string literals, small constants.
     * Must stay near .text for medany PC-relative addressing. */
    .rodata : {
        *(.rodata .rodata.*)
        *(.gnu.linkonce.r.*)
    }

    .preinit_array : {
        PROVIDE_HIDDEN (__preinit_array_start = .);
        KEEP (*(.preinit_array))
        PROVIDE_HIDDEN (__preinit_array_end = .);
    }
    .init_array : {
        PROVIDE_HIDDEN (__init_array_start = .);
        KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
        KEEP (*(.init_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .ctors))
        PROVIDE_HIDDEN (__init_array_end = .);
    }
    .fini_array : {
        PROVIDE_HIDDEN (__fini_array_start = .);
        KEEP (*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
        KEEP (*(.fini_array EXCLUDE_FILE (*crtbegin.o *crtbegin?.o *crtend.o *crtend?.o ) .dtors))
        PROVIDE_HIDDEN (__fini_array_end = .);
    }
    .ctors : { KEEP (*crtbegin.o(.ctors)) KEEP (*crtbegin?.o(.ctors)) KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o) .ctors)) KEEP (*(SORT(.ctors.*))) KEEP (*(.ctors)) }
    .dtors : { KEEP (*crtbegin.o(.dtors)) KEEP (*crtbegin?.o(.dtors)) KEEP (*(EXCLUDE_FILE (*crtend.o *crtend?.o) .dtors)) KEEP (*(SORT(.dtors.*))) KEEP (*(.dtors)) }

    /* Host-Target Interface */
    .htif ALIGN(0x40) : { *(.htif) }

    .data : ALIGN (8) { *(.data) *(.gnu.linkonce.d.*) }
    .sdata : {
        PROVIDE (__global_pointer$ = . + 0x800);
        *(.sdata .sdata.*) *(.gnu.linkonce.s.*)
        *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2)
        *(.srodata .srodata.*)
    }
    PROVIDE (_edata = .);

    .bss : ALIGN (8) {
        PROVIDE_HIDDEN (__bss_start = .);
        *(.sbss .sbss.*) *(.gnu.linkonce.sb.*)
        *(.bss .bss.*) *(.gnu.linkonce.b.*)
        *(COMMON)
        PROVIDE_HIDDEN (__bss_end = .);
    }
    . = ALIGN (8);
    PROVIDE (_end = .);

    /* Reserve heap space — BEFORE large rodata */
    PROVIDE(__heap_size = 1024M);
    . += __heap_size;
    . = ALIGN (4K);
    PROVIDE (__heap_end = .);

    /* Stack — also before large rodata */
    PROVIDE (__tcb_size = __tbss_end - __tdata_start);
    PROVIDE (__stack_align = MAX (ALIGNOF (.tdata), 0x10));
    PROVIDE (__stack_start = ALIGN (__stack_align));
    PROVIDE (__stack_shift = LOG2CEIL( ALIGN (__stack_size_min + __tcb_size, __stack_align)));
    PROVIDE (__stack_size = 1 << __stack_shift);

    /* Large VMFB weights — placed LAST to avoid medany overflow.
     * The .vmfb_data section is defined in tinyllama_vmfb_embed.S and
     * accessed via absolute pointer symbols, not PC-relative addressing.
     *
     * CRITICAL: __stack_start is at the current location counter (right
     * after heap). The stack grows DOWNWARD from __stack_start + __stack_size.
     * We must place .vmfb_data AFTER the stack end to avoid overlap.
     * Use a fixed offset: __stack_start + __stack_size (128MB).
     */
    . = __stack_start + __stack_size;
    . = ALIGN (16);
    .vmfb_data : {
        *(.vmfb_data)
    }

    PROVIDE (__boot_hart = 0);
}
