// Assembly wrapper to embed a large VMFB as .rodata without using iree-c-embed-data.
// The VMFB path is provided via -DTINYLLAMA_VMFB_PATH=... at compile time.
// Provides symbols: _binary_tinyllama_vmfb_start, _binary_tinyllama_vmfb_end

    .section .rodata, "a", @progbits
    .global _binary_tinyllama_vmfb_start
    .global _binary_tinyllama_vmfb_end
    .balign 16
_binary_tinyllama_vmfb_start:
#ifdef TINYLLAMA_VMFB_PATH
    .incbin TINYLLAMA_VMFB_PATH
#else
    .error "TINYLLAMA_VMFB_PATH must be defined"
#endif
_binary_tinyllama_vmfb_end:
