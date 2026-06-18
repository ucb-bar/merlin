"""Robust, non-regex decoders for the kernel-mining pipeline.

These replace the fragile regex-over-C / regex-over-objdump-text extraction. The asm decoder
(``asm_mc``) is the authoritative substrate every framework lowers to; the source decoder
(``clang_ast``, added later) is a typed cross-check.
"""
