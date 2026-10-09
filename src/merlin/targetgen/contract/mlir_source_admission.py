"""Caller-selected lexical limits before MLIR attribute/operation allocation.

This guard uses actual lexer tokens, never rewrites source or interprets it.
It bounds source bytes and syntax nesting, and can exclude aggregate literals
and oversized integer types before upstream parsers allocate their storage.
It grants no semantic, execution, physical or compiled resource authority.
Allowing aggregates leaves their allocation safety with the caller's parser.
"""

from __future__ import annotations


class MlirSourceUnavailable(ValueError):
    """Source exceeds the explicitly selected lexical observation domain."""


def admit_mlir_source(
    text,
    *,
    max_source_bytes,
    max_nesting,
    max_integer_bits,
    allow_dense,
    allow_dense_resource,
):
    """Check bounded text without constructing attributes, shaped data or IR.

    These are observer input limits, not a subprocess memory/time lease.
    Strings, comments and prefixed SSA/symbol names cannot supply type or
    literal tokens. The caller still verifies the actual parsed operation
    roster, types, metadata, original semantics and numerical/effect domains.
    Bare i<N>/si<N>/ui<N> identifiers are conservatively screened even where a
    dialect uses that spelling for an attribute or port name. This is an input
    domain restriction, not grammatical type classification or renaming.
    """
    from xdsl.utils.exceptions import ParseError
    from xdsl.utils.lexer import Input
    from xdsl.utils.mlir_lexer import MLIRLexer
    from xdsl.utils.mlir_lexer import MLIRTokenKind as K

    if (
        type(text) is not str
        or type(max_source_bytes) is not int
        or max_source_bytes < 1
        or type(max_nesting) is not int
        or max_nesting < 1
        or type(max_integer_bits) is not int
        or max_integer_bits < 1
        or type(allow_dense) is not bool
        or type(allow_dense_resource) is not bool
    ):
        raise MlirSourceUnavailable("MLIR source needs explicit bounded lexical selections")
    # Reject by character count first, before allocating an encoded copy of a
    # very large string. UTF-8 is then counted exactly, including comments.
    try:
        if len(text) > max_source_bytes or len(text.encode()) > max_source_bytes:
            raise MlirSourceUnavailable("MLIR source byte bound exceeded")
    except UnicodeEncodeError as error:
        raise MlirSourceUnavailable("MLIR source has no complete UTF-8 encoding") from error
    openings = {
        K.LESS: K.GREATER,
        K.L_PAREN: K.R_PAREN,
        K.L_SQUARE: K.R_SQUARE,
        K.L_BRACE: K.R_BRACE,
        K.FILE_METADATA_BEGIN: K.FILE_METADATA_END,
    }
    closing = set(openings.values())
    stack, previous = [], None
    denied = {
        name for name, allowed in (("dense", allow_dense), ("dense_resource", allow_dense_resource)) if not allowed
    }
    try:
        lexer = MLIRLexer(Input(text, "bounded-mlir-source"))
        while True:
            token = lexer.lex()
            if token.kind is K.EOF:
                break
            if (
                previous is not None
                and previous.kind is K.BARE_IDENT
                and previous.text in denied
                and token.kind is K.LESS
            ):
                raise MlirSourceUnavailable("aggregate literals are unsupported before shaped parser allocation")
            if token.kind is K.BARE_IDENT:
                for prefix in ("si", "ui", "i"):
                    digits = token.text.removeprefix(prefix) if token.text.startswith(prefix) else ""
                    if digits and digits.isascii() and digits.isdigit():
                        significant = digits.lstrip("0") or "0"
                        # The positive budget's bit length is a loose safe
                        # decimal-digit bound, without rendering huge caller
                        # integers or converting obviously excessive tokens.
                        if len(significant) > max_integer_bits.bit_length():
                            raise MlirSourceUnavailable("MLIR scalar width exceeds the selected lexical bound")
                        try:
                            width = int(significant)
                        except ValueError as error:
                            raise MlirSourceUnavailable(
                                "MLIR integer type token is not representable by this lexer domain"
                            ) from error
                        if not 1 <= width <= max_integer_bits:
                            raise MlirSourceUnavailable("MLIR scalar width exceeds the selected lexical bound")
                        break
            if token.kind in openings:
                stack.append(openings[token.kind])
                if len(stack) > max_nesting:
                    raise MlirSourceUnavailable("MLIR syntax nesting bound exceeded")
            elif token.kind in closing:
                if not stack or stack.pop() is not token.kind:
                    raise MlirSourceUnavailable("MLIR source has mismatched lexical delimiters")
            previous = token
    except ParseError as error:
        raise MlirSourceUnavailable("MLIR source cannot be lexed completely") from error
    if stack:
        raise MlirSourceUnavailable("MLIR source has incomplete lexical delimiters")
