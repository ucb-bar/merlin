"""Bounded lexical spans for compiler-owned LLVM source proposals.

No assembler, subprocess, target lookup, reference or qualification is exposed.
The lexical productions match the existing late-legalization reader; consuming
these spans establishes no semantic or executable-program proof.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LLVMToken:
    text: str
    start: int
    end: int


def llvm_tokens(text: str, *, max_bytes: int = 1 << 20) -> list[LLVMToken]:
    """LLVM punctuation, comments, identifiers and quoted strings with spans.

    This is an intentionally small lexer, not a permissive LLVM module parser.
    LLVM's assembler verifies the whole module at the public build seam. The
    matcher below accepts only explicit instruction productions with typed
    operands and a recognized statement boundary.
    """
    if (
        type(text) is not str
        or type(max_bytes) is not int
        or not 0 < max_bytes <= 1 << 20
        or len(text.encode()) > max_bytes
    ):
        raise ValueError("LLVM lexical input exceeds the explicit source bound")
    result: list[LLVMToken] = []
    i = 0
    punctuation = frozenset("(){}[],=<>:*!")
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        if text[i] == ";":
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
            continue
        start = i
        if text[i] in "%@":
            i += 1
        if i < len(text) and text[i] == '"':
            i += 1
            while i < len(text) and text[i] != '"':
                if text[i] == "\\":
                    # LLVM quoted bytes use two hexadecimal escape digits.
                    i += 3
                else:
                    i += 1
            if i >= len(text):
                raise ValueError("unterminated LLVM quoted token")
            i += 1
        elif i == start and text[i] in punctuation:
            i += 1
        else:
            while i < len(text) and not text[i].isspace() and text[i] not in punctuation and text[i] not in ';"':
                i += 1
        if i == start:
            raise ValueError("unsupported LLVM token")
        result.append(LLVMToken(text[start:i], start, i))
    return result
