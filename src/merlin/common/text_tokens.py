"""Small lexical primitives for explicitly bounded source-format readers.

These tokens preserve quoted literals and discard C-style comments. Callers
must validate the complete grammar they accept. Lexing supplies no semantics.
"""

from dataclasses import dataclass


def is_identifier(value: object) -> bool:
    """Whether a value is a nonempty portable C/LLVM or shell identifier."""
    return isinstance(value, str) and value.isascii() and value.isidentifier()


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def source_tokens(text: str) -> list[Token]:
    """Lex words, quoted literals and individual punctuation, refusing open literals/comments."""
    tokens, position = [], 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        if text.startswith("//", position):
            end = text.find("\n", position + 2)
            position = len(text) if end < 0 else end + 1
            continue
        if text.startswith("/*", position):
            end = text.find("*/", position + 2)
            if end < 0:
                raise ValueError("unterminated source comment")
            position = end + 2
            continue
        start = position
        if text[position] in "\"'":
            quote = text[position]
            position += 1
            while position < len(text) and text[position] != quote:
                position += 2 if text[position] == "\\" else 1
            if position >= len(text):
                raise ValueError("unterminated source literal")
            position += 1
        elif text[position].isalnum() or text[position] in "_$":
            position += 1
            while position < len(text) and (text[position].isalnum() or text[position] in "_$"):
                position += 1
        else:
            position += 1
        tokens.append(Token(text[start:position], start, position))
    return tokens
