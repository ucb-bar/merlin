"""Combine explicitly selected C sources into a standalone source/header pair.

This is packaging, not a C preprocessor. Providers supply headers in dependency
order. Literal project includes must resolve to an already supplied header;
system includes and preprocessor conditions remain for the consumer's compiler.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Mapping
from pathlib import PurePosixPath


def c_source_pair(
    name: str,
    *,
    public_headers: Mapping[str, str],
    private_headers: Mapping[str, str],
    sources: Mapping[str, str],
    declarations: str = "",
    implementation: str = "",
) -> dict[str, str]:
    """Render C/H files without checkout includes or a generated main function.

    File names are logical source names used in diagnostics, not local paths.
    The caller owns source selection, symbol compatibility and platform ABI.
    The consumer still supplies any required C library and platform services.
    """
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    identifier_chars = letters + "0123456789_"
    if not name or len(name) > 64 or name[0] not in letters or any(c not in identifier_chars for c in name):
        raise ValueError("package name must be a C identifier beginning with a letter")
    members = (*public_headers, *private_headers, *sources)
    if not sources or len(set(members)) != len(members):
        raise ValueError("C package needs sources and distinct logical member names")
    for member in members:
        path = PurePosixPath(member)
        if (
            not member
            or path.is_absolute()
            or path.as_posix() != member
            or ".." in path.parts
            or any(c not in identifier_chars + "./-" for c in member)
        ):
            raise ValueError("invalid logical C source name")

    available: set[str] = set()

    def flatten(member: str, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty C package member")
        lines = [f"#line 1 {json.dumps(member)}"]
        for line in text.splitlines():
            directive = line.lstrip()
            if directive.startswith("#"):
                directive = directive[1:].lstrip()
                end = 0
                while end < len(directive) and directive[end] in identifier_chars:
                    end += 1
                keyword, operand = directive[:end], directive[end:].strip()
            else:
                keyword, operand = "", ""
            if keyword in ("include", "include_next"):
                if keyword != "include" or len(operand) < 3 or operand[0] not in ('"', "<"):
                    raise ValueError(f"nonliteral or unsupported include in {member}")
                closing = '"' if operand[0] == '"' else ">"
                if not operand.endswith(closing) or closing in operand[1:-1]:
                    raise ValueError(f"nonliteral or unsupported include in {member}")
                if operand[0] == '"':
                    target = posixpath.normpath(str(PurePosixPath(member).parent / operand[1:-1]))
                    if target not in available:
                        raise ValueError(f"unresolved or out-of-order include {operand!r} in {member}")
                    # Keep line numbering for diagnostics after removing it.
                    lines.append("")
                    continue
            lines.append(line)
        return "\n".join(lines) + "\n"

    public = []
    for member, text in public_headers.items():
        public.append(flatten(member, text))
        available.add(member)
    guard = "MERLIN_C_PACKAGE_" + name.upper() + "_H"
    header = (
        f"#ifndef {guard}\n#define {guard}\n\n"
        + "\n".join(public)
        + '\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
        + declarations
        + "\n#ifdef __cplusplus\n}\n#endif\n#endif\n"
    )
    body = [f'#include "{name}.h"\n']
    # Clang normally suppresses unused inline helpers in included headers.
    # Preserve that behavior for private headers embedded in a C translation
    # unit, without suppressing warnings in the implementation below them.
    body.append(
        '#ifdef __clang__\n#pragma clang diagnostic push\n#pragma clang diagnostic ignored "-Wunused-function"\n#endif'
    )
    for member, text in private_headers.items():
        body.append(flatten(member, text))
        available.add(member)
    body.append("#ifdef __clang__\n#pragma clang diagnostic pop\n#endif")
    for member, text in sources.items():
        body.append(flatten(member, text))
    body.append(implementation)
    return {name + ".h": header, name + ".c": "\n".join(body) + "\n"}
