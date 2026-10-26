"""UNAPPROVED data-only extension proposal; not used by the native controller.

The retained integer counterexample has one exact three-line `decide` diagnostic
with a word-literal equality. This finite grammar generalizes the literal values,
not arbitrary Lean messages/propositions. It supplies no proof or launch authority.
All other multiline diagnostics remain unclassified. Review before integration.
"""

from __future__ import annotations


def _decimal(token):
    return bool(token) and token.isascii() and token.isdigit()


def _word_equality(line):
    if not line.startswith("  ") or line.startswith("   "):
        return False
    parts = line[2:].split(" ")
    return (
        len(parts) == 7
        and parts[0] == parts[4] == "word"
        and parts[3] == "="
        and all(_decimal(parts[index]) for index in (1, 2, 5, 6))
    )


def word_decide_refusal(stdout, sources):
    """Require the ENTIRE buffer to consist of exact known-format refusals.

    Sources are exact bound paths. A source header by itself grants nothing.
    Unknown/mixed/runtime text, additional continuation lines and even other
    well-formed Lean propositions return False rather than broadening the rule.
    """
    lines = stdout.splitlines()
    index, blocks = 0, 0
    while index < len(lines):
        if lines[index] == "":
            index += 1
            continue
        path, sep, rest = lines[index].partition(":")
        row, sep2, rest = rest.partition(":")
        column, sep3, message = rest.partition(":")
        if not (
            path in sources
            and sep
            and sep2
            and sep3
            and _decimal(row)
            and _decimal(column)
            and message == " error: Tactic `decide` proved that the proposition"
            and index + 2 < len(lines)
            and _word_equality(lines[index + 1])
            and lines[index + 2] == "is false"
        ):
            return False
        blocks += 1
        index += 3
    return blocks > 0
