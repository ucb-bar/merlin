"""Which capsule entries test something that behaves differently below zero.

One rule, read by the synthesizer (which gives such members a signed stimulus) and by the gate that
holds the corpus to it, so the two cannot disagree about what needs one: an entry is sign-sensitive
when it carries a readout stage, on the operation or on any of its per-command entries, or IS such
a stage standing alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_sign_sensitive(entry: Mapping[str, Any]) -> bool:
    """``entry`` is a profile entry or an operation's ``{"op": ..., **attributes}``."""
    from merlin.runtime.commandbuffer import EPILOGUE_STAGE_SET

    if str(entry.get("op") or "") in EPILOGUE_STAGE_SET:
        return True
    commands = [entry, *(c for c in entry.get("matmuls") or () if isinstance(c, Mapping))]
    return any(str(stage) in EPILOGUE_STAGE_SET for command in commands for stage in command.get("epilogue") or ())


def has_signed_integer_operands(entry: Mapping[str, Any]) -> bool:
    """``entry`` feeds a signed, element-scaled integer format to the unit.

    Operand sign extension is itself sign-sensitive: a load that zero-extends where it should
    sign-extend reads ``-4`` as ``252``, and on the non-negative default stimulus the two agree on
    every element. Kept separate from :func:`is_sign_sensitive`, which is about readout stages and is
    the rule the corpus sign-coverage gate holds historical capsules to.
    """
    from merlin.common import quant_formats

    token = entry.get("operand_dtype")
    if not isinstance(token, str) or not token:
        return False
    try:
        fmt = quant_formats.get(token)
    except KeyError:
        return False
    return bool(fmt.signed) and not fmt.is_float and not fmt.is_block_scaled
