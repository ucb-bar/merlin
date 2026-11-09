"""Checked lossless serialization at the opt-in deployable-kernel seam."""

from __future__ import annotations


def checked_generic_text(module) -> str:
    """Retain operation attributes, properties and region bodies or refuse.

    Named/custom assembly can omit extension attributes and even a named
    operation's explicit nonstandard body. Deployable code keys and compiler
    inputs therefore use generic assembly, independently round-tripped through
    the ordinary source parser. This does not alter the shared printer default
    or the ordinary monolithic build path. SSA spelling is not structural.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    serialized = text(module, generic=True)
    try:
        parsed = parse_mlir_text(serialized)
        parsed.verify()
    except Exception as error:
        raise ValueError("deployable generic IR cannot be independently round-tripped") from error
    if not module.is_structurally_equivalent(parsed):
        raise ValueError("deployable generic IR changed attributes, properties, types or region structure")
    return serialized
