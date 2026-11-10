"""Existing register-group feature declarations delegating to the selected legacy helper."""

from __future__ import annotations

from merlin.compiler.selection.features import ImprFeature

LMUL_GROUP_PREFIX = "lmul_group_m"


LMUL_GROUP_SENTINEL = "lmul_register_group"


def lmul_group_feature(lmul: int) -> str:
    """The name of the default-off feature that pins the auto-vectorizer's register group to ``lmul``.

    Every whole-register width is registered eagerly (there are four), so a name resolves in the
    lowering SUBPROCESS as well as in the parent -- the same reproducibility requirement
    :func:`_try_lazy_register` exists for.
    """
    from merlin.llvmlower.lmul_group import LMUL_LADDER

    if int(lmul) not in LMUL_LADDER:
        from merlin.llvmlower.lmul_group import LmulDerivationError

        raise LmulDerivationError(f"LMUL={lmul!r} is not one of {LMUL_LADDER}")
    return f"{LMUL_GROUP_PREFIX}{int(lmul)}"


def ensure_lmul_group(
    *, operand_bits: int, acc_bits: int, vlen: int | None = None, max_group_elems: int | None = None
) -> str:
    """DERIVE the register-group width from the datapath and return the feature that pins it.

    The width is ``lmul_group.group_lmul`` -- the smallest whole-register group at which the
    narrowest operand's group stops being a fraction, capped by what the target's VLEN can usefully
    hold. Nobody types a 4: an ``i8 x i8 -> i32`` contraction derives 4 (which is what the expert
    XNNPACK qd8 kernel runs at), an ``f32`` one derives 1, a ``bf16 -> f32`` one derives 2.
    """
    from merlin.llvmlower.lmul_group import group_lmul

    return lmul_group_feature(
        group_lmul(operand_bits=operand_bits, acc_bits=acc_bits, vlen=vlen, max_group_elems=max_group_elems)
    )


def ensure_lmul_group_for_elem_types(
    a: str, b: str, c: str, *, vlen: int | None = None, max_group_elems: int | None = None
) -> str:
    """:func:`ensure_lmul_group` for a contraction named by its MLIR element types ``a x b -> c``."""
    from merlin.llvmlower.lmul_group import group_lmul_for_elem_types

    return lmul_group_feature(group_lmul_for_elem_types(a, b, c, vlen=vlen, max_group_elems=max_group_elems))


def _register_lmul_groups() -> list[str]:
    from merlin.compiler.optimizations.registry import register
    from merlin.llvmlower.lmul_group import LMUL_LADDER, lmul_cflags

    names = []
    for _lmul in LMUL_LADDER:
        _name = f"{LMUL_GROUP_PREFIX}{_lmul}"
        register(
            ImprFeature(
                name=_name,
                action_class="KNOB",
                description=(
                    f"Pin the vector REGISTER-GROUP WIDTH of auto-vectorized code to LMUL={_lmul}, "
                    f"directly (an -mllvm backend option), without changing any tile or vector size. "
                    f"This is the seam the `vector.lmul` divergence actually wants: the N-tile route to "
                    f"the same axis also widens the tail transfer, which is a whole-model scalar-fallback "
                    f"cliff. Derive the width with `ensure_lmul_group(...)` rather than naming a number "
                    f"-- it is acc_bits/operand_bits, capped by what the VLEN can hold. Default-off; a "
                    f"build that does not name it gets byte-identical flags."
                ),
                edit_cflags=(lambda c, _l=_lmul: [*c, *lmul_cflags(_l)]),
            )
        )
        names.append(_name)
    return names
