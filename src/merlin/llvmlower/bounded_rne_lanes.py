"""CPU emission for independent, source-proven bounded binary32 RNE lanes.

This emitter implements a numeric contract; it does not select source operations
or authorize moving loads across stores. A caller must independently prove the
source clamp bounds, fixed ties-to-even rounding and absence of observable FP
exception flags. All input values are supplied by value before any output write.
The caller supplies storage for exactly ``lanes`` signed32 output words.

Proof: signed bounds through 24 bits are integral and exactly representable in
binary32. Integral clamping commutes with ties-to-even rounding. The bounded
result fits signed32, so fixed-RNE conversion has no overflow. Each lane has its
own temporary, floating inputs cannot overlap integer outputs, and declared
floating clobbers exclude every temporary from input allocation. Grouping the
stages therefore changes no lane's arithmetic or ambient rounding dependence.
This is a CPU emission primitive, not a graph transformation or alias proof.
"""

from __future__ import annotations


def emit_bounded_rne_lanes(name: str, *, bits: int, lanes: int, host_isa: str) -> str:
    """Emit an inline C helper; absent/unsupported CPU policies are refused.

    The ``portable`` implementation is a paired native oracle. ``rv64gc`` groups
    independent clamp/conversion stages so scheduling can overlap their latency.
    It uses no accelerator instruction and changes no source default policy.
    Source NaN-to-integer poison may be refined; finite inputs and infinities
    obey the exact clamp-before-RNE contract under every ambient rounding mode.
    """
    if (
        not isinstance(name, str)
        or not name
        or name[0].isdigit()
        or any(not (c.isascii() and (c.isalnum() or c == "_")) for c in name)
    ):
        raise ValueError("name must be a C identifier")
    if type(bits) is not int or not 1 <= bits <= 24:
        raise ValueError("exact binary32 signed bounds require bits in [1,24]")
    if type(lanes) is not int or not 1 <= lanes <= 8:
        raise ValueError("bounded register packet requires lanes in [1,8]")
    if host_isa not in ("portable", "rv64gc"):
        raise ValueError("explicit supported host ISA is required")
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    lo, hi = float(lower).hex() + "f", float(upper).hex() + "f"
    args = ", ".join(f"float a{i}" for i in range(lanes))
    lines = [f"static inline void {name}({args}, int32_t out[{lanes}]) {{"]
    if host_isa == "portable":
        for i in range(lanes):
            lines.extend(
                [
                    f"  float x{i}=a{i} < {lo} ? {lo} : (a{i} > {hi} ? {hi} : a{i});",
                    f"  out[{i}]=(int32_t)__builtin_roundevenf(x{i});",
                ]
            )
    else:
        lines.append("  int32_t " + ", ".join(f"r{i}" for i in range(lanes)) + ";")
        instructions = [f"fmax.s ft{i}, %{lanes + i}, %{2 * lanes}" for i in range(lanes)]
        instructions += [f"fmin.s ft{i}, ft{i}, %{2 * lanes + 1}" for i in range(lanes)]
        instructions += [f"fcvt.w.s %{i}, ft{i}, rne" for i in range(lanes)]
        outputs = ",".join(f'"=r"(r{i})' for i in range(lanes))
        inputs = ",".join([*(f'"f"(a{i})' for i in range(lanes)), f'"f"({lo})', f'"f"({hi})'])
        clobbers = ",".join(f'"ft{i}"' for i in range(lanes))
        lines.append(
            '  __asm__("' + "; ".join(instructions) + '" : ' + outputs + " : " + inputs + " : " + clobbers + ");"
        )
        lines += [f"  out[{i}]=r{i};" for i in range(lanes)]
    return "\n".join([*lines, "}", ""])
