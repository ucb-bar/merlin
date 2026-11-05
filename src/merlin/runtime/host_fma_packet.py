"""Explicit host CPU emission of source-proven scalar FMA packets.

The caller selects the CPU ISA/ABI and supplies the source/effect contract.
No accelerator facts or graph selection are encoded here.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.llvmlower.constant_float_clamp import _finite_f32
from merlin.llvmlower.constant_fma_packet import ConstantFmaPacket


@dataclass(frozen=True)
class RV64GCScalarFmaCapability:
    isa: str = "rv64gc"
    abi: str = "lp64d"
    ieee_f32_fma: bool = True
    ieee_gradual_underflow: bool = True

    def emit(self, packet: ConstantFmaPacket, temporary: str) -> str:
        """Emit exact source input order with protected destructive outputs.

        Early-clobber prevents a later input from sharing an already written
        output register, including cross-lane shared SSA. Tied inputs reduce live
        registers; LLVM preserves any other uses by ordinary register allocation.
        FMA uses the original default environment; no rounding/flag policy is
        changed by this emitter. The caller separately proves source legality.
        """
        if (
            (self.isa, self.abi) != ("rv64gc", "lp64d")
            or self.ieee_f32_fma is not True
            or self.ieee_gradual_underflow is not True
        ):
            raise ValueError("explicit RV64GC IEEE scalar FMA capability required")
        if type(packet) is not ConstantFmaPacket:
            raise ValueError("typed source FMA packet required")
        count = len(packet.calls)
        if count not in (2, 4):
            raise ValueError("supported scalar FMA resource packets have 2 or 4 lanes")
        lhs = [call.operands[0] for call in packet.calls]
        rhs = [call.operands[1] for call in packet.calls]
        addend = [call.operands[2] for call in packet.calls]
        expected = {"constant_rhs": (1,), "constant_addend": (2,), "constant_lhs_addend": (0, 2)}.get(packet.mode)
        if expected is None or len(packet.constant_words) != len(expected):
            raise ValueError("unsupported source-proven constant FMA mode")
        for axis, word in zip(expected, packet.constant_words, strict=True):
            for call in packet.calls:
                decoded = _finite_f32(call.operands[axis])
                if decoded is None or decoded[0] != word:
                    raise ValueError("source FMA constant witness changed")
        if any(
            not call.operands[axis].startswith("%")
            for call in packet.calls
            for axis in range(3)
            if axis not in expected
        ):
            raise ValueError("source FMA variable witness changed")
        dtype = "{ " + ", ".join(["float"] * count) + " }"
        outputs = ["=&f"] * count
        if packet.mode == "constant_rhs":
            arguments = [
                *(f"float {value}" for value in lhs),
                f"i32 {packet.constant_words[0]}",
                *(f"float {value}" for value in addend),
            ]
            instructions = f"fmv.w.x ft0,${2 * count}" + "".join(
                f"\\0Afmadd.s ${lane},${lane + count},ft0,${lane}" for lane in range(count)
            )
            constraints = [*outputs, *["f"] * count, "r", *(str(lane) for lane in range(count)), "~{ft0}"]
        elif packet.mode == "constant_addend":
            arguments = [
                *(f"float {value}" for value in rhs),
                f"i32 {packet.constant_words[0]}",
                *(f"float {value}" for value in lhs),
            ]
            instructions = f"fmv.w.x ft0,${2 * count}" + "".join(
                f"\\0Afmadd.s ${lane},${lane},${lane + count},ft0" for lane in range(count)
            )
            constraints = [*outputs, *["f"] * count, "r", *(str(lane) for lane in range(count)), "~{ft0}"]
        else:
            arguments = [*(f"i32 {word}" for word in packet.constant_words), *(f"float {value}" for value in rhs)]
            instructions = f"fmv.w.x ft0,${count}\\0Afmv.w.x ft1,${count + 1}" + "".join(
                f"\\0Afmadd.s ${lane},ft0,${lane},ft1" for lane in range(count)
            )
            constraints = [*outputs, "r", "r", *(str(lane) for lane in range(count)), "~{ft0}", "~{ft1}"]
        lines = [
            f'{temporary} = call {dtype} asm "{instructions}", "'
            + ",".join(constraints)
            + '"('
            + ", ".join(arguments)
            + ")"
        ]
        lines.extend(
            f"  {call.result} = extractvalue {dtype} {temporary}, {lane}" for lane, call in enumerate(packet.calls)
        )
        return "\n".join(lines)
