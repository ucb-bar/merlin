"""The RoCC coprocessor interface: decoding a trace of it, and emitting one.

RoCC is a RISC-V *standard* — a custom-opcode coprocessor interface that any accelerator on a Rocket or
BOOM core can sit behind — so tooling for it belongs to the interface, not to whichever accelerator
happens to be the first one plugged in. Both modules here take ``target`` as a REQUIRED parameter and
read every encoding fact (custom opcode, funct3, the funct-to-class table, operand field positions)
from that target's RTL-derived facts. Neither has a default target, on purpose: a decoder that guesses
whose ISA it is reading will happily decode one accelerator's trace against another's table and report
a clean result.

They stay in-tree rather than moving out with a reference target because four generic core modules
consume them (the capsule runner, the CIRCT gate and introspection, and the assembler), so evicting
them would break the core rather than de-couple it.
"""
