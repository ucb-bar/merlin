"""Building device images for targets whose ISA is a fixed-format re-encoding of a stock one.

A growing class of accelerators keeps a stock ISA's opcode/funct *values* but re-packs each
instruction's fields into a wider fixed-width word — the immediate contiguous, the registers widened.
That single change breaks the stock toolchain in exactly two places, and historically each vendor
answered both by shipping a compiler fork:

* the **assembler** does not know the target's CUSTOM-slot mnemonics, and
* the **linker** writes each relocation's bits at the stock field positions, which in a wide word land
  on register/funct fields and corrupt the image.

Neither needs a fork. :mod:`~merlin.targetgen.isa_transcode` re-maps stock words into the target's
format from its derived :class:`~merlin.targetgen.isa_model.IsaModel`; :mod:`.boot` does the same for a
stock-assembled boot object (CUSTOM slots through the derived encoder); :mod:`.link` runs a stock
``ld.lld`` for layout only and re-applies every relocation at the derived field positions.

These modules live under ``targetgen`` rather than ``runtime/backends`` because every fact they use is
an ISA fact and ``target`` is a parameter throughout — nothing here knows which device it is building
for. They were previously named after the first target that exercised them, which made a general
capability read as one vendor's plumbing.
"""
