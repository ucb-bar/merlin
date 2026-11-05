"""Target-agnostic analyses that derive facts from IR, shapes and a target's own derived tables.

Every module here reads its subject (a parsed module, a disassembly, the target's RTL-derived decode
table or contract declaration) and reports what it found; none carries a kernel, schedule, baseline,
measured answer or reference runtime. That is why these analyses are part of the reviewed public
compiler source while ``merlin.kernels`` and ``merlin.mining`` stay withheld from an author: the older
``merlin.kernels.*`` paths are compatibility identities for the modules that moved here.
"""
