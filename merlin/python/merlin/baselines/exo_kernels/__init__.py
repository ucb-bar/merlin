"""EXO kernel sources for the K1-RVV whole-model baseline arm.

These are hand-authored EXO schedules (the EXO DSL *is* the authoring surface — you write a
reference nest and schedule it down to RVV intrinsics). ``exocc`` lowers them to C that we
cross-compile for ``rv64gcv`` with the SpacemiT clang. The generated C (build output) lives under
``build/baselines/exo/`` and is NOT committed; only these ``.py`` sources are.

The K1 X60 is VLEN=256 (8x f32 per ``m1`` register), so we author an 8-wide RVV register class
(EXO ships a 4-wide ``rvv.py`` for VLEN=128). The ``vl``-parameterised ``__riscv_v*`` intrinsics
are width-agnostic at the ISA level; only the tile width the scheduler blocks by changes.
"""
