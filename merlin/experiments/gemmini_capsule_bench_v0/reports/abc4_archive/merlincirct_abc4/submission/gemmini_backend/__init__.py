"""Gemmini out-of-tree MLIR backend (xDSL), merlin_assisted arm.

A real xDSL backend: a ``merlin_iface`` input dialect and a ``gemmini`` target
dialect (IRDL ops/types/verifiers), a rewrite-pattern lowering pass between
them, and command-buffer / RoCC-LLVM emitters that walk the target IR.  Imports
only the public ``xdsl`` framework — no ``merlin`` runtime / oracle.
"""
