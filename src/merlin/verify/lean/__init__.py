"""Python side of the Lean verification layer: stage capture, certificates, export and receipts.

The Lean checkers live in ``merlin/lean`` (package ``MerlinVerify``); this package records the exact
before/after IR of each Merlin compilation boundary, hands it to the MLIR-parser-based exporter and
the Lean checker, and writes byte-bound receipts. Design: ``docs/design/merlin_lean_verification.md``.
"""
