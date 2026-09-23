# MX Gemmini Phase 1 inputs

This directory owns public target-specific experiment inputs, not reusable compiler
implementation. The descriptor explicitly selects the contract directory and ISA header.
Keep the three ISA/ABI files together: the ISA definition imports its sibling patterns.
Preserve source provenance and recorded unknowns; never invent hardware facts.
Generated prompts, bundles, capsules, compilers and certification records are artifacts.
Private goldens and holdouts must not be copied here.
