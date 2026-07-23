"""The compute-unit-kind registry — the routing axis that keeps core generation target-name-free.

Guards that the registry stays aligned with the canonical KINDS taxonomy, that endpoint defaults are
valid, and that the fork-free `.insn` path is the default wherever a command ISA exists (the
no-forked-toolchain rule) — so a new accelerator of a known kind routes by kind, never by name.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import families as F
from merlin.targetgen.compute_units import KINDS


def test_registry_covers_exactly_the_canonical_kinds():
    # every canonical compute-unit kind has a profile, and no extra invented kinds sneak in
    assert set(F.known_kinds()) == set(KINDS)


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_profile_is_well_formed(kind):
    p = F.family_profile(kind)
    assert p.kind == kind
    assert p.endpoint_kind_default in F.ENDPOINT_KINDS
    assert isinstance(p.encoding_required, bool)
    # a trace gate is only meaningful when an ISA encoding is derived
    if p.trace_gate is not None:
        assert p.encoding_required


def test_command_isa_kinds_default_to_fork_free_insn():
    # systolic + simt expose a command ISA -> default to inline_asm_insn on stock LLVM, no fork
    assert F.family_profile("systolic").endpoint_kind_default == "inline_asm_insn"
    assert F.family_profile("simt").endpoint_kind_default == "inline_asm_insn"
    # systolic derives an op->.insn encoding + runs the rocc_insn trace gate; simt does not
    assert F.family_profile("systolic").encoding_required and F.family_profile("systolic").trace_gate
    assert not F.family_profile("simt").encoding_required and F.family_profile("simt").trace_gate is None


def test_unknown_kind_fails_closed():
    with pytest.raises(KeyError):
        F.family_profile("wat")
