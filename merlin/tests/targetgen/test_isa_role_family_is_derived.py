"""The structural ISA-ROLE -> semantic-family map must track the role census, not any one target.

A self-hosted-ISA target declares no ``encoding.semantic_class``, so the shared class vocabulary
:func:`semantic_families.from_isa_class` speaks does not reach it and its family vocabulary came back
empty. The bridge is the STRUCTURAL role census: ``isa_introspect._role_for_pattern`` reads an
instruction's own typed operands and answers what datapath it describes, in a closed vocabulary that is
merlin's, not the target's.

These tests exist to keep that bridge honest in the one way it could rot: by quietly becoming a table of
one target's habits. So they assert the mapping is keyed on EXACTLY the role vocabulary the census can
emit — extracted from the census function's own source via ``ast``, never restated here as a literal
list. Add a role to the census without pinning its family and this fails; invent a key that is not a
real role (an overfit) and this fails too.
"""
from __future__ import annotations

import ast

from merlin.common.paths import merlin_dir
from merlin.targetgen import semantic_families as SF

_CENSUS = (merlin_dir() / "python/merlin/targetgen/oracle_helpers/isa_introspect.py")


def _census_roles() -> set[str]:
    """Every string ``_role_for_pattern`` can return, read structurally from its AST.

    Walks the function's ``return`` statements and collects the strings that are actually RETURNED,
    descending through the conditional expressions it uses (``"a" if cond else "b"``) into both arms
    while skipping the test — a string inside the condition (``"exponent" in kinds``) names an operand
    concept, not a role, and counting it would make this test demand a family for a non-role. No regex,
    no hand-copied list: the census source is the source of truth."""
    tree = ast.parse(_CENSUS.read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_role_for_pattern")

    def returned(expr: ast.expr) -> set[str]:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.IfExp):                      # both arms, never the test
            return returned(expr.body) | returned(expr.orelse)
        return set()

    roles: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and node.value is not None:
            roles |= returned(node.value)
    return roles


#: Roles the map leaves unpinned ON PURPOSE, each because pinning it would invent a capability.
#: ``scalar`` is host-side control. The three accumulator/weight roles are contraction PLUMBING: they
#: feed or drain the mesh, so reading them as ``contraction`` would let a target that merely pushes
#: weights claim it can multiply. Same judgement the declared-class map makes by omitting CONFIG.
_PLUMBING = {"scalar", "weight_load", "acc_seed", "acc_readout"}


def test_every_census_role_is_pinned_or_declared_plumbing():
    roles = _census_roles()
    assert roles, "extracted no roles — the census function moved or changed shape"
    unpinned = roles - set(SF.ISA_ROLE_FAMILY) - _PLUMBING
    assert not unpinned, (
        f"role(s) {sorted(unpinned)} come out of the structural census but pin to no semantic family, "
        f"so a target evidencing only those reads as having no capability. Pin them in "
        f"_ISA_ROLE_FAMILY, or add them to _PLUMBING with a reason.")


def test_no_key_is_invented_beyond_the_census():
    """The anti-overfit direction: a key that is not a role the census emits is a target habit."""
    roles = _census_roles()
    invented = set(SF.ISA_ROLE_FAMILY) - roles
    assert not invented, (
        f"_ISA_ROLE_FAMILY keys {sorted(invented)} are not roles the structural census can emit — that "
        f"is a per-target mnemonic leaking into shared code, not a derived role.")


def test_every_role_maps_to_a_declared_family():
    assert not SF.check(), SF.check()
    for role, (fam, requires) in SF.ISA_ROLE_FAMILY.items():
        assert fam in SF.FAMILIES, f"{role!r} -> {fam!r} is not a declared family"
        for req in requires:
            assert req in SF.FAMILIES, f"{role!r} requires undeclared {req!r}"


def test_unknown_and_plumbing_roles_fail_closed():
    """An unrecognized role must return None so callers record UNKNOWN, never a guessed family."""
    assert SF.from_isa_role("scalar") is None
    assert SF.from_isa_role("no_such_role_at_all") is None
    assert SF.from_isa_role(None) is None
    assert SF.from_isa_role("") is None


def test_roles_pin_to_the_family_their_datapath_describes():
    """Spot-check the judgements, stated as datapath facts rather than target facts."""
    assert SF.from_isa_role("matmul") == "contraction"
    # tensor->tensor epilogue is a per-element map, whether one source or two
    assert SF.from_isa_role("tensor_compute_unary") == "elementwise_map"
    assert SF.from_isa_role("tensor_compute_binary") == "elementwise_map"
    assert SF.from_isa_role("memory") == "movement"
    # plumbing licenses NOTHING standalone -- feeding a mesh is not multiplying
    for plumbing in ("weight_load", "acc_seed", "acc_readout"):
        assert SF.from_isa_role(plumbing) is None, plumbing
    # a scaled accumulator pop is a requant on the readout path: a map, but only fused
    assert SF.from_isa_role("acc_readout_scaled") == "elementwise_map"
    assert SF.isa_role_requires("acc_readout_scaled") == ("contraction",)
    assert SF.isa_role_requires("matmul") == ()


def test_reduction_is_not_claimed_from_the_census():
    """A reduce and a per-element map use the same tensor->tensor instructions; the difference is loop
    structure, not encoding. Claiming ``reduction`` from the census would be a guess, so no role may."""
    assert "reduction" not in {fam for fam, _ in SF.ISA_ROLE_FAMILY.values()}


def test_a_conditional_licence_needs_its_required_family_present():
    """``acc_readout_scaled`` alone proves a fused requant exists, NOT a standalone elementwise unit."""
    assert SF.families_from_roles(["acc_readout_scaled"]) == frozenset()
    assert SF.families_from_roles(["acc_readout_scaled", "matmul"]) == frozenset(
        {"contraction", "elementwise_map"})


def test_the_map_is_the_one_the_capability_ladder_uses():
    """One table, not two: a duplicate here already drifted from the ladder inside a single session."""
    from merlin.targetgen import capability_derive as CD
    assert CD._ROLE_FAMILY is SF.ISA_ROLE_FAMILY


def test_families_from_roles_drops_plumbing_and_dedupes():
    fams = SF.families_from_roles(
        ["matmul", "weight_load", "scalar", "tensor_compute_unary", "bogus"])
    assert fams == frozenset({"contraction", "elementwise_map"})
    assert SF.families_from_roles([]) == frozenset()
    assert SF.families_from_roles(None) == frozenset()


def test_no_target_name_appears_in_the_mapping():
    """The cardinal rule, asserted where it would most plausibly be broken."""
    blob = " ".join([*SF.ISA_ROLE_FAMILY,
                     *(f for f, _ in SF.ISA_ROLE_FAMILY.values())]).lower()
    for token in ("atlas", "gemmini", "radiance", "muon", "saturn", "opu", "mxu", "npu"):
        assert token not in blob, f"{token!r} leaked into the shared role->family mapping"
