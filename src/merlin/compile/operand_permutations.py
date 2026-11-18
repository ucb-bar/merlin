"""Offer private operand permutations to consumers that can absorb indexing."""
from .layout_c import permutation_axes
from .operand_layouts import OperandLayout as OperandPermutation
from .operand_layouts import private_operand_layouts


def private_operand_permutations(source, choices):
    """Retain complete-permutation admission for existing consumer integrations."""
    return private_operand_layouts(source, choices, admit=permutation_axes)
