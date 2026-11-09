"""Capacity planning for a private dense product with a complete resident RHS.

The target supplies its row geometry and a disjoint LHS staging reservation.
This plan supplies no synchronization, instruction semantics or overlap claim.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseRhsResidency:
    m: int
    k: int
    n: int
    block: int
    rhs_base: int
    k_tiles: int
    n_tiles: int
    operand_rows: int

    @property
    def padded_k(self):
        return self.k_tiles * self.block

    @property
    def rhs_rows(self):
        return self.k_tiles * self.n_tiles * self.block


def plan_dense_rhs_residency(m, k, n, *, block, operand_rows, accumulator_rows, rhs_base,
                             max_static_rhs_tiles=1024):
    """Plan only disjoint full-RHS storage and one complete output accumulator.

    Decline before emission if the padded matrix, bounded static preparation or
    private LHS/output tile does not fit. The caller must own the complete local
    storage epoch and retain ordinary external-buffer alias/lifetime checks.
    """
    values = (m, k, n, block, operand_rows, accumulator_rows, rhs_base, max_static_rhs_tiles)
    if any(type(value) is not int or value <= 0 for value in values):
        return None
    if rhs_base < block or rhs_base % block or accumulator_rows < block:
        return None
    kt, nt = (k + block - 1) // block, (n + block - 1) // block
    if (kt * nt > max_static_rhs_tiles or rhs_base + kt * nt * block > operand_rows
            or max(m * k, k * n, 4 * m * n) >= 1 << 63):
        return None
    return DenseRhsResidency(m, k, n, block, rhs_base, kt, nt, operand_rows)
