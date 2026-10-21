"""Bank-separated private full-K operand caching for a dense contraction.

The target owns physical addresses, encodings and completion. This planner
assigns enough rows for one all-K LHS tile and a capacity-bounded RHS column
panel. No overlap or measured service capability is inferred from capacity.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DenseResidentPanel:
    m: int
    k: int
    n: int
    block: int
    padded_k: int
    rhs_base: int
    n_blocks: int
    operand_rows: int
    m_blocks: int = 1

    @property
    def columns(self):
        return self.n_blocks * self.block

    @property
    def rhs_rows(self):
        return self.n_blocks * self.padded_k

    @property
    def row_group(self):
        return self.m_blocks * self.block


def plan_dense_resident_panel(m, k, n, *, block, operand_rows, bank_rows, accumulator_rows):
    """Maximize a complete-K RHS panel after reserving disjoint LHS banks."""
    if any(type(value) is not int or value <= 0 for value in
           (m, k, n, block, operand_rows, bank_rows, accumulator_rows)):
        return None
    if (bank_rows % block or operand_rows % bank_rows or accumulator_rows < block
            or max(m * k, k * n, 4 * m * n) >= 1 << 63):
        return None
    padded_k = (k + block - 1) // block * block
    rhs_base = (padded_k + bank_rows - 1) // bank_rows * bank_rows
    n_blocks = min((n + block - 1) // block, (operand_rows - rhs_base) // padded_k)
    if n_blocks < 1:
        return None
    return DenseResidentPanel(m, k, n, block, padded_k, rhs_base, n_blocks, operand_rows)


def plan_dense_stationary_panel(m, k, n, *, block, operand_rows, bank_rows, accumulator_rows):
    """Balance complete-K row groups and column panels under private capacity.

    Requested DRAM bytes order proposals, followed by stationary-weight fills.
    These are compiler heuristics, not timing observations or fitness claims.
    """
    base = plan_dense_resident_panel(m, k, n, block=block, operand_rows=operand_rows,
                                    bank_rows=bank_rows, accumulator_rows=accumulator_rows)
    if base is None:
        return None
    mt, nt, kt = (m + block - 1) // block, (n + block - 1) // block, (k + block - 1) // block
    candidates = []
    for m_blocks in range(1, min(mt, accumulator_rows // block) + 1):
        rhs_base = (m_blocks * base.padded_k + bank_rows - 1) // bank_rows * bank_rows
        n_blocks = min(nt, (operand_rows - rhs_base) // base.padded_k)
        if n_blocks < 1:
            continue
        column_panels = (nt + n_blocks - 1) // n_blocks
        row_groups = (mt + m_blocks - 1) // m_blocks
        score = m * k * column_panels + k * n, row_groups * nt * kt, -m_blocks
        candidates.append((score, DenseResidentPanel(m, k, n, block, base.padded_k,
                            rhs_base, n_blocks, operand_rows, m_blocks)))
    return min(candidates, key=lambda row: row[0])[1] if candidates else None


def plan_dense_output_panel(m, k, n, *, block, operand_rows, bank_rows, accumulator_rows):
    """Cache full-K operands and distinct full-reduction output columns.

    The target supplies the accumulator capacity and readout mapping. Each
    column block owns one accumulator block until the final readout fence.
    """
    base = plan_dense_resident_panel(m, k, n, block=block, operand_rows=operand_rows,
                                    bank_rows=bank_rows, accumulator_rows=accumulator_rows)
    if base is None:
        return None
    return replace(base, n_blocks=min(base.n_blocks, accumulator_rows // block))


def plan_dense_joint_panel(m, k, n, *, block, operand_rows, bank_rows, accumulator_rows):
    """Reuse row groups and full-K column panels with every output live.

    Total input bytes order candidates, then stationary-weight fills. The
    simultaneous M*N block product must fit private accumulator capacity.
    This heuristic grants no timing or overlap authority.
    """
    base = plan_dense_resident_panel(m, k, n, block=block, operand_rows=operand_rows,
                                    bank_rows=bank_rows, accumulator_rows=accumulator_rows)
    if base is None:
        return None
    mt, nt, kt = (m + block - 1) // block, (n + block - 1) // block, (k + block - 1) // block
    candidates = []
    for m_blocks in range(1, min(mt, accumulator_rows // block) + 1):
        rhs_base = (m_blocks * base.padded_k + bank_rows - 1) // bank_rows * bank_rows
        n_blocks = min(nt, (operand_rows - rhs_base) // base.padded_k,
                       accumulator_rows // (m_blocks * block))
        if n_blocks < 1:
            continue
        column_panels = (nt + n_blocks - 1) // n_blocks
        row_groups = (mt + m_blocks - 1) // m_blocks
        score = m * k * column_panels + k * n, row_groups * nt * kt, -m_blocks
        candidates.append((score, DenseResidentPanel(m, k, n, block, base.padded_k,
                            rhs_base, n_blocks, operand_rows, m_blocks)))
    return min(candidates, key=lambda row: row[0])[1] if candidates else None
