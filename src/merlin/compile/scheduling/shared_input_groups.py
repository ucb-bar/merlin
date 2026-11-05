"""Recognize closed sibling contractions of one immutable typed tensor value.

These source proposals preserve every operation and each independent result.
They establish no physical aliases, target instruction effects or timing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects import builtin, func

from .zero_initializer_groups import find_zero_initializer_groups


@dataclass(frozen=True)
class SharedInputGroup:
    source_ir_sha256: str
    source_op_indices: tuple[int, ...]
    contraction_ordinals: tuple[int, ...]
    m: int
    k: int
    widths: tuple[int, ...]

    def record(self):
        return {
            "schema": "merlin.shared_input_group.v1",
            **asdict(self),
            "source_semantics": "immutable tensor SSA; separate zero-seeded signed modular contractions",
            "result_boundary": "every original contraction result and later epilogue remains independent",
            "authority": "source proposal only; physical ownership and actual target effects remain required",
        }

    def verify_current(self, module, *, max_source_operations=10_000, max_source_elements=64 << 20):
        groups = find_shared_input_groups(
            module, max_source_operations=max_source_operations, max_source_elements=max_source_elements)
        if not any(self == prefix for group in groups for prefix in grow_shared_input_group(group)):
            raise ValueError("shared-input group differs from complete current typed source")


def grow_shared_input_group(group):
    """Yield consecutive prefixes, adding one sibling at a time; never subsets.

    Each prefix retains its own exact zero initializers and output identities.
    Unselected siblings and external consumers stay in the original module.
    ``verify_current`` rederives the maximal group before accepting a prefix.
    """
    from dataclasses import replace

    for count in range(2, len(group.widths) + 1):
        last = group.contraction_ordinals[count - 1]
        yield replace(group, source_op_indices=tuple(i for i in group.source_op_indices if i <= last),
                      contraction_ordinals=group.contraction_ordinals[:count], widths=group.widths[:count])


def find_shared_input_groups(module, *, max_source_operations=10_000, max_source_elements=64 << 20):
    """Group adjacent checked zero-seed regions with the identical LHS SSA value.

    No operation moves across unknown effects or an intervening consumer. The
    existing strict source checker establishes signed products, increasing K,
    ordinary tensor layouts and one-use exact zero initializers. Only source
    zero/initializer/contraction operations may occur within a group. Grouping
    does not erase them or change any use list, including later epilogues.
    """
    seeds = find_zero_initializer_groups(
        module, max_source_operations=max_source_operations, max_source_elements=max_source_elements
    )
    if not seeds:
        return ()
    # The original checker already validates the single-function/block shell.
    function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
    operations = tuple(function.body.block.ops)
    groups, run, covered, shared = [], [], set(), None

    def finish():
        if len(run) < 2:
            return
        first = run[0]
        groups.append(
            SharedInputGroup(
                first.source_ir_sha256,
                tuple(sorted(covered)),
                tuple(seed.contraction_ordinal for seed in run),
                first.geometry_m_k_n[0],
                first.geometry_m_k_n[1],
                tuple(seed.geometry_m_k_n[2] for seed in run),
            )
        )

    for seed in seeds:
        lhs = operations[seed.contraction_ordinal].operands[0]
        # The source recognizer requires immutable, unencoded tensor SSA. A
        # repeated name or equal shape never substitutes for value identity.
        assert isinstance(lhs.type, builtin.TensorType)
        new = set(seed.source_op_indices)
        contiguous = not run or min(new) == max(covered) + 1
        if run and (lhs is not shared or not contiguous):
            finish()
            run, covered = [], set()
        shared = lhs
        run.append(seed)
        covered.update(new)
    finish()
    return tuple(groups)
