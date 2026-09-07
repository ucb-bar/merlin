"""Fold a sole-use ``linalg.broadcast`` into an all-parallel generic's indexing map.

This is intentionally narrower than running upstream elementwise fusion after named-op
generalization.  That ordering was measured to absorb dequantization into contractions and slow
the K1 path.  Here the consumer must already be an all-parallel ``linalg.generic`` and the only
change is affine-map composition: a materialized broadcast becomes a projected operand access.
"""
from __future__ import annotations

from pathlib import Path


FEATURE = "fold_broadcast_into_generic"


# Spliced into each m2m lowering runner.  Keep this self-contained: it executes in the model2MLIR
# venv, not Merlin's Python environment.
RUNNER_PRELUDE = r'''
def _bf_walk(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in list(block.operations):
                yield inner.operation
                yield from _bf_walk(inner.operation)


def _bf_rank(value):
    try:
        shaped = ir.ShapedType(value.type)
        return shaped.rank if shaped.has_rank else None
    except Exception:
        return None


def _bf_maps(op):
    try:
        maps = op.attributes['indexing_maps']
        if len(maps) != len(op.operands):
            return None
        return maps
    except (KeyError, TypeError, ValueError):
        return None


def _bf_all_parallel(op):
    try:
        iters = op.attributes['iterator_types']
    except KeyError:
        return False
    return bool(iters) and all('parallel' in str(iterator) for iterator in iters)


def _fold_broadcasts(module, ctx, required_attr=None):
    """Return ``(folded, report)`` after applying only mechanically proven map folds."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    broadcasts = [op for op in _bf_walk(module.operation) if op.name == 'linalg.broadcast']
    folded = 0
    report = []
    for broadcast in broadcasts:
        if required_attr is not None and required_attr not in broadcast.attributes:
            continue
        shape = str(broadcast.results[0].type) if broadcast.results else '<no result>'
        if len(broadcast.operands) != 2 or len(broadcast.results) != 1:
            report.append(('skip', shape + ': expected one input, one init and one result'))
            continue
        src, init = broadcast.operands
        result = broadcast.results[0]
        uses = list(result.uses)
        if len(uses) != 1:
            report.append(('skip', shape + ': broadcast has %d readers, not one' % len(uses)))
            continue
        use = uses[0]
        consumer = use.owner
        operand_index = int(use.operand_number)
        if consumer.name != 'linalg.generic':
            report.append(('skip', shape + ': sole reader is %s, not linalg.generic'
                           % consumer.name))
            continue
        # This guard is the boundary from the refuted blanket fusion lever: a contraction carries a
        # reduction iterator.  Do not change contraction input maps or its scheduled access pattern.
        if not _bf_all_parallel(consumer):
            report.append(('skip', shape + ': consumer has a non-parallel iterator'))
            continue
        maps = _bf_maps(consumer)
        if maps is None:
            report.append(('skip', shape + ': consumer states no map for every operand'))
            continue
        n_outs = len(consumer.results)
        n_inputs = len(consumer.operands) - n_outs
        if n_outs < 1 or operand_index >= n_inputs:
            report.append(('skip', shape + ': broadcast is not a tensor input of its consumer'))
            continue
        try:
            dimensions = [int(dim) for dim in broadcast.attributes['dimensions']]
        except (KeyError, TypeError, ValueError):
            report.append(('skip', shape + ': dimensions are not a static integer list'))
            continue
        src_rank = _bf_rank(src)
        out_rank = _bf_rank(result)
        if src_rank is None or out_rank is None:
            report.append(('skip', shape + ': source/result is unranked'))
            continue
        if (dimensions != sorted(set(dimensions))
                or any(dim < 0 or dim >= out_rank for dim in dimensions)
                or src_rank + len(dimensions) != out_rank):
            report.append(('skip', shape + ': dimensions do not define a rank projection'))
            continue
        old_map = maps[operand_index].value
        old_results = list(old_map.results)
        if len(old_results) != out_rank:
            report.append(('skip', shape + ': consumer map/result rank disagree'))
            continue
        keep = [position for position in range(out_rank) if position not in dimensions]
        new_results = [old_results[position] for position in keep]

        # Changing the shaped operand must not change the scalar type of the corresponding generic
        # block argument.  The linalg verifier checks this too, but refusing before mutation keeps
        # the transformation atomic and makes the reason visible.
        try:
            if ir.ShapedType(src.type).element_type != ir.ShapedType(result.type).element_type:
                report.append(('skip', shape + ': source/result element types disagree'))
                continue
        except Exception:
            report.append(('skip', shape + ': source/result is not shaped'))
            continue

        with ctx:
            entries = [maps[index] for index in range(len(maps))]
            entries[operand_index] = AffineMapAttr.get(
                AffineMap.get(old_map.n_dims, old_map.n_symbols, new_results))
            consumer.attributes['indexing_maps'] = ArrayAttr.get(entries)
            consumer.operands[operand_index] = src

        # Result is now dead by construction.  Its destination is normally a tensor.empty used only
        # by the broadcast; erase it too, otherwise bufferization can still leave a pointless alloc.
        init_owner = init.owner
        broadcast.erase()
        if (init_owner is not None and init_owner.name == 'tensor.empty'
                and all(not list(value.uses) for value in init_owner.results)):
            init_owner.erase()
        folded += 1
        report.append(('fold', '%s operand %d -> rank-%d projection'
                       % (shape, operand_index, src_rank)))
    return folded, report


_FOLD_BROADCAST = len(sys.argv) > 14 and sys.argv[14] == '1'
'''


def run_source() -> str:
    """Standalone driver used by tests; it executes the exact runner rewrite."""
    return (
        "import sys\n"
        "from torch_mlir import ir\n"
        + RUNNER_PRELUDE
        + "src_path, out_path = sys.argv[1], sys.argv[2]\n"
        "ctx = ir.Context()\n"
        "with open(src_path) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "n, report = _fold_broadcasts(module, ctx)\n"
        "for kind, detail in report:\n"
        "    print(kind.upper(), detail)\n"
        "print('FOLDED', n)\n"
        "with open(out_path, 'w') as f:\n"
        "    f.write(str(module.operation))\n"
    )


def _feature():
    from .impr_features import ImprFeature
    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "fold a sole-use linalg.broadcast into an already-all-parallel linalg.generic by "
            "projecting the broadcast dimensions out of that input's indexing map. The corrected "
            "post-all17 LSTMNetVIT K1 profile attributes 4.863 ms (161 of 223 broadcasts) to this "
            "exact structurally eligible set, but that attribution is profiler-created: the marker "
            "calls inhibit the normal post-contraction fusion stage. The runner demonstrably folds "
            "all 161, yet its uninstrumented model.ll and model.o are byte-identical to the all17 "
            "control because the normal pipeline already reaches the same IR. This is a diagnostic "
            "negative lever, not a board candidate. Reduction consumers are refused, keeping "
            "contraction accesses outside the rewrite after blanket post-generalization fusion "
            "measured slower. Default-off and runner-gated; feature-off lowering is byte-identical."
        ),
    )


def ensure_registered() -> str:
    from .impr_features import known, register
    if FEATURE not in known():
        register(_feature())
    return FEATURE


_REPORT_PREFIX = "OK fold_broadcast_into_generic folded "


def _exact_nonnegative_int_reports(stdout: str, prefix: str) -> list[int]:
    """Read exact, one-integer receipt lines without treating output as a pattern language."""
    values: list[int] = []
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            continue
        value = line[len(prefix):]
        if value and value.isdecimal():
            values.append(int(value))
    return values


def require_report(stdout: str, work: str | Path) -> int:
    """Require proof that the requested runner rewrite executed and found work.

    A feature name reaching package metadata but not the runner gate otherwise produces a valid
    baseline binary and masquerades as an optimization result.  Persist the receipt next to the
    lowered IR so an artifact remains auditable after the subprocess output is gone.
    """
    matches = _exact_nonnegative_int_reports(stdout, _REPORT_PREFIX)
    if len(matches) != 1:
        raise ValueError(
            "fold_broadcast_into_generic was requested but the lowering runner did not report "
            f"exactly once (reports={matches})")
    if matches[0] < 1:
        raise ValueError(
            "fold_broadcast_into_generic was requested but folded zero broadcasts")
    path = Path(work) / "broadcast_fold_report.txt"
    path.write_text(f"folded={matches[0]}\n", encoding="utf-8")
    return matches[0]


# Direct imports (tests and controlled build scripts) should see a resolvable feature name.  The
# lowering subprocess also calls ensure_registered explicitly before normalizing its feature set.
ensure_registered()
