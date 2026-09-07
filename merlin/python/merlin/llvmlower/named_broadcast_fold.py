"""Remove broadcasts whose sole consumer is a named ``linalg.add`` or ``linalg.mul``.

The stage is deliberately placed after contraction scheduling and the existing generic
producer/consumer fusion, immediately before residual named-op generalization.  At that seam the
LSTMNetVIT all17's uninstrumented tagged input has exactly 62 such broadcasts: 52 feeding named
adds and 10 feeding named multiplies.  The existing post-contraction fusion/canonicalization removes
23 before this seam, leaving 31 add and 8 multiply chains for this rewrite.  It marks only those
producers, runs the pipeline's normal named-op generalization, and then composes each generalized
broadcast's source map into its consumer.  It does not run blanket post-generalization fusion,
which also absorbs unrelated producers.
"""
from __future__ import annotations

from pathlib import Path


FEATURE = "targeted_named_broadcast_fold"
MARKER = "__merlin_targeted_named_broadcast_fold__"


RUNNER_PRELUDE = r'''
def _fold_targeted_generalized_broadcasts(module, ctx, target_attr):
    """Fold only generic broadcasts carrying ``target_attr`` into parallel generics."""
    from torch_mlir.ir import AffineMap, AffineMapAttr, ArrayAttr

    producers = [op for op in _bf_walk(module.operation)
                 if op.name == 'linalg.generic' and target_attr in op.attributes]
    folded = 0
    report = []
    for producer in producers:
        shape = str(producer.results[0].type) if producer.results else '<no result>'
        # Stock named-op generalization turns a tensor broadcast into one source, one init, one
        # result and two maps.  The marker proves its origin, but keep the full structural check so
        # an upstream representation change fails closed.
        maps = _bf_maps(producer)
        if (len(producer.operands) != 2 or len(producer.results) != 1
                or maps is None or len(maps) != 2 or not _bf_all_parallel(producer)):
            report.append(('skip', shape + ': marked producer is not a generalized broadcast'))
            continue
        src, init = producer.operands
        result = producer.results[0]
        uses = list(result.uses)
        if len(uses) != 1:
            report.append(('skip', shape + ': generalized broadcast has %d readers' % len(uses)))
            continue
        use = uses[0]
        consumer = use.owner
        operand_index = int(use.operand_number)
        consumer_maps = _bf_maps(consumer)
        if (consumer.name != 'linalg.generic' or not _bf_all_parallel(consumer)
                or consumer_maps is None):
            report.append(('skip', shape + ': generalized consumer is not all-parallel generic'))
            continue
        n_outs = len(consumer.results)
        n_inputs = len(consumer.operands) - n_outs
        if n_outs != 1 or operand_index >= n_inputs:
            report.append(('skip', shape + ': generalized broadcast is not a consumer input'))
            continue
        source_map = maps[0].value
        output_map = maps[1].value
        consumer_map = consumer_maps[operand_index].value
        src_rank = _bf_rank(src)
        out_rank = _bf_rank(result)
        if (src_rank is None or out_rank is None
                or len(source_map.results) != src_rank
                or len(output_map.results) != out_rank
                or not source_map.is_projected_permutation
                or not output_map.is_permutation
                or len(consumer_map.results) != source_map.n_dims
                or source_map.n_symbols != 0 or consumer_map.n_symbols != 0):
            report.append(('skip', shape + ': generalized maps do not prove broadcast semantics'))
            continue
        try:
            if ir.ShapedType(src.type).element_type != ir.ShapedType(result.type).element_type:
                report.append(('skip', shape + ': source/result element types disagree'))
                continue
        except Exception:
            report.append(('skip', shape + ': source/result is not shaped'))
            continue

        # source_map maps producer points to source indices; consumer_map maps consumer points to
        # producer-result indices. Compose them to address the source directly.
        with ctx:
            composed = [expr.compose(consumer_map) for expr in source_map.results]
            entries = [consumer_maps[index] for index in range(len(consumer_maps))]
            entries[operand_index] = AffineMapAttr.get(
                AffineMap.get(consumer_map.n_dims, 0, composed))
            consumer.attributes['indexing_maps'] = ArrayAttr.get(entries)
            consumer.operands[operand_index] = src

        init_owner = init.owner
        producer.erase()
        if (init_owner is not None and init_owner.name == 'tensor.empty'
                and all(not list(value.uses) for value in init_owner.results)):
            init_owner.erase()
        folded += 1
        report.append(('fold', shape + ' operand %d -> rank-%d projection'
                       % (operand_index, src_rank)))
    return folded, report


def _targeted_named_broadcast_fold(ctx, module):
    """Generalize and fold only sole-use broadcasts feeding named add/mul operations."""
    target_attr = 'merlin.targeted_named_broadcast_fold'
    targets = []
    counts = {'linalg.add': 0, 'linalg.mul': 0}
    for broadcast in _bf_walk(module.operation):
        if broadcast.name != 'linalg.broadcast' or len(broadcast.results) != 1:
            continue
        uses = list(broadcast.results[0].uses)
        if len(uses) != 1 or uses[0].owner.name not in counts:
            continue
        # Stock generalization preserves this discardable attribute when rebuilding the producer,
        # giving the post-pass rewrite a mechanically exact provenance token.
        with ctx:
            broadcast.attributes[target_attr] = ir.UnitAttr.get()
        targets.append(broadcast)
        counts[uses[0].owner.name] += 1

    print('CENSUS targeted_named_broadcast_fold add', counts['linalg.add'],
          'mul', counts['linalg.mul'])
    if not targets:
        return 0

    # This is the same generalization the surrounding pipeline is about to run. Contractions have
    # already been scheduled and lowered before this stage; its later invocation is a no-op.
    PassManager.parse('builtin.module(func.func(linalg-generalize-named-ops))', ctx).run(
        module.operation)
    folded, report = _fold_targeted_generalized_broadcasts(module, ctx, target_attr)
    if folded != len(targets):
        details = '; '.join(kind + ': ' + detail for kind, detail in report)
        raise RuntimeError('targeted named broadcast fold planned %d but folded %d: %s'
                           % (len(targets), folded, details))
    return folded


_TARGETED_NAMED_BROADCAST_FOLD = len(sys.argv) > 15 and sys.argv[15] == '1'
_PRE_GENERALIZE_STAGES = ([('targeted_named_broadcast_fold',
                            _targeted_named_broadcast_fold)]
                          if _TARGETED_NAMED_BROADCAST_FOLD else [])
'''


def run_source() -> str:
    """Standalone driver executing the exact stage implementation used by lowering."""
    from .broadcast_fold import RUNNER_PRELUDE as _BROADCAST_PRELUDE
    return (
        "import sys\n"
        "from torch_mlir import ir\n"
        "from torch_mlir.passmanager import PassManager\n"
        + _BROADCAST_PRELUDE
        + RUNNER_PRELUDE
        + "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "n = _targeted_named_broadcast_fold(ctx, module)\n"
        "module.operation.verify()\n"
        "print('FOLDED', n)\n"
        "with open(sys.argv[2], 'w') as f:\n"
        "    f.write(str(module.operation))\n"
    )


def _edit_pipeline(passes: list[str]) -> list[str]:
    anchor = "func.func(linalg-generalize-named-ops)"
    if MARKER in passes:
        return list(passes)
    if anchor not in passes:
        raise ValueError(
            f"{FEATURE} requires {anchor!r}; without it the after-schedule/pre-generalize seam "
            "cannot be established")
    at = passes.index(anchor)
    return [*passes[:at], MARKER, *passes[at:]]


def _feature():
    from .impr_features import ImprFeature
    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "after contraction scheduling and generic fusion, fold only sole-use broadcasts whose "
            "remaining named consumer is linalg.add or linalg.mul. The uninstrumented all17 "
            "LSTMNetVIT input has 52 add and 10 mul chains; existing fusion removes 23 first, so "
            "the exact runner receipt is 31 add and 8 mul. Shared uses, generic reductions and "
            "contractions are outside the match. Accepted on full-output-gated K1 W8A8: launch "
            "medians 135.208 to 130.880 ms at one core (1.033x) and 51.005 to 49.640 ms at eight "
            "cores (1.027x); every launch retained exact SHA adf8308a. Default-off; requires a "
            "nonzero runner receipt."),
        edit_pipeline=_edit_pipeline,
    )


def ensure_registered() -> str:
    from .impr_features import known, register
    if FEATURE not in known():
        register(_feature())
    return FEATURE


_CENSUS_PREFIX = "CENSUS targeted_named_broadcast_fold add "
_FOLDED_PREFIX = "OK targeted_named_broadcast_fold "


def _exact_census_reports(stdout: str) -> list[tuple[int, int]]:
    """Read exact ``add N mul M`` census lines as tokens, never as a regex."""
    reports: list[tuple[int, int]] = []
    for line in stdout.splitlines():
        if not line.startswith(_CENSUS_PREFIX):
            continue
        fields = line[len(_CENSUS_PREFIX):].split(" ")
        if (len(fields) == 3 and fields[1] == "mul"
                and fields[0].isdecimal() and fields[2].isdecimal()):
            reports.append((int(fields[0]), int(fields[2])))
    return reports


def _exact_folded_reports(stdout: str) -> list[int]:
    reports: list[int] = []
    for line in stdout.splitlines():
        if not line.startswith(_FOLDED_PREFIX):
            continue
        value = line[len(_FOLDED_PREFIX):]
        if value and value.isdecimal():
            reports.append(int(value))
    return reports


def require_report(stdout: str, work: str | Path) -> dict[str, int]:
    census = _exact_census_reports(stdout)
    folded = _exact_folded_reports(stdout)
    if (len(census) != 1 or len(folded) != 1 or folded[0] < 1
            or folded[0] != sum(census[0])):
        raise ValueError(
            "targeted_named_broadcast_fold receipt missing or inconsistent: "
            f"census={census}, folded={folded}")
    report = {"add": census[0][0], "mul": census[0][1], "folded": folded[0]}
    Path(work, "named_broadcast_fold_report.txt").write_text(
        " ".join(f"{key}={value}" for key, value in report.items()) + "\n", encoding="utf-8")
    return report


ensure_registered()
