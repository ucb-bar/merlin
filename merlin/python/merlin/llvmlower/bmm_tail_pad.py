"""Tag statically odd batch-matmul tails for feature-scoped transform padding."""
from __future__ import annotations


FEATURE = "accumulator_resident_wholemodel_vf_bmmpad"
TAG = "merlin.bmm_pad_tail"
FULL_TAG = "merlin.bmm_full_tile"


# This source executes in the model2MLIR environment, where torch-mlir owns the
# MLIR Python objects.  Keep the selector structural: provenance is useful for
# reporting but is not required for admitting a contraction.
RUNNER_PRELUDE = r'''
def _tag_odd_batch_matmul_tails(module, ctx, mr=4, nr=8):
    from torch_mlir import ir as _btir

    tails = []
    full = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    if child.operation.name == 'linalg.generic':
                        try:
                            maps = str(child.attributes['indexing_maps']).replace(' ', '')
                            iters = str(child.attributes['iterator_types'])
                            a = _btir.RankedTensorType(child.operands[0].type)
                            b = _btir.RankedTensorType(child.operands[1].type)
                            c = _btir.RankedTensorType(child.operands[2].type)
                        except (KeyError, ValueError, TypeError, IndexError):
                            pass
                        else:
                            bmm_maps = (
                                '(d0,d1,d3)' in maps and
                                '(d0,d3,d2)' in maps and
                                maps.endswith('(d0,d1,d2)>]')
                            )
                            bmm_iters = (
                                iters.count('#linalg.iterator_type<parallel>') == 3 and
                                iters.count('#linalg.iterator_type<reduction>') == 1
                            )
                            ash, bsh, csh = list(a.shape), list(b.shape), list(c.shape)
                            static_bmm = (
                                len(ash) == len(bsh) == len(csh) == 3 and
                                min(*ash, *bsh, *csh) >= 0 and
                                ash[0] == bsh[0] == csh[0] and
                                ash[1] == csh[1] and ash[2] == bsh[1] and
                                bsh[2] == csh[2]
                            )
                            if bmm_maps and bmm_iters and static_bmm:
                                m, n = csh[1], csh[2]
                                (tails if (m % mr or n % nr) else full).append(child)
                    elif child.operation.name == 'linalg.batch_matmul':
                        try:
                            a = _btir.RankedTensorType(child.operands[0].type)
                            b = _btir.RankedTensorType(child.operands[1].type)
                            c = _btir.RankedTensorType(child.operands[2].type)
                        except (ValueError, TypeError, IndexError):
                            pass
                        else:
                            ash, bsh, csh = list(a.shape), list(b.shape), list(c.shape)
                            if (len(ash) == len(bsh) == len(csh) == 3 and
                                    min(*ash, *bsh, *csh) >= 0):
                                m, n = csh[1], csh[2]
                                (tails if (m % mr or n % nr) else full).append(child)
                    walk(child)

    walk(module.operation)
    with ctx, _btir.Location.unknown():
        for op in tails:
            op.attributes['merlin.bmm_pad_tail'] = _btir.UnitAttr.get()
        for op in full:
            op.attributes['merlin.bmm_full_tile'] = _btir.UnitAttr.get()
    print('OK bmm_tail_pad tagged', len(tails), 'full', len(full))
    return len(tails), len(full)
'''
