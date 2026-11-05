"""Column partitions retain all source coefficients and exact rounding domains."""
import numpy as np
import pytest

from merlin.llvmlower.integer_readout import evaluate, partition_column_scales


def test_complete_nonadjacent_regions_and_rounding_boundaries():
    scales=[.5]*17+[.125]*15+[.75,.5]
    result=partition_column_scales(scales,len(scales),relu=False)
    assert [(r['first_column'],r['columns'])for r in result['regions']]==[(0,17),(17,15),(32,1),(33,1)]
    consumed=[]
    for region in result['regions']:
        proof=region['readout_proof']['source']
        points={-(1<<31),(1<<31)-1,0}
        for t in proof['thresholds']:
            points.update(x for x in (t-1,t,t+1)if -(1<<31)<=x<1<<31)
        values=np.array(sorted(points),dtype=np.int64)
        expected=np.clip(np.rint(values.astype(np.float32)*np.float32(region['scale'])),-128,127).astype(np.int32)
        assert [evaluate(int(x),proof)for x in values]==expected.tolist()
        consumed.extend([region['scale']]*region['columns'])
    assert consumed==scales


@pytest.mark.parametrize('scales,columns,budget',[
    ([.5],2,128),([.5,0],2,128),([.5,-.5],2,128),([.5,float('nan')],2,128),
    ([True],1,128),([.5,.125,.5],3,2),([.5],1,0)])
def test_invalid_or_unproved_columns_never_truncate_or_approximate(scales,columns,budget):
    with pytest.raises(ValueError):partition_column_scales(scales,columns,max_partitions=budget)
