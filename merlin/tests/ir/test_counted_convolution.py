"""Counted convolution partitions execute original windows, tails and reductions."""
from dataclasses import replace
import operator
import numpy as np
import pytest
from merlin.compile.scheduling.window_transfer import Window2D,CountedConvolutionPlan,emit_counted_convolution


class Integers:
    def op(self,name,a,b,dtype="i64"):
        return {"add":operator.add,"sub":operator.sub,"mul":operator.mul,
                "udiv":operator.floordiv,"urem":operator.mod,"and":operator.and_}[name](a,b)
    def cmp(self,name,a,b):
        # The actual index builder uses unsigned i64 comparisons.
        a%=1<<64;b%=1<<64
        return {"ult":operator.lt,"ule":operator.le,"eq":operator.eq}[name](a,b)
    def loop(self,start,stop,step,body):
        for index in range(start,stop,step):body(index)
    def branch(self,predicate,yes,no):(yes if predicate else no)()


@pytest.mark.parametrize("shape,kernel,stride,padding,dilation",[
    ((1,3,5,3),(2,2),(1,1),(0,0,0,0),(1,1)),
    ((2,4,5,5),(3,2),(2,1),(1,1,1,0),(1,1)),
    ((1,2,3,1),(3,3),(1,1),(2,2,1,1),(1,1)),
    ((1,5,6,3),(2,2),(1,2),(1,0,0,1),(2,2)),
])
@pytest.mark.parametrize("biased",[False,True])
def test_full_source_numerics_and_one_store_per_output(shape,kernel,stride,padding,dilation,biased):
    window=Window2D(shape,kernel,stride,dilation,padding);oh,ow=window.output_spatial
    plan=CountedConvolutionPlan(window,5,4,min(4,ow),4,4,32,8,0,8,0,biased)
    image=(np.arange(np.prod(shape))%17-8).astype(np.int8).reshape(shape)
    weights=(np.arange(np.prod(kernel)*shape[-1]*5)%13-6).astype(np.int8).reshape(*kernel,shape[-1],5)
    bias=np.array([-(1<<31),(1<<31)-1,-9,0,7],dtype=np.int32)
    output=np.zeros((shape[0]*oh*ow,5),dtype=np.int32);written=np.zeros_like(output)
    lhs=np.zeros((plan.row_tile,plan.channel_tile),dtype=np.int8);rhs=None;acc=None
    events={"loads":0,"zero":0,"compute":0,"store":0}
    def lower(kind,a):
        nonlocal rhs,acc
        if kind=="initialize_bias":acc=np.broadcast_to(bias[a["output_column"]:a["output_column"]+a["columns"]],(a["rows"],a["columns"])).copy()
        elif kind=="load_input":
            events["loads"]+=1;events["zero"]+=int(a["zero"])
            for row in range(a["rows"]):
                first=a["first_element"]+row*a["row_stride_elements"]
                lhs[a["local_row"]+row,:a["columns"]]=0 if a["zero"]else image.ravel()[first:first+a["columns"]]
        elif kind=="load_weight":
            assert a["local_row"]==plan.rhs_row
            rhs=weights.reshape(-1,5)[a["reduction_row"]:a["reduction_row"]+a["rows"],a["output_column"]:a["output_column"]+a["columns"]]
        elif kind=="compute":
            events["compute"]+=1
            product=lhs[:a["rows"],:a["depth"]].astype(np.int32)@rhs.astype(np.int32)
            acc=product if a["first_reduction"]and not a["bias"]else acc+product
        elif kind=="store":
            events["store"]+=1;r,c=a["output_row"],a["output_column"]
            output[r:r+a["rows"],c:c+a["columns"]]=acc;written[r:r+a["rows"],c:c+a["columns"]]+=1
        else:raise AssertionError(kind)
    receipt=emit_counted_convolution(plan,Integers(),lower,synchronous_completion=True)
    expected=np.zeros_like(output);pt,pl,_,_=padding;kh,kw=kernel
    for b in range(shape[0]):
        for y in range(oh):
            for x in range(ow):
                value=bias.astype(np.int64).copy()if biased else np.zeros(5,dtype=np.int64)
                for ky in range(kh):
                    for kx in range(kw):
                        iy=y*stride[0]+ky*dilation[0]-pt;ix=x*stride[1]+kx*dilation[1]-pl
                        if 0<=iy<shape[1]and 0<=ix<shape[2]:value+=image[b,iy,ix].astype(np.int64)@weights[ky,kx].astype(np.int64)
                expected[(b*oh+y)*ow+x]=value.astype(np.int32)
    assert np.array_equal(output,expected)and np.all(written==1)
    assert receipt["static_tile_families"]<=8 and receipt["host_payload_operations"]==0
    if any(padding):assert events["zero"]>0


@pytest.mark.parametrize("change",[dict(lhs_row=7),dict(accumulator_rows=1),dict(row_tile=9),dict(channel_tile=5),dict(bias=1)])
def test_bad_lifetimes_and_geometry_refuse(change):
    p=CountedConvolutionPlan(Window2D((1,4,5,3),(2,2)),5,4,4,4,3,32,8,0,8,0)
    with pytest.raises(ValueError):replace(p,**change)


def test_completion_and_mutation_refuse():
    p=CountedConvolutionPlan(Window2D((1,2,2,1),(1,1)),1,2,2,1,1,8,4,0,4,0)
    with pytest.raises(ValueError,match="synchronous"):
        emit_counted_convolution(p,Integers(),lambda *_:None,synchronous_completion=False)
    def mutate(*_):object.__setattr__(p,"output_channels",2)
    with pytest.raises(ValueError,match="changed"):
        emit_counted_convolution(p,Integers(),mutate,synchronous_completion=True)


def test_large_shape_has_bounded_static_families():
    p=CountedConvolutionPlan(Window2D((4,56,56,64),(3,3),padding=(1,1,1,1)),64,16,16,16,16,16384,1024,0,16,0)
    assert p.coverage["static_tile_families"]==2
    assert p.coverage["reduction_products"]==4*56*56*64*3*3*64
