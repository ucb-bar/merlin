"""Shape-only pooling geometry for the portable public interface printer."""
def pool_params(attrs, op='pool'):
    return {k: attrs[k] for k in ('pool_in_dims','pool_size','pool_stride','pool_padding')}

def pool_out_dims(h,w,size,stride,padding):
    if len(padding)==2:
        pt=pb=padding[0]; pl=pr=padding[1]
    else:
        pt,pb,pl,pr=padding
    return (h+pt+pb-size[0])//stride[0]+1, (w+pl+pr-size[1])//stride[1]+1
