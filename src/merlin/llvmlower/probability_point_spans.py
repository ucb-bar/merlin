"""Specialize the owned attention producer after exact probability refinement.

This is not an admission for arbitrary C or caller-supplied point flags. The
complete owned producer/store/gather/evaluator skeleton must remain intact.
Source matching, immutable workspace ownership and retained fallback are still
proved by the normal provider binding. Every successful probability store is a
finite, equal BF16-bin snapshot; exact replay refreshes that snapshot first.
"""


def prepare_probability_point_spans(text: str) -> str:
    """Share one immutable point span across PV value/lower/upper roles."""
    store = "pl[off+j]=bins.lo;ph[off+j]=bins.hi;p[off+j]=merlin_bf16_interval_midpoint(y,bins);if(merlin_interval_bits(pl[off+j])!=merlin_interval_bits(ph[off+j]))counts[1]++;"
    replay = "y=merlin_interval_point(source_poly(exact*SCORE_SCALE-mx)); bins=merlin_bf16_interval_prepare(y); counts[2]++;"
    if text.count(store) != 2 or text.count(replay) != 2:
        raise ValueError("complete probability replay/store grammar required")
    required = (
        "if(mask[off+j] && bins.low_bits != bins.high_bits) {",
        "if(!(lo[off+j]<=exact && exact<=hi[off+j]))return 0;",
    )
    if any(text.count(x) != 2 for x in required):
        raise ValueError("source score replay enclosure changed")
    # Refuse unknown mutations between the complete probability producer and
    # its six synchronous product consumers. Template shape/loop bounds are
    # source-derived macros, never numerical samples or workload selectors.
    region = """  for(int tile=0;tile<2;tile++)for(int part=0;part<PARTS;part++){
   int begin=tile*CHUNK+part*SEGMENT,length=CHUNK-part*SEGMENT;if(length>SEGMENT)length=SEGMENT;
   for(int r=0;r<ROWS;r++)for(int z=0;z<length;z++){
    int ix=r*KEYS+begin+z;w->a[r*length+z]=h->p[ix];w->al[r*length+z]=h->plo[ix];w->ah[r*length+z]=h->phi[ix];
   }
   for(int d=0;d<DEPTH;d++)for(int z=0;z<length;z++)w->b[d*length+z]=h->v[(begin+z)*DEPTH+d];
   if(!evaluate_products(w,ROWS,DEPTH,length,product,opaque))return 0;
   for(int i=0;i<ROWS*DEPTH;i++){
    int ix=(tile*PARTS+part)*ROWS*DEPTH+i;h->partlo[ix]=w->lower[i];h->parthi[ix]=w->upper[i];h->centers[ix]=(float)w->center[i];
   }
  }"""
    marker = "if(!soft_details(h->q,h->k,h->mask,h->qlo,h->qhi,h->p,h->plo,h->phi,h->denlo,h->denhi,h->alpha,ROWS,w->softcounts,h->ylo,h->yhi,h->maxima,&spans,&span_epoch))return 0;"
    if text.count(region) != 1 or text.count(marker + "\n" + region) != 1:
        raise ValueError("probability point producer/consumer lifetime changed")
    qcopy = "memcpy(w->a,h->q,sizeof(h->q));memcpy(w->al,h->q,sizeof(h->q));memcpy(w->ah,h->q,sizeof(h->q));"
    evaluator = "static int evaluate_products(struct attention_workspace *w,int m,int n,int k,merlin_attention_product product,void *opaque){"
    encode = "w->astep,w->al,w->ah,w->encoded_a_exact,&aproof)"
    bounds = "dot_bounds(w->a,w->al,w->ah,w->b,w->ar,w->br,w->center,w->lower,w->upper,m,n,k,&w->norms,&aproof,&bproof)"
    if any(text.count(x) != 1 for x in (qcopy, evaluator, encode, bounds)):
        raise ValueError("point input encoder/bound ownership changed")
    # There are exactly two syntactic product sites (QK and PV). Both use a
    # fresh local epoch; the witness cannot escape the synchronous call.
    qcall = "if(!evaluate_products(w,ROWS,CHUNK,DEPTH,product,opaque))return 0;"
    if text.count(qcall) != 1 or text.count("if(!evaluate_products(") != 2:
        raise ValueError("unknown product site or interval input")
    point_store = "merlin_bf16_exact_point point=merlin_bf16_exact_point_finish(y,bins);if(!point.valid)return 0;p[off+j]=point.value;"
    text = text.replace(store, point_store)
    selected = region.replace(
        "w->a[r*length+z]=h->p[ix];w->al[r*length+z]=h->plo[ix];w->ah[r*length+z]=h->phi[ix];",
        "w->a[r*length+z]=h->p[ix];",
    ).replace(
        "   if(!evaluate_products(w,ROWS,DEPTH,length,product,opaque))return 0;",
        "   unsigned char product_epoch;\n"
        "   merlin_source_point_span points={w->a,(size_t)ROWS*length,&product_epoch};\n"
        "   if(!evaluate_products(w,ROWS,DEPTH,length,product,opaque,&points,&product_epoch))return 0;",
    )
    text = text.replace(region, selected).replace(qcopy, "memcpy(w->a,h->q,sizeof(h->q));")
    text = text.replace(
        qcall,
        "unsigned char product_epoch;\n"
        "   merlin_source_point_span points={w->a,(size_t)ROWS*DEPTH,&product_epoch};\n"
        "   if(!evaluate_products(w,ROWS,CHUNK,DEPTH,product,opaque,&points,&product_epoch))return 0;",
    )
    text = text.replace(
        evaluator,
        evaluator[:-2] + ",const merlin_source_point_span *points,const void *epoch){\n"
        " if(!merlin_source_point_span_matches(points,w->a,(size_t)m*k,epoch))return 0;",
    )
    # Null interval bounds mean the exact source value, not zero uncertainty
    # inferred from sampled values. Encoding still proves representation error;
    # all original source-FMA error/overflow admission remains unchanged.
    text = text.replace(encode, "w->astep,0,0,w->encoded_a_exact,&aproof)")
    text = text.replace(bounds, bounds.replace("w->a,w->al,w->ah,", "w->a,0,0,"))
    return text
