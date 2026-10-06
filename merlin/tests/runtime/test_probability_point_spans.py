"""Exact probability points, private source epochs and owned producer refusal."""

import ctypes
import shutil
import subprocess

import pytest
from test_source_attention_frontier import PLAN

from merlin.common.paths import merlin_dir
from merlin.llvmlower.probability_point_spans import prepare_probability_point_spans
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier


OPTIONS = dict(
    prepare_probability_bins=True,
    prepare_softmax_spans=True,
    prepare_encoded_rows=True,
    prepare_required_norms=True,
    prepare_product_domain=True,
    separable_source_radius=True,
    prepare_softmax_domain=True,
)


def test_default_identity_and_owned_point_calls():
    original = emit_source_attention_frontier(PLAN, symbol="test", **OPTIONS)
    assert original == emit_source_attention_frontier(
        PLAN, symbol="test", prepare_probability_points=False, **OPTIONS
    )
    selected = emit_source_attention_frontier(PLAN, symbol="test", prepare_probability_points=True, **OPTIONS)
    assert selected == prepare_probability_point_spans(original)
    assert selected.count("merlin_bf16_exact_point_finish(y,bins)") == 2
    assert selected.count("merlin_source_point_span points=") == 2
    assert "dot_bounds(w->a,0,0,w->b" in selected
    assert "w->astep,0,0,w->encoded_a_exact,&aproof)" in selected
    assert "w->al[r*length+z]=" not in selected
    assert "w->ah[r*length+z]=" not in selected
    # Private allocation/ABI remains unchanged; no storage saving is claimed.
    for token in ("float a[", "float al[", "float ah["):
        assert selected.count(token) == original.count(token)


@pytest.mark.parametrize("bad", [None, 0, 1, "true"])
def test_boolean_refusal(bad):
    with pytest.raises(ValueError, match="probability point"):
        emit_source_attention_frontier(PLAN, symbol="test", prepare_probability_points=bad, **OPTIONS)


@pytest.mark.parametrize("missing", ["prepare_probability_bins", "prepare_softmax_spans", "prepare_encoded_rows"])
def test_missing_producer_proof_refuses(missing):
    options = dict(OPTIONS, **{missing: False})
    with pytest.raises(ValueError, match="probability points"):
        emit_source_attention_frontier(PLAN, symbol="test", prepare_probability_points=True, **options)


@pytest.mark.parametrize(
    "before,after",
    [
        ("bins=merlin_bf16_interval_prepare(y); counts[2]++;", "counts[2]++;"),
        ("if(!(lo[off+j]<=exact && exact<=hi[off+j]))return 0;", ""),
        ("w->a[r*length+z]=h->p[ix];", "w->a[r*length+z]=h->plo[ix];"),
        ("for(int tile=0;tile<2;tile++)for(int part=0;part<PARTS;part++){", "h->p[0]=1;for(int tile=0;tile<2;tile++)for(int part=0;part<PARTS;part++){"),
        ("dot_bounds(w->a,w->al,w->ah,w->b", "dot_bounds(w->a,w->ah,w->al,w->b"),
        ("if(!evaluate_products(w,ROWS,CHUNK,DEPTH,product,opaque))return 0;", ""),
    ],
)
def test_owned_source_mutation_refuses(before, after):
    source = emit_source_attention_frontier(PLAN, symbol="test", **OPTIONS)
    assert before in source
    with pytest.raises(ValueError):
        prepare_probability_point_spans(source.replace(before, after))


C = r"""
#include <fenv.h>
#include "prepared_bf16_interval.h"
int probe(void){
 int old=fegetround(),modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 for(int mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 1;
  for(uint32_t word=0;word<65536;word++){
   uint32_t bits=word<<16;float value=merlin_interval_float(bits);
   merlin_f32_interval x=merlin_interval_point(value);
   merlin_bf16_interval_bins bins=merlin_bf16_interval_prepare(x);
   merlin_bf16_exact_point point=merlin_bf16_exact_point_finish(x,bins);
   int finite=(bits&0x7f800000)!=0x7f800000;
   if(point.valid!=finite)return 2;
   if(finite&&merlin_interval_bits(point.value)!=bits)return 3;
   /* Exercise both sides of every binary32->BF16 rounding transition. */
   for(int delta=-2;delta<=2;delta++){
    float a=merlin_interval_float(bits+32768u+(uint32_t)delta);
    float b=merlin_interval_float(bits+32768u+(uint32_t)(delta+1));
    if(!isfinite(a)||!isfinite(b))continue;
    x=merlin_interval(fminf(a,b),fmaxf(a,b));bins=merlin_bf16_interval_prepare(x);
    point=merlin_bf16_exact_point_finish(x,bins);
    if(point.valid){
     if(merlin_interval_bits(point.value)!=merlin_interval_bits(merlin_bf16_interval_midpoint(x,bins)))return 4;
     if(bins.low_bits!=bins.high_bits)return 5;
    }
   }
  }
  merlin_f32_interval zero={-0.f,0.f,1};
  if(merlin_bf16_exact_point_finish(zero,merlin_bf16_interval_prepare(zero)).valid)return 6;
  zero.valid=0;if(merlin_bf16_exact_point_finish(zero,merlin_bf16_interval_prepare(zero)).valid)return 7;
 }
 float a[2]={0},b[2]={0};unsigned char first,second;
 merlin_source_point_span span={a,2,&first};
 if(!merlin_source_point_span_matches(&span,a,2,&first))return 8;
 if(merlin_source_point_span_matches(&span,b,2,&first)||
    merlin_source_point_span_matches(&span,a,1,&first)||
    merlin_source_point_span_matches(&span,a,2,&second)||
    merlin_source_point_span_matches(0,a,2,&first)||
    merlin_source_point_span_matches(&span,a,2,0))return 9;
 return fesetround(old)?10:0;
}
"""


def test_all_bf16_and_f32_ties_environment_and_epoch(tmp_path):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    source = tmp_path / "probe.c"
    source.write_text(C)
    library = tmp_path / "probe.so"
    subprocess.run(
        [cc, "-O2", "-frounding-math", "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC",
         "-I", str(merlin_dir() / "runtime/c"), str(source), "-lm", "-o", str(library)],
        check=True,
    )
    assert ctypes.CDLL(str(library)).probe() == 0
