"""Whole portable source-group execution, dirty lifetime and refusal gates."""

from __future__ import annotations

import ctypes as C
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.common.paths import data_path
from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan, emit_source_attention_frontier

PLAN = SourceAttentionFrontierPlan(
    2,
    3,
    4,
    8,
    3,
    2,
    0.125,
    -87.3365478515625,
    1.4426950216293335,
    (-0.079204238951206207, -0.22433836758136749, 0.30354261398315430, 0.00010703434963943437),
    8388608.0,
    1065353216.0,
    127.0,
    float.fromhex("0x1.5p-17"),
    -127,
    127,
)


class View(C.Structure):
    _fields_ = [("data", C.c_void_p), ("offset", C.c_int64), ("sizes", C.c_int64 * 4), ("strides", C.c_int64 * 4)]


def view(a):
    return View(a.ctypes.data, 0, (C.c_int64 * 4)(*a.shape), (C.c_int64 * 4)(*[s // a.itemsize for s in a.strides]))


EXTRA = r"""
static int native_products(void*opaque,const int8_t*a,const int8_t*b,int32_t*c,int m,int n,int k,int degree){
 if(opaque)return 0;
 for(int row=0;row<m;row++)for(int col=0;col<n;col++){
  int64_t sum=0;for(int ad=0;ad<3;ad++){int bd=degree-ad;if(bd<0||bd>2)continue;
   for(int z=0;z<k;z++)sum+=(int32_t)a[ad*m*k+row*k+z]*(int32_t)b[bd*k*n+z*n+col];}
  c[row*n+col]=(int32_t)sum;
 }return 1;
}
int run(merlin_attention_view*in,merlin_attention_view*out,void*w,size_t cap,int fail){return test_provider(in,out,w,cap,native_products,(void*)(uintptr_t)fail);}
/* Independent direct original operand source replay, no integer centers or
 * interval/certificate code. Source ordered FMA, lane tree, casts and chunks. */
int oracle(merlin_attention_view*in,uint16_t*out){
 float values[HEADS][ROWS][DEPTH];
 for(int h=0;h<HEADS;h++)for(int r=0;r<ROWS;r++){
  float old=-INFINITY,den=0,acc[DEPTH]={0};
  for(int tile=0;tile<2;tile++){
   float scores[CHUNK],prob[CHUNK],lanes[LANES]={0},mx=old;
   for(int z=0;z<CHUNK;z++){
    int mi=tile?7:2;unsigned mask=((uint8_t*)in[mi].data)[physical(&in[mi],in[mi].sizes[1]==1?0:h,r,z)];
    float sum=0;for(int d=0;d<DEPTH;d++)sum=fmaf(load_bf16(&in[0],h,r,d),load_bf16(&in[tile?6:1],h,z,d),sum);
    scores[z]=mask?sum*SCORE_SCALE:-INFINITY;mx=fmaxf(mx,scores[z]);
   }
   for(int z=0;z<CHUNK;z++){
    float x=scores[z]-(mx==-INFINITY?0:mx),y;
    if(x<plan.cutoff)y=0;
    else{float scaled=x*plan.scale,fraction=scaled-floorf(scaled),poly=plan.coefficients[0];
     for(int c=1;c<4;c++)poly=fmaf(fraction,poly,plan.coefficients[c]);
     y=merlin_interval_float((uint32_t)(int32_t)fmaf(plan.bit_multiplier,scaled-poly,plan.bit_bias));}
    lanes[z%LANES]+=y;prob[z]=merlin_interval_bf16(y);
   }
   for(int n=LANES/2;n;n/=2)for(int l=0;l<n;l++)lanes[l]+=lanes[l+n];
   float alpha=mx==-INFINITY?1.f:(float)exp((double)(old-mx));den=fmaf(alpha,den,lanes[0]);
   for(int d=0;d<DEPTH;d++){
    acc[d]*=alpha;
    for(int part=0;part<PARTS;part++){
     float sum=0;int start=part*SEGMENT,len=CHUNK-start<SEGMENT?CHUNK-start:SEGMENT;
     for(int z=0;z<len;z++)sum=fmaf(prob[start+z],load_bf16(&in[(tile?8:3)+part],h,z,d),sum);
     acc[d]+=sum;
    }
   }old=mx;
  }
  float reciprocal=1.f/(den==0?1.f:den);
  for(int d=0;d<DEPTH;d++)values[h][r][d]=merlin_interval_bf16(acc[d]*reciprocal);
 }
 for(int h=0;h<HEADS;h++)for(int r=0;r<ROWS;r++)for(int d=0;d<DEPTH;d++)out[(h*ROWS+r)*DEPTH+d]=(uint16_t)(merlin_interval_bits(values[h][r][d])>>16);
 return 1;
}
"""
RETENTION_TEST = r"""
int retention_stability(void){
 float p[2*PARTS*ROWS*DEPTH]={0},c[2*PARTS*ROWS*DEPTH]={0};
 float alpha[ROWS*2],den[ROWS],lo[ROWS*DEPTH],hi[ROWS*DEPTH],out[ROWS*DEPTH];
 uint8_t certified[ROWS]={0};
 for(int r=0;r<ROWS;r++){den[r]=1;alpha[r*2]=alpha[r*2+1]=1;}
 if(!endpoint_intervals(p,p,c,alpha,den,den,lo,hi,out,ROWS,certified))return 0;
 certified[0]=1;float saved[3*DEPTH];
 memcpy(saved,lo,DEPTH*4);memcpy(saved+DEPTH,hi,DEPTH*4);memcpy(saved+2*DEPTH,out,DEPTH*4);
 for(int t=0;t<2*PARTS;t++)for(int j=0;j<DEPTH;j++)p[(t*ROWS+1)*DEPTH+j]=c[(t*ROWS+1)*DEPTH+j]=1;
 if(!endpoint_intervals(p,p,c,alpha,den,den,lo,hi,out,ROWS,certified))return 0;
 if(memcmp(saved,lo,DEPTH*4)||memcmp(saved+DEPTH,hi,DEPTH*4)||memcmp(saved+2*DEPTH,out,DEPTH*4))return 0;
 for(int j=0;j<DEPTH;j++)if(out[DEPTH+j]!=(float)(2*PARTS))return 0;
 return 1;
}
"""


@pytest.fixture(
    scope="module",
    params=[
        (row, word, zero, domain, norms, retain, radius, False)
        for row in (False, True)
        for word in (False, True)
        for zero in (False, True)
        for domain in (False, True)
        for norms in (False, True)
        for retain in (False, True)
        for radius in (False, True)
        if (not norms or domain) and (not retain or row) and (not radius or domain)
    ]
    + [
        (True, False, False, True, True, True, True, True),
        (False, False, False, False, False, False, False, True),
        (False, False, False, False, False, False, False, False, True),
        (True, False, False, True, True, True, True, True, True),
        pytest.param((True, True, False, True, True, True, True, True, True), id="word_soft_i64"),
        pytest.param((False, True, False, False, False, False, False, True), id="word_soft_basic"),
        pytest.param((True, True, False, True, True, True, True, True, True, True), id="prepared_bins_word_soft_i64"),
        pytest.param((False, False, False, False, False, False, False, False, False, True), id="prepared_bins_basic"),
        pytest.param((True, True, False, True, True, True, True, True, True, False, True), id="encoded_rows_word"),
        pytest.param(
            (False, False, False, True, True, False, False, False, False, False, True), id="encoded_rows_basic"
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, True, True), id="encoded_rows_prepared_bins"
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, False, False, True), id="producer_spans_word"
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, True, True, True), id="producer_spans_encoded_bins"
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, False, False, False, True),
            id="exact_bound_conversion",
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, True, True, True, True),
            id="exact_bound_conversion_composed",
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, False, False, False, False, True), id="batch_four"
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, True, True, True, True, True),
            id="batch_four_composed",
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, False, False, False, False, False, True),
            id="fused_integer_reconstruction",
        ),
        pytest.param(
            (True, True, False, True, True, True, True, True, True, True, True, True, True, True, True),
            id="fused_integer_reconstruction_composed",
        ),
    ],
)
def native(tmp_path_factory, request):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native compiler required")
    from merlin.llvmlower.exact_bound_conversion import ExactBoundConversionContract

    exact = len(request.param) > 12 and request.param[12]
    prefix = ""
    if exact:
        prefix = "#include <fenv.h>\n#pragma STDC FENV_ACCESS ON\nstatic float test_floor(double x){int old=fegetround();fesetround(FE_DOWNWARD);volatile double d=x;volatile float v=(float)d;fesetround(old);return v;}\nstatic float test_ceil(double x){int old=fegetround();fesetround(FE_UPWARD);volatile double d=x;volatile float v=(float)d;fesetround(old);return v;}\n#pragma STDC FENV_ACCESS OFF\n#define MERLIN_F32_EXACT_FLOOR_FROM_F64(x) test_floor(x)\n#define MERLIN_F32_EXACT_CEIL_FROM_F64(x) test_ceil(x)\n"
    w = tmp_path_factory.mktemp("source-frontier")
    (w / "test.c").write_text(
        prefix
        + emit_source_attention_frontier(
            PLAN,
            symbol="test_provider",
            prepare_endpoint_rows=request.param[0],
            word_interval_enclosure=request.param[1],
            prepare_zero_error_blocks=request.param[2],
            prepare_product_domain=request.param[3],
            prepare_required_norms=request.param[4],
            retain_certified_rows=request.param[5],
            separable_source_radius=request.param[6],
            prepare_softmax_domain=request.param[7],
            integer_reconstruction=request.param[8] if len(request.param) > 8 else False,
            fuse_integer_reconstruction=request.param[14] if len(request.param) > 14 else False,
            prepare_probability_bins=request.param[9] if len(request.param) > 9 else False,
            prepare_encoded_rows=request.param[10] if len(request.param) > 10 else False,
            prepare_softmax_spans=request.param[11] if len(request.param) > 11 else False,
            exact_bound_conversion=ExactBoundConversionContract(True, True, True, True, True) if exact else None,
            polynomial_batch_four=request.param[13] if len(request.param) > 13 else False,
        )
        + EXTRA
        + (RETENTION_TEST if request.param[5] else "")
    )
    headers = data_path("runtime", "c")
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(headers),
            str(w / "test.c"),
            "-lm",
            "-o",
            str(w / "test.so"),
        ],
        check=True,
    )
    lib = C.CDLL(str(w / "test.so"))
    lib.test_provider_workspace_bytes.restype = C.c_size_t
    lib.test_provider_workspace_alignment.restype = C.c_size_t
    if request.param[5]:
        assert lib.retention_stability() == 1
    lib.run.argtypes = [C.POINTER(View), C.POINTER(View), C.c_void_p, C.c_size_t, C.c_int]
    lib.oracle.argtypes = [C.POINTER(View), C.c_void_p]
    return lib


def inputs(seed, masked=False, strided=False):
    rng = np.random.default_rng(seed)
    arrays = []
    for i in range(11):
        if i in (2, 7):
            a = rng.integers(0, 2, (1, 1, 3, 8), dtype=np.uint8)
            a[:] = 0 if masked else a
        else:
            rows = 3 if i == 0 else 8 if i in (1, 6) else 2 if i in (5, 10) else 3
            f = rng.integers(-8, 9, (1, 2, rows, 4)).astype(np.float32) / 8
            a = (f.view(np.uint32) >> 16).astype(np.uint16)
        if strided:
            backing = np.zeros((*a.shape[:-1], a.shape[-1] * 2), dtype=a.dtype)
            backing[..., ::2] = a
            a = backing[..., ::2]
        arrays.append(a)
    return arrays


def quant(words):
    f = (words.astype(np.uint32) << 16).view(np.float32)[0].transpose(1, 0, 2).reshape(3, -1)

    def bf(x):
        a = np.ascontiguousarray(x, np.float32)
        u = a.view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)

    scale = np.maximum(bf(abs(f).max(1) / np.float32(127)), np.float32(PLAN.quant_epsilon))
    inverse = bf(np.float32(1) / scale)
    q = np.clip(bf(np.rint(bf(f * inverse[:, None]))), -127, 127).astype(np.int8)
    return q, scale.view(np.uint32)


@pytest.mark.parametrize(
    "seed,masked,strided", [(1, False, False), (2, False, True), (3, True, True), (4, False, False)]
)
def test_independent_source_quant_and_reused_dirty_workspace(native, seed, masked, strided):
    a = inputs(seed, masked, strided)
    vs = (View * 11)(*[view(x) for x in a])
    size = native.test_provider_workspace_bytes()
    storage = np.full(size + 64, 0xA5, np.uint8)
    ptr = (storage.ctypes.data + 63) & ~63
    for _ in range(2):
        out = np.full((1, 2, 3, 4), 0xDEAD, np.uint16)
        gold = np.empty_like(out)
        assert native.run(vs, C.byref(view(out)), ptr, size, 0) == 1
        assert native.oracle(vs, gold.ctypes.data) == 1
        assert all(np.array_equal(x, y) for x, y in zip(quant(out), quant(gold)))
        assert np.all(storage[: ptr - storage.ctypes.data] == 0xA5)
        assert np.all(storage[ptr - storage.ctypes.data + size :] == 0xA5)


@pytest.mark.parametrize(
    "failure", ["capacity", "alignment", "callback", "nonfinite", "stride", "shape", "overlapping_output"]
)
def test_refusal_preserves_public_destination(native, failure):
    a = inputs(31)
    vs = (View * 11)(*[view(x) for x in a])
    size = native.test_provider_workspace_bytes()
    storage = np.full(size + 64, 0xAA, np.uint8)
    ptr = (storage.ctypes.data + 63) & ~63
    cap = size
    fail = 0
    if failure == "capacity":
        cap -= 1
    if failure == "alignment":
        ptr += 1
    if failure == "callback":
        fail = 1
    if failure == "nonfinite":
        a[0].flat[0] = 0x7F80
    if failure == "stride":
        vs[0].strides[3] = -1
    if failure == "shape":
        vs[0].sizes[2] += 1
    out = np.full((1, 2, 3, 4), 0xDEAD, np.uint16)
    ov = view(out)
    if failure == "overlapping_output":
        ov.strides[3] = 0
    assert native.run(vs, C.byref(ov), ptr, cap, fail) == 0
    assert (out == 0xDEAD).all()


@pytest.mark.parametrize(
    "changes",
    [
        {"depth": 9},
        {"segment": 8},
        {"denominator_lanes": 3},
        {"heads": 2**31},
        {"quant_divisor": 0},
        {"score_scale": float("nan")},
    ],
)
def test_illegal_contracts_refuse(changes):
    with pytest.raises(ValueError):
        emit_source_attention_frontier(replace(PLAN, **changes), symbol="test_provider")


@pytest.mark.parametrize("value", [True, ".125"])
def test_source_constant_types_refuse(value):
    with pytest.raises((ValueError, TypeError)):
        emit_source_attention_frontier(replace(PLAN, score_scale=value), symbol="provider")


def test_runtime_template_uses_canonical_relocated_data(monkeypatch, tmp_path):
    from merlin.llvmlower import source_attention_frontier as emitter

    expected = emit_source_attention_frontier(PLAN, symbol="provider")
    runtime = tmp_path / "runtime"
    (runtime / "templates").mkdir(parents=True)
    for source in data_path("runtime", "c", "templates").glob("source_attention*.c.in"):
        shutil.copyfile(source, runtime / "templates" / source.name)
    monkeypatch.setattr(emitter, "data_path", lambda *parts: runtime)
    assert emit_source_attention_frontier(PLAN, symbol="provider") == expected


def test_two_distinct_provider_plans_link_without_helper_collisions(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler unavailable")
    files = []
    for name, plan in [("first", PLAN), ("second", replace(PLAN, query_rows=5))]:
        c = tmp_path / (name + ".c")
        c.write_text(emit_source_attention_frontier(plan, symbol=name))
        files.append(str(c))
    so = tmp_path / "both.so"
    subprocess.run(
        [cc, "-O2", "-shared", "-fPIC", "-I", str(data_path("runtime", "c")), *files, "-lm", "-o", str(so)], check=True
    )
    lib = C.CDLL(str(so))
    for name in ["first", "second"]:
        function = getattr(lib, name + "_workspace_bytes")
        function.restype = C.c_size_t
        assert function() > 0
    for helper in [
        "dot_bounds",
        "soft_details",
        "endpoint_intervals",
        "exact_source_denominator",
        "exact_source_partials",
    ]:
        with pytest.raises(AttributeError):
            getattr(lib, helper)


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_endpoint_policy_requires_boolean(value):
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="provider", prepare_endpoint_rows=value)


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_word_enclosure_requires_boolean(value):
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="provider", word_interval_enclosure=value)


def test_word_enclosure_only_selects_explicit_source_bound_helper():
    default = emit_source_attention_frontier(PLAN, symbol="provider")
    assert default == emit_source_attention_frontier(PLAN, symbol="provider", word_interval_enclosure=False)
    selected = emit_source_attention_frontier(PLAN, symbol="provider", word_interval_enclosure=True)
    assert selected == default.replace(
        "merlin_monotone_bit_polynomial_apply(x,&root_prepared)",
        "merlin_monotone_bit_polynomial_apply_words(x,&root_prepared)",
    )


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_zero_error_policy_requires_boolean(value):
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", prepare_zero_error_blocks=value)


@pytest.mark.parametrize("case", ["zero", "exact", "lhs_error", "rhs_error", "uncertain", "mixed"])
def test_zero_error_blocks_match_complete_checked_bounds(tmp_path, case):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler unavailable")
    wrapper = r"""
int probe_bounds(const float*a,const float*al,const float*ah,const float*b,
 const double*ar,const double*br,const double*c,float*lo,float*hi){
 struct product_scratch scratch;
 return dot_bounds(a,al,ah,b,ar,br,c,lo,hi,3,5,4,&scratch);
}
"""
    libraries = []
    for enabled in (False, True):
        source = tmp_path / (str(enabled) + ".c")
        out = source.with_suffix(".so")
        source.write_text(emit_source_attention_frontier(PLAN, symbol="p", prepare_zero_error_blocks=enabled) + wrapper)
        subprocess.run(
            [
                cc,
                "-O2",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-shared",
                "-fPIC",
                "-I",
                str(data_path("runtime", "c")),
                str(source),
                "-lm",
                "-o",
                str(out),
            ],
            check=True,
        )
        lib = C.CDLL(str(out))
        lib.probe_bounds.argtypes = [C.c_void_p] * 9
        libraries.append(lib)
    a = np.arange(-6, 6, dtype=np.float32).reshape(3, 4) / 8
    b = np.arange(-10, 10, dtype=np.float32).reshape(5, 4) / 8
    if case == "zero":
        a[:] = 0
        b[:] = -0.0
    al = a.copy()
    ah = a.copy()
    ar = a.astype(np.float64)
    br = b.astype(np.float64)
    if case in ("lhs_error", "mixed"):
        ar[1, 2] += 2.0**-12
    if case in ("rhs_error", "mixed"):
        br[4, 3] += 2.0**-12
    if case in ("uncertain", "mixed"):
        al[2, 1] -= np.float32(2.0**-10)
        ah[2, 1] += np.float32(2.0**-10)
    center = ar @ br.T
    answers = []
    for lib in libraries:
        lo = np.full((3, 5), np.nan, np.float32)
        hi = lo.copy()
        assert lib.probe_bounds(*[x.ctypes.data for x in (a, al, ah, b, ar, br, center, lo, hi)]) == 1
        answers.append((lo.view(np.uint32).copy(), hi.view(np.uint32).copy()))
    assert all(np.array_equal(x, y) for x, y in zip(*answers)), case


def test_zero_error_default_matches_pre_feature_bytes():
    import hashlib

    expected = "045d127bb2924c4ad437d930739f12c34b02a301936ad1d220de7dbbb7cce653"
    assert hashlib.sha256(emit_source_attention_frontier(PLAN, symbol="provider").encode()).hexdigest() == expected


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_product_domain_policy_is_explicit_bool(value):
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", prepare_product_domain=value)


def test_product_domain_default_bytes():
    import hashlib

    assert (
        hashlib.sha256(emit_source_attention_frontier(PLAN, symbol="provider").encode()).hexdigest()
        == "045d127bb2924c4ad437d930739f12c34b02a301936ad1d220de7dbbb7cce653"
    )


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_norm_requirements_policy_is_explicit_bool(value):
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", prepare_product_domain=True, prepare_required_norms=value)


def test_norm_requirements_need_typed_consumer():
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", prepare_required_norms=True)


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_certificate_retention_policy_is_explicit_bool(value):
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", prepare_endpoint_rows=True, retain_certified_rows=value)


def test_certificate_retention_requires_row_locality():
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", retain_certified_rows=True)


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_separable_radius_requires_boolean(value):
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", prepare_product_domain=True, separable_source_radius=value)


def test_separable_radius_requires_producer():
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", separable_source_radius=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_integer_reconstruction_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean integer reconstruction"):
        emit_source_attention_frontier(PLAN, symbol="provider", integer_reconstruction=value)


def test_integer_reconstruction_default_identity_and_private_storage():
    default = emit_source_attention_frontier(PLAN, symbol="provider")
    assert default == emit_source_attention_frontier(PLAN, symbol="provider", integer_reconstruction=False)
    selected = emit_source_attention_frontier(PLAN, symbol="provider", integer_reconstruction=True)
    assert "int64_t integer_center[ROWS*CHUNK]" in selected
    assert "merlin_radix_integer_begin_from_first_group_exact_i64(w->integer_center" in selected
    assert "merlin_radix_integer_finish_exact_f64(w->center,w->integer_center" in selected
    assert "w->center[r*n+c]*=w->astep[r]*w->bstep[c]" in selected


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_fused_integer_reconstruction_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean fused integer"):
        emit_source_attention_frontier(PLAN, symbol="provider", fuse_integer_reconstruction=value)


def test_fused_integer_reconstruction_requires_complete_integer_plan():
    with pytest.raises(ValueError, match="requires integer reconstruction"):
        emit_source_attention_frontier(PLAN, symbol="provider", fuse_integer_reconstruction=True)


def test_fused_integer_reconstruction_default_identity_and_complete_storage():
    options = dict(symbol="provider", integer_reconstruction=True)
    default = emit_source_attention_frontier(PLAN, **options)
    assert default == emit_source_attention_frontier(PLAN, **options, fuse_integer_reconstruction=False)
    selected = emit_source_attention_frontier(PLAN, **options, fuse_integer_reconstruction=True)
    assert "int32_t readout[MERLIN_RADIX_FUSED_INTEGER_GROUPS][ROWS*CHUNK]" in selected
    assert "int64_t integer_center[" not in selected
    assert "product(opaque,w->ap,w->bp,w->readout[degree],m,n,k,degree)" in selected
    assert "merlin_radix_integer_fused_exact_f64(w->center,planes,(size_t)m*n)" in selected
    assert "w->center[r*n+c]*=w->astep[r]*w->bstep[c]" in selected
