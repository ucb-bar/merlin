"""The emitted correction preserves the original arithmetic for every i8 pair."""

import ctypes
import hashlib
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from merlin.llvmlower.quantized_affine_pair import derive, emit_correction, predictor_table, source_table

SOURCE = dict(
    lhs_scale=0.011258588172495365, rhs_scale=0.00940733402967453, output_scale=0.011643771082162857, relu=True
)
PREDICTOR = dict(p=73, q=61, scale=0.013245166279375553)
CONTRACTS = [
    (SOURCE, PREDICTOR),
    (dict(lhs_scale=0.125, rhs_scale=0.03125, output_scale=0.25, relu=False), dict(p=4, q=1, scale=0.125)),
    (dict(lhs_scale=0.003, rhs_scale=0.004, output_scale=0.002, relu=False), dict(p=3, q=4, scale=0.5)),
]


def test_complete_certificate_keeps_predictor_error_and_runtime_distribution_explicit():
    proof = derive(**SOURCE, **PREDICTOR)
    assert proof["pairs"] == 65536 and proof["mismatched_pairs"] == 53
    assert proof["max_predictor_output_error"] == 1
    assert proof["packed_prefix_guard"] == dict(
        axis="lhs", negative_radius=32, positive_radius=64, safe_min=-32, safe_max=63
    )
    assert proof["correction_exact_for_all_pairs"]
    assert proof["predictor_bit_guard"] == dict(mask=1, value=1)
    assert proof["runtime_ambiguity_rate"].startswith("UNKNOWN")
    assert len(bytes.fromhex(proof["correction_bitmap_hex"])) == 8192
    assert "independently validated" in proof["predictor_arithmetic"]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("unaligned", [False, True])
@pytest.mark.parametrize("source,predictor", CONTRACTS)
def test_actual_portable_c_correction_all_pairs_tails_and_guards(tmp_path, packed, unaligned, source, predictor):
    proof = derive(**source, **predictor)
    c = tmp_path / "correction.c"
    c.write_text(emit_correction(proof, "correct", packed_prefix=packed))
    # pytest may remove and reuse a successful temporary directory while the
    # process still has its CDLL mapped. Distinct source names avoid dlopen's
    # path cache loading a prior numeric contract.
    lib = tmp_path / ("correction_" + hashlib.sha256(c.read_bytes()).hexdigest() + ".so")
    subprocess.run(["cc", "-O2", "-fPIC", "-shared", "-ffp-contract=off", str(c), "-o", str(lib)], check=True)
    correct = ctypes.CDLL(str(lib)).correct
    correct.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
    lhs = np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
    rhs = np.tile(np.arange(-128, 128, dtype=np.int8), 256)
    offset = int(unaligned)
    a = np.zeros(65536 + 8, np.int8)
    b = np.zeros_like(a)
    a[offset : offset + 65536] = lhs
    b[offset : offset + 65536] = rhs
    output = np.full(65536 + 64, 37, np.int8)
    output[:65536] = predictor_table(**predictor, relu=source["relu"]).ravel()
    correct(a.ctypes.data + offset, b.ctypes.data + offset, output.ctypes.data, 65536)
    assert np.array_equal(output[:65536], source_table(**source).ravel())
    assert (output[65536:] == 37).all()
    # An independent permutation puts exceptional pairs in final scalar tails.
    rng = np.random.default_rng(42)
    indices = rng.permutation(65536)[:1013]
    a[offset : offset + 1013] = lhs[indices]
    b[offset : offset + 1013] = rhs[indices]
    output[:1013] = predictor_table(**predictor, relu=source["relu"]).ravel()[indices]
    output[1013:] = 37
    correct(a.ctypes.data + offset, b.ctypes.data + offset, output.ctypes.data, 1013)
    assert np.array_equal(output[:1013], source_table(**source).ravel()[indices])
    assert (output[1013:] == 37).all()


def test_corrupted_certificate_and_nonfinite_or_overflowing_arithmetic_refuse():
    proof = derive(**SOURCE, **PREDICTOR)
    changed = deepcopy(proof)
    changed["packed_prefix_guard"]["safe_max"] = 127
    with pytest.raises(ValueError, match="unchanged complete pair"):
        emit_correction(changed, "correct")
    with pytest.raises(ValueError, match="positive finite"):
        derive(**dict(SOURCE, lhs_scale=float("nan")), **PREDICTOR)
    with pytest.raises(ValueError, match="signed-i32"):
        predictor_table(1 << 30, 1, 1.0)
    with pytest.raises(ValueError, match="may overflow"):
        source_table(1e38, 1e38, 1.0, relu=False)
    with pytest.raises(ValueError, match="explicit source ReLU"):
        source_table(1.0, 1.0, 1.0, relu=1)
    for symbol in ("for", "unicode_λ", "a;evil()", ""):
        with pytest.raises(ValueError, match="valid identifier"):
            emit_correction(proof, symbol)


def test_zero_adjacent_ambiguity_does_not_create_an_empty_maximum():
    from merlin.llvmlower.quantized_affine_pair import _centered_guard

    assert _centered_guard({-1, 5})["negative_radius"] == 0
    assert _centered_guard({0, 7}) is None


def test_actual_c_prefix_all_radii_lanes_bytes_and_gather_masks(tmp_path):
    from merlin.llvmlower.quantized_affine_pair import _GATHER_EXPRESSION, _prefix_expression

    cases = [(left, right) for left in (0, 1, 2, 4, 8, 16, 32, 64, 128) for right in (1, 2, 4, 8, 16, 32, 64, 128)]
    code = ["#include <stdint.h>"]
    for index, (left, right) in enumerate(cases):
        expr = _prefix_expression(dict(negative_radius=left, positive_radius=right))
        code.append(f"uint64_t prefix{index}(uint64_t raw) {{return {expr};}}")
    code.append(f"unsigned gather(uint64_t outside) {{return (unsigned)({_GATHER_EXPRESSION});}}")
    # This is the emitter's conservative byte-zero predicate, tested against
    # independent per-byte equality. Adjacent zero bytes may add flags but must
    # never remove a required one.
    code.append(
        "uint64_t zero(uint64_t difference) {return (difference-UINT64_C(0x0101010101010101))"
        "&~difference&UINT64_C(0x8080808080808080);}"
    )
    source = tmp_path / "words.c"
    source.write_text("\n".join(code))
    lib = tmp_path / "words.so"
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", str(source), "-o", str(lib)], check=True)
    library = ctypes.CDLL(str(lib))
    for index, (left, right) in enumerate(cases):
        fn = getattr(library, "prefix" + str(index))
        fn.argtypes = [ctypes.c_uint64]
        fn.restype = ctypes.c_uint64
        for lane in range(8):
            for value in range(256):
                values = [0] * 8
                values[lane] = value
                raw = sum(v << (8 * i) for i, v in enumerate(values))
                expected = sum(
                    0x80 << (8 * i)
                    for i, v in enumerate(values)
                    if (v if v < 128 else v - 256) < -left or (v if v < 128 else v - 256) >= right
                )
                assert fn(raw) == expected, (left, right, lane, value)
    gather = library.gather
    gather.argtypes = [ctypes.c_uint64]
    gather.restype = ctypes.c_uint
    for flags in range(256):
        word = sum(0x80 << (8 * lane) for lane in range(8) if flags & (1 << lane))
        assert gather(word) == flags
    zero = library.zero
    zero.argtypes = [ctypes.c_uint64]
    zero.restype = ctypes.c_uint64
    for adjacent in range(7):
        for first in range(256):
            for second in (0, 1, 2, 127, 128, 255):
                values = [1] * 8
                values[adjacent] = first
                values[adjacent + 1] = second
                word = sum(v << (8 * i) for i, v in enumerate(values))
                required = sum(0x80 << (8 * i) for i, v in enumerate(values) if v == 0)
                assert zero(word) & required == required


@pytest.mark.parametrize("lhs_scale,rhs_scale,relu,bound", [(1.,1.,False,0),(.875,.375,False,1),(.625,1.125,True,2)])
def test_loaded_sum_proof_matches_independent_compiled_f32_all_pairs(tmp_path_factory,lhs_scale,rhs_scale,relu,bound):
    from merlin.llvmlower.quantized_affine_pair import prove_loaded_sum
    from merlin.llvmlower.requantization import f32
    root=tmp_path_factory.mktemp("loaded-sum")
    store=f32(max(1,lhs_scale,rhs_scale));left=f32(lhs_scale/store);right=f32(rhs_scale/store)
    proof=prove_loaded_sum(lhs_scale,rhs_scale,load_lhs_scale=left,load_rhs_scale=right,
                           store_scale=store,bound_lsb=bound,relu=relu)
    low=0 if relu else -128
    source=f"""#include <stdint.h>
#include <stddef.h>
#include <math.h>
static int sat(float x,int lo){{float y=nearbyintf(x);if(y<lo)y=lo;if(y>127)y=127;return (int)y;}}
void run(const int8_t*a,const int8_t*b,int8_t*ref,int8_t*actual,size_t n){{
 for(size_t i=0;i<n;i++){{
  volatile float x=(float)a[i]*{float(f32(lhs_scale)).hex()}f;
  volatile float y=(float)b[i]*{float(f32(rhs_scale)).hex()}f;
  volatile float sum=x+y;ref[i]=(int8_t)sat(sum,{low});
  volatile float load_a=(float)a[i]*{float(left).hex()}f;
  volatile float load_b=(float)b[i]*{float(right).hex()}f;
  int32_t acc=sat(load_a,-128)+sat(load_b,-128);
  volatile float readout=(float)acc*{float(store).hex()}f;
  actual[i]=(int8_t)sat(readout,{low});
 }}
}}"""
    c=root/'sum.c';c.write_text(source);lib=root/'sum.so'
    subprocess.run(['cc','-O2','-ffp-contract=off','-shared','-fPIC',str(c),'-lm','-o',str(lib)],check=True,capture_output=True)
    run=ctypes.CDLL(str(lib)).run;run.argtypes=[ctypes.c_void_p]*4+[ctypes.c_size_t];run.restype=None
    a=np.repeat(np.arange(-128,128,dtype=np.int8),256);b=np.tile(np.arange(-128,128,dtype=np.int8),256)
    reference=np.full(65544,-73,dtype=np.int8);actual=reference.copy()
    run(a.ctypes.data,b.ctypes.data,reference.ctypes.data,actual.ctypes.data,65536)
    assert np.all(reference[65536:]==-73)and np.all(actual[65536:]==-73)
    errors=np.abs(reference[:65536].astype(np.int16)-actual[:65536].astype(np.int16))
    assert int(errors.max())==proof['max_error_lsb']<=bound
    assert hashlib.sha256(reference[:65536].tobytes()).hexdigest()==proof['source_sha256']
    assert hashlib.sha256(actual[:65536].tobytes()).hexdigest()==proof['proposed_sha256']


def test_loaded_sum_never_invents_error_budget_or_ignores_load_clipping():
    from merlin.llvmlower.quantized_affine_pair import prove_loaded_sum
    with pytest.raises(ValueError,match='exceeds source error bound'):
        prove_loaded_sum(.5,.5,load_lhs_scale=.5,load_rhs_scale=.5,store_scale=1,bound_lsb=0)
    with pytest.raises(ValueError,match='exceeds source error bound'):
        prove_loaded_sum(2.,.5,load_lhs_scale=2.,load_rhs_scale=.5,store_scale=1,bound_lsb=1)
    for bound in (-1,256,True):
        with pytest.raises(ValueError,match='explicit'):
            prove_loaded_sum(1.,1.,load_lhs_scale=1.,load_rhs_scale=1.,store_scale=1,bound_lsb=bound)
    for scale in (0,-1,float('inf'),float('nan')):
        with pytest.raises(ValueError):
            prove_loaded_sum(1.,1.,load_lhs_scale=scale,load_rhs_scale=1.,store_scale=1,bound_lsb=1)
