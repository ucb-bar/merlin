"""Complete-domain certificates for multiple rounded integer readouts.

No target instruction, resource, input sample or model identity is used. Each
candidate is a positive binary32 scale followed by the same nearest-even clamp.
A provider must separately prove that actual producer/storage and readout modes
implement these functions over the declared integer domain.
"""

from merlin.compiler.lower.quant.readout import derive, evaluate


def prove(source_scales, store_scales, lo, hi, *, relu=False):
    """Prove enclosure and whether two observed bytes determine source exactly.

    The union of every transition partitions the entire integer domain into
    constant cells. Repeated output pairs with distinct source values refuse a
    complete decoder and retain concrete counterexamples; no sampling is used.
    """
    if type(relu) is not bool:
        raise ValueError("relu must be boolean")
    if len(store_scales) != 2:
        raise ValueError("exactly two readout scales required")
    source = derive(source_scales, lo, hi, relu)
    stores = [derive([scale], lo, hi, relu) for scale in store_scales]
    boundaries = sorted({lo, hi + 1, *(v for p in [source, *stores] for v in p["thresholds"])})
    pairs = {}
    collisions = []
    enclosure_failures = []
    disagreements = []
    for start, stop in zip(boundaries, boundaries[1:]):
        expected = evaluate(start, source)
        a, b = [evaluate(start, p) for p in stores]
        if not min(a, b) <= expected <= max(a, b):
            enclosure_failures.append(dict(acc=start, source=expected, pair=[a, b]))
        key = (a, b)
        if key in pairs and pairs[key]["source"] != expected:
            collisions.append(dict(pair=list(key), first=pairs[key], other=dict(acc=start, source=expected)))
        else:
            pairs.setdefault(key, dict(acc=start, source=expected))
        if a != b:
            disagreements.append(dict(lo=start, hi=stop - 1, pair=[a, b], source=expected))
    exact = not collisions and not enclosure_failures
    return dict(
        schema="two_readout_enclosure_v1",
        source_scales=source["source_scales"],
        store_scales=[p["source_scales"][0] for p in stores],
        lo=lo,
        hi=hi,
        relu=relu,
        enclosed=not enclosure_failures,
        exact_pair_decoder=exact,
        enclosure_counterexamples=enclosure_failures,
        decoder_counterexamples=collisions,
        disagreement_intervals=disagreements,
        corrections=[dict(pair=list(k), source=v["source"]) for k, v in sorted(pairs.items()) if k[0] != k[1]],
        proof="Complete union of source and both store transition partitions; each pair has one source output on every reachable cell",
    )


def emit_pair_scan(certificate, symbol, *, copy_policy="runtime", checked_alignment=False):
    """Update the first byte array in place; second must be stable and disjoint.

    Complete producer-domain and both readout semantics are caller prerequisites.
    memcpy loads permit unaligned addresses. Equal eight-byte packets need no
    stores; differing packets are decoded lane by lane. The explicit compiler_builtin
    policy permits the C compiler to inline constant byte copies even under
    fno-builtin; runtime retains ordinary memcpy calls. Explicit checked_alignment
    tests both actual pointers before telling the compiler that packet loads are
    aligned. Other pointers keep byte-safe copies; complete packets and tails
    stay within count. This option requires compiler_builtin and never casts byte
    storage to an integer lvalue. Default C bytes are unchanged. No exception flags or
    floating arithmetic occur here. Unknown pairs trap instead of guessing.
    """
    if copy_policy not in ("runtime", "compiler_builtin"):
        raise ValueError("unknown portable copy policy")
    if type(checked_alignment) is not bool:
        raise ValueError("checked alignment must be boolean")
    if checked_alignment and copy_policy != "compiler_builtin":
        raise ValueError("checked alignment requires compiler builtin copies")
    copy_name = "memcpy" if copy_policy == "runtime" else "__builtin_memcpy"
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isidentifier():
        raise ValueError("C identifier required")
    expected = prove(
        certificate["source_scales"],
        certificate["store_scales"],
        certificate["lo"],
        certificate["hi"],
        relu=certificate["relu"],
    )
    if certificate != expected or not expected["exact_pair_decoder"]:
        raise ValueError("unaltered complete exact pair certificate required")
    cases = "\n".join(
        f"case {(r['pair'][0] & 255) | ((r['pair'][1] & 255) << 8)}: return (unsigned char){r['source']};"
        for r in expected["corrections"]
    )

    def packet_loop(first, second):
        return f""" for(;count-i>=8;i+=8){{uint64_t a,b;{copy_name}(&a,{first}+i,8);{copy_name}(&b,{second}+i,8);
  if(a!=b)for(size_t j=0;j<8;j++)first[i+j]={symbol}_lane(first[i+j],second[i+j]);}}"""

    packets = packet_loop("first", "second")
    if checked_alignment:
        packets = f""" if(count>=8&&(((uintptr_t)first|(uintptr_t)second)&7)==0){{
 unsigned char *aligned_first=__builtin_assume_aligned(first,8);
 const unsigned char *aligned_second=__builtin_assume_aligned(second,8);
{packet_loop("aligned_first", "aligned_second")}
 }}else{{
{packets}
 }}"""
    return f"""#include <stdint.h>
#include <stddef.h>
extern void *memcpy(void *, const void *, size_t);
static unsigned char {symbol}_lane(unsigned char a,unsigned char b){{
 if(a==b)return a;
 switch((unsigned)a|((unsigned)b<<8)){{{cases}\n default:__builtin_trap();}}
}}
void {symbol}(unsigned char *first,const unsigned char *second,size_t count){{
 size_t i=0;
{packets}
 for(;i<count;i++)first[i]={symbol}_lane(first[i],second[i]);
}}
"""


def synthesize(source_scales, lo, hi, *, relu=False, radii=(0, 1, 2, 4, 8)):
    """Try a fixed source-derived precision ladder, proving every candidate.

    A radius is a count of adjacent positive f32 encodings around the rounded
    real product of source scales. Search failure is a refusal, never a license
    to extrapolate from input samples or relax the source result.
    """
    import math
    import struct

    from merlin.compiler.lower.quant.requantization import f32

    source = derive(source_scales, lo, hi, relu)
    if type(radii) is not tuple or not radii or any(type(x) is not int or x < 0 or x > 65536 for x in radii):
        raise ValueError("finite explicit nonnegative ULP radii required")
    center = f32(math.prod(source["source_scales"]))
    if not math.isfinite(center) or center <= 0:
        return dict(accepted=False, refusal="source product has no finite positive f32 center")
    bits = struct.unpack("<I", struct.pack("<f", center))[0]
    refusals = []
    for radius in radii:
        if not 0 < bits - radius <= bits + radius < 0x7F800000:
            continue
        scales = [struct.unpack("<f", struct.pack("<I", bits + d))[0] for d in (-radius, radius)]
        certificate = prove(source["source_scales"], scales, lo, hi, relu=relu)
        if certificate["exact_pair_decoder"]:
            return dict(accepted=True, radius=radius, certificate=certificate)
        refusals.append(
            dict(
                radius=radius,
                enclosure_counterexamples=certificate["enclosure_counterexamples"][:1],
                decoder_counterexamples=certificate["decoder_counterexamples"][:1],
            )
        )
    return dict(accepted=False, refusal="no exact pair decoder in declared precision ladder", attempts=refusals)
