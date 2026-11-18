"""Complete joint-byte certificates for ordered binary32 affine arithmetic.

No predictor tuple is eligible when two source byte pairs require different
outputs. Backend implementations of both predictor contracts must be validated
independently; this proof neither assigns hardware costs nor permits aliasing.
"""

import hashlib
import math

import numpy as np

from merlin.compiler.lower.quant.affine_pair import _scales, _valid_c_identifier, predictor_table, source_table
from merlin.compiler.lower.quant.requantization import f32


def nearby_candidates(lhs_scale, rhs_scale, output_scale, *, max_denominator, relu=False):
    """Generate source-derived rational guesses; eligibility always requires a proof."""
    source, reciprocal = _scales(lhs_scale, rhs_scale, output_scale, relu)
    if type(max_denominator) is not int or not 1 <= max_denominator <= 32767:
        raise ValueError("explicit positive bounded denominator required")
    alpha, beta = source["lhs_scale"] * reciprocal, source["rhs_scale"] * reciprocal
    seen = set()
    for q in range(1, max_denominator + 1):
        for p in sorted({math.floor(alpha / beta * q), math.ceil(alpha / beta * q)}):
            if p <= 0 or 128 * (p + q) > (1 << 31) - 1:
                continue
            for scale in (beta / q, (alpha * p + beta * q) / (p * p + q * q)):
                key = (p, q, f32(scale), relu)
                if key not in seen:
                    seen.add(key)
                    yield dict(p=p, q=q, scale=key[2], relu=relu)


def derive_joint(lhs_scale, rhs_scale, output_scale, *, predictors, relu=False):
    """Prove all65,536 signed-byte pairs, returning precise collision witnesses."""
    source, reciprocal = _scales(lhs_scale, rhs_scale, output_scale, relu)
    if not isinstance(predictors, (list, tuple)) or len(predictors) != 2:
        raise ValueError("two explicit predictor contracts required")
    normalized, predictions = [], []
    for supplied in predictors:
        if (
            not isinstance(supplied, dict)
            or not {"p", "q", "scale"} <= set(supplied)
            or set(supplied) - {"p", "q", "scale", "relu"}
        ):
            raise ValueError("exact predictor coefficient/scale/ReLU contract required")
        row = dict(supplied)
        row.setdefault("relu", relu)
        prediction = predictor_table(**row).ravel()
        row["scale"] = f32(row["scale"])
        normalized.append(row)
        predictions.append(prediction)
    expected = source_table(**source).ravel()
    first, second = predictions
    keys = first.view(np.uint8).astype(np.uint16) * 256 + second.view(np.uint8)
    low, high = np.full(65536, 128, np.int16), np.full(65536, -129, np.int16)
    np.minimum.at(low, keys, expected)
    np.maximum.at(high, keys, expected)
    conflicts = np.flatnonzero(low < high)
    witnesses = []
    for key in conflicts[:8]:
        positions = np.flatnonzero(keys == key)
        pair = [int(positions[np.flatnonzero(expected[positions] == value)[0]]) for value in (low[key], high[key])]
        witnesses.append(
            dict(
                predictions=[int(first[pair[0]]), int(second[pair[0]])],
                source_pairs=[
                    dict(lhs=index // 256 - 128, rhs=index % 256 - 128, output=int(expected[index])) for index in pair
                ],
            )
        )
    equal = first == second
    identity = bool(np.all(expected[equal] == first[equal]))
    delta = int(np.abs(first.astype(np.int16) - second.astype(np.int16)).max())
    decoder, form = None, None
    if not len(conflicts):
        if identity and delta <= 1:
            form = "adjacent_byte_tuple"
            decoder = np.zeros(512, np.int8)
            different = ~equal
            compact = (second[different] > first[different]).astype(np.uint16) * 256 + first[different].view(np.uint8)
            decoder[compact] = expected[different]
            decoded = first.copy()
            decoded[different] = decoder[compact]
        else:
            form = "full_byte_tuple"
            decoder = np.zeros(65536, np.int8)
            decoder[keys] = expected
            decoded = decoder[keys]
        if not np.array_equal(decoded, expected):
            raise ValueError("complete joint decoder reconstruction failed")
    return dict(
        schema="quantized_affine_joint_certificate_v1",
        source=source,
        reciprocal=reciprocal,
        predictors=normalized,
        pairs=65536,
        decoder_exact_for_all_pairs=not len(conflicts),
        conflicting_tuples=int(len(conflicts)),
        collision_witnesses=witnesses,
        reachable_tuples=int(np.sum(low <= high)),
        equal_tuple_identity=identity,
        maximum_predictor_byte_delta=delta,
        different_prediction_pairs=int(np.sum(~equal)),
        individual_mismatched_pairs=[int(np.sum(v != expected)) for v in predictions],
        source_table_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        predictor_table_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in predictions],
        decoder_format=form,
        decoder_hex=decoder.tobytes().hex() if decoder is not None else None,
        decoder_sha256=hashlib.sha256(decoder.tobytes()).hexdigest() if decoder is not None else None,
        backend_predictor_implementation="UNKNOWN; each complete target pair domain must be independently closed",
        runtime_different_prediction_rate="UNKNOWN; complete-domain counts are not runtime frequencies",
        source_arithmetic="Separate binary32 RN-even products/add/optional ReLU/reciprocal product; RNE/clamp",
        predictor_arithmetic="Exact signed-i32 affine sum, binary32 conversion/product RN-even, RNE/clamp",
    )


def derive_first_output_guard(proof):
    """Prove a conservative byte predicate covering every inexact first output.

    Predicate rejection proves that the first output already equals the source.
    It does not permit changing either predictor, aliasing, or assuming an
    observed runtime distribution. The predicate may accept extra byte values.
    """
    if proof != derive_joint(**proof["source"], predictors=proof["predictors"]):
        raise ValueError("unchanged complete joint certificate required")
    if not proof["decoder_exact_for_all_pairs"]:
        raise ValueError("conflicting predictor tuple refuses exact decoder")
    first = predictor_table(**proof["predictors"][0]).ravel()
    expected = source_table(**proof["source"]).ravel()
    wrong = first[first != expected].view(np.uint8)
    values = sorted(map(int, np.unique(wrong)))
    ones, zeros = 255, 255
    for value in values:
        ones &= value
        zeros &= value ^ 255
    mask = ones | zeros if values else 0
    accepted = [byte for byte in range(256) if byte & mask == ones] if values else []
    if not set(values) <= set(accepted):
        raise ValueError("first-output guard excludes an inexact predictor value")
    return dict(
        schema="joint_affine_first_output_guard_v1",
        source_table_sha256=proof["source_table_sha256"],
        first_predictor_table_sha256=proof["predictor_table_sha256"][0],
        pairs=65536,
        first_exact=not values,
        inexact_first_raw_byte_values=values,
        mask=mask,
        value=ones if values else 0,
        accepted_raw_byte_values=accepted,
        scope=(
            "Predicate rejection proves source identity for every original signed-byte pair; "
            "no runtime rate or target cost inferred"
        ),
    )


def emit_joint_decoder(proof, symbol, *, packed=True, first_output_guard=False):
    """Overwrite the first prediction using two independently qualified byte buffers.

    Caller obligations: both buffers contain the certified predictors of the
    same immutable source inputs and have n accessible bytes; they do not
    overlap. Second remains read-only/alive throughout. Aligned word equality
    is endian-independent; scalar fallback/tails retain exact byte behavior.
    """
    if not _valid_c_identifier(symbol) or type(packed) is not bool or type(first_output_guard) is not bool:
        raise ValueError("valid C identifier and explicit packed boolean required")
    if proof != derive_joint(**proof["source"], predictors=proof["predictors"]):
        raise ValueError("unchanged complete joint certificate required")
    if not proof["decoder_exact_for_all_pairs"]:
        raise ValueError("conflicting predictor tuple refuses exact decoder")
    table = bytes.fromhex(proof["decoder_hex"])
    values = ",".join(map(str, np.frombuffer(table, dtype=np.int8)))
    adjacent = proof["decoder_format"] == "adjacent_byte_tuple"
    key = "((unsigned)(b>a)<<8)|(unsigned)(uint8_t)a" if adjacent else "((unsigned)(uint8_t)a<<8)|(unsigned)(uint8_t)b"
    skip = "if(a==b)return;" if proof["equal_tuple_identity"] else ""
    guard = derive_first_output_guard(proof) if first_output_guard else None
    if guard is not None and guard["first_exact"]:
        return f"""#include <stdint.h>
#include <stddef.h>
void {symbol}(int8_t *first,const int8_t *second,size_t n){{
 (void)first;(void)second;(void)n;
}}
"""
    if guard is not None and not guard["mask"]:
        guard = None
    byte_load = "int a=first[i],b=second[i];"
    if guard is not None:
        byte_load = f"int a=first[i];if(((uint8_t)a&{guard['mask']})!={guard['value']})return;int b=second[i];"
    code = f"""#include <stdint.h>
#include <stddef.h>
static const int8_t {symbol}_table[{len(table)}] __attribute__((aligned(64)))=
{{{values}}};
static inline __attribute__((always_inline)) void {symbol}_byte(int8_t *first,const int8_t *second,size_t i){{
 {byte_load}{skip}
 first[i]={symbol}_table[{key}];
}}
"""
    loop = ""
    if packed and proof["equal_tuple_identity"]:
        word_load = f"uint64_t a=*(const {symbol}_word*)(first+i),b=*(const {symbol}_word*)(second+i);"
        if guard is not None:
            mask = guard["mask"] * 0x0101010101010101
            value = guard["value"] * 0x0101010101010101
            word_load = f"""uint64_t a=*(const {symbol}_word*)(first+i);
  uint64_t d=(a&UINT64_C(0x{mask:016x}))^UINT64_C(0x{value:016x});
  if(!((d-UINT64_C(0x0101010101010101))&~d&UINT64_C(0x8080808080808080)))continue;
  uint64_t b=*(const {symbol}_word*)(second+i);"""
        loop = f"""
typedef uint64_t {symbol}_word __attribute__((may_alias));
void {symbol}(int8_t *first,const int8_t *second,size_t n){{
 size_t i=0;
 if(!(((uintptr_t)first|(uintptr_t)second)&7))for(;n-i>=8;i+=8){{
  {word_load}
  if(a!=b)for(size_t lane=0;lane<8;lane++){symbol}_byte(first,second,i+lane);
 }}
 for(;i<n;i++){symbol}_byte(first,second,i);
}}
"""
    else:
        loop = f"""void {symbol}(int8_t *first,const int8_t *second,size_t n){{
 for(size_t i=0;i<n;i++){symbol}_byte(first,second,i);
}}
"""
    return code + loop
