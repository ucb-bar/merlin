"""Full reduced-storage operand domains against each IEEE special/edge class.

Finite interior closure is the precision/RNE theorem, not extrapolation from
these checks. Distinct competing half multiply NaNs are explicitly unqualified.
"""


def arithmetic_word(a, b, token, operation, mode):
    from merlin.targetgen.native_ieee_certificate import FORMATS, dyadic_word

    fraction, exponent, bias = FORMATS[token]
    sign, infinity = 1 << (fraction + exponent), ((1 << exponent) - 1) << fraction
    ma, mb = a & (sign - 1), b & (sign - 1)
    na, nb = ma > infinity, mb > infinity
    nan = (0x7FC0 if mode == "scalar" else 0xFFFF) if token == "bf16" else infinity | sign | 1 << (fraction - 1)
    if na or nb:
        if token == "bf16":
            return nan
        chosen = b if nb and (not na or (operation in {"add", "subtract"} and mode == "vector")) else a
        return chosen | 1 << (fraction - 1)
    sa, sb = bool(a & sign), bool(b & sign)
    if operation == "multiply":
        if (ma == infinity and mb == 0) or (mb == infinity and ma == 0):
            return nan
        if ma == infinity or mb == infinity:
            return (sign if sa != sb else 0) | infinity
    else:
        if operation == "subtract":
            sb = not sb
        if ma == infinity and mb == infinity and sa != sb:
            return nan
        if ma == infinity or mb == infinity:
            return (sign if (sa if ma == infinity else sb) else 0) | infinity

    def decode(word, negative):
        field = word >> fraction & ((1 << exponent) - 1)
        mantissa = (word & ((1 << fraction) - 1)) | (1 << fraction if field else 0)
        return -mantissa if negative else mantissa, max(field, 1) - bias - fraction

    x, ex = decode(a, sa)
    y, ey = decode(b, sb)
    if operation == "multiply":
        return dyadic_word(x * y, ex + ey, token, negative_zero=sa != sb)
    power = min(ex, ey)
    return dyadic_word(
        (x << (ex - power)) + (y << (ey - power)), power, token, negative_zero=sa and sb and ma == mb == 0
    )


def generate():
    import hashlib

    import numpy as np
    import torch

    from merlin.targetgen.native_ieee_certificate import FORMATS, digest

    rows = []
    for token, dtype in (("f16", torch.float16), ("bf16", torch.bfloat16)):
        f, e, bias = FORMATS[token]
        sign, infinity = 1 << (f + e), ((1 << e) - 1) << f
        classes = [
            0,
            sign,
            1,
            sign | 1,
            (bias << f),
            sign | (bias << f),
            infinity - 1,
            infinity,
            sign | infinity,
            infinity | 1,
            sign | infinity | (1 << (f - 1)) | 3,
        ]
        words = np.arange(65536, dtype=np.uint16)
        left = torch.from_numpy(words).view(dtype)
        scalars = [left[i : i + 1] for i in range(65536)]
        for operation, fn in (("add", torch.add), ("subtract", torch.sub), ("multiply", torch.mul)):
            for mode in ("scalar", "vector"):
                sha, count = hashlib.sha256(), 0
                refused = None
                for rhs in classes:
                    right = torch.tensor([rhs], dtype=torch.uint16).view(dtype)
                    actual = (
                        fn(left, right.expand_as(left).clone()).view(torch.uint16).tolist()
                        if mode == "vector"
                        else [fn(v, right).view(torch.uint16).item() for v in scalars]
                    )
                    expected = [arithmetic_word(int(lhs), rhs, token, operation, mode) for lhs in words]
                    for i, (a, b) in enumerate(zip(actual, expected)):
                        # There is no declared universal half-multiply priority
                        # for two distinct NaN words. Admission rejects that form.
                        if (
                            token == "f16"
                            and operation == "multiply"
                            and (i & (sign - 1)) > infinity
                            and (rhs & (sign - 1)) > infinity
                            and i != rhs
                        ):
                            continue
                        if a != b:
                            refused = dict(lhs=i, rhs=rhs, cpu=a, proposed=b)
                            break
                    sha.update(np.array(actual, dtype=np.uint16).tobytes())
                    count += 65536
                    if refused:
                        break
                form = dict(
                    dtype=token,
                    operation=operation,
                    mode=mode,
                    domain="all_storage_words_x_complete_IEEE_special_and_edge_classes",
                    constraint="no_distinct_competing_NaNs" if token == "f16" and operation == "multiply" else "none",
                )
                rows.append(
                    dict(
                        form=form,
                        key=digest(form),
                        count=count,
                        output_sha256=sha.hexdigest(),
                        status="refused" if refused else "proven",
                        counterexample=refused,
                    )
                )
    return rows
