"""Exhaust every reduced operand pair with a NaN on each CPU loop path.

The strided loop uses the same scalar functor as the contiguous scalar tail;
the contiguous loop exercises the AVX2 vector functor. No model is consumed.
"""


def generate():
    import hashlib

    import numpy as np
    import torch

    from merlin.targetgen.native_ieee_certificate import digest

    rows = []
    words = np.arange(65536, dtype=np.uint16)
    nans = words[(words & 0x7FFF) > 0x7C00]
    for operation, fn in (("add", torch.add), ("subtract", torch.sub)):
        for mode in ("scalar", "vector"):
            sha, count = hashlib.sha256(), 0
            # A nonunit input stride disables vectorized_loop; unlike probing
            # small tensors this covers the entire scalar domain in bounded time.
            step = 2 if mode == "scalar" else 1
            a = torch.empty(65536 * step, dtype=torch.float16)[::step]
            b = torch.empty_like(a, memory_format=torch.contiguous_format)
            all_words = torch.from_numpy(words).view(torch.float16)
            for nan_on_left in (True, False):
                for nan in nans:
                    a.copy_(
                        torch.tensor([int(nan)], dtype=torch.uint16).view(torch.float16) if nan_on_left else all_words
                    )
                    b.copy_(
                        all_words if nan_on_left else torch.tensor([int(nan)], dtype=torch.uint16).view(torch.float16)
                    )
                    actual = fn(a, b).view(torch.uint16).numpy()
                    lhs, rhs = (np.full_like(words, nan), words) if nan_on_left else (words, np.full_like(words, nan))
                    left_nan, right_nan = (lhs & 0x7FFF) > 0x7C00, (rhs & 0x7FFF) > 0x7C00
                    chosen = np.where(right_nan, rhs, lhs) if mode == "vector" else np.where(left_nan, lhs, rhs)
                    expected = chosen | 0x200
                    if not np.array_equal(actual, expected):
                        i = int(np.flatnonzero(actual != expected)[0])
                        raise ValueError(
                            f"NaN priority refused: {operation}/{mode}: {int(lhs[i]):04x},{int(rhs[i]):04x}"
                        )
                    sha.update(actual.tobytes())
                    count += len(words)
            form = dict(
                dtype="f16",
                operation=operation,
                mode=mode,
                domain="all_NaN_x_storage_words_both_operand_orders",
                priority="right" if mode == "vector" else "left",
                quiet_mask=0x200,
            )
            rows.append(dict(form=form, key=digest(form), count=count, output_sha256=sha.hexdigest(), status="proven"))
    return rows
