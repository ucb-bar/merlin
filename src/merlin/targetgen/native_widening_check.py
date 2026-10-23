"""Exact binary32 embedding in binary64: complete exceptional and subnormal words."""


def generate():
    import hashlib

    import numpy as np
    import torch

    from merlin.targetgen.native_ieee_certificate import digest
    from merlin.targetgen.native_ieee_ir_check import conversion_function

    emitted = conversion_function("f32", "f64", "payload_quiet")
    rows = []
    for mode in ("scalar", "vector"):
        sha, count = hashlib.sha256(), 0
        # Every subnormal and NaN payload, both signs, plus both endpoints and
        # central significands of every normal binade. Normal interiors follow
        # the exact 24-bit -> 53-bit embedding theorem, not sampled agreement.
        normal = np.array(
            [
                (s << 31) | (e << 23) | m
                for s in (0, 1)
                for e in range(1, 255)
                for m in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF)
            ],
            dtype=np.uint32,
        )
        chunks = [normal]
        for sign in (0, 0x80000000):
            for field in (0, 0x7F800000):
                chunks.extend(
                    np.arange(i, min(i + 262144, 1 << 23), dtype=np.uint32) | np.uint32(sign | field)
                    for i in range(0, 1 << 23, 262144)
                )
        for words in chunks:
            tensor = torch.from_numpy(words).view(torch.float32)
            if mode == "scalar":
                strided = torch.empty(2 * len(words), dtype=torch.float32)[::2]
                strided.copy_(tensor)
                tensor = strided
            actual = tensor.double().view(torch.uint64).numpy()
            expected = emitted(words)
            if not np.array_equal(actual, expected):
                i = int(np.flatnonzero(actual != expected)[0])
                raise ValueError(f"binary64 widening refused: {mode}/{int(words[i]):08x}")
            sha.update(actual.tobytes())
            count += len(words)
        form = dict(
            source="f32",
            target="f64",
            mode=mode,
            domain="all_subnormal_and_special_words; exact_normal_embedding_24_le_53",
            nan_policy="payload_quiet",
            rounding="exact",
        )
        rows.append(
            dict(
                form=form,
                key=digest(form),
                count=count,
                output_sha256=sha.hexdigest(),
                executable_ir_sha256=hashlib.sha256(emitted.source.encode()).hexdigest(),
                status="proven",
            )
        )
    return rows
