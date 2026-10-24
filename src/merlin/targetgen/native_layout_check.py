"""Full storage-domain checks of static TensorIterator row/readout classes."""


def generate():
    import hashlib

    import numpy as np
    import torch

    from merlin.llvmlower.native_cpu_layout import profile
    from merlin.targetgen.native_ieee_certificate import digest

    # Each is an instance of the row/coalescing theorem, including vector/tail
    # seams, offsets, broadcast, dimension permutation and non-unit steps.
    layouts = [
        dict(shape=[2, 3, 65], stride=[195, 65, 1]),
        dict(shape=[2, 3, 65], stride=[390, 130, 2]),
        dict(shape=[2, 3, 65], stride=[234, 78, 1]),
        dict(shape=[2, 3, 65], stride=[0, 65, 1]),
        dict(shape=[1, 2, 3, 65], stride=[390, 1, 130, 2]),
    ]
    rows = []
    for layout in layouts:
        p = profile(layout, block=32, grain=32768)
        indices = np.indices(layout["shape"])
        addresses = sum(i * s for i, s in zip(indices, layout["stride"]))
        unique = np.unique(addresses)
        position = np.zeros(layout["shape"], dtype=np.int64)
        factor = 1
        for axis in p["axes"]:
            position += indices[axis] * factor
            factor *= p["shape"][axis]
        vector = position % max(1, p["row"]) < p["vector_prefix"]
        for token, dtype in [("f16", torch.float16), ("bf16", torch.bfloat16)]:
            sha, count = hashlib.sha256(), 0
            base = torch.empty(int(addresses.max()) + 8, dtype=torch.uint16)
            value = base.view(dtype).as_strided(layout["shape"], layout["stride"], 3)
            for start in range(0, 65536, len(unique)):
                words = np.arange(start, start + len(unique), dtype=np.uint32).astype(np.uint16)
                base.numpy()[unique + 3] = words
                raw = base.numpy()[addresses + 3]
                infinity = 0x7C00 if token == "f16" else 0x7F80
                nan = (raw & 0x7FFF) > infinity
                for family, fn in [("sign_change", torch.neg), ("positive_zero_threshold", torch.relu)]:
                    actual = fn(value).contiguous().view(torch.uint16).numpy()
                    if family == "sign_change":
                        expected = raw ^ 0x8000
                        expected = np.where(nan & ~vector, (expected | 0x200) if token == "f16" else 0x7FC0, expected)
                    else:
                        expected = np.where((raw & 0x8000 != 0) & ((raw & 0x7FFF) != 0) & ~nan, 0, raw)
                        expected = np.where(nan & vector, (raw | 0x200) if token == "f16" else 0xFFFF, expected)
                    if not np.array_equal(actual, expected):
                        i = tuple(np.argwhere(actual != expected)[0])
                        raise ValueError(
                            f"CPU layout certificate refused: {layout}/{token}/{family}/{i}: "
                            f"{actual[i]:04x} != {expected[i]:04x}"
                        )
                    sha.update(actual.tobytes())
                    count += actual.size
            form = dict(layout=layout, dtype=token, domain="all_storage_words_each_static_row_class", profile=p)
            rows.append(dict(form=form, key=digest(form), count=count, status="proven", output_sha256=sha.hexdigest()))
    return rows


def generate_joint():
    """Every bf16 NaN payload at every vector/tail position in joint rows."""
    import hashlib

    import numpy as np
    import torch

    from merlin.llvmlower.native_cpu_layout import joint_profile
    from merlin.targetgen.native_ieee_certificate import digest

    layouts = [
        ([2, 1, 65], [65, 65, 1], [1, 3, 1], [3, 1, 1]),
        ([2, 1, 65], [130, 130, 2], [1, 3, 1], [6, 2, 2]),
        ([1, 2, 3, 65], [390, 1, 130, 2], [1, 2, 3, 65], [390, 195, 65, 1]),
    ]
    rows = []
    for ash, ast, bsh, bst in layouts:
        shape = list(torch.broadcast_shapes(ash, bsh))
        declared = [dict(shape=ash, stride=ast), dict(shape=bsh, stride=bst)]
        p = joint_profile(declared, shape, block=32, grain=32768)
        indices, position, factor = np.indices(shape), np.zeros(shape, dtype=np.int64), 1
        for axis in p["axes"]:
            position += indices[axis] * factor
            factor *= shape[axis]
        expected = np.where(position % p["row"] < p["vector_prefix"], 0xFFFF, 0x7FC0)
        tensors = [
            torch.empty(1 + sum((n - 1) * s for n, s in zip(sh, st)), dtype=torch.bfloat16).as_strided(sh, st)
            for sh, st in [(ash, ast), (bsh, bst)]
        ]
        sha, count = hashlib.sha256(), 0
        for word in range(65536):
            if word & 0x7FFF <= 0x7F80:
                continue
            # fill through the storage carrier so zero-stride views remain legal.
            tensors[0].view(torch.int16).fill_(word if word < 32768 else word - 65536)
            tensors[1].fill_(1)
            actual = torch.add(*tensors).contiguous().view(torch.uint16).numpy()
            if not np.array_equal(actual, expected):
                raise ValueError("joint layout NaN readout differs from static row certificate")
            sha.update(actual.tobytes())
            count += actual.size
        form = dict(dtype="bf16", layouts=declared, profile=p, domain="all_NaN_storage_words_at_every_row_position")
        rows.append(dict(form=form, key=digest(form), count=count, status="proven", output_sha256=sha.hexdigest()))
    return rows
