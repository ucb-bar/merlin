"""Build unchanged operator fixtures plus bounded current-default reuse controls.

TVM app modules must be selected explicitly in PYTHONPATH. This creates positive
and wrong-reference ELFs; it does not run a simulator or change compiler policy.
"""
import argparse
from pathlib import Path


def reuse_fixtures():
    import numpy as np
    from tvm import te
    from verify_device_ops import graph

    result = []
    for m, k, n in ((64, 64, 48), (1, 64, 256), (7, 19, 11)):
        a, b = te.placeholder((m, k), "int8", "a"), te.placeholder((k, n), "int8", "b")
        axis = te.reduce_axis((0, k), "k")
        c = te.compute((m, n), lambda i, j: te.sum(a[i, axis].astype("int32") *
                       b[axis, j].astype("int32"), axis=axis), name="matmul")
        x = (np.arange(m*k).reshape(m, k) % 16 - 8).astype("int8")
        y = ((np.arange(k*n).reshape(k, n)*7) % 16 - 8).astype("int8")
        expected = (x.astype("int64") @ y.astype("int64")).astype("int32")
        result.append((f"default_reuse_{m}_{k}_{n}", graph(te.create_prim_func([a,b,c]), 2),
                       [x, y], [expected]))
    return result


def main():
    from operator_contracts import fixtures as baseline_fixtures
    from verify_device_ops import fixtures as device_fixtures
    from verify_tensorir import prepare

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "adapter-receipt", "riscv-gcc"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    cases = list(baseline_fixtures()) + device_fixtures() + reuse_fixtures()
    assert len(cases) == 74 and len({case[0] for case in cases}) == 74
    args.cases = [case[0] for case in cases]
    args.timing_repeats, args.rtol, args.atol, args.libm = 0, 0., 0., True
    prepare(args, cases=cases, additional_bindings=[__file__])


if __name__ == "__main__":
    main()
