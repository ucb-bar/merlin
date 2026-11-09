"""Independent ordered integer reference for the pinned public parameter chain.

Dimensions come from the selected complete typed interface. This is a fixed-input
value/prefix check, never a universal accumulator or reassociation permission.
"""

import json
from pathlib import Path

import numpy as np
from xdsl.dialects.builtin import ArrayAttr, StringAttr

from merlin.capture.safetensors import load_header
from merlin.common.digest import sha256_file
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_digest

PUBLIC_FILES = (
    "case.json",
    "capsule.pytorch.py",
    "capsule.interface.mlir",
    "capsule.weights.safetensors",
    "capsule.weights.safetensors.manifest.json",
    "frontend-trace.json",
    "independent-reference.npz",
)
POLICY = {
    "schema": "merlin.numeric_policy.per_output.v1",
    "compare": "per_output",
    "outputs": {"wide": {"compare": "exact_int", "dtype": "i32"}, "state": {"compare": "exact_int", "dtype": "i8"}},
}


def semantic_identity(source, *, weights_path):
    """Ignore two inert trace keys; preserve known semantic/source-boundary fields.

    The sole path-valued weight binding is normalized to its checked file digest,
    not discarded. Unknown provenance may carry executable meaning and refuses.
    """
    module = parse_mlir_text(source)
    module.verify()
    weights_path = Path(weights_path)
    assert weights_path.is_file() and not weights_path.is_symlink()
    bound = 0
    for op in module.walk():
        for attributes in (op.attributes, op.properties):
            for name, value in tuple(attributes.items()):
                if not name.startswith("prov."):
                    continue
                if name in {"prov.region_id", "prov.fqn"}:
                    assert type(value) is StringAttr
                    del attributes[name]
                elif name == "prov.weights_file":
                    assert op is module and type(value) is StringAttr
                    assert value.data in (str(weights_path), weights_path.name)
                    attributes[name] = StringAttr("sha256:" + sha256_file(weights_path))
                    bound += 1
                elif name in {
                    "prov.level",
                    "prov.family",
                    "prov._pattern_hint",
                    "prov.op",
                    "prov.aten",
                    "prov.trace_role",
                    "prov.orig_dtype",
                }:
                    assert type(value) is StringAttr  # Preserved, including the exact value.
                elif name in {"prov.source_node_ids", "prov.origin_node_ids"}:
                    assert type(value) is ArrayAttr and all(type(v) is StringAttr for v in value)
                else:
                    raise ValueError("unknown provenance is not inert trace attribution: " + name)
    assert bound == 1, "one explicit checked weight-source binding is required"
    return ir_digest(module)


def geometry(source):
    module = parse_mlir_text(source)
    module.verify()
    entries = tuple(module.body.block.ops)
    assert len(entries) == 1 and entries[0].name == "func.func" and entries[0].sym_name.data == "forward"
    inputs, outputs = entries[0].function_type.inputs, entries[0].function_type.outputs
    assert len(inputs) == 4 and len(outputs) == 2, "complete three-parameter/one-input/two-output ABI required"
    shapes = tuple(tuple(t.get_shape()) for t in inputs)
    n = shapes[-1][-1]
    assert type(n) is int and n > 0 and shapes == ((n, n),) * 3 + ((1, n),)
    assert tuple(map(str, inputs)) == (f"tensor<{n}x{n}xi8>",) * 3 + (f"tensor<1x{n}xi8>",)
    assert tuple(map(str, outputs)) == (f"tensor<1x{n}xi32>", f"tensor<1x{n}xi8>")
    assert 3 * n * n == 3 << 20, "this regression selects the complete public 3 MiB fixture"
    return n


def signed_weights(path, n):
    header, offset = load_header(path)
    assert set(header) - {"__metadata__"} == {"w0", "w1", "w2"}
    blob, arrays, spans = Path(path).read_bytes(), {}, []
    for name in ("w0", "w1", "w2"):
        row = header[name]
        assert row["dtype"] == "I8" and row["shape"] == [n, n]
        lo, hi = row["data_offsets"]
        assert type(lo) is int and type(hi) is int and 0 <= lo < hi and hi - lo == n * n
        assert offset + hi <= len(blob)
        arrays[name] = np.frombuffer(blob[offset + lo : offset + hi], dtype=np.int8).reshape(n, n)
        spans.append((lo, hi))
    assert sorted(spans) == [(i * n * n, (i + 1) * n * n) for i in range(3)]
    assert len(blob) - offset == 3 * n * n
    return arrays


def ordered_reference(x, weights):
    assert x.dtype == np.int8 and x.ndim == 2 and x.shape[0] == 1
    n, state, prefixes = x.shape[1], x.copy(), []
    assert set(weights) == {"w0", "w1", "w2"}
    for name in ("w0", "w1", "w2"):
        weight = weights[name]
        assert weight.dtype == np.int8 and weight.shape == (n, n)
        acc, low, high = np.zeros(n, dtype=np.int64), 0, 0
        for k in range(n):
            acc += int(state[0, k]) * weight[k].astype(np.int64)
            low, high = min(low, int(acc.min())), max(high, int(acc.max()))
            assert -(1 << 31) <= low and high < 1 << 31
        wide = acc.reshape(1, n).astype(np.int32)
        added = wide.astype(np.int64) + 4
        assert np.all(added >= -(1 << 31)) and np.all(added < 1 << 31)
        state = np.clip(np.floor_divide(added, 8), -128, 127).astype(np.int8)
        prefixes.append(
            dict(
                weight=name,
                minimum=low,
                maximum=high,
                prefix_pairs=n * n,
                fixed_input_fits_signed20=-(1 << 19) <= low and high < 1 << 19,
            )
        )
    return (wide, state), prefixes


def exact_outputs(actual, expected):
    assert len(actual) == len(expected) == 2, "complete ordered output roster required"
    for a, b in zip(actual, expected, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def exact_named_outputs(arrays, expected):
    """NPZ storage order is independent of the already checked public ABI order."""
    assert len(arrays.files) == 2 and set(arrays.files) == {"wide", "state"}
    exact_outputs((arrays["wide"], arrays["state"]), expected)


def captured_values(captured, x, wanted):
    assert len(captured.inputs) == 1 and len(captured.golden) == 2
    given = np.asarray(captured.inputs[0])
    assert given.shape == x.shape and given.dtype.kind in "iu" and np.array_equal(given, x)
    for raw, expected in zip(captured.golden, wanted, strict=True):
        raw = np.asarray(raw)
        assert raw.dtype.kind in "iu" and raw.shape == expected.shape and np.array_equal(raw, expected)


def validate_weight_manifest(manifest, n):
    """Check every original argument, including parameter storage type/extents."""
    assert type(n) is int and n > 0
    expected = {
        **{str(i): {"kind": "param", "weight": f"w{i}", "dtype": "int8", "shape": [n, n]} for i in range(3)},
        "3": {"kind": "input", "name": "x"},
    }
    assert type(manifest) is dict and manifest == expected, "complete original parameter/input ABI required"
    assert all(type(d) is int for i in range(3) for d in manifest[str(i)]["shape"])


def public_reference(paths):
    source = Path(paths["capsule.interface.mlir"]["path"]).read_text()
    n = geometry(source)
    case = json.loads(Path(paths["case.json"]["path"]).read_bytes())
    assert case["schema"] == "gemmini.public_integer_fixture.v1" and case["numeric_policy"] == POLICY
    assert [(r["ordinal"], r["shape"], r["dtype"]) for r in case["inputs"]] == [(0, [1, n], "i8")]
    assert [(r["ordinal"], r["name"], r["shape"], r["dtype"]) for r in case["outputs"]] == [
        (0, "wide", [1, n], "i32"),
        (1, "state", [1, n], "i8"),
    ]
    weights = signed_weights(Path(paths["capsule.weights.safetensors"]["path"]), n)
    with np.load(paths["independent-reference.npz"]["path"], allow_pickle=False) as values:
        assert set(values.files) == {"input_0", "output_0", "output_1"}
        x, expected = values["input_0"].copy(), (values["output_0"].copy(), values["output_1"].copy())
    assert x.dtype == np.int8 and x.shape == (1, n)
    assert [(a.shape, a.dtype) for a in expected] == [((1, n), np.dtype("int32")), ((1, n), np.dtype("int8"))]
    computed, prefixes = ordered_reference(x, weights)
    exact_outputs(computed, expected)
    assert all(p["fixed_input_fits_signed20"] for p in prefixes), "no fixed-input signed20 prefix proof"
    return source, x, weights, expected, prefixes
