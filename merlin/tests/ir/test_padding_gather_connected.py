"""Public padded convolution and an ordered, complete-output scalar reference."""

from pathlib import Path

import numpy as np


def source():
    return Path(__file__).with_name("padding_gather_connected.mlir").read_text()


def inputs(epoch, state):
    image = np.asarray([-128, -1, 0, 1, 126, 127, -127, 3, 9, -7, 11, -13], dtype=np.int16)
    image = ((image + epoch * 17 + 128) % 256 - 128).astype(np.int8).reshape(1, 1, 3, 4)
    weight = np.asarray([((index * 37 + epoch * 11) % 255) - 127 for index in range(18)], dtype=np.int8)
    return image, weight.reshape(2, 9), state.copy()


def wrap32(value):
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


def reference(image, weight, state):
    """Read the original image with explicit zero edges, never a packed operand."""
    assert image.shape == (1, 1, 3, 4) and image.dtype == np.int8
    assert weight.shape == (2, 9) and weight.dtype == np.int8
    assert state.shape == (2, 12) and state.dtype == np.int32
    columns = np.zeros((1, 3, 3, 1, 3, 4), dtype=np.int8)
    wide = np.zeros((2, 12), dtype=np.int32)
    for h in range(3):
        for w in range(4):
            for kh in range(3):
                for kw in range(3):
                    ih, iw = h + kh - 1, w + kw - 1
                    value = int(image[0, 0, ih, iw]) if 0 <= ih < 3 and 0 <= iw < 4 else 0
                    columns[0, kh, kw, 0, h, w] = value
            for f in range(2):
                acc = 0
                for kh in range(3):
                    for kw in range(3):
                        ih, iw = h + kh - 1, w + kw - 1
                        value = int(image[0, 0, ih, iw]) if 0 <= ih < 3 and 0 <= iw < 4 else 0
                        acc = wrap32(acc + int(weight[f, kh * 3 + kw]) * value)
                wide[f, h * 4 + w] = acc
    next_state = np.asarray(
        [[wrap32(int(wide[f, col]) + int(state[f, col])) for col in range(12)] for f in range(2)], dtype=np.int32
    )
    return next_state, columns, wide


def test_public_padding_reference_preserves_edges_and_state():
    image = np.ones((1, 1, 3, 4), dtype=np.int8)
    weight = np.ones((2, 9), dtype=np.int8)
    state = np.arange(24, dtype=np.int32).reshape(2, 12)
    next_state, columns, wide = reference(image, weight, state)
    expected = np.asarray([4, 6, 6, 4, 6, 9, 9, 6, 4, 6, 6, 4], dtype=np.int32)
    np.testing.assert_array_equal(wide, np.stack((expected, expected)))
    np.testing.assert_array_equal(next_state, wide + state)
    assert np.count_nonzero(columns) == 70 and columns.size == 108


def test_public_padding_source_verifies_and_states_one_complete_contraction():
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry

    module = parse_mlir_text(source())
    module.verify()
    contractions = [
        operation
        for operation in module.walk()
        if type(operation) is GenericOp
        and any(it.data is IteratorType.REDUCTION for it in operation.get_iterator_types())
    ]
    assert len(contractions) == 1
    assert static_zero_integer_matmul_geometry(contractions[0]) == (2, 9, 12)


def test_public_padding_native_baseline_preserves_complete_outputs_and_state(tmp_path):
    import hashlib
    import json

    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    (tmp_path / "source.mlir").write_text(source())
    compiled = lower_model(source(), tmp_path / "native", targets=("host",), vectorize=False, parallel=False)
    assert compiled.host_so is not None
    model = HostModel.load(
        str(compiled.host_so), image_policy=PrivateHostImagePolicy(compiled.host_so.parent.resolve())
    )
    state = np.arange(-12, 12, dtype=np.int32).reshape(2, 12)
    observations = []
    for epoch in range(3):
        arguments = inputs(epoch, state)
        originals = [value.copy() for value in arguments]
        expected = reference(*originals)
        guards = [np.full(value.size + 32, 83, dtype=value.dtype) for value in expected]
        outputs = [buffer[16:-16].reshape(value.shape) for buffer, value in zip(guards, expected, strict=True)]
        np.savez(tmp_path / f"inputs-{epoch}.npz", **{f"in{i}": value for i, value in enumerate(arguments)})
        model([(value.ctypes.data, value.shape) for value in (*arguments, *outputs)])
        actual_path = tmp_path / f"actual-{epoch}.npz"
        np.savez(
            actual_path,
            **dict(zip(("next_state", "columns", "wide"), outputs, strict=True)),
            **{f"guard{i}": value for i, value in enumerate(guards)},
            **{f"after_input{i}": value for i, value in enumerate(arguments)},
        )
        for actual, independent in zip(outputs, expected, strict=True):
            assert actual.tobytes() == independent.tobytes()
        assert all(actual.tobytes() == before.tobytes() for actual, before in zip(arguments, originals, strict=True))
        assert all(np.all(value[:16] == 83) and np.all(value[-16:] == 83) for value in guards)
        observations.append({"path": str(actual_path), "sha256": hashlib.sha256(actual_path.read_bytes()).hexdigest()})
        state = outputs[0].copy()
    artifacts = (tmp_path / "source.mlir", compiled.ll_path, compiled.host_so, compiled.host_so.with_suffix(".o"))
    (tmp_path / "baseline.json").write_text(
        json.dumps(
            {
                "scope": "ordinary native host baseline; complete outputs and external state; no target qualification",
                "artifacts": [
                    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in artifacts
                ],
                "actual_outputs": observations,
                "invocations": [str(path) for path in sorted(compiled.workdir.rglob("invocation.json"))],
            },
            indent=2,
        )
        + "\n"
    )
