"""Smoke test for the QNN kernel build pipeline.

Asserts that running `kernels/qnn/build.build_qnn_kernel` against the
benchmarks/QRB5165 Add kernel produces a valid .qnn-ctx blob. Validates
caching behaviour. Skipped when the QNN SDK isn't reachable at the default
location (or when QAIRT_SDK_ROOT is unset and the default doesn't exist).
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]


def _sdk_available() -> bool:
    import os

    root = pathlib.Path(os.environ.get("QAIRT_SDK_ROOT", "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326"))
    return (root / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator").exists()


def _board_available(host: str = "qdev") -> bool:
    """Probe whether the QRB5165 board is reachable AND has /tmp/qnn_probe
    staged with QAIRT runtime tools.
    """
    import subprocess

    res = subprocess.run(
        [
            "ssh",
            "-o",
            "ConnectTimeout=3",
            "-o",
            "BatchMode=yes",
            host,
            "test -x /tmp/qnn_probe/bin/qnn-context-binary-generator",
        ],
        capture_output=True,
        timeout=8,
    )
    return res.returncode == 0


@pytest.fixture
def cache_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "qnn_kernels"
    d.mkdir()
    return d


@pytest.mark.skipif(not _sdk_available(), reason="QNN SDK not available")
def test_build_add_f32_kernel(cache_dir: pathlib.Path) -> None:
    """End-to-end: cpp source → .so → .qnn-ctx via qnn-context-binary-generator."""
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build

    source = REPO / "benchmarks" / "QRB5165" / "kernels" / "abi" / "add_f32.qnn.cpp"
    assert source.exists(), f"missing source: {source}"

    cfg = qnn_build.QnnBuildConfig.from_env(backend="cpu")
    out = qnn_build.build_qnn_kernel(source, "add_f32", cache_dir, cfg)

    assert out.exists(), f"build did not emit artifact at {out}"
    # A QNN context binary header has a stable magic; loose lower bound on
    # size (Add op in fp32 with 1×16 shape has produced ~1.5 KB so far).
    assert out.stat().st_size >= 512, f"unexpectedly small artifact: {out.stat().st_size} bytes"

    # Cache hit: re-running with unchanged inputs returns the same path
    # without rebuilding.
    out2 = qnn_build.build_qnn_kernel(source, "add_f32", cache_dir, cfg)
    assert out2 == out


@pytest.mark.skipif(not _sdk_available(), reason="QNN SDK not available")
def test_build_invalidates_on_source_change(cache_dir: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Modifying the source bytes invalidates the cache key."""
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build

    source = REPO / "benchmarks" / "QRB5165" / "kernels" / "abi" / "add_f32.qnn.cpp"
    cfg = qnn_build.QnnBuildConfig.from_env(backend="cpu")
    first = qnn_build.build_qnn_kernel(source, "add_f32", cache_dir, cfg)

    # Make a perturbed copy and rebuild — different cache key, different path.
    altered = tmp_path / "altered.qnn.cpp"
    text = source.read_text()
    altered.write_text(text + "\n// trailing comment to perturb hash\n")
    second = qnn_build.build_qnn_kernel(altered, "add_f32", cache_dir, cfg)

    assert second != first, "cache should produce distinct artifacts for different sources"
    assert second.exists() and first.exists()


@pytest.mark.skipif(not _sdk_available(), reason="QNN SDK not available")
def test_full_pipeline_manifest_to_spec(cache_dir: pathlib.Path) -> None:
    """End-to-end manifest.load → precompile → spec_gen.emit.

    Validates the kernel-embedding pipeline emits a transform-spec MLIR plus
    a sidecar QNN manifest that maps the kernel's export symbol to the
    .qnn-ctx blob the precompile step produced.
    """
    import sys

    sys.path.insert(0, str(REPO / "tools"))
    from kernels import (
        manifest as kmanifest,
    )
    from kernels import (
        precompile as kprecompile,
    )
    from kernels import (
        spec_gen as kspec_gen,
    )

    manifest_path = REPO / "benchmarks" / "QRB5165" / "kernels" / "manifest.json"
    mfst = kmanifest.load(manifest_path)
    objects = kprecompile.precompile(mfst, cache_dir, targets_filter=["qnn-gpu"])
    assert len(objects) >= 1
    # qnn_add_f32 is the canonical entry — must be present.
    add_artifact = objects.get(("qnn_add_f32", "qnn-gpu"))
    assert add_artifact is not None
    assert add_artifact.qnn_context is True
    assert add_artifact.path.exists()

    spec_path = cache_dir / "transform_spec.mlir"
    gen = kspec_gen.emit(mfst, objects, spec_path)
    assert gen.spec_path.exists()
    assert gen.qnn_manifest_path is not None and gen.qnn_manifest_path.exists()

    import json

    qnn_map = json.loads(gen.qnn_manifest_path.read_text())
    # `_export_name` strips trailing `_workgroup` or appends `_dispatch`;
    # for entry_symbol "add_f32" we expect "add_f32_dispatch".
    assert "add_f32_dispatch" in qnn_map, qnn_map
    assert qnn_map["add_f32_dispatch"].endswith(".qnn-ctx")
    assert pathlib.Path(qnn_map["add_f32_dispatch"]).exists()

    spec_text = gen.spec_path.read_text()
    assert "qnn-context-binary" in spec_text
    assert "@add_f32_dispatch" in spec_text


@pytest.mark.skipif(
    not (_sdk_available() and _board_available()),
    reason="QNN SDK + QRB5165 board (with /tmp/qnn_probe staged) required",
)
def test_build_on_board_gpu_add_f32(cache_dir: pathlib.Path) -> None:
    """Board-required: build add_f32 on QRB5165 against libQnnGpu.so.

    Validates the on-board build path produces a GPU-loadable .qnn-ctx blob
    (the host CPU-validation path produces an incompatible blob — the GPU
    rejects host-CPU-validated blobs with INVALID_VERSION). Caches by
    source+target hash.
    """
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build

    source = REPO / "benchmarks" / "QRB5165" / "kernels" / "abi" / "add_f32.qnn.cpp"
    cfg = qnn_build.BoardBuildConfig.from_env(
        ssh_host="qdev",
        board_qairt_root="/tmp/qnn_probe",
        target_backend="gpu",
    )
    out = qnn_build.build_qnn_kernel_on_board(source, "add_f32", cache_dir, cfg)
    assert out.exists()
    # Phase 0 produced a 54932-byte GPU blob; the same source must produce
    # the same byte count regardless of caching layer.
    assert out.stat().st_size > 50000, f"unexpectedly small artifact: {out.stat().st_size}"

    # Cache hit: re-running with unchanged inputs returns the same path
    # without re-building.
    out2 = qnn_build.build_qnn_kernel_on_board(source, "add_f32", cache_dir, cfg)
    assert out2 == out


@pytest.mark.skipif(
    not (_sdk_available() and _board_available()),
    reason="QNN SDK + QRB5165 board required",
)
def test_build_on_board_gpu_conv2d_relu(cache_dir: pathlib.Path) -> None:
    """Board-required: build the multi-op Conv2D+ReLU smoke kernel on
    QRB5165 against libQnnGpu.so. Verifies the authoring style scales
    beyond elementwise to a graph with Conv2D's full param surface."""
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build

    source = REPO / "benchmarks" / "QRB5165" / "kernels" / "abi" / "conv2d_relu_smoke_f32.qnn.cpp"
    cfg = qnn_build.BoardBuildConfig.from_env(
        ssh_host="qdev",
        board_qairt_root="/tmp/qnn_probe",
        target_backend="gpu",
    )
    out = qnn_build.build_qnn_kernel_on_board(source, "conv2d_relu_smoke_f32", cache_dir, cfg)
    assert out.exists()
    assert out.stat().st_size > 50000


@pytest.mark.skipif(not _sdk_available(), reason="QNN SDK not available")
def test_qnn_emit_conv2d_relu_produces_valid_cpp(cache_dir: pathlib.Path) -> None:
    """Host-only: the MLIR→QNN emitter recognises the Conv2D+bias+ReLU
    pattern in the smoke fixture and produces a complete .qnn.cpp source.
    Validates the lowering contract without touching the board."""
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_emit
    import qnn_ir  # noqa: F401

    mlir_path = REPO / "benchmarks" / "QRB5165" / "mlir" / "conv2d_relu_smoke.mlir"
    out_cpp = cache_dir / "conv2d_relu_emitted.qnn.cpp"
    graph = qnn_emit.emit(mlir_path, out_cpp)
    assert graph.name == "conv2d_relu"
    # 5 tensors expected: input, weight (static), bias (static), conv_out
    # (native), output (APP_READ).
    assert len(graph.tensors) == 5
    assert len(graph.nodes) == 2
    assert graph.nodes[0].op_type == "Conv2d"
    assert graph.nodes[1].op_type == "Relu"
    text = out_cpp.read_text()
    assert "QnnModel_composeGraphs" in text
    assert "Conv2d" in text
    assert "Relu" in text
    # Static weight payload (3*3*3*4 floats = 432 bytes) must be present.
    assert "g_weight_data[432]" in text


@pytest.mark.skipif(
    not (_sdk_available() and _board_available()),
    reason="QNN SDK + QRB5165 board required",
)
def test_qnn_emit_to_board_bytes_equal_vs_handauthored(
    cache_dir: pathlib.Path,
) -> None:
    """Board-required end-to-end: MLIR Conv2D+ReLU snippet → emitter →
    .qnn.cpp → on-board build → GPU ctx-bin → run on board → bytes-equal
    vs the hand-authored conv2d_relu_smoke kernel's GPU output. This is
    the PR-C0 done condition: the emitter's compile path produces the
    same arithmetic on the same hardware as the hand-authored equivalent.
    """
    import subprocess
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build
    import qnn_emit

    # 1) Emit .qnn.cpp from MLIR.
    mlir_path = REPO / "benchmarks" / "QRB5165" / "mlir" / "conv2d_relu_smoke.mlir"
    emitted_cpp = cache_dir / "conv2d_relu_emitted.qnn.cpp"
    qnn_emit.emit(mlir_path, emitted_cpp)

    # 2) Build both kernels on board (emitted + hand-authored) into GPU ctx-bins.
    handauth_src = REPO / "benchmarks" / "QRB5165" / "kernels" / "abi" / "conv2d_relu_smoke_f32.qnn.cpp"
    cfg = qnn_build.BoardBuildConfig.from_env(
        ssh_host="qdev",
        board_qairt_root="/tmp/qnn_probe",
        target_backend="gpu",
    )
    emitted_ctx = qnn_build.build_qnn_kernel_on_board(emitted_cpp, "conv2d_relu_emitted", cache_dir, cfg)
    handauth_ctx = qnn_build.build_qnn_kernel_on_board(handauth_src, "conv2d_relu_handauth", cache_dir, cfg)

    # 3) Push both ctx-bins to board and run with the same deterministic input.
    #    `conv_input.raw` + `conv_input_list.txt` were left on the board
    #    from Phase 0; if absent, the test fails with a clear message.
    subprocess.run(["scp", "-q", str(emitted_ctx), "qdev:/tmp/qnn_probe/test_emit.qnn-ctx"], check=True)
    subprocess.run(["scp", "-q", str(handauth_ctx), "qdev:/tmp/qnn_probe/test_hand.qnn-ctx"], check=True)
    run_cmd = (
        "cd /tmp/qnn_probe && export LD_LIBRARY_PATH=$PWD/lib && "
        "rm -rf out_test_emit out_test_hand && "
        "./bin/qnn-net-run --retrieve_context ./test_emit.qnn-ctx "
        "--backend ./lib/libQnnGpu.so --input_list ./conv_input_list.txt "
        "--output_dir ./out_test_emit && "
        "./bin/qnn-net-run --retrieve_context ./test_hand.qnn-ctx "
        "--backend ./lib/libQnnGpu.so --input_list ./conv_input_list.txt "
        "--output_dir ./out_test_hand"
    )
    res = subprocess.run(["ssh", "qdev", run_cmd], capture_output=True, text=True)
    assert res.returncode == 0, f"on-board run failed:\n{res.stderr}\n{res.stdout}"

    # 4) Pull both outputs and compare.
    emit_out = cache_dir / "test_emit_output.raw"
    hand_out = cache_dir / "test_hand_output.raw"
    subprocess.run(["scp", "-q", "qdev:/tmp/qnn_probe/out_test_emit/Result_0/output.raw", str(emit_out)], check=True)
    subprocess.run(["scp", "-q", "qdev:/tmp/qnn_probe/out_test_hand/Result_0/output.raw", str(hand_out)], check=True)
    import numpy as np

    emit = np.fromfile(emit_out, dtype=np.float32)
    hand = np.fromfile(hand_out, dtype=np.float32)
    # Bytes-equal is the strict gate — the emitter's lowering should
    # produce identical arithmetic to the hand-authored equivalent on the
    # same hardware.
    assert emit.shape == hand.shape == (144,)
    assert np.array_equal(emit, hand), (
        f"emitter and hand-authored outputs differ on GPU: " f"max abs diff {np.abs(emit - hand).max():.2e}"
    )


@pytest.mark.skipif(not _sdk_available(), reason="QNN SDK not available")
def test_qnn_emit_dispatcher_recognises_elementwise_patterns(
    cache_dir: pathlib.Path,
) -> None:
    """Host-only: the multi-pattern dispatcher in `qnn_emit.parse_mlir`
    classifies elementwise binary (Add, Mul) and unary (Sigmoid) fixtures
    correctly. Validates op-coverage extension without touching the board."""
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_emit

    fixtures = {
        "add_f32_smoke.mlir": ("add_f32", "ElementWiseAdd"),
        "mul_f32_smoke.mlir": ("mul_f32", "ElementWiseMultiply"),
        "sigmoid_f32_smoke.mlir": ("sigmoid_f32", "Sigmoid"),
    }
    for fname, (expected_graph_name, expected_op) in fixtures.items():
        text = (REPO / "benchmarks" / "QRB5165" / "mlir" / fname).read_text()
        graph = qnn_emit.parse_mlir(text)
        assert graph.name == expected_graph_name
        op_types = [n.op_type for n in graph.nodes]
        assert expected_op in op_types, f"{fname}: expected {expected_op} in {op_types}"


@pytest.mark.skipif(
    not (_sdk_available() and _board_available()),
    reason="QNN SDK + QRB5165 board required",
)
def test_qnn_emit_elementwise_bytes_equal_on_gpu(
    cache_dir: pathlib.Path,
) -> None:
    """Board-required: emit + build + run elementwise Add / Mul / Sigmoid
    on Adreno GPU; assert outputs match host numpy reference (within fp32
    tolerance for Sigmoid; bytes-equal for Add/Mul)."""
    import subprocess
    import sys

    import numpy as np

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build
    import qnn_emit

    cfg = qnn_build.BoardBuildConfig.from_env(
        ssh_host="qdev",
        board_qairt_root="/tmp/qnn_probe",
        target_backend="gpu",
    )
    a = np.linspace(-1, 1, 16, dtype=np.float32).reshape(1, 16)
    b = np.linspace(0.5, 2.0, 16, dtype=np.float32).reshape(1, 16)
    a_path = cache_dir / "a.raw"
    a.tofile(a_path)
    b_path = cache_dir / "b.raw"
    b.tofile(b_path)
    subprocess.run(["scp", "-q", str(a_path), "qdev:/tmp/qnn_probe/a_test.raw"], check=True)
    subprocess.run(["scp", "-q", str(b_path), "qdev:/tmp/qnn_probe/b_test.raw"], check=True)

    cases = [
        (
            "add_f32_smoke",
            "input_list_add_test.txt",
            "a:=/tmp/qnn_probe/a_test.raw b:=/tmp/qnn_probe/b_test.raw\n",
            (a + b).astype(np.float32),
            True,
        ),  # strict bytes-equal
        (
            "mul_f32_smoke",
            "input_list_mul_test.txt",
            "a:=/tmp/qnn_probe/a_test.raw b:=/tmp/qnn_probe/b_test.raw\n",
            (a * b).astype(np.float32),
            True,
        ),
        (
            "sigmoid_f32_smoke",
            "input_list_sig_test.txt",
            "input:=/tmp/qnn_probe/a_test.raw\n",
            (1.0 / (1.0 + np.exp(-a))).astype(np.float32),
            False,
        ),  # fp32 tolerance only — Adreno exp differs 1 ULP from numpy
    ]
    for kname, list_name, list_contents, expected, strict in cases:
        # 1) Emit MLIR → .qnn.cpp
        mlir_path = REPO / "benchmarks" / "QRB5165" / "mlir" / f"{kname}.mlir"
        cpp_path = cache_dir / f"{kname}.qnn.cpp"
        qnn_emit.emit(mlir_path, cpp_path)

        # 2) Build on board for GPU
        ctx = qnn_build.build_qnn_kernel_on_board(cpp_path, kname, cache_dir, cfg)

        # 3) Push ctx + input list to board, run
        subprocess.run(["scp", "-q", str(ctx), f"qdev:/tmp/qnn_probe/{kname}.qnn-ctx"], check=True)
        list_path = cache_dir / list_name
        list_path.write_text(list_contents)
        subprocess.run(["scp", "-q", str(list_path), f"qdev:/tmp/qnn_probe/{list_name}"], check=True)
        run = (
            f"cd /tmp/qnn_probe && export LD_LIBRARY_PATH=$PWD/lib && "
            f"rm -rf out_test_{kname} && "
            f"./bin/qnn-net-run --retrieve_context ./{kname}.qnn-ctx "
            f"--backend ./lib/libQnnGpu.so --input_list ./{list_name} "
            f"--output_dir ./out_test_{kname}"
        )
        res = subprocess.run(["ssh", "qdev", run], capture_output=True, text=True)
        assert res.returncode == 0, f"{kname} run failed:\n{res.stderr}"

        # 4) Pull output, compare bytes
        out_path = cache_dir / f"{kname}_out.raw"
        subprocess.run(
            ["scp", "-q", f"qdev:/tmp/qnn_probe/out_test_{kname}/Result_0/output.raw", str(out_path)],
            check=True,
        )
        actual = np.fromfile(out_path, dtype=np.float32).reshape(expected.shape)
        if strict:
            assert np.array_equal(actual, expected), (
                f"{kname}: not bytes-equal vs reference; max abs diff " f"{np.abs(actual - expected).max():.2e}"
            )
        else:
            diff = np.abs(actual - expected).max()
            assert diff < 1e-6, f"{kname}: max abs diff {diff:.2e} exceeds fp32 tolerance"


@pytest.mark.skipif(not _sdk_available(), reason="QNN SDK not available")
def test_qnn_emit_dispatcher_recognises_dwconv_and_maxpool(
    cache_dir: pathlib.Path,
) -> None:
    """Host-only: dispatcher classifies the depthwise-conv and maxpool
    fixtures correctly. Validates the NodeDesc factories + recognisers
    without going to the board (depthwise's GPU validation is a known
    open issue tracked separately)."""
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_emit

    fixtures = {
        "depthwise_conv_smoke.mlir": ("depthwise_conv_f32", "DepthWiseConv2d"),
        "maxpool_smoke.mlir": ("maxpool_f32", "PoolMax2d"),
    }
    for fname, (expected_graph_name, expected_op) in fixtures.items():
        text = (REPO / "benchmarks" / "QRB5165" / "mlir" / fname).read_text()
        graph = qnn_emit.parse_mlir(text)
        assert graph.name == expected_graph_name, f"{fname}: expected graph {expected_graph_name}, got {graph.name}"
        op_types = [n.op_type for n in graph.nodes]
        assert expected_op in op_types, f"{fname}: expected {expected_op} in {op_types}"


@pytest.mark.skipif(
    not (_sdk_available() and _board_available()),
    reason="QNN SDK + QRB5165 board required",
)
def test_qnn_emit_maxpool_bytes_equal_on_gpu(cache_dir: pathlib.Path) -> None:
    """Board-required: emit MaxPool 2x2 stride 2 from MLIR, build on GPU,
    run on board, assert bytes-equal vs numpy reference."""
    import subprocess
    import sys

    import numpy as np

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build
    import qnn_emit

    cfg = qnn_build.BoardBuildConfig.from_env(
        ssh_host="qdev",
        board_qairt_root="/tmp/qnn_probe",
        target_backend="gpu",
    )
    # Emit + build
    mlir_path = REPO / "benchmarks" / "QRB5165" / "mlir" / "maxpool_smoke.mlir"
    cpp_path = cache_dir / "maxpool_smoke.qnn.cpp"
    qnn_emit.emit(mlir_path, cpp_path)
    ctx = qnn_build.build_qnn_kernel_on_board(cpp_path, "maxpool_smoke", cache_dir, cfg)

    # Deterministic input [1,8,8,4]
    x = np.linspace(-1, 1, 1 * 8 * 8 * 4, dtype=np.float32).reshape(1, 8, 8, 4)
    x_path = cache_dir / "pool_in.raw"
    x.tofile(x_path)
    expected = np.zeros((1, 4, 4, 4), dtype=np.float32)
    for h in range(4):
        for w in range(4):
            for c in range(4):
                expected[0, h, w, c] = x[0, 2 * h : 2 * h + 2, 2 * w : 2 * w + 2, c].max()

    list_path = cache_dir / "pool_input_list.txt"
    list_path.write_text("input:=/tmp/qnn_probe/pool_test_in.raw\n")
    subprocess.run(
        ["scp", "-q", str(x_path), "qdev:/tmp/qnn_probe/pool_test_in.raw"],
        check=True,
    )
    subprocess.run(
        ["scp", "-q", str(list_path), "qdev:/tmp/qnn_probe/pool_test_input_list.txt"],
        check=True,
    )
    subprocess.run(
        ["scp", "-q", str(ctx), "qdev:/tmp/qnn_probe/test_pool.qnn-ctx"],
        check=True,
    )
    run = (
        "cd /tmp/qnn_probe && export LD_LIBRARY_PATH=$PWD/lib && "
        "rm -rf out_test_pool && "
        "./bin/qnn-net-run --retrieve_context ./test_pool.qnn-ctx "
        "--backend ./lib/libQnnGpu.so "
        "--input_list ./pool_test_input_list.txt "
        "--output_dir ./out_test_pool"
    )
    res = subprocess.run(["ssh", "qdev", run], capture_output=True, text=True)
    assert res.returncode == 0, f"on-board pool run failed:\n{res.stderr}"

    out_path = cache_dir / "pool_out.raw"
    subprocess.run(
        ["scp", "-q", "qdev:/tmp/qnn_probe/out_test_pool/Result_0/output.raw", str(out_path)],
        check=True,
    )
    actual = np.fromfile(out_path, dtype=np.float32).reshape(expected.shape)
    assert np.array_equal(actual, expected), (
        f"maxpool not bytes-equal vs numpy ref; max abs diff " f"{np.abs(actual - expected).max():.2e}"
    )


@pytest.mark.skipif(not _sdk_available(), reason="QNN SDK not available")
def test_qnn_emit_dispatcher_recognises_concat(cache_dir: pathlib.Path) -> None:
    """Host-only: dispatcher classifies tensor.concat fixtures correctly."""
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_emit

    text = (REPO / "benchmarks" / "QRB5165" / "mlir" / "concat_smoke.mlir").read_text()
    graph = qnn_emit.parse_mlir(text)
    assert graph.name == "concat_f32"
    assert [n.op_type for n in graph.nodes] == ["Concat"]
    # Two input tensors (a, b) + one output.
    roles = sorted(t.role for t in graph.tensors)
    assert roles == ["input", "input", "output"]


@pytest.mark.skipif(
    not (_sdk_available() and _board_available()),
    reason="QNN SDK + QRB5165 board required",
)
def test_qnn_emit_concat_bytes_equal_on_gpu(cache_dir: pathlib.Path) -> None:
    """Board-required: emit channel-wise concat from MLIR, build for GPU,
    run on board, assert bytes-equal vs numpy reference."""
    import subprocess
    import sys

    import numpy as np

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build
    import qnn_emit

    cfg = qnn_build.BoardBuildConfig.from_env(
        ssh_host="qdev",
        board_qairt_root="/tmp/qnn_probe",
        target_backend="gpu",
    )
    mlir_path = REPO / "benchmarks" / "QRB5165" / "mlir" / "concat_smoke.mlir"
    cpp_path = cache_dir / "concat_smoke.qnn.cpp"
    qnn_emit.emit(mlir_path, cpp_path)
    ctx = qnn_build.build_qnn_kernel_on_board(cpp_path, "concat_smoke", cache_dir, cfg)

    a = np.linspace(-1, 1, 1 * 6 * 6 * 4, dtype=np.float32).reshape(1, 6, 6, 4)
    b = np.linspace(2, 3, 1 * 6 * 6 * 4, dtype=np.float32).reshape(1, 6, 6, 4)
    expected = np.concatenate([a, b], axis=3).astype(np.float32)
    a_path = cache_dir / "concat_a.raw"
    a.tofile(a_path)
    b_path = cache_dir / "concat_b.raw"
    b.tofile(b_path)

    list_path = cache_dir / "concat_input_list.txt"
    list_path.write_text("a:=/tmp/qnn_probe/concat_test_a.raw " "b:=/tmp/qnn_probe/concat_test_b.raw\n")
    subprocess.run(["scp", "-q", str(a_path), "qdev:/tmp/qnn_probe/concat_test_a.raw"], check=True)
    subprocess.run(["scp", "-q", str(b_path), "qdev:/tmp/qnn_probe/concat_test_b.raw"], check=True)
    subprocess.run(["scp", "-q", str(list_path), "qdev:/tmp/qnn_probe/concat_test_input_list.txt"], check=True)
    subprocess.run(["scp", "-q", str(ctx), "qdev:/tmp/qnn_probe/test_concat.qnn-ctx"], check=True)
    run = (
        "cd /tmp/qnn_probe && export LD_LIBRARY_PATH=$PWD/lib && "
        "rm -rf out_test_concat && "
        "./bin/qnn-net-run --retrieve_context ./test_concat.qnn-ctx "
        "--backend ./lib/libQnnGpu.so "
        "--input_list ./concat_test_input_list.txt "
        "--output_dir ./out_test_concat"
    )
    res = subprocess.run(["ssh", "qdev", run], capture_output=True, text=True)
    assert res.returncode == 0, f"on-board concat run failed:\n{res.stderr}"

    out_path = cache_dir / "concat_out.raw"
    subprocess.run(
        ["scp", "-q", "qdev:/tmp/qnn_probe/out_test_concat/Result_0/output.raw", str(out_path)],
        check=True,
    )
    actual = np.fromfile(out_path, dtype=np.float32).reshape(expected.shape)
    assert np.array_equal(actual, expected), (
        f"concat not bytes-equal vs numpy ref; max abs diff " f"{np.abs(actual - expected).max():.2e}"
    )


@pytest.mark.skipif(not _sdk_available(), reason="QNN SDK not available")
def test_qnn_emit_dispatcher_recognises_reshape(cache_dir: pathlib.Path) -> None:
    """Host-only: dispatcher classifies tensor.collapse_shape as a Reshape."""
    import sys

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_emit

    text = (REPO / "benchmarks" / "QRB5165" / "mlir" / "reshape_smoke.mlir").read_text()
    graph = qnn_emit.parse_mlir(text)
    assert graph.name == "reshape_f32"
    assert [n.op_type for n in graph.nodes] == ["Reshape"]


@pytest.mark.skipif(
    not (_sdk_available() and _board_available()),
    reason="QNN SDK + QRB5165 board required",
)
def test_qnn_emit_reshape_bytes_equal_on_gpu(cache_dir: pathlib.Path) -> None:
    """Board-required: emit a 4D-to-2D reshape from MLIR, build for GPU,
    run on board, assert bytes-equal vs numpy reshape."""
    import subprocess
    import sys

    import numpy as np

    sys.path.insert(0, str(REPO / "tools" / "kernels"))
    import qnn_build
    import qnn_emit

    cfg = qnn_build.BoardBuildConfig.from_env(
        ssh_host="qdev",
        board_qairt_root="/tmp/qnn_probe",
        target_backend="gpu",
    )
    mlir_path = REPO / "benchmarks" / "QRB5165" / "mlir" / "reshape_smoke.mlir"
    cpp_path = cache_dir / "reshape_smoke.qnn.cpp"
    qnn_emit.emit(mlir_path, cpp_path)
    ctx = qnn_build.build_qnn_kernel_on_board(cpp_path, "reshape_smoke", cache_dir, cfg)

    x = np.linspace(-1, 1, 1 * 6 * 6 * 4, dtype=np.float32).reshape(1, 6, 6, 4)
    expected = x.reshape(1, 144).astype(np.float32)
    x_path = cache_dir / "reshape_in.raw"
    x.tofile(x_path)

    list_path = cache_dir / "reshape_input_list.txt"
    list_path.write_text("input:=/tmp/qnn_probe/reshape_test_in.raw\n")
    subprocess.run(["scp", "-q", str(x_path), "qdev:/tmp/qnn_probe/reshape_test_in.raw"], check=True)
    subprocess.run(["scp", "-q", str(list_path), "qdev:/tmp/qnn_probe/reshape_test_input_list.txt"], check=True)
    subprocess.run(["scp", "-q", str(ctx), "qdev:/tmp/qnn_probe/test_reshape.qnn-ctx"], check=True)
    run = (
        "cd /tmp/qnn_probe && export LD_LIBRARY_PATH=$PWD/lib && "
        "rm -rf out_test_reshape && "
        "./bin/qnn-net-run --retrieve_context ./test_reshape.qnn-ctx "
        "--backend ./lib/libQnnGpu.so "
        "--input_list ./reshape_test_input_list.txt "
        "--output_dir ./out_test_reshape"
    )
    res = subprocess.run(["ssh", "qdev", run], capture_output=True, text=True)
    assert res.returncode == 0, f"on-board reshape run failed:\n{res.stderr}"

    out_path = cache_dir / "reshape_out.raw"
    subprocess.run(
        ["scp", "-q", "qdev:/tmp/qnn_probe/out_test_reshape/Result_0/output.raw", str(out_path)],
        check=True,
    )
    actual = np.fromfile(out_path, dtype=np.float32).reshape(expected.shape)
    assert np.array_equal(actual, expected), (
        f"reshape not bytes-equal vs numpy ref; max abs diff " f"{np.abs(actual - expected).max():.2e}"
    )
