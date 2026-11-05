"""Public benchmark intake is explicit geometry, not a tuned schedule or verdict."""

import importlib.util

import pytest
from xdsl.dialects import arith, func
from xdsl.dialects.linalg.ops import GenericOp

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text as parse_module


@pytest.fixture(scope="module")
def benchmark():
    path = repo_root() / "examples/gemmini/phase2/shared_pass_benchmarks.py"
    spec = importlib.util.spec_from_file_location("shared_pass_benchmark_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("index", range(6))
def test_complete_source_geometry_and_ordered_integer_recurrence(benchmark, index):
    m, k, n = benchmark.GEMMS_MKN[index]
    module = parse_module(benchmark.gemm_source(m, k, n))
    module.verify()
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp))
    assert [arg.type.get_shape() for arg in entry.body.block.args] == [(m, k), (k, n)]
    generic = next(op for op in module.walk() if isinstance(op, GenericOp))
    assert generic.results[0].type.get_shape() == (m, n)
    assert [str(t) for t in generic.iterator_types] == [
        "#linalg.iterator_type<parallel>",
        "#linalg.iterator_type<parallel>",
        "#linalg.iterator_type<reduction>",
    ]
    body = list(generic.body.block.ops)
    assert [op.name for op in body] == ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]
    assert isinstance(body[2], arith.MuliOp)
    assert tuple(body[2].operands) == (body[0].result, body[1].result)
    assert tuple(body[3].operands) == (generic.body.block.args[2], body[2].result)
    assert tuple(body[4].operands) == (body[3].result,)
    returned = next(op for op in module.walk() if isinstance(op, func.ReturnOp))
    assert tuple(returned.operands) == tuple(generic.results)
    row = benchmark.workload(m, k, n)
    assert row["dimensions"] == {"M": m, "K": k, "N": n}
    assert row["proposed_execution_domain"]["every_partial_sum_absolute_bound"] == k * 4
    assert k * 4 < 1 << 19  # Declared test-domain arithmetic; no runtime qualification.


@pytest.mark.parametrize("value", (0, -1, True, 3.5, "32", 1 << 31))
def test_unknown_or_invalid_extent_refuses(benchmark, value):
    with pytest.raises(ValueError, match="explicit positive bounded"):
        benchmark.gemm_source(32, value, 48)


def test_materialization_preserves_complete_roster_and_unresolved_convolutions(benchmark, tmp_path):
    import hashlib

    destination = tmp_path / "public-inputs"
    document = benchmark.generate(destination)
    assert document["dimension_order"] == "M_K_N"
    assert benchmark.GEMMS_MKN == (
        (512, 512, 512),
        (12544, 256, 64),
        (12544, 64, 256),
        (3136, 512, 128),
        (3136, 128, 512),
        (784, 1024, 256),
    )
    assert len(document["gemms"]) == 6
    assert document["gemms"][0]["dimensions"] == {"M": 512, "K": 512, "N": 512}
    assert document["execution"] == "NOT_RUN" and document["performance"] == "NOT_MEASURED"
    assert len(document["convolutions"]) == 3
    for row in document["convolutions"]:
        assert row["status"].startswith("awaiting_explicit_geometry")
    assert len(list(destination.iterdir())) == 6
    for row in document["gemms"]:
        assert hashlib.sha256((destination / row["source"]["file"]).read_bytes()).hexdigest() == row["source"]["sha256"]
    with pytest.raises(FileExistsError):
        benchmark.generate(destination)


@pytest.mark.parametrize(
    "damage",
    (
        "none",
        "compiler_identity",
        "source_bytes",
        "missing_artifacts",
        "compiler_drift",
        "earlier_source",
        "earlier_output",
        "earlier_analysis",
        "python_drift",
        "verifier_drift",
    ),
)
def test_preflight_does_not_credit_changed_inputs_or_empty_emission(benchmark, tmp_path, monkeypatch, damage):
    import json
    from pathlib import Path
    from subprocess import CompletedProcess

    from merlin.common import proc
    from merlin.compile.model_execution_inputs import strict_tree_sha256

    compiler = tmp_path / "compiler"
    compiler.mkdir()
    (compiler / "compiler.py").write_text("# independently selected test compiler\n")
    (compiler / "manifest.yaml").write_text(
        "entrypoints: {tool: compiler.py}\n"
        "commands:\n  emit_analysis_bundle:\n"
        "    argv: [python3, '{tool}', '--emit-command-buffer={output_json}', '{input_mlir}']\n"
    )
    identity = strict_tree_sha256(compiler)["sha256"]
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    real_python = tmp_path / "original-interpreter"
    real_python.write_text("original selected interpreter")
    python.symlink_to(real_python)
    install = tmp_path / "llvm"
    (install / "bin").mkdir(parents=True)
    for name in ("mlir-opt", "mlir-translate"):
        (install / "bin" / name).write_text("original selected tool")
    document = benchmark.generate(tmp_path / "inputs")
    if damage == "compiler_identity":
        identity = "0" * 64
    if damage == "source_bytes":
        (tmp_path / "inputs" / document["gemms"][0]["source"]["file"]).write_text("changed")
    called = []
    compiled = []

    def emit(argv, **kwargs):
        called.append(argv)
        argv = [str(arg) for arg in argv]
        is_compiler = any(arg.startswith("--emit-command-buffer=") for arg in argv)
        if is_compiler:
            assert argv[0] == str(python)  # Preserve the selected venv invocation.
            compiled.append(argv)
        if damage == "compiler_drift" and is_compiler:
            (compiler / "compiler.py").write_text("changed compiler")
        if is_compiler and damage != "missing_artifacts":
            output = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--emit-command-buffer="))
            Path(output).write_text(json.dumps({"synthetic": True}))
        if len(compiled) == 6 and not is_compiler:
            first = document["gemms"][0]
            targets = {
                "earlier_source": tmp_path / "inputs" / first["source"]["file"],
                "earlier_output": tmp_path / "inputs" / (first["id"] + ".target.mlir"),
                "earlier_analysis": tmp_path / "inputs" / (first["id"] + ".analysis.json"),
                "python_drift": python,
                "verifier_drift": install / "bin/mlir-opt",
            }
            if damage in targets:
                targets[damage].write_text("changed after earlier successful compilation")
        return CompletedProcess(argv, 0, "" if damage == "missing_artifacts" else "builtin.module {}", "")

    monkeypatch.setattr(proc, "run_checked", emit)
    kwargs = dict(compiler=compiler, compiler_sha256=identity, python=python, mlir_install=install)
    if damage in ("none", "missing_artifacts"):
        report = benchmark.preflight(document, tmp_path / "inputs", **kwargs)
        expected = "EMITTED" if damage == "none" else "REFUSED"
        assert all(row["status"] == expected for row in report["results"])
        if damage == "none":
            assert len(report["product_pins"]) == 12 and len(report["source_pins"]) == 6
        assert report["target_execution"] == "NOT_RUN" and report["speedup"] == "NOT_MEASURED"
    else:
        with pytest.raises(ValueError, match="compiler|source|tool|product"):
            benchmark.preflight(document, tmp_path / "inputs", **kwargs)
    if damage in ("compiler_identity", "source_bytes"):
        assert not called
