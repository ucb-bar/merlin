"""Raw-bit native evidence, preserving NaN payloads/signs and recording conservative support gaps."""

import hashlib
import json
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_lean_defined_conversions import certify
from test_lean_integer_fold import canonicalize
from test_lean_scalar_bitcast import program, source_proofs

from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import check_typed_stage
from merlin.verify.lean.export import export_file


def values(width):
    if width <= 8:
        return list(range(1 << width))
    edges = [0, 1, (1 << width) - 1, 1 << (width - 1)]
    if width in {32, 64}:
        fraction, exponent = (23, 8) if width == 32 else (52, 11)
        infinity = ((1 << exponent) - 1) << fraction
        edges += [
            infinity,
            infinity | (1 << (width - 1)),
            infinity | 1,
            infinity | 0x12345,
            infinity | (1 << (fraction - 1)) | 0x12345,
            infinity - 1,
            (1 << fraction) - 1,
        ]
    rng = random.Random(827163 + width)
    return edges + [rng.getrandbits(width) for _ in range(21)]


def certified(tmp_path, tools, source, target, *, generic=False):
    lean_tools = SimpleNamespace(m2m_python=tools.compiler_python)
    stage, before, after = canonicalize(tmp_path, lean_tools, program(source, target, generic=generic))
    if source != target:
        certify(stage, before, lean_tools, tmp_path)
    else:
        source_proofs(before, lean_tools, tmp_path)
        assert check_typed_stage(stage, m2m_python=tools.compiler_python)["verdict"] == "abstracted"
    return stage, before, after


def samples_and_expected(width, generic=False):
    samples, expected = [], []
    data = values(width)
    for n, value in enumerate(data):
        keep = 827163 + n
        if generic:
            elements = [value, data[(n + 1) % len(data)], data[(n + 2) % len(data)]]
            samples.append([elements, [0, 0, 0], [keep]])
            expected.append([*map(str, elements), str(keep)])
        else:
            samples.append([[value], [value ^ ((1 << width) - 1)], [keep]])
            expected.append([str(value), str(keep)])
    return samples, expected


def compare(tmp_path, tools, before, after, samples, expected):
    for label, path in [("before", before), ("after", after)]:
        work = tmp_path / label
        diff.compare(path.read_text(), samples, work, tools, function="forward")
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


@pytest.mark.parametrize("source,target", [("i32", "f32"), ("f32", "i32"), ("i64", "f64"), ("f64", "i64")])
@pytest.mark.parametrize("generic", [False, True])
def test_raw_bitcasts_preserve_every_observed_bit(tmp_path, fidelity_tools, source, target, generic):
    _, before, after = certified(tmp_path, fidelity_tools, source, target, generic=generic)
    samples, expected = samples_and_expected(int(source[1:]), generic)
    compare(tmp_path, fidelity_tools, before, after, samples, expected)


@pytest.mark.parametrize("width", [1, 13, 32, 64])
def test_native_integer_identity_has_source_proofs_without_alias_claim(tmp_path, fidelity_tools, width):
    _, before, after = certified(tmp_path, fidelity_tools, f"i{width}", f"i{width}")
    samples, expected = samples_and_expected(width)
    compare(tmp_path, fidelity_tools, before, after, samples, expected)


@pytest.mark.parametrize("source,target", [("i32", "f32"), ("f64", "i64")])
def test_changed_bitcast_operand_is_valid_but_observably_wrong(tmp_path, fidelity_tools, source, target):
    stage, before, after = certified(tmp_path, fidelity_tools, source, target)
    result = subprocess.run(
        [
            str(fidelity_tools.compiler_python),
            "-c",
            "from pathlib import Path; from torch_mlir import ir; import sys\n"
            "with ir.Context():\n p=Path(sys.argv[1]); m=ir.Module.parse(p.read_text())\n"
            " b=m.body.operations[0].regions[0].blocks[0]\n"
            " cast=next(o for o in b.operations if o.name=='arith.bitcast')\n"
            " cast.operation.operands[0]=b.arguments[1]\n"
            " m.operation.verify(); p.write_text(str(m))",
            str(after),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert check_typed_stage(stage, m2m_python=fidelity_tools.compiler_python)["verdict"] == "abstracted"
    width = int(source[1:])
    samples, expected = samples_and_expected(width)
    samples, expected = samples[:5], expected[:5]
    results = []
    for label, path in [("before", before), ("wrong", after)]:
        work = tmp_path / label
        diff.compare(path.read_text(), samples, work, fidelity_tools, function="forward")
        results.append(json.loads((work / "native-results.json").read_text())["results"])
    assert results[0] == expected
    assert results[1] != expected


@pytest.mark.parametrize("source,target", [("f32", "f32"), ("f64", "f64"), ("f16", "bf16"), ("bf16", "f16")])
def test_legal_float_bitcast_native_support_gap_manifest(tmp_path, fidelity_tools, source, target):
    lean_tools = SimpleNamespace(m2m_python=fidelity_tools.compiler_python)
    width = 16 if "16" in source else int(source[1:])
    path = tmp_path / "source.mlir"
    if width == 16:
        # Integer external values isolate the legal unsupported float-to-float operation without
        # pretending the fidelity adapter interprets f16/bf16 numerical arithmetic.
        text = f"""func.func @forward(%x: i16, %other: i16, %keep: i32) -> (i16, i32) {{
          %a = arith.bitcast %x : i16 to {source}
          %b = arith.bitcast %a : {source} to {target}
          %r = arith.bitcast %b : {target} to i16
          return %r, %keep : i16, i32
        }}"""
    else:
        text = program(source, target)
    path.write_text(text)
    source_proofs(path, lean_tools, tmp_path, verdict="abstracted")
    exported = tmp_path / "export.json"
    export_file(path, exported, backend="mlir", m2m_python=fidelity_tools.compiler_python)
    inputs, outputs = diff.signature(json.loads(exported.read_text()), "forward")
    samples, expected = samples_and_expected(width)
    samples = [[[str(word) for word in value] for value in sample] for sample in samples]
    if width != 16:
        lean = diff.evaluate_lean(exported, samples, tmp_path / "lean", fidelity_tools, function="forward")
        assert all(x["status"] == "stuck" for x in lean)
    native = diff.execute_mlir(
        path, samples, tmp_path / "native", fidelity_tools, inputs=inputs, outputs=outputs, function="forward"
    )
    assert native["results"] == expected
    library = subprocess.run(
        [str(fidelity_tools.compiler_python), "-c", "from torch_mlir._mlir_libs import _mlir; print(_mlir.__file__)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "classification": "conservative Lean semantic-support gap; not a compiler bug",
        "source_format": source,
        "target_format": target,
        "source_mlir": text,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mlir_revision": native["mlir_revision"],
        "compiler_python_sha256": hashlib.sha256(fidelity_tools.compiler_python.read_bytes()).hexdigest(),
        "mlir_library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(fidelity_tools.lean_exe.read_bytes()).hexdigest(),
        "native_observation": "all initialized samples preserve raw bits, including NaN payloads/signs",
        "lean_observation": "support and definedness refuse; float-result branch accepts integer inputs only",
        "fidelity_claim": False,
    }
    (tmp_path / "support-gap.json").write_text(json.dumps(manifest, indent=2))
