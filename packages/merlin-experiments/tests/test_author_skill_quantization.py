"""The quantization author skill: its script reads real case shapes and its text stays true to the library.

``scripts/quant_params.py`` prints a case's declared quantization and integer pipeline, the captured
chains of an MLIR module and the target's output path; these tests run it on synthetic workspaces
written here. Every ``PATH:LINE text`` citation in the skill must still hold its text, every Python
block of ``reference/library_api.md`` runs against the library, and every self-check field, flag and
row name the skill tells the author to read must still be produced by the code that writes it.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import merlin_experiments.phase1 as phase1
import pytest
import yaml

from merlin.common.paths import checkout_root, merlin_dir

SKILL = Path(phase1.__file__).resolve().parent / "author_skills" / "quantization"
BODY_LIMIT = 4200


def _script():
    path = SKILL / "scripts" / "quant_params.py"
    spec = importlib.util.spec_from_file_location("author_skill_quantization_quant_params", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QP = _script()


def _run(*argv) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert QP.main([*argv, "--library", "absent-library"]) == 0
    return out.getvalue()


def _json(*argv) -> dict:
    return json.loads(_run(*argv, "--json"))


def _source(module: str) -> str:
    return Path(importlib.util.find_spec(module).origin).read_text(encoding="utf-8")


def _skill_texts() -> dict[str, str]:
    files = [SKILL / "SKILL.md", *sorted((SKILL / "scripts").glob("*.py")), *sorted((SKILL / "reference").glob("*.md"))]
    return {path.relative_to(SKILL).as_posix(): path.read_text(encoding="utf-8") for path in files}


# --------------------------------------------------------------------------------- layout and text


def test_front_matter_size_and_generic_content():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    head, sep, body = text.removeprefix("---\n").partition("\n---\n")
    assert text.startswith("---\n") and sep
    fields = {}
    for line in head.splitlines():
        key, _, value = line.partition(": ")
        fields[key] = json.loads(value) if value.startswith('"') else value
    assert fields["name"] == "quantization"
    assert 80 <= len(fields["description"]) <= 1024 and "<" not in fields["description"]
    assert len(body.encode("utf-8")) <= BODY_LIMIT
    for name in (
        "reference/semantics.md",
        "reference/library_api.md",
        "reference/debugging.md",
        "scripts/quant_params.py",
    ):
        assert name in text and (SKILL / name).is_file()
    gate_path = checkout_root() / "build_tools" / "scripts" / "check_no_target_name.py"
    spec = importlib.util.spec_from_file_location("author_skill_quantization_gate", gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    names = frozenset(gate.TARGET_NAMES | gate.SUBSTRATE_NAMES)
    for label, content in _skill_texts().items():
        assert gate._name_in(content, names) is None, f"{label} names {gate._name_in(content, names)}"
    for script in (SKILL / "scripts").glob("*.py"):
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert "re" not in imported
        assert imported - set(sys.stdlib_module_names) <= {"__future__", "yaml", "merlin"}
        assert len(source.splitlines()) <= 1500
        top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        assert not top & {"yaml", "merlin"}, "library and YAML imports stay inside the functions that need them"


def _citations():
    for label, content in _skill_texts().items():
        if not label.endswith(".md"):
            continue
        for index, span in enumerate(content.split("`")):
            if index % 2 and ".py:" in span:
                yield label, span


def test_every_line_citation_still_holds_its_text():
    seen = 0
    for label, span in _citations():
        path, _, rest = span.partition(".py:")
        number, _, text = rest.partition(" ")
        assert number.isdigit() and text, f"{label}: citation {span!r} is not 'PATH:LINE text'"
        lines = _source(path.replace("/", ".")).splitlines()
        assert text in lines[int(number) - 1], f"{label}: {path}.py:{number} no longer holds {text!r}"
        seen += 1
    assert seen >= 30


# ------------------------------------------------------------------------------- library snippets


ID2, COL = "affine_map<(d0, d1) -> (d0, d1)>", "affine_map<(d0, d1) -> (d1)>"
MUL = "    ^bb0(%a: f32, %b: f32, %o: f32):\n      %c = arith.mulf %a, %b : f32\n      linalg.yield %c : f32"
ADD = "    ^bb0(%a: f32, %b: f32, %o: f32):\n      %c = arith.addf %a, %b : f32\n      linalg.yield %c : f32"
BIAS_CODES = (0, 64, -64, 128, 8, -8, 0, 256)


def _generic(result, ins, types, maps, body, out="tensor<4x8xf32>"):
    return (
        f"    %e_{result} = tensor.empty() : {out}\n"
        f"    %{result} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = "
        f'["parallel", "parallel"]}} ins({ins} : {types}) outs(%e_{result} : {out}) {{\n{body}\n    }} -> {out}'
    )


def chain_module() -> str:
    """int8 x int8 -> i32 -> f32 -> *0.25 -> *0.5 per channel -> +bias -> ReLU -> quantize(8.0)."""
    weights = (
        "dense<["
        + ", ".join("[" + ", ".join(str((i * 3 + j) % 5 - 2) for j in range(8)) + "]" for i in range(8))
        + "]>"
    )
    bias = "dense<[" + ", ".join(f"{b * 0.125:.6e}" for b in BIAS_CODES) + "]>"
    relu = (
        "    ^bb0(%a: f32, %o: f32):\n      %z = arith.constant 0.000000e+00 : f32\n"
        "      %c = arith.maximumf %a, %z : f32\n      linalg.yield %c : f32"
    )
    return "\n".join(
        [
            "builtin.module {",
            "  func.func @forward(%x: tensor<4x8xi8>) -> tensor<4x8xi8> {",
            f"    %w = arith.constant {weights} : tensor<8x8xi8>",
            "    %zero = arith.constant 0 : i32",
            "    %e0 = tensor.empty() : tensor<4x8xi32>",
            "    %init = linalg.fill ins(%zero : i32) outs(%e0 : tensor<4x8xi32>) -> tensor<4x8xi32>",
            "    %mm = linalg.matmul ins(%x, %w : tensor<4x8xi8>, tensor<8x8xi8>) outs(%init : tensor<4x8xi32>)"
            " -> tensor<4x8xi32>",
            _generic(
                "f",
                "%mm",
                "tensor<4x8xi32>",
                [ID2, ID2],
                "    ^bb0(%a: i32, %o: f32):\n      %c = arith.sitofp %a : i32 to f32\n      linalg.yield %c : f32",
            ),
            "    %sin = arith.constant 2.500000e-01 : f32",
            "    %s0 = tensor.splat %sin : tensor<4x8xf32>",
            _generic("m1", "%f, %s0", "tensor<4x8xf32>, tensor<4x8xf32>", [ID2, ID2, ID2], MUL),
            "    %sw = arith.constant dense<5.000000e-01> : tensor<8xf32>",
            _generic("m2", "%m1, %sw", "tensor<4x8xf32>, tensor<8xf32>", [ID2, COL, ID2], MUL),
            f"    %bias = arith.constant {bias} : tensor<8xf32>",
            _generic("ba", "%m2, %bias", "tensor<4x8xf32>, tensor<8xf32>", [ID2, COL, ID2], ADD),
            _generic("r", "%ba", "tensor<4x8xf32>", [ID2, ID2], relu),
            "    %qs = arith.constant dense<8.000000e+00> : tensor<f32>",
            "    %qz = arith.constant dense<0> : tensor<i64>",
            '    %q = "quant_ext.quantize_per_tensor"(%r, %qs, %qz) <{quant_min = -128 : i64, quant_max = 127 : i64, '
            'output_dtype = "int8"}> : (tensor<4x8xf32>, tensor<f32>, tensor<i64>) -> tensor<4x8xi8>',
            "    func.return %q : tensor<4x8xi8>",
            "  }",
            "}",
        ]
    )


FLOAT_MODULE = """builtin.module {
  func.func @forward(%x: tensor<4x8xf32>, %wq: tensor<8x8xi8>, %wf: tensor<8x8xf32>) -> tensor<4x8xf32> {
    %s = arith.constant dense<5.000000e-01> : tensor<f32>
    %z = arith.constant dense<0> : tensor<i64>
    %w = "quant_ext.dequantize_per_tensor"(%wq, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64}>
      : (tensor<8x8xi8>, tensor<f32>, tensor<i64>) -> tensor<8x8xf32>
    %zero = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<4x8xf32>
    %init = linalg.fill ins(%zero : f32) outs(%e : tensor<4x8xf32>) -> tensor<4x8xf32>
    %mm = linalg.matmul ins(%x, %w : tensor<4x8xf32>, tensor<8x8xf32>) outs(%init : tensor<4x8xf32>)
      -> tensor<4x8xf32>
    %mm2 = linalg.matmul ins(%mm, %wf : tensor<4x8xf32>, tensor<8x8xf32>) outs(%init : tensor<4x8xf32>)
      -> tensor<4x8xf32>
    func.return %mm2 : tensor<4x8xf32>
  }
}"""


def output_path(columns_per_scale=4) -> dict:
    return {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [
            {"kind": "preload", "dtype": "i32", "range": [-(1 << 31), (1 << 31) - 1], "rounding": "none"},
        ],
        "readout": [
            {"kind": "activation", "function": "relu", "domain": "accumulator"},
            {"kind": "convert", "to": "f32", "rounding": "half_even"},
            {
                "kind": "scale",
                "dtype": "f32",
                "rounding": "half_even",
                "granularity": "per_command",
                "columns_per_scale": columns_per_scale,
            },
            {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        ],
        "exits": {"accumulator": True, "requant": True},
    }


def _python_blocks() -> list[str]:
    text = (SKILL / "reference" / "library_api.md").read_text(encoding="utf-8")
    pieces = text.split("```")
    return [piece.removeprefix("python\n") for piece in pieces[1::2] if piece.startswith("python\n")]


def test_every_library_snippet_runs():
    blocks = _python_blocks()
    assert len(blocks) == 5
    data = merlin_dir() / "tests" / "data"
    sources = {
        "chain": chain_module(),
        "fused": (data / "fused_requant.mlir").read_text(encoding="utf-8"),
        "captured": (data / "captured_requant.mlir").read_text(encoding="utf-8"),
    }
    printed = {}
    for block in blocks:
        for label, source in sources.items() if "source_text" in block else [("none", "")]:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(block, "library_api.md", "exec"), {"source_text": source, "output_path": output_path()})
            printed[label] = out.getvalue()
    assert "requant ['convert', 'scale', 'scale', 'bias', 'relu', 'quantize']" in printed["chain"]
    assert "0.03125" in printed["fused"]  # the strict single-region reader proves the fused form's scale
    assert "neither a constant nor a stored tensor" in printed["captured"]  # a stored bias, no weights here


# ------------------------------------------------------------------------------------- case mode


def _case(root: Path, group: str, name: str, *, inputs, op, attributes, policy=None, **extra) -> None:
    folder = root / group / name
    folder.mkdir(parents=True)
    document = {
        "name": name,
        "kind": "layer",
        "label": "public",
        "inputs": inputs,
        "operation": {"op": op, "attributes": attributes},
        "numeric_policy": policy or {"compare": "exact_int", "dtype": attributes.get("output_dtype", "i32")},
        **extra,
    }
    (folder / "capsule.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _tensor(name, role, shape, dtype="i8"):
    return {"name": name, "role": role, "shape": shape, "dtype": dtype}


@pytest.fixture
def workspace(tmp_path):
    columns = [0.5] * 16 + [0.125] * 16
    _case(
        tmp_path,
        "layers",
        "CASE_COLUMNS",
        op="conv2d",
        inputs=[
            _tensor("W", "weight", [16, 32]),
            _tensor("IFM", "input", [1, 2, 3, 16]),
            _tensor("B", "bias", [32], "i32"),
        ],
        attributes={
            "ifm": "IFM",
            "weight": "W",
            "out": "Y0",
            "ci": 16,
            "kh": 1,
            "kw": 1,
            "bias": "B",
            "epilogue": ["bias_add", "acc_scale", "relu"],
            "acc_scale_by_column": columns,
            "output_dtype": "i8",
        },
        stimulus_range=[-4, 3],
        integer_partial_sum_bound={"status": "proven_safe", "bound": 260, "signed_positive_limit": 524287},
    )
    _case(
        tmp_path,
        "layers",
        "CASE_SCALAR",
        op="matmul",
        inputs=[_tensor("W", "weight", [16, 17]), _tensor("A0", "input", [1, 16])],
        attributes={
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "epilogue": ["acc_scale"],
            "acc_scale": 0.0625,
            "output_dtype": "i8",
        },
    )
    _case(
        tmp_path,
        "layers",
        "CASE_RESIDUAL",
        op="residual_add",
        inputs=[_tensor("X0", "input", [4, 4]), _tensor("X1", "input", [4, 4])],
        attributes={
            "lhs": "X0",
            "rhs": "X1",
            "out": "Y0",
            "epilogue": ["relu"],
            "lhs_scale": 0.5,
            "rhs_scale": 0.25,
            "bound_lsb": 1,
            "output_dtype": "i8",
        },
        policy={"compare": "bounded_int", "dtype": "i8", "atol": 1, "rtol": 0},
    )
    _case(
        tmp_path,
        "isa",
        "CASE_SHIFT",
        op="matmul",
        inputs=[_tensor("W", "weight", [32, 8]), _tensor("A0", "input", [4, 32])],
        attributes={
            "lhs": "A0",
            "weight": "W",
            "out": "Y0",
            "epilogue": ["requant"],
            "requant_shift": 3,
            "output_dtype": "i8",
        },
    )
    _case(
        tmp_path,
        "isa",
        "CASE_RAW",
        op="matmul",
        inputs=[_tensor("W", "weight", [32, 8]), _tensor("A0", "input", [4, 32])],
        attributes={"lhs": "A0", "weight": "W", "out": "Y0", "epilogue": [], "output_dtype": "i32"},
    )
    # A copy inside the editable tree is not a test case and must not make the name ambiguous.
    _case(tmp_path, "submission", "CASE_SCALAR", op="matmul", inputs=[], attributes={})
    return tmp_path


def test_case_prints_declared_scales_bound_and_pipeline(workspace):
    summary = _json("case", "CASE_COLUMNS", "--workspace", str(workspace))
    assert summary["epilogue"] == ["bias_add", "acc_scale", "relu"]
    assert summary["scales"]["acc_scale_by_column"] == [
        {"first_column": 0, "columns": 16, "scale": 0.5},
        {"first_column": 16, "columns": 16, "scale": 0.125},
    ]
    assert summary["accumulator_bound"] == {
        "terms": 16,
        "min": -260096,
        "max": 262144,
        "fits_i32": True,
        "exact_in_f32": True,
    }
    assert summary["integer_partial_sum_bound"]["signed_positive_limit"] == 524287
    steps = summary["pipeline"]
    assert steps[0].startswith("acc[i32] = sum over K=16 of IFM[i8] * W[i8]")
    assert steps[1].startswith("acc += B[i32, one per output column]")
    assert "32 columns, 2 equal-scale runs" in steps[2]
    assert steps[3:] == ["y = max(y, 0)", "saturate to [-128, 127] and store i8"]
    text = _run("case", "CASE_COLUMNS", "--workspace", str(workspace))
    assert "columns 16..31: 0.125" in text and "4. y = max(y, 0)" in text


def test_case_pipeline_for_shift_residual_and_raw_outputs(workspace):
    shift = _json("case", "CASE_SHIFT", "--workspace", str(workspace))["pipeline"]
    assert shift[1] == "y = (y + 2**2) >> 3  (arithmetic shift, ties round up)"
    residual = _json("case", "CASE_RESIDUAL", "--workspace", str(workspace))
    assert residual["pipeline"][0].startswith("y = round_half_even(f32(f32(f32(a) * 0.5) + f32(f32(b) * 0.25)))")
    assert "bound_lsb=1" in residual["pipeline"][0] and residual["pipeline"][1] == "y = max(y, 0)"
    assert residual["policy"].startswith("bounded_int: |observed - reference| <= 1")
    raw = _json("case", str(workspace / "isa" / "CASE_RAW"), "--workspace", str(workspace))
    assert raw["pipeline"][-1] == "store the raw i32 accumulator (no readout stage applies)"


def test_case_proofs_come_from_the_library(workspace):
    scalar = _json("case", "CASE_SCALAR", "--workspace", str(workspace), "--prove")["proofs"]
    assert scalar["exact_multiplier_shift"] == {"multiplier": 1, "shift": 4}
    assert scalar["thresholds"]["count"] == 255 and scalar["thresholds"]["output"] == [-128, 127]
    assert scalar["fixedpoint_estimate"]["max_estimate_output_error"] <= 1
    assert scalar["producer_interval"] == [-260096, 262144]
    columns = _json("case", "CASE_COLUMNS", "--workspace", str(workspace), "--prove")["proofs"]
    assert columns["column_partition"]["partitions"] == 2
    assert _json("case", "CASE_RAW", "--workspace", str(workspace), "--prove")["proofs"]["readout"] == (
        "no scaled i8 readout declared"
    )


def test_case_lookup_refuses_unknown_names(workspace):
    with pytest.raises(SystemExit, match="found none"):
        QP.main(["case", "CASE_ABSENT", "--workspace", str(workspace)])


# --------------------------------------------------------------------------------------- ir mode


def test_ir_reads_the_chain_and_the_cut_the_output_path_absorbs(tmp_path):
    module = tmp_path / "chain.mlir"
    module.write_text(chain_module(), encoding="utf-8")
    contract = tmp_path / "target_contract.yaml"
    contract.write_text(yaml.safe_dump({"output_epilogue": {"output_path": output_path()}}), encoding="utf-8")
    summary = _json("ir", str(module), "--path", "--contract", str(contract))
    (row,) = summary["chains"]
    assert row["dimensions"] == {"batch": 1, "m": 4, "n": 8, "k": 8} and row["unread"] is None
    assert [s["kind"] for s in row["stages"]] == ["convert", "scale", "scale", "bias", "relu", "quantize"]
    assert row["stages"][1]["numbers"] == {"per": "tensor", "value": 0.25}
    assert row["stages"][2]["numbers"]["per"] == "channel" and row["stages"][2]["numbers"]["n"] == 8
    assert row["stages"][-1]["zero_point"] == 0 and (row["stages"][-1]["min"], row["stages"][-1]["max"]) == (-128, 127)
    fusion = row["fusion"]
    assert fusion["cut"] == "requant" and fusion["commands"] == 2 and fusion["scale_reconfigurations"] == 0
    assert fusion["preload_offsets"]["min"] == -64 and fusion["preload_offsets"]["max"] == 256
    assert summary["quantizers"]["quant_ext.quantize_per_tensor"]["zero_points"] == [0]
    text = _run("ir", str(module), "--path", "--contract", str(contract))
    assert "output path takes:" in text and "refused pool: stage_undeclared" in text


def test_ir_reports_unread_chains_and_float_contraction_evidence(tmp_path):
    captured = _json("ir", str(merlin_dir() / "tests" / "data" / "captured_requant.mlir"))
    (row,) = captured["chains"]
    assert row["dimensions"]["k"] == 64 and "stored tensor" in row["unread"]
    module = tmp_path / "float.mlir"
    module.write_text(FLOAT_MODULE, encoding="utf-8")
    floats = _json("ir", str(module))["float_contractions"]
    assert len(floats["with_captured_quantization"]) == 1 and len(floats["without"]) == 1


# ----------------------------------------------------------------------------------- target mode


def test_target_prints_output_path_and_methodology(tmp_path):
    (tmp_path / "target_contract.yaml").write_text(
        yaml.safe_dump({"output_epilogue": {"schema": "output_epilogue_v1", "output_path": output_path(16)}}),
        encoding="utf-8",
    )
    (tmp_path / "public_declarations").mkdir()
    spec = {
        "quantization": {
            "formats": [{"operand_dtype": "int8", "accumulator_dtype": "i32"}],
            "methodology": {
                "schema": "merlin.quantization_methodology.v1",
                "scale_granularity": "from_output_path",
                "model_variants": [
                    {
                        "model": "m",
                        "name": "v",
                        "status": "reviewed",
                        "reason": "long text",
                        "changes": {"calibration_count": 8},
                    }
                ],
            },
        }
    }
    (tmp_path / "public_declarations" / "software-spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    summary = _json("target", "--workspace", str(tmp_path))
    assert [s["kind"] for s in summary["output_path"]["readout"]] == ["activation", "convert", "scale", "requant"]
    assert summary["software_spec"]["methodology"]["model_variants"] == [
        {"model": "m", "name": "v", "status": "reviewed", "changes": {"calibration_count": 8}}
    ]
    text = _run("target", "--workspace", str(tmp_path))
    assert 'readout 3: {"columns_per_scale": 16' in text and 'scale_granularity: "from_output_path"' in text


# ------------------------------------------------------------- names the skill tells the author to read


def test_named_fields_flags_and_rows_still_exist():
    golden = _source("merlin.targetgen.capsule_golden")
    for name in ("mismatch_indices", "output_never_written", "output_shape_mismatch", "saturated"):
        assert f'"{name}"' in golden
    assert "_MISMATCH_INDEX_CAP = 64" in golden
    feedback = _source("merlin_experiments.phase1.feedback.selfcheck")
    assert '"your_value": fm.get("observed")' in feedback and '"your_artifacts": own' in feedback
    assert '"missing_outputs"' in _source("merlin.targetgen.capsule_runner")
    assert '_SAFE_NUMERIC = {"status", "policy", "mismatch_count"}' in _source("merlin_experiments.phase1.feedback.qa")
    shim = _source("merlin_experiments.phase1.tools.selfcheck")
    assert '"--capsules"' in shim and '"--out"' in shim
    model_check = _source("merlin_experiments.phase1.tools.model_check")
    for flag in ('"replay-kernel"', '"--model"', '"--program"', '"--group"'):
        assert flag in model_check
    assert 'result["epilogue_store_path"] = record' in _source("merlin.targetgen.epilogue_store_path")
    assert 'result["host_compute"] = {' in _source("merlin.targetgen.offload_census")
    assert '"[quant_scope] respected captured quantization evidence: "' in _source(
        "merlin.runtime.backends.zephyr_model"
    )
